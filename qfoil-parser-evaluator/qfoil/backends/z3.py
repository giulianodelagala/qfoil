"""Optional Z3-backed CEGIS solver for rank>=1 QFOIL queries.

This module is not imported by :mod:`qfoil` unless explicitly requested.
Parsing and ordinary evaluation therefore work identically whether or not
`z3-solver` is installed.

Why this exists (see SIGMA_EVALUATION_PROPOSAL.md, "Fase 2"): queries like
`MinimumPSR`/`MPSR` need to answer "does there exist a smaller/better
candidate?" - i.e. prove the ABSENCE of a counterexample. The exact
brute-force evaluator (`SigmaEvaluator.evaluate_exact`) can do this but pays
the full O(3^n) domain; the grow-shrink heuristic search (`SigmaSearch.local_search`,
"Fase 1") structurally cannot do it at all - it can only ever propose
witnesses for EXISTENCE, never prove non-existence, and its violation
scorer (`SigmaSearch.soft_violation`) only covers a pure-conjunction fragment with no
`NOT`, no nested comparisons, no probabilistic functions.

Z3 fixes the "prove absence" problem for the part of the query it can
actually reason about: the tuple-algebra *skeleton* (`<<=`, `<<`, `FULL`,
`L0`, `L1`/`FEAT`, `SIZE`, `PARTITION`, `MEET`, `JOIN`, `OPPOSITE`, `SUF`,
`COMP`, `NOT`, `*`, and numeric `==`/`<=`/`>=` over `SIZE` and `FREE`) - the same
fragment `SigmaSearch.soft_violation` already recognizes, just interpreted here as Z3
constraints instead of a float violation score. It still cannot reason
about `VAL`/`COND`/`EP`/`CEP` (the actual PyJuice circuit) at all - those
stay exactly what they always were: an oracle, called through the *real*
`QFOILEvaluator.evaluate()`, once per Z3-proposed candidate.

This is why `Z3CegisSolver` never hooks into `QFOILEvaluator.evaluate_function`'s
dispatch table: it doesn't evaluate a fixed formula, it *searches* for a
`var` satisfying a cheap Z3-checked `skeleton_expr`, verified by an exact
`oracle_expr` (which may itself use anything the plain evaluator supports,
including nested SIGMA - the oracle is never touched by the Z3 encoder).

`find_argopt`'s optional `warm_start` (off by default) is how Fase 1
"informs" this solver rather than leaving it to search blind from
scratch: `QFOILEvaluator.sigma.search.soft_violation` (Fase 1) learned to score
numeric `>=`/`<=`/`==` comparisons by evaluating both sides with the real
evaluator, so it can now attempt the *entire* `(skeleton) * (oracle)`
conjunction directly - including `EP`/`CEP`/`VAL` threshold conjuncts,
not just tuple algebra. When that succeeds, the verified witness turns
`minimize`/`maximize` into a tighter question for Z3 - "does anything
strictly beat this known-achievable value?" - instead of an unconstrained
search that has to individually discover and reject every worse
candidate on the way there. Fase 1 still can't do what Z3 does here: it
can find *a* witness fast, never prove none-better exists, or that the
skeleton is UNSAT - that half of the job stays Z3's alone.
"""

from typing import Any, Dict, Optional, Tuple

from ..language.ast import (
    ASTNode,
    BinaryOp,
    FunctionCall,
    NumberLiteral,
    Sigma,
    UnaryOp,
    Variable,
)

# Tuple-algebra relations this module can translate into Z3 constraints -
# exactly the fragment `SigmaSearch.VIOLATION_FUNCTIONS` covers for Fase 1,
# reused here as the boundary of what the skeleton encoder understands.
_SKELETON_FUNCS = {
    "FULL",
    "L0",
    "L1",
    "FEAT",
    "PARTITION",
    "MEET",
    "JOIN",
    "OPPOSITE",
    "SUF",
    "COMP",
}


def _concrete_to_bits(t: tuple) -> Tuple[int, int]:
    """Python tuple over {0, 1, None} -> (value_int, mask_int).

    mask bit i = 1 iff position i is defined; value bit i is only
    meaningful where mask bit i = 1 (mirrors Z3_Sandbox/sandbox.ipynb's
    `partial_instance_to_bitvec`).
    """
    value = mask = 0
    for i, v in enumerate(t):
        if v is not None:
            mask |= 1 << i
            if v:
                value |= 1 << i
    return value, mask


def _bits_to_concrete(value: int, mask: int, n: int) -> tuple:
    """Inverse of `_concrete_to_bits`."""
    return tuple(
        ((value >> i) & 1) if ((mask >> i) & 1) else None for i in range(n)
    )


class Z3CegisSolver:
    """CEGIS search over an existing evaluator/parser pair.

    Does not modify or wrap either of them - it only ever calls
    `evaluator.evaluate(...)`, the same call any other code makes, to
    verify candidates it proposes. `z3` is imported lazily in `__init__`
    so constructing anything else in this package never requires it.
    """

    def __init__(self, evaluator, parser):
        try:
            import z3
        except ImportError as e:
            raise RuntimeError(
                "Z3CegisSolver requires the 'z3-solver' package "
                "(pip install z3-solver)."
            ) from e
        self._z3 = z3
        self.evaluator = evaluator
        self.parser = parser
        # Set by find_argopt on every return - see its docstring. Exists
        # because `None` alone can't tell a caller "proved no witness
        # exists" (UNSAT - trustworthy) apart from "gave up at max_iters"
        # (exhausted - not a proof of anything). Conflating the two was a
        # real gap: any code meaning to *trust* a `None`/witness result
        # (e.g. treating it as evidence for a universal statement) needs
        # this distinction, not just the return value.
        self.last_status: Optional[str] = None

    # ------------------------------------------------------------------
    # AST -> Z3 encoding (the "skeleton" fragment only)
    # ------------------------------------------------------------------

    def _encode_tuple(
        self,
        ast: ASTNode,
        var: str,
        sym: Tuple[Any, Any],
        context: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """Encode a tuple-valued AST node as a (value_expr, mask_expr) pair."""
        if isinstance(ast, Variable):
            if ast.name == var:
                return sym
            if ast.name not in context:
                raise ValueError(f"Undefined variable in skeleton: {ast.name}")
            value_int, mask_int = _concrete_to_bits(context[ast.name])
            n = sym[0].size()
            return self._z3.BitVecVal(value_int, n), self._z3.BitVecVal(mask_int, n)

        raise ValueError(f"Cannot encode as a tuple: {ast}")

    def _encode_num(
        self,
        ast: ASTNode,
        var: str,
        sym: Tuple[Any, Any],
        context: Dict[str, Any],
    ):
        """Encode a numeric-valued AST node as a Z3 arithmetic expression."""
        z3 = self._z3

        if isinstance(ast, NumberLiteral):
            return ast.value

        if isinstance(ast, Variable):
            if ast.name in context and not isinstance(context[ast.name], tuple):
                return context[ast.name]
            raise ValueError(f"Undefined numeric variable in skeleton: {ast.name}")

        if isinstance(ast, FunctionCall) and ast.function_name.upper() == "SIZE":
            (x_ast,) = ast.arguments
            _, mask = self._encode_tuple(x_ast, var, sym, context)
            n = mask.size()
            return z3.Sum(
                [z3.If(z3.Extract(i, i, mask) == 1, 1, 0) for i in range(n)]
            )

        if isinstance(ast, FunctionCall) and ast.function_name.upper() == "FREE":
            x_ast, y_ast = ast.arguments
            vx, mx = self._encode_tuple(x_ast, var, sym, context)
            vy, my = self._encode_tuple(y_ast, var, sym, context)
            n = mx.size()

            # FREE(x, y) counts positions defined by x whose value is no
            # longer preserved by y. A position is therefore free when x is
            # defined and either y is undefined or y has the opposite value.
            free_mask = mx & (~my | (vx ^ vy))
            return z3.Sum(
                [
                    z3.If(z3.Extract(i, i, free_mask) == 1, 1, 0)
                    for i in range(n)
                ]
            )

        raise ValueError(f"Cannot encode as a number: {ast}")

    def _encode_bool(
        self,
        ast: ASTNode,
        var: str,
        sym: Tuple[Any, Any],
        context: Dict[str, Any],
    ):
        """Encode a Boolean-valued AST node as a Z3 BoolRef.

        Supports exactly the fragment described at the top of this module:
        `NOT`, `*` (AND), `<<=`/`<<` (SUBSETEQ/SUBSET), `==`/`<=`/`>=` over
        numeric terms (`SIZE(...)` and `FREE(...)`), and the relations in
        `_SKELETON_FUNCS`. Anything else (nested SIGMA, `+`, probabilistic
        functions) raises `ValueError` - callers should keep those out of
        `skeleton_expr` and put them in `oracle_expr` instead, where the
        exact evaluator handles them with no restriction.
        """
        z3 = self._z3

        if isinstance(ast, UnaryOp) and ast.operator.upper() in ("NOT", "!"):
            return z3.Not(self._encode_bool(ast.operand, var, sym, context))

        if isinstance(ast, BinaryOp):
            op = ast.operator.upper()

            if op in ("SUBSETEQ", "<<="):
                va, ma = self._encode_tuple(ast.left, var, sym, context)
                vb, mb = self._encode_tuple(ast.right, var, sym, context)
                return z3.And((ma & mb) == ma, (va & ma) == (vb & ma))

            if op in ("SUBSET", "<<"):
                va, ma = self._encode_tuple(ast.left, var, sym, context)
                vb, mb = self._encode_tuple(ast.right, var, sym, context)
                subseteq = z3.And((ma & mb) == ma, (va & ma) == (vb & ma))
                return z3.And(subseteq, z3.Or(va != vb, ma != mb))

            if ast.operator == "*":
                return z3.And(
                    self._encode_bool(ast.left, var, sym, context),
                    self._encode_bool(ast.right, var, sym, context),
                )

            if ast.operator in ("==", "<=", ">="):
                left = self._encode_num(ast.left, var, sym, context)
                right = self._encode_num(ast.right, var, sym, context)
                if ast.operator == "==":
                    return left == right
                if ast.operator == "<=":
                    return left <= right
                return left >= right

            raise ValueError(f"Unsupported operator in skeleton: {ast.operator}")

        if isinstance(ast, FunctionCall):
            fname = ast.function_name.upper()

            if fname not in _SKELETON_FUNCS:
                raise ValueError(
                    f"Unsupported function in skeleton: {fname} "
                    "(probabilistic functions belong in oracle_expr, not "
                    "skeleton_expr)"
                )

            args = [
                self._encode_tuple(a, var, sym, context) for a in ast.arguments
            ]
            n = sym[0].size()

            if fname == "FULL":
                ((_, mask),) = args
                return mask == z3.BitVecVal((1 << n) - 1, n)

            if fname == "L0":
                ((_, mask),) = args
                return mask == z3.BitVecVal(0, n)

            if fname in ("L1", "FEAT"):
                ((_, mask),) = args
                popcount = z3.Sum(
                    [z3.If(z3.Extract(i, i, mask) == 1, 1, 0) for i in range(n)]
                )
                return popcount == 1

            if fname == "PARTITION":
                (_, ma), (_, mb) = args
                return (ma & mb) == z3.BitVecVal(0, n)

            if fname == "MEET":
                (vx, mx), (vy, my), (vz, mz) = args
                mask_meet = mx & my & ~(vx ^ vy)
                val_meet = vx & mask_meet
                return z3.And(vz == val_meet, mz == mask_meet)

            if fname == "JOIN":
                (vx, mx), (vy, my), (vz, mz) = args
                compatible = ((vx ^ vy) & mx & my) == z3.BitVecVal(0, n)
                mask_join = mx | my
                val_join = (vx & mx) | (vy & my)
                return z3.And(compatible, vz == val_join, mz == mask_join)

            if fname == "OPPOSITE":
                (vx, mx), (vy, my) = args
                shared = mx & my
                shared_count = z3.Sum(
                    [z3.If(z3.Extract(i, i, shared) == 1, 1, 0) for i in range(n)]
                )
                same_support = mx == my
                differ = (vx & shared) != (vy & shared)
                return z3.And(same_support, shared_count == 1, differ)

            if fname == "SUF":
                (_, mx), (_, my) = args
                return mx == my

            if fname == "COMP":
                (vx, mx), (vy, my) = args
                subseteq = z3.And((mx & my) == mx, (vx & mx) == (vy & mx))
                return z3.And(subseteq, my == z3.BitVecVal((1 << n) - 1, n))

        raise ValueError(f"Unsupported node in skeleton: {ast}")

    # ------------------------------------------------------------------
    # Fase 1 warm start (optional, off by default)
    # ------------------------------------------------------------------

    def _warm_start_witness(
        self,
        var: str,
        skeleton_expr: str,
        oracle_expr: str,
        context: Dict[str, Any],
        verbose: bool,
        fase1_budget: Optional[int] = None,
    ) -> Optional[tuple]:
        """Try the evaluator's Fase 1 searches on `(skeleton) * (oracle) >=
        1` before involving Z3 at all.

        This only works now because `SigmaSearch.soft_violation` (Fase 1) learned to
        score numeric `>=`/`<=`/`==` comparisons by calling the real
        evaluator on both sides - so `oracle_expr` shapes like
        `EP(y, k) >= delta` no longer disable it. Still fully optional and
        sound: `None` (unsupported fragment, budget exhausted, or a parse
        error) just means `find_argopt` proceeds exactly as if warm start
        didn't exist. Any witness returned here has already been
        re-verified against the exact evaluator, not just the heuristic
        score.

        Tries `SigmaSearch.greedy_ascent` first: for the "informing a
        `minimize`/`maximize` search" use case this warm start exists
        for, growing a witness (its only mode) is exactly the right
        shape, and it doesn't waste budget re-probing positions a `seed`
        already had right (unlike `SigmaSearch.local_search`'s UCB1 bandit,
        which resets its explore/exploit state on every fresh call - see
        SIGMA_EVALUATION_PROPOSAL.md). Falls back to
        `SigmaSearch.local_search`'s more general grow-shrink search only if
        the greedy ascent finds nothing, so anything it used to catch it
        still catches.
        """
        try:
            combined_ast = self.parser.parse(
                f"(({skeleton_expr}) * ({oracle_expr}))"
            )
        except Exception:
            return None

        reference = self.evaluator.sigma.domain_template()
        sigma_ast = Sigma(var, combined_ast)
        budget = (
            fase1_budget if fase1_budget is not None
            else self.evaluator.sigma.search_budget
        )

        witness = self.evaluator.sigma.search.greedy_ascent(
            sigma_ast, context, reference, budget
        )
        engine = "greedy ascent"
        if witness is None:
            witness = self.evaluator.sigma.search.local_search(
                sigma_ast, context, reference, budget
            )
            engine = "local search"
        if witness is None:
            return None

        verified = self.evaluator.evaluate(combined_ast, {**context, var: witness})
        verified_true = (
            verified > 0 if isinstance(verified, (int, float)) else bool(verified)
        )
        if not verified_true:
            return None

        if verbose:
            print(f"[warm start] Fase 1 ({engine}) found and verified witness {witness}")
        return witness

    # ------------------------------------------------------------------
    # CEGIS loop
    # ------------------------------------------------------------------

    def find_argopt(
        self,
        var: str,
        skeleton_expr: str,
        oracle_expr: str,
        context: Dict[str, Any],
        minimize: Optional[str] = None,
        maximize: Optional[str] = None,
        max_iters: int = 100,
        verbose: bool = False,
        warm_start: bool = False,
        fase1_budget: Optional[int] = None,
    ) -> Optional[tuple]:
        """Find a value for `var` satisfying `skeleton_expr` and `oracle_expr`.

        `skeleton_expr` and (if given) `minimize`/`maximize` are parsed
        with the same QFOIL parser as everything else and translated to
        Z3 constraints (see `_encode_bool`/`_encode_num`) - they must stay
        inside the fragment documented at the top of this module.

        `oracle_expr` is parsed the same way but never touches Z3: each
        candidate Z3 proposes is substituted for `var` and checked with
        `self.evaluator.evaluate(...)`, the exact evaluator, so it may use
        anything the evaluator supports (`EP`, `CEP`, `VAL`, `COND`, nested
        SIGMA, `L_*` functions, ...).

        `warm_start` (off by default, to keep existing callers' iteration
        counts unchanged): before building the Z3 problem, try the
        evaluator's Fase 1 heuristic on `(skeleton) * (oracle) >= 1`. If it
        finds and verifies a witness `w` with objective value `v`:
        - with no `minimize`/`maximize`, `w` is returned directly and Z3
          never runs - `find_argopt` without an objective just returns
          *a* satisfying value, and the warm-started one already
          qualifies;
        - with `minimize`/`maximize`, instead of handing Z3 the original
          problem, this recurses *once* (with `warm_start=False`, so no
          further recursion) on the skeleton tightened to demand
          something strictly better than `v` (`(minimize) < v` /
          `(maximize) > v`). If that inner search finds one, it's
          returned (genuinely better than `w`); if it proves none exists
          (UNSAT) or exhausts `max_iters` without finding one, `w` itself
          is returned. This is the actual point of warm starting a
          `minimize`/`maximize` search: "is there anything better than
          `v`?" is one tighter question for Z3 to dispatch - often a
          single UNSAT check - instead of `Optimize` walking the
          objective up from its own unconstrained best (which may need
          many individually-blocked, individually oracle-rejected
          candidates to get past before ever reaching `v`).

        Returns the first (or, with `minimize`/`maximize`, the Z3-optimal
        among tried) candidate tuple for which both hold, or `None` if the
        skeleton is exhausted (UNSAT) or `max_iters` is reached without a
        verified candidate.

        Also sets `self.last_status`, one of:
        - `"proved_optimal"`: a witness is returned, and (since `minimize`/
          `maximize` was given) Z3 has proven nothing satisfying both
          `skeleton_expr` and `oracle_expr` scores better - true whether
          that proof came from the main loop's own blocking history or
          from `warm_start`'s "is there anything better?" sub-search
          hitting real UNSAT.
        - `"found"`: a witness is returned, but with no proof of
          optimality - either there was no `minimize`/`maximize` to be
          optimal about, or the "anything better?" sub-search ran out of
          `max_iters` without resolving (inconclusive, not disproof).
        - `"unsat"`: `None` is returned, and it's a real proof - no value
          of `var` satisfies `skeleton_expr` at all. Trustworthy for
          e.g. checking a universal statement by searching for a
          counterexample: `unsat` here means none exists.
        - `"exhausted"`: `None` is returned, but `max_iters` simply ran
          out - some value might still satisfy both `skeleton_expr` and
          `oracle_expr`, this search just didn't find it in time. Never
          treat this the same as `"unsat"`.

        `fase1_budget` (only meaningful when `warm_start=True`) overrides
        `self.evaluator.sigma.search_budget` for the warm-start attempt
        only, without mutating shared evaluator state - `None` (default)
        just uses the evaluator's own configured budget.
        """
        if warm_start:
            seed_witness = self._warm_start_witness(
                var, skeleton_expr, oracle_expr, context, verbose, fase1_budget
            )
            if seed_witness is not None:
                # The language has no strict `<`/`>` (only `==`/`<=`/`>=`,
                # see qfoil/language/parser.py's tokenizer) - "strictly better" is
                # expressed as the negation of "not better": minimize
                # wants NOT(objective >= bound) i.e. objective < bound;
                # maximize wants NOT(objective <= bound) i.e. objective >
                # bound. This clause only ever reaches `_encode_bool`
                # (which does handle NOT), never `SigmaSearch.soft_violation` again -
                # the recursive call below passes warm_start=False.
                if minimize is not None:
                    objective, not_better_op = minimize, ">="
                elif maximize is not None:
                    objective, not_better_op = maximize, "<="
                else:
                    if verbose:
                        print(
                            "[warm start] no minimize/maximize given - "
                            "returning the warm-start witness directly"
                        )
                    self.last_status = "found"
                    return seed_witness

                bound_value = self.evaluator.evaluate(
                    self.parser.parse(objective), {**context, var: seed_witness}
                )
                if verbose:
                    print(
                        f"[warm start] seed {seed_witness} scores "
                        f"{objective}={bound_value}; searching for "
                        f"something strictly better"
                    )
                improved = self.find_argopt(
                    var=var,
                    skeleton_expr=(
                        f"({skeleton_expr}) * "
                        f"NOT(({objective}) {not_better_op} {bound_value})"
                    ),
                    oracle_expr=oracle_expr,
                    context=context,
                    minimize=minimize,
                    maximize=maximize,
                    max_iters=max_iters,
                    verbose=verbose,
                    warm_start=False,
                )
                if improved is not None:
                    # self.last_status was already set correctly by the
                    # recursive call above (it's the same object).
                    return improved
                # self.last_status here is whatever the recursive "is
                # there better?" call left behind: "unsat" means Z3
                # proved nothing beats the seed - the seed itself is
                # therefore proved optimal, not just "the best we found".
                # "exhausted" means that sub-search simply ran out of
                # max_iters - the seed is still a valid witness, but its
                # optimality was never established.
                proved_optimal = self.last_status == "unsat"
                self.last_status = "proved_optimal" if proved_optimal else "found"
                if verbose:
                    verdict = "proved optimal" if proved_optimal else "not proven optimal"
                    print(
                        f"[warm start] nothing improves on {objective}="
                        f"{bound_value} - returning the seed ({verdict})"
                    )
                return seed_witness

        z3 = self._z3

        reference = self.evaluator.sigma.domain_template()
        n = len(reference)
        sym = (z3.BitVec(f"{var}__value", n), z3.BitVec(f"{var}__mask", n))

        skeleton_ast = self.parser.parse(skeleton_expr)
        oracle_ast = self.parser.parse(oracle_expr)

        opt = z3.Optimize()
        opt.add(self._encode_bool(skeleton_ast, var, sym, context))

        if minimize is not None:
            term = self._encode_num(self.parser.parse(minimize), var, sym, context)
            opt.minimize(term)
        elif maximize is not None:
            term = self._encode_num(self.parser.parse(maximize), var, sym, context)
            opt.maximize(term)

        for iteration in range(max_iters):
            if opt.check() != z3.sat:
                if verbose:
                    print(f"[iter {iteration}] UNSAT - skeleton exhausted")
                self.last_status = "unsat"
                return None

            model = opt.model()
            value_int = model[sym[0]].as_long()
            mask_int = model[sym[1]].as_long()
            candidate = _bits_to_concrete(value_int, mask_int, n)

            verdict = self.evaluator.evaluate(oracle_ast, {**context, var: candidate})
            ok = verdict > 0 if isinstance(verdict, (int, float)) else bool(verdict)

            if verbose:
                print(f"[iter {iteration}] candidate {candidate} -> oracle={verdict}")

            if ok:
                # Z3's Optimize() proves this candidate optimal among
                # everything not yet blocked, and every blocked candidate
                # was individually confirmed oracle-rejected above - so
                # when there's an objective, this really is the global
                # optimum, not just "the first one that happened to work".
                self.last_status = (
                    "proved_optimal" if minimize is not None or maximize is not None
                    else "found"
                )
                return candidate

            # Counterexample: block this exact semantic tuple and retry. Bits
            # of value_int outside mask_int are "don't care" - Z3 is free to
            # pick anything there, so blocking on raw (value, mask) would let
            # it "escape" by flipping an irrelevant bit while proposing the
            # very same tuple again. Mask value_int before comparing.
            opt.add(
                z3.Or(
                    (sym[0] & z3.BitVecVal(mask_int, n))
                    != z3.BitVecVal(value_int & mask_int, n),
                    sym[1] != z3.BitVecVal(mask_int, n),
                )
            )

        if verbose:
            print(f"Max iterations ({max_iters}) reached without a verified candidate.")
        self.last_status = "exhausted"
        return None

    # ------------------------------------------------------------------
    # Universal (forall) orchestration
    # ------------------------------------------------------------------

    def check_universal(
        self,
        var: str,
        formula_expr: str,
        context: Dict[str, Any],
        skeleton_expr: str,
        fase1_budget: Optional[int] = None,
        max_iters: int = 100,
        verbose: bool = False,
    ) -> Tuple[str, Optional[tuple]]:
        """Check `forall var (in skeleton_expr). formula_expr` honestly.

        This is the "orchestrator" role Z3 plays for universal-shaped
        SIGMA queries (`SIGMA z. formula(z) == 0` after De Morgan, i.e.
        `forall z. NOT(formula(z))` restated as `forall z. formula'(z)` -
        see SIGMA_EVALUATION_PROPOSAL.md's "Orquestación de universales"
        plan). We deliberately do NOT try to prove or refute the universal
        by iterating candidates - over any real tuple domain that's
        `3^n`, exponential, and exactly the cost this whole module exists
        to avoid. Instead:

        1. Cheap first: search for a COUNTEREXAMPLE to `formula_expr`
           within `skeleton_expr` using the evaluator's Fase 1 heuristics
           (via `find_argopt`'s own `warm_start`). If one exists and Fase
           1 happens to find it, that's an exact, fully-verified refutation
           - no need to ever involve Z3's CEGIS loop.
        2. Otherwise, let Z3 search for that same counterexample
           (`NOT(formula_expr)` inside `skeleton_expr`) via CEGIS. Two
           honest outcomes:
           - genuine UNSAT (`last_status == "unsat"`): no counterexample
             exists anywhere in `skeleton_expr` - the universal is
             PROVEN, full stop.
           - `max_iters` exhausted (`last_status == "exhausted"`): we
             don't know. This is reported as `"unknown"`, never as a
             proof either way - trusting a `None` result here would mean
             trusting the caller's `delta`/`k`/time budget, which is
             exactly what the "no false claims" rule in this module says
             not to do silently.

        Returns `(status, witness)`:
        - `("disproved", witness)`: `witness` fails `formula_expr` (the
          universal is false) - a genuine counterexample, verified by the
          exact evaluator, not just Z3's skeleton encoding.
        - `("proved", None)`: Z3 proved no counterexample exists - the
          universal holds over the entire `skeleton_expr` domain.
        - `("unknown", None)`: neither Fase 1 nor Z3 (within `max_iters`)
          settled it. Not evidence for or against the universal - just an
          honest "ran out of budget".
        """
        witness = self.find_argopt(
            var=var,
            skeleton_expr=skeleton_expr,
            oracle_expr=f"NOT(({formula_expr}))",
            context=context,
            max_iters=max_iters,
            verbose=verbose,
            warm_start=True,
            fase1_budget=fase1_budget,
        )

        if witness is not None:
            return "disproved", witness

        if self.last_status == "unsat":
            return "proved", None

        # self.last_status == "exhausted" - budget ran out, no proof either way.
        return "unknown", None
__all__ = ["Z3CegisSolver"]
