"""Tests for exact SIGMA evaluation and heuristic witness search."""

import pytest

from qfoil import QFOIL, Inconclusive


def parse_and_evaluate(expression, context, *, mode="exact"):
    """Evaluate through the canonical API with an explicitly sized runtime.

    Defaults to ``mode="exact"``: most call sites here are ground-truth
    semantic checks on small domains. Call sites that specifically exercise
    the heuristic search succeeding on a large domain pass
    ``mode="heuristic"`` explicitly.
    """
    instance_len = len(
        next(value for value in context.values() if isinstance(value, tuple))
    )
    return QFOIL(instance_len=instance_len, sigma_mode=mode).evaluate(
        expression, bindings=context
    )


class TestSigmaExpressions:
    """Test SIGMA quantification expressions."""

    @pytest.mark.parametrize(
        "context,expected,description",
        [
            ({"x": (1, 0)}, 1, "Full tuples have no proper supersets"),
            ({"x": (1, 0, None)}, 0, "Partial tuples have proper supersets"),
            ({"x": (None, None, None)}, 0, "All None tuples have proper supersets"),
        ],
    )
    def test_sigma_superset_check(self, context, expected, description):
        """Test SIGMA checking for proper supersets: ((x <<= y) * NOT (x == y)) == 0."""
        expr = "SIGMA y. ((x <<= y) * NOT (x == y)) == 0"
        result = parse_and_evaluate(expr, context)
        assert result == expected  # description

    @pytest.mark.parametrize(
        "context,expected,description",
        [
            (
                {"x": (None, None, None)},
                1,
                "Empty tuple pattern has only itself as subset",
            ),
            ({"x": (1, 0, None)}, 0, "Non-empty tuples have multiple subsets"),
        ],
    )
    @pytest.mark.parametrize("separator", [".", ""])
    def test_sigma_subset_count_check(self, context, expected, description, separator):
        """Test SIGMA checking if tuple has exactly one subset: (y <<= x) == 1."""
        expr = f"SIGMA y{separator} (y <<= x) == 1"
        result = parse_and_evaluate(expr, context)
        assert result == expected  # description

    def test_sigma_meet_operation(self):
        """Test SIGMA with MEET operation."""
        expr = (
            "(z <<= x) * (z <<= y) * (SIGMA w. (w <<= x) * (w <<= y) * "
            "(w <<= z) >= SIGMA w. (w <<= x) * (w <<= y))"
        )
        context = {"x": (1, 0, None), "y": (1, None, 1), "z": (1, None, None)}
        result = parse_and_evaluate(expr, context)
        assert result == 1  # z is the meet of x and y

    @pytest.mark.parametrize(
        "context,expected,description",
        [
            (
                {"x": (1, 0, None), "y": (1, None, None)},
                1,
                "Compatible tuples have a join",
            ),
            (
                {"x": (0, 0, None), "y": (1, None, None)},
                0,
                "Incompatible tuples have no join",
            ),
        ],
    )
    def test_sigma_join_existence(self, context, expected, description):
        """Test SIGMA for JOIN existence: (FULL(z) * (x <<= z) * (y <<= z)) >= 1."""
        expr = "SIGMA z. (FULL(z) * (x <<= z) * (y <<= z)) >= 1"
        result = parse_and_evaluate(expr, context)
        assert result == expected  # description


class TestSigmaEarlyExit:
    """Test the early-exit optimization for `SIGMA ... OP threshold`.

    `SigmaEvaluator.decide` must remain exact (same result as the brute-force
    oracle would give) while stopping enumeration as soon as the
    (monotonically non-decreasing) running count decides the comparison.
    The n=200 cases below are the real point: at that domain size, naive
    enumeration (3**200 candidates) is not just slow, it's physically
    impossible - these only terminate because early exit makes them O(1).
    """

    def test_geq_true_resolves_without_full_domain_scan(self):
        """`>= 1` should decide True on the very first candidate.

        itertools.product's first candidate is all-zeros, which is FULL,
        so `SIGMA z. FULL(z) >= 1` is decided immediately regardless of n.
        """
        result = parse_and_evaluate("SIGMA z. FULL(z) >= 1", {"x": (0,) * 200})
        assert result == 1

    def test_leq_false_resolves_without_full_domain_scan(self):
        """`<= 0` should decide False as soon as the first candidate hits."""
        result = parse_and_evaluate("SIGMA z. FULL(z) <= 0", {"x": (0,) * 200})
        assert result == 0

    def test_mirrored_operator_threshold_on_left(self):
        """`threshold <= SIGMA ...` mirrors to `SIGMA ... >= threshold`."""
        result = parse_and_evaluate("1 <= SIGMA z. FULL(z)", {"x": (0,) * 200})
        assert result == 1

    def test_mirrored_operator_threshold_on_left_false(self):
        # n=0: the only candidate is the empty tuple, vacuously FULL -> count == 1
        result = parse_and_evaluate("1 >= SIGMA z. FULL(z)", {"x": ()})
        assert result == 1

        # n=3: 2**3 = 8 FULL candidates ({0,1}^3), so `1 >= 8` is False
        result = parse_and_evaluate("1 >= SIGMA z. FULL(z)", {"x": (0,) * 3})
        assert result == 0

    def test_both_sides_sigma(self):
        """Both operands SIGMA (the shape L_SUF's body uses)."""
        expr = "(SIGMA u. SIGMA v. FULL(u)) == (SIGMA u. SIGMA v. FULL(u))"
        assert parse_and_evaluate(expr, {"x": (0, 0, 0)}) == 1

        expr_diff = "(SIGMA u. SIGMA v. FULL(u)) == (SIGMA u. FULL(u))"
        assert parse_and_evaluate(expr_diff, {"x": (0, 0, 0)}) == 0

    def test_negative_summand_is_rejected(self):
        """Early exit assumes non-negative summands; guard must reject them.

        The grammar itself has no unary minus, so a negative summand can't
        be produced by parsing user input - build the AST directly to
        exercise the guard.
        """
        from qfoil.language import NumberLiteral, Sigma

        evaluator = QFOIL(sigma_mode="exact").evaluator
        sigma_ast = Sigma("z", NumberLiteral(-1))

        with pytest.raises(ValueError, match="non-negative summands"):
            evaluator.sigma.decide(sigma_ast, ">=", 1, {"x": (0, 0)})

    @pytest.mark.parametrize("n", range(0, 5))
    def test_matches_brute_force_across_all_thresholds(self, n):
        """Differential test: early-exit decision vs. the exact count.

        For every threshold and every comparison operator, the early-exit
        result must equal `(exact SIGMA count) OP threshold` - i.e. early
        exit changes nothing about correctness, only how much of the
        domain gets enumerated to reach it.
        """
        full_count = 2**n  # number of FULL candidates in {0,1,None}^n
        context = {"x": (0,) * n}

        for k in range(0, full_count + 2):
            for op, py_op in [
                (">=", lambda a, b: a >= b),
                ("<=", lambda a, b: a <= b),
                ("==", lambda a, b: a == b),
            ]:
                expr = f"SIGMA z. FULL(z) {op} {k}"
                got = parse_and_evaluate(expr, context)
                expected = 1 if py_op(full_count, k) else 0
                assert got == expected, f"{expr} with n={n}: got {got}, want {expected}"


class TestSigmaLocalSearch:
    """Test the Fase 1 grow-shrink heuristic witness search.

    The heuristic is only ever a fast path in front of the exact
    evaluator: it must never change the *result* of a query, only how
    quickly it's reached. Every test here holds regardless of whether
    `sigma_search_budget` is on (default) or off (0).
    """

    def _adversarial_expr_and_context(self, n):
        """`SIGMA z. (FULL(z) * (k <<= z)) >= 1` with a single witness (z==k)
        sitting at canonical-order index 3**(n-1) - i.e. Fase 0 alone has to
        enumerate roughly a third of the domain to reach it, but grow-shrink
        should nail it in ~O(n) evaluations by greedily matching k position
        by position.
        """
        k = (1,) + (0,) * (n - 1)
        return "SIGMA z. (FULL(z) * (k <<= z)) >= 1", {"k": k}

    def test_finds_witness_far_from_canonical_order(self):
        """n=30: witness sits at index 3**29 (~6.9e13) - unreachable by
        canonical-order scanning alone, but heuristic search resolves it.
        """
        expr, context = self._adversarial_expr_and_context(30)
        assert parse_and_evaluate(expr, context, mode="heuristic") == 1

    def test_agrees_with_exact_reference_when_mode_is_exact(self):
        """Same query, resolved via `sigma_mode="exact"` instead of the
        heuristic: proves the heuristic isn't the thing making the answer
        right, only how fast it's reached when it succeeds -- the exact
        path is a separately selected, equally correct reference.
        """

        expr, context = self._adversarial_expr_and_context(6)
        parser = QFOIL(instance_len=6, sigma_mode="exact")
        ast = parser.parse(expr)
        assert parser.evaluator.evaluate(ast, context) == 1

    def test_unsupported_fragment_is_honestly_inconclusive_in_heuristic_mode(self):
        """A formula outside the v1 fragment (here: `+`, used to combine
        two relations as if it were boolean OR - NOT wraps this and
        inherits its unsupported status too, see
        test_soft_violation_unsupported_returns_none) makes
        `SigmaSearch.soft_violation` bail out (None) rather than guess.
        Heuristic mode must report that honestly as Inconclusive instead of
        silently falling through to the exact scan -- only `sigma_mode=
        "exact"`, selected explicitly, resolves it (ground truth: 1).
        """

        expr = "SIGMA z. NOT(FULL(z) + L0(z)) >= 1"
        context = {"x": (0, 0, 0)}

        parser_heuristic = QFOIL()
        parser_exact = QFOIL(sigma_mode="exact")

        result_heuristic = parser_heuristic.evaluator.evaluate(
            parser_heuristic.parse(expr), context
        )
        result_exact = parser_exact.evaluator.evaluate(
            parser_exact.parse(expr), context
        )
        assert isinstance(result_heuristic, Inconclusive)
        assert result_exact == 1  # e.g. z=(None,None,None) satisfies

    def test_spurious_witness_is_rejected_without_enumerating(self):
        """If the heuristic ever proposed a candidate that doesn't really
        satisfy the formula, `SigmaEvaluator.decide` must catch it via
        independent re-verification and report Inconclusive -- never trust
        it blind, and never silently fall through to an exact scan to
        rescue the answer (that scan is exactly what heuristic mode exists
        to avoid paying). `sigma_mode="exact"` remains the way to get the
        real ground truth, unaffected by the patched heuristic since exact
        mode never consults it.
        """
        parser = QFOIL()
        evaluator = parser.evaluator
        evaluator.sigma.search.local_search = lambda *a, **k: (1, 1, 1)  # bogus

        # FULL(z) >= 1 is actually true (e.g. z=(0,0,0)), but the witness
        # the (patched) heuristic proposes, (1,1,1), doesn't satisfy the
        # *real* formula used here: only L0(z) do, and (1,1,1) isn't L0.
        ast = parser.parse("SIGMA z. L0(z) >= 1")
        result = evaluator.evaluate(ast, {"x": (0, 0, 0)})
        assert isinstance(result, Inconclusive)

        exact_parser = QFOIL(sigma_mode="exact")
        exact_result = exact_parser.evaluator.evaluate(
            exact_parser.parse("SIGMA z. L0(z) >= 1"), {"x": (0, 0, 0)}
        )
        assert exact_result == 1  # z=(None,None,None) is L0

    @pytest.mark.parametrize(
        "func_name,args,expected_violation",
        [
            ("FULL", [(1, 0, 1)], 0),
            ("FULL", [(1, None, 1)], 1),
            ("L0", [(None, None, None)], 0),
            ("L0", [(1, None, None)], 1),
            ("L1", [(1, None, None)], 0),
            ("L1", [(None, None, None)], 1),
            ("L1", [(1, 0, None)], 1),
            ("FEAT", [(1, None, None)], 0),
            ("PARTITION", [(1, None, None), (None, 0, None)], 0),
            ("PARTITION", [(1, 0, None), (None, 0, None)], 1),
            ("OPPOSITE", [(1, None, None), (0, None, None)], 0),
            ("SUF", [(1, None, 1), (0, None, 1)], 0),
            ("SUF", [(1, None, 1), (0, 1, 1)], 1),
        ],
    )
    def test_violation_scores_are_zero_iff_satisfied(
        self, func_name, args, expected_violation
    ):
        """`_violation_for_native` must be 0 exactly when the real native
        function would return 1 (satisfied), matching the exact evaluator.
        """

        evaluator = QFOIL().evaluator
        violation = evaluator.sigma.search.violation_for_native(func_name, args)
        assert violation == expected_violation

        native_result = evaluator.function_runtime.evaluate_native(func_name, args)
        assert (violation == 0) == (native_result == 1)

    def test_soft_violation_unsupported_returns_none(self):
        """`+` is outside the v1 fragment - and so is anything under NOT
        that bottoms out on `+` (NOT recurses into whatever
        `SigmaSearch.soft_violation` already supports - a compound `*` or even a
        nested NOT both resolve fine via that recursion, see
        test_soft_violation_negates_a_single_leaf and
        test_not_recurses_into_compound_subformulas below - so `+` is the
        actual boundary, not NOT itself)."""
        from qfoil.language import UnaryOp, FunctionCall, Variable, BinaryOp

        evaluator = QFOIL().evaluator
        context = {"z": (1, 0, None)}

        full_plus_full = BinaryOp(
            FunctionCall("FULL", [Variable("z")]),
            "+",
            FunctionCall("FULL", [Variable("z")]),
        )
        assert evaluator.sigma.search.soft_violation(full_plus_full, context) is None

        not_of_plus = UnaryOp("!", full_plus_full)
        assert evaluator.sigma.search.soft_violation(not_of_plus, context) is None

    def test_not_recurses_into_compound_subformulas(self):
        """NOT isn't limited to a single leaf: it recurses into whatever
        `SigmaSearch.soft_violation` already handles, falling back to a step-function
        score (satisfied/violated, no gradient) for anything past a
        numeric comparison - a `*`-conjunction, or even a double negation.
        Still exact where it matters: 0 iff the real formula is false."""

        parser = QFOIL()
        evaluator = parser.evaluator

        # (0,0,0) is FULL but not L0 - FULL(z)*L0(z) is violated (>0), so
        # NOT of it is satisfied (0).
        conj = parser.parse("NOT(FULL(z) * L0(z))")
        assert evaluator.sigma.search.soft_violation(conj, {"z": (0, 0, 0)}) == 0.0
        # BOTTOM is L0 (and hence not FULL) - FULL(z)*L0(z) is still
        # violated (FULL fails), NOT of it is still satisfied.
        assert (
            evaluator.sigma.search.soft_violation(conj, {"z": (None, None, None)}) == 0.0
        )

        # Double negation of FULL(z) is exactly FULL(z) again.
        double_not = parser.parse("NOT(NOT(FULL(z)))")
        assert evaluator.sigma.search.soft_violation(double_not, {"z": (0, 0, 0)}) == 0.0
        assert (
            evaluator.sigma.search.soft_violation(double_not, {"z": (0, 0, None)}) > 0.0
        )

    @pytest.mark.parametrize(
        "expr,context,expected_violated",
        [
            ("NOT(FULL(z))", {"z": (1, 0, None)}, False),  # z has a None
            ("NOT(FULL(z))", {"z": (1, 0, 1)}, True),  # z is full
            ("NOT(L0(z))", {"z": (1, None, None)}, False),
            ("NOT(L0(z))", {"z": (None, None, None)}, True),
            ("NOT(SIZE(z) >= 2)", {"z": (1, 0, None)}, True),  # SIZE=2
            ("NOT(SIZE(z) >= 2)", {"z": (1, None, None)}, False),  # SIZE=1
            ("NOT(SIZE(z) <= 1)", {"z": (1, 0, None)}, False),  # SIZE=2
            ("NOT(SIZE(z) <= 1)", {"z": (1, None, None)}, True),  # SIZE=1
            ("NOT(SIZE(z) == 1)", {"z": (1, None, None)}, True),  # SIZE=1
            ("NOT(SIZE(z) == 1)", {"z": (1, 0, None)}, False),  # SIZE=2
        ],
    )
    def test_soft_violation_negates_a_single_leaf(
        self, expr, context, expected_violated
    ):
        """NOT of a single leaf (a relation or a numeric comparison) IS
        supported - real universal-statement bodies are shaped exactly
        like `(skeleton) * NOT(oracle)`, not `NOT(skeleton * oracle)` (see
        `examples/intro_explainability.ipynb`'s `all_comp_pred`), so this
        is the case that actually matters for `SigmaEvaluator.decide`'s
        universal fast path."""

        parser = QFOIL()
        violation = parser.evaluator.sigma.search.soft_violation(parser.parse(expr), context)
        assert violation is not None
        assert (violation > 0) == expected_violated

    @pytest.mark.parametrize(
        "op,left,right,expected_violation",
        [
            (">=", 3, 2, 0.0),
            (">=", 2, 3, 1.0),
            ("<=", 2, 3, 0.0),
            ("<=", 3, 2, 1.0),
            ("==", 3, 3, 0.0),
            ("==", 3, 5, 2.0),
        ],
    )
    def test_soft_violation_numeric_comparison(
        self, op, left, right, expected_violation
    ):
        """`SigmaSearch.soft_violation` scores `>=`/`<=`/`==` as a continuous gap
        between the two (evaluated) sides instead of bailing out to None.
        This is what lets PSR/PCE-shaped conjuncts like `EP(z, k) >=
        delta` or `SIZE(z) <= size_max` drive the local search - the score
        only depends on the two numbers `evaluate()` produces, never on
        how they were computed, so it works the same whether the operands
        are plain arithmetic or a real PyJuiceModel query.
        """
        from qfoil.language import BinaryOp, NumberLiteral

        evaluator = QFOIL().evaluator
        ast = BinaryOp(NumberLiteral(left), op, NumberLiteral(right))
        assert evaluator.sigma.search.soft_violation(ast, {}) == expected_violation

    def test_finds_witness_with_numeric_comparison_in_conjunction(self):
        """Before this fix, ANY numeric comparison inside the SIGMA
        formula made `SigmaSearch.soft_violation` return None for the *entire*
        `*`-conjunction (one None operand poisons the whole product),
        disabling heuristic search outright and forcing the exact scan -
        intractable for the adversarial n=30 witness used above. Same
        setup, with a `SIZE(z) <= n` factor appended (trivially true once
        FULL(z) holds, but previously enough to kill the heuristic for
        the whole formula): must still resolve via local search.
        """
        n = 30
        k = (1,) + (0,) * (n - 1)
        expr = f"SIGMA z. (FULL(z) * (k <<= z) * (SIZE(z) <= {n})) >= 1"
        assert parse_and_evaluate(expr, {"k": k}, mode="heuristic") == 1

    def test_seed_used_as_starting_point(self):
        """A `seed` matching the domain length is used as the starting
        candidate instead of BOTTOM - lets a caller warm-start from an
        already-known-decent point (see SIGMA_EVALUATION_PROPOSAL.md,
        'Fase 1'). Here the seed already satisfies the formula, so it
        must come back unchanged as the very first check.
        """
        from qfoil.language import Sigma, FunctionCall, Variable

        evaluator = QFOIL().evaluator
        sigma_ast = Sigma("z", FunctionCall("FULL", [Variable("z")]))
        seed = (1, 0, 1)
        witness = evaluator.sigma.search.local_search(
            sigma_ast, {}, (None, None, None), budget=64, seed=seed
        )
        assert witness == seed

    def test_seed_length_mismatch_uses_random_start(self):
        """A `seed` that doesn't match the domain length is ignored
        (uses a random start) rather than corrupting the search. L0 still
        guides removal of every defined value."""
        from qfoil.language import Sigma, FunctionCall, Variable

        evaluator = QFOIL().evaluator
        sigma_ast = Sigma("z", FunctionCall("L0", [Variable("z")]))
        witness = evaluator.sigma.search.local_search(
            sigma_ast, {}, (None, None, None), budget=64, seed=(1, 1)
        )
        assert witness == (None, None, None)


class TestSigmaDecisionUniversalFastPath:
    """`SIGMA z. formula(z) <= 0` / `== 0` restates `forall z.
    NOT(formula(z))`. A witness for the raw `formula` is a counterexample
    that DISPROVES the universal cheaply - this is step 3 of the
    "Orquestación de universales" plan (SIGMA_EVALUATION_PROPOSAL.md):
    a Z3-free, evaluator-only refutation attempt via the same Fase 1
    search already used for plain existentials. It can only ever disprove;
    proving one true still needs the exact scan (or, out of process,
    `Z3CegisSolver.check_universal`).
    """

    def test_disproves_a_false_universal_via_heuristic_search(self):
        """`(z <<= x) * NOT(SIZE(z) <= 1) == 0` claims every subset of a
        30-bit x has size <= 1 - false (e.g. x itself). The witness sits
        far from canonical `itertools.product` order, so this only
        resolves fast if the universal fast path actually engages."""

        n = 30
        x = (1,) * n
        expr = "SIGMA z. ((z <<= x) * NOT(SIZE(z) <= 1)) == 0"
        parser = QFOIL(instance_len=n)
        assert parser.evaluator.evaluate(parser.parse(expr), {"x": x}) == 0

    def test_true_universal_is_inconclusive_in_heuristic_mode(self):
        """No heuristic-found counterexample exists here (every z <<= x
        genuinely has SIZE <= 3) - heuristic mode has no certification
        strategy left once the search for a counterexample fails, so it
        must report Inconclusive rather than silently falling through to
        the exact scan to rescue the answer."""

        x = (1, 1, 1)
        expr = "SIGMA z. ((z <<= x) * NOT(SIZE(z) <= 3)) == 0"
        parser = QFOIL()
        result = parser.evaluator.evaluate(parser.parse(expr), {"x": x})
        assert isinstance(result, Inconclusive)

    def test_true_universal_resolved_exactly_in_exact_mode(self):
        """Same universal, `sigma_mode="exact"` selected explicitly: the
        exact scan reports the universal as holding (1), not a guess."""

        x = (1, 1, 1)
        expr = "SIGMA z. ((z <<= x) * NOT(SIZE(z) <= 3)) == 0"
        parser = QFOIL(sigma_mode="exact")
        assert parser.evaluator.evaluate(parser.parse(expr), {"x": x}) == 1

    def test_agrees_with_exact_mode_for_the_false_case(self):
        """Same disproof as the n=30 heuristic case above, resolved via
        `sigma_mode="exact"` instead: the exact scan reaches the same
        answer - the procedure only changes speed, never the result."""

        x = (1, 1, 1)
        expr = "SIGMA z. ((z <<= x) * NOT(SIZE(z) <= 1)) == 0"
        parser = QFOIL(sigma_mode="exact")
        assert parser.evaluator.evaluate(parser.parse(expr), {"x": x}) == 0

    def test_le_zero_shape_also_engages(self):
        """`<= 0` is the same universal shape as `== 0` (count is always
        >= 0) - the fast path must fire for it too."""

        n = 30
        x = (1,) * n
        expr = "SIGMA z. ((z <<= x) * NOT(SIZE(z) <= 1)) <= 0"
        parser = QFOIL(instance_len=n)
        assert parser.evaluator.evaluate(parser.parse(expr), {"x": x}) == 0


class TestComparisonViolationAvoidsBareSigmaEvaluation:
    """Regression coverage for a real bug found validating step 3 against
    `examples/intro_explainability.ipynb`'s `DFS`/`threshold_determined_pred`
    shape: `NOT(SIGMA w. formula(w) == 0)` (and the un-negated comparison
    case too) used to evaluate the nested Sigma's operand directly
    (`self.evaluate(cmp_ast.left, ...)`), which always pays the full
    O(3^n) brute-force count - none of `SigmaEvaluator.decide`'s early-exit/
    Fase-1 machinery applies to a bare Sigma value, only to one directly
    wrapped in a comparison. This silently turned every heuristic
    evaluation of an outer universal into an exponential-time one instead
    of the O(1)-ish check it looks like. `SigmaSearch.comparison_violation` fixes
    this by evaluating the *whole* inner comparison (still fast) instead
    of splitting it apart whenever either side contains a Sigma.
    """

    def test_contains_sigma_detects_nested_quantifiers(self):

        parser = QFOIL()
        ev = parser.evaluator

        assert ev.sigma.search.contains_sigma(parser.parse("SIGMA w. FULL(w)")) is True
        assert (
            ev.sigma.search.contains_sigma(parser.parse("NOT(SIGMA w. FULL(w) >= 1)")) is True
        )
        assert (
            ev.sigma.search.contains_sigma(parser.parse("(SIGMA w. FULL(w)) == 0")) is True
        )
        assert ev.sigma.search.contains_sigma(parser.parse("SIZE(z) >= 2")) is False
        assert ev.sigma.search.contains_sigma(parser.parse("FULL(z) * L0(z)")) is False

    def test_not_of_nested_sigma_comparison_stays_fast(self):
        """`NOT(SIGMA w. (w <<= z) * NOT(SIZE(w) >= k) == 0)` (the
        `threshold_determined_pred` shape from `all_comp_pred`/`DFS` in
        `examples/intro_explainability.ipynb`) must resolve in roughly
        constant time as the domain grows, not exponentially."""
        import time


        def td_like(var):
            a = f"SIGMA w. ((w <<= {var}) * NOT(SIZE(w) >= 2)) == 0"
            b = f"SIGMA w. ((w <<= {var}) * NOT(SIZE(w) >= 5)) == 0"
            return f"NOT(NOT({a}) * NOT({b}))"

        elapsed_by_n = {}
        for n in (10, 30):
            parser = QFOIL(instance_len=n)
            x = (1,) * n
            expr = f'SIGMA z. ((z <<= x) * NOT({td_like("z")})) == 0'
            t0 = time.time()
            result = parser.evaluator.evaluate(parser.parse(expr), {"x": x})
            elapsed_by_n[n] = time.time() - t0
            assert result == 0  # BOTTOM is a genuine counterexample here

        # Before the fix this went from ~0.3s (n=10) to indefinite hangs
        # by n=30 (exponential). Generous bound: still well under a
        # second even at n=30 if the fast path is actually engaging.
        assert elapsed_by_n[30] < 2.0

        # `SigmaSearch.comparison_violation`'s fix is shared by the un-negated branch
        # too (`SigmaSearch.soft_violation`'s plain `==`/`<=`/`>=` case, `negate=False`
        # - same helper, same `SigmaSearch.contains_sigma` guard), not just the NOT
        # one exercised above - no separate large-n test for that side:
        # constructing one that isn't accidentally a "prove this SIGMA is
        # truly 0 with no witness at all" case (which genuinely has no
        # fast path - see test_true_universal_still_resolved_exactly - and
        # is orthogonal to this bug) is fiddly, and the code path is
        # identical modulo the `negate` flag.


class TestApproximateModeNeverFallsBackToExactScan:
    """Once `SigmaEvaluator.decide` is in approximate mode (nested SIGMA
    detected - statically, on the formula it's deciding, or dynamically,
    because an ancestor search is already in that mode), it must NEVER
    fall back to the exact scan when Fase 1 fails to find a witness/
    counterexample - that scan is exactly the O(3^n) cost approximate
    mode exists to avoid paying (`_approximate_mode_for_nested_sigma`
    already warned the caller this is unverified). Instead it trusts
    Fase 1's negative verdict as final: existential -> 0, universal -> 1.

    Regression coverage for a real bug found validating this against
    MinimumPCE at MNIST scale (196 features, see
    SIGMA_EVALUATION_PROPOSAL.md's "Búsqueda aproximada con SIGMA
    anidado"): capping the inner search budget made Fase 1 fail almost
    every time (too little budget to reach a witness from BOTTOM in a
    196-dimensional space), and every failure fell through to a 3^196
    exact scan that never finished - the opposite of what the budget cap
    was supposed to achieve.
    """

    def test_inner_search_without_its_own_nesting_still_stays_bounded(self):
        """The inner `z` check here (`SCE(z) * NOT(SIZE(z) <= SIZE(y))`)
        has no nested SIGMA of its own - only the OUTER `y` formula does.
        Must still avoid the exact scan via the *dynamic*
        `SigmaEvaluator.approximate_depth` signal (set while the outer search
        runs), not just the static `SigmaSearch.contains_sigma` check on its own
        formula - `SigmaSearch.contains_sigma` alone would miss this exact shape,
        which is precisely the MinimumPCE bug above."""
        import time

        n = 40
        x = (1,) * n
        # From BOTTOM, reaching SIZE >= 20 needs 20 correct commits -
        # inner_budget's tiny default (8) cannot get there, so the inner
        # minimality check is expected to fail every time.
        size_min = 20

        def sce(var):
            return f"({var} <<= x) * (SIZE({var}) >= {size_min})"

        formula = (
            f"({sce('y')}) * "
            f"(SIGMA z. (({sce('z')}) * NOT(SIZE(z) <= SIZE(y))) == 0)"
        )

        parser = QFOIL()
        t0 = time.time()
        witness = parser.grow_witness(
            formula, var="y", context={"x": x}, budget=200
        )
        elapsed = time.time() - t0

        # The point isn't whether a witness is found (approximate mode
        # may or may not, since the minimality check itself is
        # unverified) - it's that this returns promptly instead of
        # hanging on a 3^40 exact scan for the nested `z` check.
        assert elapsed < 5.0
        if witness is not None:
            # SCE(y) itself has no nested SIGMA - it's exactly verified,
            # regardless of how the (unverified) minimality claim landed.
            assert parser.evaluator.evaluate(
                parser.parse(sce("y")), {"x": x, "y": witness}
            )


class TestLastWitnessAndSeed:
    """`evaluate()`'s side channel for a SIGMA query's own witness
    (`last_witness`) and its optional search starting point (`seed`) -
    both exposed without changing `evaluate()`'s contract (still always
    returns a number). See SIGMA_EVALUATION_PROPOSAL.md's "`last_witness`
    y `seed`" section for the empirical case that motivated these."""

    def test_last_witness_set_for_fase1_existential(self):
        """A witness `SigmaSearch.local_search` finds via the heuristic fast
        path (not the exact scan) must end up in `last_witness`, verified."""

        n = 30
        k_tuple = (1,) + (0,) * (n - 1)  # far from canonical order
        parser = QFOIL(instance_len=n)
        expr = "SIGMA z. (FULL(z) * (k <<= z)) >= 1"
        result = parser.evaluator.evaluate(parser.parse(expr), {"k": k_tuple})

        assert result == 1
        witness = parser.evaluator.sigma.last_witness
        assert witness is not None
        assert all(v is not None for v in witness)  # FULL(witness)
        assert all(w == k for w, k in zip(witness, k_tuple) if k is not None)

    def test_last_witness_set_for_exact_scan_candidate(self):
        """With `sigma_mode="exact"` selected explicitly, the exact scan
        itself must still expose the exact candidate that decided the
        comparison."""

        parser = QFOIL(sigma_mode="exact")
        expr = "SIGMA z. FULL(z) >= 1"
        result = parser.evaluator.evaluate(parser.parse(expr), {"x": (0, 0, 0)})

        assert result == 1
        witness = parser.evaluator.sigma.last_witness
        assert witness is not None
        assert all(v is not None for v in witness)

    def test_last_witness_none_when_universal_holds(self):
        """A universal that genuinely holds (no counterexample anywhere)
        has nothing witness-shaped to report. Uses `sigma_mode="exact"`
        so the exact scan actually certifies the universal as 1 (in
        heuristic mode this same query is Inconclusive -- see
        test_true_universal_is_inconclusive_in_heuristic_mode)."""

        parser = QFOIL(sigma_mode="exact")
        expr = "SIGMA z. ((z <<= x) * NOT(SIZE(z) <= 3)) == 0"
        result = parser.evaluator.evaluate(parser.parse(expr), {"x": (1, 1, 1)})

        assert result == 1
        assert parser.evaluator.sigma.last_witness is None

    def test_last_witness_none_in_approximate_mode_guess(self):
        """Approximate mode's "assume satisfied/holds" verdict is a guess,
        never promoted to a witness."""

        n = 40
        x = (1,) * n
        size_min = 20  # unreachable from BOTTOM within the tiny inner_budget

        def sce(var):
            return f"({var} <<= x) * (SIZE({var}) >= {size_min})"

        formula = (
            f"({sce('y')}) * "
            f"(SIGMA z. (({sce('z')}) * NOT(SIZE(z) <= SIZE(y))) == 0)"
        )
        parser = QFOIL()
        outer_ast = parser.parse(f"SIGMA y. ({formula}) >= 1")
        # `outer_ast.left` (the Sigma node) itself nests a SIGMA -
        # `SigmaEvaluator.decide` enters approximate mode for this outer `y`
        # search too (static `SigmaSearch.contains_sigma` check), so whatever Fase 1
        # decides here, it must never fall back to the exact scan (would
        # hang at n=40) nor promote its guess to a witness.
        parser.evaluator.evaluate(outer_ast, {"x": x})
        assert parser.evaluator.sigma.last_witness is None

    def test_seed_lets_evaluate_find_a_witness_far_from_canonical_order(self):
        """With the actual witness given as `seed`, `SigmaSearch.local_search`
        accepts it on its very first check, regardless of how far it sits
        from canonical `itertools.product` order.

        Does NOT compare against a no-seed/crippled-budget run here: for
        a flat formula (no nested SIGMA) a failed Fase 1 attempt falls
        back to the exact scan (`SigmaEvaluator.decide`'s documented, correct
        behavior for small domains) - at n=30 that's genuinely 3**30
        candidates, not a quick "no" the way approximate mode's guess is.
        Exercising that path is `TestSigmaLocalSearch`'s job at a size
        that stays exact-scannable; this test only needs to show `seed`
        works through the public `evaluate()` API.
        """

        n = 30
        k_tuple = (1,) + (0,) * (n - 1)
        parser = QFOIL(instance_len=n)
        expr = "SIGMA z. (FULL(z) * (k <<= z)) >= 1"

        result = parser.evaluator.evaluate(
            parser.parse(expr), {"k": k_tuple}, seed=k_tuple
        )
        assert result == 1
        assert parser.evaluator.sigma.last_witness == k_tuple

    def test_seed_ignored_for_non_sigma_ast(self):
        """`seed` only means something for a top-level SIGMA comparison -
        passing it for anything else must be harmless."""

        parser = QFOIL()
        result = parser.evaluator.evaluate(
            parser.parse("FULL(x)"), {"x": (1, 1)}, seed=(0, 0)
        )
        assert result == 1


class TestSigmaGreedyAscent:
    """Test `SigmaSearch.greedy_ascent`: the monotone-growth alternative to
    `SigmaSearch.local_search`, used to warm-start `minimize`/`maximize`
    searches (see `Z3CegisSolver._warm_start_witness`). Not wired into
    `SigmaEvaluator.decide`/`evaluate()` - it's reached directly (as these
    tests do) or via `qfoil/backends/z3.py`, never as a `SIGMA ... >= 1`
    accelerator for the general public API.
    """

    def test_grows_bottom_to_a_full_witness(self):
        """SIGMA z. FULL(z) >= 1: any fully-defined tuple works, so
        growth just needs to fill in every position."""
        from qfoil.language import Sigma, FunctionCall, Variable

        evaluator = QFOIL().evaluator
        sigma_ast = Sigma("z", FunctionCall("FULL", [Variable("z")]))
        n = 10
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {}, (None,) * n, budget=200
        )
        assert witness is not None
        assert all(v is not None for v in witness)
        assert evaluator.function_runtime.evaluate_native("FULL", [witness]) == 1

    def test_grows_toward_a_specific_target_like_the_adversarial_case(self):
        """SIGMA z. (FULL(z) * (k <<= z)) >= 1 with k fixed and fully
        defined forces z == k exactly - same adversarial shape used for
        `SigmaSearch.local_search`, checking growth converges to the unique
        witness rather than some arbitrary full tuple."""
        from qfoil.language import Sigma, BinaryOp, FunctionCall, Variable

        n = 30
        k = (1,) + (0,) * (n - 1)
        evaluator = QFOIL().evaluator
        formula = BinaryOp(
            FunctionCall("FULL", [Variable("z")]),
            "*",
            BinaryOp(Variable("k"), "<<=", Variable("z")),
        )
        sigma_ast = Sigma("z", formula)
        # Each round scores the whole remaining pool (up to n positions x 2
        # values) before committing just one, so full convergence over n
        # rounds costs O(n^2) evaluations in the worst case - unlike
        # SigmaSearch.local_search's UCB1, which can find this same witness in
        # O(n) evaluations (see the analogous test above). That trade-off
        # (worse for blind existence search, better for warm-starting an
        # already-decent seed) is exactly why the two coexist.
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {"k": k}, (None,) * n, budget=2000
        )
        assert witness == k

    def test_seed_used_as_starting_point(self):
        """A seed that already satisfies the formula comes back unchanged,
        with zero growth needed."""
        from qfoil.language import Sigma, FunctionCall, Variable

        evaluator = QFOIL().evaluator
        sigma_ast = Sigma("z", FunctionCall("FULL", [Variable("z")]))
        seed = (1, 0, 1)
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {}, (None, None, None), budget=200, seed=seed
        )
        assert witness == seed

    def test_seed_length_mismatch_uses_random_start(self):
        from qfoil.language import Sigma, FunctionCall, Variable

        evaluator = QFOIL().evaluator
        sigma_ast = Sigma("z", FunctionCall("FULL", [Variable("z")]))
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {}, (None, None, None), budget=200, seed=(1, 1)
        )
        assert witness is not None and all(v is not None for v in witness)

    def test_unsupported_fragment_returns_none(self):
        """`+` is outside `SigmaSearch.soft_violation`'s fragment, and NOT wrapping it
        inherits that (NOT of a compound `*`, or even a nested NOT, IS
        supported via recursion - see test_grows_a_witness_for_a_negated_leaf
        below and TestSigmaLocalSearch.test_not_recurses_into_compound_subformulas)
        - must bail out cleanly, same contract as `SigmaSearch.local_search`."""
        from qfoil.language import (
            Sigma,
            UnaryOp,
            BinaryOp,
            FunctionCall,
            Variable,
        )

        evaluator = QFOIL().evaluator
        sigma_ast = Sigma(
            "z",
            UnaryOp(
                "!",
                BinaryOp(
                    FunctionCall("FULL", [Variable("z")]),
                    "+",
                    FunctionCall("L0", [Variable("z")]),
                ),
            ),
        )
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {}, (None, None, None), budget=200
        )
        assert witness is None

    def test_grows_a_witness_for_a_negated_leaf(self):
        """NOT(FULL(z)) is satisfied by anything with at least one None -
        greedy ascent must find one, matching `SigmaEvaluator.decide`'s
        universal-statement fast path (see `SigmaSearch.soft_violation`'s NOT case)."""
        from qfoil.language import Sigma, UnaryOp, FunctionCall, Variable

        evaluator = QFOIL().evaluator
        sigma_ast = Sigma("z", UnaryOp("!", FunctionCall("FULL", [Variable("z")])))
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {}, (None, None, None), budget=200
        )
        assert witness is not None
        assert evaluator.function_runtime.evaluate_native("FULL", [witness]) == 0

    def test_numeric_comparison_drives_growth(self):
        """SIGMA z. (z <<= x) * (SIZE(z) >= k) >= 1: the same numeric-
        comparison support `SigmaSearch.soft_violation` gained for `SigmaSearch.local_search`
        (SIZE/EP/CEP thresholds) drives greedy ascent too, since both
        share the same scorer."""
        from qfoil.language import (
            Sigma,
            BinaryOp,
            FunctionCall,
            Variable,
            NumberLiteral,
        )

        evaluator = QFOIL().evaluator
        x = (1, 1, 0, 1, 1, 0, 1, 1)
        formula = BinaryOp(
            BinaryOp(Variable("z"), "<<=", Variable("x")),
            "*",
            BinaryOp(FunctionCall("SIZE", [Variable("z")]), ">=", NumberLiteral(5)),
        )
        sigma_ast = Sigma("z", formula)
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {"x": x}, (None,) * len(x), budget=200
        )
        assert witness is not None
        assert evaluator.function_runtime.evaluate_native("SIZE", [witness]) >= 5
        assert all(
            witness[i] is None or witness[i] == x[i] for i in range(len(x))
        )

    def test_stalls_and_returns_none_when_infeasible(self):
        """SIGMA z. (FULL(z) * (SIZE(z) <= 0)) >= 1: no witness can ever
        satisfy both (FULL needs every position set, the size cap forbids
        setting any) - every single-position commit trades one violation
        for another with zero net gain, so no pool move ever improves and
        the search must stop, not loop or wrongly claim success."""
        from qfoil.language import (
            Sigma,
            BinaryOp,
            FunctionCall,
            Variable,
            NumberLiteral,
        )

        evaluator = QFOIL().evaluator
        formula = BinaryOp(
            FunctionCall("FULL", [Variable("z")]),
            "*",
            BinaryOp(FunctionCall("SIZE", [Variable("z")]), "<=", NumberLiteral(0)),
        )
        sigma_ast = Sigma("z", formula)
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {}, (None, None, None, None), budget=200
        )
        assert witness is None

    def test_agrees_with_local_search_on_who_it_trusts(self):
        """Whatever it proposes, once verified, must actually satisfy the
        exact evaluator - the caller-verification contract is identical
        to `SigmaSearch.local_search`'s."""
        from qfoil.language import Sigma, FunctionCall, Variable

        evaluator = QFOIL().evaluator
        sigma_ast = Sigma("z", FunctionCall("L1", [Variable("z")]))
        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {}, (None, None, None, None, None), budget=200
        )
        assert witness is not None
        assert evaluator.function_runtime.evaluate_native("L1", [witness]) == 1


class TestGrowWitness:
    """`grow_witness` is the public entry point for `SigmaSearch.greedy_ascent`:
    takes a plain formula string (same language as `evaluate()`), not a
    hand-built `Sigma` AST node and an explicit reference tuple - so the
    call site reads as the logical formula being solved, not as
    evaluator internals, and callers never need to reach for a name
    starting with `_`.
    """

    def test_grows_toward_a_specific_target(self):
        """Same shape as the adversarial `SigmaSearch.greedy_ascent` test, but
        entirely through the public expression-string API."""

        n = 30
        k = (1,) + (0,) * (n - 1)
        parser = QFOIL(instance_len=n)
        witness = parser.grow_witness(
            "FULL(z) * (k <<= z)", var="z", context={"k": k}, budget=2000
        )
        assert witness == k

    def test_numeric_comparison_drives_growth(self):
        """The same `EP`/`CEP`/`SIZE`-threshold support `SigmaSearch.greedy_ascent`
        has carries through the public wrapper."""

        x = (1, 1, 0, 1, 1, 0, 1, 1)
        parser = QFOIL(instance_len=len(x))
        witness = parser.grow_witness(
            "(z <<= x) * (SIZE(z) >= 5)", var="z", context={"x": x}, budget=200
        )
        assert witness is not None
        assert parser.evaluator.function_runtime.evaluate_native("SIZE", [witness]) >= 5
        assert all(
            witness[i] is None or witness[i] == x[i] for i in range(len(x))
        )

    def test_unsupported_fragment_returns_none(self):
        """`+` is outside the supported fragment, and NOT wrapping it
        inherits that (NOT of a compound `*` IS supported - see
        `TestSigmaGreedyAscent.test_grows_a_witness_for_a_negated_leaf`)."""

        parser = QFOIL()
        witness = parser.grow_witness(
            "NOT(FULL(z) + L0(z))", var="z", context={"x": (0, 0, 0)}, budget=200
        )
        assert witness is None

    def test_seed_used_as_starting_point(self):

        parser = QFOIL()
        seed = (1, 0, 1)
        witness = parser.grow_witness(
            "FULL(z)", var="z", context={"x": seed}, budget=200, seed=seed
        )
        assert witness == seed

    def test_uses_configured_domain_length(self):
        """The public wrapper uses the runtime's configured domain length."""

        x = (1, 0, 1, 1, 0)
        parser = QFOIL(instance_len=len(x))
        witness = parser.grow_witness(
            "z <<= x", var="z", context={"x": x}, budget=200
        )
        assert witness is not None
        assert len(witness) == len(x)

    def test_requires_parser_to_be_set(self):
        """Mirrors the existing `Parser not set` guard used by logic-
        defined functions - clear error instead of an AttributeError deep
        inside `SigmaSearch.greedy_ascent`."""
        from qfoil import QFOILEvaluator

        evaluator = QFOILEvaluator()  # no parser passed
        with pytest.raises(RuntimeError, match="Parser not set"):
            evaluator.sigma.grow_witness("FULL(z)", var="z", context={"x": (1, 0)})
