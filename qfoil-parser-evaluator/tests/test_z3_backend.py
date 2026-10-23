"""Tests for the optional QFOIL Z3-backed CEGIS solver.

This module is entirely optional: QFOIL's core must
work whether or not `z3-solver` is installed, and `Z3CegisSolver` must raise a
clear error (not import-crash the whole package) when it isn't. The rest
of this file needs `z3-solver` and is skipped otherwise.
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent

from qfoil import QFOIL

MODEL_DIR = PROJECT_ROOT / "examples" / "models" / "spam_binary_small"


# Everything below needs a real z3-solver install.
z3 = pytest.importorskip("z3")

from qfoil.backends.z3 import Z3CegisSolver


@pytest.fixture()
def solver(request):
    parser = QFOIL(instance_len=getattr(request, "param", 4))
    return Z3CegisSolver(parser.evaluator, parser), parser


class TestSkeletonEncodingMatchesNativeFunctions:
    """Differential test: encoding a fully-concrete skeleton in Z3 and
    checking its (constant) truth value must agree with the corresponding
    native function - same fragment, same semantics, two interpreters."""

    def _check(self, solver_bundle, expr, context, expected):
        solver, parser = solver_bundle
        ast = parser.parse(expr)
        # No free "var" here - every operand comes from context, so the
        # symbolic (value, mask) pair is never actually referenced.
        n = len(next(value for value in context.values() if isinstance(value, tuple)))
        dummy_sym = (z3.BitVec("dummy_v", n), z3.BitVec("dummy_m", n))
        encoded = solver._encode_bool(ast, "__unused__", dummy_sym, context)
        simplified = z3.simplify(encoded)
        assert z3.is_true(simplified) == bool(expected), (expr, context, simplified)

    @pytest.mark.parametrize(
        "expr,context,expected",
        [
            ("FULL(x)", {"x": (1, 0, 1)}, 1),
            ("FULL(x)", {"x": (1, 0, None)}, 0),
            ("L0(x)", {"x": (None, None, None)}, 1),
            ("L0(x)", {"x": (1, None, None)}, 0),
            ("L1(x)", {"x": (1, None, None)}, 1),
            ("L1(x)", {"x": (1, 0, None)}, 0),
            ("FEAT(x)", {"x": (None, 1, None)}, 1),
            (
                "FREE(x,y) == 2",
                {"x": (1, 0, None), "y": (None, 1, 1)},
                1,
            ),
            (
                "FREE(x,y) == 1",
                {"x": (1, 0, None), "y": (1, None, None)},
                1,
            ),
            ("x <<= y", {"x": (1, None), "y": (1, 0)}, 1),
            ("x <<= y", {"x": (1, 0), "y": (1, 1)}, 0),
            ("x << y", {"x": (1, 1), "y": (1, 1)}, 0),  # strict: equal excluded
            ("x << y", {"x": (1, None), "y": (1, 0)}, 1),
            ("PARTITION(x,y)", {"x": (1, None), "y": (None, 0)}, 1),
            ("PARTITION(x,y)", {"x": (1, 0), "y": (1, None)}, 0),
            ("SUF(x,y)", {"x": (1, None, 1), "y": (0, None, 1)}, 1),
            ("SUF(x,y)", {"x": (1, None, 1), "y": (0, 1, 1)}, 0),
            ("COMP(x,y)", {"x": (1, None), "y": (1, 0)}, 1),
            ("COMP(x,y)", {"x": (1, None), "y": (1, None)}, 0),  # y not full
            ("NOT(FULL(x))", {"x": (1, 0, None)}, 1),
            (
                "MEET(x,y,z)",
                {"x": (1, 0, None), "y": (1, None, 1), "z": (1, None, None)},
                1,
            ),
            (
                "MEET(x,y,z)",
                {"x": (1, 0, None), "y": (1, None, 1), "z": (0, None, None)},
                0,
            ),
            ("JOIN(x,y,z)", {"x": (1, 0, None), "y": (1, None, 1), "z": (1, 0, 1)}, 1),
            ("OPPOSITE(x,y)", {"x": (1, None, None), "y": (0, None, None)}, 1),
            ("OPPOSITE(x,y)", {"x": (1, 0, None), "y": (0, None, None)}, 0),
        ],
    )
    def test_encoding_matches_native(self, solver, expr, context, expected):
        self._check(solver, expr, context, expected)


class TestFindArgopt:
    """End-to-end CEGIS loop, with a pure-logic oracle (no PyJuice model
    needed) to isolate the search mechanics from probabilistic inference."""

    def test_finds_minimum_size_subset_with_exact_property(self, solver):
        solver_obj, _ = solver
        x = (1, 1, 0, 1)

        result = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="SIZE(y) == 2",
            context={"x": x},
            minimize="SIZE(y)",
        )
        assert result is not None
        assert sum(1 for e in result if e is not None) == 2
        assert all(result[i] == x[i] for i in range(len(x)) if result[i] is not None)
        # minimize was given and the main loop's own blocking history
        # proves nothing smaller also satisfies the oracle - genuinely
        # optimal, not just "the first one that worked".
        assert solver_obj.last_status == "proved_optimal"

    def test_free_can_constrain_a_symbolic_candidate(self, solver):
        solver_obj, parser = solver
        x = (1, 0, 1, 0)

        result = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x) * (FREE(x, y) >= 2)",
            oracle_expr="SIZE(y) >= 1",
            context={"x": x},
            maximize="SIZE(y)",
        )

        assert result is not None
        assert solver_obj.evaluator.evaluate(
            parser.parse("FREE(x, y)"), {"x": x, "y": result}
        ) == 2
        assert solver_obj.evaluator.evaluate(
            parser.parse("SIZE(y)"), {"y": result}
        ) == 2
        assert solver_obj.last_status == "proved_optimal"

    def test_found_without_objective_is_not_claimed_optimal(self, solver):
        """No `minimize`/`maximize` given - "optimal" isn't even a
        meaningful claim here, so the status must say plain `"found"`."""
        solver_obj, _ = solver
        x = (1, 1, 0, 1)

        result = solver_obj.find_argopt(
            var="y", skeleton_expr="(y <<= x)", oracle_expr="FULL(y)",
            context={"x": x},
        )
        assert result == x
        assert solver_obj.last_status == "found"

    def test_blocking_actually_excludes_tried_candidates(self, solver, capsys):
        """Regression test for a real bug: blocking on raw (value, mask)
        instead of (value & mask, mask) let Z3 "re-propose" the same
        semantic tuple by flipping a don't-care bit outside the mask,
        looping forever on the same candidate.

        `evaluate()` is recursive, so spying on it directly overcounts
        (each candidate check makes several nested calls); instead, parse
        `verbose=True`'s one-line-per-iteration output, which reports
        exactly one candidate per CEGIS iteration.
        """
        solver_obj, _ = solver
        x = (1, 1, 0, 1)

        result = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="SIZE(y) == 2",
            context={"x": x},
            minimize="SIZE(y)",
            verbose=True,
        )
        candidates_per_iteration = [
            line.split("candidate ")[1].split(" ->")[0]
            for line in capsys.readouterr().out.splitlines()
            if "candidate" in line
        ]

        assert result is not None
        assert len(candidates_per_iteration) > 1  # actually exercises blocking
        assert len(candidates_per_iteration) == len(set(candidates_per_iteration)), (
            candidates_per_iteration
        )

    def test_unsat_skeleton_returns_none(self, solver):
        solver_obj, _ = solver
        x = (1, 0)

        # No y <<= x has size >= 5 when len(x) == 2 - skeleton is UNSAT.
        result = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x) * (SIZE(y) >= 5)",
            oracle_expr="FULL(y)",
            context={"x": x},
        )
        assert result is None
        # A real proof of absence, not a timeout - this is exactly the
        # distinction a caller checking a universal statement needs.
        assert solver_obj.last_status == "unsat"

    def test_oracle_never_satisfied_exhausts_skeleton(self, solver):
        solver_obj, _ = solver
        x = (1, 1)

        # Skeleton has candidates (e.g. y=x), but the oracle is never true.
        result = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="FULL(y) * NOT(FULL(y))",  # never true
            context={"x": x},
            max_iters=20,
        )
        assert result is None
        # x has only 2 bits, so the skeleton has just 9 candidates -
        # CEGIS blocks all of them well within max_iters=20 and Z3 itself
        # goes UNSAT. That's a real proof here, precisely because the
        # domain is tiny; see test_exhausts_iterations_without_a_proof
        # for the "ran out of time, not a proof" case on a bigger domain.
        assert solver_obj.last_status == "unsat"

    def test_exhausts_iterations_without_a_proof(self, solver):
        """Same never-satisfiable oracle, but a domain too big to
        exhaust within a small `max_iters` - must report `"exhausted"`,
        not `"unsat"`, since Z3 never actually got to prove anything."""
        solver_obj, _ = solver
        x = (1,) * 10  # 3**10 candidates - nowhere near exhausted in 2 tries

        result = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="FULL(y) * NOT(FULL(y))",  # never true
            context={"x": x},
            max_iters=2,
        )
        assert result is None
        assert solver_obj.last_status == "exhausted"


class TestWarmStart:
    """`find_argopt(..., warm_start=True)`: Fase 1 informs Z3 instead of
    letting it search blind from BOTTOM. Off by default - every test in
    `TestFindArgopt` above is unaffected by any of this.
    """

    def test_matches_plain_search_optimal_value(self, solver):
        """Warm-started and plain search must reach the same *optimal
        value* - not necessarily the same tuple, since several minimum-
        size subsets can tie (SIZE(y) == 2 here has more than one)."""
        solver_obj, parser = solver
        x = (1, 1, 0, 1)
        kwargs = dict(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="SIZE(y) == 2",
            context={"x": x},
            minimize="SIZE(y)",
        )

        plain = solver_obj.find_argopt(**kwargs)
        warm = solver_obj.find_argopt(**kwargs, warm_start=True)

        assert plain is not None and warm is not None
        size_of = lambda w: solver_obj.evaluator.evaluate(  # noqa: E731
            parser.parse("SIZE(y)"), {"y": w}
        )
        assert size_of(plain) == size_of(warm) == 2
        for w in (plain, warm):
            assert all(w[i] == x[i] for i in range(len(x)) if w[i] is not None)

    def test_proves_optimality_via_unsat_not_blocking(self, solver, capsys):
        """The recursive 'is anything strictly better?' sub-search should
        reach UNSAT - a real proof, not iteration exhaustion - and then
        return the seed. Regression target: if this ever silently
        degrades to exhausting max_iters instead, we'd rather know."""
        solver_obj, _ = solver
        x = (1, 1, 0, 1)

        warm = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="SIZE(y) == 2",
            context={"x": x},
            minimize="SIZE(y)",
            warm_start=True,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert warm is not None
        assert "[warm start]" in out
        assert "UNSAT" in out
        assert "nothing improves on SIZE(y)=2" in out
        assert solver_obj.last_status == "proved_optimal"

    def test_warm_start_exhausted_sub_search_is_not_claimed_optimal(self, solver):
        """When the 'is anything strictly better?' sub-search runs out of
        `max_iters` instead of reaching UNSAT, the seed it falls back to
        must be reported as `"found"`, not `"proved_optimal"` - it might
        not be optimal, we just didn't find better in time."""
        x = (1,) * 8  # large enough that 3 iterations can't exhaust it
        parser = QFOIL(instance_len=len(x))
        solver_obj = Z3CegisSolver(parser.evaluator, parser)

        warm = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="SIZE(y) == 3",
            context={"x": x},
            minimize="SIZE(y)",
            warm_start=True,
            max_iters=3,
        )
        assert warm is not None
        assert solver_obj.last_status == "found"

    def test_no_objective_returns_witness_directly_and_skips_z3(self, solver, capsys):
        """With neither minimize nor maximize, a successful warm start is
        just returned - the Z3 CEGIS loop should never even run."""
        solver_obj, _ = solver
        x = (1, 1, 0, 1)

        result = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="FULL(y)",
            context={"x": x},
            warm_start=True,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert result == x  # the only y with y <<= x and FULL(y) is x itself
        assert "[warm start]" in out
        assert "candidate" not in out  # CEGIS loop never entered

    def test_prefers_greedy_ascent_over_local_search(self, solver, capsys):
        """`_warm_start_witness` tries `SigmaSearch.greedy_ascent` first - the
        verbose engine label should say so for a case squarely within its
        (grow-only) capability, falling back to `SigmaSearch.local_search`
        only when that fails (see the NOT-fragment test below, which
        can't be reached by either engine and never gets an engine
        label at all)."""
        solver_obj, _ = solver
        x = (1, 1, 0, 1)

        result = solver_obj.find_argopt(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="FULL(y)",
            context={"x": x},
            warm_start=True,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert result == x
        assert "(greedy ascent)" in out

    @pytest.mark.parametrize("solver", [2], indirect=True)
    def test_unsupported_oracle_fragment_falls_through_cleanly(self, solver):
        """`NOT` in the oracle is outside `SigmaSearch.soft_violation`'s fragment, so
        the warm-start pre-pass finds nothing - find_argopt must proceed
        exactly as with warm_start=False, reaching the same optimal size.
        """
        solver_obj, parser = solver
        x = (1, 0)
        kwargs = dict(
            var="y",
            skeleton_expr="(y <<= x)",
            oracle_expr="NOT(L0(y))",
            context={"x": x},
            minimize="SIZE(y)",
        )

        plain = solver_obj.find_argopt(**kwargs)
        warm = solver_obj.find_argopt(**kwargs, warm_start=True)

        assert plain is not None and warm is not None
        size_of = lambda w: solver_obj.evaluator.evaluate(  # noqa: E731
            parser.parse("SIZE(y)"), {"y": w}
        )
        assert size_of(plain) == size_of(warm) == 1


class TestCheckUniversal:
    """`check_universal`: the honest orchestrator for universal-shaped
    SIGMA queries (step 2 of the plan in SIGMA_EVALUATION_PROPOSAL.md's
    "Orquestación de universales" section) - never iterates the domain
    itself, only ever returns "proved"/"unknown" backed by `last_status`.
    """

    @pytest.mark.parametrize("solver", [3], indirect=True)
    def test_disproved_returns_a_verified_counterexample(self, solver):
        solver_obj, parser = solver
        x = (1, 1, 1)

        # NOT every subset of a 3-bit x has size <= 1 - e.g. x itself.
        status, witness = solver_obj.check_universal(
            var="y",
            formula_expr="SIZE(y) <= 1",
            context={"x": x},
            skeleton_expr="(y <<= x)",
        )

        assert status == "disproved"
        assert witness is not None
        # The witness must genuinely violate the formula, not just look
        # like one from the skeleton's point of view.
        size = solver_obj.evaluator.evaluate(parser.parse("SIZE(y)"), {"y": witness})
        assert size > 1

    @pytest.mark.parametrize("solver", [3], indirect=True)
    def test_proved_when_no_counterexample_exists(self, solver):
        solver_obj, _ = solver
        x = (1, 1, 1)

        # Every y <<= x has at most 3 defined positions - a genuine
        # tautology over this skeleton, provable by real UNSAT on
        # NOT(formula_expr).
        status, witness = solver_obj.check_universal(
            var="y",
            formula_expr="SIZE(y) <= 3",
            context={"x": x},
            skeleton_expr="(y <<= x)",
        )

        assert status == "proved"
        assert witness is None

    @pytest.mark.parametrize("solver", [10], indirect=True)
    def test_unknown_when_budget_is_exhausted(self, solver):
        """Same never-satisfiable-oracle-under-tiny-max_iters shape as
        `TestFindArgopt.test_exhausts_iterations_without_a_proof`, phrased
        as a universal: a domain too big to exhaust in 2 tries must be
        reported as "unknown", never conflated with a real proof."""
        solver_obj, _ = solver
        x = (1,) * 10  # 3**10 candidates

        # NOT(formula_expr) reduces to "FULL(y) * NOT(FULL(y))" - the same
        # always-false oracle as the reference test - after the double
        # negation `check_universal` introduces cancels out.
        status, witness = solver_obj.check_universal(
            var="y",
            formula_expr="NOT((FULL(y)) * (NOT(FULL(y))))",
            context={"x": x},
            skeleton_expr="(y <<= x)",
            max_iters=2,
        )

        assert status == "unknown"
        assert witness is None
        assert solver_obj.last_status == "exhausted"

    @pytest.mark.parametrize("solver", [3], indirect=True)
    def test_fase1_budget_is_scoped_to_the_call(self, solver):
        """`fase1_budget` must not leak into the evaluator's own
        persistent `sigma_search_budget`."""
        solver_obj, _ = solver
        original_budget = solver_obj.evaluator.sigma.search_budget
        x = (1, 1, 1)

        solver_obj.check_universal(
            var="y",
            formula_expr="SIZE(y) <= 1",
            context={"x": x},
            skeleton_expr="(y <<= x)",
            fase1_budget=5,
        )

        assert solver_obj.evaluator.sigma.search_budget == original_budget


@pytest.fixture(scope="module")
def real_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyjuice")
    if not torch.cuda.is_available():
        pytest.skip("PyJuice circuit compilation requires CUDA")
    if not MODEL_DIR.exists():
        pytest.skip(f"{MODEL_DIR} not found - run examples/train_spam_model.py first")

    from qfoil.backends.pyjuice import PyJuiceModel, PyJuiceProbabilityBackend

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PyJuiceModel.load(str(MODEL_DIR), device=device)
    return QFOIL(
        probabilistic_backend=PyJuiceProbabilityBackend(model),
        instance_len=model.num_features,
    )


class TestMinimumPSRIntegration:
    """The motivating use case: find a MinimumPSR-style witness with EP
    verified against a real PyJuiceModel, without Z3 ever seeing EP."""

    def test_finds_known_minimum_psr(self, real_model):
        parser = real_model
        solver = Z3CegisSolver(parser.evaluator, parser)

        x = (1, 1, 0, 0)
        k = 1
        delta = 0.75
        size_max = 2

        result = solver.find_argopt(
            var="y",
            skeleton_expr=(
                f"(y <<= x) * (1 <= SIZE(y)) * (SIZE(y) <= {size_max})"
            ),
            oracle_expr=f"EP(y, k) >= {delta}",
            context={"x": x, "k": k},
            minimize="SIZE(y)",
        )

        # Hand-validated in examples/intro_explainability.ipynb: the unique
        # size-1 sufficient reason for this instance/threshold is "C=0".
        assert result == (None, None, 0, None)

        # And it really does satisfy EP >= delta under the exact evaluator.
        ep = parser.evaluator.evaluate(
            parser.parse("EP(y, k)"), {"y": result, "k": k}
        )
        assert ep >= delta

    def test_warm_start_finds_the_same_minimum_psr(self, real_model):
        """Same query, `warm_start=True`: Fase 1 (now able to score
        `EP(y, k) >= delta` directly via `SigmaSearch.soft_violation`)
        should reach the same unique answer as plain
        Z3 search."""
        parser = real_model
        solver = Z3CegisSolver(parser.evaluator, parser)

        x = (1, 1, 0, 0)
        k = 1
        delta = 0.75
        size_max = 2

        result = solver.find_argopt(
            var="y",
            skeleton_expr=(
                f"(y <<= x) * (1 <= SIZE(y)) * (SIZE(y) <= {size_max})"
            ),
            oracle_expr=f"EP(y, k) >= {delta}",
            context={"x": x, "k": k},
            minimize="SIZE(y)",
            warm_start=True,
        )

        assert result == (None, None, 0, None)
