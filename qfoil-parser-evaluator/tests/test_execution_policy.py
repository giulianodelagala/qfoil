"""Explicit execution-policy contract: modes vs. budgets.

`sigma_mode` selects the *procedure* ("heuristic" never enumerates;
"exact" always decides by exhaustive enumeration, selected explicitly).
The local budgets (`sigma_search_budget`, the independent witness-search
`budget`, `inner_budget`) and the new global `evaluation_budget` only bound
*how much work* an attempt may spend -- none of them selects the mode, and
`sigma_search_budget=0` is not a back door into exact enumeration.
"""

import pytest

import qfoil.quantifiers.sigma as sigma_module
from qfoil import QFOIL, Inconclusive


@pytest.fixture
def forbid_enumeration(monkeypatch):
    """Make any call to the shared ``product`` helper fail the test.

    `SigmaEvaluator.evaluate_exact` and `_decide_by_enumeration` are the
    only two places in the runtime that enumerate the ternary domain, and
    both reach it through this one imported name -- patching it here is a
    single, sufficient guard that heuristic mode never enumerates via
    either path, regardless of which formula shape triggered it.
    """

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("heuristic mode must never enumerate the domain")

    monkeypatch.setattr(sigma_module, "product", _forbidden)


class TestModeIsExplicitAndSeparateFromBudgets:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="sigma_mode"):
            QFOIL(sigma_mode="eager")

    def test_default_mode_is_heuristic(self):
        assert QFOIL().evaluator.sigma.mode == "heuristic"

    def test_zero_local_budget_does_not_select_exact_mode(self):
        """`sigma_search_budget=0` only skips the heuristic search attempt;
        it must not fall back to enumeration the way the old, budget-based
        selector did. Mode stays "heuristic" and the result is Inconclusive,
        not a silently computed exact answer."""

        qfoil = QFOIL(instance_len=3, sigma_search_budget=0)
        assert qfoil.evaluator.sigma.mode == "heuristic"
        result = qfoil.evaluate("SIGMA z. FULL(z) >= 1", x=(0, 0, 0))
        assert isinstance(result, Inconclusive)
        assert result.reason == "bounded_search_incomplete"

    def test_exact_mode_ignores_local_search_budget_value(self):
        """Exact mode always enumerates, regardless of `sigma_search_budget`
        (including its default, non-zero value) -- the two are orthogonal."""

        qfoil = QFOIL(instance_len=3, sigma_mode="exact")
        assert qfoil.evaluator.sigma.search_budget == 64  # untouched default
        assert qfoil.evaluate("SIGMA z. FULL(z) >= 1", x=(0, 0, 0)) == 1


class TestHeuristicNeverEnumerates:
    """Covers evaluation, subqueries, logic functions, scoring and
    verification, per the policy in `SigmaEvaluator.decide`/`evaluate_exact`.
    Every case here holds even when the local search budget is 0 or 1 --
    the old code used exactly that budget to decide whether to fall back
    to enumeration; heuristic mode must never make that choice at all.
    """

    @pytest.mark.parametrize("budget", [0, 1, 64])
    def test_flat_existential_never_enumerates(self, forbid_enumeration, budget):
        """FULL(z) needs every position set, SIZE(z) <= 0 forbids setting
        any -- structurally unsatisfiable, so no budget ever makes the
        search succeed, and heuristic mode must still never fall through
        to a scan to double-check that for itself."""

        qfoil = QFOIL(instance_len=3, sigma_search_budget=budget)
        result = qfoil.evaluate(
            "SIGMA z. (FULL(z) * (SIZE(z) <= 0)) >= 1", x=(0, 0, 0)
        )
        assert isinstance(result, Inconclusive)

    @pytest.mark.parametrize("budget", [0, 1, 64])
    def test_flat_universal_never_enumerates(self, forbid_enumeration, budget):
        qfoil = QFOIL(instance_len=3, sigma_search_budget=budget)
        expr = "SIGMA z. ((z <<= x) * NOT(SIZE(z) <= 3)) == 0"
        result = qfoil.evaluate(expr, x=(1, 1, 1))  # genuinely holds, no counterexample
        assert isinstance(result, Inconclusive)

    def test_non_witness_shape_never_enumerates(self, forbid_enumeration):
        """A threshold shape with no witness-search strategy (neither
        existential nor universal) must refuse immediately, not scan."""

        qfoil = QFOIL(instance_len=3)
        result = qfoil.evaluate("(SIGMA z. SIZE(z)) == 2", x=(0, 0, 0))
        assert isinstance(result, Inconclusive)

    def test_bare_count_never_enumerates(self, forbid_enumeration):
        """A SIGMA count with no wrapping comparison has no certification
        strategy at all in heuristic mode -- not even an attempt."""

        qfoil = QFOIL(instance_len=3)
        result = qfoil.evaluate("SIGMA z. SIZE(z)", x=(0, 0, 0))
        assert isinstance(result, Inconclusive)

    def test_both_sides_sigma_comparison_never_enumerates(self, forbid_enumeration):
        qfoil = QFOIL(instance_len=3)
        expr = "(SIGMA u. SIGMA v. FULL(u)) == (SIGMA u. SIGMA v. FULL(u))"
        result = qfoil.evaluate(expr, x=(0, 0, 0))
        assert isinstance(result, Inconclusive)

    def test_nested_sigma_hidden_in_logic_function_never_enumerates(
        self, forbid_enumeration
    ):
        qfoil = QFOIL(instance_len=2)
        qfoil.define_function("INNER", ["x"], "(SIGMA z. (z == x)) >= 1")
        result = qfoil.evaluate(
            "(SIGMA y. ((y == x) * INNER(x))) >= 1", x=(1, 1)
        )
        assert isinstance(result, Inconclusive)

    def test_scoring_an_unsupported_fragment_never_enumerates(
        self, forbid_enumeration
    ):
        """`+` is outside `SigmaSearch.soft_violation`'s scored fragment, so
        scoring the very first candidate already bails out to `None` --
        heuristic mode must stop there, not escalate to a scan."""

        qfoil = QFOIL(instance_len=3)
        result = qfoil.evaluate(
            "SIGMA z. NOT(FULL(z) + L0(z)) >= 1", x=(0, 0, 0)
        )
        assert isinstance(result, Inconclusive)

    def test_rejected_witness_verification_never_enumerates(
        self, forbid_enumeration, monkeypatch
    ):
        """A witness the search proposes but that fails independent
        re-verification must be refused honestly, never rescued by a scan."""

        qfoil = QFOIL(instance_len=3)
        qfoil.evaluator.sigma.search.local_search = lambda *a, **k: (1, 1, 1)
        result = qfoil.evaluate("SIGMA z. L0(z) >= 1", x=(0, 0, 0))
        assert isinstance(result, Inconclusive)


class TestHeuristicValidation:
    @pytest.mark.parametrize("mode", ["heuristic", "exact", "restricted"])
    @pytest.mark.parametrize("operator", ["<<", "<<="])
    def test_nested_query_rejects_incompatible_tuple_lengths(self, mode, operator):
        qfoil = QFOIL(instance_len=4, sigma_mode=mode)
        expression = (
            f"SIGMA y. ((y {operator} x)"
            " * (SIGMA z. (COMP(y,z) * (VAL(z) == 0)) == 0)"
            " * (SIZE(y) <= 6)) >= 1"
        )
        with pytest.raises(
            ValueError, match="SUBSETEQ operation requires equal-length tuples"
        ):
            qfoil.evaluate(expression, x=(1, None, 0))

    @pytest.mark.parametrize("relation", [
        "FULL(y, x)", "L0(1)", "L1(1)", "FEAT(1)",
        "PARTITION(y, x)", "MEET(y, y, x)", "JOIN(y, y, x)",
        "OPPOSITE(y, x)", "SUF(y, x)", "COMP(y, x)",
    ])
    def test_native_scoring_preserves_exact_validation_errors(self, relation):
        expression = f"SIGMA y. ({relation}) >= 1"
        with pytest.raises(ValueError) as exact_error:
            QFOIL(instance_len=4, sigma_mode="exact").evaluate(
                expression, x=(1, None, 0)
            )
        with pytest.raises(ValueError) as heuristic_error:
            QFOIL(instance_len=4).evaluate(expression, x=(1, None, 0))
        assert str(heuristic_error.value) == str(exact_error.value)

    @pytest.mark.parametrize("method", ["grow_witness", "search_witness"])
    def test_public_witness_search_rejects_incompatible_lengths(self, method):
        qfoil = QFOIL(instance_len=4)
        with pytest.raises(
            ValueError, match="SUBSETEQ operation requires equal-length tuples"
        ):
            getattr(qfoil, method)("y << x", "y", {"x": (1, None, 0)})

    @pytest.mark.parametrize("relation", ["y << x", "y <<= x", "COMP(y, x)"])
    @pytest.mark.parametrize("budget", [1, 2, 3])
    def test_unresolved_operands_keep_budget_exhaustion(self, relation, budget):
        qfoil = QFOIL(instance_len=3, evaluation_budget=budget)
        result = qfoil.evaluate(f"SIGMA y. ({relation}) >= 1", x=(1, None, 0))
        assert isinstance(result, Inconclusive)
        assert result.reason == "evaluation_budget_exhausted"


class TestGlobalEvaluationBudget:
    """The shared, per-operation budget: one unit per call to
    `QFOILEvaluator.evaluate()`, covering flat evaluation, subqueries,
    logic functions, heuristic scoring and witness verification alike --
    see `qfoil.runtime.budget.EvaluationBudget`. It is independent from
    `sigma_search_budget` (64 by default), the witness-search `budget`
    (1000 by default) and `inner_budget` (8 by default).
    """

    def test_unlimited_by_default(self):
        assert QFOIL().evaluator.budget.limit is None

    def test_flat_expression_exhausts_with_reason(self):
        """`SIZE(x)` costs exactly 2 units: one for the `SIZE(x)` call
        node, one for its `x` argument."""

        exhausted = QFOIL(instance_len=2, evaluation_budget=1)
        result = exhausted.evaluate("SIZE(x)", x=(1, 0))
        assert isinstance(result, Inconclusive)
        assert result.reason == "evaluation_budget_exhausted"

        sufficient = QFOIL(instance_len=2, evaluation_budget=2)
        assert sufficient.evaluate("SIZE(x)", x=(1, 0)) == 2

    def test_budget_is_shared_across_subexpressions_not_per_subquery(self):
        """`SIZE(x) + SIZE(y)` costs 5 units (1 root `+` node, 2 for each
        `SIZE` call). A budget that comfortably covers the left operand
        alone (3 units) but not both must still exhaust -- proving the
        counter is one shared pool, not reset per operand/subquery."""

        qfoil = QFOIL(instance_len=2, evaluation_budget=3)
        result = qfoil.evaluate("SIZE(x) + SIZE(y)", x=(1, 0), y=(1, None))
        assert isinstance(result, Inconclusive)
        assert result.reason == "evaluation_budget_exhausted"

        qfoil_full = QFOIL(instance_len=2, evaluation_budget=5)
        assert qfoil_full.evaluate("SIZE(x) + SIZE(y)", x=(1, 0), y=(1, None)) == 3

    def test_nested_sigma_subquery_shares_the_same_budget(self):
        """The inner `SIGMA z` subquery does not get its own fresh budget:
        a tight global budget exhausts the whole nested operation, in
        `sigma_mode="exact"` where enumeration would otherwise run deep."""

        qfoil = QFOIL(instance_len=2, sigma_mode="exact", evaluation_budget=5)
        expr = "(SIGMA y. SIGMA z. (y == z)) == 9"
        result = qfoil.evaluate(expr, bindings={})
        assert isinstance(result, Inconclusive)
        assert result.reason == "evaluation_budget_exhausted"
        assert qfoil.evaluator.sigma.last_witness is None

        qfoil_full = QFOIL(instance_len=2, sigma_mode="exact")
        assert qfoil_full.evaluate(expr, bindings={}) == 1

    def test_exhaustion_during_witness_verification_is_reported(self):
        """A known seed separates scoring from independent verification.

        Four node visits reach verification but cannot finish it; five
        suffice. This assertion is independent of random initialization.
        """
        exhausted = QFOIL(instance_len=3, evaluation_budget=4)
        result = exhausted.evaluate("SIGMA z. FULL(z) >= 1", seed=(0, 0, 0))
        assert isinstance(result, Inconclusive)
        assert result.reason == "evaluation_budget_exhausted"
        assert exhausted.evaluator.sigma.last_witness is None

        sufficient = QFOIL(instance_len=3, evaluation_budget=5)
        assert sufficient.evaluate("SIGMA z. FULL(z) >= 1", seed=(0, 0, 0)) == 1

    def test_state_does_not_leak_between_top_level_operations(self):
        """Exhausting the budget on one `evaluate()` call must not affect
        the next: the counter is rearmed at the start of every new
        top-level operation, on the same long-lived evaluator."""

        qfoil = QFOIL(instance_len=2, evaluation_budget=2)
        exhausted = qfoil.evaluate("SIZE(x) + SIZE(y)", x=(1, 0), y=(1, None))
        assert isinstance(exhausted, Inconclusive)

        fresh = qfoil.evaluate("SIZE(x)", x=(1, 0))
        assert fresh == 2  # the second, independent call gets its full budget back

    def test_rejects_invalid_budget_value(self):
        with pytest.raises(ValueError, match="evaluation_budget"):
            QFOIL(evaluation_budget=-1)
        with pytest.raises(ValueError, match="evaluation_budget"):
            QFOIL(evaluation_budget=1.5)
