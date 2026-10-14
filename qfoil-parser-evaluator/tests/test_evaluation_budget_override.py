"""Per-operation work limits apply to nested SIGMA and restore defaults."""

import pytest

from qfoil import QFOIL
from qfoil.results import Inconclusive


def test_budget_override_is_shared_by_nested_queries_and_restored():
    runtime = QFOIL(instance_len=2, evaluation_budget=1)
    expression = "SIGMA y. ((SIGMA z. FULL(z)) == 4)"
    result = runtime.evaluate(expression, sigma_mode="exact", evaluation_budget=5)
    assert isinstance(result, Inconclusive)
    assert result.reason == "evaluation_budget_exhausted"
    assert runtime.evaluator.budget.limit == 1
    assert runtime.evaluate(expression, sigma_mode="exact", evaluation_budget=1000) == 9
    assert runtime.evaluator.budget.limit == 1
    assert isinstance(runtime.evaluate(expression, sigma_mode="exact"), Inconclusive)


def test_budget_override_restored_after_evaluation_error():
    runtime = QFOIL(evaluation_budget=1)
    with pytest.raises(ValueError, match="missing"):
        runtime.evaluate("missing", evaluation_budget=1000)
    assert runtime.evaluator.budget.limit == 1


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budget_override(budget):
    with pytest.raises(ValueError, match="evaluation_budget"):
        QFOIL().evaluate("1", evaluation_budget=budget)
