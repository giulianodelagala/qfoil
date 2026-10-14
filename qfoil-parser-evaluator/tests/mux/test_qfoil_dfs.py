"""Validate threshold determination and DFS against the MUX oracle."""

import pytest

from .global_oracle import (
    is_determinant_feature_set,
    is_threshold_determined,
)

pytestmark = pytest.mark.cuda


def _all_completions_expression(score_function, variable):
    return (
        "SIGMA completion. ("
        f"    COMP({variable}, completion) * "
        f"    NOT({score_function}(completion, k) >= delta)"
        ") == 0"
    )


def _threshold_determined_expression(variable):
    all_sr = _all_completions_expression("EP", variable)
    all_ce = _all_completions_expression("CEP", variable)
    return f"NOT(NOT({all_sr}) * NOT({all_ce}))"


def _dfs_expression():
    determined = _threshold_determined_expression("y")
    return f"SIGMA y. (SUF(x, y) * NOT({determined})) == 0"


@pytest.mark.parametrize(
    "partial",
    [
        (0, 1, None, None, 1, None, None, None),
        (0, 1, None, None, 0, None, None, None),
        (0, 1, None, None, None, None, None, None),
    ],
)
def test_threshold_determined_matches_oracle(
    bernoulli_mux_distribution,
    bernoulli_mux_evaluator,
    partial,
):
    parser, evaluator = bernoulli_mux_evaluator
    expression = parser.parse(_threshold_determined_expression("x"))
    context = {"x": partial, "k": 1, "delta": 0.9}

    actual = bool(evaluator.evaluate(expression, context))
    expected = is_threshold_determined(
        bernoulli_mux_distribution, partial, 1, 0.9
    )

    assert actual is expected


@pytest.mark.slow
@pytest.mark.parametrize(
    "partial",
    [
        (0, 1, None, None, 1, None, None, None),
        (0, 1, 0, 1, 1, 0, None, None),
    ],
)
def test_dfs_matches_oracle_for_partial_and_full_support(
    bernoulli_mux_distribution,
    bernoulli_mux_evaluator,
    partial,
):
    parser, evaluator = bernoulli_mux_evaluator
    expression = parser.parse(_dfs_expression())
    context = {"x": partial, "k": 1, "delta": 0.9}

    actual = bool(evaluator.evaluate(expression, context))
    expected = is_determinant_feature_set(
        bernoulli_mux_distribution, partial, 1, 0.9
    )

    assert actual is expected
