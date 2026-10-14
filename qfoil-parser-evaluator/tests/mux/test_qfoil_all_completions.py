"""Validate all-completions SR/CE queries against the exhaustive oracle."""

import pytest

from .global_oracle import (
    all_completions_prob_ce,
    all_completions_prob_sr,
)

pytestmark = pytest.mark.cuda


def _all_completions_expression(score_function):
    return (
        "SIGMA y. ("
        "    COMP(x, y) * "
        f"    NOT({score_function}(y, k) >= delta)"
        ") == 0"
    )


@pytest.mark.parametrize(
    "partial",
    [
        (0, 1, None, None, 1, None, None, None),
        (0, 1, None, None, 0, None, None, None),
        (0, 1, None, None, None, None, None, None),
        (0, 1, 0, 1, 1, 0, 0, 1),
        (0, 1, 0, 1, 0, 0, 0, 1),
    ],
)
@pytest.mark.parametrize(
    "distribution_fixture,evaluator_fixture",
    [
        ("uniform_mux_distribution", "uniform_mux_evaluator"),
        ("bernoulli_mux_distribution", "bernoulli_mux_evaluator"),
    ],
)
def test_all_completions_sr_and_ce_match_oracle(
    request,
    distribution_fixture,
    evaluator_fixture,
    partial,
):
    distribution = request.getfixturevalue(distribution_fixture)
    parser, evaluator = request.getfixturevalue(evaluator_fixture)
    class_id = 1
    delta = 0.9
    all_sr = parser.parse(_all_completions_expression("EP"))
    all_ce = parser.parse(_all_completions_expression("CEP"))
    context = {"x": partial, "k": class_id, "delta": delta}

    actual_sr = bool(evaluator.evaluate(all_sr, context))
    actual_ce = bool(evaluator.evaluate(all_ce, context))
    expected_sr = all_completions_prob_sr(
        distribution, partial, class_id, delta
    )
    expected_ce = all_completions_prob_ce(
        distribution, partial, class_id, delta
    )

    assert actual_sr is expected_sr
    assert actual_ce is expected_ce
