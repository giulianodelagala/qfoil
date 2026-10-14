"""Validate PSR, MinimumPSR, and MPSR against an exhaustive MUX oracle."""

import pytest

from examples.models.mux import mux_output

from .psr_oracle import (
    is_psr,
    minimum_sufficient_reasons,
    most_probable_sufficient_reasons,
    subsets_of,
)

pytestmark = pytest.mark.cuda

INSTANCE = (0, 1, 0, 1, 1, 0, 0, 1)
CLASS_ID = mux_output(INSTANCE)


def _se_expression(variable):
    return (
        f"FULL(x) * ({variable} <<< x) * (EP({variable}, k) >= delta) * "
        f"(1 <= SIZE({variable})) * (SIZE({variable}) <= size_max)"
    )


def _minimum_psr_expression(variable):
    candidate = _se_expression(variable)
    smaller = _se_expression("z")
    return (
        f"({candidate}) * "
        f"(SIGMA z. (({smaller}) * NOT(SIZE({variable}) <= SIZE(z))) == 0)"
    )


def _mpsr_expression(variable):
    candidate = _minimum_psr_expression(variable)
    competitor = _minimum_psr_expression("w")
    return (
        f"({candidate}) * "
        f"(SIGMA w. (({competitor}) * NOT(VAL(w) <= VAL({variable}))) == 0)"
    )


@pytest.mark.parametrize(
    "distribution_fixture,evaluator_fixture,delta",
    [
        ("uniform_mux_distribution", "uniform_mux_evaluator", 0.75),
        ("bernoulli_mux_distribution", "bernoulli_mux_evaluator", 0.9),
    ],
)
def test_psr_matches_oracle_over_every_subset_of_the_instance(
    request, distribution_fixture, evaluator_fixture, delta
):
    distribution = request.getfixturevalue(distribution_fixture)
    parser, evaluator = request.getfixturevalue(evaluator_fixture)
    expression = parser.parse(
        "FULL(x) * (y <<< x) * (EP(y, k) >= delta)"
    )

    for candidate in subsets_of(INSTANCE, min_size=0):
        actual = evaluator.evaluate(
            expression,
            {
                "x": INSTANCE,
                "y": candidate,
                "k": CLASS_ID,
                "delta": delta,
            },
        )
        expected = is_psr(
            distribution, INSTANCE, candidate, CLASS_ID, delta
        )
        assert bool(actual) is expected


@pytest.mark.parametrize(
    "distribution_fixture,evaluator_fixture,delta",
    [
        ("uniform_mux_distribution", "uniform_mux_evaluator", 0.75),
        ("bernoulli_mux_distribution", "bernoulli_mux_evaluator", 0.9),
    ],
)
def test_minimum_psr_matches_representative_candidates(
    request, distribution_fixture, evaluator_fixture, delta
):
    distribution = request.getfixturevalue(distribution_fixture)
    parser, evaluator = request.getfixturevalue(evaluator_fixture)
    expression = parser.parse(_minimum_psr_expression("y"))
    expected_reasons = set(
        minimum_sufficient_reasons(
            distribution, INSTANCE, CLASS_ID, delta, max_size=8
        )
    )

    noise_only = (None, None, None, None, None, None, 0, 1)
    sufficient_but_not_minimum = INSTANCE
    candidates = expected_reasons | {noise_only, sufficient_but_not_minimum}

    for candidate in candidates:
        actual = evaluator.evaluate(
            expression,
            {
                "x": INSTANCE,
                "y": candidate,
                "k": CLASS_ID,
                "delta": delta,
                "size_max": 8,
            },
        )
        assert bool(actual) is (candidate in expected_reasons)


@pytest.mark.slow
def test_mpsr_ranks_two_reasons_above_the_class_prior(
    bernoulli_mux_distribution, bernoulli_mux_evaluator
):
    parser, evaluator = bernoulli_mux_evaluator
    delta = 0.65
    expression = parser.parse(_mpsr_expression("y"))
    winners = most_probable_sufficient_reasons(
        bernoulli_mux_distribution,
        INSTANCE,
        CLASS_ID,
        delta,
        max_size=8,
    )
    winner = winners[0]
    lower_probability_minimum = (
        None,
        1,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    context = {
        "x": INSTANCE,
        "k": CLASS_ID,
        "delta": delta,
        "size_max": 8,
    }

    assert evaluator.evaluate(expression, {**context, "y": winner}) == 1
    assert (
        evaluator.evaluate(
            expression,
            {**context, "y": lower_probability_minimum},
        )
        == 0
    )
