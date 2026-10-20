"""Validate PCE, MinimumPCE, and MPCE against an exhaustive MUX oracle."""

import pytest

from examples.models.mux import mux_output

from .pce_oracle import (
    is_pce,
    minimum_contrastive_explanations,
    most_contrastive_explanations,
    released_features,
)
from .psr_oracle import subsets_of

pytestmark = pytest.mark.cuda

INSTANCE = (0, 0, 1, 0, 0, 0, 0, 1)
CLASS_ID = mux_output(INSTANCE)


def _free_expression(variable):
    return f"FREE(x, {variable})"


def _sce_expression(variable):
    free = _free_expression(variable)
    return (
        f"FULL(x) * ({variable} <<= x) * (CEP({variable}, k) >= delta) * "
        f"(1 <= ({free})) * (({free}) <= release_max)"
    )


def _minimum_pce_expression(variable):
    candidate = _sce_expression(variable)
    competitor = _sce_expression("z")
    candidate_free = _free_expression(variable)
    competitor_free = _free_expression("z")
    return (
        f"({candidate}) * "
        f"(SIGMA z. (({competitor}) * "
        f"NOT(({candidate_free}) <= ({competitor_free}))) == 0)"
    )


def _mpce_expression(variable):
    candidate = _minimum_pce_expression(variable)
    competitor = _minimum_pce_expression("w")
    return (
        f"({candidate}) * "
        f"(SIGMA w. (({competitor}) * "
        f"NOT(CEP(w, k) <= CEP({variable}, k))) == 0)"
    )


def test_free_matches_the_number_of_released_features(
    bernoulli_mux_evaluator,
):
    parser, evaluator = bernoulli_mux_evaluator
    expression = parser.parse("FREE(x, y)")

    for candidate in subsets_of(INSTANCE, min_size=0):
        actual = evaluator.evaluate(
            expression, {"x": INSTANCE, "y": candidate}
        )
        assert actual == released_features(INSTANCE, candidate)


@pytest.mark.parametrize(
    "distribution_fixture,evaluator_fixture,delta",
    [
        ("uniform_mux_distribution", "uniform_mux_evaluator", 0.75),
        ("bernoulli_mux_distribution", "bernoulli_mux_evaluator", 0.8),
    ],
)
def test_pce_matches_oracle_over_every_subset_of_the_instance(
    request, distribution_fixture, evaluator_fixture, delta
):
    distribution = request.getfixturevalue(distribution_fixture)
    parser, evaluator = request.getfixturevalue(evaluator_fixture)
    expression = parser.parse(
        "FULL(x) * (y <<= x) * (CEP(y, k) >= delta)"
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
        expected = is_pce(
            distribution, INSTANCE, candidate, CLASS_ID, delta
        )
        assert bool(actual) is expected


@pytest.mark.parametrize(
    "distribution_fixture,evaluator_fixture,delta",
    [
        ("uniform_mux_distribution", "uniform_mux_evaluator", 0.75),
        ("bernoulli_mux_distribution", "bernoulli_mux_evaluator", 0.8),
    ],
)
def test_minimum_pce_matches_representative_candidates(
    request, distribution_fixture, evaluator_fixture, delta
):
    distribution = request.getfixturevalue(distribution_fixture)
    parser, evaluator = request.getfixturevalue(evaluator_fixture)
    expression = parser.parse(_minimum_pce_expression("y"))
    expected_explanations = set(
        minimum_contrastive_explanations(
            distribution,
            INSTANCE,
            CLASS_ID,
            delta,
            max_released=8,
        )
    )

    release_noise_only = (0, 0, 1, 0, 0, 0, None, 1)
    candidates = expected_explanations | {INSTANCE, release_noise_only}

    for candidate in candidates:
        actual = evaluator.evaluate(
            expression,
            {
                "x": INSTANCE,
                "y": candidate,
                "k": CLASS_ID,
                "delta": delta,
                "release_max": 8,
            },
        )
        assert bool(actual) is (candidate in expected_explanations)


@pytest.mark.slow
def test_mpce_selects_the_high_threshold_explanation_without_noise(
    bernoulli_mux_distribution, bernoulli_mux_evaluator
):
    parser, evaluator = bernoulli_mux_evaluator
    delta = 0.8
    expression = parser.parse(_mpce_expression("y"))
    winners = most_contrastive_explanations(
        bernoulli_mux_distribution,
        INSTANCE,
        CLASS_ID,
        delta,
        max_released=8,
    )
    nonminimum_with_released_noise = (0, 0, None, 0, 0, 0, None, 1)
    context = {
        "x": INSTANCE,
        "k": CLASS_ID,
        "delta": delta,
        "release_max": 8,
    }

    assert evaluator.evaluate(
        expression, {**context, "y": winners[0]}
    ) == 1
    assert evaluator.evaluate(
        expression, {**context, "y": nonminimum_with_released_noise}
    ) == 0
