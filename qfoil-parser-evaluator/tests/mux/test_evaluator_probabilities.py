"""Validate QFOIL's probabilistic primitives against the MUX oracle."""

import pytest

from examples.models.mux import enumerate_partial_assignments

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize(
    "distribution_fixture,evaluator_fixture",
    [
        ("uniform_mux_distribution", "uniform_mux_evaluator"),
        ("bernoulli_mux_distribution", "bernoulli_mux_evaluator"),
    ],
)
def test_val_matches_every_partial_assignment(
    request, distribution_fixture, evaluator_fixture
):
    distribution = request.getfixturevalue(distribution_fixture)
    parser, evaluator = request.getfixturevalue(evaluator_fixture)
    val = parser.parse("VAL(x)")
    val_for_class = parser.parse("VAL(x, k)")

    for evidence in enumerate_partial_assignments():
        actual = evaluator.evaluate(val, {"x": evidence})
        expected = distribution.evidence_probability(evidence)
        assert actual == pytest.approx(expected, abs=1e-6)

        for class_id in (0, 1):
            actual_joint = evaluator.evaluate(
                val_for_class, {"x": evidence, "k": class_id}
            )
            expected_joint = distribution.evidence_probability(
                evidence, class_id=class_id
            )
            assert actual_joint == pytest.approx(expected_joint, abs=1e-6)


@pytest.mark.parametrize(
    "distribution_fixture,evaluator_fixture",
    [
        ("uniform_mux_distribution", "uniform_mux_evaluator"),
        ("bernoulli_mux_distribution", "bernoulli_mux_evaluator"),
    ],
)
def test_ep_and_cep_match_every_partial_assignment(
    request, distribution_fixture, evaluator_fixture
):
    distribution = request.getfixturevalue(distribution_fixture)
    parser, evaluator = request.getfixturevalue(evaluator_fixture)
    ep = parser.parse("EP(x, k)")
    cep = parser.parse("CEP(x, k)")

    for evidence in enumerate_partial_assignments():
        for class_id in (0, 1):
            expected = distribution.posterior_class(evidence, class_id)
            actual_ep = evaluator.evaluate(ep, {"x": evidence, "k": class_id})
            actual_cep = evaluator.evaluate(cep, {"x": evidence, "k": class_id})

            assert actual_ep == pytest.approx(expected, abs=1e-6)
            assert actual_cep == pytest.approx(1.0 - expected, abs=1e-6)
            assert actual_ep + actual_cep == pytest.approx(1.0, abs=1e-6)


def test_ep_and_cep_are_invariant_to_noise_observations(
    bernoulli_mux_evaluator,
):
    parser, evaluator = bernoulli_mux_evaluator
    ep = parser.parse("EP(x, k)")
    cep = parser.parse("CEP(x, k)")
    core = (None, 1, None, None, None, None, None, None)
    with_noise = (None, 1, None, None, None, None, 0, 1)

    core_context = {"x": core, "k": 1}
    noise_context = {"x": with_noise, "k": 1}

    assert evaluator.evaluate(ep, noise_context) == pytest.approx(
        evaluator.evaluate(ep, core_context), abs=1e-6
    )
    assert evaluator.evaluate(cep, noise_context) == pytest.approx(
        evaluator.evaluate(cep, core_context), abs=1e-6
    )


@pytest.mark.parametrize(
    "event,given,class_id",
    [
        ((None,) * 8, (None,) * 8, 0),
        ((None,) * 8, (None,) * 8, 1),
        (
            (None, None, 1, None, None, None, None, None),
            (0, 0, None, None, None, None, None, None),
            1,
        ),
        (
            (None, None, 1, None, None, None, None, None),
            (0, 0, None, None, None, None, None, None),
            0,
        ),
        (
            (None, None, None, 0, None, None, None, None),
            (1, 0, None, None, None, None, None, None),
            0,
        ),
        (
            (None, None, None, None, 1, None, None, None),
            (0, 1, None, None, None, None, None, None),
            1,
        ),
        (
            (None, None, None, None, None, 0, None, None),
            (1, 1, None, None, None, None, None, None),
            0,
        ),
        (
            (None, None, 1, 0, None, None, None, None),
            (0, 0, None, None, None, None, None, None),
            1,
        ),
        (
            (1, None, None, None, None, None, None, None),
            (0, None, None, None, None, None, None, None),
            1,
        ),
        (
            (None, None, None, None, None, None, 1, None),
            (0, 1, None, None, 1, None, None, None),
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "distribution_fixture,evaluator_fixture",
    [
        ("uniform_mux_distribution", "uniform_mux_evaluator"),
        ("bernoulli_mux_distribution", "bernoulli_mux_evaluator"),
    ],
)
def test_cond_matches_representative_ground_truth_cases(
    request,
    distribution_fixture,
    evaluator_fixture,
    event,
    given,
    class_id,
):
    distribution = request.getfixturevalue(distribution_fixture)
    parser, evaluator = request.getfixturevalue(evaluator_fixture)
    cond = parser.parse("COND(event, given, k)")

    if any(
        event_value is not None
        and given_value is not None
        and event_value != given_value
        for event_value, given_value in zip(event, given)
    ):
        expected = 0.0
    else:
        fused = tuple(
            event_value if event_value is not None else given_value
            for event_value, given_value in zip(event, given)
        )
        numerator = distribution.evidence_probability(fused, class_id=class_id)
        denominator = distribution.evidence_probability(given)
        expected = numerator / denominator

    actual = evaluator.evaluate(
        cond,
        {"event": event, "given": given, "k": class_id},
    )
    assert actual == pytest.approx(expected, abs=1e-6)
