from itertools import product

import pytest
import torch

from examples.models.mux import (
    FEATURE_NAMES,
    MuxDistribution,
    enumerate_assignments,
    enumerate_partial_assignments,
    mux_output,
)


def test_mux_uses_s0_as_low_selector_bit():
    data_bits = (0, 1, 1, 0)
    for s0, s1 in product((0, 1), repeat=2):
        assignment = (s0, s1, *data_bits, 0, 1)
        assert mux_output(assignment) == data_bits[2 * s1 + s0]


def test_domains_have_expected_sizes():
    complete = list(enumerate_assignments())
    partial = list(enumerate_partial_assignments())
    assert len(FEATURE_NAMES) == 8
    assert len(complete) == len(set(complete)) == 256
    assert len(partial) == len(set(partial)) == 6561


def test_uniform_distribution_is_normalized(uniform_mux_distribution):
    probabilities = [
        uniform_mux_distribution.assignment_probability(assignment)
        for assignment in enumerate_assignments()
    ]
    assert probabilities == pytest.approx([1.0 / 256.0] * 256)
    assert sum(probabilities) == pytest.approx(1.0)
    assert uniform_mux_distribution.priors == pytest.approx({0: 0.5, 1: 0.5})


def test_bernoulli_distribution_and_class_priors_are_normalized(
    bernoulli_mux_distribution,
):
    distribution = bernoulli_mux_distribution
    total = sum(
        distribution.assignment_probability(assignment)
        for assignment in enumerate_assignments()
    )
    joint_totals = {
        class_id: sum(
            distribution.joint_probability(assignment, class_id)
            for assignment in enumerate_assignments()
        )
        for class_id in (0, 1)
    }
    assert total == pytest.approx(1.0)
    assert sum(distribution.priors.values()) == pytest.approx(1.0)
    assert joint_totals == pytest.approx(distribution.priors)


@pytest.mark.parametrize(
    "evidence,expected",
    [
        ((None,) * 8, 1.0),
        ((1, None, None, None, None, None, None, None), 0.35),
        ((None, 1, None, None, None, None, None, None), 0.65),
        ((None, None, None, None, None, None, 1, None), 0.20),
        ((None, None, None, None, None, None, None, 1), 0.80),
        (
            (1, 1, None, None, None, 1, 0, 1),
            0.35 * 0.65 * 0.90 * 0.80 * 0.80,
        ),
    ],
)
def test_bernoulli_evidence_probability_matches_independence(
    bernoulli_mux_distribution, evidence, expected
):
    assert bernoulli_mux_distribution.evidence_probability(evidence) == pytest.approx(
        expected
    )


def test_deterministic_label_has_unit_posterior(uniform_mux_distribution):
    for assignment in enumerate_assignments():
        label = mux_output(assignment)
        posterior = uniform_mux_distribution.posterior_class(assignment, label)
        assert posterior == pytest.approx(1.0)
        assert uniform_mux_distribution.posterior_class(
            assignment, 1 - label
        ) == pytest.approx(0.0)


@pytest.mark.parametrize("class_id", [0, 1])
def test_noise_does_not_change_class_posterior(
    bernoulli_mux_distribution, class_id
):
    core_evidence = (None, 1, None, None, None, None, None, None)
    with_noise = (None, 1, None, None, None, None, 0, 1)

    assert bernoulli_mux_distribution.posterior_class(
        with_noise, class_id
    ) == pytest.approx(
        bernoulli_mux_distribution.posterior_class(core_evidence, class_id)
    )


def test_seeded_sampling_returns_mux_labels(bernoulli_mux_distribution):
    data, labels = bernoulli_mux_distribution.sample(200, seed=1234)
    assert data.shape == (200, 8)
    assert labels.shape == (200,)
    expected = torch.tensor([mux_output(row.tolist()) for row in data])
    assert torch.equal(labels, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "selector_probs": (0.5,),
            "data_probs": (0.5,) * 4,
            "noise_probs": (0.5, 0.5),
        },
        {
            "selector_probs": (0.5, 0.5),
            "data_probs": (0.5,) * 3,
            "noise_probs": (0.5, 0.5),
        },
        {
            "selector_probs": (0.5, 0.5),
            "data_probs": (0.5,) * 4,
            "noise_probs": (0.5,),
        },
        {
            "selector_probs": (0.0, 0.5),
            "data_probs": (0.5,) * 4,
            "noise_probs": (0.5, 0.5),
        },
        {
            "selector_probs": (0.5, 0.5),
            "data_probs": (0.5, 0.5, 1.0, 0.5),
            "noise_probs": (0.5, 0.5),
        },
        {
            "selector_probs": (0.5, 0.5),
            "data_probs": (0.5,) * 4,
            "noise_probs": (0.5, 1.0),
        },
    ],
)
def test_invalid_distribution_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MuxDistribution(**kwargs)
