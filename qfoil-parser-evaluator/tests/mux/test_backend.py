import pytest
import torch

from examples.models.mux import (
    enumerate_assignments,
    enumerate_partial_assignments,
    mux_output,
)
from qfoil.backends.pyjuice import PyJuiceInference

pytestmark = pytest.mark.cuda


def _partial_batch(device):
    partial = list(enumerate_partial_assignments())
    data = torch.tensor(
        [[0 if value is None else value for value in row] for row in partial],
        dtype=torch.long,
        device=device,
    )
    observed = torch.tensor(
        [[value is not None for value in row] for row in partial],
        dtype=torch.bool,
        device=device,
    )
    return partial, data, observed


@pytest.mark.parametrize(
    "distribution_fixture,model_fixture",
    [
        ("uniform_mux_distribution", "uniform_mux_model"),
        ("bernoulli_mux_distribution", "bernoulli_mux_model"),
    ],
)
def test_backend_marginals_match_all_partial_assignments(
    request, cuda_device, distribution_fixture, model_fixture
):
    distribution = request.getfixturevalue(distribution_fixture)
    model = request.getfixturevalue(model_fixture)
    partial, data, observed = _partial_batch(cuda_device)

    marginal = torch.exp(
        PyJuiceInference.marginal_query(
            model.pc_per_class, model.priors, data, observed
        )
    ).reshape(-1)
    expected = torch.tensor(
        [distribution.evidence_probability(row) for row in partial],
        dtype=torch.float32,
        device=cuda_device,
    )
    assert torch.allclose(marginal, expected, atol=1e-6, rtol=1e-5)

    for class_id in (0, 1):
        joint = torch.exp(
            PyJuiceInference.marginal_query(
                model.pc_per_class,
                model.priors,
                data,
                observed,
                class_value=class_id,
            )
        ).reshape(-1)
        expected_joint = torch.tensor(
            [
                distribution.evidence_probability(row, class_id=class_id)
                for row in partial
            ],
            dtype=torch.float32,
            device=cuda_device,
        )
        assert torch.allclose(joint, expected_joint, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "distribution_fixture,model_fixture",
    [
        ("uniform_mux_distribution", "uniform_mux_model"),
        ("bernoulli_mux_distribution", "bernoulli_mux_model"),
    ],
)
def test_backend_posteriors_match_all_partial_assignments(
    request, cuda_device, distribution_fixture, model_fixture
):
    distribution = request.getfixturevalue(distribution_fixture)
    model = request.getfixturevalue(model_fixture)
    partial, data, observed = _partial_batch(cuda_device)
    target = torch.zeros_like(observed)

    log_posteriors = PyJuiceInference.conditional_query(
        model.pc_per_class,
        model.priors,
        data,
        target,
        observed,
        class_value=None,
        class_as_target=True,
    )
    actual = torch.exp(log_posteriors)
    expected = torch.tensor(
        [
            [distribution.posterior_class(row, class_id) for class_id in (0, 1)]
            for row in partial
        ],
        dtype=torch.float32,
        device=cuda_device,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    expected_totals = torch.ones(len(partial), device=cuda_device)
    assert torch.allclose(actual.sum(dim=1), expected_totals)


@pytest.mark.parametrize("model_fixture", ["uniform_mux_model", "bernoulli_mux_model"])
def test_backend_classifies_every_complete_assignment(
    request, cuda_device, model_fixture
):
    model = request.getfixturevalue(model_fixture)
    assignments = list(enumerate_assignments())
    data = torch.tensor(assignments, dtype=torch.long, device=cuda_device)
    predictions, probabilities = PyJuiceInference.classify(
        model.pc_per_class, model.priors, data
    )
    expected = torch.tensor(
        [mux_output(row) for row in assignments], dtype=torch.long, device=cuda_device
    )
    assert torch.equal(predictions, expected)
    assert torch.allclose(
        probabilities[torch.arange(len(assignments), device=cuda_device), expected],
        torch.ones(len(assignments), device=cuda_device),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "event,given,class_id",
    [
        (
            (None, None, 1, None, None, None, None, None),
            (0, 0, None, None, None, None, None, None),
            None,
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
            None,
        ),
        (
            (None, None, None, None, None, None, 1, None),
            (0, 1, None, None, 1, None, None, None),
            1,
        ),
    ],
)
def test_backend_feature_conditionals_match_ground_truth(
    bernoulli_mux_distribution,
    bernoulli_mux_model,
    cuda_device,
    event,
    given,
    class_id,
):
    fused = tuple(
        given_value if given_value is not None else event_value
        for event_value, given_value in zip(event, given)
    )
    data = torch.tensor(
        [[0 if value is None else value for value in fused]],
        dtype=torch.long,
        device=cuda_device,
    ).repeat(4, 1)
    target = torch.tensor(
        [[value is not None for value in event]], dtype=torch.bool, device=cuda_device
    ).repeat(4, 1)
    observed = torch.tensor(
        [[value is not None for value in given]], dtype=torch.bool, device=cuda_device
    ).repeat(4, 1)

    actual = torch.exp(
        PyJuiceInference.conditional_query(
            bernoulli_mux_model.pc_per_class,
            bernoulli_mux_model.priors,
            data,
            target,
            observed,
            class_value=class_id,
            class_as_target=False,
        )
    ).reshape(-1)[0]
    expected = bernoulli_mux_distribution.conditional_probability(
        event, given, class_id=class_id
    )
    assert actual.item() == pytest.approx(expected, abs=1e-6)
