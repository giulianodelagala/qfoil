from math import prod

import pyjuice as juice
import pytest
import torch

from examples.models.mux import build_mux_pc_node, enumerate_assignments


def _complete_batch(device):
    return torch.tensor(list(enumerate_assignments()), dtype=torch.long, device=device)


def _evaluate_uncompiled_pc(root, assignment):
    """Evaluate this deliberately simple sum-of-products graph on CPU."""
    weights = root.get_params().reshape(-1).tolist()
    branch_probabilities = []
    for branch in root.chs:
        leaf_probabilities = []
        for leaf in branch.chs:
            variable = next(iter(leaf.scope))
            leaf_probabilities.append(leaf.get_params()[assignment[variable]].item())
        branch_probabilities.append(prod(leaf_probabilities))
    return sum(
        weight * probability
        for weight, probability in zip(weights, branch_probabilities)
    )


@pytest.mark.parametrize(
    "distribution_fixture",
    ["uniform_mux_distribution", "bernoulli_mux_distribution"],
)
def test_uncompiled_class_conditional_graph_matches_ground_truth(
    request, distribution_fixture
):
    distribution = request.getfixturevalue(distribution_fixture)
    for class_id in (0, 1):
        root = build_mux_pc_node(distribution, class_id)
        prior = distribution.class_probability(class_id)
        for assignment in enumerate_assignments():
            expected = distribution.joint_probability(assignment, class_id) / prior
            assert _evaluate_uncompiled_pc(root, assignment) == pytest.approx(
                expected, abs=1e-7
            )


@pytest.mark.parametrize(
    "distribution_fixture,model_fixture",
    [
        ("uniform_mux_distribution", "uniform_mux_model"),
        ("bernoulli_mux_distribution", "bernoulli_mux_model"),
    ],
)
@pytest.mark.cuda
def test_class_conditional_pcs_match_all_complete_assignments(
    request, cuda_device, distribution_fixture, model_fixture
):
    distribution = request.getfixturevalue(distribution_fixture)
    model = request.getfixturevalue(model_fixture)
    data = _complete_batch(cuda_device)
    missing_mask = torch.zeros_like(data, dtype=torch.bool)

    for class_id in (0, 1):
        actual = torch.exp(
            juice.queries.marginal(
                model.pc_per_class[class_id], data, missing_mask
            ).reshape(-1)
        ).cpu()
        expected = torch.tensor(
            [
                distribution.joint_probability(tuple(row), class_id)
                / distribution.class_probability(class_id)
                for row in data.cpu().tolist()
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        assert actual.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_uncompiled_pc_has_four_normalized_selector_branches(
    bernoulli_mux_distribution,
):
    for class_id in (0, 1):
        root = build_mux_pc_node(bernoulli_mux_distribution, class_id)
        assert root.num_chs == 4
        assert len(root.scope) == 8
        assert root.get_params().sum().item() == pytest.approx(1.0)
