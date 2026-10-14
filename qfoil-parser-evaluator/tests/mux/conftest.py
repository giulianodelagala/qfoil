import pytest
import torch

from examples.models.mux import MuxDistribution, build_mux_model
from qfoil import QFOIL
from qfoil.backends.pyjuice import PyJuiceProbabilityBackend


def _make_evaluator(model):
    qfoil = QFOIL(
        probabilistic_backend=PyJuiceProbabilityBackend(model),
        instance_len=8,
        sigma_search_budget=0,
        sigma_mode="exact",
    )
    return qfoil, qfoil.evaluator


@pytest.fixture(scope="session")
def uniform_mux_distribution():
    return MuxDistribution.uniform()


@pytest.fixture(scope="session")
def bernoulli_mux_distribution():
    return MuxDistribution.bernoulli()


@pytest.fixture(scope="session")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("PyJuice 2.3.x requires CUDA for compiled circuit inference")
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def uniform_mux_model(uniform_mux_distribution, cuda_device):
    return build_mux_model(uniform_mux_distribution, device=cuda_device)


@pytest.fixture(scope="session")
def bernoulli_mux_model(bernoulli_mux_distribution, cuda_device):
    return build_mux_model(bernoulli_mux_distribution, device=cuda_device)


@pytest.fixture(scope="session")
def uniform_mux_evaluator(uniform_mux_model):
    return _make_evaluator(uniform_mux_model)


@pytest.fixture(scope="session")
def bernoulli_mux_evaluator(bernoulli_mux_model):
    return _make_evaluator(bernoulli_mux_model)
