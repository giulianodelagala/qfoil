"""Unit tests for backend-neutral probabilistic runtime functions."""

import pytest

from qfoil.runtime import ProbabilityRuntime


class CountingBackend:
    num_features = 2
    class_ids = {0, 1}

    def __init__(self):
        self.probability_calls = 0
        self.conditional_calls = 0

    def probability(self, instance, class_id=None):
        self.probability_calls += 1
        return 0.4 if class_id is None else 0.2

    def conditional(self, target, evidence, class_id):
        self.conditional_calls += 1
        return 0.75 if class_id == 1 else 0.25


@pytest.mark.parametrize("cache_size", [-1, 1.5, True])
def test_probability_runtime_rejects_invalid_cache_size(cache_size):
    with pytest.raises(ValueError, match="non-negative integer"):
        ProbabilityRuntime(None, cache_size=cache_size)


def test_probability_runtime_requires_a_backend_only_when_queried():
    runtime = ProbabilityRuntime(None)

    with pytest.raises(RuntimeError, match="requires a probabilistic backend"):
        runtime.val([(1, None)])


def test_val_is_cached_by_arguments():
    backend = CountingBackend()
    runtime = ProbabilityRuntime(backend)

    assert runtime.val([(1, None)]) == 0.4
    assert runtime.val([(1, None)]) == 0.4
    assert backend.probability_calls == 1
    assert runtime.cache_misses == 1
    assert runtime.cache_hits == 1


def test_ep_and_cep_share_the_conditional_cache():
    backend = CountingBackend()
    runtime = ProbabilityRuntime(backend)

    assert runtime.ep([(1, None), 1]) == 0.75
    assert runtime.cep([(1, None), 1]) == 0.25
    assert backend.conditional_calls == 1
    assert runtime.cache_hits == 1


def test_lru_cache_respects_its_capacity():
    backend = CountingBackend()
    runtime = ProbabilityRuntime(backend, cache_size=1)

    runtime.val([(1, None)])
    runtime.val([(None, 1)])

    assert len(runtime.cache) == 1
    assert ("VAL", (None, 1), None) in runtime.cache


@pytest.mark.parametrize("class_id", [True, 1.5, "1", 3])
def test_class_id_validation(class_id):
    runtime = ProbabilityRuntime(CountingBackend())

    with pytest.raises(ValueError):
        runtime.val([(1, None), class_id])
