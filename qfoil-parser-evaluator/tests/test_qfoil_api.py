"""Contract tests for the QFOIL public API."""

import subprocess
import sys

import pytest
import qfoil as qfoil_package

from qfoil import QFOIL, QFOILEvaluator, QFOILParser
from qfoil.backends import ProbabilisticBackend


class FakeProbabilityBackend:
    num_features = 2
    class_ids = {0, 1}

    def probability(self, instance, class_id=None):
        return 0.25 if class_id is None else 0.125

    def conditional(self, target, evidence, class_id):
        return 0.8 if class_id == 1 else 0.2


def test_root_package_exports_only_the_composed_public_api():
    assert qfoil_package.__all__ == [
        "QFOIL",
        "QFOILError",
        "QFOILEvaluator",
        "QFOILParser",
        "QFOILSyntaxError",
        "QFOILValidationError",
        "UnsupportedOperationError",
        "WitnessSearchResult",
        "Inconclusive",
        "INCONCLUSIVE",
    ]


def test_core_package_does_not_import_optional_backends():
    script = """
import sys
before = set(sys.modules)
import qfoil
loaded = set(sys.modules) - before
assert not ({'torch', 'pyjuice', 'z3'} & loaded)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_qfoil_parser_is_independent_from_evaluator():
    parser = QFOILParser(instance_len=4)

    assert parser.instance_len == 4
    assert not hasattr(parser, "evaluator")
    assert parser.parse("SIGMA x. FULL(x)").bound_var == "x"


def test_qfoil_facade_parses_and_evaluates_pure_logic():
    qfoil = QFOIL(instance_len=3)

    assert qfoil.evaluate("SIZE(x)", x=(1, None, 0)) == 2


def test_qfoil_facade_configures_runtime_explicitly():
    qfoil = QFOIL(
        instance_len=4,
        sigma_search_budget=0,
        prob_cache_size=7,
    )

    assert qfoil.parser.instance_len == 4
    assert qfoil.evaluator.instance_len == 4
    assert qfoil.evaluator.sigma.search_budget == 0
    assert qfoil.evaluator.probability_runtime.cache_size == 7


def test_qfoil_facade_rejects_unknown_constructor_options():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        QFOIL(batch_size=8)


@pytest.mark.parametrize("sigma_search_budget", [-1, 1.5, True])
def test_qfoil_facade_rejects_invalid_sigma_budget(sigma_search_budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        QFOIL(sigma_search_budget=sigma_search_budget)


def test_qfoil_facade_accepts_ast_and_explicit_bindings():
    qfoil = QFOIL()
    ast = qfoil.parse("x <<= y")

    assert qfoil.evaluate(
        ast,
        bindings={"x": (1, None), "y": (1, 0)},
    ) == 1


def test_qfoil_facade_exposes_witness_search():
    qfoil = QFOIL()

    witness = qfoil.grow_witness(
        "FULL(z) * (z <<= x)",
        var="z",
        context={"x": (1, 0, 1)},
    )

    assert witness == (1, 0, 1)


def test_probability_functions_use_backend_contract():
    backend = FakeProbabilityBackend()
    qfoil = QFOIL(probabilistic_backend=backend)

    assert isinstance(backend, ProbabilisticBackend)
    assert qfoil.evaluate("VAL(x)", x=(1, None)) == 0.25
    assert qfoil.evaluate("VAL(x, c)", x=(1, None), c=1) == 0.125
    assert qfoil.evaluate(
        "COND(x, y, c)",
        x=(1, None),
        y=(None, 0),
        c=1,
    ) == 0.8
    assert qfoil.evaluate("EP(x, c)", x=(1, None), c=1) == 0.8
    assert abs(qfoil.evaluate("CEP(x, c)", x=(1, None), c=1) - 0.2) < 1e-12


def test_sigma_domain_uses_configured_instance_len_with_empty_bindings():
    qfoil = QFOIL(instance_len=4)

    assert qfoil.evaluate("(SIGMA x. (SIZE(x) <= 1)) >= 1", bindings={}) == 1


def test_sigma_rejects_binding_for_quantified_variable():
    qfoil = QFOIL(instance_len=4)

    with pytest.raises(
        ValueError,
        match="bindings must not include variables bound by SIGMA: x",
    ):
        qfoil.evaluate(
            "(SIGMA x. (SIZE(x) <= 1)) >= 1",
            bindings={"x": (1, 0, 1, 0)},
        )


def test_sigma_with_probability_backend_needs_only_free_variable_bindings():
    backend = FakeProbabilityBackend()
    backend.num_features = 4
    qfoil = QFOIL(
        probabilistic_backend=backend,
        instance_len=4,
    )

    assert qfoil.evaluate(
        "(SIGMA x. (EP(x, c) >= 0.5)) >= 1",
        bindings={"c": 1},
    ) == 1


@pytest.mark.parametrize("aux", [(0,), (0, 1, None, 0, 1, None)])
def test_tuple_bindings_do_not_change_sigma_domain(aux):
    qfoil = QFOIL(instance_len=4, sigma_mode="exact")

    assert qfoil.evaluate(
        "(SIGMA x. (SIZE(x) == limit)) == 16",
        bindings={"limit": 4, "aux": aux},
    ) == 1


def test_function_registry_handles_native_and_defined_functions():
    qfoil = QFOIL()
    registry = qfoil.evaluator.function_runtime.registry

    assert registry.has_native("size")
    assert registry.has_defined("l_size")

    qfoil.define_function("HAS_ONE", ["x"], "SIZE(x) == 1")
    assert registry.has_defined("has_one")
    assert qfoil.evaluate("HAS_ONE(x)", x=(1, None)) == 1

    qfoil.define_function("HAS_ONE", ["x"], "SIZE(x) <= 1")
    assert qfoil.evaluate("HAS_ONE(x)", x=(None, None)) == 1
