"""Unit tests for QFOIL function registration and execution."""

import pytest

from qfoil.language import QFOILParser
from qfoil.runtime import FunctionRuntime
from qfoil.semantics import DefinedFunction


def test_native_and_defined_functions_share_one_registry():
    runtime = FunctionRuntime(
        {"SIZE": lambda args: len(args[0]), "UNIMPLEMENTED": None},
        [DefinedFunction("IDENTITY", ("x",), "x")],
        parser=QFOILParser(),
    )

    assert runtime.evaluate_native("size", [(1, 2)]) == 2
    assert runtime.registry.has_defined("identity")
    with pytest.raises(NotImplementedError, match="UNIMPLEMENTED"):
        runtime.evaluate_native("unimplemented", [])


def test_defined_function_compiles_once_and_binds_local_context():
    class CountingParser:
        def __init__(self):
            self.calls = 0

        def parse(self, body):
            self.calls += 1
            return body

    parser = CountingParser()
    runtime = FunctionRuntime(
        {},
        [DefinedFunction("IDENTITY", ("x",), "x")],
        parser=parser,
    )

    def evaluate_body(body, context):
        return context[body]

    assert runtime.evaluate_defined("IDENTITY", [2], {}, evaluate_body) == 2
    assert runtime.evaluate_defined("IDENTITY", [3], {}, evaluate_body) == 3
    assert parser.calls == 1


def test_defining_a_function_compiles_eagerly_and_replaces_its_body():
    parser = QFOILParser()
    runtime = FunctionRuntime({}, parser=parser)

    runtime.define("IS_ONE", ["x"], "x == 1")
    first_ast = runtime.compiled_cache["IS_ONE"]
    runtime.define("IS_ONE", ["x"], "x <= 1")

    assert runtime.registry.defined_function("IS_ONE").body == "x <= 1"
    assert runtime.compiled_cache["IS_ONE"] is not first_ast


@pytest.mark.parametrize(
    ("name", "parameters", "body", "message"),
    [
        ("1BAD", ["x"], "x", "Invalid logic function name"),
        ("GOOD", "x", "x", "params must be a sequence"),
        ("GOOD", ["x", "x"], "x", "must be unique"),
        ("GOOD", ["bad-name"], "x", "valid identifiers"),
        ("GOOD", ["x"], "", "non-empty string"),
    ],
)
def test_definition_validation(name, parameters, body, message):
    runtime = FunctionRuntime({}, parser=QFOILParser())

    with pytest.raises(ValueError, match=message):
        runtime.define(name, parameters, body)


def test_native_functions_cannot_be_redefined():
    runtime = FunctionRuntime({"SIZE": lambda args: 0}, parser=QFOILParser())

    with pytest.raises(ValueError, match="Cannot redefine native function SIZE"):
        runtime.define("size", ["x"], "x")


def test_definition_requires_a_parser():
    runtime = FunctionRuntime({})

    with pytest.raises(RuntimeError, match="Parser not set"):
        runtime.define("IDENTITY", ["x"], "x")
