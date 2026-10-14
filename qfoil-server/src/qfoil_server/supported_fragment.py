"""Reject recursive logic-function definitions before execution.

Nested SIGMA execution is supported. The service defaults to heuristic
mode, which never enumerates. Evaluation may explicitly select exact mode;
both procedures retain the operation's global work budget.
"""

from qfoil import QFOIL, UnsupportedOperationError
from qfoil.language.ast import BinaryOp, FunctionCall, Sigma, TupleLiteral, UnaryOp
from qfoil.semantics.standard_library import STANDARD_DEFINITIONS


def check_supported_fragment(expression: str, runtime: QFOIL) -> None:
    """Raise ``UnsupportedOperationError`` if ``expression`` calls a logic
    function recursively, directly or through another function's body.
    """
    parser = QFOIL()
    ast = parser.parse(expression)
    definitions = {item.name: item.body for item in STANDARD_DEFINITIONS}
    functions = getattr(runtime.evaluator, "function_runtime", None)
    if functions is not None:
        registry = functions.registry
        definitions = {
            name: registry.defined_function(name).body
            for name in registry.defined_names
        }

    def visit(node, stack=()):
        if isinstance(node, Sigma):
            visit(node.formula, stack)
        elif isinstance(node, BinaryOp):
            visit(node.left, stack)
            visit(node.right, stack)
        elif isinstance(node, UnaryOp):
            visit(node.operand, stack)
        elif isinstance(node, TupleLiteral):
            for element in node.elements:
                visit(element, stack)
        elif isinstance(node, FunctionCall):
            for argument in node.arguments:
                visit(argument, stack)
            name = node.function_name.upper()
            if name in definitions:
                if name in stack:
                    raise UnsupportedOperationError(
                        "Recursive logic functions are outside the supported fragment"
                    )
                visit(parser.parse(definitions[name]), (*stack, name))

    visit(ast)
