"""Unit tests for backend-independent QFOIL binary operators."""

import pytest

from qfoil.semantics import OperatorSemantics, TupleSemantics


@pytest.fixture
def operators():
    return OperatorSemantics(TupleSemantics())


@pytest.mark.parametrize(
    ("operator", "left", "right", "expected"),
    [
        ("==", 1, 1.0, 1),
        ("<=", 1, 2, 1),
        (">=", 1, 2, 0),
        ("+", 1, 2.5, 3.5),
        ("*", 3, 2, 6),
        ("<<=", (1, None), (1, 0), 1),
        ("SUBSET", (1, None), (1, 0), 1),
        ("<<", (1, 0), (1, 0), 0),
    ],
)
def test_binary_operators(operators, operator, left, right, expected):
    assert operators.evaluate(operator, left, right) == expected


def test_equality_rejects_unrelated_types(operators):
    with pytest.raises(ValueError, match="same type"):
        operators.evaluate("==", (1,), 1)


@pytest.mark.parametrize("operator", ["+", "*", "<=", ">="])
def test_numeric_operators_reject_non_numeric_values(operators, operator):
    with pytest.raises(ValueError, match="Invalid operation"):
        operators.evaluate(operator, (1,), 1)


def test_arithmetic_rejects_booleans(operators):
    with pytest.raises(ValueError, match="Invalid operation"):
        operators.evaluate("+", True, 1)


def test_unknown_operator_is_explicit(operators):
    with pytest.raises(ValueError, match="Unknown binary operator"):
        operators.evaluate("/", 4, 2)
