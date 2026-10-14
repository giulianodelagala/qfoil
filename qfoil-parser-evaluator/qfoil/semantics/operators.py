"""Semantics of QFOIL binary operators."""

from __future__ import annotations

from typing import Any

from .tuples import TupleSemantics


class OperatorSemantics:
    """Evaluate binary operations after their operands have been resolved."""

    def __init__(self, tuples: TupleSemantics) -> None:
        self.tuples = tuples

    def evaluate(self, operator: str, left: Any, right: Any) -> Any:
        normalized = operator.upper()
        if normalized in ("SUBSETEQ", "<<<"):
            return self.tuples.subseteq(left, right)
        if normalized in ("SUBSET", "<<"):
            return self.tuples.subset(left, right)
        if operator == "==":
            return self.equality(left, right)
        if operator in ("<=", ">="):
            return self.ordering(left, right, operator)
        if operator in ("+", "*"):
            return self.arithmetic(left, right, operator)
        raise ValueError(f"Unknown binary operator: {operator}")

    @staticmethod
    def equality(left: Any, right: Any) -> int:
        both_numeric = isinstance(left, (int, float)) and isinstance(
            right, (int, float)
        )
        if not both_numeric and not isinstance(left, type(right)):
            raise ValueError("EQUAL operation requires same type values")
        return int(left == right)

    @staticmethod
    def ordering(left: Any, right: Any, operator: str) -> int:
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError(f"Invalid operation: {left} {operator} {right}")
        return int(left <= right) if operator == "<=" else int(left >= right)

    @staticmethod
    def arithmetic(left: Any, right: Any, operator: str) -> Any:
        values = (left, right)
        if any(isinstance(value, bool) for value in values) or not all(
            isinstance(value, (int, float)) for value in values
        ):
            raise ValueError(f"Invalid operation: {left} {operator} {right}")
        return left + right if operator == "+" else left * right


__all__ = ["OperatorSemantics"]
