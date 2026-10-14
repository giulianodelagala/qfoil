"""Render the canonical QFOIL AST as KaTeX-compatible LaTeX."""

from __future__ import annotations

import re
from typing import Final

from ..errors import QFOILValidationError, UnsupportedOperationError
from .ast import (
    ASTNode,
    BinaryOp,
    FunctionCall,
    NumberLiteral,
    Sigma,
    TupleLiteral,
    UnaryOp,
    Variable,
)


class QFOILLatexRenderer:
    """Pure renderer for parsed QFOIL expressions."""

    _IDENTIFIER: Final = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
    _BINARY_OPERATORS: Final = {
        "==": ("=", 10),
        "<=": (r"\leq", 20),
        ">=": (r"\geq", 30),
        "<<": (r"\subset", 40),
        "SUBSET": (r"\subset", 40),
        "<<<": (r"\subseteq", 40),
        "SUBSETEQ": (r"\subseteq", 40),
        "+": ("+", 50),
        "*": (r"\cdot", 60),
    }

    def render(self, ast: ASTNode) -> str:
        """Render one AST without math delimiters or HTML."""
        if not isinstance(ast, ASTNode):
            raise QFOILValidationError("to_latex expects a QFOIL AST")
        return self._render(ast, parent_precedence=0)

    def _render(self, node: ASTNode, parent_precedence: int) -> str:
        if isinstance(node, Variable):
            return self._render_identifier(node.name)

        if isinstance(node, NumberLiteral):
            return self._render_number(node.value)

        if isinstance(node, TupleLiteral):
            elements = ", ".join(self._render(item, 0) for item in node.elements)
            if len(node.elements) == 1:
                elements += ","
            return rf"\left({elements}\right)"

        if isinstance(node, FunctionCall):
            name = self._escape_identifier(node.function_name)
            arguments = ", ".join(
                self._render(argument, 0) for argument in node.arguments
            )
            return rf"\operatorname{{{name}}}\left({arguments}\right)"

        if isinstance(node, UnaryOp):
            if node.operator.upper() not in ("NOT", "!"):
                raise UnsupportedOperationError(
                    f"unsupported unary operator: {node.operator}"
                )
            rendered = rf"\neg {self._render(node.operand, 70)}"
            return self._parenthesize(rendered, 70, parent_precedence)

        if isinstance(node, BinaryOp):
            normalized = node.operator.upper()
            try:
                operator, precedence = self._BINARY_OPERATORS[normalized]
            except KeyError as exc:
                raise UnsupportedOperationError(
                    f"unsupported binary operator: {node.operator}"
                ) from exc
            left = self._render(node.left, precedence)
            right_precedence = (
                precedence if normalized in ("+", "*") else precedence + 1
            )
            right = self._render(node.right, right_precedence)
            rendered = f"{left} {operator} {right}"
            return self._parenthesize(rendered, precedence, parent_precedence)

        if isinstance(node, Sigma):
            variable = self._render_identifier(node.bound_var)
            formula = self._render(node.formula, 0)
            rendered = rf"\Sigma {variable} \, \left({formula}\right)"
            return self._parenthesize(rendered, 5, parent_precedence)

        raise UnsupportedOperationError(f"unsupported AST node: {type(node).__name__}")

    @staticmethod
    def _parenthesize(text: str, precedence: int, parent_precedence: int) -> str:
        if precedence < parent_precedence:
            return rf"\left({text}\right)"
        return text

    @classmethod
    def _escape_identifier(cls, value: str) -> str:
        if cls._IDENTIFIER.fullmatch(value) is None:
            raise QFOILValidationError(f"invalid QFOIL identifier: {value!r}")
        return value.replace("_", r"\_")

    @classmethod
    def _render_identifier(cls, value: str) -> str:
        escaped = cls._escape_identifier(value)
        return escaped if len(value) == 1 else rf"\mathit{{{escaped}}}"

    @staticmethod
    def _render_number(value: object) -> str:
        if value is None:
            return r"\bot"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise QFOILValidationError(f"invalid QFOIL number literal: {value!r}")


__all__ = ["QFOILLatexRenderer"]
