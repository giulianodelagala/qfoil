"""Canonical abstract syntax tree for the QFOIL language."""

from dataclasses import dataclass
from enum import Enum
from typing import List


class TokenType(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PLUS = "+"
    PROD = "*"
    SUBSET = "SUBSET"
    SUBSETEQ = "SUBSETEQ"
    EQUAL = "=="
    LEQ = "<="
    GEQ = ">="
    LPAREN = "("
    RPAREN = ")"
    IDENTIFIER = "ID"
    EOF = "EOF"
    NUMBER = "NUMBER"
    SIGMA = "SIGMA"
    DOT = "."
    COMMA = ","
    BOTTOM = "BOTTOM"


@dataclass
class Token:
    type: TokenType
    value: str
    position: int = 0


@dataclass
class ASTNode:
    pass


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    operator: str
    right: ASTNode


@dataclass
class UnaryOp(ASTNode):
    operator: str
    operand: ASTNode


@dataclass
class Variable(ASTNode):
    name: str


@dataclass
class NumberLiteral(ASTNode):
    value: float


@dataclass
class TupleLiteral(ASTNode):
    elements: List[ASTNode]


@dataclass
class Sigma(ASTNode):
    """Count instances satisfying ``formula`` while binding one variable."""

    bound_var: str
    formula: ASTNode


@dataclass
class FunctionCall(ASTNode):
    """Call a native or QFOIL-defined function."""

    function_name: str
    arguments: List[ASTNode]


__all__ = [
    "ASTNode",
    "BinaryOp",
    "FunctionCall",
    "NumberLiteral",
    "Sigma",
    "Token",
    "TokenType",
    "TupleLiteral",
    "UnaryOp",
    "Variable",
]
