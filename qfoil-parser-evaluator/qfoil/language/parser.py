import re
from typing import List

from .ast import (
    ASTNode,
    Variable,
    NumberLiteral,
    BinaryOp,
    UnaryOp,
    TokenType,
    Token,
    Sigma,
    FunctionCall,
    TupleLiteral,
)


class QFOILParser:
    """Parse QFOIL expressions without constructing an execution runtime.

    Parsing is intentionally independent from evaluation.  This keeps the
    language frontend usable without PyJuice, Torch, or any other execution
    backend.
    """

    _TOKEN_PATTERNS = tuple(
        (re.compile(pattern), token_type)
        for pattern, token_type in (
            (r"\s+", None),
            (r"(?:SIGMA|sigma)(?![A-Za-z0-9_])|∑", TokenType.SIGMA),
            (r"(?:DOT|dot)(?![A-Za-z0-9_])|\.", TokenType.DOT),
            (r"(?:NOT|not)(?![A-Za-z0-9_])|!", TokenType.NOT),
            (r"(?:SUBSETEQ|subseteq)(?![A-Za-z0-9_])|<<<", TokenType.SUBSETEQ),
            (r"(?:SUBSET|subset)(?![A-Za-z0-9_])|<<", TokenType.SUBSET),
            (r"==", TokenType.EQUAL),
            (r"<=", TokenType.LEQ),
            (r">=", TokenType.GEQ),
            (r"\(", TokenType.LPAREN),
            (r"\)", TokenType.RPAREN),
            (r",", TokenType.COMMA),
            (r"[A-Za-z][A-Za-z0-9_]*", TokenType.IDENTIFIER),
            (r"\+", TokenType.PLUS),
            (r"\*", TokenType.PROD),
            (r"\d+(?:\.\d+)?", TokenType.NUMBER),
        )
    )

    def __init__(self, instance_len: int = 3) -> None:
        if (
            isinstance(instance_len, bool)
            or not isinstance(instance_len, int)
            or instance_len < 0
        ):
            raise ValueError("instance_len must be a non-negative integer")
        self.tokens: List[Token] = []
        self.pos = 0
        self.instance_len = instance_len

    def tokenize(self, text: str) -> List[Token]:
        """Convert text to tokens without splitting reserved-word prefixes."""
        tokens: List[Token] = []

        pos = 0
        while pos < len(text):
            matched = False
            for regex, token_type in self._TOKEN_PATTERNS:
                match = regex.match(text, pos)
                if match:
                    value = match.group(0)
                    if token_type is not None:
                        tokens.append(Token(token_type, value, pos))
                    pos = match.end()
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Invalid character at position {pos}: {text[pos]}")

        tokens.append(Token(TokenType.EOF, "", pos))
        return tokens

    def parse(self, text: str) -> ASTNode:
        """Parse exactly one complete expression into an AST."""
        self.tokens = self.tokenize(text)
        self.pos = 0
        expression = self.parse_equal()
        if self.current_token().type != TokenType.EOF:
            token = self.current_token()
            raise ValueError(
                f"Unexpected token at position {token.position}: {token.value!r}"
            )
        return expression

    def current_token(self) -> Token:
        return (
            self.tokens[self.pos]
            if self.pos < len(self.tokens)
            else Token(TokenType.EOF, "", 0)
        )

    def consume(self, expected_type: TokenType | None = None) -> Token:
        token = self.current_token()
        if expected_type and token.type != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token.type}")
        self.pos += 1
        return token

    def parse_equal(self) -> ASTNode:
        """Parse EQUAL expressions (lowest precedence)"""
        left = self.parse_leq()

        while self.current_token().type == TokenType.EQUAL:
            op = self.consume().value
            right = self.parse_leq()
            left = BinaryOp(left, op, right)

        return left

    def parse_leq(self) -> ASTNode:
        """Parse LEQ expressions"""
        left = self.parse_geq()

        while self.current_token().type == TokenType.LEQ:
            op = self.consume().value
            right = self.parse_geq()
            left = BinaryOp(left, op, right)

        return left

    def parse_geq(self) -> ASTNode:
        """Parse GEQ expressions"""
        left = self.parse_subset()

        while self.current_token().type == TokenType.GEQ:
            op = self.consume().value
            right = self.parse_subset()
            left = BinaryOp(left, op, right)

        return left

    def parse_subset(self) -> ASTNode:
        """Parse SUBSET (<<, strict) and SUBSETEQ (<<<, non-strict) expressions"""
        left = self.parse_plus()

        while self.current_token().type in (TokenType.SUBSET, TokenType.SUBSETEQ):
            op = self.consume().value
            right = self.parse_plus()
            left = BinaryOp(left, op, right)

        return left

    def parse_plus(self) -> ASTNode:
        """Parse PLUS expressions"""
        left = self.parse_prod()

        while self.current_token().type == TokenType.PLUS:
            op = self.consume().value
            right = self.parse_prod()
            left = BinaryOp(left, op, right)

        return left

    def parse_prod(self) -> ASTNode:
        """Parse PROD expressions"""
        left = self.parse_not()

        while self.current_token().type == TokenType.PROD:
            op = self.consume().value
            right = self.parse_not()
            left = BinaryOp(left, op, right)

        return left

    def parse_not(self) -> ASTNode:
        """Parse NOT expressions"""
        if self.current_token().type == TokenType.NOT:
            op = self.consume().value
            operand = self.parse_not()  # Right associative
            return UnaryOp(op, operand)

        return self.parse_sigma()

    def parse_sigma(self) -> ASTNode:
        """Parse SIGMA expressions with an optional dot after the bound variable."""
        if self.current_token().type == TokenType.SIGMA:
            self.consume(TokenType.SIGMA)
            bound_var = self.consume(TokenType.IDENTIFIER).value
            if self.current_token().type == TokenType.DOT:
                self.consume(TokenType.DOT)
            # Parse the formula at subset level to avoid recursion
            formula = self.parse_plus()

            return Sigma(bound_var, formula)

        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        """Parse primary expressions (variables, numbers,
        parentheses, tuples)"""
        token = self.current_token()

        if token.type == TokenType.IDENTIFIER:
            # Check for special keywords
            if token.value == "None":
                self.consume()
                return NumberLiteral(None)
                # Represent None as special NumberLiteral

            # Check if this is a function call
            elif (
                self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1].type == TokenType.LPAREN
            ):
                return self.parse_function_call()
            else:
                # Regular variable
                self.consume()
                return Variable(token.value)

        elif token.type == TokenType.NUMBER:
            self.consume()
            return NumberLiteral(float(token.value))

        elif token.type == TokenType.LPAREN:
            # Check if this is a tuple literal or parenthesized expression
            self.consume(TokenType.LPAREN)

            # If we see a right paren immediately, it's empty tuple
            if self.current_token().type == TokenType.RPAREN:
                self.consume(TokenType.RPAREN)
                return TupleLiteral([])

            # Parse first element
            first_elem = self.parse_equal()

            # Check if we have a comma (tuple) or right paren (parentheses)
            if self.current_token().type == TokenType.COMMA:
                # This is a tuple literal
                elements = [first_elem]

                while self.current_token().type == TokenType.COMMA:
                    self.consume(TokenType.COMMA)
                    # Handle trailing comma case
                    if self.current_token().type == TokenType.RPAREN:
                        break
                    elements.append(self.parse_equal())

                self.consume(TokenType.RPAREN)
                return TupleLiteral(elements)

            elif self.current_token().type == TokenType.RPAREN:
                # This is a parenthesized expression
                self.consume(TokenType.RPAREN)
                return first_elem

            else:
                # fmt: off
                raise ValueError(
                    f"Expected ',' or ')' in tuple/parentheses, "
                    f"got {self.current_token().type}"
                    )
                # fmt: on

        else:
            raise ValueError(f"Unexpected token: {token}")

    def parse_function_call(self) -> FunctionCall:
        """Parse function calls: FUNC(arg1, arg2, ...)"""
        func_name = self.consume().value
        self.consume(TokenType.LPAREN)

        arguments = []
        if self.current_token().type != TokenType.RPAREN:
            arguments.append(self.parse_sigma())

            while self.current_token().type == TokenType.COMMA:
                self.consume(TokenType.COMMA)
                arguments.append(self.parse_sigma())

        self.consume(TokenType.RPAREN)
        return FunctionCall(func_name, arguments)
