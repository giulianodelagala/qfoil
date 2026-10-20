"""Tests for the backend-independent QFOIL parser."""

import pytest

from qfoil import QFOILParser
from qfoil.language import (
    Variable,
    NumberLiteral,
    BinaryOp,
    UnaryOp,
    Sigma,
    FunctionCall,
    TupleLiteral,
    TokenType,
)


class TestQFOILParser:
    """Test parser functionality without a probabilistic backend."""

    def test_parser_initialization(self):
        """Test that parser can be initialized without a PyJuice model."""
        parser = QFOILParser(instance_len=3)
        assert parser.instance_len == 3
        assert parser.pos == 0
        assert parser.tokens == []

    @pytest.mark.parametrize("instance_len", [-1, 1.5, True])
    def test_parser_rejects_invalid_instance_length(self, instance_len):
        with pytest.raises(ValueError, match="non-negative integer"):
            QFOILParser(instance_len=instance_len)

    def test_tokenize_simple_variable(self):
        """Test tokenization of simple variable."""
        parser = QFOILParser()
        tokens = parser.tokenize("x")
        assert len(tokens) == 2  # "x" + EOF
        assert tokens[0].value == "x"

    def test_tokenize_number(self):
        """Test tokenization of numbers."""
        parser = QFOILParser()
        tokens = parser.tokenize("42")
        assert tokens[0].value == "42"

    def test_tokenize_float(self):
        """Test tokenization of floats."""
        parser = QFOILParser()
        tokens = parser.tokenize("3.14")
        assert tokens[0].value == "3.14"

    def test_tokenize_operators(self):
        """Test tokenization of operators."""
        parser = QFOILParser()
        tokens = parser.tokenize("x + y * z")
        values = [t.value for t in tokens[:-1]]  # Exclude EOF
        assert "x" in values
        assert "+" in values
        assert "*" in values
        assert "y" in values
        assert "z" in values

    def test_tokenize_sigma(self):
        """Test tokenization of SIGMA operator."""
        parser = QFOILParser()
        tokens = parser.tokenize("SIGMA x. x + y")
        values = [t.value for t in tokens[:-1]]  # Exclude EOF
        assert "SIGMA" in values
        assert "x" in values
        assert "." in values

    def test_tokenize_comparison_operators(self):
        """Test tokenization of comparison operators."""
        parser = QFOILParser()
        tokens = parser.tokenize("x == y")
        values = [t.value for t in tokens[:-1]]
        assert "==" in values

    def test_parse_variable(self):
        """Test parsing a simple variable."""
        parser = QFOILParser()
        ast = parser.parse("x")
        assert isinstance(ast, Variable)
        assert ast.name == "x"

    def test_parse_number(self):
        """Test parsing a number literal."""
        parser = QFOILParser()
        ast = parser.parse("42")
        assert isinstance(ast, NumberLiteral)
        assert ast.value == 42.0

    def test_parse_binary_op_addition(self):
        """Test parsing binary addition."""
        parser = QFOILParser()
        ast = parser.parse("x + y")
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "+"
        assert isinstance(ast.left, Variable)
        assert isinstance(ast.right, Variable)

    def test_parse_binary_op_multiplication(self):
        """Test parsing binary multiplication."""
        parser = QFOILParser()
        ast = parser.parse("x * y")
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "*"

    def test_parse_binary_op_comparison(self):
        """Test parsing binary comparison."""
        parser = QFOILParser()
        ast = parser.parse("x == y")
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "=="

    def test_parse_unary_not(self):
        """Test parsing unary NOT."""
        parser = QFOILParser()
        ast = parser.parse("!x")
        assert isinstance(ast, UnaryOp)
        assert ast.operator == "!"
        assert isinstance(ast.operand, Variable)

    def test_parse_sigma_expression(self):
        """Test parsing SIGMA expression."""
        parser = QFOILParser()
        ast = parser.parse("SIGMA x. x + y")
        assert isinstance(ast, Sigma)
        assert ast.bound_var == "x"
        assert isinstance(ast.formula, BinaryOp)
        assert ast.formula.operator == "+"

    @pytest.mark.parametrize("keyword", ["SIGMA", "sigma", "∑"])
    @pytest.mark.parametrize(
        "formula",
        [
            "x + y",
            "(x <<= y)",
            "!FULL(x)",
            "0.5",
            "SIGMA y. (x == y)",
            "SIGMA y (x == y)",
        ],
    )
    def test_sigma_optional_dot_preserves_ast(self, keyword, formula):
        parser = QFOILParser()
        assert parser.parse(f"{keyword} x {formula}") == parser.parse(
            f"{keyword} x. {formula}"
        )

    @pytest.mark.parametrize("expression", ["SIGMA x", "SIGMA x.", "SIGMA (x)"])
    def test_sigma_requires_bound_variable_and_body(self, expression):
        with pytest.raises(ValueError):
            QFOILParser().parse(expression)

    def test_parse_nested_sigma(self):
        """Test parsing nested SIGMA expressions."""
        parser = QFOILParser()
        ast = parser.parse("SIGMA x. SIGMA y. x * y")
        assert isinstance(ast, Sigma)
        assert ast.bound_var == "x"
        assert isinstance(ast.formula, Sigma)
        assert ast.formula.bound_var == "y"

    def test_parse_function_call(self):
        """Test parsing function calls."""
        parser = QFOILParser()
        ast = parser.parse("BOTTOM(x)")
        assert isinstance(ast, FunctionCall)
        assert ast.function_name == "BOTTOM"
        assert len(ast.arguments) == 1

    def test_parse_function_call_multiple_args(self):
        """Test parsing function calls with multiple arguments."""
        parser = QFOILParser()
        ast = parser.parse("MEET(x, y, z)")
        assert isinstance(ast, FunctionCall)
        assert ast.function_name == "MEET"
        assert len(ast.arguments) == 3

    def test_parse_parenthesized_expression(self):
        """Test parsing parenthesized expressions."""
        parser = QFOILParser()
        ast = parser.parse("(x + y) * z")
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "*"
        # Left side should be the addition
        assert isinstance(ast.left, BinaryOp)
        assert ast.left.operator == "+"

    def test_parse_tuple_literal(self):
        """Test parsing tuple literals."""
        parser = QFOILParser()
        ast = parser.parse("(x, y, z)")
        assert isinstance(ast, TupleLiteral)
        assert len(ast.elements) == 3

    def test_parse_empty_tuple(self):
        """Test parsing empty tuple."""
        parser = QFOILParser()
        ast = parser.parse("()")
        assert isinstance(ast, TupleLiteral)
        assert len(ast.elements) == 0

    def test_parse_complex_expression(self):
        """Test parsing complex nested expression."""
        parser = QFOILParser()
        expr = "SIGMA x. (MEET(x, y) + BOTTOM(z)) * !w"
        ast = parser.parse(expr)
        assert isinstance(ast, Sigma)

    def test_parse_subseteq_operator(self):
        """Test parsing the non-strict SUBSETEQ operator (<<=)."""
        parser = QFOILParser()
        ast = parser.parse("x <<= y")
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "<<="

    def test_parse_subset_operator(self):
        """Test parsing the strict SUBSET operator (<<)."""
        parser = QFOILParser()
        ast = parser.parse("x << y")
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "<<"

    def test_parse_leq_operator(self):
        """Test parsing LEQ operator."""
        parser = QFOILParser()
        ast = parser.parse("x <= y")
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "<="

    def test_parse_geq_operator(self):
        """Test parsing GEQ operator."""
        parser = QFOILParser()
        ast = parser.parse("x >= y")
        assert isinstance(ast, BinaryOp)
        assert ast.operator == ">="

    def test_tokenize_invalid_character(self):
        """Test that tokenizer rejects invalid characters."""
        parser = QFOILParser()
        with pytest.raises(ValueError):
            parser.tokenize("x @ y")

    @pytest.mark.parametrize(
        "identifier",
        ["NOTHING", "SIGMA_value", "SUBSET_candidate", "SUBSETEQ2", "DOT_matrix"],
    )
    def test_reserved_word_prefixes_remain_identifiers(self, identifier):
        parser = QFOILParser()

        token = parser.tokenize(identifier)[0]
        assert token.type == TokenType.IDENTIFIER
        assert token.value == identifier
        assert parser.parse(identifier) == Variable(identifier)

    @pytest.mark.parametrize("expression", ["x y", "x)", "x, y", "x 2"])
    def test_parse_rejects_trailing_tokens(self, expression):
        with pytest.raises(ValueError, match="Unexpected token at position"):
            QFOILParser().parse(expression)

    def test_parse_precedence_multiplication_over_addition(self):
        """Test operator precedence: * binds tighter than +."""
        parser = QFOILParser()
        ast = parser.parse("x + y * z")
        # Should parse as: x + (y * z)
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "+"
        assert isinstance(ast.right, BinaryOp)
        assert ast.right.operator == "*"

    def test_parse_precedence_not_over_and_product(self):
        """Test operator precedence: ! binds tighter than *."""
        parser = QFOILParser()
        ast = parser.parse("!x * y")
        # Should parse as: (!x) * y
        assert isinstance(ast, BinaryOp)
        assert ast.operator == "*"
        assert isinstance(ast.left, UnaryOp)
        assert ast.left.operator == "!"

    def test_parse_none_literal(self):
        """Test parsing None literal."""
        parser = QFOILParser()
        ast = parser.parse("None")
        assert isinstance(ast, NumberLiteral)
        assert ast.value is None

    def test_convenience_function_parse_and_evaluate(self):
        """The canonical parser can be used without constructing a runtime."""
        parser = QFOILParser()
        expr = "x + y"
        ast = parser.parse(expr)
        assert ast is not None


class TestParserTokenRegexPatterns:
    """Test individual regex patterns used in tokenizer."""

    def test_sigma_aliases(self):
        """Test all SIGMA aliases: SIGMA, sigma, ∑."""
        parser = QFOILParser()

        # Test SIGMA
        tokens = parser.tokenize("SIGMA x. x")
        assert tokens[0].value == "SIGMA"

        # Test sigma
        tokens = parser.tokenize("sigma x. x")
        assert tokens[0].value == "sigma"

        # Test ∑ (mathematical symbol)
        tokens = parser.tokenize("∑ x. x")
        assert tokens[0].value == "∑"

    def test_dot_aliases(self):
        """Test DOT aliases: DOT, dot, ."""
        parser = QFOILParser()

        # Test dot
        tokens = parser.tokenize("SIGMA x. x")
        dot_token = [t for t in tokens if t.value == "."][0]
        assert dot_token.value == "."

    def test_not_aliases(self):
        """Test NOT aliases: NOT, not, !."""
        parser = QFOILParser()

        # Test !
        tokens = parser.tokenize("!x")
        assert tokens[0].value == "!"

        # Test NOT
        tokens = parser.tokenize("NOT x")
        assert tokens[0].value == "NOT"

        # Test not
        tokens = parser.tokenize("not x")
        assert tokens[0].value == "not"

    def test_subseteq_aliases(self):
        """Test SUBSETEQ aliases: SUBSETEQ, subseteq, <<=."""
        parser = QFOILParser()

        for text in ("x <<= y", "x SUBSETEQ y", "x subseteq y"):
            tokens = parser.tokenize(text)
            middle_token = tokens[1]
            assert middle_token.type == TokenType.SUBSETEQ

    def test_subset_aliases(self):
        """Test SUBSET aliases: SUBSET, subset, <<."""
        parser = QFOILParser()

        for text in ("x << y", "x SUBSET y", "x subset y"):
            tokens = parser.tokenize(text)
            middle_token = tokens[1]
            assert middle_token.type == TokenType.SUBSET

    @pytest.mark.parametrize("expression", ["x<<=y", "x <<= y"])
    def test_subseteq_symbol(self, expression):
        ast = QFOILParser().parse(expression)
        assert ast.operator == "<<="

    def test_removed_subseteq_symbol_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid character"):
            QFOILParser().parse("x <<< y")

    def test_subset_and_subseteq_are_distinct_tokens(self):
        """<< and <<= must tokenize differently, even back to back."""
        parser = QFOILParser()

        tokens = parser.tokenize("x << y <<= z")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.IDENTIFIER,
            TokenType.SUBSET,
            TokenType.IDENTIFIER,
            TokenType.SUBSETEQ,
            TokenType.IDENTIFIER,
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
