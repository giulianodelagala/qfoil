"""Integration tests for the backend-independent QFOIL language runtime."""

import pytest

from qfoil import QFOIL


def parse_and_evaluate(expression, context):
    """Evaluate through the canonical high-level API."""
    return QFOIL().evaluate(expression, bindings=context)


class TestBasicArithmetic:
    """Test basic arithmetic operations."""

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("A + B", {"A": 1, "B": 1}, 2),
            ("A + B", {"A": 1, "B": 0}, 1),
            ("A + B", {"A": 5, "B": 3}, 8),
            ("A * B", {"A": 3, "B": 4}, 12),
            ("A * B", {"A": 0, "B": 5}, 0),
            ("A * 0.5", {"A": 2}, 1.0),
        ],
    )
    def test_arithmetic_operations(self, expression, context, expected):
        """Test basic arithmetic operations work correctly."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    @pytest.mark.parametrize(
        "expression,context,error_message",
        [
            ("A + B", {"A": True, "B": 2}, r"Invalid operation: True \+ 2"),
            ("A * B", {"A": True, "B": 2}, r"Invalid operation: True \* 2"),
        ],
    )
    def test_arithmetic_type_errors(self, expression, context, error_message):
        """Test that invalid arithmetic operations raise proper errors."""
        with pytest.raises(ValueError, match=error_message):
            parse_and_evaluate(expression, context)


class TestComparisonOperators:
    """Test comparison operators."""

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("A <= B", {"A": 1, "B": 2}, 1),
            ("A <= B", {"A": 2, "B": 2}, 1),
            ("A <= B", {"A": 3, "B": 2}, 0),
            ("A >= B", {"A": 2, "B": 1}, 1),
            ("A >= B", {"A": 2, "B": 2}, 1),
            ("A >= B", {"A": 1, "B": 2}, 0),
        ],
    )
    def test_comparison_operations(self, expression, context, expected):
        """Test comparison operations work correctly."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    @pytest.mark.parametrize(
        "expression,context,error_message",
        [
            ("A <= B", {"A": (1, 2), "B": 2}, "Invalid operation: \\(1, 2\\) <= 2"),
            ("A >= B", {"A": (1, 2), "B": 2}, "Invalid operation: \\(1, 2\\) >= 2"),
        ],
    )
    def test_comparison_type_errors(self, expression, context, error_message):
        """Test that invalid comparison operations raise proper errors."""
        with pytest.raises(ValueError, match=error_message):
            parse_and_evaluate(expression, context)


class TestTupleOperations:
    """Test tuple-specific operations."""

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("A == B", {"A": (1, 2), "B": (1, 2)}, 1),
            ("A == B", {"A": (1, 2), "B": (1, 2, 3)}, 0),
            ("A == B", {"A": (1, 0, 1), "B": (1, 0, 1)}, 1),
        ],
    )
    def test_tuple_equality(self, expression, context, expected):
        """Test tuple equality operations."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    def test_tuple_equality_type_error(self):
        """Test that EQUAL operation requires same type values."""
        with pytest.raises(
            ValueError, match="EQUAL operation requires same type values"
        ):
            parse_and_evaluate("A == B", {"A": True, "B": (1, 2, 3)})

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("A <<< B", {"A": (1, 1), "B": (1, 1)}, 1),  # equal tuples: reflexive
            ("A <<< B", {"A": (1, None), "B": (1, 0)}, 1),
            ("A <<< B", {"A": (1, 0), "B": (1, 1)}, 0),
            ("A <<< B", {"A": (None, None), "B": (1, 0)}, 1),
            ("A <<< B", {"A": (None, 1), "B": (1, 0)}, 0),
        ],
    )
    def test_subseteq_operations(self, expression, context, expected):
        """Test SUBSETEQ (<<<) operations: non-strict, reflexive refinement."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    def test_subseteq_length_error(self):
        """Test that SUBSETEQ operation requires equal-length tuples."""
        with pytest.raises(
            ValueError, match="SUBSETEQ operation requires equal-length tuples"
        ):
            parse_and_evaluate("A <<< B", {"A": (1, 1), "B": (1, 1, 0)})

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            # equal tuples: SUBSETEQ holds but SUBSET is strict -> 0
            ("A << B", {"A": (1, 1), "B": (1, 1)}, 0),
            ("A << B", {"A": (1, None), "B": (1, 0)}, 1),
            ("A << B", {"A": (1, 0), "B": (1, 1)}, 0),
            ("A << B", {"A": (None, None), "B": (1, 0)}, 1),
            ("A << B", {"A": (None, 1), "B": (1, 0)}, 0),
        ],
    )
    def test_subset_operations(self, expression, context, expected):
        """Test SUBSET (<<) operations: strict, proper refinement."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    def test_subset_length_error(self):
        """Test that SUBSET operation requires equal-length tuples."""
        with pytest.raises(
            ValueError, match="SUBSETEQ operation requires equal-length tuples"
        ):
            parse_and_evaluate("A << B", {"A": (1, 1), "B": (1, 1, 0)})


class TestComplexExpressions:
    """Test complex logical expressions."""

    def test_negation_complex(self):
        """Test negation of complex expressions."""
        expr = "NOT (A == B)"
        context = {"A": (1, 2), "B": (1, 2)}
        result = parse_and_evaluate(expr, context)
        assert result == 0

    def test_arithmetic_with_negation(self):
        """Test arithmetic combined with logical operations."""
        expr = "A * (0.5 + NOT (C == B))"
        context = {"A": 1, "B": (1, 1), "C": (1, 1)}
        result = parse_and_evaluate(expr, context)
        assert result == 0.5

    def test_invalid_arithmetic_combination(self):
        """Test invalid arithmetic combinations raise errors."""
        expr = "A * (B + NOT (C == B))"
        context = {"A": 1, "B": (1, 1), "C": (1, 1)}
        with pytest.raises(ValueError, match="Invalid operation: \\(1, 1\\) \\+ 0"):
            parse_and_evaluate(expr, context)


class TestRuntimeLogicFunctionDefinitions:
    def test_define_and_call_parameterized_function(self):
        parser = QFOIL()
        parser.define_function(
            "IS_SINGLE",
            parameters=["candidate"],
            body="SIZE(candidate) == 1",
        )

        assert parser.evaluator.evaluate(
            parser.parse("IS_SINGLE(x)"), {"x": (1, None, None)}
        ) == 1
        assert parser.evaluator.evaluate(
            parser.parse("IS_SINGLE(x)"), {"x": (1, 0, None)}
        ) == 0

    def test_redefinition_replaces_the_cached_body(self):
        parser = QFOIL()
        parser.define_function("SELECTED", ["x"], "SIZE(x) == 1")
        expression = parser.parse("SELECTED(x)")
        context = {"x": (1, None, None)}
        assert parser.evaluator.evaluate(expression, context) == 1

        parser.define_function("SELECTED", ["x"], "SIZE(x) == 2")
        assert parser.evaluator.evaluate(expression, context) == 0

    @pytest.mark.parametrize(
        "name,params,body,error",
        [
            ("FULL", ["x"], "SIZE(x) == 1", "native function"),
            ("1BAD", ["x"], "SIZE(x) == 1", "function name"),
            ("DUPLICATE", ["x", "x"], "SIZE(x) == 1", "unique"),
            ("BAD_PARAM", ["not-valid"], "SIZE(x) == 1", "identifiers"),
            ("EMPTY", ["x"], "", "non-empty"),
        ],
    )
    def test_invalid_definitions_are_rejected(self, name, params, body, error):
        parser = QFOIL()
        with pytest.raises(ValueError, match=error):
            parser.define_function(name, params, body)


