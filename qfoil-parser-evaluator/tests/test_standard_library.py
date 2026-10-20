"""Integration tests for QFOIL's native and declarative standard library."""

import pytest

from qfoil import QFOIL


def parse_and_evaluate(expression, context):
    """Evaluate through the canonical API with an explicitly sized runtime."""
    tuple_value = next(
        (value for value in context.values() if isinstance(value, tuple)), None
    )
    instance_len = len(tuple_value) if tuple_value is not None else 3
    return QFOIL(
        instance_len=instance_len, sigma_search_budget=0, sigma_mode="exact"
    ).evaluate(expression, bindings=context)


class TestQFOILFunctions:
    """Test logic parser defined functions (L_FULL, L_MEET, etc.)."""

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("L_FULL(x)", {"x": (1, 0, 1)}, 1),  # x is full
            ("L_FULL(x)", {"x": (1, 0, None)}, 0),  # x is not full
        ],
    )
    def test_l_full_function(self, expression, context, expected):
        """Test L_FULL logic function."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    @pytest.mark.parametrize(
        "context,expected",
        [
            # x refines to y and y is full -> y is a completion of x
            ({"x": (1, None, 1), "y": (1, 0, 1)}, 1),
            ({"x": (None, None, None), "y": (1, 0, 1)}, 1),
            ({"x": (1, 0, 1), "y": (1, 0, 1)}, 1),  # x already full and equal
            ({"x": (1, None, 1), "y": (1, 0, None)}, 0),  # y not full
            ({"x": (1, None, 1), "y": (0, 0, 1)}, 0),  # x doesn't refine to y
        ],
    )
    def test_l_comp_function(self, context, expected):
        """Test L_COMP logic function: (x <<= y) * L_FULL(y)."""
        result = parse_and_evaluate("L_COMP(x, y)", context)
        assert result == expected

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("L_FEAT(x)", {"x": (1, None, None)}, 1),  # exactly one feature
            ("L_FEAT(x)", {"x": (None, None, None)}, 0),  # no features
            ("L_FEAT(x)", {"x": (1, 0, None)}, 0),  # two features
            ("L_FEAT(x)", {"x": (1, 0, 1)}, 0),  # three features
        ],
    )
    def test_l_feat_function(self, expression, context, expected):
        """Test L_FEAT logic function: SIGMA y. (y << x) == 1."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("L_SIZE(x)", {"x": (1, None, None)}, 1),
            ("L_SIZE(x)", {"x": (None, None, None)}, 0),
            ("L_SIZE(x)", {"x": (1, 0, None)}, 2),
            ("L_SIZE(x)", {"x": (1, 0, 1)}, 3),
        ],
    )
    def test_l_size_function(self, expression, context, expected):
        """Test L_SIZE logic function: SIGMA y. (L_FEAT(y) * (y <<= x))."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    @pytest.mark.parametrize(
        "context,expected",
        [
            ({"x": (1, 0, 1), "y": (1, 0, 1)}, 0),  # y matches x exactly
            ({"x": (1, 0, 1), "y": (None, None, None)}, 3),  # y is BOTTOM
            ({"x": (1, 0, 1), "y": (1, None, None)}, 2),  # only position 0 kept
            ({"x": (1, 0, 1), "y": (0, 0, 1)}, 1),  # disagreement also counts as free
            ({"x": (None, None, None), "y": (1, 0, 1)}, 0),  # x has nothing to free
        ],
    )
    def test_l_free_function(self, context, expected):
        """Test L_FREE logic function: SIGMA u. (L_FEAT(u)*(u<<=x)*!(u<<=y))."""
        result = parse_and_evaluate("L_FREE(x, y)", context)
        assert result == expected

    @pytest.mark.parametrize(
        "context,expected",
        [
            ({"x": (1, 0, None, 1), "y": (1, None, 1, 1), "z": (1, None, None, 1)}, 1),
            (
                {
                    "x": (1, 0, None, 1, 1),
                    "y": (1, 1, 1, 1, 0),
                    "z": (1, None, None, 1, None),
                },
                1,
            ),
            ({"x": (1, 0, None, 1), "y": (1, None, 1, 1), "z": (0, None, None, 1)}, 0),
        ],
    )
    def test_l_meet_function(self, context, expected):
        """Test L_MEET logic function."""
        expr = "L_MEET(x, y, z)"
        result = parse_and_evaluate(expr, context)
        assert result == expected

    @pytest.mark.parametrize(
        "context,expected",
        [
            (
                {"x": (1, None), "y": (None, 1), "z": (1, 1)},
                1,
            ),  # Join exists
            (
                {"x": (1, 0), "y": (0, 1), "z": (1, 1)},
                0,
            ),  # No join
            (
                {"x": (1, None, 1), "y": (None, 1, None), "z": (1, 1, 1)},
                1,
            ),  # Join exists
            (
                {"x": (1, 0, None), "y": (0, 1, 0), "z": (1, 0, 1)},
                0,
            ),  # No join
        ],
    )
    def test_l_join_function(self, context, expected):
        """Test L_JOIN logic function."""
        expr = "L_JOIN(x, y, z)"
        result = parse_and_evaluate(expr, context)
        assert result == expected

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("L_L0(x)", {"x": (1, 0, None, 1)}, 0),  # Not empty tuple
            ("L_L0(x)", {"x": (None,) * 5}, 1),  # Empty tuple
        ],
    )
    def test_l_l0_function(self, expression, context, expected):
        """Test L_L0 logic function."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    @pytest.mark.parametrize(
        "context,expected",
        [
            # Different features can be partitioned
            ({"x": (1, 0, 1, 1, None), "y": (None, None, None, None, 1)}, 1),
            ({"x": (1, None, 1, 1, None), "y": (None, 0, None, None, 0)}, 1),
            # Same features cannot be partitioned
            ({"x": (1, 0, 1, 1, None), "y": (None, None, 1, None, 1)}, 0),
        ],
    )
    def test_l_partition_function(self, context, expected):
        """Test L_PARTITION logic function."""
        expr = "L_PARTITION(x, y)"
        result = parse_and_evaluate(expr, context)
        assert result == expected

    def test_sigma_with_bottom_reference(self):
        """Test SIGMA with BOTTOM function reference."""
        expr = "SIGMA y. (y <<= x)"
        context = {"x": parse_and_evaluate("BOTTOM(5)", {})}
        result = parse_and_evaluate(expr, context)
        assert result == 1  # The empty tuple is a subset of itself

    def test_complex_partition_expression(self):
        """Test complex PARTITION expression."""
        expr = (
            "(SIGMA z. (L_FULL(z) * (x <<= z) * (y <<= z)) >= 1) * "
            "SIGMA z. (L_L0(z) * L_MEET(x, y, z))"
        )
        context = {"x": (1, 0, None, None), "y": (None, None, 1, 1)}
        result = parse_and_evaluate(expr, context)
        assert result == 1  # x and y mention different features


class TestNativeFunctions:
    """Test native functions that don't require server connection."""

    @pytest.mark.parametrize(
        "length,expected",
        [
            (3, (None, None, None)),
            (1, (None,)),
            (12, (None,) * 12),
            (5, (None, None, None, None, None)),
        ],
    )
    def test_bottom_function(self, length, expected):
        """Test BOTTOM native function."""
        expr = f"BOTTOM({length})"
        result = parse_and_evaluate(expr, {})
        assert result == expected

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("FULL(x)", {"x": (1, 0, 1)}, 1),  # x is full
            ("FULL(x)", {"x": (1, 0, None)}, 0),  # x is not full
            ("FULL(x)", {"x": (1, 0, 1) * 4}, 1),  # x is full (repeated)
        ],
    )
    def test_full_function(self, expression, context, expected):
        """Test FULL native function."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    def test_full_function_with_bottom(self):
        """Test FULL function with BOTTOM reference."""
        expr = "FULL(x)"
        context = {"x": parse_and_evaluate("BOTTOM(12)", {})}
        result = parse_and_evaluate(expr, context)
        assert result == 0  # BOTTOM result is not full

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("L0(x)", {"x": (None,) * 5}, 1),  # L0 tuple
            ("L0(x)", {"x": (1, 0, None, 1)}, 0),  # Not L0 tuple
        ],
    )
    def test_l0_function(self, expression, context, expected):
        """Test L0 native function."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("FEAT(x)", {"x": (1, None, None)}, 1),  # exactly one feature
            ("FEAT(x)", {"x": (None, None, None)}, 0),  # no features
            ("FEAT(x)", {"x": (1, 0, None)}, 0),  # two features
            ("FEAT(x)", {"x": (1, 0, 1)}, 0),  # three features
        ],
    )
    def test_feat_function(self, expression, context, expected):
        """Test FEAT native function."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    def test_feat_function_type_validation(self):
        """Test that FEAT function validates argument types."""
        with pytest.raises(ValueError, match="FEAT argument must be tuple"):
            parse_and_evaluate("FEAT(x)", {"x": [1, 0]})

    @pytest.mark.parametrize(
        "expression,context,expected",
        [
            ("SIZE(x)", {"x": (1, None, None)}, 1),
            ("SIZE(x)", {"x": (None, None, None)}, 0),
            ("SIZE(x)", {"x": (1, 0, None)}, 2),
            ("SIZE(x)", {"x": (1, 0, 1)}, 3),
            ("SIZE(x)", {"x": ()}, 0),
        ],
    )
    def test_size_function(self, expression, context, expected):
        """Test SIZE native function."""
        result = parse_and_evaluate(expression, context)
        assert result == expected

    def test_size_function_type_validation(self):
        """Test that SIZE function validates argument types."""
        with pytest.raises(ValueError, match="SIZE argument must be tuple"):
            parse_and_evaluate("SIZE(x)", {"x": [1, 0]})

    @pytest.mark.parametrize(
        "context,expected",
        [
            ({"x": (1, 0, None, 1), "y": (1, None, 1, 1), "z": (1, None, None, 1)}, 1),
            (
                {
                    "x": (1, 0, None, 1, 1),
                    "y": (1, 1, 1, 1, 0),
                    "z": (1, None, None, 1, None),
                },
                1,
            ),
            ({"x": (1, 0, None, 1), "y": (1, None, 1, 1), "z": (0, None, None, 1)}, 0),
        ],
    )
    def test_meet_function(self, context, expected):
        """Test MEET native function."""
        expr = "MEET(x, y, z)"
        result = parse_and_evaluate(expr, context)
        assert result == expected

    @pytest.mark.parametrize(
        "context,expected",
        [
            # Valid JOIN cases - z is the least upper bound of x and y
            ({"x": (1, None, 0), "y": (1, None, 1), "z": (1, None, None)}, 0),
            ({"x": (None, 0, None), "y": (None, None, 1), "z": (None, None, None)}, 0),
            ({"x": (1, 0, None), "y": (1, None, 1), "z": (1, None, None)}, 0),
            ({"x": (1, 1, 1), "y": (1, 1, 1), "z": (1, 1, 1)}, 1),
            # Invalid JOIN cases - z is not the correct join
            # z should be (1, None, None)
            ({"x": (1, None, 0), "y": (1, None, 1), "z": (1, 0, None)}, 0),
            # x and y conflict, z should be None
            ({"x": (1, 0, None), "y": (0, 1, None), "z": (1, 1, None)}, 0),
            # z should equal common values
            ({"x": (1, 1, None), "y": (1, 1, None), "z": (1, 0, None)}, 0),
            # z should equal common value
            ({"x": (None, None, 1), "y": (None, None, 1), "z": (None, None, 0)}, 0),
            # Edge cases
            # All None
            (
                {
                    "x": (None, None, None),
                    "y": (None, None, None),
                    "z": (None, None, None),
                },
                1,
            ),
            ({"x": (1, 0, 1), "y": (1, 0, 1), "z": (1, 0, 1)}, 1),  # Identical tuples
        ],
    )
    def test_join_function(self, context, expected):
        """Test JOIN native function."""
        expr = "JOIN(x, y, z)"
        result = parse_and_evaluate(expr, context)
        assert result == expected

    def test_join_function_length_validation(self):
        """Test that JOIN function validates tuple lengths."""
        with pytest.raises(
            ValueError, match="JOIN arguments must be tuples of same length"
        ):
            parse_and_evaluate(
                "JOIN(x, y, z)", {"x": (1, 0), "y": (1, 0, None), "z": (1, 0)}
            )

    def test_join_function_type_validation(self):
        """Test that JOIN function validates argument types."""
        with pytest.raises(ValueError, match="JOIN arguments must be tuples"):
            parse_and_evaluate("JOIN(x, y, z)", {"x": [1, 0], "y": (1, 0), "z": (1, 0)})

    @pytest.mark.parametrize(
        "context,expected",
        [
            # Different features can be partitioned
            ({"x": (1, 0, 1, 1, None), "y": (None, None, None, None, 1)}, 1),
            ({"x": (1, None, 1, 1, None), "y": (None, 0, None, None, 0)}, 1),
            ({"x": (1, None, None, 1, None), "y": (None, None, None, None, 0)}, 1),
            # Same features cannot be partitioned
            ({"x": (1, 0, 1, 1, None), "y": (None, None, 1, None, 1)}, 0),
        ],
    )
    def test_partition_function(self, context, expected):
        """Test PARTITION native function."""
        expr = "PARTITION(x, y)"
        result = parse_and_evaluate(expr, context)
        assert result == expected

    @pytest.mark.parametrize(
        "context,expected",
        [
            # x refines to y and y is full -> y is a completion of x
            ({"x": (1, None, 1), "y": (1, 0, 1)}, 1),
            ({"x": (None, None, None), "y": (1, 0, 1)}, 1),
            ({"x": (1, 0, 1), "y": (1, 0, 1)}, 1),  # x already full and equal
            ({"x": (1, None, 1), "y": (1, 0, None)}, 0),  # y not full
            ({"x": (1, None, 1), "y": (0, 0, 1)}, 0),  # x doesn't refine to y
        ],
    )
    def test_comp_function(self, context, expected):
        """Test COMP native function: COMP(x, y) = (x <<= y) * FULL(y)."""
        result = parse_and_evaluate("COMP(x, y)", context)
        assert result == expected

    def test_comp_function_length_validation(self):
        """Test that COMP function validates tuple lengths."""
        with pytest.raises(
            ValueError, match="COMP operation requires equal-length tuples"
        ):
            parse_and_evaluate("COMP(x, y)", {"x": (1, 0), "y": (1, 0, None)})

    def test_comp_function_type_validation(self):
        """Test that COMP function validates argument types."""
        with pytest.raises(ValueError, match="COMP arguments must be tuples"):
            parse_and_evaluate("COMP(x, y)", {"x": [1, 0], "y": (1, 0)})

    @pytest.mark.parametrize(
        "context,expected",
        [
            ({"x": (1, 0, 1), "y": (1, 0, 1)}, 0),  # y matches x exactly
            ({"x": (1, 0, 1), "y": (None, None, None)}, 3),  # y is BOTTOM
            ({"x": (1, 0, 1), "y": (1, None, None)}, 2),  # only position 0 kept
            ({"x": (1, 0, 1), "y": (0, 0, 1)}, 1),  # disagreement also counts as free
            ({"x": (None, None, None), "y": (1, 0, 1)}, 0),  # x has nothing to free
            ({"x": (), "y": ()}, 0),  # empty tuples
        ],
    )
    def test_free_function(self, context, expected):
        """Test FREE native function: # positions x defines that y releases."""
        result = parse_and_evaluate("FREE(x, y)", context)
        assert result == expected

    def test_free_function_length_validation(self):
        """Test that FREE function validates tuple lengths."""
        with pytest.raises(
            ValueError, match="FREE operation requires equal-length tuples"
        ):
            parse_and_evaluate("FREE(x, y)", {"x": (1, 0), "y": (1, 0, None)})

    def test_free_function_type_validation(self):
        """Test that FREE function validates argument types."""
        with pytest.raises(ValueError, match="FREE requires tuple argument"):
            parse_and_evaluate("FREE(x, y)", {"x": [1, 0], "y": (1, 0)})


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_context(self):
        """Test expressions that don't need context variables."""
        result = parse_and_evaluate("BOTTOM(3)", {})
        assert result == (None, None, None)

    def test_nested_function_calls(self):
        """Test nested function calls."""
        expr = "FULL(BOTTOM(5))"
        result = parse_and_evaluate(expr, {})
        assert result == 0  # BOTTOM creates all None tuple, which is not FULL

    def test_complex_meet_expressions(self):
        """Test complex MEET expressions with various tuple lengths."""
        test_cases = [
            {
                "context": {
                    "x": (1, 0, None, 1),
                    "y": (1, None, 1, 1),
                    "z": (1, None, None, 1),
                },
                "expected": 1,
            },
            {
                "context": {"x": (1, 0, None), "y": (1, None, 1), "z": (0, None, None)},
                "expected": 0,
            },
        ]

        expr = (
            "(z <<= x) * (z <<= y) * (SIGMA w. (w <<= x) * (w <<= y) * "
            "(w <<= z) >= SIGMA w. (w <<= x) * (w <<= y))"
        )

        for case in test_cases:
            result = parse_and_evaluate(expr, case["context"])
            assert result == case["expected"]


class TestL1FunctionEquivalence:
    """Test that L_L1 and L1 functions return equal results."""

    @pytest.mark.parametrize(
        "tuple_input,expected",
        [
            # L1 tuples (exactly one element is not None) - should return 1
            ((0, None, None, None), 1),
            ((None, None, None, 1, None), 1),
            # Non-L1 tuples - both functions should return 0
            # Empty tuples (all None)
            ((None, None, None), 0),
            ((None, None, None, None, None), 0),
            # Multiple non-None elements
            ((1, 0, 1), 0),
            ((0, 1, None, 1), 0),
            ((1, None, 1), 0),
            # All elements are non-None
            ((0, 0, 0), 0),
            ((1, 1, 1), 0),
            # Edge cases
            ((None,), 0),  # Single None element
            ((0,), 1),  # Single zero element
        ],
    )
    def test_l1_and_l_l1_equivalence(self, tuple_input, expected):
        """Test that L1 and L_L1 functions return identical results."""
        context = {"x": tuple_input}

        # Test native L1 function
        l1_result = parse_and_evaluate("L1(x)", context)

        # Test logic-defined L_L1 function
        l_l1_result = parse_and_evaluate("L_L1(x)", context)

        # Both should return the expected value
        assert (
            l1_result == expected
        ), f"L1({tuple_input}) expected {expected}, got {l1_result}"
        assert (
            l_l1_result == expected
        ), f"L_L1({tuple_input}) expected {expected}, got {l_l1_result}"

        # Most importantly, both should return the same value
        assert l1_result == l_l1_result, (
            f"L1 and L_L1 differ for {tuple_input}: "
            f"L1={l1_result}, L_L1={l_l1_result}"
        )

    @pytest.mark.parametrize(
        "tuple_input",
        [
            (0, None, None, None),
            (None, None, None, None, None),
            (1, 0, 1),
            (None,),
            (0,),
            (),
        ],
    )
    def test_l1_is_alias_of_feat(self, tuple_input):
        """L1 is an alias of FEAT: both must agree on every input."""
        context = {"x": tuple_input}
        assert parse_and_evaluate("L1(x)", context) == parse_and_evaluate(
            "FEAT(x)", context
        )


class TestOppositeFunctionEquivalence:
    """Test that L_OPPOSITE and OPPOSITE functions return equal results."""

    @pytest.mark.parametrize(
        "tuple_x,tuple_y,expected",
        [
            # Valid OPPOSITE cases - should return 1
            # Both are L1 tuples, same position, different values
            ((None, 1, None), (None, 0, None), 1),
            ((None, None, 1), (None, None, 0), 1),
            # Invalid OPPOSITE cases - should return 0
            # Different positions (even if both L1)
            ((1, None, None), (None, 0, None), 0),
            ((None, 1, None), (None, None, 0), 0),
            ((None, 0, None), (None, 0, None), 0),
            # Multiple positions with values (violates L1 constraint)
            ((1, 1, None), (None, None, 0), 0),
            ((1, None, None), (0, 1, None), 0),
            # All None tuples (violates L1 constraint)
            ((None, None, None), (1, None, None), 0),  # x is L0, not L1
            ((None, None, None), (None, None, None), 0),  # Both are L0
            # Edge cases - single element tuples
            ((None,), (1,), 0),  # x is L0, not L1
            ((1,), (1,), 0),  # Same position, same value
            ((1,), (0,), 1),  # Same pos (0), diff values (1≠0)
        ],
    )
    def test_opposite_and_l_opposite_equivalence(self, tuple_x, tuple_y, expected):
        """Test that OPPOSITE and L_OPPOSITE functions return identical results."""

        # Test native OPPOSITE function
        opposite_result = parse_and_evaluate(
            "OPPOSITE(x, y)", {"x": tuple_x, "y": tuple_y}
        )

        # Test logic-defined L_OPPOSITE function
        l_opposite_result = parse_and_evaluate(
            "L_OPPOSITE(x, y)", {"x": tuple_x, "y": tuple_y}
        )

        # Both should return the expected value
        assert opposite_result == expected, (
            f"OPPOSITE({tuple_x}, {tuple_y}) expected {expected}, "
            f"got {opposite_result}"
        )
        assert l_opposite_result == expected, (
            f"L_OPPOSITE({tuple_x}, {tuple_y}) expected {expected}, "
            f"got {l_opposite_result}"
        )

        # Most importantly, both should return the same value
        assert opposite_result == l_opposite_result, (
            f"OPPOSITE and L_OPPOSITE differ for ({tuple_x}, {tuple_y}): "
            f"OPPOSITE={opposite_result}, L_OPPOSITE={l_opposite_result}"
        )


class TestSufFunctionEquivalence:
    """Test SUF and L_SUF function equivalence with comprehensive test cases."""

    @pytest.mark.parametrize(
        "tuple_x,tuple_y,expected",
        [
            # Compatible patterns (both should return 1)
            ((1, None, None, None), (0, 1, None, None), 0),
            ((1, None, None, 1), (0, None, None, 1), 1),
            ((None, None, 1, None), (None, None, 0, None), 1),
            # SUF should work with same tuples
            ((None, None, None), (None, None, None), 1),
            ((1, None, None), (None, 1, None), 0),
            # Edge cases
            ((1,), (0,), 1),
            ((1,), (None,), 0),
        ],
    )
    def test_suf_and_l_suf_equivalence(self, tuple_x, tuple_y, expected):
        """Test that SUF and L_SUF functions return identical results."""
        # Test native SUF function
        suf_result = parse_and_evaluate("SUF(x, y)", {"x": tuple_x, "y": tuple_y})

        # Test logic-defined L_SUF function
        l_suf_result = parse_and_evaluate("L_SUF(x, y)", {"x": tuple_x, "y": tuple_y})

        # Both should return the expected value
        assert suf_result == expected, (
            f"SUF({tuple_x}, {tuple_y}) expected {expected}, " f"got {suf_result}"
        )
        assert l_suf_result == expected, (
            f"L_SUF({tuple_x}, {tuple_y}) expected {expected}, " f"got {l_suf_result}"
        )

        # Most importantly, both should return the same value
        assert suf_result == l_suf_result, (
            f"SUF and L_SUF differ for ({tuple_x}, {tuple_y}): "
            f"SUF={suf_result}, L_SUF={l_suf_result}"
        )


class TestCompFunctionEquivalence:
    """Test COMP and L_COMP function equivalence with comprehensive test cases."""

    @pytest.mark.parametrize(
        "tuple_x,tuple_y,expected",
        [
            # x refines to y and y is full -> y is a completion of x
            ((1, None, 1), (1, 0, 1), 1),
            ((None, None, None), (1, 0, 1), 1),
            ((1, 0, 1), (1, 0, 1), 1),  # x already full and equal to y
            ((1, None, 1), (1, 0, None), 0),  # y not full
            ((1, None, 1), (0, 0, 1), 0),  # x doesn't refine to y
            ((), (), 1),  # empty tuples: vacuously refines, FULL(()) is True
        ],
    )
    def test_comp_and_l_comp_equivalence(self, tuple_x, tuple_y, expected):
        """Test that COMP and L_COMP functions return identical results."""
        # Test native COMP function
        comp_result = parse_and_evaluate("COMP(x, y)", {"x": tuple_x, "y": tuple_y})

        # Test logic-defined L_COMP function
        l_comp_result = parse_and_evaluate(
            "L_COMP(x, y)", {"x": tuple_x, "y": tuple_y}
        )

        # Both should return the expected value
        assert comp_result == expected, (
            f"COMP({tuple_x}, {tuple_y}) expected {expected}, " f"got {comp_result}"
        )
        assert l_comp_result == expected, (
            f"L_COMP({tuple_x}, {tuple_y}) expected {expected}, "
            f"got {l_comp_result}"
        )

        # Most importantly, both should return the same value
        assert comp_result == l_comp_result, (
            f"COMP and L_COMP differ for ({tuple_x}, {tuple_y}): "
            f"COMP={comp_result}, L_COMP={l_comp_result}"
        )


class TestFeatFunctionEquivalence:
    """Test FEAT and L_FEAT function equivalence with comprehensive test cases."""

    @pytest.mark.parametrize(
        "tuple_x,expected",
        [
            ((1, None, None), 1),
            ((None, 1, None), 1),
            ((None, None, None), 0),  # no features
            ((1, 0, None), 0),  # two features
            ((1, 0, 1), 0),  # three features
            ((1,), 1),  # single-position tuple, specified
            ((None,), 0),  # single-position tuple, unspecified
        ],
    )
    def test_feat_and_l_feat_equivalence(self, tuple_x, expected):
        """Test that FEAT and L_FEAT functions return identical results."""
        feat_result = parse_and_evaluate("FEAT(x)", {"x": tuple_x})
        l_feat_result = parse_and_evaluate("L_FEAT(x)", {"x": tuple_x})

        assert feat_result == expected, f"FEAT({tuple_x}) expected {expected}"
        assert l_feat_result == expected, f"L_FEAT({tuple_x}) expected {expected}"
        assert feat_result == l_feat_result, (
            f"FEAT and L_FEAT differ for {tuple_x}: "
            f"FEAT={feat_result}, L_FEAT={l_feat_result}"
        )


class TestSizeFunctionEquivalence:
    """Test SIZE and L_SIZE function equivalence with comprehensive test cases."""

    @pytest.mark.parametrize(
        "tuple_x,expected",
        [
            ((1, None, None), 1),
            ((None, None, None), 0),
            ((1, 0, None), 2),
            ((1, 0, 1), 3),
            ((1, 0, 1, None, 1), 4),
        ],
    )
    def test_size_and_l_size_equivalence(self, tuple_x, expected):
        """Test that SIZE and L_SIZE functions return identical results."""
        size_result = parse_and_evaluate("SIZE(x)", {"x": tuple_x})
        l_size_result = parse_and_evaluate("L_SIZE(x)", {"x": tuple_x})

        assert size_result == expected, f"SIZE({tuple_x}) expected {expected}"
        assert l_size_result == expected, f"L_SIZE({tuple_x}) expected {expected}"
        assert size_result == l_size_result, (
            f"SIZE and L_SIZE differ for {tuple_x}: "
            f"SIZE={size_result}, L_SIZE={l_size_result}"
        )


class TestFreeFunctionEquivalence:
    """Test FREE and L_FREE function equivalence with comprehensive test cases."""

    @pytest.mark.parametrize(
        "tuple_x,tuple_y,expected",
        [
            ((1, 0, 1), (1, 0, 1), 0),  # y matches x exactly
            ((1, 0, 1), (None, None, None), 3),  # y is BOTTOM
            ((1, 0, 1), (1, None, None), 2),  # only position 0 kept
            ((1, 0, 1), (0, 0, 1), 1),  # disagreement also counts as free
            ((None, None, None), (1, 0, 1), 0),  # x has nothing to free
            ((1, None, 1, None), (1, 0, None, None), 1),  # mixed
        ],
    )
    def test_free_and_l_free_equivalence(self, tuple_x, tuple_y, expected):
        """Test that FREE and L_FREE functions return identical results."""
        free_result = parse_and_evaluate("FREE(x, y)", {"x": tuple_x, "y": tuple_y})
        l_free_result = parse_and_evaluate("L_FREE(x, y)", {"x": tuple_x, "y": tuple_y})

        assert free_result == expected, f"FREE({tuple_x}, {tuple_y}) expected {expected}"
        assert l_free_result == expected, f"L_FREE({tuple_x}, {tuple_y}) expected {expected}"
        assert free_result == l_free_result, (
            f"FREE and L_FREE differ for ({tuple_x}, {tuple_y}): "
            f"FREE={free_result}, L_FREE={l_free_result}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
