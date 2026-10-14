"""Integration tests for QFOIL evaluation with a PyJuice backend."""

import pytest
import torch
from pathlib import Path

from qfoil import QFOILEvaluator, QFOILParser
from qfoil.backends.pyjuice import PyJuiceModel, PyJuiceProbabilityBackend

pytestmark = pytest.mark.cuda

# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "examples" / "models" / "mnist_binary_small_0_vs_1"


@pytest.fixture(scope="module")
def device():
    """Get compute device."""
    if not torch.cuda.is_available():
        pytest.skip("PyJuice circuit compilation requires CUDA")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def pyjuice_model(device):
    """Load pre-trained PyJuice model."""
    model = PyJuiceModel.load(str(MODEL_DIR), device=device)
    return model


@pytest.fixture(scope="module")
def evaluator(pyjuice_model):
    """Create PyJuice evaluator instance."""
    instance_len = pyjuice_model.num_features
    parser = QFOILParser(instance_len=instance_len)
    backend = PyJuiceProbabilityBackend(pyjuice_model)
    evaluator = QFOILEvaluator(
        probabilistic_backend=backend,
        instance_len=instance_len,
        parser=parser,
    )
    evaluator.set_parser(parser)
    return evaluator


class TestQFOILEvaluatorWithPyJuice:
    """Test basic evaluator setup and infrastructure."""

    def test_evaluator_creation(self, evaluator, pyjuice_model):
        """Test evaluator is created correctly."""
        assert evaluator is not None
        assert evaluator.probabilistic_backend.model == pyjuice_model
        assert evaluator.instance_len == pyjuice_model.num_features
        assert evaluator.domain == {0, 1, None}

    def test_native_functions_defined(self, evaluator):
        """Test native functions are registered."""
        expected_natives = {
            "BOTTOM",
            "FULL",
            "L0",
            "L1",
            "FEAT",
            "SIZE",
            "FREE",
            "MEET",
            "JOIN",
            "PARTITION",
            "VAL",
            "COND",
            "EP",
            "CEP",
            "OPPOSITE",
            "SUF",
            "COMP",
        }
        assert evaluator.function_runtime.native_names == expected_natives

    def test_backend_initialized(self, evaluator):
        """Test PyJuice backend is initialized."""
        assert evaluator.probabilistic_backend is not None
        assert (
            evaluator.probabilistic_backend.device
            == evaluator.probabilistic_backend.model.device
        )


class TestBasicTupleFunctions:
    """Test basic tuple functions: BOTTOM, FULL, L0, L1."""

    @pytest.mark.parametrize(
        "length,expected",
        [
            (3, (None, None, None)),
            (1, (None,)),
            (0, ()),
            (5, (None, None, None, None, None)),
        ],
    )
    def test_bottom_function(self, evaluator, length, expected):
        """Test BOTTOM native function."""
        result = evaluator.function_runtime.evaluate_native("BOTTOM", [length])
        assert result == expected

    @pytest.mark.parametrize(
        "input_tuple,expected",
        [
            ((1, 0, 1), 1),  # Full tuple
            ((1, 0, None), 0),  # Partial tuple
            ((None, None, None), 0),  # All None
            ((), 1),  # Empty tuple is logically FULL
        ],
    )
    def test_full_function(self, evaluator, input_tuple, expected):
        """Test FULL native function."""
        result = evaluator.function_runtime.evaluate_native("FULL", [input_tuple])
        assert result == expected

    @pytest.mark.parametrize(
        "input_tuple,expected",
        [
            ((None, None, None), 1),  # All None (L0)
            ((1, 0, None), 0),  # Partial
            ((1, 0, 1), 0),  # Full
            ((), 1),  # Empty tuple is L0
        ],
    )
    def test_l0_function(self, evaluator, input_tuple, expected):
        """Test L0 native function."""
        result = evaluator.function_runtime.evaluate_native("L0", [input_tuple])
        assert result == expected

    def test_l1_true(self, evaluator):
        """Test L1 returns 1 for single-element tuples."""
        result = evaluator.function_runtime.evaluate_native("L1", [(1, None, None)])
        assert result == 1

        result = evaluator.function_runtime.evaluate_native("L1", [(None, 0, None)])
        assert result == 1

    def test_l1_false(self, evaluator):
        """Test L1 returns 0 for non-single-element tuples."""
        result = evaluator.function_runtime.evaluate_native("L1", [(None, None, None)])
        assert result == 0

        result = evaluator.function_runtime.evaluate_native("L1", [(1, 0, None)])
        assert result == 0


class TestEvaluateBasicExpressions:
    """Test basic expression evaluation (variables, numbers, operators)."""

    def test_evaluate_variable(self, evaluator):
        """Test variable evaluation from context."""
        context = {"x": (1, 0, None)}
        from qfoil.language import Variable

        ast = Variable("x")
        result = evaluator.evaluate(ast, context)
        assert result == (1, 0, None)

    def test_evaluate_number(self, evaluator):
        """Test number literal evaluation."""
        from qfoil.language import NumberLiteral

        ast = NumberLiteral(42)
        result = evaluator.evaluate(ast, {})
        assert result == 42

    def test_evaluate_tuple(self, evaluator):
        """Test tuple literal evaluation."""
        from qfoil.language import NumberLiteral, TupleLiteral

        ast = TupleLiteral([NumberLiteral(1), NumberLiteral(0), NumberLiteral(None)])
        result = evaluator.evaluate(ast, {})
        assert result == (1, 0, None)

    def test_evaluate_subseteq_true(self, evaluator):
        """Test SUBSETEQ (<<<) operator returns 1 when true."""
        result = evaluator.tuple_semantics.subseteq((1, None, None), (1, 0, 1))
        assert result == 1

        result = evaluator.tuple_semantics.subseteq((None, 0, None), (1, 0, 1))
        assert result == 1

        # Reflexive: a tuple is a SUBSETEQ of itself.
        result = evaluator.tuple_semantics.subseteq((1, 0, 1), (1, 0, 1))
        assert result == 1

    def test_evaluate_subseteq_false(self, evaluator):
        """Test SUBSETEQ (<<<) operator returns 0 when false."""
        result = evaluator.tuple_semantics.subseteq((1, None, None), (0, 0, 1))
        assert result == 0

        result = evaluator.tuple_semantics.subseteq((None, 1, None), (1, 0, 1))
        assert result == 0

    def test_evaluate_subset_true(self, evaluator):
        """Test SUBSET (<<) strict operator returns 1 when true."""
        result = evaluator.tuple_semantics.subset((1, None, None), (1, 0, 1))
        assert result == 1

        result = evaluator.tuple_semantics.subset((None, 0, None), (1, 0, 1))
        assert result == 1

    def test_evaluate_subset_false(self, evaluator):
        """Test SUBSET (<<) strict operator returns 0 when false."""
        result = evaluator.tuple_semantics.subset((1, None, None), (0, 0, 1))
        assert result == 0

        result = evaluator.tuple_semantics.subset((None, 1, None), (1, 0, 1))
        assert result == 0

    def test_evaluate_subset_excludes_equal_tuples(self, evaluator):
        """SUBSET is strict: identical tuples satisfy SUBSETEQ but not SUBSET."""
        result = evaluator.tuple_semantics.subset((1, 0, 1), (1, 0, 1))
        assert result == 0
        assert evaluator.tuple_semantics.subseteq((1, 0, 1), (1, 0, 1)) == 1


class TestTupleOperations:
    """Test tuple operation functions (MEET, JOIN, PARTITION, OPPOSITE, SUF)."""

    def test_meet_valid(self, evaluator):
        """Test MEET with valid inputs."""
        # x=(1, 2, 3), y=(1, 5, 3) differ at position 1
        # So z must have None at position 1 and match at positions 0 and 2
        result = evaluator.function_runtime.evaluate_native(
            "MEET", [(1, 2, 3), (1, 5, 3), (1, None, 3)]
        )
        assert result == 1

        # x=(1, None, 3), y=(1, None, 3) are identical
        # So z must be (1, None, 3)
        result = evaluator.function_runtime.evaluate_native(
            "MEET", [(1, None, 3), (1, None, 3), (1, None, 3)]
        )
        assert result == 1

    def test_meet_invalid(self, evaluator):
        """Test MEET with invalid inputs."""
        # x=(1, 2, 3), y=(1, 5, 3) differ at position 1
        # z should have None at position 1, but has 2 instead
        result = evaluator.function_runtime.evaluate_native(
            "MEET", [(1, 2, 3), (1, 5, 3), (1, 2, 3)]
        )
        assert result == 0

        # x=(1, 2, 3), y=(1, 2, 3) agree at position 1
        # z should have 2 at position 1, but has None instead
        result = evaluator.function_runtime.evaluate_native(
            "MEET", [(1, 2, 3), (1, 2, 3), (1, None, 3)]
        )
        assert result == 0

    def test_join_invalid(self, evaluator):
        """Test JOIN with invalid inputs."""
        # x=(1, 2, 3), y=(1, 5, 3) differ at position 1 (both non-None)
        # So theres no valid join
        result = evaluator.function_runtime.evaluate_native(
            "JOIN", [(1, 2, 3), (1, 5, 3), (1, None, 3)]
        )
        assert result == 0

        # x=(1, None, 3), y=(1, 2, 3) differ at position 1 (y is non-None, x is None)
        # So z must have 2 at position 1
        result = evaluator.function_runtime.evaluate_native(
            "JOIN", [(1, None, 3), (1, 2, 3), (1, None, 3)]
        )
        assert result == 0

    def test_join_valid(self, evaluator):
        """Test JOIN with valid inputs."""

        result = evaluator.function_runtime.evaluate_native(
            "JOIN", [(1, 2, 3), (1, None, None), (1, 2, 3)]
        )
        assert result == 1

        result = evaluator.function_runtime.evaluate_native(
            "JOIN", [(1, None, 3), (1, 2, 3), (1, 2, 3)]
        )
        assert result == 1

    def test_partition_valid(self, evaluator):
        """Test PARTITION with valid inputs."""
        result = evaluator.function_runtime.evaluate_native(
            "PARTITION", [(None, None, 3), (1, 2, None)]
        )
        assert result == 1

        result = evaluator.function_runtime.evaluate_native(
            "PARTITION", [(None, None, None), (1, 2, 3)]
        )
        assert result == 1

    def test_partition_invalid(self, evaluator):
        """Test PARTITION with invalid inputs."""
        # x=(1, 2, 3), y=(1, 5, 3) have different non-None values at position 1
        result = evaluator.function_runtime.evaluate_native("PARTITION", [(1, 2, 3), (1, 5, 3)])
        assert result == 0

    def test_opposite_valid(self, evaluator):
        """Test OPPOSITE with valid inputs."""
        # x=(1, None, None), y=(5, None, None) differ at exactly one position
        result = evaluator.function_runtime.evaluate_native(
            "OPPOSITE", [(1, None, None), (5, None, None)]
        )
        assert result == 1

        # x=(None, 2, None), y=(None, 5, None) differ at exactly one position
        result = evaluator.function_runtime.evaluate_native(
            "OPPOSITE", [(None, 2, None), (None, 5, None)]
        )
        assert result == 1

    def test_opposite_invalid(self, evaluator):
        """Test OPPOSITE with invalid inputs."""
        # x=(1, 2, 3), y=(5, 6, 3) differ at two positions (0 and 1)
        result = evaluator.function_runtime.evaluate_native("OPPOSITE", [(1, 2, 3), (5, 6, 3)])
        assert result == 0

        # x=(1, 2, 3), y=(1, 2, 3) are identical
        result = evaluator.function_runtime.evaluate_native("OPPOSITE", [(1, 2, 3), (1, 2, 3)])
        assert result == 0

    def test_suf_valid(self, evaluator):
        """Test SUF with valid inputs."""
        result = evaluator.function_runtime.evaluate_native("SUF", [(1, None, 3), (2, None, 3)])
        assert result == 1

        result = evaluator.function_runtime.evaluate_native(
            "SUF", [(1, None, None), (2, None, None)]
        )
        assert result == 1

        result = evaluator.function_runtime.evaluate_native(
            "SUF", [(None, None, None), (None, None, None)]
        )
        assert result == 1

    def test_suf_invalid(self, evaluator):
        """Test SUF with invalid inputs."""
        result = evaluator.function_runtime.evaluate_native("SUF", [(1, None, 3), (1, 5, 3)])
        assert result == 0

        result = evaluator.function_runtime.evaluate_native("SUF", [(None, 2, 3), (1, 5, None)])
        assert result == 0


class TestProbabilisticFunctions:
    """Test probabilistic functions (VAL, COND, EP, CEP).

    Class is always an explicit extra argument (the QFOIL-PC "Alternative 2"
    vocabulary), never embedded as a tuple position - tuples here have
    exactly `evaluator.probabilistic_backend.model.num_features` (196) positions, no +1 for class.
    """

    def test_val_marginal(self, evaluator):
        """VAL(x) = Pr(x), marginalized over the class variable."""
        num_features = evaluator.probabilistic_backend.model.num_features

        zeros = tuple([0] * num_features)
        result = evaluator.function_runtime.evaluate_native("VAL", [zeros])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

        ones = tuple([1] * num_features)
        result = evaluator.function_runtime.evaluate_native("VAL", [ones])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_val_joint_with_class(self, evaluator):
        """VAL(x, c) = Pr(x, Y=c) for each class the model knows about."""
        num_features = evaluator.probabilistic_backend.model.num_features
        zeros = tuple([0] * num_features)

        for c in evaluator.probabilistic_backend.model.pc_per_class:
            result = evaluator.function_runtime.evaluate_native("VAL", [zeros, c])
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_val_errors(self, evaluator):
        """Test VAL error handling."""
        num_features = evaluator.probabilistic_backend.model.num_features

        with pytest.raises(
            ValueError, match=r"VAL requires 1 argument \(x\) or 2 arguments"
        ):
            evaluator.function_runtime.evaluate_native("VAL", [])

        with pytest.raises(ValueError, match="VAL's x argument must be a tuple"):
            evaluator.function_runtime.evaluate_native("VAL", [[1, 2, 3]])

        with pytest.raises(ValueError, match="doesn't match model's num_features"):
            wrong_size = tuple([0] * 100)
            evaluator.function_runtime.evaluate_native("VAL", [wrong_size])

        with pytest.raises(ValueError, match="Unknown class id"):
            evaluator.function_runtime.evaluate_native(
                "VAL", [tuple([0] * num_features), 99]
            )

    def test_cond_basic(self, evaluator):
        """COND(x, y, c) = Pr(x, Y=c | y), for disjoint partial x, y."""
        num_features = evaluator.probabilistic_backend.model.num_features

        x = tuple([1] + [None] * (num_features - 1))
        y = tuple([None, 0] + [None] * (num_features - 2))
        for c in evaluator.probabilistic_backend.model.pc_per_class:
            result = evaluator.function_runtime.evaluate_native("COND", [x, y, c])
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_cond_conflicting_features(self, evaluator):
        """x and y disagreeing at a shared position -> probability 0."""
        num_features = evaluator.probabilistic_backend.model.num_features

        x = tuple([1] + [None] * (num_features - 1))
        y = tuple([0] + [None] * (num_features - 1))  # conflicts with x at 0

        result = evaluator.function_runtime.evaluate_native("COND", [x, y, 0])
        assert result == 0.0

    def test_cond_errors(self, evaluator):
        """Test COND error handling."""
        num_features = evaluator.probabilistic_backend.model.num_features
        full = tuple([1] * num_features)

        with pytest.raises(
            ValueError, match=r"COND requires exactly 3 arguments \(x, y, c\)"
        ):
            evaluator.function_runtime.evaluate_native("COND", [full, full])

        with pytest.raises(ValueError, match="COND's x and y arguments must be tuples"):
            evaluator.function_runtime.evaluate_native("COND", [[1, 0, 0], full, 0])

        with pytest.raises(ValueError, match="doesn't match model's num_features"):
            wrong_size = tuple([0] * 100)
            evaluator.function_runtime.evaluate_native("COND", [wrong_size, wrong_size, 0])

        with pytest.raises(ValueError, match="Unknown class id"):
            evaluator.function_runtime.evaluate_native("COND", [full, full, 99])

    def test_ep_basic(self, evaluator):
        """EP(x, c) = Pr(Y=c | x) is a valid probability for every class."""
        num_features = evaluator.probabilistic_backend.model.num_features
        x = tuple([1, None] + [1] * (num_features - 2))

        for c in evaluator.probabilistic_backend.model.pc_per_class:
            result = evaluator.function_runtime.evaluate_native("EP", [x, c])
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_ep_equals_cond_with_bottom(self, evaluator):
        """EP(x, c) must agree with its defining identity COND(BOTTOM, x, c)."""
        num_features = evaluator.probabilistic_backend.model.num_features
        x = tuple([1, None] + [1] * (num_features - 2))
        bottom = (None,) * num_features

        ep_result = evaluator.function_runtime.evaluate_native("EP", [x, 0])
        cond_result = evaluator.function_runtime.evaluate_native("COND", [bottom, x, 0])
        assert ep_result == pytest.approx(cond_result)

    def test_ep_errors(self, evaluator):
        """Test EP error handling."""
        num_features = evaluator.probabilistic_backend.model.num_features
        full = tuple([1] * num_features)

        with pytest.raises(
            ValueError, match=r"EP requires exactly 2 arguments \(x, c\)"
        ):
            evaluator.function_runtime.evaluate_native("EP", [full])

        with pytest.raises(ValueError, match="EP's x argument must be a tuple"):
            evaluator.function_runtime.evaluate_native("EP", [[1, 1, 1], 0])

        with pytest.raises(ValueError, match="Unknown class id"):
            evaluator.function_runtime.evaluate_native("EP", [full, 99])

    def test_cep_is_one_minus_ep(self, evaluator):
        """CEP(x, c) = 1 - EP(x, c), for every class."""
        num_features = evaluator.probabilistic_backend.model.num_features
        x = tuple([1, None] + [1] * (num_features - 2))

        for c in evaluator.probabilistic_backend.model.pc_per_class:
            ep_result = evaluator.function_runtime.evaluate_native("EP", [x, c])
            cep_result = evaluator.function_runtime.evaluate_native("CEP", [x, c])
            assert cep_result == pytest.approx(1.0 - ep_result)

    def test_cep_errors(self, evaluator):
        """Test CEP error handling."""
        num_features = evaluator.probabilistic_backend.model.num_features
        full = tuple([1] * num_features)

        with pytest.raises(
            ValueError, match=r"CEP requires exactly 2 arguments \(x, c\)"
        ):
            evaluator.function_runtime.evaluate_native("CEP", [full])

        with pytest.raises(ValueError, match="Unknown class id"):
            evaluator.function_runtime.evaluate_native("CEP", [full, 99])


class TestSigmaLocalSearchWithProbabilisticFunctions:
    """The motivating use case for the Fase 1 numeric-comparison extension:
    a PSR-shaped `SIGMA z. (skeleton) * (EP(z, c) >= delta) >= 1` existence
    query, at real MNIST scale (196 features).

    Before this extension, `EP(z, c) >= delta` inside the conjunction made
    `SigmaSearch.soft_violation` return None for the whole formula (one unsupported
    conjunct poisons the `*`), disabling heuristic search entirely and
    falling back to the exact lazy scan - which, at n=196, essentially
    never finds a small-SIZE witness (`itertools.product`'s canonical
    order keeps almost every leading position pinned at 0 until the
    trailing ~3^195 combinations are exhausted). This is not tested
    against that fallback here (it would never finish); it only checks
    that the heuristic path itself resolves correctly and quickly, using
    the real PyJuiceModel as the oracle for every candidate it tries.
    """

    def test_finds_small_witness_via_ep_threshold(self, evaluator):
        num_features = evaluator.probabilistic_backend.model.num_features
        c = next(iter(evaluator.probabilistic_backend.model.pc_per_class))
        x = tuple([1] * num_features)
        bottom = tuple([None] * num_features)

        # delta set just above the unconditional EP(BOTTOM, c) (computed
        # from the real model, not hardcoded) so BOTTOM itself does NOT
        # already satisfy it - the search has to actually add evidence.
        ep_bottom = evaluator.function_runtime.evaluate_native("EP", [bottom, c])
        delta = ep_bottom + 0.05
        size_max = 3

        expr = (
            f"(SIGMA z. (z <<< x) * (SIZE(z) <= {size_max}) "
            f"* (EP(z, c) >= {delta})) >= 1"
        )
        ast = evaluator.parser.parse(expr)

        import time

        t0 = time.perf_counter()
        result = evaluator.evaluate(ast, {"x": x, "c": c})
        elapsed = time.perf_counter() - t0

        assert result == 1
        # Not a hard perf assertion (machine-dependent), just a sanity
        # bound that this used the heuristic path and not an accidental
        # full/near-full scan of a 196-feature domain.
        assert elapsed < 30.0


class TestSigmaGreedyAscentWithProbabilisticFunctions:
    """`SigmaSearch.greedy_ascent`'s motivating use case: warm-starting a
    `minimize`/`maximize` search that needs to grow a witness while an
    `EP`/`CEP` threshold holds, at real MNIST scale. Unlike
    `SigmaSearch.local_search`, this isn't wired into `SigmaEvaluator.decide`/
    `evaluate()` - it's called directly, exactly as `Z3CegisSolver.
    _warm_start_witness` does.
    """

    def test_grows_witness_via_ep_threshold(self, evaluator):
        from qfoil.language import Sigma

        num_features = evaluator.probabilistic_backend.model.num_features
        c = next(iter(evaluator.probabilistic_backend.model.pc_per_class))
        x = tuple([1] * num_features)
        bottom = tuple([None] * num_features)

        ep_bottom = evaluator.function_runtime.evaluate_native("EP", [bottom, c])
        delta = ep_bottom + 0.05

        formula_ast = evaluator.parser.parse(f"(z <<< x) * (EP(z, c) >= {delta})")
        sigma_ast = Sigma("z", formula_ast)
        reference = tuple([None] * num_features)

        witness = evaluator.sigma.search.greedy_ascent(
            sigma_ast, {"x": x, "c": c}, reference, budget=2000
        )

        assert witness is not None
        verified = evaluator.evaluate(formula_ast, {"x": x, "c": c, "z": witness})
        assert verified == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
