"""Tests for low-level PyJuice model storage and inference.

This module tests probabilistic circuit inference using PyJuice with
pre-trained MNIST binary classification models (class 0 vs class 1).

Test data:
- examples/models/mnist_binary_model_class_0.jpc
- examples/models/mnist_binary_model_class_1.jpc
- examples/models/mnist_binary_metadata.json
- examples/models/mnist_test_data_binary_small.pt
"""

import pytest
import torch
import pyjuice as juice
from pathlib import Path

from qfoil.backends.pyjuice import PyJuiceInference, PyJuiceModel

pytestmark = pytest.mark.cuda

# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "examples" / "models" / "mnist_binary_small_0_vs_1"
TEST_DATA_PATH = MODEL_DIR / "mnist_test_data_binary_small.pt"
TEST_LABELS_PATH = MODEL_DIR / "test_targets.pt"


@pytest.fixture(scope="module")
def device():
    """Get compute device."""
    if not torch.cuda.is_available():
        pytest.skip("PyJuice circuit compilation requires CUDA")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def test_data(device):
    """Load test data."""
    data = torch.load(TEST_DATA_PATH, map_location=device, weights_only=True)
    return data


@pytest.fixture(scope="module")
def test_labels(device):
    """Load test labels."""
    labels = torch.load(TEST_LABELS_PATH, map_location=device, weights_only=True)
    return labels


@pytest.fixture(scope="module")
def pyjuice_model(device):
    """Load pre-trained PyJuice model."""
    model = PyJuiceModel.load(str(MODEL_DIR), device=device)
    return model


@pytest.fixture(scope="module")
def backend(device):
    """Create PyJuice backend instance."""
    return PyJuiceInference(device=str(device))


class TestPyJuiceModel:
    """Test PyJuiceModel container class."""

    def test_load_model(self, pyjuice_model):
        """Test loading pre-trained model."""
        assert pyjuice_model is not None
        assert pyjuice_model.num_classes == 2
        assert pyjuice_model.num_features == 196
        assert pyjuice_model.img_size == 14

    def test_model_structure(self, pyjuice_model):
        """Test model has correct structure."""
        # Check pc_per_class
        assert len(pyjuice_model.pc_per_class) == 2
        assert 0 in pyjuice_model.pc_per_class
        assert 1 in pyjuice_model.pc_per_class

        # Check priors
        assert len(pyjuice_model.priors) == 2
        assert 0 in pyjuice_model.priors
        assert 1 in pyjuice_model.priors

        # Priors should sum to approximately 1
        prior_sum = sum(pyjuice_model.priors.values())
        assert abs(prior_sum - 1.0) < 1e-5

    def test_save_load_roundtrip(self, pyjuice_model, tmp_path, device):
        """Test save and load model."""
        # Save to temporary directory
        save_dir = tmp_path / "test_model"
        pyjuice_model.save(str(save_dir))

        # Check files exist
        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "class_0.jpc").exists()
        assert (save_dir / "class_1.jpc").exists()

        # Load model back
        loaded_model = PyJuiceModel.load(str(save_dir), device=device)

        # Check structure matches
        assert loaded_model.num_classes == pyjuice_model.num_classes
        assert loaded_model.num_features == pyjuice_model.num_features
        assert loaded_model.img_size == pyjuice_model.img_size

        # Check priors match
        for k in pyjuice_model.priors.keys():
            assert abs(loaded_model.priors[k] - pyjuice_model.priors[k]) < 1e-6


class TestPyJuiceInferenceMarginal:
    """Test marginal query functionality."""

    def test_marginal_all_observed(self, pyjuice_model, test_data):
        """Test marginal query with all features observed."""
        # Use first 10 samples
        data_batch = test_data[:10]
        observed_mask = torch.ones_like(data_batch, dtype=torch.bool)

        # Compute P(x) - marginalize over classes
        log_p_x = PyJuiceInference.marginal_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask,
            class_value=None,
        )

        # Check output shape
        assert log_p_x.shape == (10,)

        # Probabilities should be finite
        assert torch.all(torch.isfinite(log_p_x))

        # Log probabilities should be negative (0 < P(x) < 1)
        assert torch.all(log_p_x <= 0)

    def test_marginal_specific_class(self, pyjuice_model, test_data):
        """Test marginal query for specific class."""
        data_batch = test_data[:10]
        observed_mask = torch.ones_like(data_batch, dtype=torch.bool)

        # Compute P(x, k=0)
        log_p_x_and_k0 = PyJuiceInference.marginal_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask,
            class_value=0,
        )

        # Compute P(x, k=1)
        log_p_x_and_k1 = PyJuiceInference.marginal_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask,
            class_value=1,
        )

        # Both should be valid
        assert log_p_x_and_k0.shape == (10,)
        assert log_p_x_and_k1.shape == (10,)
        assert torch.all(torch.isfinite(log_p_x_and_k0))
        assert torch.all(torch.isfinite(log_p_x_and_k1))

        # P(x) should be close to P(x, k=0) + P(x, k=1)
        log_p_x = PyJuiceInference.marginal_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask,
            class_value=None,
        )

        # log(a + b) = logsumexp([log(a), log(b)])
        log_p_x_reconstructed = torch.logsumexp(
            torch.stack([log_p_x_and_k0, log_p_x_and_k1], dim=0), dim=0
        )

        # Should match within numerical precision
        assert torch.allclose(log_p_x, log_p_x_reconstructed, atol=1e-5)

    def test_marginal_partial_evidence(self, pyjuice_model, test_data):
        """Test marginal query with partial evidence."""
        data_batch = test_data[:10]

        # Observe only 50% of features
        observed_mask = PyJuiceInference.create_random_mask(
            batch_size=10,
            num_features=196,
            observed_ratio=0.5,
            device=data_batch.device,
        )

        # Compute P(x) with partial evidence
        log_p_x = PyJuiceInference.marginal_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask,
            class_value=None,
        )

        assert log_p_x.shape == (10,)
        assert torch.all(torch.isfinite(log_p_x))


class TestPyJuiceInferenceConditional:
    """Test conditional query functionality."""

    def test_conditional_simple(self, pyjuice_model, test_data):
        """Test P(targets | observed) with no class conditioning."""
        data_batch = test_data[:10]

        # Split features: first 100 observed, last 96 targets
        observed_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        observed_mask[:, :100] = True

        target_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        target_mask[:, 100:] = True

        # Compute P(targets | observed)
        log_cond = PyJuiceInference.conditional_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            target_mask,
            observed_mask,
            class_value=None,
            class_as_target=False,
        )

        assert log_cond.shape == (10,)
        assert torch.all(torch.isfinite(log_cond))

    def test_conditional_with_class(self, pyjuice_model, test_data):
        """Test P(targets | observed, k) conditioned on class."""
        data_batch = test_data[:10]

        observed_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        observed_mask[:, :100] = True

        target_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        target_mask[:, 100:] = True

        # Compute P(targets | observed, k=0)
        log_cond_k0 = PyJuiceInference.conditional_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            target_mask,
            observed_mask,
            class_value=0,
            class_as_target=False,
        )

        # Compute P(targets | observed, k=1)
        log_cond_k1 = PyJuiceInference.conditional_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            target_mask,
            observed_mask,
            class_value=1,
            class_as_target=False,
        )

        assert log_cond_k0.shape == (10,)
        assert log_cond_k1.shape == (10,)
        assert torch.all(torch.isfinite(log_cond_k0))
        assert torch.all(torch.isfinite(log_cond_k1))

    def test_conditional_class_as_target(self, pyjuice_model, test_data):
        """Test P(k | observed) with class as target."""
        data_batch = test_data[:10]

        # Observe first 100 features
        observed_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        observed_mask[:, :100] = True

        # Target: remaining features
        target_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        target_mask[:, 100:] = True

        # Compute P(k, targets | observed) for all classes
        log_cond = PyJuiceInference.conditional_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            target_mask,
            observed_mask,
            class_value=None,
            class_as_target=True,
        )

        # Should return [batch, num_classes]
        assert log_cond.shape == (10, 2)
        assert torch.all(torch.isfinite(log_cond))

    def test_conditional_specific_class_as_target(self, pyjuice_model, test_data):
        """Test P(k=0, targets | observed) with specific class."""
        data_batch = test_data[:10]

        observed_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        observed_mask[:, :100] = True

        target_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        target_mask[:, 100:] = True

        # Compute P(k=0, targets | observed)
        log_cond_k0 = PyJuiceInference.conditional_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            target_mask,
            observed_mask,
            class_value=0,
            class_as_target=True,
        )

        assert log_cond_k0.shape == (10,)
        assert torch.all(torch.isfinite(log_cond_k0))


class TestPyJuiceInferenceClassification:
    """Test classification functionality."""

    def test_classify_full_evidence(self, pyjuice_model, test_data):
        """Test classification with all features observed."""
        data_batch = test_data[:100]

        # Classify with full evidence
        predictions, class_probs = PyJuiceInference.classify(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask=None,
        )

        # Check output shapes
        assert predictions.shape == (100,)
        assert class_probs.shape == (100, 2)

        # Predictions should be 0 or 1
        assert torch.all((predictions == 0) | (predictions == 1))

        # Probabilities should sum to 1
        prob_sums = class_probs.sum(dim=1)
        expected = torch.ones(100, device=data_batch.device)
        assert torch.allclose(prob_sums, expected, atol=1e-5)

        # Probabilities should be in [0, 1]
        assert torch.all(class_probs >= 0)
        assert torch.all(class_probs <= 1)

        # Predictions should match argmax of probabilities
        assert torch.all(predictions == torch.argmax(class_probs, dim=1))

    def test_classify_partial_evidence(self, pyjuice_model, test_data):
        """Test classification with partial evidence."""
        data_batch = test_data[:50]

        # Observe 70% of features
        observed_mask = PyJuiceInference.create_random_mask(
            batch_size=50,
            num_features=196,
            observed_ratio=0.7,
            device=data_batch.device,
        )

        # Classify with partial evidence
        predictions, class_probs = PyJuiceInference.classify(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask=observed_mask,
        )

        # Check outputs are valid
        assert predictions.shape == (50,)
        assert class_probs.shape == (50, 2)
        assert torch.all((predictions == 0) | (predictions == 1))

        prob_sums = class_probs.sum(dim=1)
        expected = torch.ones(50, device=data_batch.device)
        assert torch.allclose(prob_sums, expected, atol=1e-5)

    def test_classify_accuracy(self, pyjuice_model, test_data, test_labels):
        """Test classification accuracy on labeled test data."""
        # Use subset of test data
        data_batch = test_data[:500]
        labels_batch = test_labels[:500]

        predictions, class_probs = PyJuiceInference.classify(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask=None,
        )

        # Compute accuracy
        correct = (predictions == labels_batch).sum().item()
        accuracy = correct / len(labels_batch)

        print(f"\nClassification accuracy: {accuracy:.4f}")
        print(f"Correct: {correct}/{len(labels_batch)}")

        # Model should achieve reasonable accuracy (> 80%)
        assert accuracy > 0.80, f"Accuracy {accuracy:.4f} is below 80%"

        # Check that model makes confident predictions
        max_probs = class_probs.max(dim=1).values
        confident_predictions = (max_probs > 0.9).sum()
        conf_ratio = f"{confident_predictions}/{len(data_batch)}"
        print(f"Confident predictions (>0.9): {conf_ratio}")

        # At least some predictions should be confident
        assert confident_predictions > 0

        # Model should predict both classes
        unique_preds = torch.unique(predictions)
        assert len(unique_preds) == 2, f"Expected 2 classes, got {len(unique_preds)}"

        # Check per-class accuracy
        for class_id in [0, 1]:
            class_mask = labels_batch == class_id
            if class_mask.sum() > 0:
                class_correct = (predictions[class_mask] == class_id).sum().item()
                class_total = class_mask.sum().item()
                class_acc = class_correct / class_total
                print(f"Class {class_id} accuracy: {class_acc:.4f}")
                print(f"  Correct: {class_correct}/{class_total}")
                assert class_acc > 0.7, f"Class {class_id} accuracy too low"


class TestPyJuiceInferenceUtilities:
    """Test utility functions."""

    def test_create_random_mask(self, device):
        """Test random mask creation."""
        mask = PyJuiceInference.create_random_mask(
            batch_size=32,
            num_features=196,
            observed_ratio=0.5,
            device=device,
        )

        # Check shape and type
        assert mask.shape == (32, 196)
        assert mask.dtype == torch.bool

        # Check approximately 50% are True
        observed_fraction = mask.float().mean().item()
        assert abs(observed_fraction - 0.5) < 0.1  # Within 10% tolerance

    def test_create_random_mask_ratios(self, device):
        """Test different observation ratios."""
        for ratio in [0.1, 0.3, 0.7, 0.9]:
            mask = PyJuiceInference.create_random_mask(
                batch_size=100,
                num_features=196,
                observed_ratio=ratio,
                device=device,
            )

            observed_fraction = mask.float().mean().item()
            # Should be close to target ratio
            assert abs(observed_fraction - ratio) < 0.1


class TestPyJuiceInferenceIntegration:
    """Integration tests combining multiple operations."""

    def test_marginal_conditional_consistency(self, pyjuice_model, test_data):
        """Test that conditional query is consistent with marginal queries."""
        data_batch = test_data[:5]

        # Define observed and target features
        observed_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        observed_mask[:, :100] = True

        target_mask = torch.zeros_like(data_batch, dtype=torch.bool)
        target_mask[:, 100:] = True

        both_mask = observed_mask | target_mask

        # Method 1: Use conditional_query
        log_cond = PyJuiceInference.conditional_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            target_mask,
            observed_mask,
            class_value=None,
            class_as_target=False,
        )

        # Method 2: Compute manually using marginals
        # P(targets | observed) = P(targets, observed) / P(observed)
        log_p_both = PyJuiceInference.marginal_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            both_mask,
            class_value=None,
        )

        log_p_obs = PyJuiceInference.marginal_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask,
            class_value=None,
        )

        log_cond_manual = log_p_both - log_p_obs

        # Should match
        assert torch.allclose(log_cond, log_cond_manual, atol=1e-5)

    def test_classification_via_conditional(self, pyjuice_model, test_data):
        """Test that classification is consistent with conditional queries."""
        data_batch = test_data[:10]
        observed_mask = torch.ones_like(data_batch, dtype=torch.bool)

        # Method 1: Use classify
        predictions, class_probs = PyJuiceInference.classify(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            observed_mask=observed_mask,
        )

        # Method 2: Use conditional with class as target
        # P(k | observed) with empty target mask
        empty_target = torch.zeros_like(data_batch, dtype=torch.bool)

        log_class_probs = PyJuiceInference.conditional_query(
            pyjuice_model.pc_per_class,
            pyjuice_model.priors,
            data_batch,
            empty_target,
            observed_mask,
            class_value=None,
            class_as_target=True,
        )

        class_probs_conditional = torch.exp(log_class_probs)

        # Should match
        assert torch.allclose(class_probs, class_probs_conditional, atol=1e-5)

        # Predictions should match
        predictions_conditional = torch.argmax(class_probs_conditional, dim=1)
        assert torch.all(predictions == predictions_conditional)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
