"""Low-level PyJuice model storage and inference for QFOIL.

This module provides probabilistic circuit inference using PyJuice as a
replacement for Julia-based computations. It ports the inference functions
from examples/mnist_sandbox.ipynb and provides a clean interface for
probabilistic queries.

Phase 2 of Julia → Python migration.

Key classes:
- PyJuiceModel: Container for PC models, metadata, and test data
- PyJuiceInference: Static methods for probabilistic inference

Key functions:
- PyJuiceInference.marginal_query(): Compute P(X|evidence)
- PyJuiceInference.conditional_query(): Compute P(targets|observed)
- PyJuiceInference.classify(): Classification using Bayes rule
- PyJuiceInference.create_random_mask(): Generate random evidence masks
"""

from typing import Dict, Tuple, Optional, Any
import torch
import pyjuice as juice
import json
from pathlib import Path


class PyJuiceModel:
    """Container for PyJuice probabilistic circuit models with metadata.

    This class encapsulates:
    - Per-class PC models (split models)
    - Class priors and metadata
    - Load/save functionality
    """

    def __init__(
        self,
        pc_per_class: Dict[int, Any],
        priors: Dict[int, float],
        metadata: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """Initialize PyJuice model container.

        Args:
            pc_per_class: Dict mapping class_id -> compiled PC model
            priors: Dict mapping class_id -> prior probability P(k)
            metadata: Dict with model info (img_size, num_features, etc.)
            device: Device to use (auto-detect if None)
        """
        self.pc_per_class = pc_per_class
        self.priors = priors
        self.metadata = metadata

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Move models to device
        for pc in self.pc_per_class.values():
            pc.to(self.device)

    @property
    def num_classes(self) -> int:
        """Number of classes in the model."""
        return len(self.pc_per_class)

    @property
    def num_features(self) -> int:
        """Number of input features."""
        return self.metadata.get("num_features", None)

    @property
    def img_size(self) -> Optional[int]:
        """Image size (if applicable)."""
        return self.metadata.get("img_size", None)

    def save(self, model_dir: str):
        """Save model to directory.

        Creates directory structure:
        - model_dir/
          - metadata.json
          - class_0.jpc
          - class_1.jpc
          - ...

        Args:
            model_dir: Directory to save models
        """
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save metadata (includes priors)
        # Convert integer keys to strings for JSON compatibility
        priors_for_json = {str(k): v for k, v in self.priors.items()}
        metadata_with_priors = {
            **self.metadata,
            "priors": priors_for_json,
            "num_classes": self.num_classes,
        }
        with open(model_path / "metadata.json", "w") as f:
            json.dump(metadata_with_priors, f, indent=4)

        # Save per-class models
        for class_id, pc in self.pc_per_class.items():
            # Get the node structure before compilation
            # PyJuice models need to save the node structure
            model_file = model_path / f"class_{class_id}.jpc"
            juice.io.save(str(model_file), pc)

        print(f"✅ Model saved to {model_dir}")
        print(f"   - {self.num_classes} class models")
        print(f"   - Metadata: {model_path / 'metadata.json'}")

    @classmethod
    def load(
        cls,
        model_dir: str,
        device: Optional[torch.device] = None,
    ) -> "PyJuiceModel":
        """Load model from directory.

        Args:
            model_dir: Directory containing saved models
            device: Device to load models on (auto-detect if None)

        Returns:
            PyJuiceModel instance
        """
        model_path = Path(model_dir)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load metadata
        with open(model_path / "metadata.json", "r") as f:
            metadata_with_priors = json.load(f)

        # Extract priors from metadata and convert string keys to int
        priors_raw = metadata_with_priors.pop("priors")
        priors = {int(k): v for k, v in priors_raw.items()}
        num_classes = metadata_with_priors.pop("num_classes", len(priors))
        metadata = metadata_with_priors

        # Load per-class models
        pc_per_class = {}
        for class_id in sorted(priors):
            model_file = model_path / f"class_{class_id}.jpc"
            if not model_file.is_file():
                raise FileNotFoundError(f"Missing circuit for class {class_id}: {model_file}")
            if model_file.exists():
                # Load node structure
                loaded_ns = juice.io.load(str(model_file))
                # Compile to get executable PC
                pc = juice.compile(loaded_ns)
                pc.to(device)
                pc_per_class[class_id] = pc

        print(f"✅ Model loaded from {model_dir}")
        print(f"   - {len(pc_per_class)} class models")

        return cls(
            pc_per_class=pc_per_class,
            priors=priors,
            metadata=metadata,
            device=device,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PyJuiceModel(num_classes={self.num_classes}, "
            f"num_features={self.num_features})"
        )


class PyJuiceInference:
    """Low-level probabilistic-circuit inference using PyJuice."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the PyJuice inference engine.

        Args:
            device: Device to use ('cuda:0', 'cpu', etc). Auto-detect if None.
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @staticmethod
    def marginal_query(
        pc_per_class: Dict[int, Any],
        priors: Dict[int, float],
        data: torch.Tensor,
        observed_mask: torch.Tensor,
        class_value: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Marginal query with split PCs (one per class).

        Computes different query types based on class_value:
        - class_value=None: P(x) - marginalize over classes
        - class_value=k: P(x, k) - joint probability

        Args:
            pc_per_class: Dict {class_id: pc_model}
            priors: Dict {class_id: prior_probability}
            data: Input data [batch, features]
            observed_mask: Boolean mask [batch, features] (True = observed)
            class_value: None for marginalizing, int for specific class

        Returns:
            log_prob: Log probability [batch]
        """
        with torch.no_grad():
            # Convert observed_mask to missing_mask for PyJuice API
            missing_mask = ~observed_mask

            if class_value is None:
                # Case 1: P(x) = Σ_k P(x, k) = Σ_k P(x|k) × P(k)
                log_probs = []
                for k, pc in pc_per_class.items():
                    # P(x|k) - conditional on class k
                    log_p_x_given_k = juice.queries.marginal(
                        pc, data, missing_mask
                    ).squeeze()
                    # P(x, k) = P(x|k) × P(k)
                    log_p_x_and_k = log_p_x_given_k + torch.log(
                        torch.tensor(priors[k]).to(data.device)
                    )
                    log_probs.append(log_p_x_and_k)

                # P(x) = Σ_k P(x, k)
                log_p_x = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0)
                return log_p_x

            else:
                # Case 2: P(x, k) = P(x|k) × P(k)
                pc = pc_per_class[class_value]

                # P(x|k) - conditional on class
                log_p_x_given_k = juice.queries.marginal(
                    pc, data, missing_mask
                ).squeeze()

                log_p_x_and_k = log_p_x_given_k + torch.log(
                    torch.tensor(priors[class_value]).to(data.device)
                )
                return log_p_x_and_k

    @staticmethod
    def conditional_query(
        pc_per_class: Dict[int, Any],
        priors: Dict[int, float],
        data: torch.Tensor,
        target_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        class_value: Optional[int] = None,
        class_as_target: bool = False,
    ) -> torch.Tensor:
        """
        Classical conditional query: P(targets|observed) = P(targets, observed) / P(observed)
        Using ratio of marginals approach.

        Args:
            pc_per_class: Dict {class_id: pc_model}
            priors: Dict {class_id: prior_probability}
            data: Input data [batch, features]
            target_mask: Boolean mask [batch, features] (True = target variables)
            observed_mask: Boolean mask [batch, features] (True = observed variables)
            class_value:
                - If class_as_target=False: None for marginalizing, int for specific class
                - If class_as_target=True: None for all classes, int for specific class
            class_as_target: If True, class is a target variable to infer

        Returns:
            log_conditional: Log conditional probability
                - If class_as_target=False: [batch] - P(targets | observed)
                - If class_as_target=True and class_value=None: [batch, num_classes]
                - If class_as_target=True and class_value=k: [batch]
        """
        with torch.no_grad():
            # Create observed masks for marginal queries
            both_observed_mask = target_mask | observed_mask

            if class_as_target:
                # Class is a target variable: compute P(class, targets | observed)
                # Formula: P(k, targets | observed) = P(k, targets, observed) / P(observed)

                # Compute P(observed)
                log_p_obs = PyJuiceInference.marginal_query(
                    pc_per_class, priors, data, observed_mask, class_value=None
                )

                if class_value is None:
                    # Compute for ALL classes: return [batch, num_classes]
                    log_conditionals = []
                    class_ids = sorted(pc_per_class.keys())

                    for k in class_ids:
                        # P(targets, observed, k) = P(targets, observed | k) × P(k)
                        log_p_both_and_k = PyJuiceInference.marginal_query(
                            pc_per_class,
                            priors,
                            data,
                            both_observed_mask,
                            class_value=k,
                        )

                        # P(k, targets | observed) = P(targets, observed, k) / P(observed)
                        log_cond_k = log_p_both_and_k - log_p_obs
                        log_conditionals.append(log_cond_k)

                    # Stack: [batch, num_classes]
                    log_conditional = torch.stack(log_conditionals, dim=1)
                    return log_conditional

                else:
                    # Compute for SPECIFIC class: return [batch]
                    log_p_both_and_k = PyJuiceInference.marginal_query(
                        pc_per_class,
                        priors,
                        data,
                        both_observed_mask,
                        class_value=class_value,
                    )

                    # P(k, targets | observed) = P(targets, observed, k) / P(observed)
                    log_conditional = log_p_both_and_k - log_p_obs
                    return log_conditional

            else:
                # class is NOT a target
                if class_value is not None:
                    # P(targets | observed, k) = P(targets, observed, k) / P(observed, k)
                    log_p_both_and_k = PyJuiceInference.marginal_query(
                        pc_per_class,
                        priors,
                        data,
                        both_observed_mask,
                        class_value=class_value,
                    )

                    log_p_obs_and_k = PyJuiceInference.marginal_query(
                        pc_per_class,
                        priors,
                        data,
                        observed_mask,
                        class_value=class_value,
                    )

                    # P(targets | observed, k) = P(targets, observed, k) / P(observed, k)
                    log_conditional = log_p_both_and_k - log_p_obs_and_k

                else:
                    # P(targets | observed) = P(targets, observed) / P(observed)
                    log_p_both = PyJuiceInference.marginal_query(
                        pc_per_class, priors, data, both_observed_mask, class_value=None
                    )

                    log_p_obs = PyJuiceInference.marginal_query(
                        pc_per_class, priors, data, observed_mask, class_value=None
                    )

                    log_conditional = log_p_both - log_p_obs

                return log_conditional

    @staticmethod
    def classify(
        pc_per_class: Dict[int, Any],
        priors: Dict[int, float],
        data: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classification using split PCs via Bayes' rule.

        Args:
            pc_per_class: Dict {class_id: pc_model}
            priors: Dict {class_id: prior_probability}
            data: Input data [batch, features]
            observed_mask: Optional mask for partial evidence [batch, features]

        Returns:
            predictions: Predicted class labels [batch]
            class_probs: P(k | x) for each class [batch, num_classes]
        """
        with torch.no_grad():
            if observed_mask is None:
                # No missing data - observe everything
                observed_mask = torch.ones_like(data, dtype=torch.bool)

            # Convert to missing_mask for PyJuice API
            missing_mask = ~observed_mask

            # Compute log P(observed | k) for each class
            log_p_obs_given_k = {}
            for k, pc in pc_per_class.items():
                log_p_obs_given_k[k] = juice.queries.marginal(
                    pc, data=data, missing_mask=missing_mask
                ).squeeze()

            # Apply Bayes' rule: P(k | observed) = P(observed | k) × P(k) / P(observed)
            log_p_k_and_obs = []
            class_ids = sorted(pc_per_class.keys())

            for k in class_ids:
                log_p_and = log_p_obs_given_k[k] + torch.log(
                    torch.tensor(priors[k]).to(data.device)
                )
                log_p_k_and_obs.append(log_p_and)

            # Stack: [batch, num_classes]
            log_p_k_and_obs = torch.stack(log_p_k_and_obs, dim=1)

            # P(observed) = Σ_k P(observed, k)
            log_p_obs = torch.logsumexp(log_p_k_and_obs, dim=1, keepdim=True)

            # P(k | observed)
            log_posteriors = log_p_k_and_obs - log_p_obs
            class_probs = torch.exp(log_posteriors)

            # Predict class with highest posterior
            class_labels = torch.tensor(class_ids, device=data.device)
            predictions = class_labels[torch.argmax(class_probs, dim=1)]

            return predictions, class_probs

    @staticmethod
    def create_random_mask(
        batch_size: int,
        num_features: int,
        observed_ratio: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Create random observation masks for partial queries.

        Args:
            batch_size: Batch size
            num_features: Number of features
            observed_ratio: Fraction of features to observe (0-1)
            device: Device to create tensor on

        Returns:
            mask: Boolean mask [batch_size, num_features] (True = observed)
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mask = torch.rand(batch_size, num_features, device=device) < observed_ratio
        return mask
