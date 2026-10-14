"""PyJuice implementation of QFOIL's probabilistic backend contract."""

from math import exp
from typing import Collection, Optional

from .probabilistic import ProbabilisticBackend
from .pyjuice_inference import PyJuiceInference, PyJuiceModel


class PyJuiceProbabilityBackend(ProbabilisticBackend):
    """Adapt a :class:`PyJuiceModel` to QFOIL probability operations."""

    def __init__(self, model: PyJuiceModel, *, min_batch_size: int = 4) -> None:
        if min_batch_size <= 0:
            raise ValueError("min_batch_size must be positive")

        self.model = model
        self.device = model.device
        self.min_batch_size = min_batch_size
        self.inference = PyJuiceInference(device=str(model.device))

    @property
    def num_features(self) -> int:
        return self.model.num_features

    @property
    def class_ids(self) -> Collection[int]:
        return self.model.pc_per_class.keys()

    def probability(
        self,
        instance: tuple,
        class_id: Optional[int] = None,
    ) -> float:
        data = self._tuple_to_tensor(instance)
        observed_mask = self._tuple_to_mask(instance)

        try:
            log_probability = self._marginal_query(
                data,
                observed_mask,
                class_id=class_id,
            ).item()
            return self._from_log_probability(log_probability)
        except (ValueError, OverflowError):
            return 0.0

    def conditional(
        self,
        target: tuple,
        evidence: tuple,
        class_id: int,
    ) -> float:
        try:
            fused = self._fuse_tuples(evidence, target)
        except ValueError:
            return 0.0

        data = self._tuple_to_tensor(fused)
        target_mask = self._tuple_to_mask(target)
        observed_mask = self._tuple_to_mask(evidence)

        try:
            log_probability = self._conditional_query(
                data,
                target_mask,
                observed_mask,
                class_id=class_id,
            ).item()
            return self._from_log_probability(log_probability)
        except (ValueError, OverflowError):
            return 0.0

    def _tuple_to_tensor(self, value: tuple):
        import torch

        concrete = [item if item is not None else 0 for item in value]
        return torch.tensor([concrete], dtype=torch.long, device=self.device)

    def _tuple_to_mask(self, value: tuple):
        import torch

        observed = [item is not None for item in value]
        return torch.tensor([observed], dtype=torch.bool, device=self.device)

    def _marginal_query(self, data, observed_mask, *, class_id=None):
        original_size = data.shape[0]
        if original_size < self.min_batch_size:
            repeats = self.min_batch_size
            data = data.repeat(repeats, 1)
            observed_mask = observed_mask.repeat(repeats, 1)

        result = self.inference.marginal_query(
            self.model.pc_per_class,
            self.model.priors,
            data,
            observed_mask,
            class_value=class_id,
        )
        return result[:original_size]

    def _conditional_query(
        self,
        data,
        target_mask,
        observed_mask,
        *,
        class_id: int,
    ):
        original_size = data.shape[0]
        if original_size < self.min_batch_size:
            repeats = self.min_batch_size
            data = data.repeat(repeats, 1)
            target_mask = target_mask.repeat(repeats, 1)
            observed_mask = observed_mask.repeat(repeats, 1)

        result = self.inference.conditional_query(
            self.model.pc_per_class,
            self.model.priors,
            data,
            target_mask,
            observed_mask,
            class_value=class_id,
            class_as_target=True,
        )
        return result[:original_size]

    @staticmethod
    def _fuse_tuples(evidence: tuple, target: tuple) -> tuple:
        if len(evidence) != len(target):
            raise ValueError("Tuples must have same length")

        fused = []
        for index, (evidence_value, target_value) in enumerate(
            zip(evidence, target)
        ):
            if (
                evidence_value is not None
                and target_value is not None
                and evidence_value != target_value
            ):
                raise ValueError(
                    "Invalid fusion: overlapping features at position "
                    f"{index} have different values "
                    f"({evidence_value} vs {target_value})"
                )
            fused.append(target_value if target_value is not None else evidence_value)
        return tuple(fused)

    @staticmethod
    def _from_log_probability(log_probability: float) -> float:
        if log_probability == float("-inf"):
            return 0.0
        if log_probability == 0.0:
            return 1.0
        return exp(log_probability)


__all__ = [
    "PyJuiceInference",
    "PyJuiceModel",
    "PyJuiceProbabilityBackend",
]
