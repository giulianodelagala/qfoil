"""Backend-neutral access to one model's indexed test split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TestDataset(Protocol):
    """Minimal interface required by the read-only sample API."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Any, Any]: ...


@dataclass(frozen=True)
class PairedTestDataset:
    """Index aligned sample and target collections."""

    samples: Any
    targets: Any

    def __post_init__(self) -> None:
        if len(self.samples) != len(self.targets):
            raise ValueError("test samples and targets have different lengths")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.samples[index], self.targets[index]


def load_pyjuice_test_dataset(model_dir: str) -> TestDataset | None:
    """Load the optional tensor test split stored beside a PyJuice model.

    Existing QFOIL model artifacts use a model-specific data filename and the
    common ``test_targets.pt`` target filename. Missing either file means the
    model does not expose a complete test dataset.
    """
    import torch

    model_path = Path(model_dir)
    targets_path = model_path / "test_targets.pt"
    sample_paths = sorted(model_path.glob("*_test_data*.pt"))
    if len(sample_paths) != 1 or not targets_path.is_file():
        return None

    load_options = {"map_location": "cpu", "weights_only": True}
    try:
        samples = torch.load(sample_paths[0], mmap=True, **load_options)
        targets = torch.load(targets_path, mmap=True, **load_options)
    except (TypeError, RuntimeError):
        # ``mmap`` is unavailable for some older Torch/file-format pairs.
        samples = torch.load(sample_paths[0], **load_options)
        targets = torch.load(targets_path, **load_options)
    return PairedTestDataset(samples=samples, targets=targets)


def load_random_forest_test_dataset(
    model_dir: str, num_features: int, class_ids: tuple[int, ...]
) -> TestDataset | None:
    """Memory-map optional binary RF test arrays in their original order."""
    import numpy as np

    path = Path(model_dir)
    samples_path, targets_path = path / "test_samples.npy", path / "test_targets.npy"
    if not samples_path.exists() and not targets_path.exists():
        return None
    samples = np.load(samples_path, mmap_mode="r", allow_pickle=False)
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    if samples.ndim != 2 or samples.shape[1] != num_features or targets.ndim != 1:
        raise ValueError("Invalid Random Forest test dataset shape")
    if not np.isin(samples, [0, 1]).all() or not np.isin(targets, class_ids).all():
        raise ValueError("Invalid Random Forest test dataset values")
    return PairedTestDataset(samples, targets)


__all__ = [
    "PairedTestDataset",
    "TestDataset",
    "load_pyjuice_test_dataset",
    "load_random_forest_test_dataset",
]
