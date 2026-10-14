"""Lifecycle manager for the single optional model slot.

The manager always keeps a backend-independent QFOIL runtime available for
SIZE/FULL/LaTeX-style operations. Loading a model builds a second,
backend runtime out-of-line and only swaps it in once it is fully
built; unloading restores the backend-independent runtime. Exactly one
model may be active per process.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, TypeVar

from starlette.concurrency import run_in_threadpool

from qfoil import QFOIL

from .config import (
    DEFAULT_EVALUATION_BUDGET,
    DEFAULT_SIGMA_MODE,
    DEFAULT_SIGMA_SEARCH_BUDGET,
)
from .model_catalog import ModelCatalogEntry
from .model_metadata import ModelMetadata, ModelSummary
from .test_dataset import (
    TestDataset,
    load_pyjuice_test_dataset,
    load_random_forest_test_dataset,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
LoadedModel = (
    tuple[ModelMetadata, QFOIL] | tuple[ModelMetadata, QFOIL, TestDataset | None]
)
Loader = Callable[[ModelCatalogEntry], LoadedModel]


class ModelManagerError(Exception):
    """Base class for model lifecycle errors."""


class UnknownModelError(ModelManagerError):
    """The requested model_id is not in the catalog."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"unknown model: {model_id!r}")
        self.model_id = model_id


class ModelAlreadyLoadedError(ModelManagerError):
    """A model is already active; unload it before loading another."""

    def __init__(self, active_model_id: Optional[str]) -> None:
        super().__init__(f"a model is already loaded: {active_model_id!r}")
        self.active_model_id = active_model_id


class ModelBusyError(ModelManagerError):
    """Another load or unload is already in flight."""

    def __init__(self, status: str) -> None:
        super().__init__(f"model manager is busy: {status}")
        self.status = status


class BackendUnavailableError(ModelManagerError):
    """The requested backend is unsupported, or its dependencies are missing."""

    def __init__(self, backend: str) -> None:
        super().__init__(f"backend unavailable: {backend!r}")
        self.backend = backend


class ModelLoadError(ModelManagerError):
    """The backend rejected the model or failed to build a runtime for it."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"failed to load model: {model_id!r}")
        self.model_id = model_id


class ModelNotLoadedError(ModelManagerError):
    """No model is currently active."""

    def __init__(self) -> None:
        super().__init__("no model is loaded")


class TestDatasetUnavailableError(ModelManagerError):
    """The active model has no indexed test dataset."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"model {model_id!r} does not expose a test dataset")


class TestDatasetEmptyError(ModelManagerError):
    """The active model's test dataset has no samples."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"model {model_id!r} has an empty test dataset")


class TestDatasetIndexError(ModelManagerError):
    """A requested test dataset index is outside its stable range."""

    def __init__(self, index: int, dataset_size: int) -> None:
        super().__init__(f"test dataset index {index} is outside [0, {dataset_size})")
        self.index = index
        self.dataset_size = dataset_size


class TestDatasetSerializationError(ModelManagerError):
    """A sample or target cannot be represented by the JSON contract."""

    def __init__(self) -> None:
        super().__init__("test sample or expected target is not JSON serializable")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_pyjuice_runtime(entry: ModelCatalogEntry) -> LoadedModel:
    """Build a probabilistic QFOIL runtime for one catalog entry.

    PyJuice and Torch are imported here, not at module scope, so installing
    qfoil-server without the ``pyjuice`` extra never pulls either in.
    """
    if entry.backend != "pyjuice":
        raise BackendUnavailableError(entry.backend)

    try:
        import torch
        from qfoil.backends.pyjuice import PyJuiceModel, PyJuiceProbabilityBackend
    except ImportError as exc:
        raise BackendUnavailableError(entry.backend) from exc

    try:
        device = torch.device(entry.device)
        model = PyJuiceModel.load(entry.path, device=device)
        backend = PyJuiceProbabilityBackend(model, min_batch_size=entry.min_batch_size)
        runtime = QFOIL(
            probabilistic_backend=backend,
            instance_len=model.num_features,
            sigma_mode=DEFAULT_SIGMA_MODE,
            sigma_search_budget=DEFAULT_SIGMA_SEARCH_BUDGET,
            evaluation_budget=DEFAULT_EVALUATION_BUDGET,
        )
        test_dataset = load_pyjuice_test_dataset(entry.path)
    except Exception as exc:
        logger.exception("failed to load model %r", entry.model_id)
        raise ModelLoadError(entry.model_id) from exc

    metadata = ModelMetadata(
        status="ready",
        id=entry.model_id,
        backend=entry.backend,
        num_features=model.num_features,
        num_classes=model.num_classes,
        device=str(model.device),
        loaded_at=_utcnow_iso(),
    )
    return metadata, runtime, test_dataset


def _load_random_forest_runtime(entry: ModelCatalogEntry) -> LoadedModel:
    """Load a trusted local sklearn artifact only when RF is requested."""
    try:
        import joblib
        import sklearn  # noqa: F401  # Ensure the optional backend is installed.
        from qfoil.backends.random_forest import RandomForestOracle
    except ImportError as exc:
        raise BackendUnavailableError(entry.backend) from exc

    try:
        import json
        from pathlib import Path

        path = Path(entry.path)
        if entry.device != "cpu":
            raise ValueError("Random Forest requires device=cpu")
        manifest = json.loads((path / "metadata.json").read_text())
        backend = RandomForestOracle(joblib.load(path / "model.joblib"))
        if (
            manifest["backend"] != "random_forest"
            or manifest["num_features"] != backend.num_features
            or manifest["class_ids"] != list(backend.class_ids)
            or manifest["num_classes"] != len(backend.class_ids)
        ):
            raise ValueError("Random Forest metadata does not match the model")
        dataset = load_random_forest_test_dataset(
            entry.path, backend.num_features, backend.class_ids
        )
        runtime = QFOIL(
            oracle_backend=backend,
            instance_len=backend.num_features,
            sigma_mode=DEFAULT_SIGMA_MODE,
            sigma_search_budget=DEFAULT_SIGMA_SEARCH_BUDGET,
            evaluation_budget=DEFAULT_EVALUATION_BUDGET,
        )
    except Exception as exc:
        logger.exception("failed to load model %r", entry.model_id)
        raise ModelLoadError(entry.model_id) from exc

    return (
        ModelMetadata(
            status="ready",
            id=entry.model_id,
            backend=entry.backend,
            num_features=backend.num_features,
            num_classes=len(backend.class_ids),
            device="cpu",
            loaded_at=_utcnow_iso(),
        ),
        runtime,
        dataset,
    )


def _load_runtime(entry: ModelCatalogEntry) -> LoadedModel:
    if entry.backend == "pyjuice":
        return _load_pyjuice_runtime(entry)
    if entry.backend == "random_forest":
        return _load_random_forest_runtime(entry)
    raise BackendUnavailableError(entry.backend)


class ModelManager:
    """Own the catalog, lifecycle state, and active QFOIL runtime.

    ``execute`` serializes calls against the currently active runtime
    (evaluate, search_witness) because QFOIL's evaluator keeps mutable
    internal state and is not safe under concurrent use. ``execute_latex``
    serializes calls against a separate, never-swapped backend-independent
    runtime so LaTeX rendering keeps working during a load or unload and
    never contends with it. ``load``/``unload`` guard only the bookkeeping
    around the runtime swap; the slow part of a load runs outside that
    section so a concurrent load/unload attempt observes the in-flight
    state and fails fast instead of queueing behind it.
    """

    def __init__(
        self,
        *,
        catalog: Optional[Mapping[str, ModelCatalogEntry]] = None,
        default_runtime: Optional[QFOIL] = None,
        loader: Optional[Loader] = None,
    ) -> None:
        self._catalog: dict[str, ModelCatalogEntry] = dict(catalog or {})
        self._default_runtime = default_runtime or QFOIL(
            sigma_mode=DEFAULT_SIGMA_MODE,
            sigma_search_budget=DEFAULT_SIGMA_SEARCH_BUDGET,
            evaluation_budget=DEFAULT_EVALUATION_BUDGET,
        )
        self._latex_runtime = QFOIL()
        self._loader: Loader = loader or _load_runtime
        self._runtime: QFOIL = self._default_runtime
        self._test_dataset: TestDataset | None = None
        self._metadata = ModelMetadata(status="unloaded")
        self._state_lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()
        self._latex_lock = asyncio.Lock()

    async def available(self) -> list[ModelSummary]:
        """List the catalog without loading anything."""
        return [
            ModelSummary(id=entry.model_id, backend=entry.backend)
            for entry in self._catalog.values()
        ]

    async def current(self) -> ModelMetadata:
        """Return an immutable snapshot of the active model's metadata."""
        return self._metadata

    async def load(self, model_id: str) -> ModelMetadata:
        """Load one catalog model, replacing the active runtime atomically."""
        entry = self._catalog.get(model_id)
        if entry is None:
            raise UnknownModelError(model_id)

        async with self._state_lock:
            status = self._metadata.status
            if status == "ready":
                raise ModelAlreadyLoadedError(self._metadata.id)
            if status != "unloaded":
                raise ModelBusyError(status)
            self._metadata = ModelMetadata(
                status="loading", id=model_id, backend=entry.backend
            )

        try:
            loaded = await run_in_threadpool(self._loader, entry)
            metadata, runtime = loaded[0], loaded[1]
            test_dataset = loaded[2] if len(loaded) == 3 else None
        except ModelManagerError:
            async with self._state_lock:
                self._metadata = ModelMetadata(status="unloaded")
            raise
        except Exception as exc:
            async with self._state_lock:
                self._metadata = ModelMetadata(status="unloaded")
            raise ModelLoadError(model_id) from exc

        async with self._state_lock:
            self._runtime = runtime
            self._test_dataset = test_dataset
            self._metadata = metadata
        return metadata

    async def unload(self) -> ModelMetadata:
        """Restore the backend-independent runtime. Idempotent."""
        async with self._state_lock:
            status = self._metadata.status
            if status == "unloaded":
                return self._metadata
            if status != "ready":
                raise ModelBusyError(status)
            self._runtime = self._default_runtime
            self._test_dataset = None
            self._metadata = ModelMetadata(status="unloaded")
            return self._metadata

    async def test_sample(
        self, *, index: int | None = None, seed: int | None = None
    ) -> tuple[str, int, int, Any, Any]:
        """Read one item from the active model's test split without mutation."""
        async with self._execution_lock:
            metadata = self._metadata
            dataset = self._test_dataset
            if metadata.status != "ready" or metadata.id is None:
                raise ModelNotLoadedError()
            if dataset is None:
                raise TestDatasetUnavailableError(metadata.id)

            dataset_size = len(dataset)
            if dataset_size == 0:
                raise TestDatasetEmptyError(metadata.id)
            selected = (
                random.SystemRandom().randrange(dataset_size)
                if index is None and seed is None
                else random.Random(seed).randrange(dataset_size)
                if index is None
                else index
            )
            if selected < 0 or selected >= dataset_size:
                raise TestDatasetIndexError(selected, dataset_size)
            # Providers are required to offer direct indexed access; a single
            # tensor/list lookup is bounded and does not materialize the split.
            sample, expected = dataset[selected]
            return metadata.id, selected, dataset_size, sample, expected

    async def execute(self, operation: Callable[[QFOIL], T]) -> T:
        """Run ``operation`` against the active runtime, serialized."""
        async with self._execution_lock:
            runtime = self._runtime
            return await run_in_threadpool(operation, runtime)

    async def execute_latex(self, operation: Callable[[QFOIL], T]) -> T:
        """Run ``operation`` against the dedicated LaTeX runtime, serialized."""
        async with self._latex_lock:
            return await run_in_threadpool(operation, self._latex_runtime)


__all__ = [
    "ModelManager",
    "ModelManagerError",
    "UnknownModelError",
    "ModelAlreadyLoadedError",
    "ModelBusyError",
    "BackendUnavailableError",
    "ModelLoadError",
    "ModelNotLoadedError",
    "TestDatasetUnavailableError",
    "TestDatasetEmptyError",
    "TestDatasetIndexError",
    "TestDatasetSerializationError",
]
