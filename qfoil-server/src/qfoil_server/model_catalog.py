"""Static catalog of models available for on-demand loading.

Loading this module never imports Torch or PyJuice: it only knows how to
read and validate the catalog file's structure. Model backends are only
imported when a client actually requests a load.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATALOG_ENV_VAR = "QFOIL_MODEL_CATALOG"


@dataclass(frozen=True)
class ModelCatalogEntry:
    """One statically configured, not-yet-loaded model."""

    model_id: str
    backend: str
    path: str
    device: str
    min_batch_size: int


def load_catalog_from_env() -> dict[str, ModelCatalogEntry]:
    """Load the catalog named by ``QFOIL_MODEL_CATALOG``, or an empty one."""
    catalog_path = os.environ.get(CATALOG_ENV_VAR)
    if not catalog_path:
        return {}
    return load_catalog(catalog_path)


def load_catalog(catalog_path: str) -> dict[str, ModelCatalogEntry]:
    """Parse and validate a model catalog file without loading any model."""
    try:
        raw_text = Path(catalog_path).read_text()
    except OSError as exc:
        raise ValueError(f"cannot read model catalog at {catalog_path}: {exc}") from exc

    try:
        raw_catalog = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid model catalog JSON at {catalog_path}: {exc}"
        ) from exc

    if not isinstance(raw_catalog, dict):
        raise ValueError(f"model catalog at {catalog_path} must be a JSON object")

    return {
        model_id: _parse_entry(model_id, raw_entry, catalog_path)
        for model_id, raw_entry in raw_catalog.items()
    }


def _parse_entry(
    model_id: str, raw_entry: Any, catalog_path: str
) -> ModelCatalogEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError(
            f"model catalog entry {model_id!r} in {catalog_path} must be an object"
        )

    backend = raw_entry.get("backend")
    path = raw_entry.get("path")
    device = raw_entry.get("device", "cpu")
    min_batch_size = raw_entry.get("min_batch_size", 4)

    if not isinstance(backend, str) or not backend:
        raise ValueError(
            f"model catalog entry {model_id!r} is missing a valid 'backend'"
        )
    if not isinstance(path, str) or not path:
        raise ValueError(f"model catalog entry {model_id!r} is missing a valid 'path'")
    if not isinstance(device, str) or not device:
        raise ValueError(f"model catalog entry {model_id!r} has an invalid 'device'")
    if (
        isinstance(min_batch_size, bool)
        or not isinstance(min_batch_size, int)
        or min_batch_size <= 0
    ):
        raise ValueError(
            f"model catalog entry {model_id!r} has an invalid 'min_batch_size'"
        )

    return ModelCatalogEntry(
        model_id=model_id,
        backend=backend,
        path=path,
        device=device,
        min_batch_size=min_batch_size,
    )


__all__ = [
    "ModelCatalogEntry",
    "load_catalog",
    "load_catalog_from_env",
    "CATALOG_ENV_VAR",
]
