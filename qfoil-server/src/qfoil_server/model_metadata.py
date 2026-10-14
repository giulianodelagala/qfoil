"""Public, transport-safe descriptions of the model lifecycle.

These types never carry a filesystem path, a Python object, or circuit
parameters -- only what a client is allowed to see over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

ModelStatus = Literal["unloaded", "loading", "ready", "unloading", "error"]


@dataclass(frozen=True)
class ModelSummary:
    """One catalog entry, without its filesystem path."""

    id: str
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "backend": self.backend}


@dataclass(frozen=True)
class ModelMetadata:
    """Publicly visible lifecycle state of the single active model slot."""

    status: ModelStatus
    id: Optional[str] = None
    backend: Optional[str] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    device: Optional[str] = None
    loaded_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "id": self.id,
            "backend": self.backend,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "device": self.device,
            "loaded_at": self.loaded_at,
        }


__all__ = ["ModelStatus", "ModelSummary", "ModelMetadata"]
