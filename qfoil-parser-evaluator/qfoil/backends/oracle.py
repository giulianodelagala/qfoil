"""Minimal numeric VAL service, independent of probability semantics."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ValueOracle(Protocol):
    @property
    def num_features(self) -> int: ...

    def value(self, instance: tuple) -> int | float:
        """Return the backend's numeric interpretation of VAL(instance)."""
        ...
