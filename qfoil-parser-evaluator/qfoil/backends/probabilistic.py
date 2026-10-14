"""Backend contract for QFOIL probabilistic operations."""

from typing import Collection, Optional, Protocol, runtime_checkable


@runtime_checkable
class ProbabilisticBackend(Protocol):
    """Probability service required by ``VAL``, ``COND``, ``EP`` and ``CEP``.

    Implementations own all representation-specific work such as tensor
    conversion, batching, device placement, and numerical conversions.
    """

    @property
    def num_features(self) -> int:
        """Number of feature positions accepted by the backend."""
        ...

    @property
    def class_ids(self) -> Collection[int]:
        """Class identifiers known by the probabilistic model."""
        ...

    def probability(
        self,
        instance: tuple,
        class_id: Optional[int] = None,
    ) -> float:
        """Return ``Pr(instance)`` or ``Pr(instance, Y=class_id)``."""
        ...

    def conditional(
        self,
        target: tuple,
        evidence: tuple,
        class_id: int,
    ) -> float:
        """Return ``Pr(target, Y=class_id | evidence)``."""
        ...
