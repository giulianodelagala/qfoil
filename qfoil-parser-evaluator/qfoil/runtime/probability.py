"""Numeric model functions: probability queries or a single-argument oracle."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..errors import UnsupportedOperationError

if TYPE_CHECKING:
    from ..backends.probabilistic import ProbabilisticBackend
    from ..backends.oracle import ValueOracle


class ProbabilityRuntime:
    """Validate, dispatch, and cache QFOIL probability queries.

    This component depends only on the probability and value-oracle contracts. It has
    no knowledge of AST evaluation, quantification, PyJuice, or tensors.
    """

    def __init__(
        self,
        backend: Optional[ProbabilisticBackend],
        cache_size: int = 10000,
        *,
        oracle_backend: Optional[ValueOracle] = None,
    ) -> None:
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, int)
            or cache_size < 0
        ):
            raise ValueError("prob_cache_size must be a non-negative integer")
        if backend is not None and oracle_backend is not None:
            raise ValueError("Supply only one active backend")
        self.oracle_backend = oracle_backend
        self.backend = backend
        self.cache: "OrderedDict[tuple, int | float]" = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def handlers(self) -> dict[str, Callable[[list[Any]], int | float]]:
        return {
            "VAL": self.val,
            "COND": self.cond,
            "EP": self.ep,
            "CEP": self.cep,
        }

    def require_backend(self, func_name: str) -> ProbabilisticBackend:
        if self.oracle_backend is not None:
            raise UnsupportedOperationError("Consulta no soportada por Random Forest")
        if self.backend is None:
            raise RuntimeError(
                f"{func_name} requires a probabilistic backend. Pass "
                "`probabilistic_backend=` when constructing QFOILEvaluator."
            )
        return self.backend

    def require_feature_length(self, value: tuple, func_name: str) -> None:
        backend = self.require_backend(func_name)
        if len(value) != backend.num_features:
            raise ValueError(
                f"{func_name} tuple size {len(value)} doesn't match model's "
                f"num_features {backend.num_features}"
            )

    def require_class_id(self, value: Any) -> int:
        backend = self.require_backend("Probability query")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Class id must be a number, got {value!r}")
        class_id = int(value)
        if class_id != value:
            raise ValueError(f"Class id must be an integer, got {value!r}")
        if class_id not in backend.class_ids:
            raise ValueError(
                f"Unknown class id {class_id}; model has classes "
                f"{sorted(backend.class_ids)}"
            )
        return class_id

    def cached(self, key: tuple, compute: Callable[[], int | float]) -> int | float:
        if self.cache_size == 0:
            return compute()
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        value = compute()
        self.cache[key] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def val(self, args: list[Any]) -> int | float:
        """PC: probabilities; oracle: backend-specific numeric VAL(x)."""
        if self.oracle_backend is not None:
            if len(args) not in (1, 2):
                raise ValueError("VAL requires 1 argument (x) or 2 arguments (x, c)")
            if len(args) == 2:
                raise UnsupportedOperationError("Consulta no soportada por Random Forest")
            x = args[0]
            if not isinstance(x, tuple):
                raise ValueError("VAL's x argument must be a tuple")
            return self.cached(("ORACLE_VAL", x), lambda: self.oracle_backend.value(x))
        backend = self.require_backend("VAL")
        if len(args) not in (1, 2):
            raise ValueError("VAL requires 1 argument (x) or 2 arguments (x, c)")
        x = args[0]
        if not isinstance(x, tuple):
            raise ValueError("VAL's x argument must be a tuple")
        self.require_feature_length(x, "VAL")
        class_id = self.require_class_id(args[1]) if len(args) == 2 else None
        return self.cached(
            ("VAL", x, class_id),
            lambda: backend.probability(x, class_id),
        )

    def cond(self, args: list[Any]) -> float:
        """COND(x, y, c) = Pr(x, Y=c | y)."""
        self.require_backend("COND")
        if len(args) != 3:
            raise ValueError("COND requires exactly 3 arguments (x, y, c)")
        x, y, class_value = args
        if not isinstance(x, tuple) or not isinstance(y, tuple):
            raise ValueError("COND's x and y arguments must be tuples")
        if len(x) != len(y):
            raise ValueError("COND requires x and y of equal length")
        self.require_feature_length(x, "COND")
        return self.conditional(x, y, self.require_class_id(class_value))

    def conditional(self, x: tuple, y: tuple, class_id: int) -> float:
        backend = self.require_backend("COND")
        return self.cached(
            ("COND", x, y, class_id),
            lambda: backend.conditional(x, y, class_id),
        )

    def ep(self, args: list[Any]) -> float:
        """EP(x, c) = Pr(Y=c | x)."""
        return self.class_posterior(args, complement=False)

    def cep(self, args: list[Any]) -> float:
        """CEP(x, c) = Pr(Y!=c | x)."""
        return self.class_posterior(args, complement=True)

    def class_posterior(
        self,
        args: list[Any],
        *,
        complement: bool,
    ) -> float:
        func_name = "CEP" if complement else "EP"
        self.require_backend(func_name)
        if len(args) != 2:
            raise ValueError(f"{func_name} requires exactly 2 arguments (x, c)")
        x, class_value = args
        if not isinstance(x, tuple):
            raise ValueError(f"{func_name}'s x argument must be a tuple")
        self.require_feature_length(x, func_name)
        probability = self.conditional(
            (None,) * len(x),
            x,
            self.require_class_id(class_value),
        )
        return 1.0 - probability if complement else probability


__all__ = ["ProbabilityRuntime"]
