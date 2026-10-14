"""Random Forest label oracle. Importing this module does not import sklearn."""

from numbers import Integral, Real


class RandomForestOracle:
    """Adapt a fitted binary-feature classifier without probability methods."""

    def __init__(self, model) -> None:
        self.model = model
        self.num_features = int(model.n_features_in_)
        if self.num_features <= 0:
            raise ValueError("num_features must be positive")
        labels = list(model.classes_)
        if not labels or any(
            isinstance(c, bool) or not isinstance(c, Integral) for c in labels
        ):
            raise ValueError("Random Forest requires integer class labels")
        self.class_ids = tuple(int(c) for c in labels)
        self.device = "cpu"

    def value(self, instance: tuple) -> int:
        if not isinstance(instance, tuple):
            raise ValueError("VAL's x argument must be a tuple")
        if len(instance) != self.num_features:
            raise ValueError(
                f"VAL tuple size {len(instance)} doesn't match model's num_features {self.num_features}"
            )
        if any(
            v is not None
            and (isinstance(v, bool) or not isinstance(v, Real) or v not in (0, 1))
            for v in instance
        ):
            raise ValueError("Random Forest attributes must be in {0,1,None}")
        # Validate every position before FULL: invalid partial inputs are rejected.
        if any(v is None for v in instance):
            return 0
        label = self.model.predict([instance])[0]
        if isinstance(label, bool) or not isinstance(label, Integral):
            raise ValueError("Random Forest prediction must be an integer label")
        return int(label)


__all__ = ["RandomForestOracle"]
