"""Small exhaustive oracle for probable contrastive explanations on the MUX."""

from math import isclose

from .psr_oracle import is_subset, subsets_of


def released_features(instance, candidate):
    """Count features present in ``instance`` but released in ``candidate``."""
    return sum(
        instance_value is not None and candidate_value is None
        for instance_value, candidate_value in zip(instance, candidate)
    )


def is_pce(distribution, instance, candidate, class_id, delta):
    return (
        all(value is not None for value in instance)
        and is_subset(candidate, instance)
        and 1.0 - distribution.posterior_class(candidate, class_id) >= delta
    )


def contrastive_explanations(
    distribution,
    instance,
    class_id,
    delta,
    *,
    min_released=1,
    max_released=None,
):
    """Return PCEs obtained by releasing features from ``instance``."""
    if max_released is None:
        max_released = len(instance)

    min_size = len(instance) - max_released
    max_size = len(instance) - min_released
    return [
        candidate
        for candidate in subsets_of(
            instance,
            min_size=max(0, min_size),
            max_size=max_size,
        )
        if is_pce(distribution, instance, candidate, class_id, delta)
    ]


def minimum_contrastive_explanations(
    distribution, instance, class_id, delta, *, max_released=None
):
    explanations = contrastive_explanations(
        distribution,
        instance,
        class_id,
        delta,
        max_released=max_released,
    )
    if not explanations:
        return []
    minimum_released = min(
        released_features(instance, explanation)
        for explanation in explanations
    )
    return [
        explanation
        for explanation in explanations
        if released_features(instance, explanation) == minimum_released
    ]


def most_contrastive_explanations(
    distribution, instance, class_id, delta, *, max_released=None
):
    """Return minimum PCEs with maximum contrastive probability (CEP)."""
    explanations = minimum_contrastive_explanations(
        distribution,
        instance,
        class_id,
        delta,
        max_released=max_released,
    )
    if not explanations:
        return []

    probabilities = [
        1.0 - distribution.posterior_class(explanation, class_id)
        for explanation in explanations
    ]
    maximum = max(probabilities)
    return [
        explanation
        for explanation, probability in zip(explanations, probabilities)
        if isclose(probability, maximum, rel_tol=1e-12, abs_tol=1e-12)
    ]
