"""Small exhaustive oracle for probable sufficient reasons on the MUX."""

from itertools import combinations
from math import isclose


def tuple_size(candidate):
    return sum(value is not None for value in candidate)


def subsets_of(instance, min_size=1, max_size=None):
    """Return the partial assignments contained in a complete instance."""
    if max_size is None:
        max_size = len(instance)

    candidates = []
    for size in range(min_size, max_size + 1):
        for positions in combinations(range(len(instance)), size):
            candidate = tuple(
                value if index in positions else None
                for index, value in enumerate(instance)
            )
            candidates.append(candidate)
    return candidates


def is_subset(candidate, instance):
    return all(
        value is None or value == instance[index]
        for index, value in enumerate(candidate)
    )


def is_psr(distribution, instance, candidate, class_id, delta):
    return (
        all(value is not None for value in instance)
        and is_subset(candidate, instance)
        and distribution.posterior_class(candidate, class_id) >= delta
    )


def sufficient_reasons(
    distribution,
    instance,
    class_id,
    delta,
    *,
    min_size=1,
    max_size=None,
):
    return [
        candidate
        for candidate in subsets_of(instance, min_size=min_size, max_size=max_size)
        if is_psr(distribution, instance, candidate, class_id, delta)
    ]


def minimum_sufficient_reasons(
    distribution, instance, class_id, delta, *, max_size=None
):
    reasons = sufficient_reasons(
        distribution,
        instance,
        class_id,
        delta,
        max_size=max_size,
    )
    if not reasons:
        return []
    minimum_size = min(tuple_size(reason) for reason in reasons)
    return [reason for reason in reasons if tuple_size(reason) == minimum_size]


def most_probable_sufficient_reasons(
    distribution, instance, class_id, delta, *, max_size=None
):
    reasons = minimum_sufficient_reasons(
        distribution,
        instance,
        class_id,
        delta,
        max_size=max_size,
    )
    if not reasons:
        return []
    probabilities = [distribution.evidence_probability(reason) for reason in reasons]
    maximum = max(probabilities)
    return [
        reason
        for reason, probability in zip(reasons, probabilities)
        if isclose(probability, maximum, rel_tol=1e-12, abs_tol=1e-12)
    ]
