"""Exhaustive oracle for global explanation queries on the MUX."""

from itertools import combinations, product
from math import isclose, prod

from examples.models.mux import mux_output


def completions(distribution, partial):
    """Return every complete MUX input compatible with ``partial``."""
    assignments = []
    for values in product((0, 1), repeat=sum(value is None for value in partial)):
        value_iterator = iter(values)
        assignments.append(
            tuple(
                next(value_iterator) if value is None else value
                for value in partial
            )
        )
    return assignments


def all_completions_prob_sr(
    distribution, partial, class_id, delta
):
    """Whether every completion predicts ``class_id`` above ``delta``."""
    return all(
        float(mux_output(assignment) == class_id) >= delta
        for assignment in completions(distribution, partial)
    )


def all_completions_prob_ce(
    distribution, partial, class_id, delta
):
    """Whether every completion predicts a different class above ``delta``."""
    return all(
        float(mux_output(assignment) != class_id) >= delta
        for assignment in completions(distribution, partial)
    )


def same_support_assignments(partial):
    """Return all partial assignments with the same defined positions."""
    defined = [value is not None for value in partial]
    assignments = []
    for values in product((0, 1), repeat=sum(defined)):
        value_iterator = iter(values)
        assignments.append(
            tuple(next(value_iterator) if keep else None for keep in defined)
        )
    return assignments


def is_threshold_determined(
    distribution, partial, class_id, delta
):
    """Whether all completions agree for or against ``class_id``."""
    return all_completions_prob_sr(
        distribution, partial, class_id, delta
    ) or all_completions_prob_ce(distribution, partial, class_id, delta)


def determined_assignment_count(
    distribution, partial, class_id, delta
):
    """Count threshold-determined assignments sharing ``partial``'s support."""
    return sum(
        is_threshold_determined(distribution, item, class_id, delta)
        for item in same_support_assignments(partial)
    )


def is_determinant_feature_set(
    distribution, partial, class_id, delta
):
    """Whether every assignment with this support is threshold-determined."""
    assignments = same_support_assignments(partial)
    return determined_assignment_count(
        distribution, partial, class_id, delta
    ) == len(assignments)


def strict_subassignments(partial):
    """Return every strict subassignment contained in ``partial``."""
    defined_positions = [
        index for index, value in enumerate(partial) if value is not None
    ]
    assignments = []
    for size in range(len(defined_positions)):
        for positions in combinations(defined_positions, size):
            kept = set(positions)
            assignments.append(
                tuple(
                    value if index in kept else None
                    for index, value in enumerate(partial)
                )
            )
    return assignments


def is_minimal_determinant_feature_set(
    distribution, partial, class_id, delta
):
    """Whether ``partial`` is a DFS with no strict DFS subassignment."""
    if not is_determinant_feature_set(
        distribution, partial, class_id, delta
    ):
        return False
    return all(
        not is_determinant_feature_set(
            distribution, smaller, class_id, delta
        )
        for smaller in strict_subassignments(partial)
    )


def threshold_determined_mass_score(
    distribution, partial, class_id, delta
):
    """Return the probability mass of determined same-support assignments."""
    return sum(
        prod(
            probability if value else 1.0 - probability
            for value, probability in zip(
                assignment, distribution.feature_probs
            )
            if value is not None
        )
        for assignment in same_support_assignments(partial)
        if is_threshold_determined(
            distribution, assignment, class_id, delta
        )
    )


def is_probabilistic_determinant_feature_set(
    distribution, partial, class_id, delta, rho
):
    """Whether the threshold-determined mass is at least ``rho``."""
    score = threshold_determined_mass_score(
        distribution, partial, class_id, delta
    )
    return score >= rho or isclose(
        score, rho, rel_tol=1e-12, abs_tol=1e-12
    )


def support_patterns():
    """Return one canonical partial assignment for every feature support."""
    return [
        tuple(0 if included else None for included in support)
        for support in product((False, True), repeat=8)
    ]


def minimum_probabilistic_determinant_feature_sets(
    distribution, class_id, delta, rho
):
    """Return all ProbDFS supports having minimum cardinality."""
    candidates = [
        partial
        for partial in support_patterns()
        if is_probabilistic_determinant_feature_set(
            distribution, partial, class_id, delta, rho
        )
    ]
    if not candidates:
        return []
    minimum_size = min(
        sum(value is not None for value in partial)
        for partial in candidates
    )
    return [
        partial
        for partial in candidates
        if sum(value is not None for value in partial) == minimum_size
    ]


def most_probable_determinant_feature_sets(
    distribution, class_id, delta, rho
):
    """Return minimum ProbDFS supports with maximum DFSScore."""
    candidates = minimum_probabilistic_determinant_feature_sets(
        distribution, class_id, delta, rho
    )
    if not candidates:
        return []
    scores = [
        threshold_determined_mass_score(
            distribution, partial, class_id, delta
        )
        for partial in candidates
    ]
    maximum = max(scores)
    return [
        partial
        for partial, score in zip(candidates, scores)
        if isclose(score, maximum, rel_tol=1e-12, abs_tol=1e-12)
    ]
