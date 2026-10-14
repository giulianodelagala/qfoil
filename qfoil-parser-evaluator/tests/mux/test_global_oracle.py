from itertools import product

import pytest

from examples.models.mux import (
    MuxDistribution,
    enumerate_partial_assignments,
    mux_output,
)

from .global_oracle import (
    all_completions_prob_ce,
    all_completions_prob_sr,
    completions,
    determined_assignment_count,
    is_determinant_feature_set,
    is_minimal_determinant_feature_set,
    is_probabilistic_determinant_feature_set,
    is_threshold_determined,
    minimum_probabilistic_determinant_feature_sets,
    most_probable_determinant_feature_sets,
    same_support_assignments,
    strict_subassignments,
    support_patterns,
    threshold_determined_mass_score,
)


@pytest.mark.parametrize(
    "distribution", [MuxDistribution.uniform(), MuxDistribution.bernoulli()]
)
def test_completions_are_full_and_compatible(distribution):
    partial = (0, 1, None, None, 1, None, None, None)
    actual = completions(distribution, partial)

    assert len(actual) == 32
    assert all(all(value is not None for value in item) for item in actual)
    assert all(distribution.matches(partial, item) for item in actual)


@pytest.mark.parametrize(
    "distribution", [MuxDistribution.uniform(), MuxDistribution.bernoulli()]
)
def test_all_completion_queries_match_mux_labels_exhaustively(distribution):
    delta = 0.9

    for partial in enumerate_partial_assignments():
        labels = [mux_output(item) for item in completions(distribution, partial)]

        for class_id in (0, 1):
            expected_sr = all(label == class_id for label in labels)
            expected_ce = all(label != class_id for label in labels)

            assert (
                all_completions_prob_sr(
                    distribution, partial, class_id, delta
                )
                is expected_sr
            )
            assert (
                all_completions_prob_ce(
                    distribution, partial, class_id, delta
                )
                is expected_ce
            )


def test_representative_guaranteed_and_ambiguous_evidence():
    distribution = MuxDistribution.bernoulli()
    guarantees_one = (0, 1, None, None, 1, None, None, None)
    guarantees_zero = (0, 1, None, None, 0, None, None, None)
    ambiguous = (0, 1, None, None, None, None, None, None)

    assert all_completions_prob_sr(distribution, guarantees_one, 1, 0.9)
    assert all_completions_prob_ce(distribution, guarantees_zero, 1, 0.9)
    assert not all_completions_prob_sr(distribution, ambiguous, 1, 0.9)
    assert not all_completions_prob_ce(distribution, ambiguous, 1, 0.9)


def test_threshold_determined_accepts_either_class():
    distribution = MuxDistribution.bernoulli()
    guarantees_one = (0, 1, None, None, 1, None, None, None)
    guarantees_zero = (0, 1, None, None, 0, None, None, None)
    ambiguous = (0, 1, None, None, None, None, None, None)

    assert is_threshold_determined(distribution, guarantees_one, 1, 0.9)
    assert is_threshold_determined(distribution, guarantees_zero, 1, 0.9)
    assert not is_threshold_determined(distribution, ambiguous, 1, 0.9)


def test_same_support_assignments_change_values_but_not_support():
    partial = (0, 1, None, None, 1, None, None, None)
    assignments = same_support_assignments(partial)

    assert len(assignments) == 8
    assert (1, 0, None, None, 0, None, None, None) in assignments
    assert all(
        tuple(value is None for value in item)
        == tuple(value is None for value in partial)
        for item in assignments
    )


def test_noise_features_are_optional_in_a_determinant_feature_set():
    distribution = MuxDistribution.bernoulli()
    determinant_supports = []

    for support in product((False, True), repeat=8):
        pattern = tuple(
            0 if included else None
            for included in support
        )
        if is_determinant_feature_set(distribution, pattern, 1, 0.9):
            determinant_supports.append(pattern)

    assert determinant_supports == [
        (0, 0, 0, 0, 0, 0, None, None),
        (0, 0, 0, 0, 0, 0, None, 0),
        (0, 0, 0, 0, 0, 0, 0, None),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ]


def test_a_determined_assignment_need_not_have_determinant_support():
    distribution = MuxDistribution.bernoulli()
    partial = (0, 1, None, None, 1, None, None, None)

    assert is_threshold_determined(distribution, partial, 1, 0.9)
    assert determined_assignment_count(distribution, partial, 1, 0.9) == 2
    assert not is_determinant_feature_set(distribution, partial, 1, 0.9)


def test_strict_subassignments_remove_at_least_one_defined_feature():
    candidate = (0, 1, None, None, 1, None, None, None)
    smaller = strict_subassignments(candidate)

    assert len(smaller) == 7
    assert (0, None, None, None, 1, None, None, None) in smaller
    assert candidate not in smaller


def test_functional_mux_support_is_the_unique_minimal_dfs():
    distribution = MuxDistribution.bernoulli()
    core_support = (0, 0, 0, 0, 0, 0, None, None)
    with_n0 = (0, 0, 0, 0, 0, 0, 0, None)
    with_n1 = (0, 0, 0, 0, 0, 0, None, 0)
    with_both_noise_features = (0, 0, 0, 0, 0, 0, 0, 0)

    assert is_minimal_determinant_feature_set(
        distribution, core_support, 1, 0.9
    )
    assert not is_minimal_determinant_feature_set(
        distribution, with_n0, 1, 0.9
    )
    assert not is_minimal_determinant_feature_set(
        distribution, with_n1, 1, 0.9
    )
    assert not is_minimal_determinant_feature_set(
        distribution, with_both_noise_features, 1, 0.9
    )


def test_dfs_score_is_probability_mass_not_assignment_count():
    distribution = MuxDistribution.bernoulli()
    support = (0, 0, 0, None, 0, None, None, None)

    score = threshold_determined_mass_score(
        distribution, support, 1, 0.9
    )

    assert score == pytest.approx(0.65)


def test_noise_features_do_not_change_dfs_score():
    distribution = MuxDistribution.bernoulli()
    without_noise = (0, 0, 0, None, 0, 0, None, None)
    with_noise = (0, 0, 0, None, 0, 0, 0, 0)

    score_without_noise = threshold_determined_mass_score(
        distribution, without_noise, 1, 0.9
    )
    score_with_noise = threshold_determined_mass_score(
        distribution, with_noise, 1, 0.9
    )

    assert score_without_noise == pytest.approx(0.8775)
    assert score_with_noise == pytest.approx(score_without_noise)


def test_prob_dfs_at_rho_one_matches_dfs_under_full_support():
    distribution = MuxDistribution.bernoulli()

    for support in support_patterns():
        assert is_probabilistic_determinant_feature_set(
            distribution, support, 1, 0.9, 1.0
        ) is is_determinant_feature_set(
            distribution, support, 1, 0.9
        )


def test_minimum_prob_dfs_and_mdfs_use_lexicographic_ranking():
    distribution = MuxDistribution.bernoulli()
    expected_minimum = {
        (0, 0, None, 0, 0, 0, None, None),
        (0, 0, 0, None, 0, 0, None, None),
        (0, 0, 0, 0, 0, None, None, None),
    }
    expected_most_probable = {
        (0, 0, 0, None, 0, 0, None, None),
    }

    minimum = set(
        minimum_probabilistic_determinant_feature_sets(
            distribution, 1, 0.9, 0.75
        )
    )
    most_probable = set(
        most_probable_determinant_feature_sets(
            distribution, 1, 0.9, 0.75
        )
    )

    assert minimum == expected_minimum
    assert most_probable == expected_most_probable
