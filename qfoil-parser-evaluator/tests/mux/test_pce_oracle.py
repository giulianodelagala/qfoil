from examples.models.mux import MuxDistribution, mux_output

from .pce_oracle import (
    contrastive_explanations,
    minimum_contrastive_explanations,
    most_contrastive_explanations,
    released_features,
)

INSTANCE = (0, 0, 1, 0, 0, 0, 0, 1)
CLASS_ID = mux_output(INSTANCE)

RELEASE_S0_D0 = (None, 0, None, 0, 0, 0, 0, 1)
RELEASE_S1_D0 = (0, None, None, 0, 0, 0, 0, 1)
RELEASE_S0_S1 = (None, None, 1, 0, 0, 0, 0, 1)
RELEASE_D0 = (0, 0, None, 0, 0, 0, 0, 1)


def test_released_features_counts_only_removed_values():
    assert released_features(INSTANCE, INSTANCE) == 0
    assert released_features(INSTANCE, RELEASE_D0) == 1
    assert released_features(INSTANCE, (None,) * 8) == 8


def test_uniform_oracle_has_three_tied_two_feature_explanations():
    distribution = MuxDistribution.uniform()
    minimum = minimum_contrastive_explanations(
        distribution, INSTANCE, CLASS_ID, 0.75
    )
    most_contrastive = most_contrastive_explanations(
        distribution, INSTANCE, CLASS_ID, 0.75
    )

    expected = {RELEASE_S0_D0, RELEASE_S1_D0, RELEASE_S0_S1}
    assert set(minimum) == expected
    assert set(most_contrastive) == expected


def test_high_threshold_bernoulli_oracle_excludes_noise():
    distribution = MuxDistribution.bernoulli()
    explanations = contrastive_explanations(
        distribution, INSTANCE, CLASS_ID, 0.8, max_released=8
    )
    minimum = minimum_contrastive_explanations(
        distribution, INSTANCE, CLASS_ID, 0.8, max_released=8
    )
    most_contrastive = most_contrastive_explanations(
        distribution, INSTANCE, CLASS_ID, 0.8, max_released=8
    )

    assert len(explanations) > len(minimum)
    assert minimum == [RELEASE_D0]
    assert most_contrastive == [RELEASE_D0]
