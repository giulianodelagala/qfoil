from examples.models.mux import MuxDistribution, mux_output

from .psr_oracle import (
    minimum_sufficient_reasons,
    most_probable_sufficient_reasons,
    sufficient_reasons,
)

INSTANCE = (0, 1, 0, 1, 1, 0, 0, 1)
CLASS_ID = mux_output(INSTANCE)


def test_uniform_oracle_has_expected_minimum_reasons():
    distribution = MuxDistribution.uniform()
    reasons = sufficient_reasons(distribution, INSTANCE, CLASS_ID, 0.75)
    minimum = minimum_sufficient_reasons(distribution, INSTANCE, CLASS_ID, 0.75)
    most_probable = most_probable_sufficient_reasons(
        distribution, INSTANCE, CLASS_ID, 0.75
    )

    assert len(reasons) == 68
    assert minimum == [
        (0, None, None, None, 1, None, None, None),
        (None, 1, None, None, 1, None, None, None),
        (None, None, None, 1, 1, None, None, None),
    ]
    assert most_probable == minimum


def test_high_threshold_bernoulli_oracle_excludes_noise():
    distribution = MuxDistribution.bernoulli()
    minimum = minimum_sufficient_reasons(distribution, INSTANCE, CLASS_ID, 0.9)
    most_probable = most_probable_sufficient_reasons(
        distribution, INSTANCE, CLASS_ID, 0.9
    )

    assert minimum == [
        (None, 1, None, None, 1, None, None, None),
    ]
    assert most_probable == minimum


def test_mpsr_ranks_two_reasons_above_the_class_prior():
    distribution = MuxDistribution.bernoulli()
    minimum = minimum_sufficient_reasons(distribution, INSTANCE, CLASS_ID, 0.65)
    most_probable = most_probable_sufficient_reasons(
        distribution, INSTANCE, CLASS_ID, 0.65
    )

    assert minimum == [
        (None, 1, None, None, None, None, None, None),
        (None, None, None, None, 1, None, None, None),
    ]
    assert most_probable == [
        (None, None, None, None, 1, None, None, None)
    ]
