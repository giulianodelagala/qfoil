"""Reproduce the article's 16 exhaustive MUX contrastive-query cases.

Reference-set checks run on CPU; compiled PyJuice comparisons require CUDA.
"""

from itertools import product

import pytest

from examples.models.mux import enumerate_partial_assignments, mux_output
from tests.mux.pce_oracle import is_pce


INSTANCE = (0, 0, 1, 0, 0, 0, 0, 1)
CLASS_ID = mux_output(INSTANCE)
CASES = [
    pytest.param(
        (distribution, delta, release), id=f"{distribution}-{delta}-{release}"
    )
    for distribution, delta, release in product(
        ("uniform", "bernoulli"), (0.0, 0.75, 0.8, 1.0), (False, True)
    )
]


@pytest.fixture(scope="module", params=CASES)
def article_case(request):
    distribution_name, delta, require_release = request.param
    distribution = request.getfixturevalue(f"{distribution_name}_mux_distribution")
    candidates = tuple(enumerate_partial_assignments())
    # This oracle uses only the known input distribution and deterministic MUX
    # rule, not the parser, evaluator, tuple predicates, or compiled circuits.
    satisfying = {
        y
        for y in candidates
        if (not require_release or sum(v is not None for v in y) <= 7)
        and is_pce(distribution, INSTANCE, y, CLASS_ID, delta)
    }
    body = "FULL(x) * (y <<= x) * (CEP(y,c) >= delta)"
    if require_release:
        body += " * (SIZE(y) <= 7)"
    return {
        "distribution": distribution_name,
        "delta": delta,
        "require_release": require_release,
        "candidates": candidates,
        "satisfying": satisfying,
        "body": body,
        "bindings": {"x": INSTANCE, "c": CLASS_ID, "delta": delta},
    }


def test_article_reference_sets(article_case):
    """Check domain coverage, boundary counts, and independent noise bits."""
    case = article_case
    satisfying = case["satisfying"]
    assert CLASS_ID == 1
    assert len(case["candidates"]) == 3**8
    assert len(set(case["candidates"])) == 3**8
    assert all(
        all(v is None or v == original for v, original in zip(y, INSTANCE))
        for y in satisfying
    )
    if case["delta"] == 0:
        assert len(satisfying) == (255 if case["require_release"] else 256)
        assert (INSTANCE in satisfying) is (not case["require_release"])
    elif case["delta"] == 1:
        assert not satisfying
    else:
        assert 0 < len(satisfying) < 256
    # Releasing additional independent noise cannot change the posterior or
    # violate the upper size bound.
    assert all(y[:6] + (None, None) in satisfying for y in satisfying)


@pytest.mark.cuda
def test_article_pce_count_decision_and_witness(request, article_case):
    """Compare the actual quantified queries with the independent oracle."""
    case = article_case
    runtime, _ = request.getfixturevalue(f"{case['distribution']}_mux_evaluator")
    count_expr = f"SIGMA y. ({case['body']})"
    decision_expr = f"({count_expr}) >= 1"
    expected = case["satisfying"]

    # Exercise both exact execution policies, including the full ternary domain.
    for mode in ("restricted", "exact"):
        count = runtime.evaluate(count_expr, case["bindings"], sigma_mode=mode)
        assert count == len(expected), (mode, case["delta"], count, len(expected))
        decision = runtime.evaluate(decision_expr, case["bindings"], sigma_mode=mode)
        # Save the witness before another evaluation clears it.
        witness = runtime.evaluator.sigma.last_witness
        assert decision == int(bool(expected))
        if expected:
            assert witness in expected
            assert runtime.evaluate(
                case["body"], {**case["bindings"], "y": witness}, sigma_mode=mode
            ) == 1
        else:
            assert witness is None
