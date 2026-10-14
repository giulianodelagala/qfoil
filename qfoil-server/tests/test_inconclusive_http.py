"""The transport preserves unknown evaluation rather than serializing a guess."""
import asyncio

import httpx
from qfoil import QFOIL, INCONCLUSIVE

from qfoil_server import create_app


def _post(app, path, json):
    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(path, json=json)

    return asyncio.run(send())


def test_search_verification_shares_global_budget_over_http():
    runtime = QFOIL(instance_len=2, evaluation_budget=2)
    app = create_app(qfoil=runtime)
    response = _post(app, "/v1/witnesses/search", {
        "expression": "FULL(y)", "variable": "y", "bindings": {},
        "options": {"budget": 0, "seed": [1, 1]},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "inconclusive"
    assert body["reason"] == "evaluation_budget_exhausted"
    assert body["witness"] is None and body["verified"] is False
    following = _post(app, "/v1/expressions/evaluate", {
        "expression": "SIZE(x)", "bindings": {"x": [1, 1]},
    })
    assert following.json()["status"] == "decided"
    assert following.json()["value"] == 2


def test_unknown_evaluation_has_null_value_and_no_witness():
    runtime = QFOIL()
    runtime.evaluate = lambda *args, **kwargs: INCONCLUSIVE
    app = create_app(qfoil=runtime)

    response = _post(app, "/v1/expressions/evaluate", {
        "expression": "FULL(x)", "bindings": {"x": [1, 1, 1]},
    })
    assert response.status_code == 200
    assert response.json() == {
        "status": "inconclusive", "value": None, "last_witness": None,
        "reason": "bounded_search_incomplete",
    }


def test_nested_sigma_evaluates_to_a_decided_value():
    """A nested SIGMA whose outer decision the heuristic can actually
    certify (search finds and verifies a witness) decides normally end to
    end, through the real heuristic runtime -- no mock, no 501."""

    app = create_app(qfoil=QFOIL(instance_len=2))
    response = _post(app, "/v1/expressions/evaluate", {
        "expression": "(SIGMA y. (FULL(y) * ((SIGMA z. FULL(z)) >= 1))) >= 1",
        "bindings": {},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "decided"
    assert body["value"] == 1


def test_uncertified_nested_sigma_is_inconclusive_not_501():
    """A nested decision the heuristic cannot certify through a verified
    witness (here: the inner comparison scores to an un-scoreable tuple
    equality, see qfoil.quantifiers.witness_search.SigmaSearch) reports
    `inconclusive`, end to end through the real heuristic runtime -- never
    `501 unsupported_operation`, and never a silently wrong decision."""

    app = create_app(qfoil=QFOIL(instance_len=3))
    response = _post(app, "/v1/expressions/evaluate", {
        "expression": "(SIGMA y. ((y == x) * ((SIGMA z. (z == x)) >= 1))) >= 1",
        "bindings": {"x": [1, 1, 1]},
    })
    assert response.status_code == 200
    assert response.json() == {
        "status": "inconclusive", "value": None, "last_witness": None,
        "reason": "bounded_search_incomplete",
    }


def test_global_evaluation_budget_exhaustion_is_reported_over_http():
    """`SIZE(x)` costs exactly 2 evaluation units (see
    qfoil.runtime.budget.EvaluationBudget): one for the call node, one for
    its argument. A runtime configured with a 1-unit global budget must
    report `evaluation_budget_exhausted` over HTTP, not a value."""

    app = create_app(qfoil=QFOIL(instance_len=2, evaluation_budget=1))
    response = _post(app, "/v1/expressions/evaluate", {
        "expression": "SIZE(x)", "bindings": {"x": [1, 0]},
    })
    assert response.status_code == 200
    assert response.json() == {
        "status": "inconclusive", "value": None, "last_witness": None,
        "reason": "evaluation_budget_exhausted",
    }


def test_witness_search_body_may_contain_sigma_and_reports_inconclusive():
    """The witness-search endpoint's body may itself contain SIGMA; when
    the heuristic cannot certify it, the response is `inconclusive`, not
    `501 unsupported_operation`."""

    app = create_app(qfoil=QFOIL(instance_len=3))
    response = _post(app, "/v1/witnesses/search", {
        "expression": "(SIGMA z. (z == x)) == 0", "variable": "y",
        "bindings": {"x": [1, 1, 1]},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "inconclusive"
    assert body["witness"] is None
    assert body["verified"] is False
    assert body["reason"] == "bounded_search_incomplete"
