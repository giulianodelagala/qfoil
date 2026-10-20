"""Configuration discovery and per-request search option contract."""

import pytest
from types import SimpleNamespace

from qfoil import QFOIL

from qfoil_server.config import (
    DEFAULT_EVALUATION_BUDGET,
    DEFAULT_INNER_BUDGET,
    DEFAULT_POOL_SIZE,
    DEFAULT_SEARCH_BUDGET,
    DEFAULT_SIGMA_SEARCH_BUDGET,
    MAX_EVALUATION_BUDGET,
    MAX_SIGMA_SEARCH_BUDGET,
    MAX_INNER_BUDGET,
    MAX_POOL_SIZE,
    MAX_SEARCH_BUDGET,
)
from test_api import request


DEFAULTS = {
    "budget": DEFAULT_SEARCH_BUDGET,
    "inner_budget": DEFAULT_INNER_BUDGET,
    "pool_size": DEFAULT_POOL_SIZE,
}
MAXIMUMS = {
    "budget": MAX_SEARCH_BUDGET,
    "inner_budget": MAX_INNER_BUDGET,
    "pool_size": MAX_POOL_SIZE,
}


@pytest.fixture(autouse=True)
def capture_search_options(monkeypatch):
    """Check HTTP forwarding without spending the maximum search budgets."""
    def capture(self, expression, var, context, **options):
        assert context == {"x": (1, 0, 1)}
        found = options["budget"] > 0
        return SimpleNamespace(to_dict=lambda: {
            "status": "found" if found else "inconclusive",
            "variable": var,
            "witness": [1, 0, 1] if found else None,
            "verified": found,
            "seed_used": options["seed"],
            "search": {key: options[key] for key in DEFAULTS},
        })

    monkeypatch.setattr(QFOIL, "search_witness", capture)


def search(options=None):
    body = {
        "expression": "FULL(z) * (z <<= x)",
        "variable": "z",
        "bindings": {"x": [1, 0, 1]},
    }
    if options is not None:
        body["options"] = options
    return request("POST", "/v1/witnesses/search", json=body)


def test_config_exposes_central_defaults_and_limits_without_a_model():
    response = request("GET", "/v1/config")
    assert response.status_code == 200
    assert response.json() == {
        "evaluation": {
            "defaults": {"mode": "heuristic", "sigma_search_budget": 64, "evaluation_budget": None, "rng_seed": None, "seed": None},
            "modes": ["heuristic", "exact"],
            "max_sigma_search_budget": MAX_SIGMA_SEARCH_BUDGET,
            "max_evaluation_budget": MAX_EVALUATION_BUDGET,
        },
        "search": {"defaults": DEFAULTS, "max": MAXIMUMS},
        "evaluation_budget": DEFAULT_EVALUATION_BUDGET,
        "sigma_search_budget": DEFAULT_SIGMA_SEARCH_BUDGET,
    }


@pytest.mark.parametrize("options", [None, {}, {"pool_size": 3}])
def test_omitted_and_partial_options_report_effective_defaults(options):
    response = search(options)
    assert response.status_code == 200
    assert response.json()["status"] == "found"
    assert response.json()["search"] == {**DEFAULTS, **(options or {})}


@pytest.mark.parametrize("options", [
    {"budget": 100, "inner_budget": 2, "pool_size": 3},
    {"budget": 0, "inner_budget": 0, "pool_size": 1},
    MAXIMUMS,
])
def test_explicit_options_and_inclusive_limits_are_reported(options):
    response = search(options)
    assert response.status_code == 200
    assert response.json()["search"] == options
    if options["budget"] == 0:
        assert response.json()["status"] == "inconclusive"
    # A subsequent request still uses defaults, without session configuration.
    assert search().json()["search"] == DEFAULTS


@pytest.mark.parametrize("field,value", [
    ("budget", -1), ("budget", MAX_SEARCH_BUDGET + 1),
    ("inner_budget", -1), ("inner_budget", MAX_INNER_BUDGET + 1),
    ("pool_size", 0), ("pool_size", MAX_POOL_SIZE + 1),
    ("evaluation_budget", 1), ("sigma_search_budget", 1),
])
def test_invalid_or_read_only_options_are_rejected(field, value):
    assert search({field: value}).status_code == 422


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_config_has_no_update_endpoint(method):
    assert request(method, "/v1/config", json={"evaluation_budget": 1}).status_code == 405


@pytest.mark.parametrize("endpoint", ["evaluate", "search"])
@pytest.mark.parametrize("field", ["evaluation_budget", "sigma_search_budget"])
def test_read_only_budgets_are_rejected_at_request_top_level(endpoint, field):
    path = "/v1/expressions/evaluate" if endpoint == "evaluate" else "/v1/witnesses/search"
    body = {"expression": "FULL(z)", "bindings": {}, field: 1}
    if endpoint == "search":
        body["variable"] = "z"
    assert request("POST", path, json=body).status_code == 422
