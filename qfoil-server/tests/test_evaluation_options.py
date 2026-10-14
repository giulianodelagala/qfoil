"""Per-request evaluation modes and random initialization."""
import asyncio
import httpx
import pytest
from qfoil import QFOIL
from qfoil_server import create_app
from qfoil_server.model_manager import ModelManager


def evaluate(app, expression, options=None):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            body = {"expression": expression}
            if options is not None:
                body["options"] = options
            return await client.post("/v1/expressions/evaluate", json=body)
    return asyncio.run(send())


@pytest.fixture
def runtime_app():
    runtime = QFOIL(instance_len=3, evaluation_budget=200000)
    return runtime, create_app(model_manager=ModelManager(default_runtime=runtime))


def test_exact_count_and_following_default_do_not_share_mode(runtime_app):
    runtime, app = runtime_app
    exact = evaluate(app, "SIGMA z. FULL(z)", {"mode": "exact"})
    assert exact.status_code == 200
    assert exact.json()["value"] == 8
    assert runtime.evaluator.sigma.mode == "heuristic"
    assert evaluate(app, "SIGMA z. FULL(z)").json()["status"] == "inconclusive"


def test_rng_reproducibility_and_initial_tuple(runtime_app):
    runtime, app = runtime_app
    expr = "(SIGMA z. FULL(z)) >= 1"
    options = {"rng_seed": 38, "sigma_search_budget": 100}
    first = evaluate(app, expr, options).json()
    evaluate(app, expr)
    assert evaluate(app, expr, options).json() == first
    assert first["status"] == "decided"
    assert runtime.evaluator.sigma.search_budget == 64
    seeded = evaluate(app, expr, {"seed": [1, 0, 1], "sigma_search_budget": 1}).json()
    assert seeded["last_witness"] == [1, 0, 1]


def test_exact_mode_inherits_global_budget():
    runtime = QFOIL(instance_len=3, evaluation_budget=5)
    app = create_app(model_manager=ModelManager(default_runtime=runtime))
    response = evaluate(app, "SIGMA z. FULL(z)", {"mode": "exact"})
    assert response.json()["reason"] == "evaluation_budget_exhausted"
    assert runtime.evaluator.sigma.mode == "heuristic"


def test_mode_restored_after_validation_failure(runtime_app):
    runtime, app = runtime_app
    rng = runtime.evaluator.sigma.search.rng
    response = evaluate(app, "missing", {"mode": "exact", "rng_seed": 7, "sigma_search_budget": 1})
    assert response.status_code == 422
    assert runtime.evaluator.sigma.mode == "heuristic"
    assert runtime.evaluator.sigma.search_budget == 64
    assert runtime.evaluator.sigma.search.rng is rng


def test_exact_mode_applies_to_nested_queries(runtime_app):
    _, app = runtime_app
    response = evaluate(app, "SIGMA z. (FULL(z) * ((SIGMA w. FULL(w)) == 8))", {"mode": "exact"})
    assert response.status_code == 200
    assert response.json()["value"] == 8


@pytest.mark.parametrize("options", [
    {"mode": "exhaustive"}, {"sigma_search_budget": -1},
    {"sigma_search_budget": 20000001}, {"sigma_search_budget": True},
    {"rng_seed": True}, {"rng_seed": 1.5},
    {"seed": [True, 0, 1]}, {"seed": [2, 0, 1]},
    {"seed": [0, 1]}, {"evaluation_budget": 20000001},
    {"evaluation_budget": -1}, {"evaluation_budget": True},
])
def test_invalid_options(runtime_app, options):
    _, app = runtime_app
    assert evaluate(app, "(SIGMA z. FULL(z)) >= 1", options).status_code == 422



@pytest.mark.parametrize("mode", ["heuristic", "exact"])
def test_per_request_total_budget_restores_runtime_default(mode):
    runtime = QFOIL(instance_len=2, evaluation_budget=1)
    app = create_app(model_manager=ModelManager(default_runtime=runtime))
    expression = "(SIGMA z. FULL(z)) >= 1"
    result = evaluate(app, expression, {
        "mode": mode, "evaluation_budget": 20_000_000,
        "sigma_search_budget": 20_000_000, "seed": [1, 0],
    })
    assert result.status_code == 200
    assert result.json()["status"] == "decided"
    assert result.json()["value"] == 1
    assert runtime.evaluator.budget.limit == 1
    exhausted = evaluate(app, expression, {"mode": mode})
    assert exhausted.json()["reason"] == "evaluation_budget_exhausted"


def test_request_budget_is_shared_by_nested_queries():
    runtime = QFOIL(instance_len=2, evaluation_budget=1000)
    app = create_app(model_manager=ModelManager(default_runtime=runtime))
    result = evaluate(app, "SIGMA y. ((SIGMA z. FULL(z)) == 4)", {
        "mode": "exact", "evaluation_budget": 5,
    })
    assert result.json()["reason"] == "evaluation_budget_exhausted"
    assert runtime.evaluator.budget.limit == 1000
