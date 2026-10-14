"""HTTP contract tests for the qfoil-server proof of concept."""

import asyncio
from types import SimpleNamespace

import httpx

from qfoil_server import create_app


app = create_app()


def request(method: str, path: str, *, json=None) -> httpx.Response:
    """Exercise the ASGI app without opening a socket or worker thread."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=json)

    return asyncio.run(send())


def test_health():
    response = request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "qfoil-server",
        "model_status": "unloaded",
    }


def test_evaluate_converts_json_arrays_to_qfoil_tuples():
    response = request(
        "POST",
        "/v1/expressions/evaluate",
        json={
            "expression": "SIZE(x) <= limit",
            "bindings": {"x": [1, None, 0], "limit": 2},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "decided", "value": 1, "last_witness": None, "reason": None}


def test_evaluate_captures_value_and_witness_inside_one_execute_operation():
    class FakeRuntime:
        def __init__(self):
            self.evaluator = SimpleNamespace(
                sigma=SimpleNamespace(last_witness=(1, 1, 1))
            )

        def evaluate(self, expression, *, bindings, **options):
            assert self.evaluator.sigma.last_witness is None
            self.evaluator.sigma.last_witness = (None, 1, 0, None)
            return 1

    class BoundaryManager:
        def __init__(self):
            self.runtime = FakeRuntime()
            self.execute_calls = 0

        async def execute(self, operation):
            self.execute_calls += 1
            result = operation(self.runtime)
            self.runtime.evaluator.sigma.last_witness = (0, 0, 0, 0)
            return result

    manager = BoundaryManager()
    isolated_app = create_app(model_manager=manager)

    async def send():
        transport = httpx.ASGITransport(app=isolated_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/expressions/evaluate",
                json={"expression": "SIGMA x. predicate", "bindings": {}},
            )

    response = asyncio.run(send())

    assert response.status_code == 200
    assert response.json() == {
        "value": 1,
        "last_witness": [None, 1, 0, None],
        "status": "decided",
        "reason": None,
    }
    assert manager.execute_calls == 1


def test_render_latex():
    response = request(
        "POST",
        "/v1/expressions/latex",
        json={"expression": "SIZE(x) <= 2"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "latex": r"\operatorname{SIZE}\left(x\right) \leq 2"
    }


def test_search_for_verified_witness():
    response = request(
        "POST",
        "/v1/witnesses/search",
        json={
            "expression": "FULL(z) * (z <<< x)",
            "variable": "z",
            "bindings": {"x": [1, 0, 1]},
            "options": {"budget": 100, "pool_size": 3},
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "found"
    assert result["witness"] == [1, 0, 1]
    assert result["verified"] is True


def test_search_reports_inconclusive_when_budget_is_exhausted():
    response = request(
        "POST",
        "/v1/witnesses/search",
        json={
            "expression": "FULL(z) * (z <<< x)",
            "variable": "z",
            "bindings": {"x": [1, 0, 1]},
            "options": {"budget": 0, "pool_size": 3},
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "inconclusive"
    assert result["witness"] is None
    assert result["verified"] is False


def test_search_seed_length_mismatch_uses_public_error_envelope():
    response = request(
        "POST",
        "/v1/witnesses/search",
        json={
            "expression": "FULL(z) * (z <<< x)",
            "variable": "z",
            "bindings": {"x": [1, 0, 1]},
            "options": {"budget": 10, "pool_size": 3, "seed": [0]},
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "qfoil_validation_error"


def test_qfoil_syntax_error_uses_public_error_envelope():
    response = request(
        "POST",
        "/v1/expressions/latex",
        json={"expression": "SIZE("},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "qfoil_syntax_error"


def test_search_limits_are_validated_before_execution():
    response = request(
        "POST",
        "/v1/witnesses/search",
        json={
            "expression": "FULL(z)",
            "variable": "z",
            "bindings": {"x": [1, 0]},
            "options": {"budget": 100_001},
        },
    )

    assert response.status_code == 422
