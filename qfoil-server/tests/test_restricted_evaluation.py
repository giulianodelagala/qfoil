"""HTTP exhaustive mode uses the restricted procedure, preserving Python exact."""
import asyncio
import httpx
from qfoil import QFOIL, Inconclusive
from qfoil_server import create_app
from qfoil_server.model_manager import ModelManager


def post(runtime, expression, bindings=None, budget=None):
    async def send():
        app = create_app(model_manager=ModelManager(default_runtime=runtime))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            options = {"mode": "exact"}
            if budget is not None:
                options["evaluation_budget"] = budget
            return await client.post("/v1/expressions/evaluate", json={
                "expression": expression, "bindings": bindings or {}, "options": options,
            })
    return asyncio.run(send())


def test_http_exhaustive_prunes_while_python_total_mode_is_preserved():
    runtime = QFOIL(instance_len=3, evaluation_budget=40)
    assert isinstance(runtime.evaluate("SIGMA y. FULL(y)", sigma_mode="exact"), Inconclusive)
    response = post(runtime, "SIGMA y. FULL(y)")
    assert response.status_code == 200
    assert response.json()["status"] == "decided"
    assert response.json()["value"] == 8
    assert runtime.evaluator.sigma.mode == "heuristic"


def test_http_nested_sr_for_all_ones_is_verified_under_small_budget():
    class MuxOracle:
        num_features = 8
        def value(self, x):
            return 0 if None in x else x[2 + 2*x[1] + x[0]]
    runtime = QFOIL(instance_len=8, oracle_backend=MuxOracle())
    expression = "SIGMA y. ((y <<= x) * (SIGMA z. (COMP(y,z) * NOT(VAL(z) == VAL(x))) == 0) * (SIZE(y) <= 6)) >= 1"
    response = post(runtime, expression, {"x": [1]*8}, budget=100_000)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "decided" and data["value"] == 1
    assert data["last_witness"] == [1,1,1,1,1,1,None,None]
    witness = tuple(data["last_witness"])
    body = "(y <<= x) * (SIGMA z. (COMP(y,z) * NOT(VAL(z) == VAL(x))) == 0) * (SIZE(y) <= 6)"
    assert runtime.evaluate(body, x=(1,)*8, y=witness, sigma_mode="exact") == 1
    exhausted = post(runtime, expression, {"x": [1]*8}, budget=2)
    assert exhausted.json()["reason"] == "evaluation_budget_exhausted"
