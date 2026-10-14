"""Optional end-to-end PyJuice/CUDA lifecycle test.

Skipped unless both QFOIL_MODEL_CATALOG and QFOIL_CUDA_MODEL_ID name a real
catalog file and a real, CUDA-loadable model, and a CUDA device is actually
available. Never runs as part of the default, dependency-free test suite.
"""

import asyncio
import os

import httpx
import pytest

from qfoil_server import create_app

CUDA_MODEL_ID_ENV = "QFOIL_CUDA_MODEL_ID"
CUDA_CLASS_ID_ENV = "QFOIL_CUDA_CLASS_ID"


def request(app, method: str, path: str, *, json=None) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=json)

    return asyncio.run(send())


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.cuda
def test_full_pyjuice_lifecycle_on_cuda():
    model_id = os.environ.get(CUDA_MODEL_ID_ENV)
    if not os.environ.get("QFOIL_MODEL_CATALOG") or not model_id:
        pytest.skip("QFOIL_MODEL_CATALOG and QFOIL_CUDA_MODEL_ID must be set")
    if not _cuda_available():
        pytest.skip("CUDA is not available")

    class_id = int(os.environ.get(CUDA_CLASS_ID_ENV, "0"))
    app = create_app()

    current = request(app, "GET", "/v1/models/current")
    assert current.json() == {"model": {"status": "unloaded"}}

    loaded = request(app, "POST", f"/v1/models/{model_id}/load")
    assert loaded.status_code == 200
    model_info = loaded.json()["model"]
    assert model_info["status"] == "ready"

    unobserved = [None] * model_info["num_features"]

    ep_response = request(
        app,
        "POST",
        "/v1/expressions/evaluate",
        json={
            "expression": "EP(x, c)",
            "bindings": {"x": unobserved, "c": class_id},
        },
    )
    assert ep_response.status_code == 200
    assert isinstance(ep_response.json()["value"], float)

    cep_response = request(
        app,
        "POST",
        "/v1/expressions/evaluate",
        json={
            "expression": "CEP(x, c)",
            "bindings": {"x": unobserved, "c": class_id},
        },
    )
    assert cep_response.status_code == 200
    assert isinstance(cep_response.json()["value"], float)

    current_after = request(app, "GET", "/v1/models/current")
    assert current_after.json()["model"]["status"] == "ready"

    unloaded = request(app, "DELETE", "/v1/models/current")
    assert unloaded.json() == {"model": {"status": "unloaded"}}

    size_response = request(
        app,
        "POST",
        "/v1/expressions/evaluate",
        json={"expression": "SIZE(x) <= 2", "bindings": {"x": [1, None, 0]}},
    )
    assert size_response.status_code == 200
    assert size_response.json() == {"status": "decided", "value": 1, "last_witness": None, "reason": None}
