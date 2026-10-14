"""Tests for the PyJuice model lifecycle (load/unload) endpoints.

None of these tests require Torch, PyJuice or CUDA: every ``ModelManager``
here is built with an explicit catalog and a fake loader, so the real
catalog file and the real PyJuice backend are never touched.
"""

import asyncio
import subprocess
import sys
import textwrap
import threading

import httpx
import pytest

from qfoil import QFOIL
from qfoil_server import create_app
from qfoil_server.model_catalog import ModelCatalogEntry
from qfoil_server.model_manager import ModelBusyError, ModelManager
from qfoil_server.model_metadata import ModelMetadata


class FakeProbabilityBackend:
    num_features = 4
    class_ids = {0, 1}

    def probability(self, instance, class_id=None):
        return 0.25

    def conditional(self, target, evidence, class_id):
        return 0.8 if class_id == 1 else 0.2


def request(app, method: str, path: str, *, json=None) -> httpx.Response:
    """Exercise one ASGI app without opening a socket or worker thread."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=json)

    return asyncio.run(send())


def make_catalog(**entries: dict) -> dict[str, ModelCatalogEntry]:
    return {
        model_id: ModelCatalogEntry(
            model_id=model_id,
            backend=fields.get("backend", "pyjuice"),
            path=fields.get("path", f"/fake/models/{model_id}"),
            device=fields.get("device", "cpu"),
            min_batch_size=fields.get("min_batch_size", 4),
        )
        for model_id, fields in entries.items()
    }


def make_fake_loader(*, num_features=8, num_classes=2, calls=None):
    """A loader that never imports Torch/PyJuice and just counts calls."""
    calls = calls if calls is not None else []

    def loader(entry: ModelCatalogEntry):
        calls.append(entry.model_id)
        metadata = ModelMetadata(
            status="ready",
            id=entry.model_id,
            backend=entry.backend,
            num_features=num_features,
            num_classes=num_classes,
            device=entry.device,
            loaded_at="2026-09-24T17:32:00Z",
        )
        return metadata, QFOIL(instance_len=num_features)

    return loader, calls


def make_failing_loader(calls=None):
    calls = calls if calls is not None else []

    def loader(entry: ModelCatalogEntry):
        calls.append(entry.model_id)
        raise RuntimeError("boom")

    return loader, calls


def test_server_starts_without_torch_or_pyjuice():
    # Other test modules may import optional backends during collection.
    # Check startup in a fresh interpreter, using the same environment.
    script = textwrap.dedent(
        """
        import asyncio
        import sys

        import httpx

        from qfoil_server import create_app

        async def check_health():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get("/health")
            assert response.status_code == 200

        asyncio.run(check_health())
        assert "torch" not in sys.modules
        assert "pyjuice" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_current_model_is_unloaded_initially():
    app = create_app()
    response = request(app, "GET", "/v1/models/current")

    assert response.status_code == 200
    assert response.json() == {"model": {"status": "unloaded"}}


def test_list_models_returns_catalog_without_loading():
    loader, calls = make_fake_loader()
    catalog = make_catalog(mux={}, spam={})
    manager = ModelManager(catalog=catalog, loader=loader)
    app = create_app(model_manager=manager)

    response = request(app, "GET", "/v1/models")

    assert response.status_code == 200
    models = {entry["id"]: entry["backend"] for entry in response.json()["models"]}
    assert models == {"mux": "pyjuice", "spam": "pyjuice"}
    assert calls == []


def test_unknown_model_id_returns_404():
    manager = ModelManager(catalog=make_catalog(mux={}))
    app = create_app(model_manager=manager)

    response = request(app, "POST", "/v1/models/ghost/load")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_model"


def test_load_invokes_fake_loader_once_and_publishes_safe_metadata():
    loader, calls = make_fake_loader(num_features=8, num_classes=2)
    manager = ModelManager(catalog=make_catalog(mux={}), loader=loader)
    app = create_app(model_manager=manager)

    response = request(app, "POST", "/v1/models/mux/load")

    assert response.status_code == 200
    assert calls == ["mux"]
    assert response.json() == {
        "model": {
            "id": "mux",
            "status": "ready",
            "backend": "pyjuice",
            "num_features": 8,
            "num_classes": 2,
            "device": "cpu",
            "loaded_at": "2026-09-24T17:32:00Z",
        }
    }


def test_second_load_returns_409_already_loaded():
    loader, _ = make_fake_loader()
    manager = ModelManager(catalog=make_catalog(mux={}, spam={}), loader=loader)
    app = create_app(model_manager=manager)

    first = request(app, "POST", "/v1/models/mux/load")
    second = request(app, "POST", "/v1/models/spam/load")

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "model_already_loaded"


def test_loaded_model_configures_sigma_domain_without_bound_variable_binding():
    metadata = ModelMetadata(
        status="ready",
        id="mux",
        backend="pyjuice",
        num_features=4,
        num_classes=2,
        device="cpu",
        loaded_at="2026-09-24T17:32:00Z",
    )
    runtime = QFOIL(
        probabilistic_backend=FakeProbabilityBackend(),
        instance_len=metadata.num_features,
    )
    manager = ModelManager(catalog=make_catalog(mux={}))
    manager._metadata = metadata
    manager._runtime = runtime
    app = create_app(model_manager=manager)

    response = request(
        app,
        "POST",
        "/v1/expressions/evaluate",
        json={
            "expression": "(SIGMA x. (EP(x, c) >= 0.5)) >= 1",
            "bindings": {"c": 1},
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["value"] == 1
    witness = result["last_witness"]
    assert len(witness) == metadata.num_features
    assert all(value in {0, 1, None} for value in witness)
    assert (
        runtime.evaluate(
            "EP(x, c) >= 0.5", bindings={"x": tuple(witness), "c": 1}
        )
        == 1
    )

    without_witness = request(
        app,
        "POST",
        "/v1/expressions/evaluate",
        json={"expression": "SIZE(x) <= 4", "bindings": {"x": witness}},
    )
    assert without_witness.json() == {"status": "decided", "value": 1, "last_witness": None, "reason": None}

    current = request(app, "GET", "/v1/models/current")
    assert current.json()["model"]["num_features"] == 4


def test_failed_load_preserves_previous_runtime():
    loader, calls = make_failing_loader()
    manager = ModelManager(catalog=make_catalog(broken={}), loader=loader)
    app = create_app(model_manager=manager)

    response = request(app, "POST", "/v1/models/broken/load")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "model_load_error"
    assert calls == ["broken"]

    current = request(app, "GET", "/v1/models/current")
    assert current.json() == {"model": {"status": "unloaded"}}

    evaluated = request(
        app,
        "POST",
        "/v1/expressions/evaluate",
        json={"expression": "SIZE(x) <= 2", "bindings": {"x": [1, None, 0]}},
    )
    assert evaluated.status_code == 200
    assert evaluated.json() == {"status": "decided", "value": 1, "last_witness": None, "reason": None}


def test_delete_restores_backend_independent_runtime():
    loader, _ = make_fake_loader()
    manager = ModelManager(catalog=make_catalog(mux={}), loader=loader)
    app = create_app(model_manager=manager)

    request(app, "POST", "/v1/models/mux/load")
    assert manager._runtime is not manager._default_runtime

    response = request(app, "DELETE", "/v1/models/current")

    assert manager._runtime is manager._default_runtime
    assert response.status_code == 200
    assert response.json() == {"model": {"status": "unloaded"}}

    evaluated = request(
        app,
        "POST",
        "/v1/expressions/evaluate",
        json={"expression": "SIZE(x) <= 2", "bindings": {"x": [1, None, 0]}},
    )
    assert evaluated.status_code == 200
    assert evaluated.json() == {"status": "decided", "value": 1, "last_witness": None, "reason": None}


def test_delete_is_idempotent():
    app = create_app()

    response = request(app, "DELETE", "/v1/models/current")

    assert response.status_code == 200
    assert response.json() == {"model": {"status": "unloaded"}}


def test_evaluate_and_search_use_the_active_runtime():
    async def scenario():
        marker_runtime = QFOIL()

        def loader(entry):
            metadata = ModelMetadata(
                status="ready", id=entry.model_id, backend=entry.backend
            )
            return metadata, marker_runtime

        manager = ModelManager(catalog=make_catalog(mux={}), loader=loader)

        before = await manager.execute(lambda runtime: runtime is marker_runtime)
        assert before is False

        await manager.load("mux")
        during = await manager.execute(lambda runtime: runtime is marker_runtime)
        assert during is True

        await manager.unload()
        after = await manager.execute(lambda runtime: runtime is marker_runtime)
        assert after is False

    asyncio.run(scenario())


def test_load_does_not_block_evaluate_and_reports_busy_while_loading():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def slow_loader(entry: ModelCatalogEntry):
            calls.append(entry.model_id)
            started.set()
            release.wait(timeout=5)
            metadata = ModelMetadata(
                status="ready", id=entry.model_id, backend=entry.backend
            )
            return metadata, QFOIL()

        manager = ModelManager(
            catalog=make_catalog(slow={}, other={}), loader=slow_loader
        )

        load_task = asyncio.ensure_future(manager.load("slow"))
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()

        value = await manager.execute(
            lambda runtime: runtime.evaluate(
                "SIZE(x) <= 2", bindings={"x": (1, None, 0)}
            )
        )
        assert value == 1

        with pytest.raises(ModelBusyError):
            await manager.load("other")
        with pytest.raises(ModelBusyError):
            await manager.unload()

        release.set()
        metadata = await load_task
        assert metadata.status == "ready"
        assert calls == ["slow"]

    asyncio.run(scenario())


def test_latex_works_with_and_without_a_loaded_model():
    loader, _ = make_fake_loader()
    manager = ModelManager(catalog=make_catalog(mux={}), loader=loader)
    app = create_app(model_manager=manager)

    expected = {"latex": r"\operatorname{SIZE}\left(x\right) \leq 2"}
    payload = {"expression": "SIZE(x) <= 2"}

    before = request(app, "POST", "/v1/expressions/latex", json=payload)
    assert before.status_code == 200
    assert before.json() == expected

    request(app, "POST", "/v1/models/mux/load")
    during = request(app, "POST", "/v1/expressions/latex", json=payload)
    assert during.status_code == 200
    assert during.json() == expected

    request(app, "DELETE", "/v1/models/current")
    after = request(app, "POST", "/v1/expressions/latex", json=payload)
    assert after.status_code == 200
    assert after.json() == expected


def test_health_reflects_unloaded_and_ready():
    loader, _ = make_fake_loader()
    manager = ModelManager(catalog=make_catalog(mux={}), loader=loader)
    app = create_app(model_manager=manager)

    before = request(app, "GET", "/health")
    assert before.json() == {
        "status": "ok",
        "service": "qfoil-server",
        "model_status": "unloaded",
    }

    request(app, "POST", "/v1/models/mux/load")
    after = request(app, "GET", "/health")
    assert after.json() == {
        "status": "ok",
        "service": "qfoil-server",
        "model_status": "ready",
        "model_id": "mux",
    }
