"""Contract tests for read-only access to the active model's test split."""

import asyncio

import httpx

from qfoil import QFOIL
from qfoil_server import create_app
from qfoil_server.model_catalog import ModelCatalogEntry
from qfoil_server.model_manager import ModelManager
from qfoil_server.model_metadata import ModelMetadata
from qfoil_server.test_dataset import PairedTestDataset


def request(app, path: str) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(path)

    return asyncio.run(send())


def make_app(samples=((1, None, 0), (0, 1, 1)), targets=(1, 0), *, exposed=True):
    entry = ModelCatalogEntry(
        model_id="example-model",
        backend="pyjuice",
        path="/not/read/by/fake/loader",
        device="cpu",
        min_batch_size=4,
    )

    metadata = ModelMetadata(
        status="ready",
        id="example-model",
        backend="pyjuice",
        num_features=3,
        num_classes=2,
        device="cpu",
        loaded_at="2026-09-24T17:32:00Z",
    )
    manager = ModelManager(catalog={entry.model_id: entry})
    manager._metadata = metadata
    manager._runtime = QFOIL(instance_len=3)
    manager._test_dataset = PairedTestDataset(samples, targets) if exposed else None
    return create_app(model_manager=manager), manager


def test_gets_first_sample():
    app, _ = make_app()

    response = request(app, "/v1/models/current/dataset/test/samples/0")

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "example-model",
        "split": "test",
        "index": 0,
        "dataset_size": 2,
        "sample": [1, None, 0],
        "expected": 1,
    }


def test_gets_last_sample():
    app, _ = make_app()
    response = request(app, "/v1/models/current/dataset/test/samples/1")
    assert response.status_code == 200
    assert response.json()["sample"] == [0, 1, 1]


def test_negative_index_is_rejected():
    app, _ = make_app()
    response = request(app, "/v1/models/current/dataset/test/samples/-1")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "qfoil_validation_error"


def test_index_equal_to_dataset_size_is_out_of_range():
    app, _ = make_app()
    response = request(app, "/v1/models/current/dataset/test/samples/2")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "test_dataset_index_out_of_range"


def test_invalid_index_is_rejected_with_public_error():
    app, _ = make_app()
    response = request(app, "/v1/models/current/dataset/test/samples/not-an-int")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "qfoil_validation_error"


def test_model_not_loaded():
    response = request(create_app(), "/v1/models/current/dataset/test/samples/0")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_not_loaded"


def test_model_without_test_dataset():
    app, _ = make_app(exposed=False)
    response = request(app, "/v1/models/current/dataset/test/samples/0")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "test_dataset_unavailable"


def test_empty_test_dataset():
    app, _ = make_app(samples=(), targets=())
    response = request(app, "/v1/models/current/dataset/test/samples/random")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "test_dataset_empty"


def test_random_sample_without_seed_uses_the_same_schema():
    app, _ = make_app()
    response = request(app, "/v1/models/current/dataset/test/samples/random")
    assert response.status_code == 200
    assert set(response.json()) == {
        "model_id",
        "split",
        "index",
        "dataset_size",
        "sample",
        "expected",
    }
    assert response.json()["index"] in {0, 1}


def test_random_sample_is_reproducible_with_seed():
    samples = tuple((index,) for index in range(20))
    app, _ = make_app(samples=samples, targets=tuple(range(20)))
    first = request(app, "/v1/models/current/dataset/test/samples/random?seed=123")
    second = request(app, "/v1/models/current/dataset/test/samples/random?seed=123")
    assert first.json() == second.json()


def test_invalid_seed_is_rejected_with_public_error():
    app, _ = make_app()
    response = request(app, "/v1/models/current/dataset/test/samples/random?seed=1.5")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "qfoil_validation_error"


def test_backend_values_are_converted_recursively_to_json():
    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    app, _ = make_app(samples=(FakeTensor([1, None, 0]),), targets=(FakeTensor(1),))
    response = request(app, "/v1/models/current/dataset/test/samples/0")
    assert response.status_code == 200
    assert response.json()["sample"] == [1, None, 0]
    assert response.json()["expected"] == 1


def test_unserializable_sample_uses_explicit_error():
    app, _ = make_app(samples=(object(),), targets=(1,))
    response = request(app, "/v1/models/current/dataset/test/samples/0")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "test_dataset_serialization_error"


def test_reading_samples_does_not_change_active_model():
    app, manager = make_app()
    before = request(app, "/v1/models/current").json()
    request(app, "/v1/models/current/dataset/test/samples/0")
    request(app, "/v1/models/current/dataset/test/samples/random?seed=5")
    after = request(app, "/v1/models/current").json()
    assert after == before
    assert asyncio.run(manager.current()).id == "example-model"
