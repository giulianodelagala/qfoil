"""RF integration through the production loader and existing HTTP endpoints."""

import asyncio
import json

import httpx
import pytest

from qfoil_server import create_app
from qfoil_server.model_catalog import ModelCatalogEntry
from qfoil_server.model_manager import ModelManager
from qfoil_server.test_dataset import load_random_forest_test_dataset


@pytest.fixture
def artifact(tmp_path):
    np = pytest.importorskip("numpy")
    ensemble = pytest.importorskip("sklearn.ensemble")
    joblib = pytest.importorskip("joblib")
    samples = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    targets = np.array([3, 3, 8, 8], dtype=np.int64)
    model = ensemble.RandomForestClassifier(n_estimators=5, random_state=38).fit(
        samples, targets
    )
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            dict(
                backend="random_forest", num_features=2, num_classes=2, class_ids=[3, 8]
            )
        )
    )
    np.save(tmp_path / "test_samples.npy", samples)
    np.save(tmp_path / "test_targets.npy", targets)
    return tmp_path, model


def test_rf_http_lifecycle_samples_values_errors_and_witnesses(artifact):
    path, forest = artifact
    entry = ModelCatalogEntry("rf", "random_forest", str(path), "cpu", 4)
    app = create_app(model_manager=ModelManager(catalog={"rf": entry}))

    async def verify():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")
            assert response.json()["models"] == [
                {"id": "rf", "backend": "random_forest"}
            ]
            response = await client.post("/v1/models/rf/load")
            assert response.status_code == 200, response.text
            metadata = response.json()["model"]
            assert (
                metadata["num_features"] == 2
                and metadata["num_classes"] == 2
                and metadata["device"] == "cpu"
            )
            for index in range(4):
                sample = (
                    await client.get(f"/v1/models/current/dataset/test/samples/{index}")
                ).json()
                assert sample["dataset_size"] == 4
                row = sample["sample"]
                value = await client.post(
                    "/v1/expressions/evaluate",
                    json={"expression": "VAL(x)", "bindings": {"x": row}},
                )
                assert value.status_code == 200
                assert type(value.json()["value"]) is int
                assert value.json()["value"] == int(forest.predict([row])[0])
            random_sample = await client.get(
                "/v1/models/current/dataset/test/samples/random?seed=38"
            )
            assert random_sample.status_code == 200
            partial = await client.post(
                "/v1/expressions/evaluate",
                json={"expression": "VAL(x)", "bindings": {"x": [None, 0]}},
            )
            assert type(partial.json()["value"]) is int and partial.json()["value"] == 0
            for row in [[0], [2, 1], [None, 2]]:
                invalid = await client.post(
                    "/v1/expressions/evaluate",
                    json={"expression": "VAL(x)", "bindings": {"x": row}},
                )
                assert invalid.status_code == 422
            for expression in [
                "VAL(x,3)",
                "EP(x,3)",
                "CEP(x,3)",
                "COND(x,x,3)",
                "(SIGMA z. EP(z,3)) >= 1",
            ]:
                unsupported = await client.post(
                    "/v1/expressions/evaluate",
                    json={"expression": expression, "bindings": {"x": [0, 1]}},
                )
                assert unsupported.status_code == 501, unsupported.text
                assert unsupported.json()["error"] == {
                    "code": "unsupported_operation",
                    "message": "Consulta no soportada por Random Forest",
                }
            formula = "FULL(x) * (VAL(x) == 3)"
            sigma = await client.post(
                "/v1/expressions/evaluate",
                json={"expression": f"(SIGMA x. {formula}) >= 1", "bindings": {}},
            )
            assert sigma.status_code == 200 and sigma.json()["value"] == 1
            witness = sigma.json()["last_witness"]
            assert witness is not None and all(v in [0, 1] for v in witness)
            assert int(forest.predict([witness])[0]) == 3
            verification = await client.post(
                "/v1/expressions/evaluate",
                json={"expression": formula, "bindings": {"x": witness}},
            )
            assert verification.json()["value"] == 1
            search = await client.post(
                "/v1/witnesses/search",
                json={
                    "expression": formula,
                    "variable": "x",
                    "bindings": {},
                    "options": {"seed": [0, 1], "budget": 30},
                },
            )
            assert search.status_code == 200, search.text
            assert search.json()["status"] == "found" and search.json()["verified"]
            unsupported_search = await client.post(
                "/v1/witnesses/search",
                json={
                    "expression": "EP(x,3) >= 0",
                    "variable": "x",
                    "bindings": {},
                    "options": {"seed": [0, 1]},
                },
            )
            assert unsupported_search.status_code == 501
            unload = await client.delete("/v1/models/current")
            assert (
                unload.status_code == 200
                and unload.json()["model"]["status"] == "unloaded"
            )
            assert (
                await client.get("/v1/models/current/dataset/test/samples/0")
            ).status_code == 409
            pure = await client.post(
                "/v1/expressions/evaluate",
                json={"expression": "FULL(x)", "bindings": {"x": [1, 0]}},
            )
            assert pure.json()["value"] == 1

    asyncio.run(verify())


def test_dataset_validation(artifact):
    np = pytest.importorskip("numpy")
    path, _ = artifact
    np.save(path / "test_samples.npy", [[1, 2]])
    with pytest.raises(ValueError, match="values"):
        load_random_forest_test_dataset(str(path), 2, (3, 8))


def test_bad_metadata_load_restores_unloaded_state(artifact):
    from qfoil_server.model_manager import ModelLoadError

    path, _ = artifact
    (path / "metadata.json").write_text(
        json.dumps(
            dict(
                backend="random_forest",
                num_features=99,
                num_classes=2,
                class_ids=[3, 8],
            )
        )
    )
    manager = ModelManager(
        catalog={"rf": ModelCatalogEntry("rf", "random_forest", str(path), "cpu", 4)}
    )

    async def verify():
        with pytest.raises(ModelLoadError):
            await manager.load("rf")
        assert (await manager.current()).status == "unloaded"

    asyncio.run(verify())
