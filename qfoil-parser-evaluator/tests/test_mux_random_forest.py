"""Exhaustive MUX RF predictions and integration with the label runtime."""

import json
from pathlib import Path

import pytest

from qfoil import QFOIL
from qfoil.backends.random_forest import RandomForestOracle

ARTIFACT = Path(__file__).resolve().parents[1] / "examples/models/mux_random_forest"


def test_mux_artifact_exhaustive_labels_and_noise_invariance():
    np = pytest.importorskip("numpy")
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    forest = joblib.load(ARTIFACT / "model.joblib")
    samples = np.load(ARTIFACT / "test_samples.npy")
    targets = np.load(ARTIFACT / "test_targets.npy")
    assert samples.shape == (256, 8)
    assert len({tuple(row) for row in samples}) == 256
    expected = samples[np.arange(256), 2 + 2 * samples[:, 1] + samples[:, 0]]
    np.testing.assert_array_equal(targets, expected)
    np.testing.assert_array_equal(forest.predict(samples), expected)
    flipped_noise = samples.copy()
    flipped_noise[:, 6:] = 1 - flipped_noise[:, 6:]
    np.testing.assert_array_equal(forest.predict(flipped_noise), expected)
    metadata = json.loads((ARTIFACT / "metadata.json").read_text())
    assert metadata["num_features"] == forest.n_features_in_ == 8
    assert metadata["class_ids"] == list(forest.classes_) == [0, 1]
    qfoil = QFOIL(oracle_backend=RandomForestOracle(forest), instance_len=8)
    for row, target in zip(samples, expected):
        assert qfoil.evaluate("VAL(x)", x=tuple(row)) == int(target)
    assert qfoil.evaluate("VAL(x)", x=(None,) * 8) == 0
    assert qfoil.evaluate("FULL(x)", x=(None,) * 8) == 0
