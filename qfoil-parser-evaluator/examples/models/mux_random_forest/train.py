"""Build a reproducible MUX RF artifact without Torch, PyJuice, or CUDA.

Run from the repository root with:
    conda run -n pc python -m examples.models.mux_random_forest.train
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

FEATURE_NAMES = ("S0", "S1", "D0", "D1", "D2", "D3", "N0", "N1")


def build_mux_random_forest():
    """Fit the full truth table and verify every possible input.

    Using every feature and disabling bootstrap lets each tree learn the
    complete MUX rule. The final two inputs are irrelevant noise bits.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    samples = np.array(list(product((0, 1), repeat=8)), dtype=np.int64)
    selectors = 2 * samples[:, 1] + samples[:, 0]
    targets = samples[np.arange(len(samples)), 2 + selectors]
    forest = RandomForestClassifier(
        n_estimators=100,
        bootstrap=False,
        max_features=None,
        random_state=38,
        n_jobs=1,
    ).fit(samples, targets)
    if not np.array_equal(forest.predict(samples), targets):
        raise RuntimeError("MUX RF must reproduce the complete truth table")
    return forest, samples, targets


def save_artifact(output: Path) -> dict:
    """Save the classifier, exhaustive verification inputs, and manifest."""
    import joblib
    import numpy as np
    import sklearn

    forest, samples, targets = build_mux_random_forest()
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, output / "model.joblib", compress=3)
    np.save(output / "test_samples.npy", samples)
    np.save(output / "test_targets.npy", targets)
    metadata = {
        "backend": "random_forest",
        "num_features": len(FEATURE_NAMES),
        "num_classes": 2,
        "class_ids": [0, 1],
        "device": "cpu",
        "feature_names": list(FEATURE_NAMES),
        "label_rule": "D[2*S1 + S0]",
        "train_split": "complete binary truth table",
        "test_split": "exhaustive verification of the training truth table",
        "train_samples": len(samples),
        "test_samples": len(samples),
        "truth_table_accuracy": 1.0,
        "parameters": forest.get_params(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "artifact_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("model.joblib", "test_samples.npy", "test_targets.npy")
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    metadata = save_artifact(args.output)
    print(f"Saved MUX RF to {args.output}: all {metadata['test_samples']} inputs verified")
