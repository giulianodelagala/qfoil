"""Sparse digit identifiers must survive loading and classification."""
import json

import pytest

torch = pytest.importorskip("torch")
juice = pytest.importorskip("pyjuice")
from qfoil.backends.pyjuice import PyJuiceModel, PyJuiceInference


def test_loader_uses_prior_class_ids(tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "priors": {"3": 0.5, "8": 0.5}, "num_classes": 2, "num_features": 196,
    }))
    for digit in (3, 8):
        (tmp_path / f"class_{digit}.jpc").touch()
    loaded = []

    class Circuit:
        def to(self, device):
            return self

    monkeypatch.setattr(juice.io, "load", lambda path: loaded.append(path))
    monkeypatch.setattr(juice, "compile", lambda nodes: Circuit())
    model = PyJuiceModel.load(str(tmp_path), device=torch.device("cpu"))
    assert set(model.pc_per_class) == {3, 8}
    assert model.num_classes == 2
    assert len(loaded) == 2
    (tmp_path / "class_8.jpc").unlink()
    with pytest.raises(FileNotFoundError):
        PyJuiceModel.load(str(tmp_path), device=torch.device("cpu"))


def test_classifier_returns_digits_not_column_positions(monkeypatch):
    monkeypatch.setattr(
        juice.queries, "marginal",
        lambda pc, **kwargs: torch.tensor([0.0, -10.0]) if pc == 3
        else torch.tensor([-10.0, 0.0]),
    )
    prediction, posterior = PyJuiceInference.classify(
        {3: 3, 8: 8}, {3: 0.5, 8: 0.5}, torch.zeros((2, 196), dtype=torch.long)
    )
    assert prediction.tolist() == [3, 8]
    assert posterior.shape == (2, 2)
