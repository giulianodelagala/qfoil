"""RF label semantics, propagation, and backend independence."""

import subprocess
import sys
from unittest.mock import Mock

import pytest
from qfoil import QFOIL, QFOILEvaluator, UnsupportedOperationError
from qfoil.backends.random_forest import RandomForestOracle


def runtime():
    model = Mock(n_features_in_=2, classes_=[3, 8])
    model.predict.return_value = [3]
    return QFOIL(
        oracle_backend=RandomForestOracle(model), instance_len=2, sigma_mode="exact"
    ), model


def test_complete_label_and_partial_zero_without_prediction():
    q, model = runtime()
    assert type(q.evaluate("VAL(x)", x=(None, 1))) is int
    assert q.evaluate("VAL(x)", x=(None, 1)) == 0
    model.predict.assert_not_called()
    assert q.evaluate("VAL(x)", x=(0, 1)) == 3
    model.predict.assert_called_once_with([(0, 1)])
    assert q.evaluate("FULL(x) * (VAL(x) == 3)", x=(0, 1)) == 1


@pytest.mark.parametrize(
    "x", [(1,), (1, 0, 1), (2, 1), (None, 2), ("1", 0), (float("nan"), 0), (True, 0)]
)
def test_invalid_instances_do_not_predict(x):
    q, model = runtime()
    with pytest.raises(ValueError):
        q.evaluate("VAL(x)", x=x)
    model.predict.assert_not_called()


@pytest.mark.parametrize(
    "expr",
    [
        "VAL(x,3)",
        "VAL(x,99)",
        "EP(x,3)",
        "CEP(x,3)",
        "COND(x,x,3)",
        "SIGMA x. EP(x,3)",
        "(SIGMA x. EP(x,3)) >= 1",
    ],
)
def test_unsupported_operations_propagate(expr):
    q, model = runtime()
    bindings = {} if "SIGMA" in expr else {"x": (0, 1)}
    with pytest.raises(
        UnsupportedOperationError, match="^Consulta no soportada por Random Forest$"
    ):
        q.evaluate(expr, bindings=bindings)
    model.predict.assert_not_called()


def test_search_unsupported_operation_propagates():
    q, _ = runtime()
    with pytest.raises(UnsupportedOperationError):
        q.search_witness("FULL(x) * (EP(x,3) >= 0)", "x", {}, seed=(0, 1))


def test_sigma_and_verified_witness():
    q, _ = runtime()
    assert q.evaluate("(SIGMA x. FULL(x) * (VAL(x) == 3)) >= 1") == 1
    witness = q.evaluator.sigma.last_witness
    assert witness is not None
    assert q.evaluate("FULL(x) * (VAL(x) == 3)", x=witness) == 1
    result = q.search_witness("FULL(x) * (VAL(x) == 3)", "x", {}, seed=(0, 1))
    assert result.status == "found" and result.verified


def test_real_sklearn_predictions_are_python_ints():
    np = pytest.importorskip("numpy")
    sklearn = pytest.importorskip("sklearn.ensemble")
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    model = sklearn.RandomForestClassifier(n_estimators=5, random_state=38).fit(
        x, [3, 3, 8, 8]
    )
    q = QFOIL(oracle_backend=RandomForestOracle(model), instance_len=2)
    for row, predicted in zip(x, model.predict(x)):
        value = q.evaluate("VAL(x)", x=tuple(row))
        assert type(value) is int and value == predicted


def test_mutually_exclusive_backends():
    q, _ = runtime()
    with pytest.raises(ValueError, match="only one"):
        QFOIL(probabilistic_backend=Mock(), oracle_backend=q.evaluator.oracle_backend)
    with pytest.raises(ValueError, match="only one"):
        QFOILEvaluator(
            probabilistic_backend=Mock(), oracle_backend=q.evaluator.oracle_backend
        )


def test_imports_do_not_load_optional_ml_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import qfoil; from qfoil.backends.random_forest import RandomForestOracle; assert not any(k in sys.modules for k in ('sklearn','numpy','torch','pyjuice')); qfoil.QFOIL().evaluate('FULL(x)', x=(1,0,1))",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "expr", ["SIGMA x. EP(x,3)", "(SIGMA x. EP(x,3)) >= 1", "(SIGMA x. VAL(x,3)) >= 2"]
)
def test_heuristic_cannot_hide_unsupported_operations(expr):
    q, _ = runtime()
    q.evaluator.sigma.mode = "heuristic"
    q.evaluator.sigma.search_budget = 0
    with pytest.raises(UnsupportedOperationError):
        q.evaluate(expr)


def test_defined_function_and_zero_search_budget_do_not_hide_errors():
    q, _ = runtime()
    q.define_function("LABELPROB", ["x"], "EP(x,3)")
    q.evaluator.sigma.mode = "heuristic"
    with pytest.raises(UnsupportedOperationError):
        q.evaluate("SIGMA x. LABELPROB(x)")
    with pytest.raises(UnsupportedOperationError):
        q.search_witness("LABELPROB(x)", "x", {}, budget=0)
