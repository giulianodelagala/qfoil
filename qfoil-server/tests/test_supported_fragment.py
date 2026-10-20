"""Transport rejects only recursive logic-function definitions.

Nested SIGMA execution is supported: HTTP runs qfoil-parser-evaluator's heuristic
mode exclusively, which never enumerates and reports `inconclusive`
whenever a nested decision cannot be certified (see
`test_inconclusive_http.py`). The fixed SIGMA depth-1 restriction this
module used to enforce is gone; only recursive function definitions -- a
real technical limit, unrelated to nesting -- are still rejected before
execution.
"""

import pytest
from qfoil import QFOIL, UnsupportedOperationError

from qfoil_server.supported_fragment import check_supported_fragment
from test_api import request


@pytest.mark.parametrize("expression", [
    "(SIGMA y. ((y == x) * ((SIGMA z. (z == x)) >= 1))) >= 1",
    "(SIGMA y. L_FULL(y)) >= 1",
    "L_SIZE(x)",
])
def test_nested_evaluation_is_accepted(expression):
    response = request("POST", "/v1/expressions/evaluate", json={
        "expression": expression, "bindings": {"x": [1, 1, 1]},
    })
    assert response.status_code == 200
    assert response.json()["status"] in ("decided", "inconclusive")


def test_search_body_may_contain_sigma():
    response = request("POST", "/v1/witnesses/search", json={
        "expression": "(SIGMA z. (z == x)) == 0", "variable": "y",
        "bindings": {"x": [1, 1, 1]},
    })
    assert response.status_code == 200
    assert response.json()["status"] in ("found", "inconclusive")


def test_nested_expression_can_still_be_rendered():
    response = request("POST", "/v1/expressions/latex", json={
        "expression": "SIGMA y. SIGMA z. (y == z)",
    })
    assert response.status_code == 200


def test_separate_quantifiers_and_article_body_are_supported():
    runtime = QFOIL(instance_len=2)
    check_supported_fragment("(SIGMA y. FULL(y)) == (SIGMA z. FULL(z))", runtime)
    check_supported_fragment(
        "FULL(x) * (y <<= x) * (CEP(y,c) >= delta) * (SIZE(y) <= 7)", runtime,
    )


def test_nested_sigma_hidden_in_function_body_is_accepted():
    runtime = QFOIL(instance_len=2)
    runtime.define_function("INNER", ["x"], "(SIGMA z. (z == x)) >= 1")
    check_supported_fragment("SIGMA y. INNER(y)", runtime)


def test_directly_recursive_function_is_rejected():
    runtime = QFOIL(instance_len=2)
    runtime.define_function("LOOP", ["x"], "LOOP(x) >= 1")
    with pytest.raises(UnsupportedOperationError):
        check_supported_fragment("LOOP(x)", runtime)


def test_indirectly_recursive_function_is_rejected():
    runtime = QFOIL(instance_len=2)
    runtime.define_function("PING", ["x"], "PONG(x) >= 1")
    runtime.define_function("PONG", ["x"], "PING(x) >= 1")
    with pytest.raises(UnsupportedOperationError):
        check_supported_fragment("PING(x)", runtime)
