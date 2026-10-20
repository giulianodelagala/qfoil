"""Proof-of-concept contracts for the extended Python API."""

import json

import pytest

from qfoil import QFOIL, QFOILSyntaxError, QFOILValidationError
from qfoil.language import QFOILLatexRenderer, Variable
from qfoil.results import WitnessSearchResult


def test_facade_wraps_parser_errors_in_public_syntax_error():
    qfoil = QFOIL()

    with pytest.raises(QFOILSyntaxError):
        qfoil.parse("SIZE(")


def test_to_latex_accepts_text_and_ast():
    qfoil = QFOIL()
    ast = qfoil.parse("SIZE(x) <= 2")
    expected = r"\operatorname{SIZE}\left(x\right) \leq 2"

    assert qfoil.to_latex("SIZE(x) <= 2") == expected
    assert qfoil.to_latex(ast) == expected


def test_latex_renderer_covers_sigma_tuples_and_precedence():
    qfoil = QFOIL()

    assert qfoil.to_latex("x + y * z") == r"x + y \cdot z"
    assert qfoil.to_latex("(1, None, 0)") == r"\left(1, \bot, 0\right)"
    assert qfoil.to_latex("SIGMA z. FULL(z) * (z <<= x)") == (
        r"\Sigma z \, \left("
        r"\operatorname{FULL}\left(z\right) \cdot "
        r"\left(z \subseteq x\right)\right)"
    )


def test_latex_renderer_rejects_unsafe_manual_identifier():
    with pytest.raises(QFOILValidationError, match="invalid QFOIL identifier"):
        QFOILLatexRenderer().render(Variable(r"x\end{HTML}"))


def test_search_witness_returns_verified_structured_result():
    qfoil = QFOIL()

    result = qfoil.search_witness(
        "FULL(z) * (z <<= x)",
        var="z",
        context={"x": (1, 0, 1)},
        budget=100,
        pool_size=3,
    )

    assert isinstance(result, WitnessSearchResult)
    assert result.found
    assert result.status == "found"
    assert result.witness == (1, 0, 1)
    assert result.verified
    assert result.search == {
        "budget": 100,
        "pool_size": 3,
        "inner_budget": 8,
    }
    json.dumps(result.to_dict())


def test_search_witness_distinguishes_budget_exhaustion_from_error():
    qfoil = QFOIL()

    result = qfoil.search_witness(
        "FULL(z)",
        var="z",
        context={"x": (1, 0)},
        budget=0,
    )

    assert not result.found
    assert result.status == "inconclusive"
    assert result.witness is None
    assert not result.verified


def test_search_witness_accepts_empty_context_without_free_variables():
    qfoil = QFOIL(instance_len=2)

    result = qfoil.search_witness("FULL(z)", var="z", context={}, budget=100)

    assert result.found
    assert result.verified
    assert len(result.witness) == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"budget": -1}, "budget"),
        ({"pool_size": 0}, "pool_size"),
        ({"inner_budget": -1}, "inner_budget"),
        ({"seed": (1,)}, "instance_len"),
    ],
)
def test_search_witness_validates_public_options(kwargs, message):
    qfoil = QFOIL()

    with pytest.raises(QFOILValidationError, match=message):
        qfoil.search_witness(
            "FULL(z)",
            var="z",
            context={"x": (1, 0)},
            **kwargs,
        )
