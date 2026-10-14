"""Restricted exact enumeration agrees with the original total domain."""

from itertools import product

import pytest

from qfoil import QFOIL, Inconclusive
from qfoil.quantifiers.domains import restricted_candidates


@pytest.mark.parametrize("body", [
    "FULL(y)", "COMP(x,y)", "COMP(y,x)", "FULL(x)", "(y <<< x)", "(x <<< y)",
    "(y << x)", "(x << y)",
    "FULL(y) * (y <<< x)", "COMP(x,y) * (SIZE(y) <= 2)",
    "(y <<< x) * (SIZE(y) <= 2)", "(x <<< y) * (SIZE(y) >= 2)",
    "(SIZE(y) == 1.5)", "(SIZE(y) <= 1.5)", "(SIZE(y) >= 1.5)",
    "(2 >= SIZE(y)) * (1 <= SIZE(y))", "(SIZE(y) == k)",
    "FULL(y) * (SIZE(y) <= 0)", "(SIZE(y) >= 4)",
    "NOT(FULL(y))", "FULL(y) + (SIZE(y) <= 1)",
    "NOT(y <<< x)", "NOT(COMP(x,y))", "FULL(y) == 0",
    "(y <<< x) * (SIGMA z. (COMP(y,z) * (SIZE(z) >= 2)) == 0)",
    "(y <<< x) * (SIGMA y. FULL(y) == 8)",
    "COMP((1,None,0),y)", "((None,1,None) <<< y) * (y <<< x)",
])
def test_counts_match_total_enumeration_for_every_partial_binding(body):
    runtime = QFOIL(instance_len=3)
    expression = f"SIGMA y. ({body})"
    for x in product((0, 1, None), repeat=3):
        bindings = {"x": x, "k": 2}
        expected = runtime.evaluate(expression, bindings, sigma_mode="exact")
        assert runtime.evaluate(expression, bindings, sigma_mode="restricted") == expected


@pytest.mark.parametrize("comparison", ["== 0", "== 2", "<= 1", ">= 1", ">= 0"])
def test_decisions_match_total_enumeration(comparison):
    runtime = QFOIL(instance_len=3)
    for x in product((0, 1, None), repeat=3):
        expr = f"(SIGMA y. ((y <<< x) * (SIZE(y) <= 2))) {comparison}"
        assert runtime.evaluate(expr, x=x, sigma_mode="restricted") == runtime.evaluate(expr, x=x, sigma_mode="exact")


def test_generator_enumerates_only_valid_candidates_in_original_order():
    runtime = QFOIL(instance_len=8)
    ast = runtime.parse("SIGMA y. ((y <<< x) * (SIZE(y) <= 6))")
    domain = list(restricted_candidates(ast, {"x": (1,) * 8}, 8))
    assert len(domain) == 247
    assert domain == [y for y in product((0, 1, None), repeat=8)
                      if all(v in (1, None) for v in y) and sum(v is not None for v in y) <= 6]
    completions = runtime.parse("SIGMA z. COMP(y,z)")
    assert len(list(restricted_candidates(completions, {"y": domain[0]}, 8))) == 4


def test_constraints_combine_and_contradictions_have_no_candidates():
    runtime = QFOIL(instance_len=3)
    expr = "SIGMA y. ((a <<< y) * (y <<< b) * FULL(y))"
    assert runtime.evaluate(expr, a=(1,None,None), b=(0,1,0), sigma_mode="restricted") == 0
    assert runtime.evaluate(expr, a=(1,None,None), b=(1,1,0), sigma_mode="restricted") == 1


def test_empty_tuple_and_large_size_pruning_without_recursion():
    assert QFOIL(instance_len=0, sigma_mode="restricted").evaluate("SIGMA y. FULL(y)") == 1
    runtime = QFOIL(instance_len=1500, sigma_mode="restricted", evaluation_budget=10)
    assert runtime.evaluate("SIGMA y. (SIZE(y) <= 0)") == 1


def test_original_total_mode_and_shared_budget_remain_available():
    runtime = QFOIL(instance_len=3, evaluation_budget=40)
    expr = "SIGMA y. FULL(y)"
    assert isinstance(runtime.evaluate(expr, sigma_mode="exact"), Inconclusive)
    assert runtime.evaluate(expr, sigma_mode="restricted") == 8
    result = runtime.evaluate("SIGMA y. FULL(y)", sigma_mode="restricted", evaluation_budget=2)
    assert isinstance(result, Inconclusive)
    assert result.reason == "evaluation_budget_exhausted"
    assert runtime.evaluator.sigma.mode == "heuristic"


def test_nested_domains_respect_budget_and_do_not_replace_outer_witness():
    class MuxOracle:
        num_features = 8

        def value(self, x):
            return 0 if None in x else x[2 + 2*x[1] + x[0]]

    runtime = QFOIL(instance_len=8, oracle_backend=MuxOracle(), evaluation_budget=100_000)
    expr = "SIGMA y. ((y <<< x) * (SIGMA z. (COMP(y,z) * NOT(VAL(z) == VAL(x))) == 0) * (SIZE(y) <= 6)) >= 1"
    assert runtime.evaluate(expr, x=(1,)*8, sigma_mode="restricted") == 1
    witness = runtime.evaluator.sigma.last_witness
    assert witness == (1,1,1,1,1,1,None,None)
    assert runtime.evaluate("(y <<< x) * (SIGMA z. (COMP(y,z) * NOT(VAL(z) == VAL(x))) == 0) * (SIZE(y) <= 6)",
                            x=(1,)*8, y=witness, sigma_mode="exact") == 1
