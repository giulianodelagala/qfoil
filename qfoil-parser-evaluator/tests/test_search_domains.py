"""Heuristic candidates obey necessary constraints before expensive scoring."""

from itertools import product

import pytest

from qfoil import Inconclusive, QFOIL
from qfoil.language.ast import Sigma
from qfoil.quantifiers.domains import plan_domain


@pytest.mark.parametrize("engine", ["local_search", "greedy_ascent"])
@pytest.mark.parametrize("body", [
    "(y <<= x) * (SIZE(y) <= 2)",
    "(k <<= y) * (y <<= x) * (SIZE(y) >= 2)",
    "FULL(y)", "COMP(k,y)", "COMP(y,full)",
    "y << full", "k << y", "(SIZE(y) == 2)",
    "(2 >= SIZE(y)) * (1 <= SIZE(y))",
    "((1,None,None,0) <<= y) * (SIZE(y) == 3)",
])
def test_every_scored_candidate_satisfies_mandatory_factors(monkeypatch, engine, body):
    runtime = QFOIL(instance_len=4)
    context = {"x": (1, None, 0, 1), "k": (1, None, None, None), "full": (1, 0, 0, 1)}
    ast = Sigma("y", runtime.parse(body))
    search = runtime.evaluator.sigma.search
    scored = []

    def score(formula, bindings):
        scored.append(bindings["y"])
        assert runtime.evaluator.evaluate(formula, bindings) == 1
        # Keep local search running through moves and restarts, instead of
        # accepting its already-feasible initial candidate as a witness.
        return 1.0

    monkeypatch.setattr(search, "soft_violation", score)
    assert getattr(search, engine)(ast, context, (None,) * 4, budget=35) is None
    assert scored
    if engine == "local_search":
        assert len(scored) == 35


@pytest.mark.parametrize("body", [
    "(y <<= x) * (SIZE(y) <= 2)", "x << y", "y << x",
    "COMP(x,y)", "COMP(y,x)", "FULL(y) * (SIZE(y) <= 2)",
    "(SIZE(y) == 1.5)", "(SIZE(y) == 2)",
])
def test_domain_samples_and_moves_agree_with_tuple_semantics(body):
    runtime = QFOIL(instance_len=3)
    ast = Sigma("y", runtime.parse(body))
    rng = runtime.evaluator.sigma.search.rng
    for x in product((0, 1, None), repeat=3):
        context = {"x": x}
        valid = [y for y in product((0, 1, None), repeat=3)
                 if runtime.evaluate(body, x=x, y=y) == 1]
        domain = plan_domain(ast, context, 3)
        assert [y for y in product((0, 1, None), repeat=3) if domain.accepts(y)] == valid
        candidate = domain.sample(rng, initial=True)
        if not valid:
            assert candidate is None
            continue
        assert tuple(candidate) in valid
        for i in range(3):
            for value in (0, 1, None):
                proposal = domain.move(candidate, i, value, rng)
                assert proposal is None or tuple(proposal) in valid


@pytest.mark.parametrize("engine", ["local_search", "greedy_ascent"])
def test_size_equality_can_exchange_observed_positions(engine):
    runtime = QFOIL(instance_len=4)
    ast = Sigma("y", runtime.parse("(SIZE(y) == 2) * ((y <<= x))"))
    search = runtime.evaluator.sigma.search
    # Every legal witness must replace the two initially observed positions.
    witness = getattr(search, engine)(
        ast, {"x": (None, None, 1, 0)}, (None,) * 4,
        budget=30, seed=(1, 0, None, None),
    )
    assert witness == (None, None, 1, 0)


@pytest.mark.parametrize("engine", ["local_search", "greedy_ascent"])
def test_moves_exchange_support_to_improve_an_unplanned_predicate(monkeypatch, engine):
    runtime = QFOIL(instance_len=4)
    ast = Sigma("y", runtime.parse("SIZE(y) == 2"))
    search = runtime.evaluator.sigma.search
    scored = []

    def score(_, context):
        y = context["y"]
        assert sum(v is not None for v in y) == 2
        scored.append(y)
        return float(sum(v is None for v in y[2:]))

    monkeypatch.setattr(search, "soft_violation", score)
    witness = getattr(search, engine)(
        ast, {}, (None,) * 4, budget=60, seed=(1, 0, None, None),
    )
    assert witness is not None and witness[:2] == (None, None)
    assert scored[0] == (1, 0, None, None)
    assert len(scored) > 1


@pytest.mark.parametrize("body", [
    "NOT(FULL(y))", "FULL(y) + (SIZE(y) <= 1)",
    "NOT(y <<= x)", "FULL(y) == 0", "SIGMA z. COMP(y,z)",
])
def test_nonmandatory_factors_do_not_constrain_the_outer_domain(body):
    runtime = QFOIL(instance_len=3)
    domain = plan_domain(Sigma("y", runtime.parse(body)), {"x": (1, None, 0)}, 3)
    assert all(domain.accepts(y) for y in product((0, 1, None), repeat=3))


def test_nested_domains_are_planned_from_the_current_outer_binding(monkeypatch):
    runtime = QFOIL(instance_len=4)
    search = runtime.evaluator.sigma.search
    original = search.soft_violation
    inner_scored = []

    def score(ast, context):
        if "z" in context:
            y, z = context["y"], context["z"]
            assert None not in z
            assert all(a is None or a == b for a, b in zip(y, z))
            inner_scored.append((y, z))
        return original(ast, context)

    monkeypatch.setattr(search, "soft_violation", score)
    result = runtime.evaluate(
        "SIGMA y. ((y <<= x) * (SIZE(y) == 2)"
        " * (SIGMA z. COMP(y,z) >= 1)) >= 1", x=(1, 0, 1, 0),
    )
    assert result == 1
    assert inner_scored
    witness = runtime.evaluator.sigma.last_witness
    assert runtime.evaluate("(y <<= x) * (SIZE(y) == 2)", y=witness, x=(1, 0, 1, 0)) == 1


@pytest.mark.parametrize("engine", ["local_search", "greedy_ascent"])
def test_empty_plan_does_not_hide_validation_errors(engine):
    runtime = QFOIL(instance_len=4)
    ast = Sigma("y", runtime.parse("FULL(y) * (SIZE(y) <= 0) * (y <<= x)"))
    with pytest.raises(ValueError, match="equal-length"):
        getattr(runtime.evaluator.sigma.search, engine)(
            ast, {"x": (1, None, 0)}, (None,) * 4, budget=20,
        )


@pytest.mark.parametrize("body", [
    "FULL(y) * (SIZE(y) <= 0)", "y << x", "COMP(y,x)",
])
def test_empty_or_excluded_domain_remains_inconclusive(body):
    runtime = QFOIL(instance_len=3)
    result = runtime.evaluate(f"SIGMA y. ({body}) >= 1", x=(None,) * 3)
    assert isinstance(result, Inconclusive)
    assert runtime.evaluator.sigma.last_witness is None


def test_reduction_finds_a_sparse_witness_with_one_candidate():
    runtime = QFOIL(instance_len=196, sigma_search_budget=1)
    x = (1,) * 3 + (None,) * 193
    assert runtime.evaluate("SIGMA y. ((y <<= x) * (SIZE(y) == 3)) >= 1", x=x) == 1
    assert runtime.evaluator.sigma.last_witness == x
