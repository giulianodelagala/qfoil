"""Per-operation configuration must not leak into the shared evaluator."""
import pytest
from qfoil import QFOIL, Inconclusive


def test_rng_seed_reproduces_search_despite_prior_operations():
    q = QFOIL(instance_len=16)
    expr = '(SIGMA z. FULL(z)) >= 1'
    assert q.evaluate(expr, rng_seed=7) == 1
    first = q.evaluator.sigma.last_witness
    q.evaluate(expr)
    assert q.evaluate(expr, rng_seed=7) == 1
    assert q.evaluator.sigma.last_witness == first


def test_rng_and_local_budget_are_restored_after_evaluation():
    q = QFOIL(instance_len=3)
    rng = q.evaluator.sigma.search.rng
    state = rng.getstate()
    assert isinstance(q.evaluate('(SIGMA z. FULL(z)) >= 1', rng_seed=9, sigma_search_budget=0), Inconclusive)
    assert q.evaluator.sigma.search_budget == 64
    assert q.evaluator.sigma.search.rng is rng
    assert rng.getstate() == state
    assert q.evaluate('(SIGMA z. FULL(z)) >= 1') == 1


def test_overrides_apply_to_logic_function_and_nested_queries(monkeypatch):
    q = QFOIL(instance_len=3)
    q.define_function('EXISTS_FULL', [], '(SIGMA z. FULL(z)) >= 1')
    seen = []
    original = q.evaluator.sigma.search.local_search
    def capture(*args, **kwargs):
        seen.append((args[3], q.evaluator.sigma.search.rng))
        return original(*args, **kwargs)
    monkeypatch.setattr(q.evaluator.sigma.search, 'local_search', capture)
    assert q.evaluate('EXISTS_FULL() * ((SIGMA y. FULL(y)) >= 1)', rng_seed=3, sigma_search_budget=20) == 1
    assert len(seen) == 2
    assert all(budget == 20 for budget, _ in seen)
    assert seen[0][1] is seen[1][1]
    assert seen[0][1] is not q.evaluator.sigma.search.rng


def test_exception_restores_settings():
    q = QFOIL()
    rng = q.evaluator.sigma.search.rng
    with pytest.raises(ValueError, match='Undefined variable'):
        q.evaluate('missing', rng_seed=2, sigma_search_budget=10)
    assert q.evaluator.sigma.search.rng is rng
    assert q.evaluator.sigma.search_budget == 64
    assert q.evaluator._operation_depth == 0
    assert q.evaluate('1') == 1


@pytest.mark.parametrize('options', [
    {'rng_seed': True}, {'rng_seed': 1.2},
    {'sigma_search_budget': True}, {'sigma_search_budget': -1},
])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        QFOIL().evaluate('1', **options)


def test_low_level_evaluator_supports_operation_options():
    q = QFOIL(instance_len=3)
    ast = q.parse('(SIGMA z. FULL(z)) >= 1')
    assert isinstance(q.evaluator.evaluate(ast, {}, rng_seed=7, sigma_search_budget=0), Inconclusive)
    assert q.evaluator.sigma.search_budget == 64
    assert q.evaluator.evaluate(ast, {}, rng_seed=7) == 1


def test_initial_tuple_and_rng_seed_are_distinct():
    q = QFOIL(instance_len=3)
    assert q.evaluate('(SIGMA z. FULL(z)) >= 1', seed=(1, 0, 1), rng_seed=7, sigma_search_budget=1) == 1
    assert q.evaluator.sigma.last_witness == (1, 0, 1)


def test_flat_evaluation_clears_previous_witness():
    q = QFOIL(instance_len=3)
    q.evaluate('(SIGMA z. FULL(z)) >= 1', rng_seed=7)
    assert q.evaluator.sigma.last_witness is not None
    assert q.evaluate('1') == 1
    assert q.evaluator.sigma.last_witness is None


def test_exact_mode_override_and_nested_queries():
    q = QFOIL(instance_len=3)
    expr = 'SIGMA z. (FULL(z) * ((SIGMA w. FULL(w)) == 8))'
    assert q.evaluate(expr, sigma_mode='exact') == 8
    assert q.evaluator.sigma.mode == 'heuristic'
    assert isinstance(q.evaluate(expr), Inconclusive)


def test_invalid_mode_is_rejected_without_changing_state():
    q = QFOIL()
    with pytest.raises(ValueError, match='sigma_mode'):
        q.evaluate('1', sigma_mode='invalid')
    assert q.evaluator.sigma.mode == 'heuristic'
