"""Correctness contracts for random-start SIGMA search, not benchmarks."""
from unittest.mock import Mock

from qfoil import QFOIL
from qfoil.backends.random_forest import RandomForestOracle
from qfoil.language.ast import Sigma


def test_rf_random_start_is_complete_and_verified():
    model = Mock(n_features_in_=30, classes_=[3, 8])
    model.predict.return_value = [8]
    q = QFOIL(oracle_backend=RandomForestOracle(model), instance_len=30)
    assert q.evaluate('(SIGMA z. FULL(z) * (VAL(z) == 8)) >= 1') == 1
    witness = q.evaluator.sigma.last_witness
    assert all(v in (0, 1) for v in witness)
    assert q.evaluate('FULL(z) * (VAL(z) == 8)', z=witness) == 1


def test_rf_label_mismatch_has_no_numeric_distance():
    model = Mock(n_features_in_=2, classes_=[3, 8])
    model.predict.return_value = [3]
    q = QFOIL(oracle_backend=RandomForestOracle(model), instance_len=2)
    search = q.evaluator.sigma.search
    for target in (8, 99):
        assert search.soft_violation(q.parse('VAL(z) == c'), {'z': (1, 0), 'c': target}) == 1
    assert search.soft_violation(q.parse('!(VAL(z) == 3)'), {'z': (1, 0)}) == 1
    assert search.soft_violation(q.parse('VAL(z) == 3'), {'z': (1, 0)}) == 0


def test_random_starts_respect_positive_tuple_bounds():
    q = QFOIL(instance_len=4)
    search = q.evaluator.sigma.search
    ast = Sigma('z', q.parse('(z <<= x) * (k <<= z)'))
    x = (1, None, 0, 1)
    k = (1, None, None, None)
    for _ in range(20):
        values = search._random_candidate(ast, {'x': x, 'k': k}, 4)
        assert values[0] == 1 and values[1] is None
        assert all(v is None or v == expected for v, expected in zip(values, x))


def test_restarts_and_moves_share_candidate_budget(monkeypatch):
    q = QFOIL(instance_len=2)
    search = q.evaluator.sigma.search
    ast = Sigma('z', q.parse('FULL(z)'))
    starts, scores = [], []
    def start(*args, **kwargs):
        starts.append(1)
        return [0, 0]
    def score(_, context):
        scores.append(context['z'])
        return 1.0
    monkeypatch.setattr(search, '_random_candidate', start)
    monkeypatch.setattr(search, 'soft_violation', score)
    assert search.local_search(ast, {}, (None, None), budget=25) is None
    assert len(scores) == 25
    assert len(starts) > 1
    scores.clear()
    assert search.local_search(ast, {}, (None, None), budget=0) is None
    assert not scores


def test_random_search_is_reproducible_with_same_rng_seed():
    results = []
    for _ in range(2):
        q = QFOIL(instance_len=12)
        q.evaluator.sigma.search.rng.seed(7)
        assert q.evaluate('(SIGMA z. FULL(z)) >= 1') == 1
        results.append(q.evaluator.sigma.last_witness)
    assert results[0] == results[1]


def test_explicit_seed_is_scored_first(monkeypatch):
    q = QFOIL(instance_len=3)
    search = q.evaluator.sigma.search
    def unexpected_start(*args):
        raise AssertionError('must score the explicit seed first')
    monkeypatch.setattr(search, '_random_candidate', unexpected_start)
    ast = Sigma('z', q.parse('FULL(z)'))
    seed = (1, 0, 1)
    assert search.local_search(ast, {}, (None,) * 3, budget=1, seed=seed) == seed
