"""Regressions for unknown results: never turn heuristic failure into false."""
import pytest
from qfoil import QFOIL, Inconclusive, INCONCLUSIVE

BODY = '(y == x) * ((SIGMA z. ((z == x) * ((SIGMA w. (w == x)) >= 1))) >= 1)'

@pytest.mark.parametrize('budget',[1,64])
@pytest.mark.parametrize('comparison,expected',[('>= 1',1),('== 0',0)])
def test_nested_counterexample_is_unknown_not_wrong(budget,comparison,expected):
    expression=f'(SIGMA y. ({BODY})) {comparison}'
    exact=QFOIL(instance_len=2,sigma_mode="exact")
    assert exact.evaluate(expression,x=(1,1))==expected
    bounded=QFOIL(instance_len=2,sigma_search_budget=budget)
    assert isinstance(bounded.evaluate(expression,x=(1,1)),Inconclusive)
    assert bounded.evaluator.sigma.last_witness is None
    assert bounded.evaluator.sigma.approximate_depth==0

@pytest.mark.parametrize('expression',[
 'NOT(u)','u + 1','u * 1','u == 0','u <= 0','FULL(u)',
 '(u,0)','SIGMA y. u','(SIGMA y. u) >= 1','(SIGMA y. FULL(y)) >= u',
])
def test_unknown_propagates(expression):
    runtime=QFOIL(instance_len=2,sigma_mode="exact")
    assert isinstance(runtime.evaluate(expression,u=INCONCLUSIVE),Inconclusive)

def test_zero_product_is_conclusive():
    assert QFOIL().evaluate('0 * u',u=INCONCLUSIVE)==0
    with pytest.raises(TypeError):
        bool(INCONCLUSIVE)

def test_unknown_candidate_is_not_verified(monkeypatch):
    runtime=QFOIL(instance_len=2)
    monkeypatch.setattr(runtime,'grow_witness',lambda *a,**k:(1,1))
    monkeypatch.setattr(runtime,'evaluate',lambda *a,**k:INCONCLUSIVE)
    result=runtime.search_witness('FULL(y)','y',{})
    assert result.status=='inconclusive' and result.witness is None and not result.verified

def test_empty_bounded_search_is_inconclusive():
    result=QFOIL(instance_len=2).search_witness('FULL(y)','y',{},budget=0)
    assert result.status=='inconclusive' and not result.verified

def test_defined_function_does_not_hide_sigma():
    runtime=QFOIL(instance_len=2)
    runtime.define_function('INNER',['x'],'(SIGMA z. (z == x)) >= 1')
    assert runtime.evaluator.sigma.search.contains_sigma(runtime.parse('INNER(x)'))
    assert isinstance(runtime.evaluate('(SIGMA y. ((y == x) * INNER(x))) >= 1',x=(1,1)),Inconclusive)


def test_zero_inner_budget_does_not_enumerate(monkeypatch):
    runtime=QFOIL(instance_len=200,sigma_search_budget=64)
    def forbidden(*args,**kwargs):
        raise AssertionError("approximate budget zero must not enumerate")
    monkeypatch.setattr(runtime.evaluator.sigma,'_decide_by_enumeration',forbidden)
    with runtime.evaluator.sigma.search.temporarily_limit_budget(0):
        value=runtime.evaluate('(SIGMA y. FULL(y)) >= 1')
        assert isinstance(value,Inconclusive)
    assert runtime.evaluator.sigma.search_budget==64
    assert runtime.evaluator.sigma.approximate_depth==0
