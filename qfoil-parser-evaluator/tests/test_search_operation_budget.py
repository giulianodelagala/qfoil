import pytest
from qfoil import QFOIL


def test_scoring_and_verification_share_budget():
    # With a complete seed and budget=0, scoring costs one variable visit;
    # independent FULL(y) verification costs two additional node visits.
    short=QFOIL(instance_len=2,evaluation_budget=2)
    result=short.search_witness('FULL(y)','y',{},budget=0,seed=(1,1))
    assert result.status=='inconclusive' and not result.verified
    assert result.reason=='evaluation_budget_exhausted' and result.witness is None
    assert short.evaluator.budget.remaining==0
    assert short.evaluator._operation_depth==0
    assert short.evaluator._evaluation_depth==0
    sufficient=QFOIL(instance_len=2,evaluation_budget=3)
    assert sufficient.search_witness('FULL(y)','y',{},budget=0,seed=(1,1)).found


def test_grow_scoring_does_not_rearm_between_candidates():
    runtime=QFOIL(instance_len=2,evaluation_budget=1)
    # L1 is scored but not planned: finding it still requires a move after
    # the initial BOTTOM score. FULL now starts with a complete candidate.
    assert runtime.grow_witness('L1(y)','y',{},budget=20) is None
    assert runtime.evaluator.budget.remaining==0
    assert runtime.grow_witness('L1(y)','y',{},budget=0,seed=(1,None))==(1,None)
    assert runtime.evaluator.budget.remaining==0


def test_next_evaluation_rearms_after_failed_search():
    runtime=QFOIL(instance_len=2,evaluation_budget=2)
    assert not runtime.search_witness('FULL(y)','y',{},budget=0,seed=(1,1)).found
    assert runtime.evaluate('SIZE(x)',x=(1,1))==2


def test_exception_restores_operation_state(monkeypatch):
    runtime=QFOIL(instance_len=2,evaluation_budget=2)
    def failure(*args,**kwargs):
        raise RuntimeError('backend failed')
    with monkeypatch.context() as m:
        m.setattr(runtime.evaluator.sigma,'grow_witness',failure)
        with pytest.raises(RuntimeError):
            runtime.search_witness('FULL(y)','y',{})
    assert runtime.evaluator._operation_depth==0
    assert runtime.evaluate('SIZE(x)',x=(1,1))==2


def test_nested_operation_does_not_reset_pool():
    runtime=QFOIL(instance_len=2,evaluation_budget=2)
    with runtime.evaluator.operation():
        assert runtime.evaluate('1')==1
        with runtime.evaluator.operation():
            assert runtime.evaluate('1')==1
        assert runtime.evaluator.budget.remaining==0
    assert runtime.evaluate('SIZE(x)',x=(1,1))==2
