"""共享GPU场景采满额度后等待参考边界的回归，仅在服务器1执行。

用真实故障的5100/5380 tick边界检查：已采满的环境不能继续耗尽GMT未来窗口，
必须保存有效bootstrap、标记行政截断并要求下轮reset，不能报告物理失败或伪造
奖励。另检查充足参考、部分真实等待和提前共同结束；每个额外控制步必须严格
对应12 tick、4物理步和一条真实奖励，预占预算只覆盖支持的控制范围。
"""
from types import SimpleNamespace as NS
import pytest
import torch
from gem.closedloop.dppo.vector_boundary import fragment_reference_boundary,FRAGMENT_CONTRACT
from gem.closedloop.dppo.vector_environment import bounded_reference_deadline
from gem.closedloop.dppo.env_adapter import ExecutionIntegrityError
from tests.closedloop.dppo.test_data_learning import transition,context


@pytest.mark.parametrize('tick,source,valid,boundary',[(5100,5380,5364,5100),(4800,6280,6264,6000),
                                                       (600,2980,2964,2700)])
def test_boundary_still_supports_real_prefix_and_lookahead(tick,source,valid,boundary):
    snapshot=dict(tick=tick,source_end_tick=source,reference_valid_end_tick=valid)
    assert fragment_reference_boundary(snapshot)==boundary
    snapshot['tick']=boundary
    bounded_reference_deadline(snapshot,boundary+840)
    snapshot['tick']=boundary+300
    with pytest.raises(ExecutionIntegrityError):bounded_reference_deadline(snapshot,boundary+1140)










@pytest.mark.parametrize('delay,controls,truncated',[(.764757,25,True),(.7,50,False)])
def test_late_active_generation_keeps_real_controls_and_bootstrap(tmp_path,delay,controls,truncated):
    from gem.closedloop.dppo.vector_environment import VectorUpperEnvironment
    from tests.closedloop.dppo.test_env_adapter import adapter
    from gem.closedloop.dppo.buffer import RolloutBuffer
    env,backend=adapter(tmp_path,elapsed=delay)
    env.__class__=VectorUpperEnvironment
    old_snapshot=backend.snapshot
    def snapshot():
        # 将真实故障4800/5380整体平移到夹具600/1180；及时提交会延长参考。
        end=2980 if any(k=='commit_plan' for k,_ in backend.calls) else 1180
        return dict(old_snapshot(),source_end_tick=end,reference_valid_end_tick=end-16)
    backend.snapshot=snapshot;env.snapshot=snapshot()
    generate=env.generate
    def generation(**kwargs):
        value=generate(**kwargs);value['meta']['deadline_tick']=1044
        return value
    env.generate=generation
    result=env.step()
    assert result.executed_control_steps==controls and result.truncated==truncated
    assert not result.terminated and result.next_context is not None
    assert result.control_tick_end==600+12*controls
    assert sum(env.budget.actual)==controls
    assert result.metadata['event_penalty_total']==0
    assert any(method=='commit_plan' for method,_ in backend.calls)==(not truncated)
    if truncated:
        assert result.reason=='reference_horizon_truncated'
        assert result.metadata['reference_execution_boundary']['simulated_arrival_tick']==1068
        assert any(method=='discard_plan' for method,_ in backend.calls)
    RolloutBuffer().append(result)
