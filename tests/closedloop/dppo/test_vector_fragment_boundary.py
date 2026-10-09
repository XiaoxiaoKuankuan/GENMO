"""共享GPU场景采满额度后等待参考边界的回归，仅在服务器1执行。

用真实故障的5100/5380 tick边界检查：已采满的环境不能继续耗尽GMT未来窗口，
必须保存有效bootstrap、标记行政截断并要求下轮reset，不能报告物理失败或伪造
奖励。另检查充足参考、部分真实等待和提前共同结束；每个额外控制步必须严格
对应12 tick、4物理步和一条真实奖励，预占预算只覆盖支持的控制范围。
"""
from types import SimpleNamespace as NS
import pytest
import torch
from gem.closedloop.dppo.vector_boundary import finish_vector_fragment,fragment_reference_boundary,FRAGMENT_CONTRACT
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


def fixture_env(*,source=5380,actual=0):
    snapshot=dict(episode_id='e',env_id=5,tick=5100,done=False,terminated=False,reason=None,
                  source_end_tick=source,reference_valid_end_tick=source-16)
    row=transition();row.identity['env_id']=5
    row.control_tick_begin=5076;row.control_tick_end=5100
    row.metadata=dict(reward_details=[],event_penalty_total=0.,consumed_plan_ids=[])
    calls=[]
    def call(method,**payload):
        assert method=='drain_fragment'
        assert actual<=payload['max_control_steps']
        calls.append(payload)
        trace=[dict(episode_id='e',env_id=5,tick=5100+12*(i+1),consumed_plan_ids=['p']) for i in range(actual)]
        return dict(executed_control_steps=actual,executed_physics_steps=actual*4,physics_count_exact=True,
                    transition_valid=True,trace=trace,snapshot=dict(snapshot,tick=5100+actual*12),mutation_seq=12)
    def no_failure_event(*args):raise AssertionError('Administrative boundary is not physical failure')
    budget=[]
    env=NS(snapshot=snapshot,music_end_tick=18000,soft_end_tick=18000,phase='train',
           backend=NS(call=call,sequence=11),
           budget=NS(reserve=lambda *a,**k:budget.append(k),settle_control=lambda *a:None),
           reward=NS(evaluate_step=lambda step:dict(transition_valid=True,reward=.25),event_reward=no_failure_event),
           remaining_music=lambda:20.,preview_context=lambda:(context(),None))
    return env,row,calls,budget


def test_exhausted_extra_wait_is_truncated_with_bootstrap_and_no_phantom_budget():
    env,row,calls,budget=fixture_env()
    old=row.rewards.clone()
    controls,reset=finish_vector_fragment(env,row,continue_episode=True)
    assert controls==0 and reset and row.truncated and not row.terminated
    assert row.next_context is not None and row.reason=='reference_horizon_truncated'
    assert calls[0]['max_control_steps']==0 and not budget
    torch.testing.assert_close(row.rewards,old)
    assert row.metadata['fragment_tail']['contract']==FRAGMENT_CONTRACT


@pytest.mark.parametrize('actual,reset',[(25,False),(75,True)])
def test_partial_wait_only_resets_when_supported_boundary_reached(actual,reset):
    env,row,calls,budget=fixture_env(source=6280,actual=actual)
    controls,ended=finish_vector_fragment(env,row,continue_episode=True)
    assert controls==actual and ended==reset
    assert row.control_tick_end==5100+12*actual and row.executed_control_steps==2+actual
    assert budget==[dict(control_steps=75,physics_steps=300)]
    assert calls[0]['max_control_steps']==75
    assert row.next_context is not None and not row.terminated and row.truncated
    assert row.metadata['fragment_tail']['reference_horizon_truncated']==reset


def test_music_limit_with_sufficient_reference_is_unchanged():
    env,row,calls,budget=fixture_env(source=9280,actual=25)
    env.soft_end_tick=5700
    controls,ended=finish_vector_fragment(env,row,continue_episode=True)
    assert controls==25 and not ended
    assert calls[0]['max_control_steps']==50 and calls[0]['end_reason']=='collection_limit'
    assert not row.metadata['fragment_tail']['reference_horizon_truncated']
