"""连续滚动模式的调度、变长奖励与兼容性回归，不以替身冒充物理验收。

执行真实UpperEnvironment状态机，以可计数控制后端核对20ms奖励唯一、动态网格、
超时取消及重置前bootstrap；并独立手算10/15/20控制步GAE。新旧配置比较要求除
runtime调度声明外逐字段一致，优化算法绑定完全相同。测试仅写pytest临时目录。
"""
import copy
from pathlib import Path
import pytest
import torch
import yaml
from gem.closedloop.dppo.rolling import (SCHEMA, REFERENCE_CONTRACT, decision_grid,
    next_decision_tick, generation_capacity, validate_schedule, store_rollout_on_host)
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.actor_optimization import optimization_binding
from gem.closedloop.dppo.execution_compatibility import same_execution
from gem.closedloop.dppo.returns import compute_gae
from tests.closedloop.dppo.test_env_adapter import adapter

SCHEDULE=dict(schema=SCHEMA,mode='continuous_rolling',decision_grid_ticks=60,
    reference_contract=REFERENCE_CONTRACT,minimum_prefix_frames=6)
ROOT=Path(__file__).resolve().parents[3]

@pytest.mark.parametrize('arrival,expected',[(720,720),(756,780),(780,780),(792,840),(960,960)])
def test_dynamic_grid_is_not_fixed_three_tenths(arrival,expected):
    assert next_decision_tick(600,arrival,60)==expected
    assert next_decision_tick(600,arrival,300)>=900



@pytest.mark.parametrize('delay,controls',[(.20,10),(.26,15),(.32,20),(.44,25)])
def test_real_step_flow_advances_all_waiting_controls_and_replans(delay,controls,tmp_path):
    env,backend=adapter(tmp_path,elapsed=delay)
    env.config['runtime']={'generation_schedule':SCHEDULE}
    first=env.step()
    assert first.executed_control_steps==controls
    assert len(first.rewards)==controls
    assert first.control_tick_end==600+12*controls
    assert backend.calls[[k for k,(name,_) in enumerate(backend.calls) if name=='commit_plan'][0]][1]['expected_control_tick']==600+round(delay*600)
    second=env.step()
    assert second.control_tick_begin==first.control_tick_end
    assert second.control_tick_end-first.control_tick_begin==24*controls
    store_rollout_on_host(first)
    assert first.chain.device.type=='cpu'

@pytest.mark.parametrize('failure',[False,True])
def test_cancel_pending_at_time_limit_and_failure_precedes_bootstrap(tmp_path,failure):
    env,backend=adapter(tmp_path,elapsed=.32)
    env.config['runtime']={'generation_schedule':SCHEDULE}
    env.soft_end_tick=720
    if failure:backend.fail_at=720
    row=env.step()
    assert row.executed_control_steps==10 and row.control_tick_end==720
    assert row.terminated==failure and row.truncated!=failure
    assert (row.next_context is not None)!=failure
    assert row.metadata['pending_cancelled_at_boundary']
    assert any(name=='discard_plan' for name,_ in backend.calls)
    assert not any(name=='commit_plan' for name,_ in backend.calls)


def test_rejected_commit_retries_on_next_grid_with_old_reference(tmp_path):
    env,backend=adapter(tmp_path,elapsed=.26)
    env.config['runtime']={'generation_schedule':SCHEDULE}
    backend.reject_commit=True
    row=env.step()
    assert row.executed_control_steps==15 and row.metadata['published'] is None
    assert row.metadata['rejection']['policy_penalty'] is False
    assert env.step().control_tick_begin==780


def test_variable_duration_gae_hand_calculation_and_terminal_tail():
    counts=[10,15,20];values=[2.,3.,4.];next_values=[3.,4.,999.]
    rewards=[torch.arange(n,dtype=torch.float64)/100 for n in counts]
    result=compute_gae(rewards,values,next_values,counts,[True,True,False],[True,True,False],gamma_upper=.95,lambda_upper=.98)
    expected=[];tail=0.
    for i in reversed(range(3)):
        r=sum(float(x)*.95**(j/25) for j,x in enumerate(rewards[i]))
        delta=r-values[i]+(.95**(counts[i]/25)*next_values[i] if i<2 else 0.)
        tail=delta+((.95*.98)**(counts[i]/25)*tail if i<2 else 0.)
        expected.insert(0,tail)
    torch.testing.assert_close(result['advantages_raw'],torch.tensor(expected,dtype=torch.float64),rtol=1e-12,atol=1e-12)


def test_rolling_window_prepayment_covers_inflight_twenty_five_control_warmup():
    from gem.closedloop.dppo.memory_execution import PrepaidMemoryBudget
    # 第399个实际控制后仍能预付一次25步；后端只执行最后1步并退款。
    from gem.closedloop.dppo.rolling import world_window_credit
    limits=world_window_credit({'runtime':{'generation_schedule':SCHEDULE}},8.)
    assert limits['generations']==87
    credit=PrepaidMemoryBudget({key:limits[key] for key in ('control_steps','physics_steps')})
    credit.reserve('collect',control_steps=399,physics_steps=1596)
    credit.reserve('warmup',control_steps=25,physics_steps=100)
    credit.settle_control('warmup',25,dict(executed_control_steps=1,executed_physics_steps=4))
    assert credit.used==dict(control_steps=400,physics_steps=1600)


def test_legacy_fragment_migration_cannot_bypass_rolling_reference_identity():
    from gem.closedloop.dppo.training_checkpoint import _legacy_fragment_execution
    old=dict(runtime={'num_envs':128},training={'bc_enabled':False})
    new=copy.deepcopy(old)
    new['training'].update(actor_objective='full_chain_clipped_ppo',actor_trainable_scope='all')
    assert _legacy_fragment_execution(old,new)
    new['runtime']['generation_schedule']=SCHEDULE
    assert not _legacy_fragment_execution(old,new)


def test_pending_generation_advances_to_arrival_event_without_unneeded_grid_snapshots(tmp_path):
    env,backend=adapter(tmp_path,elapsed=.26)
    env.config['runtime']={'generation_schedule':SCHEDULE}
    row=env.step()
    assert [payload['control_steps'] for method,payload in backend.calls if method=='advance']==[13,2]
    assert row.executed_control_steps==15 and row.control_tick_end==780
    assert not row.metadata['events'] # 忙碌期间经过100ms不是错过一次固定周期决策。
