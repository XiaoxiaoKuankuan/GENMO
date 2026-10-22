"""连续滚动调度的CPU因果与时间归一化测试，不替代PhysX实测。

复用能逐步计数的后端替身，验证每份生成使用旧快照，实测延迟的每个50Hz控制
都结算一次，然后才提交并在下一个0.1秒共同格读取新状态。另与独立逐控制步
公式比较变长GAE，核验8秒容量和6～30帧前缀合同；不更新网络或生产资产。
"""
from pathlib import Path
import math
import pytest
import torch

from gem.closedloop.dppo.rolling_schedule import capacity, next_decision_tick, validate_schedule
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.returns import compute_gae
from tests.closedloop.dppo.test_env_adapter import adapter


@pytest.mark.parametrize('seconds,controls',[(.06,5),(.26,15),(.30,15),(.301,20),(.46,25),(.66,35)])
def test_latency_uses_all_old_reference_physics_then_latest_state(tmp_path,seconds,controls):
    env,backend=adapter(tmp_path,elapsed=seconds)
    env.config['runtime']={'generation_schedule':'continuous_rolling.v1'}
    env.rolling=True;env.decision_ticks=60
    row=env.step()
    assert row.control_tick_begin==600 and row.control_tick_end==600+12*controls
    assert row.executed_control_steps==controls and row.executed_physics_steps==4*controls
    assert len(row.rewards)==controls==sum(env.budget.actual)
    commit=next(v for k,v in backend.calls if k=='commit_plan')
    assert commit['expected_control_tick']==600+math.ceil(seconds*50)*12
    assert float(row.next_context['decision_time'][0])==pytest.approx(row.control_tick_end/600.)
    assert env._request()['decision_tick']==row.control_tick_end
    assert env._request()['min_prefix']==6


def test_normal_rejection_keeps_old_motion_and_next_request(tmp_path):
    env,backend=adapter(tmp_path,elapsed=.26)
    env.config['runtime']={'generation_schedule':'continuous_rolling.v1'}
    env.rolling=True;env.decision_ticks=60;backend.reject_commit=True
    row=env.step()
    assert row.executed_control_steps==15 and row.metadata['rejection']['code']=='late_plan'
    assert not row.terminated and not row.truncated and row.next_context is not None


def test_grid_is_eligibility_not_fixed_point_three_period():
    c={'runtime':{'generation_schedule':'continuous_rolling.v1'}}
    assert [next_decision_tick(600,t,c) for t in (612,756,804,1056)]==[660,780,840,1080]
    assert capacity(c,8)==80
    assert capacity({},8)==16
    with pytest.raises(ValueError):capacity(c,8.01)


def test_production_config_preserves_algorithm_and_only_expands_capacity():
    root=Path(__file__).resolve().parents[3]
    old=configuration(root/'configs/closedloop/stage2_stepwise_mean.yaml')
    new=configuration(root/'configs/closedloop/stage2_continuous_rolling.yaml')
    assert new['stage10']['derived']['real_chains']==81920
    for key in ('actor_objective','ppo_clip','eta','std_schedule','bc_enabled','actor_lr_multipliers',
                'gamma_upper','lambda_upper','actor_trainable_scope'):
        assert new['stage10']['training'][key]==old['stage10']['training'][key]
    for key in ('reward','physical_episode','generation_horizon','critic'):
        assert new['stage10'][key]==old['stage10'][key]
    validate_schedule(new)
    new['timing']['decision_ticks']=180
    with pytest.raises(ValueError):validate_schedule(new)


def test_variable_interval_gae_matches_independent_physical_time_equations():
    steps=[15,20,5];gamma=.95**(1/25);lam=.98**(1/25)
    rewards=[torch.arange(n,dtype=torch.float64)/100 for n in steps]
    values=[1.,2.,3.];next_values=[2.,3.,4.]
    got=compute_gae(rewards,values,next_values,steps,[True]*3,[True,True,False],
                    gamma_upper=.95,lambda_upper=.98)
    following=0.;expected=[]
    for i in reversed(range(3)):
        delta=sum(float(v)*gamma**k for k,v in enumerate(rewards[i]))+gamma**steps[i]*next_values[i]-values[i]
        following=delta+(gamma*lam)**steps[i]*following
        expected.insert(0,following)
    torch.testing.assert_close(got['advantages_raw'],torch.tensor(expected,dtype=torch.float64))
    # 正常采集截断自举终态，不能被误做真实失败而删除V(next)。
    terminated=compute_gae([rewards[-1]],[3.],[4.],[5],[False],[False],gamma_upper=.95,lambda_upper=.98)
    assert got['advantages_raw'][-1]-terminated['advantages_raw'][0]==pytest.approx(gamma**5*4)


def test_velocity_lineage_resolves_prefix_and_is_not_position_weight():
    from tools.analyze_continuous_rolling import velocity_dependencies
    plans={'a':dict(decision_tick=300,prefix_frames=0,parent_plan_id='bootstrap:x',generation_frames=60),
           'b':dict(decision_tick=600,prefix_frames=6,parent_plan_id='a',generation_frames=60)}
    ticks=list(range(600,841,12))
    evidence=[dict(steps=[dict(control_tick_begin=720,consumed_reference_ticks=ticks,
        consumed_plan_ids=['a' if t<=720 else 'b' for t in ticks])])]
    frames,rows=velocity_dependencies(evidence,plans)
    assert rows[0]['control_tick']==720 and not rows[0]['has_bootstrap_support']
    assert frames['a']==[20,21] and frames['b']==[6,7]
    coefficients=rows[0]['linear_joint_velocity_source_support']
    assert sum(v['coefficient_per_second'] for v in coefficients)==pytest.approx(0.)
    assert sum(abs(v['coefficient_per_second']) for v in coefficients)==pytest.approx(50.)


def test_dynamic_prefix_estimate_does_not_peek_future_and_roundtrips(tmp_path):
    from gem.closedloop.dppo.execution_state import capture_execution_state,restore_execution_state
    env,_=adapter(tmp_path,elapsed=.26)
    env.rolling=True;env.config['runtime']={'generation_schedule':'continuous_rolling.v1'}
    env.rolling_delay_samples=[72,84]
    # 已测最大0.14秒+40ms，deadline为708；不读取任何下一请求延迟。
    assert env._request()['deadline_tick']==708
    env.attempt=2;env.decision=1
    saved=capture_execution_state(env)
    saved.update(policy_version=2,iteration=2)
    env.rolling_delay_samples=[]
    restore_execution_state(env,saved)
    assert env._request()['deadline_tick']==708 and env.rolling_delay_samples==[72,84]


def test_rollout_end_prepays_full_warmup_request_but_settles_actual_five_controls():
    from gem.closedloop.dppo.memory_execution import PrepaidMemoryBudget
    credit=dict(control_steps=425,physics_steps=1700,generations=82,accepted_iterations=1,optimizer_attempts=1)
    budget=PrepaidMemoryBudget(credit)
    budget.reserve('train',control_steps=395,physics_steps=1580)
    budget.reserve('train',control_steps=25,physics_steps=100)
    budget.settle_control('train',25,dict(executed_control_steps=5,executed_physics_steps=20,
        physics_count_exact=True,transition_valid=True))
    assert budget.used['control_steps']==400 and budget.used['physics_steps']==1600


def test_world_audit_rejects_missing_evidence_and_noncontiguous_windows():
    from copy import deepcopy
    from tools.audit_continuous_rolling_windows import audit_world_windows
    case=dict(full_zero_update_probability=dict(passed=True,checked_steps=list(range(19))),
        collection_by_rank=[dict(world_window=dict(begin_control=0,current_control=400,
            target_control=400,actual_world_controls=400,reached=True))])
    following=deepcopy(case)
    following['collection_by_rank'][0]['world_window'].update(begin_control=400,current_control=800,target_control=800)
    assert len(audit_world_windows([case,following])[0])==2
    missing=deepcopy(case);missing['collection_by_rank'][0].pop('world_window')
    with pytest.raises(ValueError,match='Missing actual'):audit_world_windows([missing])
    with pytest.raises(ValueError,match='contiguous'):audit_world_windows([case,case])
    missing=deepcopy(case);missing.pop('full_zero_update_probability')
    with pytest.raises(ValueError,match='zero-update'):audit_world_windows([missing])


def test_mc_and_evaluation_credit_covers_non_grid_music_tail():
    from gem.closedloop.dppo.rolling_schedule import generation_credit
    from gem.closedloop.dppo.memory_execution import PrepaidMemoryBudget
    config={'runtime':{'generation_schedule':'continuous_rolling.v1'}}
    count=generation_credit(config,1.23,overhead=5)
    budget=PrepaidMemoryBudget(dict(control_steps=200,physics_steps=800,generations=count))
    # 最快合法生成时刻0.0～1.2秒，共13请求；尾段不能向下取整，也不能用旧2Hz额度。
    for _ in range(13):budget.reserve('train',generations=1)
    assert budget.used['generations']==13 and count==18
    assert generation_credit({},1.23,overhead=5)==8
    with pytest.raises(ValueError):generation_credit(config,float('nan'),overhead=5)
