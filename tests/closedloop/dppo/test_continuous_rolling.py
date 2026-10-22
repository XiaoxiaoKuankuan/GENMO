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
