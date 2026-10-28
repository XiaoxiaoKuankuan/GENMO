"""延迟信用分配的独立CPU回归：不依赖GPU或正式数据。

用明确50Hz控制记录证明前瞻观察、当前参考消费及奖励参考是三个时刻，计划账本
跨Actor发布保持原生成身份；重复物理步失败。手算延迟奖励的GAE传播、真实终止
和行为版本边界，检查奖励未挪动且未二次乘dt。正式配置验证16384链只更新一次，
MC起点独立确定性且不会改变完整歌曲评估计划，每百轮checkpoint不被预热触发。
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
import yaml
from gem.closedloop.dppo.reference_credit import ReferenceCreditLedger, validate_reward_intervals
from gem.closedloop.dppo.returns import compute_gae
from gem.closedloop.dppo.trainer import fixed_targets
from gem.closedloop.dppo.run_control import iteration_checkpoint_reason,validate_stopping_iteration
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.mc_calibration import matched_reset_mc_plan

ROOT=Path(__file__).resolve().parents[3]


def step(begin, *, current_from=72):
    ticks=list(range(begin-120,begin+121,12))
    return dict(episode_id='e',control_tick_begin=begin,tick=begin+12,
        consumed_reference_ticks=ticks,consumed_plan_ids=['p' if t>=current_from else 'bootstrap:e:1' for t in ticks],
        reference_tick=begin+12,reference_plan_id='p' if begin+12>=current_from else 'bootstrap:e:1')


def test_plan_first_observation_and_consumption_survive_rollout_and_policy_change():
    ledger=ReferenceCreditLedger();ledger.reset('e')
    ledger.register('p',decision_tick=0,policy_version=7,parent_plan_id='bootstrap:e:1',horizon_ticks=1200)
    ledger.committed('p',0)
    ledger.observe([step(t) for t in range(0,48,12)])
    saved=ledger.drain_updates();p=next(r for r in saved if r['plan_id']=='p')
    assert p['first_window_observed_tick']==0 and p['first_current_consumed_tick'] is None
    # 第二轮发布version8，但旧承诺计划仍归version7；不重新计算首次时间。
    ledger.observe([step(t) for t in range(48,96,12)])
    p=next(r for r in ledger.drain_updates() if r['plan_id']=='p')
    assert p['policy_version']==7 and p['first_window_observed_tick']==0
    assert p['first_current_consumed_tick']==72 and p['first_reward_reference_tick']==72
    with pytest.raises(ValueError,match='Duplicate'):ledger.observe([step(84)])
    ledger.reset('next');assert ledger.records=={}


@pytest.mark.parametrize('fault',['uncommitted','unknown','window','reference_time'])
def test_invalid_reference_clock_rejected(fault):
    ledger=ReferenceCreditLedger();ledger.reset('e')
    ledger.register('p',decision_tick=0,policy_version=0,parent_plan_id=None,horizon_ticks=1200)
    if fault!='uncommitted':ledger.committed('p',0)
    row=step(0)
    if fault=='unknown':row['consumed_plan_ids'][20]='invented'
    if fault=='window':row['consumed_reference_ticks'][20]+=12
    if fault=='reference_time':row['reference_tick']=0
    with pytest.raises(ValueError):ledger.observe([row])


def transition(begin, *, policy=0,episode='e',terminal=False,reward=0.):
    return SimpleNamespace(identity=dict(backend_session_id='s',env_id=0,episode_id=episode,policy_version=policy),
        control_tick_begin=begin,control_tick_end=begin+300,executed_control_steps=25,
        rewards=torch.tensor([reward]+[0.]*24,dtype=torch.float64),transition_valid=True,
        terminated=terminal,truncated=False,next_context=None if terminal else {},
        old_value=0.,next_value=0.,metadata=dict(value_snapshot_version=0,event_reward=0.))


def test_lambda98_propagates_real_delayed_reward_without_reassignment():
    rows=[transition(600+i*300,reward=1. if i==15 else 0.,terminal=i==15) for i in range(16)]
    original=[r.rewards.clone() for r in rows]
    out=fixed_targets(rows,None,'cpu',reuse_values=True,critic_version=0,gamma_upper=.99,lambda_upper=.98)
    assert out['advantages'][0]==pytest.approx((.99*.98)**15,rel=1e-12)
    assert out['advantages'][-1]==1.
    for r,before in zip(rows,original):torch.testing.assert_close(r.rewards,before)
    assert (.99*.98)**15>(.99*.95)**15
    validate_reward_intervals(rows)
    with pytest.raises(ValueError,match='Repeated'):validate_reward_intervals(rows+[rows[-1]])


@pytest.mark.parametrize('boundary',['policy','episode','truncation'])
def test_gae_cannot_cross_wrong_boundary(boundary):
    a=transition(600);b=transition(900,reward=1.,terminal=True)
    if boundary=='policy':b.identity['policy_version']=1
    elif boundary=='episode':b.identity['episode_id']='other'
    else:a.truncated=True
    out=fixed_targets([a,b],None,'cpu',reuse_values=True,critic_version=0,gamma_upper=.99,lambda_upper=.98)
    assert out['advantages'].tolist()==[0.,1.]


def test_fragment_bootstrap_uses_actual_control_discount_only_once():
    r=transition(600);r.truncated=True;r.next_value=2.
    out=fixed_targets([r],None,'cpu',reuse_values=True,critic_version=0,gamma_upper=.99,lambda_upper=.98)
    assert out['returns'].item()==pytest.approx(1.98)
    r.terminated=True
    out=fixed_targets([r],None,'cpu',reuse_values=True,critic_version=0,gamma_upper=.99,lambda_upper=.98)
    assert out['returns'].item()==0.


def test_formal_configuration_and_milestone_cumulative_restore():
    c=configuration(ROOT/'tests/closedloop/fixtures/dppo_contract.yaml');s=c['stage10']
    assert s['derived']['maximum_actor_steps']==s['derived']['planned_actor_steps']==8
    assert s['derived']['real_chains']==16384 and s['derived']['bc_per_rollout']==2048
    assert s['training']['actor_lr']==1e-6 and s['generation_horizon']['frames']==60
    assert validate_stopping_iteration(2000,s)==2000
    with pytest.raises(ValueError):validate_stopping_iteration(2001,s)
    assert iteration_checkpoint_reason(100,s['storage'])=='iteration_000100'
    assert iteration_checkpoint_reason(2000,s['storage'])=='iteration_002000'
    assert iteration_checkpoint_reason(101,s['storage']) is None
    assert iteration_checkpoint_reason(100,s['storage'],rejected=True) is None
    assert iteration_checkpoint_reason(100,s['storage'],is_warmup=True) is None


def test_mc_reset_plan_is_independent_reproducible_and_complete():
    c=yaml.safe_load((ROOT/'tests/closedloop/fixtures/dppo_contract.yaml').read_text())
    tasks=[dict(task_id=str(i),dataset='Mine',sample_id=str(i),split='val',seed=i,
        full_music_frames=3000,music_start_frame=0,remaining_music_seconds=100.,initialization='standing') for i in range(200)]
    plan=dict(tasks=tasks,episode_seconds=None);before=copy.deepcopy(plan)
    a=matched_reset_mc_plan(plan,c['stage10']);b=matched_reset_mc_plan(plan,c['stage10'])
    assert a==b and plan==before and a['episode_seconds'] is None
    assert a['start_mode']=='training_reset_mixture'
    counts=sum(t['initialization']=='standing' for t in a['tasks'])
    assert 20<counts<60
    assert any(t['music_start_frame']>2000 for t in a['tasks'])
    assert all(t['remaining_music_seconds']>=8 for t in a['tasks'])
    assert all(t['remaining_music_seconds']==(3000-t['music_start_frame'])/30 for t in a['tasks'])
