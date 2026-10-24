"""逐步骤DPPO、困难采样与姿态诊断的独立回归验收。

用手写PPO公式覆盖步骤概率相互抵消、逐步裁剪和K步平均；解析四元数覆盖纯yaw、
85度俯仰和实际倾斜，证明heading病态不会被当成无物理风险。合成多首长音乐
检查困难概率增加、均匀下限、固定rank分片、跨PPO窗口去重、隔离证据门槛、
隔离窗口不被20秒任务跨入以及checkpoint精确恢复。不读正式数据或启动仿真。
新算法迁移测试使用真实小网络非空Adam，验证只重置Actor动量，保留Critic，
并拒绝普通resume和未经授权的物理/奖励/噪声变化；临时资产只写pytest目录。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.step_probability import step_statistics, step_surrogate, step_diagnostics
from gem.closedloop.dppo.adaptive_sampling import AdaptiveTimePool
from gem.closedloop.dppo.full_dataset import SOURCES
from gem.closedloop.dppo.feasibility import pose_diagnostics


def adaptive_config():
    return dict(ema_alpha=.2, uniform_fraction=.3, maximum_weight=10., failure_weight=.65,
        neighbor_radius=2, minimum_visits=12, minimum_rounds=3, failure_rate_threshold=.85,
        severe_fraction=.5, unlearnable_visits=32, unlearnable_rounds=8,
        minimum_improvement=.02, episode_seconds=20., excluded_segments=[])


def make_pool(rank=0, world_size=1):
    catalog=SimpleNamespace(identity={'test':'two_long_musics'}, samples={'train':{
        source:[{'row':{'sample_id':str(i),'num_frames':6000}} for i in range(2)] for source in SOURCES}})
    return AdaptiveTimePool(catalog,adaptive=adaptive_config(),seed=42,rank=rank,world_size=world_size,
        minimum_remaining_seconds=8.,bin_seconds=1.,standing_start_probability=0.)


@pytest.mark.parametrize('count',[4,5,10])
def test_independent_step_clip_and_average_not_product(count):
    # 每步1.08均未裁剪，连乘早已越界；生产必须保留每一步的有效梯度。
    log=torch.full((3,count),math.log(1.08),dtype=torch.float64,requires_grad=True)
    stats=step_statistics(log,log*0,torch.full_like(log,722))
    advantage=torch.tensor([1.,-1.,.5],dtype=torch.float64)
    result=step_surrogate(stats,advantage,clip=.1)
    loss=-result['objective'].mean()
    reference=-torch.minimum(log.exp()*advantage[:,None],log.exp().clamp(.9,1.1)*advantage[:,None]).mean()
    torch.testing.assert_close(loss,reference)
    grad=torch.autograd.grad(loss,log)[0]
    torch.testing.assert_close(grad,-1.08*advantage[:,None].expand_as(log)/(3*count))
    report=step_diagnostics(stats,advantage,clip=.1)
    assert report['complete_chain_count']==3 and report['internal_transition_count']==3*count
    assert report['active_clip_fraction']==0 and not report['ratio_is_joint_probability']


def test_opposite_steps_cannot_cancel_before_clipping():
    log=torch.tensor([[math.log(2),-math.log(2)]],dtype=torch.float64,requires_grad=True)
    stats=step_statistics(log,log*0,torch.ones_like(log))
    loss=-step_surrogate(stats,torch.ones(1),clip=.1)['objective'].mean()
    loss.backward()
    assert float(loss)==pytest.approx(-.8)
    torch.testing.assert_close(log.grad,torch.tensor([[0.,-.25]],dtype=torch.float64))


def test_zero_free_chain_and_mismatched_steps():
    log=torch.zeros(2,5,dtype=torch.float64)
    counts=torch.ones_like(log);counts[0]=0
    stats=step_statistics(log,log,counts)
    result=step_surrogate(stats,torch.tensor([100.,1.]),clip=.2)
    assert result['objective'][0].sum()==0
    assert step_diagnostics(stats,torch.tensor([100.,1.]),clip=.2)['complete_chain_count']==1
    counts[0,1]=1
    with pytest.raises(ValueError,match='same free-coordinate'):
        step_statistics(log,log,counts)


def pitch(degrees):
    a=math.radians(degrees)/2
    return [math.cos(a),0.,math.sin(a),0.]


def test_heading_degeneracy_is_separate_from_real_tilt_and_yaw():
    result=pose_diagnostics(pitch(85),pitch(78.5),non_yaw=.85)
    assert result['reference_forward_horizontal_norm']==pytest.approx(math.cos(math.radians(85)))
    assert result['gravity_direction_error_rad']==pytest.approx(math.radians(6.5))
    assert result['combined_feasibility_risk'] and result['actual_large_tilt']
    assert result['production_non_yaw_error_rad']==.85
    upright=pose_diagnostics(pitch(85),pitch(0))
    assert upright['reference_heading_degenerate'] and not upright['combined_feasibility_risk']
    yaw=pose_diagnostics([1,0,0,0],[math.cos(.7),0,0,math.sin(.7)])
    assert yaw['gravity_direction_error_rad']==pytest.approx(0.) and not yaw['reference_heading_degenerate']
    assert not pose_diagnostics([0,0,0,0],pitch(0))['available']


def event(bin_id=80,fail=True,severe=False,score=.2):
    return dict(source='FineDance',bin_id=bin_id,episode='e',controls=50,
        track_sum=score*50,failure=fail,severe=severe)


def test_difficult_bin_more_sampling_uniform_floor_cap_and_restore():
    pool=make_pool();pool.update([event()])
    ids,p=pool._probabilities['FineDance'];lookup=dict(zip(ids,p))
    assert lookup[80]>lookup[10]*4
    assert p.min()>=.3/len(ids)
    assert p.max()<=10/len(ids)
    state=pool.state_dict();expected=[pool.draw('FineDance') for _ in range(50)]
    pool.load_state_dict(state)
    assert expected==[pool.draw('FineDance') for _ in range(50)]
    broken=copy.deepcopy(state);broken['statistics']['FineDance/80']['failure_ema']=float('nan')
    before=pool.state_dict()
    with pytest.raises(ValueError):pool.load_state_dict(broken)
    assert pool.state_dict()==before


def test_ranks_keep_disjoint_starts_but_share_new_feedback():
    ranks=[make_pool(i,8) for i in range(8)]
    events=[event(80,score=.2),event(90,fail=False,score=.9)]
    for pool in ranks:pool.update(events)
    assert all(p.statistics==ranks[0].statistics for p in ranks)
    owned=[set(p._probabilities['FineDance'][0]) for p in ranks]
    assert sum(map(len,owned))==len(set.union(*owned))


def test_single_risk_not_excluded_repeated_failure_quarantines_short_window():
    pool=make_pool()
    pool.update([event(severe=True)]*10)
    assert not pool.quarantine
    pool.update([event(severe=True)]);assert not pool.quarantine
    report=pool.update([event(severe=True)])
    assert len(report['newly_quarantined'])==1
    item=pool.quarantine[0]
    assert item['sample_id']=='0' and item['begin_seconds']==80 and item['end_seconds']==81
    assert item['evidence']['visits']==12 and item['evidence']['observed_rounds']==3
    ids,_=pool._probabilities['FineDance']
    assert all(not pool._blocked('FineDance',int(i)) for i in ids)
    for _ in range(1000):
        sample,frame,_=pool.draw('FineDance')
        if sample==0:assert frame>=81*30 or frame+20*30<=80*30


def test_persistent_failure_without_pose_risk_needs_more_rounds():
    pool=make_pool()
    for _ in range(7):pool.update([event()]*4)
    assert not pool.quarantine
    pool.update([event()]*4)
    assert pool.quarantine[0]['reason']=='persistent_failure_without_tracking_improvement'


def test_window_boundary_is_not_failure_and_same_bin_visit_is_counted_once():
    pool=make_pool()
    def row(ticks,failed=False):
        return SimpleNamespace(identity={'episode_id':'one'},reason=None,
            metadata=dict(training_task={'dataset':'FineDance','sample_id':'0','music_start_frame':0},
                collector_env_slot=0,terminal_snapshot={'terminated':failed},
                reward_details=[dict(tick=t,components={'track':{'score':.8}}) for t in ticks],
                feasibility={'terminal':{'combined_feasibility_risk':failed}}))
    assert pool.feedback([row([612,624])])==[]
    saved=pool.state_dict();pool.load_state_dict(saved)
    events=pool.feedback([row([636,648],True)])
    assert len(events)==1 and events[0]['controls']==4 and events[0]['failure']
    pool.update(events);assert pool.statistics['FineDance/0']['visits']==1


def test_song_tail_failure_uses_true_time_bin_not_last_legal_start():
    pool=make_pool()
    assert pool._bin('FineDance','0',197*30)==197
    for _ in range(3):pool.update([event(197,severe=True)]*4)
    assert pool.quarantine[0]['begin_seconds']==197
    for _ in range(500):
        sample,start,_=pool.draw('FineDance')
        assert start <= (200-8)*30
        if sample==0:assert start+20*30<=197*30


def test_production_compact_rewards_feed_same_real_tracking_scores():
    from gem.runtime.trajectory_blocks import pack_trace,ColumnarTrace
    from test_vector_reward_adapter import calculators,packet
    from test_data_learning import actual_step
    oracle,_,reward=calculators();reward.evidence_level='training'
    data=[]
    for i in range(4):
        row=actual_step(i,speed=1.+.1*i);row['env_id']=0
        row['reward_primitives']=packet(row,oracle);data.append(row)
    details=reward.evaluate_batch(ColumnarTrace([pack_trace(data)]))
    assert 'components' in details[0] and 'components' not in details[1]
    expected=sum(r['reward_primitives']['components']['track']['score'] for r in data)
    row=SimpleNamespace(identity={'episode_id':'one'},reason=None,metadata=dict(
        training_task={'dataset':'FineDance','sample_id':'0','music_start_frame':0},
        collector_env_slot=0,terminal_snapshot={'terminated':True},reward_details=details))
    pool=make_pool();events=pool.feedback([row])
    assert len(events)==1 and events[0]['controls']==4
    assert events[0]['track_sum']==pytest.approx(expected)
    pool.update(events)
    assert pool.statistics['FineDance/0']['track_ema']==pytest.approx(expected/4)


def test_warmup_failure_does_not_fabricate_tracking_score():
    pool=make_pool()
    task=dict(sample=dict(dataset='FineDance',row={'sample_id':'0'}),music_start_frame=80*30)
    for _ in range(8):
        for _ in range(4):pool.record_warmup_failure(task,{})
        pool.update(pool.feedback([]))
    stat=pool.statistics['FineDance/80']
    assert stat['warmup_failures']==32 and stat['tracking_observed_rounds']==0
    assert stat['first_track']==[] and stat['recent_track']==[] and stat['track_ema']==1.
    assert pool.quarantine[0]['reason']=='persistent_warmup_failure'


def test_new_configuration_rejects_old_joint_training_contract():
    from gem.closedloop.dppo.training_config import configuration
    root=Path(__file__).resolve().parents[3]
    config=configuration(root/'configs/closedloop/stage2_step_dppo_adaptive_server2.yaml')
    assert config['stage9']['actor_objective']=='denoising_step_clipped_ppo'
    assert config['stage9']['objective_step_reduction']=='mean'
    assert config['stage9']['actor_lr']==1e-7
    with pytest.raises(ValueError):
        configuration(root/'configs/closedloop/stage2_ddim10_last5_motion30_music60_lr3e8_server2.yaml')


def migration_fixture(tmp_path):
    from tests.closedloop.dppo.test_actor_lr_initialization import _fixture
    from gem.closedloop.dppo.training_checkpoint import LEGACY_JOINT_SCHEMA
    path,payload,target,identity=_fixture(tmp_path)
    payload['schema']=LEGACY_JOINT_SCHEMA
    payload['identity']['training']['actor_objective']='full_chain_clipped_ppo'
    payload['identity']['dataset']={'sampling_contract':'rank_shared_time_bins.v1'}
    identity['training'].update(actor_objective='denoising_step_clipped_ppo',objective_step_reduction='mean')
    identity['dataset']={'sampling_contract':'rank_shared_adaptive_time_bins.v1','adaptive':adaptive_config()}
    torch.save(payload,path)
    return path,payload,target,identity


def test_explicit_step_migration_resets_actor_adam_preserves_critic_and_rejects_resume(tmp_path):
    from gem.closedloop.dppo.training_checkpoint import load_training_checkpoint, SCHEMA
    from tests.closedloop.dppo.test_actor_lr_initialization import _equal
    path,payload,target,identity=migration_fixture(tmp_path)
    with pytest.raises(ValueError):load_training_checkpoint(path,identity=identity,**target)
    loaded=load_training_checkpoint(path,identity=identity,**target,new_experiment=True,step_dppo_transfer=True)
    _equal(target['actor'].state_dict(),payload['actor'])
    _equal(target['critic'].state_dict(),payload['critic'])
    _equal(target['critic_optimizer'].state_dict(),payload['critic_optimizer'])
    assert target['actor_optimizer'].state_dict()['state']=={}
    assert loaded['step_dppo_transfer']['old_rollout_reused'] is False
    # 新完整schema可以严格恢复，不能依赖一直放开旧身份。
    loaded['schema']=SCHEMA;loaded['identity']=identity
    loaded['actor_optimizer']=target['actor_optimizer'].state_dict()
    torch.save(loaded,tmp_path/'new.pt')
    load_training_checkpoint(tmp_path/'new.pt',identity=identity,**target)


@pytest.mark.parametrize('field',['kernel','reward','critic','physical_episode'])
def test_step_migration_rejects_unapproved_changes_without_mutation(tmp_path,field):
    from gem.closedloop.dppo.training_checkpoint import load_training_checkpoint
    from tests.closedloop.dppo.test_actor_lr_initialization import _equal, _snapshot
    path,payload,target,identity=migration_fixture(tmp_path)
    before=_snapshot(target);identity[field]['unauthorized']=True
    with pytest.raises(ValueError,match='unapproved'):
        load_training_checkpoint(path,identity=identity,**target,new_experiment=True,step_dppo_transfer=True)
    _equal(_snapshot(target),before)
