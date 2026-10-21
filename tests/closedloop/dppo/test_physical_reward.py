"""关闭音乐奖励与活动度系数后的真实控制分项、诊断、积分和策略参数验收。

使用含四个物理子步的既有夹具，对标量和列式路径逐控制步独立核对；把节拍函数
替换成必定报错的探针，确认禁用后完全不调用。tracking 仍要求配对活动监督。
额外检查 eta/CFG 显式传参、音乐条件不受奖励开关影响和与奖励独立的音乐条件。
测试只产生 pytest 临时数据，不加载正式权重、不启动仿真或修改已有运行。
"""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.closedloop.dppo.vector_reward_adapter import VectorExecutionReward, compare_reward_evidence
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.evaluation import _decision_evidence
from gem.runtime.trajectory_blocks import ColumnarTrace, pack_trace
from tests.closedloop.dppo.test_data_learning import actual_step, target_activity_fixture
from tests.closedloop.dppo.test_vector_reward_adapter import packet

CONFIG = dict(version='stage2.gmt8_reference_limits.v2',music_reward_enabled=False,music_weight=0.)


def test_disabled_music_skips_beats_intensity_and_preserves_every_integral(monkeypatch):
    from gem.closedloop.dppo import rewards, columnar_reward
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled music reward computed a music score')
    for module in (rewards,columnar_reward):
        monkeypatch.setattr(module,'_derive_motion_beats',forbidden)
        monkeypatch.setattr(module,'_beat_alignment',forbidden)
    oracle=ExecutionReward(CONFIG,target_activity=target_activity_fixture)
    reference=ExecutionReward(CONFIG,target_activity=target_activity_fixture)
    vector=VectorExecutionReward(CONFIG,target_activity=target_activity_fixture)
    rows=[];expected=[]
    for i in range(65):
        row=actual_step(i,speed=.3);row['env_id']=3
        row['reward_primitives']=packet(row,oracle)
        oracle.evaluate_step(row)
        expected.append(reference.evaluate_step(row));rows.append(row)
    actual=vector.evaluate_batch(ColumnarTrace([pack_trace(rows[i:i+25]) for i in range(0,len(rows),25)]))
    for a,b in zip(actual,expected):
        compare_reward_evidence(a,b)
        assert a['transition_valid'],a['errors']
        assert a['components']['music']==dict(enabled=False,valid=True,raw={'reason':'music_reward_disabled'},
            normalized={},score=0.,weight=0.,gate=1.,activity_gate=1.,weighted_rate=0.,integrated_reward=0.)
        assert a['activity']['intensity_score'] is None
        assert a['reward']==pytest.approx(.02*sum(c['score']*c['weight']*c['gate'] for c in a['components'].values()))
        assert a['components']['track']['gate']==1.
    assert vector.columnar_fallbacks==0
    assert vector.event_reward('physical_failure')==-5. and vector.event_reward('music_end')==0.


def test_disabled_music_retains_required_paired_activity_and_rejects_bad_switch():
    bad=ExecutionReward(CONFIG,target_activity=lambda tick:dict(valid=False))
    assert not bad.evaluate_step(actual_step())['transition_valid']
    with pytest.raises(ValueError,match='music_weight'):
        ExecutionReward(dict(CONFIG,music_weight=2.))
    with pytest.raises(ValueError,match='disabled'):
        ExecutionReward(dict(CONFIG,music_reward_enabled=True))


@pytest.mark.parametrize('speed',[0.,.03])
def test_activity_changes_diagnostics_but_never_scales_physical_rewards(speed):
    """同一真实控制夹具只改配对活动度，零/低活动时仍完整评价跟踪质量。"""
    reference_rates=None
    for target_value in (0.,.1,3.):
        def target(tick):
            return dict(target_activity_fixture(tick),activity_rad_s=target_value)
        oracle=ExecutionReward(CONFIG,target_activity=target)
        scalar=ExecutionReward(CONFIG,target_activity=target)
        vector=VectorExecutionReward(CONFIG,target_activity=target)
        rows=[];expected=[]
        for index in range(30):
            row=actual_step(index,speed=speed);row['env_id']=3
            row['reward_primitives']=packet(row,oracle)
            oracle.evaluate_step(row)
            rows.append(row);expected.append(scalar.evaluate_step(row))
        actual=vector.evaluate_batch(ColumnarTrace([pack_trace(rows[i:i+25]) for i in range(0,len(rows),25)]))
        for left,right in zip(actual,expected):
            compare_reward_evidence(left,right)
            assert left['transition_valid'],left['errors']
            assert all(c['gate']==c['activity_gate']==1. for c in left['components'].values())
            tracking=left['components']['track']
            assert tracking['weighted_rate']==pytest.approx(tracking['score']*tracking['weight'])
            assert tracking['weighted_rate']>0
        rates=[r['reward_rate'] for r in actual]
        if reference_rates is None:reference_rates=rates
        else:assert rates==reference_rates
        if target_value==3.:
            assert actual[-1]['activity']['gate']<.1
            assert actual[-1]['activity']['actual_activity_rad_s']==pytest.approx(speed)
        assert vector.columnar_fallbacks==0


def test_failure_event_is_integrated_once_and_not_multiplied_by_dt():
    reward=ExecutionReward(CONFIG,target_activity=target_activity_fixture)
    detail=reward.evaluate_step(actual_step())
    metadata=dict(latency_seconds=.2,reward_details=[detail],generated=dict(prefix_frames=0,joint_limit_diagnostics={}),event_penalty_total=-5.,event_reward=0.,
        terminal_snapshot=dict(terminated=True))
    transition=SimpleNamespace(transition_valid=True,metadata=metadata,rewards=[detail['reward']-5.],
        executed_control_steps=1,executed_physics_steps=4,terminated=True,truncated=False,reason='physical_failure')
    evidence=_decision_evidence(transition,0)
    assert evidence['reward_sum']==pytest.approx(detail['reward']-5.)




def test_policy_receives_actual_eta_cfg_and_precision(monkeypatch):
    from gem.closedloop.dppo import frozen_prefix,actor_scope
    monkeypatch.setattr(frozen_prefix,'FrozenPrefixPolicy',lambda actor,**kwargs:SimpleNamespace(**kwargs))
    monkeypatch.setattr(actor_scope,'configure_actor_scope',lambda actor,scope:dict(scope=scope))
    result=build_policy(object(),dict(std_schedule=[.02]*20,denoising_steps=20,eta=.23,guidance_scale=.7),
        dict(mode='native_fp32',attention='sdpa_math',linear_execution='stable_tf32x3',compile=False,precise_output=True))
    assert (result.eta,result.guidance_scale,result.steps)==(.23,.7,20)
    assert result.actor_scope==dict(scope='all')
