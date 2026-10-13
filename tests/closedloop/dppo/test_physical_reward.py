"""关闭音乐奖励后的真实控制分项、门控、积分与恢复期限验收。

使用含四个物理子步的既有夹具，对标量和列式路径逐控制步独立核对；把节拍函数
替换成必定报错的探针，确认禁用后完全不调用。tracking 仍要求配对活动监督。
额外检查 eta/CFG 显式传参、音乐条件不受奖励开关影响和跨恢复的固定运行期限。
测试只产生 pytest 临时数据，不加载正式权重、不启动仿真或修改已有运行。
"""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.closedloop.dppo.vector_reward_adapter import VectorExecutionReward, compare_reward_evidence
from gem.closedloop.dppo.run_control import RunWallBudget, build_policy
from gem.closedloop.dppo.evaluation import _decision_evidence
from gem.runtime.trajectory_blocks import ColumnarTrace, pack_trace
from tests.closedloop.dppo.test_data_learning import actual_step, target_activity_fixture
from tests.closedloop.dppo.test_vector_reward_adapter import packet

CONFIG = dict(version='stage10.physical_reward.v1',music_reward_enabled=False,music_weight=0.)


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
        assert a['components']['track']['gate']==a['activity']['gate']
    assert vector.columnar_fallbacks==0
    assert vector.event_reward('physical_failure')==-5. and vector.event_reward('music_end')==0.


def test_disabled_music_retains_required_paired_activity_and_rejects_bad_switch():
    bad=ExecutionReward(CONFIG,target_activity=lambda tick:dict(valid=False))
    assert not bad.evaluate_step(actual_step())['transition_valid']
    with pytest.raises(ValueError,match='music_weight'):
        ExecutionReward(dict(CONFIG,music_weight=2.))
    with pytest.raises(ValueError,match='disabled'):
        ExecutionReward(dict(CONFIG,music_reward_enabled=True))


def test_failure_event_is_integrated_once_and_not_multiplied_by_dt():
    reward=ExecutionReward(CONFIG,target_activity=target_activity_fixture)
    detail=reward.evaluate_step(actual_step())
    metadata=dict(latency_seconds=.2,reward_details=[detail],generated=dict(prefix_frames=0),event_penalty_total=-5.,event_reward=0.,
        terminal_snapshot=dict(terminated=True))
    transition=SimpleNamespace(transition_valid=True,metadata=metadata,rewards=[detail['reward']-5.],
        executed_control_steps=1,executed_physics_steps=4,terminated=True,truncated=False,reason='physical_failure')
    evidence=_decision_evidence(transition,0)
    assert evidence['reward_sum']==pytest.approx(detail['reward']-5.)


def test_run_deadline_is_restored_without_another_48_hours():
    now=[1000.];clock=lambda:now[0]
    budget=RunWallBudget(48*3600,clock=clock)
    now[0]+=24*3600;saved=budget.state_dict()
    resumed=RunWallBudget(48*3600,clock=clock);resumed.restore(saved)
    now[0]+=24*3600
    assert resumed.expired and resumed.state_dict()['deadline_unix_seconds']==1000.+48*3600
    with pytest.raises(ValueError,match='identity'):
        RunWallBudget(60,clock=clock).restore(saved)


def test_policy_receives_actual_eta_cfg_and_precision(monkeypatch):
    from gem.closedloop.dppo import dense_policy
    monkeypatch.setattr(dense_policy,'DenseDiffusionPolicy',lambda actor,**kwargs:kwargs)
    result=build_policy(object(),dict(std_schedule=[.02]*20,denoising_steps=20,eta=.23,guidance_scale=.7),
        dict(mode='native_fp32',attention='sdpa_math',linear_execution='stable_tf32x3',compile=False,precise_output=True))
    assert (result['eta'],result['guidance_scale'],result['steps'])==(.23,.7,20)
