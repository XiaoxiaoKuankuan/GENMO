"""新八项奖励、无饱和限位以及三条计算路径的服务器2数值回归。

使用具名GMT机器人夹具及独立平方误差指数公式，逐控制步比较标量、设备批量数学
和列式消费。检查实际/参考分别越界及连续时长；完整生成窗口仅在可修改坐标诊断。
合成数据用于数值合约验证，真实GMT原函数/真实物理仍由配套后端测试单独核验。
"""
import copy
import math
import numpy as np
import pytest
import torch
from test_data_learning import actual_step,target_activity_fixture
from gem.closedloop.dppo.rewards import ExecutionReward,TRACKING_WEIGHTS,TRACKING_STD
from gem.closedloop.dppo.vector_reward_math import VectorRewardMath
from gem.closedloop.dppo.vector_reward_adapter import VectorExecutionReward,compare_reward_evidence
from gem.closedloop.dppo.joint_limits import generated_boundaries
from gem.runtime.trajectory_blocks import pack_trace,ColumnarTrace


def device_packet(row,engine):
    def tensors(v):
        if isinstance(v,dict):return {k:tensors(x) for k,x in v.items()}
        if isinstance(v,np.ndarray):return torch.as_tensor(v)[None]
        return v
    tracking=tensors(row['motion_tracking'])
    for item in tracking['terms'].values():item['error']=torch.tensor([item['error']],dtype=torch.float64)
    substeps=[]
    for step in row['physics_substeps']:
        d=tensors(step['physical_diagnostics']);d.update(physics_tick=torch.tensor([step['physics_tick']]),
            env_index=torch.tensor([0]),joint_position_target=torch.as_tensor(step['joint_position_target'])[None])
        for field in ('foot_support_sphere_tangent_speed_m_s','foot_support_sphere_clearance_m'):
            d[field]={n:torch.zeros(1,1,dtype=torch.float64) for n in d['foot_body_names']}
        substeps.append(d)
    packet=engine.compute(tracking=tracking,errors=tensors(row['errors']),substeps=substeps,
        actual_joint_pos=torch.as_tensor(row['actual_joint_pos_gmt'])[None],
        reference_joint_pos=torch.as_tensor(row['reference']['joint_pos'])[None],terminated=torch.tensor([False]),
        active=torch.tensor([True]),ticks=torch.tensor([row['tick']]))
    def host(v):
        if torch.is_tensor(v):
            v=v[0].numpy();return v.item() if v.ndim==0 else v
        if isinstance(v,dict):return {k:host(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return type(v)(host(x) for x in v)
        return v
    packet=host(packet);packet['episode_id']=row['episode_id'];return packet


def test_scalar_gpu_math_and_columnar_rewards():
    scalar=ExecutionReward(target_activity=target_activity_fixture)
    vector=VectorExecutionReward(target_activity=target_activity_fixture)
    engine=VectorRewardMath(scalar.config,1,'cpu')
    # 实际训练在启动站立预热后才开始音乐奖励，前一步目标已由真实控制建立。
    scalar.seed_previous_target(np.zeros(21));vector.seed_previous_target(np.zeros(21));engine.previous_valid.fill_(True)
    for i in range(3):
        row=actual_step(i,speed=1.);row['env_id']=0
        for j,(name,item) in enumerate(row['motion_tracking']['terms'].items()):
            item['error']=(.01*(j+1))**2
            item['source_score']=math.exp(-item['error']/TRACKING_STD[name]**2)
        for substep in row['physics_substeps']:
            d=substep['physical_diagnostics']
            for field in ('foot_support_sphere_tangent_speed_m_s','foot_support_sphere_clearance_m'):
                d[field]={n:np.zeros(1) for n in d['foot_body_names']}
        row['physical_diagnostics']=copy.deepcopy(row['physics_substeps'][-1]['physical_diagnostics'])
        row['actual_joint_pos_gmt'][0]=2.+.1*i
        row['reference']['joint_pos'][1]=2.+.2*i
        row['reward_primitives']=device_packet(row,engine)
        expected=scalar.evaluate_step(row)
        assert expected['transition_valid'],expected['errors']
        assert expected['components']['track']['score']==pytest.approx(sum(TRACKING_WEIGHTS[k]*v['source_score'] for k,v in row['motion_tracking']['terms'].items()))
        # 第一控制步没有前一PD目标时cmd有效性为False，不影响真实奖励可用性。
        observed=vector.evaluate_batch(ColumnarTrace([pack_trace([row])]))[0]
        compare_reward_evidence(observed,expected)
        assert expected['reward']==pytest.approx(.02*sum(v['weighted_rate'] for v in expected['components'].values()))
        assert not expected['components']['music']['enabled'] and not expected['components']['stable']['enabled']
        assert expected['joint_limit_diagnostics']['actual']['soft']['maximum_consecutive_seconds']==pytest.approx(.02*(i+1))
        assert expected['components']['joint_limit']['score']==pytest.approx(.3*(1+.5*i)+.7*(1+i))


def test_full_generated_boundary_uses_only_free_coordinates():
    row=actual_step();names=row['physical_diagnostics']['joint_names']
    snapshot=dict(joint_names=names,joint_hard_limits_rad=np.tile([-2,2],(21,1)),joint_soft_limits_rad=np.tile([-1.8,1.8],(21,1)))
    q=np.zeros((60,28));q[:10,7:]=100.;q[59,7]=2.5
    before=q.copy();mask=np.ones((60,21),bool);mask[:10]=False
    report=generated_boundaries(q,mask,snapshot,names)
    assert report['hard']['max_excess_rad']==.5
    assert report['hard']['violating_joint_names']==[names[0]]
    assert report['hard']['valid_coordinates']==50*21
    assert report['reward_weight']==0 and np.array_equal(q,before)


def test_evaluation_accepts_new_reward_and_keeps_separate_boundary_costs():
    from types import SimpleNamespace
    from gem.closedloop.dppo.evaluation import _decision_evidence
    reward=ExecutionReward(target_activity=target_activity_fixture)
    details=[reward.evaluate_step(actual_step(i,speed=1.)) for i in range(2)]
    row=actual_step();names=row['physical_diagnostics']['joint_names']
    snapshot=dict(joint_names=names,joint_hard_limits_rad=np.tile([-2,2],(21,1)),joint_soft_limits_rad=np.tile([-1.8,1.8],(21,1)))
    q=np.zeros((60,28));q[-1,7]=2.4
    boundary=generated_boundaries(q,np.ones((60,21),bool),snapshot,names)
    result=SimpleNamespace(transition_valid=True,rewards=[d['reward'] for d in details],
        executed_control_steps=2,executed_physics_steps=8,terminated=True,truncated=False,reason='music_end',identity={},
        metadata=dict(reward_details=details,event_reward=0.,latency_seconds=.25,
            generated=dict(prefix_frames=12,generation_horizon_frames=60,joint_limit_diagnostics=boundary)))
    record=_decision_evidence(result,0)
    assert record['reward_sum']==pytest.approx(sum(d['reward'] for d in details))
    assert record['controls'][0]['joint_limit_diagnostics']['actual_cost']==0.
    assert record['controls'][0]['joint_limit_diagnostics']['reference_cost']==0.
    assert record['controls'][0]['raw_errors']['root_height_error_m']==0.
    from gem.closedloop.dppo.evaluation import aggregate_evaluation
    aggregate=aggregate_evaluation([dict(transition_valid=True,physical_failure=False,truncated=False,
        end_reason='music_end',reward_sum=record['reward_sum'],dataset='test',seed=42,decisions=[record])])['overall']
    assert aggregate['joint_limits']['generated_proposals']==1
    assert aggregate['joint_limits']['generated']['hard']['max_excess_rad']==pytest.approx(.4)
    assert aggregate['reward_sum']==record['reward_sum']  # 未执行末帧越界只统计，不重复罚分。
