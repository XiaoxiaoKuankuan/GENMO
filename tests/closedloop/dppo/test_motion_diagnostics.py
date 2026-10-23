"""完整生成动作诊断的帧数兼容与实际时间单位回归。

使用纯CPU合成的世界坐标qpos和已有真实关节边界统计函数构造生成提案，覆盖
通用动作合同的15/30/60/120帧。按时间秒定义匀速、匀加速/三次多项式关节轨迹，
以解析有限区间速度、加速度和恒定jerk作为独立期望，不复制被测np.diff实现。
短动作仍以30Hz采样，不能因音乐上下文为60帧而拉伸动作时间或误用50Hz控制频率。

测试同时核对非法形状/非有限提案计数、混合时长的样本加权RMS、root与关节轴
分离、已承诺前缀也纳入完整生成诊断、输入不变及reward_weight为零。这里的
available/invalid_proposals仅属于episode结束后的诊断，不表示生产参考被拒绝，
也不检验Isaac物理效果、生成模型质量或真实机器人安全性。
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER
from gem.closedloop.dppo.joint_limits import generated_boundaries
from gem.closedloop.dppo.motion_diagnostics import generated_reference_diagnostic


def _plan(joints, *, prefix=6):
    """提供与运行时相同的21关节边界证据；没有模拟生成/奖励/保护逻辑。"""
    joints=np.asarray(joints,dtype=np.float64)
    q=np.zeros((len(joints),28),dtype=np.float64)
    q[:,3]=1.
    q[:,7:]=joints
    limits=np.tile([-100.,100.],(21,1))
    snapshot=dict(joint_names=list(GMT_EXPECTED_JOINT_ORDER),
        joint_hard_limits_rad=limits,joint_soft_limits_rad=limits)
    boundaries=generated_boundaries(q,np.ones_like(joints,dtype=bool),snapshot,
        GMT_EXPECTED_JOINT_ORDER)
    return dict(qpos_world=q,prefix_frames=prefix,joint_limit_diagnostics=boundaries)


@pytest.mark.parametrize('frames',[15,30,60,120])
def test_supported_horizons_use_analytic_30hz_motion_units(frames):
    # q(t)=v*t+a*t²/2+j*t³/6。有限差分的时间位置是区间中点，
    # 三次项的区间平均速度另含j*dt²/24；二阶差分位于中间样本时刻。
    dt=1./30.
    time=np.arange(frames,dtype=np.float64)*dt
    velocity=np.linspace(-.4,.6,21)
    acceleration=np.linspace(.1,.5,21)
    jerk=np.linspace(-.15,.2,21)
    joints=(time[:,None]*velocity+time[:,None]**2*acceleration/2
            +time[:,None]**3*jerk/6)
    plan=_plan(joints)
    before=copy.deepcopy(plan)
    report=generated_reference_diagnostic([plan])
    midtime=(np.arange(frames-1)+.5)*dt
    expected_velocity=(velocity+midtime[:,None]*acceleration
                       +(midtime[:,None]**2/2+dt**2/24)*jerk)
    expected_acceleration=acceleration+time[1:-1,None]*jerk
    assert report['available'] is True
    assert report['valid_proposals']==1 and report['invalid_proposals']==0
    assert report['generated_joint_speed_rms_rad_s']==pytest.approx(
        float(np.sqrt(np.mean(expected_velocity**2))),rel=1e-11,abs=1e-12)
    assert report['generated_joint_acceleration_rms_rad_s2']==pytest.approx(
        float(np.sqrt(np.mean(expected_acceleration**2))),rel=1e-11,abs=1e-12)
    assert report['generated_joint_jerk_rms_rad_s3']==pytest.approx(
        float(np.sqrt(np.mean(jerk**2))),rel=1e-10,abs=1e-12)
    assert report['joint_limits']['generated']['hard']['valid_coordinates']==frames*21
    assert report['quaternion_norm_max_error']==0.
    assert report['prefix_min_frames']==report['prefix_max_frames']==6
    assert report['reward_weight']==0.
    assert report['scope']=='generated_full_horizon_proposals_including_committed_prefix_not_executed_robot'
    np.testing.assert_array_equal(plan['qpos_world'],before.pop('qpos_world'))
    assert {k:v for k,v in plan.items() if k!='qpos_world'}==before


@pytest.mark.parametrize('shape',[(0,28),(1,28),(29,28),(31,28),(59,28),(121,28),
                                  (30,27),(30,29),(30,),(1,30,28)])
def test_unsupported_shapes_remain_unavailable_without_claiming_zero_motion(shape):
    report=generated_reference_diagnostic([dict(qpos_world=np.zeros(shape))])
    assert report==dict(available=False,reason='no_valid_generated_proposals',
                        invalid_proposals=1,reward_weight=0.)


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_proposal_remains_invalid(value):
    q=np.zeros((30,28));q[7,11]=value
    report=generated_reference_diagnostic([dict(qpos_world=q)])
    assert report['available'] is False and report['invalid_proposals']==1
    assert 'generated_joint_speed_rms_rad_s' not in report


def test_empty_or_missing_proposals_are_explicitly_unavailable():
    assert generated_reference_diagnostic([])['invalid_proposals']==0
    assert generated_reference_diagnostic([{}])['invalid_proposals']==1
    assert generated_reference_diagnostic([])['available'] is False


def test_mixed_horizons_weight_actual_differences_without_cross_plan_jump():
    plans=[]
    for frames,speed,offset,prefix in [(30,.2,0.,6),(60,.4,20.,12),(120,.8,-20.,18)]:
        joints=np.tile((offset+np.arange(frames)*speed/30)[:,None],(1,21))
        plans.append(_plan(joints,prefix=prefix))
    report=generated_reference_diagnostic(plans)
    expected=np.sqrt((29*.2**2+59*.4**2+119*.8**2)/(29+59+119))
    assert report['valid_proposals']==3 and report['invalid_proposals']==0
    assert report['generated_joint_speed_rms_rad_s']==pytest.approx(expected,rel=1e-12)
    assert report['generated_joint_acceleration_rms_rad_s2']<1e-10
    assert report['generated_joint_jerk_rms_rad_s3']<1e-8
    assert report['prefix_min_frames']==6 and report['prefix_max_frames']==18


def test_root_motion_and_invalid_shape_do_not_replace_valid_joint_diagnostic():
    plan=_plan(np.tile((np.arange(30)*.6/30)[:,None],(1,21)))
    plan['qpos_world'][:,:3]=np.arange(30)[:,None]*100.
    plan['qpos_world'][:,3]=1.25
    invalid=copy.deepcopy(plan)
    invalid['qpos_world']=invalid['qpos_world'][:-1]
    report=generated_reference_diagnostic([invalid,plan])
    assert report['available'] is True
    assert report['valid_proposals']==1 and report['invalid_proposals']==1
    assert report['generated_joint_speed_rms_rad_s']==pytest.approx(.6,rel=1e-12)
    assert report['quaternion_norm_max_error']==pytest.approx(.25)


def test_committed_prefix_is_included_in_full_generated_horizon():
    # 仅前缀发生一次0.1rad变化，仍须贡献完整提案的29个速度区间；
    # 这不是仅统计可替换后缀，也不是对实际已执行动作进行二次计分。
    joints=np.full((30,21),.1);joints[0]=0.
    report=generated_reference_diagnostic([_plan(joints,prefix=6)])
    assert report['generated_joint_speed_rms_rad_s']==pytest.approx(3./np.sqrt(29))
    assert report['reward_weight']==0.
