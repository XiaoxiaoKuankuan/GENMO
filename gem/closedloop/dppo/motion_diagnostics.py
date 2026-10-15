"""独立物理评估中的动作活动度、音乐同步与平滑性诊断。

真实运动诊断消费已经执行的50Hz关节状态与音乐；另行统计完整30Hz生成提案，
明确后缀未必执行。两者均只在episode结束后计算一次。
节拍对齐使用既有 BUMI 定义；活动度、加速度与速度差分分别报告实际单位。
这些数值不进入 Stage2 奖励、GAE 或 optimizer，也不在普通控制步热路径计算。
缺失真实执行数据时明确报告不可用，不能用生成参考冒充机器人动作质量。
"""
from __future__ import annotations
import numpy as np
import torch
from gem.robots.bumi.metrics import _derive_motion_beats, _beat_alignment


def episode_motion_diagnostic(rows, music, *, music_start_tick=600):
    if not rows:
        return dict(available=False, reason='no_actual_control_trace', reward_weight=0.)
    ticks = np.asarray([r['tick'] for r in rows], dtype=np.int64)
    speeds = np.asarray([r['actual_joint_vel_gmt'] for r in rows], dtype=np.float64)
    positions = np.asarray([r['actual_joint_pos_gmt'] for r in rows], dtype=np.float64)
    if speeds.shape != (len(rows),21) or positions.shape != speeds.shape or not np.isfinite(speeds).all():
        raise ValueError('Physical diagnostic requires actual finite 21-joint control samples')
    if len(ticks)>1 and not np.all(np.diff(ticks)==12):
        raise ValueError('Physical diagnostic controls must remain continuous')
    source = np.flatnonzero(np.asarray(music)[:,34] > .5)
    beat_ticks = music_start_tick + np.rint(source*20/12).astype(np.int64)*12
    beats = torch.from_numpy(np.isin(ticks,beat_ticks))[None]
    valid = torch.ones_like(beats)
    motion = _derive_motion_beats(torch.from_numpy(np.abs(speeds))[None],valid)
    alignment = float(_beat_alignment(beats,motion,valid,50)[1]) if beats.any() else None
    acceleration = np.diff(speeds,axis=0)/.02
    jerk = np.diff(acceleration,axis=0)/.02
    return dict(available=True,reward_weight=0.,control_samples=len(rows),
        actual_activity_rad_s=float(np.sqrt(np.mean(speeds**2))),
        music_alignment=alignment,music_beat_count=int(beats.sum()),motion_beat_count=int(motion.sum()),
        joint_acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))) if len(acceleration) else None,
        joint_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2))) if len(jerk) else None,
        position_velocity_difference_rms_rad_s=float(np.sqrt(np.mean((np.diff(positions,axis=0)/.02-speeds[1:])**2))) if len(rows)>1 else None,
        scope='actual_robot_music_sync_and_smoothness_not_policy_reward')


def generated_reference_diagnostic(plans):
    """单独统计完整生成提案的30Hz质量，不把未执行的后缀当成物理运动。"""
    speeds=[];accelerations=[];jerks=[];quaternion_error=[];prefixes=[];invalid=0
    for plan in plans:
        q=np.asarray(plan.get('qpos_world',[]),dtype=np.float64)
        if q.shape not in ((60,28),(120,28)) or not np.isfinite(q).all():invalid+=1;continue
        velocity=np.diff(q[:,7:],axis=0)*30
        acceleration=np.diff(velocity,axis=0)*30
        speeds.append(velocity.reshape(-1));accelerations.append(acceleration.reshape(-1))
        jerks.append((np.diff(acceleration,axis=0)*30).reshape(-1))
        quaternion_error.append(float(np.max(np.abs(np.linalg.norm(q[:,3:7],axis=-1)-1))))
        prefixes.append(int(plan['prefix_frames']))
    if not speeds:return dict(available=False,reason='no_valid_generated_proposals',invalid_proposals=invalid,reward_weight=0.)
    def rms(values):return float(np.sqrt(np.mean(np.concatenate(values)**2)))
    from .joint_limits import summarize_boundaries
    boundaries=summarize_boundaries([],generated=[p['joint_limit_diagnostics'] for p in plans])
    return dict(joint_limits=boundaries,available=True,reward_weight=0.,valid_proposals=len(speeds),invalid_proposals=invalid,
        generated_joint_speed_rms_rad_s=rms(speeds),generated_joint_acceleration_rms_rad_s2=rms(accelerations),
        generated_joint_jerk_rms_rad_s3=rms(jerks),quaternion_norm_max_error=max(quaternion_error),
        prefix_min_frames=min(prefixes),prefix_max_frames=max(prefixes),
        scope='generated_full_horizon_proposals_including_committed_prefix_not_executed_robot')
