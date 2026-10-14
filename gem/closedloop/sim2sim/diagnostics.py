"""MuJoCo真实闭环记录与按真实控制时间计算的诊断。

逐50Hz保存真实qpos/qvel、参考、GMT动作、四个PD子步力矩和边界事件；JSON只存
轻量摘要，数值数组集中保存为NPZ。活动度采用最近0.5秒实际21关节速度RMS；
音乐同步性仅作为非奖励的节拍邻近诊断。没有配对动作时目标活动度和门控为null，
不把这些诊断合成为未经核验的Isaac训练奖励。视频只能从本次MuJoCo状态生成。
"""
from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import numpy as np


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class EpisodeDiagnostics:
    def __init__(self, music, music_start_tick=600):
        self.music=music;self.music_start_tick=music_start_tick
        self.velocity=deque(maxlen=25);self.records=[];self.requests=[];self.events=[]
        self.paired_target=None
        if music.paired_qpos is not None:
            from gem.closedloop.dppo.target_activity import PairedActivityTarget
            self.paired_target=PairedActivityTarget(music.paired_qpos[:,7:],
                joint_names=music.paired_joint_names,source=music.identity,music_start_tick=music_start_tick)

    def append(self, record, *, lateness, wall_elapsed, audio):
        if record['tick']==self.music_start_tick+12:self.velocity.clear()
        self.velocity.append(record['joint_velocity'].copy())
        activity=float(np.sqrt(np.mean(np.square(self.velocity))))
        t=(record['tick']-self.music_start_tick)/600
        target=None
        if self.paired_target is not None and 0<t<=self.paired_target.num_frames/30:
            target=self.paired_target(record['tick'])['activity_rad_s']
        gate=None if target is None else (1. if target<=.1 else min(1.,activity/(.5*target)))
        row={**record,'activity_rad_s':activity,'target_activity_rad_s':target,
             'activity_gate_diagnostic':gate,'music_seconds':max(0.,t),
             'control_lateness_seconds':float(lateness),'wall_elapsed_seconds':float(wall_elapsed),
             'audio_drift_seconds':audio['drift_seconds']}
        self.records.append(row)
        return row

    def finish(self, directory, *, reason, actor, backend, seed, active_wall_seconds, calibration):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        rows=self.records
        if rows:
            keys=[k for k,v in rows[0].items() if isinstance(v,(int,float,bool,np.ndarray,np.generic))]
            arrays={k:np.asarray([r[k] for r in rows]) for k in keys}
            # 不强迫缺失的配对目标成为数字；缺失记录显式valid-mask。
            for k in ('target_activity_rad_s','activity_gate_diagnostic','audio_drift_seconds'):
                arrays[k]=np.array([0. if r[k] is None else r[k] for r in rows])
                arrays[k+'_valid']=np.array([r[k] is not None for r in rows])
            np.savez_compressed(directory/'control_trace.npz',**arrays)
        musical=[r for r in rows if r['tick']>self.music_start_tick]
        beat_metric=None
        if len(musical)>2:
            activity=np.array([r['activity_rad_s'] for r in musical]);times=np.array([r['music_seconds'] for r in musical])
            minima=np.flatnonzero((activity[1:-1]<activity[:-2]) & (activity[1:-1]<=activity[2:]))+1
            beats=np.flatnonzero(self.music.features[:,34]>.5)/30
            if len(minima) and len(beats):
                distance=np.min(np.abs(times[minima,None]-beats[None]),axis=1)
                beat_metric=float(np.mean(np.exp(-distance**2/(2*.1**2))))
        sim=len(rows)*.02
        report=dict(schema='genmo.gmt.mujoco_episode.v1',physics_backend='mujoco',
            execution='online_GENMO_frozen_GMT_real_mj_step',reason=reason,
            completed=reason=='music_end',seed=seed,music=self.music.identity,actor=actor,
            backend=backend,calibration=calibration,control_steps=len(rows),physics_steps=len(rows)*4,
            simulated_seconds=sim,music_executed_seconds=len(musical)*.02,
            active_wall_seconds=active_wall_seconds,real_time_factor=sim/active_wall_seconds if active_wall_seconds else None,
            real_time_pass=bool(rows and .95<=sim/max(active_wall_seconds,1e-9)<=1.05 and
                not reason.startswith(('realtime_','reference_','generation_'))),
            actual_activity_mean_rad_s=float(np.mean([r['activity_rad_s'] for r in musical])) if musical else None,
            joint_tracking_rmse_mean_rad=float(np.mean([r['joint_tracking_rmse_rad'] for r in musical])) if musical else None,
            min_joint_limit_margin_rad=min((r['joint_limit_margin_rad'] for r in rows),default=None),
            joint_limit_violation_steps=sum(r['joint_limit_violation_count']>0 for r in rows),
            music_sync_diagnostic=beat_metric,music_sync_definition='Actual velocity RMS minima to nearest audio beat, sigma=0.1s; not a reward',
            training_reward=None,training_reward_reason='MuJoCo contact/explicit-PD not equated with Isaac reward sensors',
            target_activity_applicable=self.paired_target is not None,
            max_control_compute_seconds=max((r['control_compute_seconds'] for r in rows),default=None),
            max_control_lateness_seconds=max((r['control_lateness_seconds'] for r in rows),default=None),
            requests=self.requests,events=self.events,trace='control_trace.npz' if rows else None)
        write_json(directory/'report.json',report)
        return report
