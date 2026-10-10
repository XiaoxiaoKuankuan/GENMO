"""已执行轨迹的列式奖励消费，不重新计算GPU连续分项或展开完整物理字典。

输入是同一个环境、同一episode的已持久化ColumnarTrace。活动窗口按原时间顺序
构造连续数组，成批计算RMS、门控和积分；音乐仍使用原动作节拍定义及FP32对齐
归约。四个真实物理子步的功率与时间证据按列验证并计算，原跟踪raw/normalized
由只读列视图提供，完整字段仍可回放。每100步的独立标量审计由调用者执行。

本模块不提交参考、不推进物理、不ACK、不改变事件罚分和GAE。输出仍保持既有
奖励字段，列式优化只改变计算组织。异常由调用者回到原标量诊断路径，不能把
缺失字段或非有限值当作有效奖励。环境窗口、目标示范及音乐始终独立维护。
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
from gem.robots.bumi.metrics import _derive_motion_beats, _beat_alignment

WEIGHTS = dict(track='track_weight', music='music_weight', stable='stable_weight',
    alive='alive_weight', cmd='cmd_penalty_weight', torque='torque_penalty_weight',
    contact='contact_penalty_weight', joint_limit='joint_limit_penalty_weight')


def _finite(value, name, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if shape is not None and value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'Invalid columnar {name}')
    return value


def evaluate_columns(reward, trace):
    """纯计算本批结果及后继窗口；核对结束前不修改调用者的因果状态。"""
    n = len(trace)
    if not n: return [], []
    cfg = reward.config
    col = trace.column
    if any(block['count'] and len(block['columns']['fields']['physics_substeps']['fields']) != 4
           for block in trace.blocks):
        raise ValueError('normal control interval requires four actual physical substeps')
    if any(block['count'] and set(block['columns']['fields']['reward_primitives']['fields']['components']['fields'])
           != set(WEIGHTS)-{'music'} for block in trace.blocks):
        raise ValueError('GPU reward identity/configuration/fields mismatch')
    ticks = col('tick')
    episodes = col('episode_id')
    if (ticks.dtype.kind not in 'iu' or np.any(ticks % 12) or
            np.any(np.diff(ticks) != 12) or np.any(episodes != episodes[0]) or
            reward._last_tick is not None and
            (ticks[0] != reward._last_tick + 12 or episodes[0] != reward._episode)):
        raise ValueError('reward stream must be continuous within an episode; reset explicitly')
    if (np.any(col('reward_primitives', 'schema') != reward.vector_version) or
            np.any(col('reward_primitives', 'config_sha256') != reward.config_sha256) or
            any(np.any(col('reward_primitives', key) != col(key)) for key in ('env_id','episode_id','tick'))):
        raise ValueError('GPU reward identity/configuration/fields mismatch')
    if (not col('transition_valid').all() or not col('state_valid').all() or
            np.any(col('completed_physics_steps') != 4)):
        raise ValueError('Invalid physical control interval')
    positions = _finite(col('actual_joint_pos_gmt'), 'position', (n,21))
    velocities = _finite(col('actual_joint_vel_gmt'), 'velocity', (n,21))
    history = list(reward.window)
    new = [(int(ticks[i]), positions[i].copy(), velocities[i].copy()) for i in range(n)]
    all_rows = history + new
    all_speed = np.stack([row[2] for row in all_rows])
    all_ticks = np.asarray([row[0] for row in all_rows])
    ends = len(history) + np.arange(1,n+1)
    counts = np.minimum(ends, reward.activity_steps)
    actual = np.empty(n, dtype=np.float64)
    # 同长度窗口沿最后两个轴归约，与逐行连续数组相同，不使用改变求和顺序的前缀和。
    for count in np.unique(counts):
        ids = np.flatnonzero(counts == count)
        windows = all_speed[ends[ids,None]-count+np.arange(count)]
        actual[ids] = np.sqrt(np.mean(windows ** 2, axis=(1,2)))
    targets = [reward.target_activity(int(t)) for t in ticks]
    activities, intensity, gates = [], [], []
    activity_cfg = cfg['activity']
    for i, target in enumerate(targets):
        begin = int(all_ticks[ends[i]-counts[i]]-12)
        count = int(counts[i]); tick = int(ticks[i])
        if (count != min(reward.activity_steps, (tick-reward.music_start_tick)//12) or
                not target.get('valid') or not target.get('source') or
                target.get('window_count') != count or target.get('window_begin_tick') != begin or
                target.get('window_end_tick') != tick or target.get('window_complete') != (count == reward.activity_steps)):
            raise ValueError('A_target and actual activity windows differ')
        a, target_value = float(actual[i]), float(_finite(target['activity_rad_s'], 'target', ()))
        if target_value < 0: raise ValueError('A_target activity must be nonnegative')
        ratio = a/(activity_cfg['full_gate_ratio']*target_value+activity_cfg['epsilon'])
        gate = 1. if target_value <= activity_cfg['inactive_target_rad_s'] else min(1.,ratio)
        log_ratio = math.log((a+activity_cfg['intensity_epsilon'])/(target_value+activity_cfg['intensity_epsilon']))/math.log(activity_cfg['intensity_log_ratio'])
        score = math.exp(-min(abs(log_ratio),1e150)**2)
        gates.append(gate); intensity.append(score)
        activities.append(dict(actual_activity_rad_s=a,target_activity_rad_s=target_value,
            window_count=count,window_complete=count==reward.activity_steps,window_begin_tick=begin,
            window_end_tick=tick,target_source=copy.deepcopy(target['source']),target_evidence=target,
            valid=True,gate=gate,gate_unclamped_ratio=ratio,intensity_log_ratio=log_ratio,intensity_score=score))
    if reward.music is None: raise ValueError('music features not_available')
    beat_counts = np.minimum(ends, reward.beat_steps)
    beats, music_count, motion_count = np.zeros(n), np.zeros(n,dtype=int), np.zeros(n,dtype=int)
    ids = np.flatnonzero(beat_counts == reward.beat_steps)
    if len(ids):
        indices = ends[ids,None]-reward.beat_steps+np.arange(reward.beat_steps)
        window_ticks = all_ticks[indices]
        if np.any(window_ticks < reward.music_start_tick) or np.any(window_ticks > reward.music_start_tick+len(reward.music)*20):
            raise ValueError('music outside paired task')
        music = torch.from_numpy(np.isin(window_ticks,reward._beat_ticks))
        valid = torch.ones(music.shape,dtype=torch.bool)
        motion = _derive_motion_beats(torch.from_numpy(np.abs(all_speed[indices])),valid)
        for j,i in enumerate(ids):
            music_count[i] = int(music[j].sum()); motion_count[i] = int(motion[j].sum())
            if music_count[i]: beats[i] = float(_beat_alignment(music[j:j+1],motion[j:j+1],valid[j:j+1],round(1/cfg['dt']))[1])
    power = [[] for _ in range(n)]
    for s in range(4):
        path = ('physics_substeps',s)
        physics_ticks = col(*path,'physics_tick')
        durations=_finite(col(*path,'dt_s'),'physical sample duration',(n,))
        if np.any(physics_ticks != ticks-12+(s+1)*3) or np.any(np.abs(durations-cfg['dt']/4)>1e-12):
            raise ValueError('Invalid physical sample timestamp/duration')
        p = (*path,'physical_diagnostics')
        torque = _finite(col(*p,'pd_torque_estimate_nm'),'torque',(n,21))
        velocity = _finite(col(*path,'joint_vel_gmt'),'substep velocity',(n,21))
        values = _finite(np.abs(torque*velocity),'power',(n,21))
        meta = (*p,'mechanical_power_pd_estimate')
        tt = _finite(col(*meta,'torque_sample_tick'),'torque ticks',(n,))
        vt = _finite(col(*meta,'velocity_sample_tick'),'velocity ticks',(n,))
        seconds = _finite(col(*meta,'time_s'),'seconds',(n,))
        sync = col(*meta,'sampling_synchronized')
        if (sync.dtype.kind != 'b' or np.any(sync != (tt==vt)) or np.any(vt!=physics_ticks) or
                not all(math.isclose(float(t),int(p_tick)/600,abs_tol=1e-12) for t,p_tick in zip(seconds,physics_ticks))):
            raise ValueError('PD power sampling-time evidence differs')
        for i in range(n):
            power[i].append(dict(per_joint_w=values[i].tolist(),mean_w=float(values[i].mean()),
                sum_w=float(values[i].sum()),max_w=float(values[i].max()),
                torque_sample_tick=col(*meta,'torque_sample_tick')[i].item(),
                velocity_sample_tick=col(*meta,'velocity_sample_tick')[i].item(),
                time_s=col(*meta,'time_s')[i].item(),sampling_synchronized=bool(sync[i])))
    consistency_raw, consistency_norm = {}, {}
    consistency_valid=col('reference_consistency','valid')
    if consistency_valid.dtype.kind!='b' or not consistency_valid.all(): raise ValueError('Invalid reference consistency')
    for key,tolerance in cfg['consistency'].items():
        value = _finite(col('reference_consistency',key),key,(n,))
        if np.any(np.abs(value)>tolerance): raise ValueError('Reference consistency construction error')
        consistency_raw[key],consistency_norm[key] = value, np.abs(value)/tolerance
    scores = {}
    for name in WEIGHTS:
        if name == 'music': scores[name] = cfg['music_mix']['beat']*beats+cfg['music_mix']['intensity']*np.asarray(intensity)
        else: scores[name] = _finite(col('reward_primitives','components',name,'score'),name,(n,))
        if np.any((scores[name]<0)|(scores[name]>1+1e-12)): raise ValueError('Reward score outside [0,1]')
        scores[name] = np.minimum(scores[name],1.)
    rates = {name: scores[name]*cfg[key]*(-1 if 'penalty' in key else 1)*
             (np.asarray(gates) if name in ('track','music') else 1.) for name,key in WEIGHTS.items()}
    total = sum(rates.values())
    results = []
    for i,row in enumerate(trace):
        music_raw = dict(beat_definition='bumi.metrics._derive_motion_beats+_beat_alignment',
            beat_window_count=int(beat_counts[i]),beat_window_complete=bool(beat_counts[i]==reward.beat_steps),
            beat_valid=bool(music_count[i]),intensity_valid=True,activity=copy.deepcopy(activities[i]),
            beat_alignment=float(beats[i]),intensity_alignment=intensity[i])
        if beat_counts[i]<reward.beat_steps: music_raw['beat_reason']='insufficient_causal_history'
        else:
            music_raw.update(music_beat_count=int(music_count[i]),motion_beat_count=int(motion_count[i]))
            if not music_count[i]: music_raw['beat_reason']='no_music_beats'
        components = {}
        for name,key in WEIGHTS.items():
            primitive = row['reward_primitives']['components'][name] if name!='music' else None
            gate = gates[i] if name in ('track','music') else 1.
            components[name] = dict(enabled=True,valid=True if primitive is None else bool(primitive['valid']),
                raw=music_raw if primitive is None else primitive['raw'],
                normalized=dict(beat=float(beats[i]),intensity=intensity[i]) if primitive is None else primitive['normalized'],
                score=float(scores[name][i]),weight=cfg[key]*(-1 if 'penalty' in key else 1),gate=gate,
                activity_gate=gate,weighted_rate=float(rates[name][i]),integrated_reward=float(cfg['dt']*rates[name][i]))
        power_info = dict(valid=True,sampling='four_physics_substeps',
            semantics='PD_estimated_mechanical_power_proxy_not_electrical_power_or_energy',
            sampling_synchronized=all(p['sampling_synchronized'] for p in power[i]),
            sampling_note='PD torque estimate and velocity may come from opposite boundaries of a physics substep; inspect each timestamp',
            samples=power[i],mean_w=float(np.mean([p['mean_w'] for p in power[i]])),
            mean_sum_w=float(np.mean([p['sum_w'] for p in power[i]])),max_w=max(p['max_w'] for p in power[i]),reward_weight=0.)
        results.append(dict(version='stage9.execution_reward.v2',tick=int(ticks[i]),episode_id=str(episodes[i]),
            reward=float(cfg['dt']*total[i]),reward_rate=float(total[i]),reward_is_integrated=True,dt_s=cfg['dt'],
            transition_valid=True,errors=[],components=components,activity=activities[i],
            diagnostics=dict(power=power_info,consistency=dict(valid=True,
                raw={k:float(v[i]) for k,v in consistency_raw.items()},
                normalized={k:float(v[i]) for k,v in consistency_norm.items()},
                tolerances=copy.deepcopy(cfg['consistency']),reward_weight=0.)),scales=copy.deepcopy(cfg['scales'])))
    return results, new
