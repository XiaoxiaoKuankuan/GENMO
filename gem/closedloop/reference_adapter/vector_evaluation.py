"""向量环境中的固定真实评价与独立困难验证采样。

每首评价歌曲从相同固定初态、无观测噪声开始，仅记录首个真实episode直到失败或
歌曲结束；已结束环境的后续复位不会给本episode补状态或音乐分数。所有统计来自
真实物理状态，六项Tracking、根误差、动作幅度、jerk和求解器接触脚滑均可追溯。
评价好坏仅写报告，不触发训练回滚或停机；明确错误的物理方向由独立审计处理。
"""

import numpy as np
import torch

from .evaluation import beat_report
from .vector_audit import contact_tensors


@torch.no_grad()
def evaluate_physical(collector, *, zero=False, split='evaluation'):
    bank = collector.bank
    selected = bank.by_split[split]
    count = len(selected)
    songs = selected[torch.arange(collector.num_envs, device=bank.device)%count]
    frames = torch.zeros_like(songs)
    collector.reset(songs=songs, frames=frames, perturb=False)
    initial = collector.state['state'][:count].clone()
    active = torch.ones(count, device=bank.device, dtype=torch.bool)
    states, live, raws, weighted, feet, times = [], [], [], [], [], []
    end_failed = torch.zeros_like(active)
    foot_names = tuple(collector.control.bundle.spec.tracked_bodies)
    foot_indices = [foot_names.index(n) for n in ('l_ankle_roll_link', 'r_ankle_roll_link')]
    while bool(active.any()):
        row = collector.step(zero=zero, auto_reset=True, split=split, perturb=False, contact_count=count)
        states.append(row['end']['state'][:count].clone())
        live.append(active.clone())
        raws.append({k: v[:count].clone() for k, v in row['raw'].items() if isinstance(v, torch.Tensor)})
        weighted.append({k: v[:count].clone() for k, v in row['weighted'].items()})
        feet.append(row['contacts'][1])
        end_failed |= active & row['failed'][:count]
        active &= ~row['done'][:count]
        times.append(row['timing'])
    states, live, feet = torch.stack(states), torch.stack(live), torch.stack(feet)
    raw_all = {k: torch.stack([r[k] for r in raws]) for k in raws[0]}
    weighted_all = {k: torch.stack([r[k] for r in weighted]) for k in weighted[0]}
    reports = []
    for i, song in enumerate(selected.tolist()):
        mask = live[:, i]
        n = int(mask.sum())
        state = torch.cat((initial[i:i+1], states[:n, i]))
        q, qd = state[:, 13:34], state[:, 34:]
        terms = {k: float(v[:n, i].float().mean()) for k, v in raw_all.items() if v.ndim == 2}
        velocities = raw_all['tracked_body_v'][:n, i, foot_indices, :2].norm(dim=-1)
        contact = feet[:n, i]
        slip = float(velocities[contact].mean()) if bool(contact.any()) else 0.
        reports.append(dict(task=bank.tasks[song]['id'], song=bank.tasks[song]['song_id'], split=split,
            steps=n, reason='physical_failure' if bool(end_failed[i]) else 'music_end',
            full_music_complete=not bool(end_failed[i]), physical_failure=bool(end_failed[i]),
            failure_time_s=n*.02 if bool(end_failed[i]) else None,
            sm_terms=terms, weighted_step_terms={k: float(v[:n, i].mean()) for k, v in weighted_all.items()},
            root_position_rmse_m=terms['error_root_pos']**.5, root_orientation_rmse_rad=terms['error_root_ori']**.5,
            music=beat_report(qd, torch.arange(n+1)*12, bank.audio[song]),
            mean_abs_joint_speed=float(qd.abs().mean()), joint_range_mean=float((q.amax(0)-q.amin(0)).mean()),
            acceleration_rms=float((qd.diff(dim=0)/.02).square().mean().sqrt()),
            jerk_rms=float((qd.diff(n=2, dim=0)/.02**2).square().mean().sqrt()) if n > 1 else None,
            contact_foot_slip_mps=slip, foot_slip_status='solver_contact_measured_shared_reward_fk',
            timing_ms={k: {f'p{p}': float(np.percentile([t[k] for t in times[:n]], p)) for p in (50, 95, 99)} for k in times[0]},
            latency_scope=f'{collector.num_envs}_environments_batched_physics_not_single_robot_deployment'))
    return dict(episodes=reports, completion_rate=sum(r['full_music_complete'] for r in reports)/len(reports),
        mean_tracking=sum(r['sm_terms']['track'] for r in reports)/len(reports),
        mean_beat_f1=sum(r['music']['f1'] or 0. for r in reports)/len(reports),
        zero_adapter=zero, seed=42, observation_noise=False, num_envs=collector.num_envs,
        evidence='actual_physics_fixed_full_song_until_real_failure_or_music_end',
        ranking='completion_then_original_nominal_tracking_then_music')


@torch.no_grad()
def collect_validation(collector, replay, steps=300, *, zero=False):
    # 整个仿真临时切换到隔离验证任务；训练旧片段仅行政结束，不能标为MC终止。
    collector.reset(split='validation', random_start=True, perturb=True)
    ids = torch.arange(replay.num_envs, device=collector.bank.device)
    for _ in range(steps):
        collector.step(replay, replay_ids=ids, zero=zero, split='validation', perturb=True)
    return dict(actual_transitions=steps*collector.num_envs, retained_transitions=steps*replay.num_envs,
                completed_segments=replay.completed_segments, head_truncated_terminals=replay.head_truncated_terminals)
