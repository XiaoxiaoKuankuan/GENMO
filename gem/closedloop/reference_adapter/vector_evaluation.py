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
def evaluate_physical(collector, *, zero=False, split='evaluation', background_replay=None):
    bank = collector.bank
    prior_diagnostics = collector.control.detailed_reference
    collector.control.detailed_reference = True
    selected = bank.by_split[split]
    count = len(selected)
    songs = selected[torch.arange(collector.num_envs, device=bank.device)%count]
    frames = torch.zeros_like(songs)
    if background_replay is not None:
        if background_replay.num_envs != collector.num_envs-count:
            raise ValueError('评价背景Replay只能包含隔离评价机器人以外的训练环境')
        songs[count:], frames[count:] = collector.sampling(collector.num_envs-count, 'train')
    collector.reset(songs=songs, frames=frames, perturb=False)
    initial = collector.state['state'][:count].clone()
    active = torch.ones(count, device=bank.device, dtype=torch.bool)
    states, live, raws, weighted, feet, times = [], [], [], [], [], []
    end_failed = torch.zeros_like(active)
    foot_names = tuple(collector.control.bundle.spec.tracked_bodies)
    foot_indices = [foot_names.index(n) for n in ('l_ankle_roll_link', 'r_ankle_roll_link')]
    while bool(active.any()):
        row = collector.step(background_replay, zero=zero, auto_reset=True,
            split='train' if background_replay is not None else split, perturb=False, contact_count=count,
            replay_ids=torch.arange(count, collector.num_envs, device=bank.device) if background_replay is not None else None)
        if background_replay is not None:
            collector.real_by_role['train'] -= int(active.sum())
            collector.real_by_role[split] = collector.real_by_role.get(split, 0)+int(active.sum())
        states.append(row['end']['state'][:count].clone())
        live.append(active.clone())
        raws.append({k: v[:count].clone() for k, v in row['raw'].items() if isinstance(v, torch.Tensor)})
        weighted.append({k: v[:count].clone() for k, v in row['weighted'].items()})
        feet.append(row['contacts'][1])
        end_failed |= active & row['failed'][:count]
        active &= ~row['done'][:count]
        times.append(row['timing'])
        if len(times) % 256 == 0:
            print(f'真实整歌评价 split={split} zero={zero} vector_steps={len(times)} active={int(active.sum())}', flush=True)
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
        from .reference_constraints import quantiles
        distributions = {k: quantiles(v[:n, i].norm(dim=-1) if k in
            ('reference_root_v', 'reference_root_a', 'reference_root_w', 'reference_root_alpha',
             'reference_goal_root_v', 'reference_goal_root_a', 'reference_goal_root_w', 'reference_goal_root_alpha') else v[:n, i])
            for k, v in raw_all.items() if k.startswith('reference_') and v.ndim > 2}
        per_joint = {name: {key: quantiles(raw_all['reference_'+key][:n, i, ..., j]) for key in ('q', 'qd', 'qdd', 'goal_qd', 'goal_qdd')}
                     for j, name in enumerate(collector.control.bundle.spec.joint_names)}
        # 失败之后的未执行时段只给累计Tracking分母贡献，不伪造执行状态/音乐分数。
        planned_steps = int(bank.ends[song])
        normalized_tracking = float(raw_all['track'][:n, i].sum()/planned_steps)
        names = [k for k in ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori',
                             'root_non_saturating', 'magnitude', 'action_change') if k in raw_all]
        samples = torch.stack([raw_all[k][:n, i] for k in names]+[(velocities*contact).mean(-1)], -1).float()
        centered = samples-samples.mean(0)
        cov = centered.T@centered/max(n, 1)
        corr = cov/(cov.diag().sqrt()[:, None]*cov.diag().sqrt()[None, :]).clamp_min(1e-12)
        reports.append(dict(task=bank.tasks[song]['id'], song=bank.tasks[song]['song_id'], split=split,
            steps=n, reason='physical_failure' if bool(end_failed[i]) else 'music_end',
            planned_steps=planned_steps, fraction_of_song_executed=n/planned_steps,
            full_horizon_tracking=normalized_tracking, reference_distributions=distributions,
            reference_per_joint=per_joint, reference_stat_units='q rad; qd rad/s; qdd rad/s2; root v/a norm m/s,m/s2; root w/alpha norm rad/s,rad/s2',
            reference_derivative_scheme='plain=forward_segments_including_seam_and_halo; goal_=actual_Actor_central_difference_targets',
            correlation_names=names+['foot_slip'], correlations=corr.cpu().tolist(),
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
    collector.control.detailed_reference = prior_diagnostics
    return dict(episodes=reports, completion_rate=sum(r['full_music_complete'] for r in reports)/len(reports),
        mean_full_horizon_tracking=sum(r['full_horizon_tracking'] for r in reports)/len(reports),
        mean_root_position_rmse_m=sum(r['root_position_rmse_m'] for r in reports)/len(reports),
        mean_jerk_rms=sum(r['jerk_rms'] or 0 for r in reports)/len(reports),
        mean_tracking=sum(r['sm_terms']['track'] for r in reports)/len(reports),
        mean_beat_f1=sum(r['music']['f1'] or 0. for r in reports)/len(reports),
        zero_adapter=zero, seed=42, observation_noise=False, num_envs=collector.num_envs,
        measured_evaluation_envs=count, concurrent_train_replay_envs=background_replay.num_envs if background_replay is not None else 0,
        evidence='actual_physics_fixed_full_song_until_real_failure_or_music_end',
        ranking='joint_completion_root_rmse_full_horizon_tracking_jerk_then_music')


@torch.no_grad()
def collect_validation(collector, replay, steps=300, *, zero=False):
    # 整个仿真临时切换到隔离验证任务；训练旧片段仅行政结束，不能标为MC终止。
    collector.reset(split='validation', random_start=True, perturb=True)
    ids = torch.arange(replay.num_envs, device=collector.bank.device)
    for _ in range(steps):
        collector.step(replay, replay_ids=ids, zero=zero, split='validation', perturb=True)
    return dict(actual_transitions=steps*collector.num_envs, retained_transitions=steps*replay.num_envs,
                completed_segments=replay.completed_segments, head_truncated_terminals=replay.head_truncated_terminals)
