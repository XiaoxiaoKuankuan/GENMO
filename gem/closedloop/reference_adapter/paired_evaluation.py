"""已有Adapter的无探索真实物理配对评价，不执行任何网络或归一化更新。

从固定歌曲初态重置，原始GENMO缓存始终是误差目标。仅对尚未结束的环境批量
运行Adapter/Actor，已结束环境不会向报告补状态或补音乐奖励。正式入口直接使用
有界网络参考，以音乐结束或真实物理失败终止；旧投影审计仍单独记录参考拒绝。
报告实际执行前缀与共同评价预算两种Tracking，保存q/qd/姿态误差、真实接触脚滑、
jerk、完成率、回报与初态/物理参数身份。短评价明确行政截断，不能称整歌完成。
"""

import hashlib
import time

import torch

from .evaluation import beat_report
from .vector_control import select_state
from .vector_audit import contact_tensors
from .rewards import sm_mdp_contract
from .tasks import condition_kind, modality_metrics


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def merge_fixed_reports(reports):
    """评价任务多于物理环境时合并完整episode，按实际样本聚合而非平均批次均值。"""
    result = dict(reports[0])
    rows = [row for report in reports for row in report['episodes']]
    result.update(episodes=rows, by_modality=modality_metrics(rows), measured_evaluation_envs=len(rows),
        completion_rate=sum(r['full_task_complete'] for r in rows)/len(rows),
        mean_tracking=sum(r['sm_terms'].get('track', 0.) for r in rows)/len(rows))
    for field, source in (
            ('mean_full_horizon_tracking', 'full_horizon_tracking'),
            ('mean_root_position_rmse_m', 'root_position_rmse_m'), ('mean_root_xy_rmse_m', 'root_xy_rmse_m'),
            ('mean_jerk_rms', 'jerk_rms'), ('mean_joint_q_rmse_rad', 'joint_q_rmse_rad'),
            ('mean_joint_qd_rmse_rad_s', 'joint_qd_rmse_rad_s'),
            ('mean_body_orientation_rmse_rad', 'body_orientation_rmse_rad'),
            ('mean_contact_foot_slip_mps', 'contact_foot_slip_mps')):
        values = [row[source] for row in rows if row.get(source) is not None]
        result[field] = sum(values)/len(values) if values else None
    beats = [row['music'].get('f1') for row in rows if row.get('condition_kind', 'music') == 'music']
    beats = [v for v in beats if v is not None]
    result['mean_beat_f1'] = sum(beats)/len(beats) if beats else None
    for key in ('constraint_failure_count', 'wall_seconds', 'measured_transitions', 'physics_transitions'):
        result[key] = sum(report[key] for report in reports)
    result['constraint_failures'] = [r for report in reports for r in report['constraint_failures']]
    return result


def _feasible_evaluation_proposal(control, full_state, ids, *, zero):
    """只剔除失败环境后重算批量提案；求解异常也必须留下逐episode证据。"""
    failures = []
    while len(ids):
        z = select_state(full_state, ids)
        try:
            proposal = control.propose(z, zero=zero)
            certificate = control.constraint_certificate(z, proposal)
            bad = (~certificate['feasible'] if certificate is not None else
                   torch.zeros(len(ids), dtype=torch.bool, device=ids.device))
            error_text = 'reference_certificate_failed'
        except RuntimeError as error:
            if not hasattr(error, 'bad_mask'):
                raise
            bad = error.bad_mask.to(device=ids.device, dtype=torch.bool)
            certificate = getattr(error, 'certificate', {})
            error_text = str(error)
        if bad.shape != ids.shape:
            raise RuntimeError('投影失败mask不匹配实际评价环境，禁止推进物理')
        if not bool(bad.any()):
            return ids, z, proposal, failures
        for local in bad.nonzero().flatten().tolist():
            failures.append(dict(env=int(ids[local]), frame=int(z['frame'][local]), error=error_text,
                certificate={k: v[local].detach().cpu().tolist() for k,v in certificate.items()
                    if isinstance(v, torch.Tensor) and v.ndim and len(v)==len(ids)}))
        ids = ids[~bad]
    return ids, None, None, failures


@torch.no_grad()
def evaluate_fixed(collector, *, zero=False, split='evaluation', max_steps=None, trajectory_dir=None):
    control, bank = collector.control, collector.bank
    selected = bank.by_split[split]
    count = len(selected)
    if not count:
        raise ValueError('评价任务数必须非零')
    if count > collector.num_envs:
        reports = []
        try:
            for offset in range(0, count, collector.num_envs):
                bank.by_split[split] = selected[offset:offset+collector.num_envs]
                destination = None
                if trajectory_dir is not None:
                    from pathlib import Path
                    destination = Path(trajectory_dir)/f'batch_{offset:04d}'
                reports.append(evaluate_fixed(collector, zero=zero, split=split, max_steps=max_steps,
                                              trajectory_dir=destination))
        finally:
            bank.by_split[split] = selected
        return merge_fixed_reports(reports)
    songs = selected[torch.arange(collector.num_envs, device=bank.device)%count]
    collector.reset(songs=songs, frames=torch.zeros_like(songs), perturb=False)
    initial = collector.state['state'][:count].clone()
    initial_xi = collector.state['xi'][:count].clone()
    active = torch.ones(count, dtype=torch.bool, device=bank.device)
    reasons = ['']*count
    states, lives, records, weighted, contacts = [], [], [], [], []
    reference_frames, action_frames = [], []
    constraint_failures = []
    prior_diagnostics = control.detailed_reference
    control.detailed_reference = False
    foot_names = tuple(control.bundle.spec.tracked_bodies)
    feet = [foot_names.index(n) for n in ('l_ankle_roll_link', 'r_ankle_roll_link')]
    begin = time.perf_counter()
    try:
        while bool(active.any()) and (max_steps is None or len(states) < max_steps):
            ids = active.nonzero().flatten()
            ids, z, proposal, failures = _feasible_evaluation_proposal(
                control, collector.state, ids, zero=zero)
            for failure in failures:
                env_id = failure['env']
                reasons[env_id] = 'constraint_failure'
                active[env_id] = False
                failure['task'] = bank.tasks[int(songs[env_id])]['id']
                constraint_failures.append(failure)
            if not len(ids):
                break
            action = initial.new_zeros(collector.num_envs, 21)
            action[ids] = proposal['action']
            if not bool(torch.isfinite(action[ids]).all()):
                raise FloatingPointError('评价控制动作出现非有限值，未推进物理')
            nxt = collector.env.step_physics(action)
            if not bool(nxt[ids].isfinite().all()):
                raise FloatingPointError('真实评价状态非有限')
            end, reward, raw, terms, _, done, failed = control.advance(z, proposal, nxt[ids])
            target = bank.ref(z['song'], z['frame']+1)
            if trajectory_dir is not None:
                from supermimic.rotations import mat_to_quat
                corrected = proposal['future']
                poses = torch.cat((corrected.root_p[:, 0], mat_to_quat(corrected.root_R[:, 0]),
                                   corrected.q[:, 0]), -1)
                frame = initial.new_zeros(count, 28)
                frame[ids] = poses
                reference_frames.append(frame)
                action_frames.append(action[:count].clone())
            raw['joint_q_mse'] = (nxt[ids, 13:34]-target.q).square().mean(-1)
            raw['joint_qd_mse'] = (nxt[ids, 34:]-target.qd).square().mean(-1)
            raw['total_objective'] = reward
            raw['foot_speed_xy'] = raw['tracked_body_v'][:, feet, :2].norm(dim=-1)
            _, contact = contact_tensors(collector.env, count)
            row = {}
            for name, value in raw.items():
                if value.ndim == 1 or name == 'foot_speed_xy':
                    out = value.new_zeros(count, *value.shape[1:])
                    out[ids] = value
                    row[name] = out
            term_row = {}
            for name, value in terms.items():
                out = value.new_zeros(count); out[ids] = value; term_row[name] = out
            states.append(nxt[:count].clone())
            lives.append(active.clone())
            records.append(row); weighted.append(term_row); contacts.append(contact)
            for key in collector.state:
                collector.state[key][ids] = end[key]
            for local in done.nonzero().flatten().tolist():
                env_id = int(ids[local])
                task = bank.tasks[int(songs[env_id])]
                reasons[env_id] = ('physical_failure' if bool(failed[local]) else
                                   'task_end' if condition_kind(task) == 'text' else 'music_end')
            active[ids[done]] = False
            collector.vector_steps += 1
            collector.real_transitions += collector.num_envs
            collector.real_by_role[split] = collector.real_by_role.get(split, 0)+len(ids)
            if len(states) % 256 == 0:
                print(f'固定真实评价 split={split} zero={zero} steps={len(states)} active={int(active.sum())}', flush=True)
        for i in active.nonzero().flatten().tolist():
            reasons[i] = 'administrative_truncation'
    finally:
        control.detailed_reference = prior_diagnostics
    elapsed = time.perf_counter()-begin
    state_all = torch.stack(states) if states else initial.new_empty(0, count, 55)
    live = torch.stack(lives) if lives else active.new_empty(0, count)
    raw_all = {k: torch.stack([r[k] for r in records]) for k in records[0]} if records else {}
    term_all = {k: torch.stack([r[k] for r in weighted]) for k in weighted[0]} if weighted else {}
    contact_all = torch.stack(contacts) if contacts else active.new_empty(0, count, 2)
    if trajectory_dir is not None:
        from .trajectory_export import export_trajectories
        export_trajectories(trajectory_dir, collector, selected, initial, initial_xi, state_all, live,
                            reference_frames, action_frames, raw_all, reasons, zero=zero)
    episodes = []
    for i, song in enumerate(selected.tolist()):
        n = int(live[:, i].sum())
        state = torch.cat((initial[i:i+1], state_all[:n, i]))
        q, qd = state[:, 13:34], state[:, 34:]
        terms = {k: float(v[:n, i].float().mean()) for k, v in raw_all.items() if v.ndim == 2} if n else {}
        planned = int(bank.ends[song]); budget = min(planned, max_steps) if max_steps is not None else planned
        slip_mask = contact_all[:n, i]
        slip = float(raw_all['foot_speed_xy'][:n, i][slip_mask].mean()) if n and bool(slip_mask.any()) else 0.
        tracking_sum = float(raw_all['track'][:n, i].sum()) if n else 0.
        from supermimic.mdp import unflatten_state
        final_p, final_R, *_ = unflatten_state(state[-1:])
        final_nominal = bank.ref(songs[i:i+1], songs.new_tensor([n]))
        height_error = float((final_p[..., 2]-final_nominal.root_p[..., 2]).abs()[0])
        gravity_z_error = float((final_R[..., 2, 2]-final_nominal.root_R[..., 2, 2]).abs()[0])
        failure_conditions = dict(root_height_error_m=height_error, gravity_projection_z_difference=gravity_z_error,
            root_height_exceeded=height_error > control.runtime.mdp.cfg.term_height,
            gravity_projection_z_exceeded=gravity_z_error > control.runtime.mdp.cfg.term_tilt)
        kind = condition_kind(bank.tasks[song])
        episodes.append(dict(task=bank.tasks[song]['id'], song=bank.tasks[song]['song_id'], split=split,
            condition_kind=kind, source_group_id=bank.tasks[song].get('source_group_id', bank.tasks[song]['song_id']),
            generation_contract=bank.task_contracts[song],
            steps=n, planned_steps=planned, evaluation_budget_steps=budget, reason=reasons[i],
            fraction_of_song_executed=n/planned, full_music_complete=reasons[i]=='music_end',
            full_task_complete=reasons[i] in ('music_end', 'task_end'), fraction_of_task_executed=n/planned,
            physical_failure=reasons[i]=='physical_failure', constraint_failure=reasons[i]=='constraint_failure',
            failure_conditions_at_last_executed_state=failure_conditions,
            failure_time_s=n*.02 if reasons[i] in ('physical_failure', 'constraint_failure') else None,
            initial_state_sha256=tensor_digest(initial[i]), physical_xi_sha256=tensor_digest(initial_xi[i]),
            sm_terms=terms, weighted_step_terms={k: float(v[:n, i].mean()) for k, v in term_all.items()} if n else {},
            full_horizon_tracking=tracking_sum/max(budget, 1), full_song_tracking=tracking_sum/max(planned, 1),
            root_position_rmse_m=terms['error_root_pos']**.5 if n else None,
            root_xy_rmse_m=terms['root_xy_squared_m']**.5 if n and 'root_xy_squared_m' in terms else None,
            root_xy_heading_bias_m=[terms.get('root_xy_heading_x_m'), terms.get('root_xy_heading_y_m')],
            root_orientation_rmse_rad=terms['error_root_ori']**.5 if n else None,
            joint_q_rmse_rad=terms['joint_q_mse']**.5 if n else None,
            joint_qd_rmse_rad_s=terms['joint_qd_mse']**.5 if n else None,
            body_orientation_rmse_rad=terms['error_body_ori']**.5 if n else None,
            body_orientation_absolute_mean_rad=terms.get('absolute_body_orientation'),
            total_return=float(raw_all['total_objective'][:n, i].sum()) if n else 0.,
            discounted_return=float((raw_all['total_objective'][:n, i]*.99**torch.arange(n,device=bank.device)).sum()) if n else 0.,
            music=beat_report(qd, torch.arange(n+1)*12, bank.audio[song]) if n else {},
            mean_abs_joint_speed=float(qd.abs().mean()), joint_range_mean=float((q.amax(0)-q.amin(0)).mean()),
            jerk_rms=float((qd.diff(n=2, dim=0)/.02**2).square().mean().sqrt()) if n>1 else None,
            contact_foot_slip_mps=slip, foot_slip_status='solver_contact_measured_shared_reward_fk'))
    def average(key):
        values=[r[key] for r in episodes if r[key] is not None]
        return sum(values)/len(values) if values else None
    groups = modality_metrics(episodes)
    music_scores = [r['music'].get('f1') for r in episodes if r['condition_kind'] == 'music']
    music_scores = [score for score in music_scores if score is not None]
    return dict(episodes=episodes, by_modality=groups,
        completion_rate=sum(r['full_task_complete'] for r in episodes)/count,
        mean_full_horizon_tracking=average('full_horizon_tracking'), mean_root_position_rmse_m=average('root_position_rmse_m'),
        mean_root_xy_rmse_m=average('root_xy_rmse_m'),
        mean_jerk_rms=average('jerk_rms'), mean_tracking=sum(r['sm_terms'].get('track',0.) for r in episodes)/count,
        mean_joint_q_rmse_rad=average('joint_q_rmse_rad'), mean_joint_qd_rmse_rad_s=average('joint_qd_rmse_rad_s'),
        mean_body_orientation_rmse_rad=average('body_orientation_rmse_rad'), mean_contact_foot_slip_mps=average('contact_foot_slip_mps'),
        mean_beat_f1=sum(music_scores)/len(music_scores) if music_scores else None,
        constraint_failure_count=len(constraint_failures), constraint_failures=constraint_failures,
        zero_adapter=zero, max_steps=max_steps, seed=42, observation_noise=False, exploration_noise=0.,
        num_envs=collector.num_envs, measured_evaluation_envs=count, wall_seconds=elapsed,
        measured_transitions=sum(r['steps'] for r in episodes), physics_transitions=len(states)*collector.num_envs,
        evidence='actual_physics_paired_initial_state_no_exploration',
        reference_execution=('bounded_direct_no_derivative_constraints'
            if control.projection is None and control.constraint_beta == 1. else 'historical_reference_constraint_audit'),
        sm_mdp_contract=sm_mdp_contract(control.runtime), soft_joint_limit_cost=control.soft_joint_limit_cost,
        full_song_evaluation=max_steps is None, full_task_evaluation=max_steps is None)


def paired_comparison(zero, current):
    if zero.get('sm_mdp_contract') != current.get('sm_mdp_contract'):
        raise ValueError('物理评价配对条件不一致: SM奖励尺度或终止标准不同，必须重算零基线')
    baseline={r['task']:r for r in zero['episodes']}
    rows=[]
    metrics=('full_horizon_tracking','root_position_rmse_m','joint_q_rmse_rad','joint_qd_rmse_rad_s',
             'body_orientation_rmse_rad','jerk_rms','contact_foot_slip_mps')
    for row in current['episodes']:
        old=baseline[row['task']]
        if row.get('condition_kind', 'music') != old.get('condition_kind', 'music'):
            raise ValueError('物理评价配对条件不一致: 条件模态不同')
        if row.get('generation_contract') != old.get('generation_contract'):
            raise ValueError('物理评价配对条件不一致: Stage1缓存或条件身份不同')
        for key in ('initial_state_sha256','physical_xi_sha256','evaluation_budget_steps'):
            if row[key]!=old[key]:
                raise ValueError(f'物理评价配对条件不一致: {row["task"]} {key}')
        rows.append(dict(task=row['task'], condition_kind=row.get('condition_kind', 'music'),
            zero_reason=old['reason'],current_reason=row['reason'],
            zero_steps=old['steps'],current_steps=row['steps'],
            differences={k:row[k]-old[k] if row[k] is not None and old[k] is not None else None for k in metrics},
            tracking_six_differences={k:row['sm_terms'][k]-old['sm_terms'][k]
                if k in row['sm_terms'] and k in old['sm_terms'] else None
                for k in ('body_pos','body_ori','body_linvel','body_angvel','root_pos','root_ori')}))
    groups = {}
    for kind in set(zero.get('by_modality', {})) & set(current.get('by_modality', {})):
        a, b = zero['by_modality'][kind], current['by_modality'][kind]
        groups[kind] = {key: b[key]-a[key] if b.get(key) is not None and a.get(key) is not None else None
                       for key in ('completion_rate', 'mean_root_position_rmse_m', 'mean_full_horizon_tracking', 'mean_beat_f1')}
    return dict(cases=rows, by_modality=groups, completion_delta=current['completion_rate']-zero['completion_rate'],
        tracking_delta=current['mean_full_horizon_tracking']-zero['mean_full_horizon_tracking'],
        interpretation='negative errors/slip/jerk are better; positive tracking is better; failed prefixes are not full songs')
