"""已有Adapter的无探索真实物理配对评价，不执行任何网络或归一化更新。

从固定歌曲初态重置，原始GENMO缓存始终是误差目标。仅对尚未结束的环境批量
运行Adapter/Actor，已结束环境不会向报告补状态或补音乐奖励。参考不可执行时
终止对应评价episode并记录constraint_failure，绝不将该参考送入物理或伪称跌倒。
报告实际执行前缀与共同评价预算两种Tracking，保存q/qd/姿态误差、真实接触脚滑、
jerk、完成率、回报与初态/物理参数身份。短评价明确行政截断，不能称整歌完成。
"""

import hashlib
import time

import torch

from .evaluation import beat_report
from .vector_control import select_state
from .vector_audit import contact_tensors


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def evaluate_fixed(collector, *, zero=False, split='evaluation', max_steps=None):
    control, bank = collector.control, collector.bank
    selected = bank.by_split[split]
    count = len(selected)
    if not count or count > collector.num_envs:
        raise ValueError('评价歌曲数必须非零且不超过真实环境数')
    songs = selected[torch.arange(collector.num_envs, device=bank.device)%count]
    collector.reset(songs=songs, frames=torch.zeros_like(songs), perturb=False)
    initial = collector.state['state'][:count].clone()
    initial_xi = collector.state['xi'][:count].clone()
    active = torch.ones(count, dtype=torch.bool, device=bank.device)
    reasons = ['']*count
    states, lives, records, weighted, contacts = [], [], [], [], []
    constraint_failures = []
    prior_diagnostics = control.detailed_reference
    control.detailed_reference = False
    foot_names = tuple(control.bundle.spec.tracked_bodies)
    feet = [foot_names.index(n) for n in ('l_ankle_roll_link', 'r_ankle_roll_link')]
    begin = time.perf_counter()
    try:
        while bool(active.any()) and (max_steps is None or len(states) < max_steps):
            ids = active.nonzero().flatten()
            z = select_state(collector.state, ids)
            proposal = control.propose(z, zero=zero)
            certificate = control.constraint_certificate(z, proposal)
            if certificate is not None:
                bad = ~certificate['feasible']
                if bool(bad.any()):
                    for local in bad.nonzero().flatten().tolist():
                        env_id = int(ids[local])
                        reasons[env_id] = 'constraint_failure'
                        constraint_failures.append(dict(env=env_id, task=bank.tasks[int(songs[env_id])]['id'],
                            frame=int(z['frame'][local]), certificate={k: v[local].cpu().tolist()
                                for k, v in certificate.items() if isinstance(v, torch.Tensor)}))
                    active[ids[bad]] = False
                    ids = ids[~bad]
                    if not len(ids):
                        break
                    z = select_state(collector.state, ids)
                    proposal = control.propose(z, zero=zero)
                    checked = control.constraint_certificate(z, proposal)
                    if checked is not None and not bool(checked['feasible'].all()):
                        raise RuntimeError('相同评价输入重复投影不一致，不能发送未经确认参考')
            action = initial.new_zeros(collector.num_envs, 21)
            action[ids] = proposal['action']
            nxt = collector.env.step_physics(action)
            if not bool(nxt[ids].isfinite().all()):
                raise FloatingPointError('真实评价状态非有限')
            end, reward, raw, terms, _, done, failed = control.advance(z, proposal, nxt[ids])
            target = bank.ref(z['song'], z['frame']+1)
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
                reasons[env_id] = 'physical_failure' if bool(failed[local]) else 'music_end'
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
    episodes = []
    for i, song in enumerate(selected.tolist()):
        n = int(live[:, i].sum())
        state = torch.cat((initial[i:i+1], state_all[:n, i]))
        q, qd = state[:, 13:34], state[:, 34:]
        terms = {k: float(v[:n, i].mean()) for k, v in raw_all.items() if v.ndim == 2} if n else {}
        planned = int(bank.ends[song]); budget = min(planned, max_steps) if max_steps is not None else planned
        slip_mask = contact_all[:n, i]
        slip = float(raw_all['foot_speed_xy'][:n, i][slip_mask].mean()) if n and bool(slip_mask.any()) else 0.
        tracking_sum = float(raw_all['track'][:n, i].sum()) if n else 0.
        episodes.append(dict(task=bank.tasks[song]['id'], song=bank.tasks[song]['song_id'], split=split,
            steps=n, planned_steps=planned, evaluation_budget_steps=budget, reason=reasons[i],
            fraction_of_song_executed=n/planned, full_music_complete=reasons[i]=='music_end',
            physical_failure=reasons[i]=='physical_failure', constraint_failure=reasons[i]=='constraint_failure',
            failure_time_s=n*.02 if reasons[i] in ('physical_failure', 'constraint_failure') else None,
            initial_state_sha256=tensor_digest(initial[i]), physical_xi_sha256=tensor_digest(initial_xi[i]),
            sm_terms=terms, weighted_step_terms={k: float(v[:n, i].mean()) for k, v in term_all.items()} if n else {},
            full_horizon_tracking=tracking_sum/max(budget, 1), full_song_tracking=tracking_sum/max(planned, 1),
            root_position_rmse_m=terms['error_root_pos']**.5 if n else None,
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
    return dict(episodes=episodes, completion_rate=sum(r['full_music_complete'] for r in episodes)/count,
        mean_full_horizon_tracking=average('full_horizon_tracking'), mean_root_position_rmse_m=average('root_position_rmse_m'),
        mean_jerk_rms=average('jerk_rms'), mean_tracking=sum(r['sm_terms'].get('track',0.) for r in episodes)/count,
        mean_joint_q_rmse_rad=average('joint_q_rmse_rad'), mean_joint_qd_rmse_rad_s=average('joint_qd_rmse_rad_s'),
        mean_body_orientation_rmse_rad=average('body_orientation_rmse_rad'), mean_contact_foot_slip_mps=average('contact_foot_slip_mps'),
        mean_beat_f1=sum(r['music'].get('f1') or 0. for r in episodes)/count,
        constraint_failure_count=len(constraint_failures), constraint_failures=constraint_failures,
        zero_adapter=zero, max_steps=max_steps, seed=42, observation_noise=False, exploration_noise=0.,
        num_envs=collector.num_envs, measured_evaluation_envs=count, wall_seconds=elapsed,
        measured_transitions=sum(r['steps'] for r in episodes), physics_transitions=len(states)*collector.num_envs,
        evidence='actual_physics_paired_initial_state_no_exploration',
        full_song_evaluation=max_steps is None)


def paired_comparison(zero, current):
    baseline={r['task']:r for r in zero['episodes']}
    rows=[]
    metrics=('full_horizon_tracking','root_position_rmse_m','joint_q_rmse_rad','joint_qd_rmse_rad_s',
             'body_orientation_rmse_rad','jerk_rms','contact_foot_slip_mps')
    for row in current['episodes']:
        old=baseline[row['task']]
        for key in ('initial_state_sha256','physical_xi_sha256','evaluation_budget_steps'):
            if row[key]!=old[key]:
                raise ValueError(f'物理评价配对条件不一致: {row["task"]} {key}')
        rows.append(dict(task=row['task'],zero_reason=old['reason'],current_reason=row['reason'],
            zero_steps=old['steps'],current_steps=row['steps'],
            differences={k:row[k]-old[k] if row[k] is not None and old[k] is not None else None for k in metrics},
            tracking_six_differences={k:row['sm_terms'][k]-old['sm_terms'][k]
                if k in row['sm_terms'] and k in old['sm_terms'] else None
                for k in ('body_pos','body_ori','body_linvel','body_angvel','root_pos','root_ori')}))
    return dict(cases=rows,completion_delta=current['completion_rate']-zero['completion_rate'],
        tracking_delta=current['mean_full_horizon_tracking']-zero['mean_full_horizon_tracking'],
        interpretation='negative errors/slip/jerk are better; positive tracking is better; failed prefixes are not full songs')
