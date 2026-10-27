"""Stage2-A真实物理可补偿性与参考响应采集，不用Oracle最优选择冒充在线策略。

先在训练歌曲的多个合法时刻运行零Adapter，按真实误差保存因果状态和控制历史。
同一真实起点复制为基线、重复、正负参考补偿条件，经过同一投影和冻结Actor后执行
真实物理。复制起点是显式重新初始化的片段，不能称为求解器隐藏状态连续恢复。
接触、重复噪声、初态偏差和失败均逐条件记录。所有候选只读取可观测关节/姿态误差
和已可见名义参考；离线挑选最佳候选仅诊断可补偿空间，不直接监督/更新Adapter。
"""

import copy
import torch
from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_log
from .vector_control import VectorControl, select_state, detach_state
from .vector_audit import contact_tensors
from .precision import errors, calibrate, NAMES
from .update_diagnostics import parameter_identity


@torch.no_grad()
def difficulty_scan(collector, steps=96):
    bank = collector.bank
    ids = torch.arange(collector.num_envs, device=bank.device)
    train = bank.by_split['train']
    songs = train[ids % len(train)]
    fractions = ((ids//len(train)) % 8).float()/8
    frames = (fractions*(bank.ends[songs]-steps-64).clamp_min(0)).long()
    collector.reset(songs=songs, frames=frames, perturb=False)
    candidates, scores = [], []
    error_rows = {k: [] for k in NAMES}
    for step in range(steps):
        row = collector.step(zero=True, split='train', perturb=False)
        nominal = bank.ref(row['end']['song'], row['end']['frame'])
        err = errors(collector.control, row['start'], row['end']['state'], nominal, row['raw'])
        for k, value in err.items():
            error_rows[k].append(value.detach())
        if step >= 16 and step % 8 == 0:
            valid = ~row['done'] & (row['end']['frame']+80 < bank.ends[row['end']['song']])
            if bool(valid.any()):
                candidates.append(detach_state(select_state(row['end'], valid)))
                scores.append(err['joint_position'].abs().mean(-1)[valid]
                              +.2*err['body_orientation'].mean(-1)[valid])
    pool = {k: torch.cat([z[k] for z in candidates]) for k in candidates[0]}
    score = torch.cat(scores)
    selected = []
    for song in train:
        indices = (pool['song'] == song).nonzero().flatten()
        selected.append(indices[score[indices].argsort(descending=True)[:8]])
    selected = torch.cat(selected)
    return select_state(pool, selected), dict(scales=calibrate({k: torch.cat(v) for k, v in error_rows.items()}),
        selected_count=len(selected), songs=len(train), scan_steps=steps, num_envs=collector.num_envs,
        selected_frames=[dict(song=int(s), frame=int(f), error=float(e)) for s, f, e in
                         zip(pool['song'][selected], pool['frame'][selected], score[selected])],
        semantics='real_reset_segments_from_legal_times_not_complete_songs',
        sampling='zero_policy_real_high_error_train_only')


@torch.no_grad()
def install_repeated_start(collector, source, repeats):
    count = len(source['state'])*repeats
    if count > collector.num_envs:
        raise ValueError('真实配对环境数不足')
    # 空余环境仍真实采集当前策略；前count个为同一起点的独立物理条件。
    collector.reset(split='train', perturb=False)
    repeated = {k: v.repeat_interleave(repeats, 0).clone() for k, v in source.items()}
    ids = torch.arange(count, device=collector.device)
    collector.reset(ids, songs=repeated['song'], frames=repeated['frame'], initial=repeated['state'],
                    joint_bias=repeated['xi'][:, 1:22], perturb=False)
    episodes = collector.state['episode'][:count].clone()
    for k, value in repeated.items():
        collector.state[k][:count] = value
    collector.env.xi[:count] = repeated['xi']
    collector.env.a_prev[:count] = repeated['action']
    actual = collector.env.observe()['state'][:count]
    mismatch = (actual-repeated['state']).abs().amax(-1).reshape(-1, repeats).amax(-1)
    collector.state['state'][:count] = actual
    collector.state['episode'][:count] = episodes
    collector.state['start_frame'][:count] = repeated['frame']
    return count, mismatch


class OracleControl(VectorControl):
    conditions = ('zero', 'repeat', 'lead_small', 'lag_small', 'lead', 'lag',
                  'joint_error_plus', 'joint_error_minus', 'orientation_plus', 'orientation_minus',
                  'combined_plus', 'combined_minus', 'lead_old_beta', 'lag_old_beta',
                  'joint_old_beta', 'orientation_old_beta')

    def __init__(self, control, count):
        self.__dict__.update(control.__dict__)
        self.paired_count = count

    def propose(self, z, **kwargs):
        current = self.bank.ref(z['song'], z['frame'])
        future = self.bank.ref(z['song'][:, None], z['frame'][:, None]+self.bank.offset11[1:])
        _, R, _, _, q, _ = unflatten_state(z['state'])
        lead = (future.qd*.02).clamp(-.025, .025)
        joint = ((current.q-q)*.25).clamp(-.025, .025)[:, None].expand(-1, 10, -1)
        orientation = (so3_log(R.transpose(-1, -2)@current.root_R)*.15).clamp(-.008, .008)
        ids = torch.arange(len(q), device=q.device)
        mode = ids % 16
        lead_factor = q.new_tensor([0, 0, .5, -.5, 1, -1, 0, 0, 0, 0, .5, -.5, 1, -1, 0, 0])[mode]
        joint_factor = q.new_tensor([0, 0, 0, 0, 0, 0, 1, -1, 0, 0, .5, -.5, 0, 0, 1, 0])[mode]
        rot_factor = q.new_tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, -1, .5, -.5, 0, 0, 0, 1])[mode]
        active = (ids < self.paired_count).to(q)
        residual = q.new_zeros(len(q), 10, 27)
        residual[..., 6:] = (lead*lead_factor[:, None, None]+joint*joint_factor[:, None, None])*active[:, None, None]
        residual[..., 3:6] = orientation[:, None]*rot_factor[:, None, None]*active[:, None, None]
        result = super().propose(z, zero=True, residual_offset=residual)
        old = (mode >= 12) & (ids < self.paired_count)
        if bool(old.any()):
            from .reference_constraints import constrain_plan
            filtered = constrain_plan(residual, z['left'], z['anchor'], .05)
            result['residual'] = torch.where(old[:, None, None], filtered, result['residual'])
            result['current'], result['future'], result['goal'], result['halo'] = self.reference(z, result['residual'])
            result['action'] = self.bundle.act(z['history'], result['goal'])
        return result


@torch.no_grad()
def oracle_experiment(collector, source, horizons=(8, 16, 32, 64)):
    control = collector.control
    count, mismatch = install_repeated_start(collector, source, 16)
    variant = OracleControl(control, count)
    collector.control = variant
    start = detach_state(select_state(collector.state, slice(0, count)))
    total = start['state'].new_zeros(count)
    tracking = total.clone()
    precision = total.clone()
    active = torch.ones(count, device=collector.device, dtype=torch.bool)
    rows, state_series, action_series, contact_series, reference_series = [], [], [], [], []
    try:
        for t in range(max(horizons)):
            row = collector.step(auto_reset=False, split='oracle_train', contact_count=count)
            total += .99**t*row['reward'][:count]*active
            tracking += .99**t*row['raw']['track'][:count]*active
            precision += .99**t*sum(v[:count] for k, v in row['weighted'].items() if k.startswith('precision_'))*active
            active &= ~row['done'][:count]
            state_series.append(row['end']['state'][:count].clone())
            action_series.append(row['proposal']['action'][:count].clone())
            contact_series.append(row['contacts'][0])
            reference_series.append(row['proposal']['residual'][:count].clone())
            if t+1 in horizons:
                values, tracks, precise = (x.reshape(-1, 16) for x in (total, tracking, precision))
                for i, song in enumerate(source['song'].tolist()):
                    repeat = abs(float(values[i, 1]-values[i, 0]))
                    noise = max(1e-6, repeat*10)
                    repeat_state = float((torch.stack(state_series)[:, i*16+1]-torch.stack(state_series)[:, i*16]).abs().max())
                    for j, name in enumerate(variant.conditions):
                        delta = float(values[i, j]-values[i, 0])
                        contacts = torch.stack(contact_series)
                        action_delta = torch.stack(action_series)[:, i*16+j]-torch.stack(action_series)[:, i*16]
                        rows.append(dict(song=int(song), task=collector.bank.tasks[song]['id'],
                            frame=int(source['frame'][i]), horizon=t+1, candidate=name,
                            objective_delta=delta, tracking_delta=float(tracks[i, j]-tracks[i, 0]),
                            precision_reward_delta=float(precise[i, j]-precise[i, 0]), noise_floor=noise,
                            repeat_state_max_abs=repeat_state, initial_mismatch=float(mismatch[i]),
                            action_rms_delta=float(action_delta.square().mean().sqrt()),
                            contact_changed=bool((contacts[:, i*16+j] != contacts[:, i*16]).any()),
                            real_terminal=not bool(active[i*16+j]),
                            credible_gain=delta > noise and repeat_state <= 1e-4 and float(mismatch[i]) <= 1e-5
                                and bool(active[i*16+j]) and j not in (0, 1) and j < 12))
        data = dict(origin='real_train_oracle_reference_pairs', start=start,
            states=torch.stack(state_series), actions=torch.stack(action_series),
            references=torch.stack(reference_series), contacts=torch.stack(contact_series),
            conditions=variant.conditions, rows=rows, adapter_identity=parameter_identity(control.adapter),
            offline_selection_only=True, policy_optimization='WM_BPTT_only')
        return rows, data
    finally:
        collector.control = control
        collector.state = None

