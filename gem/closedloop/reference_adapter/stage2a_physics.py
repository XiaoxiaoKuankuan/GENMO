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
from .precision import errors, calibrate, ERROR_NAMES
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
    error_rows = {k: [] for k in ERROR_NAMES}
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
                # 仅训练集真实执行误差用于挑选难例，不添加任何Adapter输入。
                scores.append((err['joint_position'].abs().mean(-1)/.1
                    +.2*err['body_orientation'].mean(-1)/.15
                    +.5*err['joint_velocity'].abs().mean(-1)/.7
                    +.5*err['body_angular_velocity'].mean(-1)/1.1)[valid])
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
        sampling='zero_policy_real_q_orientation_qd_body_w_error_train_only',
        score_scales=dict(q=.1, orientation=.15, qd=.7, body_w=1.1),
        score_weights=dict(q=1., orientation=.2, qd=.5, body_w=.5))


@torch.no_grad()
def install_repeated_start(collector, source, repeats):
    count = len(source['state'])*repeats
    if count > collector.num_envs:
        raise ValueError('真实配对环境数不足')
    # 空余环境仍真实采集当前策略；前count个为同一起点的独立物理条件。
    if collector.state is None:
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
    raw_reference_series, goal_series, derivative_series, foot_series, slip_series = [], [], [], [], []
    foot_names = tuple(control.bundle.spec.tracked_bodies)
    foot_ids = [foot_names.index(n) for n in ('l_ankle_roll_link', 'r_ankle_roll_link')]
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
            raw_reference_series.append(row['proposal']['raw_residual'][:count].clone())
            goal_series.append(row['proposal']['goal'][:count].clone())
            derivative_series.append({k: v[:count].clone() for k, v in control.reference_diagnostics(row['start'], row['proposal']).items()})
            foot_series.append(row['contacts'][1])
            slip_series.append(row['raw']['tracked_body_v'][:count, foot_ids, :2].norm(dim=-1))
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
            raw_references=torch.stack(raw_reference_series), goals=torch.stack(goal_series),
            reference_derivatives={k: torch.stack([v[k] for v in derivative_series]) for k in derivative_series[0]},
            foot_contact=torch.stack(foot_series), foot_speed=torch.stack(slip_series),
            conditions=variant.conditions, rows=rows, adapter_identity=parameter_identity(control.adapter),
            offline_selection_only=True, policy_optimization='WM_BPTT_only')
        return rows, data
    finally:
        collector.control = control


@torch.no_grad()
def collect_reference_response(collector, source, *, horizon=64, amplitude=.005, replay=None, split='train',
                               intervention_steps=1):
    """同起点正负参考干预，按声明的1/8/32步持续时间采集，部署策略不添加控制器。"""
    control = collector.control
    if not bool(torch.isin(source['song'], control.bank.by_split[split]).all()):
        raise ValueError('参考响应起点与声明的隔离数据划分不符')
    if replay is not None and split != 'train':
        raise ValueError('隔离参考响应不能进入训练Replay')
    if not bool((source['frame']+horizon < control.bank.ends[source['song']]).all()):
        raise ValueError('参考响应窗口越过歌曲终点；不能补齐或外推真实状态')
    count, mismatch = install_repeated_start(collector, source, 4)
    groups = len(source['state'])
    choices = torch.as_tensor([intervention_steps] if isinstance(intervention_steps, int) else intervention_steps,
                              device=collector.device, dtype=torch.long)
    if not len(choices) or bool(((choices < 1) | (choices > horizon)).any()):
        raise ValueError('参考干预时长必须落在实际采集窗口内')
    duration = choices[torch.arange(groups, device=collector.device)%len(choices)]
    start = detach_state(select_state(collector.state, slice(0, count)))
    noise = torch.randn(groups, 10, 27, device=collector.device, generator=collector.generator)
    # 方向是独立十帧；可执行性由同一QP负责，不用另一套低通探索控制器。
    noise = noise/noise.flatten(1).norm(dim=-1)[:, None, None].clamp_min(1e-12)*amplitude
    noise[..., :2] = 0
    offset = torch.zeros(collector.num_envs, 10, 27, device=collector.device)
    offset[2:count:4], offset[3:count:4] = noise, -noise
    states, actions, switches, reliable = [], [], [], []
    live = torch.ones(count, dtype=torch.bool, device=collector.device)
    previous_contact = None
    for t in range(horizon):
        held = offset.clone()
        held[:count] *= (t < duration.repeat_interleave(4))[:, None, None]
        row = collector.step(auto_reset=True, split=split, perturb=False,
                             residual_offset=held, contact_count=count)
        current = row['end']['state'][:count].reshape(groups, 4, 55)
        action = row['proposal']['action'][:count].reshape(groups, 4, 21)
        contact = row['contacts'][0].reshape(groups, 4, -1)
        change = (contact[:, 2] != contact[:, 3]).any(-1)
        if previous_contact is not None:
            change |= (contact != previous_contact).any((1, 2))
        repeat = (current[:, 0]-current[:, 1]).abs().amax(-1)
        size = (action[:, 2]-action[:, 3]).square().mean(-1).sqrt()
        live &= ~row['done'][:count]
        good = ((mismatch <= 1e-5) & (repeat <= 1e-4) & ~change & (size >= 1e-4)
                & live.reshape(groups, 4).all(-1))
        if reliable:
            good &= reliable[-1]
        states.append(current); actions.append(action); switches.append(change); reliable.append(good)
        previous_contact = contact
        # 可选辅助真实转移仍保留实际episode/time；不把不同条件拼成连续轨迹。
        if replay is not None:
            replay.add(row['start'], row['proposal']['action'], row['end']['state'], row['reward'],
                       row['continuation'], row['done'], row['failed'], collector.policy_version)
    states, actions = torch.stack(states, 1), torch.stack(actions, 1)
    valid, switch = torch.stack(reliable, 1), torch.stack(switches, 1)
    return dict(origin='current_policy_real_reference_response', split=split,
        state=start['state'][::4], xi=start['xi'][::4], song=source['song'], frame=source['frame'],
        plus_action=actions[:, :, 2], minus_action=actions[:, :, 3],
        plus_next=states[:, :, 2], minus_next=states[:, :, 3], reliable=valid, contact_switch=switch,
        policy_version=torch.full_like(source['song'], collector.policy_version),
        intervention_steps=duration, intervention_kind=torch.zeros_like(duration),
        initial_mismatch=mismatch, repeated_states=states[:, :, :2],
        rejected=dict(unreliable_steps=int((~valid).sum()), contact_switch_steps=int(switch.sum()),
                      initial_mismatch=int((mismatch > 1e-5).sum())),
        adapter_sha256=parameter_identity(control.adapter),
        response_semantics='same_real_reset_start_current_policy_closed_loop_real_action_sequences')

