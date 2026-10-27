"""当前Adapter经投影和冻结Actor产生的完整真实参考响应监督。

按实际H8/H16/H32/H64长度分桶保存正负响应，短窗口不补齐，长窗口不截成8步。
监督标签只有训练集真实状态、实际动作和物理条件；接触切换只作诊断；重复性不足、终止及
动作变化过小的步骤仍有可靠mask。跨接触变化监督有限幅度响应，不冒充平滑导数。位置/姿态/关节速度、根角速度及现有SM FK的
身体角速度分别归一化和加权，不添加WM预测头，也不增加Adapter奖励或输入。
H64可以保存、审计；是否参与监督由训练入口显式选择，默认仅验证。所有报告绑定
确切WM身份。验证和审计歌曲禁止进入本Replay，统计为零不能冒充长时方向通过。
"""

import torch
from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_log

GROUP_SCALES = dict(root_position=.05, root_orientation=.1, joint_position=.1,
                    joint_velocity=2., root_angular_velocity=.5, body_angular_velocity=1.)
GROUP_WEIGHTS = dict(root_position=.15, root_orientation=.15, joint_position=.35,
                     joint_velocity=.35, root_angular_velocity=.1, body_angular_velocity=.1)
TEMPORAL = {'plus_action', 'minus_action', 'plus_next', 'minus_next', 'reliable',
            'contact_switch', 'repeated_states', 'trajectory_valid', 'legacy_reliable',
            'all_action', 'all_next', 'all_residual', 'all_zero_action', 'all_rewards',
            'all_frame', 'all_episode', 'all_done'}


def body_velocity(kin, state):
    if kin is None:
        raise ValueError('身体角速度响应必须使用现有SuperMimic FK，不能添加替代预测头')
    return kin.forward(*unflatten_state(state))[3]


def response_groups(origin, plus, minus, kin):
    _, R0, *_ = unflatten_state(origin)
    p1, R1, _, w1, q1, qd1 = unflatten_state(plus)
    p2, R2, _, w2, q2, qd2 = unflatten_state(minus)
    local = R0.transpose(-1, -2)
    bw = body_velocity(kin, plus)-body_velocity(kin, minus)
    return dict(root_position=(local@(p1-p2)[..., None]).squeeze(-1),
        root_orientation=so3_log(local@R1)-so3_log(local@R2), joint_position=q1-q2,
        joint_velocity=qd1-qd2, root_angular_velocity=(local@(w1-w2)[..., None]).squeeze(-1),
        body_angular_velocity=(local[:, None]@bw[..., None]).squeeze(-1).flatten(1))


class ReferenceResponseReplay:
    def __init__(self, capacity=8192):
        self.capacity, self.buckets = capacity, {}
        self.total_pairs, self.rejected = 0, {}

    def add(self, data):
        if data.get('origin') != 'current_policy_real_reference_response' or data.get('split') != 'train':
            raise ValueError('参考响应监督只允许当前策略训练歌曲真实配对')
        length = data['plus_action'].shape[1]
        if not 1 <= length <= 64 or data['reliable'].shape != data['plus_action'].shape[:2]:
            raise ValueError('响应必须保留真实的1至64步窗口与对应可靠mask')
        count = len(data['state'])
        for key, value in data['rejected'].items():
            self.rejected[key] = self.rejected.get(key, 0)+int(value)
        values = {k: v.detach() for k, v in data.items() if isinstance(v, torch.Tensor)}
        values.setdefault('intervention_steps', torch.ones(count, dtype=torch.long, device=data['state'].device))
        values.setdefault('intervention_kind', torch.zeros(count, dtype=torch.long, device=data['state'].device))
        values.setdefault('trajectory_valid', data['reliable'].detach().clone())
        for key in TEMPORAL & values.keys():
            if values[key].shape[1] != length:
                raise ValueError('同一真实响应的时间维度不一致，禁止补齐标签')
        values['_sequence'] = torch.arange(self.total_pairs, self.total_pairs+count, device=data['state'].device)
        self.total_pairs += count
        bucket = self.buckets.get(length)
        if bucket is not None and set(bucket) != set(values):
            raise ValueError('同长度响应字段不一致')
        self.buckets[length] = ({k: torch.cat((bucket[k], v)) for k, v in values.items()}
                                if bucket is not None else values)
        sequences = torch.cat([b['_sequence'] for b in self.buckets.values()])
        if len(sequences) > self.capacity:
            lower = sequences.sort().values[-self.capacity]
            for h, b in list(self.buckets.items()):
                keep = b['_sequence'] >= lower
                if bool(keep.any()):
                    self.buckets[h] = {k: v[keep] for k, v in b.items()}
                else:
                    del self.buckets[h]

    def sample(self, count, *, horizon=8):
        # 指定H32就采可靠到H32的真实对；不能用仅首步可靠的标签冒充长响应监督。
        eligible = [(h, b, (b['reliable'][:, :horizon] & b['trajectory_valid'][:, :horizon]).all(-1).nonzero().flatten())
                    for h, b in self.buckets.items() if h >= horizon]
        eligible = [(h, b, ids) for h, b, ids in eligible if len(ids)]
        if not eligible:
            return None
        sizes = [len(ids) for _, _, ids in eligible]
        device = eligible[0][1]['state'].device
        choices = torch.randint(sum(sizes), (count,), device=device)
        result, start = {}, 0
        for (_, b, ids), size in zip(eligible, sizes):
            selected = choices[(choices >= start) & (choices < start+size)]-start
            start += size
            if not len(selected):
                continue
            for key, value in b.items():
                if key == '_sequence':
                    continue
                selected_value = value[ids[selected]]
                if key in TEMPORAL:
                    selected_value = selected_value[:, :horizon]
                result.setdefault(key, []).append(selected_value)
        return {k: torch.cat(v) for k, v in result.items()}

    def coverage(self):
        songs = [b['song'] for b in self.buckets.values() if 'song' in b]
        return dict(songs=int(torch.cat(songs).unique().numel()) if songs else 0,
            windows={str(h): dict(pairs=len(b['state']), reliable_endpoint=int(b['reliable'][:, -1].sum()),
                valid_steps=int(b['reliable'].sum()),
                reliable_at={str(t): int((b['reliable'][:, :t] & b['trajectory_valid'][:, :t]).all(-1).sum()) for t in (8, 16, 32, 64) if t <= h},
                legacy_contact_filtered_at={str(t): int(b.get('legacy_reliable', b['reliable'])[:, :t].all(-1).sum()) for t in (8, 16, 32, 64) if t <= h},
                intervention_steps={str(int(t)): int((b['intervention_steps'] == t).sum())
                                    for t in b['intervention_steps'].unique()}) for h, b in self.buckets.items()})

    def state_dict(self):
        return dict(format='true_length_buckets_v2', buckets=self.buckets, total_pairs=self.total_pairs,
                    rejected=self.rejected, capacity=self.capacity)

    def load_state_dict(self, state, device):
        if state.get('format') != 'true_length_buckets_v2':
            raise ValueError('旧8步截断Replay不能冒充完整长响应恢复；旧模型可仅作显式对照')
        self.capacity, self.total_pairs, self.rejected = state['capacity'], state['total_pairs'], state['rejected']
        self.buckets = {int(h): {k: v.to(device) for k, v in b.items()} for h, b in state['buckets'].items()}
        for b in self.buckets.values():
            # 旧数据只将已有可靠前缀认作连续，不为未知/终止后时段补出有效标签。
            b.setdefault('trajectory_valid', b['reliable'].clone())


def consistency_loss(dynamics, pairs, horizon=8, *, kin, scales=None, weights=None, checkpoint_steps=False, action_norm_scale=.01):
    from torch.utils.checkpoint import checkpoint
    scales, weights = dict(GROUP_SCALES if scales is None else scales), dict(GROUP_WEIGHTS if weights is None else weights)
    if set(scales) != set(GROUP_SCALES) or set(weights) != set(GROUP_WEIGHTS) or min(scales.values()) <= 0:
        raise ValueError('各响应字段必须使用完整、独立且为正的归一化尺度')
    if action_norm_scale <= 0:
        raise ValueError('稳健动作归一化尺度必须为正')
    if pairs['plus_action'].shape[1] < horizon:
        raise ValueError('缺少真实长响应，禁止补齐或循环使用8步标签')
    origin = pairs['state']; plus, minus = origin, origin
    errors = {k: [] for k in weights}
    for t in range(horizon):
        if checkpoint_steps:
            plus = checkpoint(dynamics.step, plus, pairs['plus_action'][:, t], pairs['xi'], use_reentrant=False)
            minus = checkpoint(dynamics.step, minus, pairs['minus_action'][:, t], pairs['xi'], use_reentrant=False)
        else:
            plus = dynamics.step(plus, pairs['plus_action'][:, t], pairs['xi'])
            minus = dynamics.step(minus, pairs['minus_action'][:, t], pairs['xi'])
        with torch.no_grad():
            actual = response_groups(origin, pairs['plus_next'][:, t], pairs['minus_next'][:, t], kin)
        predicted = response_groups(origin, plus, minus, kin)
        action_size = (pairs['plus_action'][:, :t+1]-pairs['minus_action'][:, :t+1]).square().mean((1, 2)).sqrt()
        for key in errors:
            # 平滑有界的动作归一化：默认上限100，替代旧下限1e-4对应的一万倍。
            # 仍保留响应量纲；不能再乘action_norm_scale把该监督整体缩小一万倍。
            gain = (action_size.square()+action_norm_scale**2).rsqrt()
            error = (predicted[key]-actual[key])/scales[key]*gain[:, None]
            errors[key].append(torch.nn.functional.huber_loss(error, torch.zeros_like(error), reduction='none').mean(-1))
    mask = pairs['reliable'][:, :horizon]
    losses = {k: (torch.stack(v, 1)*mask).sum()/mask.sum().clamp_min(1) for k, v in errors.items()}
    loss = sum(weights[k]*v for k, v in losses.items())
    return loss, dict(reference_response_valid_steps=int(mask.sum()), reference_response_pairs=len(origin),
        reference_response_horizon=horizon, reference_response_reliable_endpoint=int(mask[:, -1].sum()),
        response_group_losses={k: float(v.detach()) for k, v in losses.items()},
        response_group_scales=scales, response_group_weights=weights,
        response_semantics='finite_amplitude_not_smooth_tangent', action_normalization_scale=action_norm_scale,
        action_normalization_gain_max=1/action_norm_scale)


@torch.no_grad()
def validate_response(dynamics, pairs, horizons=(1, 4, 8, 16, 32, 64), *, kin, scales=None):
    from .audit import _state_error
    from .update_diagnostics import parameter_identity
    scales = GROUP_SCALES if scales is None else scales
    available = pairs['plus_action'].shape[1]
    horizons = tuple(h for h in horizons if h <= available)
    origin = pairs['state']; plus, minus = origin, origin
    result = dict(wm_version=dynamics.model_version, wm_sha256=parameter_identity(dynamics),
                  actual_window=available, horizons={}, scales=dict(scales))
    for t in range(max(horizons)):
        plus = dynamics.step(plus, pairs['plus_action'][:, t], pairs['xi'])
        minus = dynamics.step(minus, pairs['minus_action'][:, t], pairs['xi'])
        if t+1 not in horizons:
            continue
        actual = response_groups(origin, pairs['plus_next'][:, t], pairs['minus_next'][:, t], kin)
        predicted = response_groups(origin, plus, minus, kin)
        valid = pairs['reliable'][:, :t+1].all(-1)
        continuous = pairs.get('trajectory_valid', pairs['reliable'])[:, :t+1].all(-1)
        fields = {}
        for key in actual:
            target, model = actual[key]/scales[key], predicted[key]/scales[key]
            cosine = (target*model).sum(-1)/(target.norm(dim=-1)*model.norm(dim=-1)).clamp_min(1e-12)
            relative = (target-model).norm(dim=-1)/target.norm(dim=-1).clamp_min(1e-6)
            fields[key] = dict(cosine=cosine.cpu().tolist(), relative_error=relative.cpu().tolist(),
                physical_response_rmse=float((actual[key][continuous]-predicted[key][continuous]).square().mean().sqrt())
                    if bool(continuous.any()) else None,
                high_snr_reverse_count=int(((cosine < 0) & valid & (target.norm(dim=-1) > 1e-4)).sum()))
        result['horizons'][str(t+1)] = dict(reliable_pairs=int(valid.sum()), total_pairs=len(valid),
            fields=fields, reliable=valid.cpu().tolist(), continuous_real_pairs=int(continuous.sum()),
            plus_state_error=_state_error(plus[continuous], pairs['plus_next'][continuous, t]) if bool(continuous.any()) else None,
            minus_state_error=_state_error(minus[continuous], pairs['minus_next'][continuous, t]) if bool(continuous.any()) else None,
            contact_switch=pairs['contact_switch'][:, :t+1].any(-1).cpu().tolist())
    return result


TRACKING_NAMES = ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori')
REWARD_NAMES = (*TRACKING_NAMES, 'full_objective')


def reward_vector(raw, total):
    """原六项乘原权重及一次dt；完整目标已经含dt，直接使用。"""
    return torch.stack([.02*(.5 if k.startswith('root_') else 1.)*raw[k] for k in TRACKING_NAMES]+[total], -1)


class ResponseRecorder:
    """保存同起点四条件的真实有限幅度响应，接触变化不再否决样本。

    同时记录旧接触筛选的覆盖作前后对照。保留真实episode/frame、终止、初态匹配、
    重复性、数值和动作可辨识检查；context只保存起点动态状态，不复制名义歌曲缓存。
    """
    def __init__(self, start, groups, mismatch):
        self.groups, self.count, self.mismatch = groups, groups*4, mismatch
        self.start = {k: v[:self.count].detach().clone() for k, v in start.items()}
        self.live = torch.ones(self.count, device=start['state'].device, dtype=torch.bool)
        self.rows, self.previous_contact = [], None

    def append(self, row):
        g, n, t = self.groups, self.count, len(self.rows)
        state = row['end']['state'][:n].reshape(g, 4, 55)
        action = row['proposal']['action'][:n].reshape(g, 4, 21)
        contact = row['contacts'][0].reshape(g, 4, -1)
        changed = (contact[:, 2] != contact[:, 3]).any(-1)
        if self.previous_contact is not None:
            changed |= (contact != self.previous_contact).any((1, 2))
        self.previous_contact = contact.clone()
        episode = row['start']['episode'][:n]
        frame = row['start']['frame'][:n]
        continuity = (episode == self.start['episode']) & (frame == self.start['frame']+t) & self.live
        continuous = continuity.reshape(g, 4).all(-1)
        self.live &= ~row['done'][:n]
        reward = reward_vector({k: row['raw'][k][:n] for k in TRACKING_NAMES}, row['reward'][:n]).reshape(g, 4, -1)
        finite = torch.cat((state.flatten(1), action.flatten(1), reward.flatten(1)), -1).isfinite().all(-1)
        repeat = (state[:, 0]-state[:, 1]).abs().amax(-1)
        size = (action[:, 2]-action[:, 3]).square().mean(-1).sqrt()
        good = ((self.mismatch <= 1e-5) & (repeat <= 1e-4) & finite & (size >= 1e-4)
                & continuous & self.live.reshape(g, 4).all(-1))
        if self.rows:
            good &= self.rows[-1]['reliable']
        legacy = good & ~changed
        if self.rows:
            legacy &= self.rows[-1]['legacy_reliable']
        self.rows.append(dict(all_next=state.detach().clone(), all_action=action.detach().clone(),
            all_residual=row['proposal']['residual'][:n].reshape(g, 4, 10, 27).detach().clone(),
            all_zero_action=row['proposal']['zero_action'][:n].reshape(g, 4, 21).detach().clone(),
            all_rewards=reward.detach().clone(), all_frame=frame.reshape(g, 4).clone(),
            all_episode=episode.reshape(g, 4).clone(), all_done=row['done'][:n].reshape(g, 4).clone(),
            reliable=good, legacy_reliable=legacy, contact_switch=changed, trajectory_valid=continuous))

    def finish(self, source, policy_version, duration, kind, split):
        values = {k: torch.stack([r[k] for r in self.rows], 1) for k in self.rows[0]}
        return dict(origin='current_policy_real_reference_response', split=split,
            state=self.start['state'][::4], xi=self.start['xi'][::4], song=source['song'], frame=source['frame'],
            plus_action=values['all_action'][:, :, 2], minus_action=values['all_action'][:, :, 3],
            plus_next=values['all_next'][:, :, 2], minus_next=values['all_next'][:, :, 3],
            initial_mismatch=self.mismatch, repeated_states=values['all_next'][:, :, :2],
            policy_version=torch.full_like(source['song'], policy_version), intervention_steps=duration,
            intervention_kind=torch.full_like(source['song'], kind),
            **{'context_'+k: v.reshape(self.groups, 4, *v.shape[1:]) for k, v in self.start.items()}, **values,
            rejected=dict(unreliable_steps=int((~values['reliable']).sum()),
                contact_switch_steps=int(values['contact_switch'].sum()), initial_mismatch=int((self.mismatch > 1e-5).sum())),
            response_semantics='finite_amplitude_contact_changes_allowed_not_smooth_derivative')


def score_response_sequences(dynamics, control, pairs, horizon, *, use_real_labels=False, checkpoint_steps=False):
    """按记录的参考/动作/上一动作重算原奖励；不重新运行Actor或修改投影。"""
    from torch.utils.checkpoint import checkpoint
    if 'all_rewards' not in pairs or pairs['all_action'].shape[1] < horizon:
        raise ValueError('奖励响应缺少完整真实四条件和参考身份，不能从旧正负数据伪造')
    groups = len(pairs['state'])
    z = {k.removeprefix('context_'): v.flatten(0, 1) for k, v in pairs.items() if k.startswith('context_')}
    def step(current, action, residual, zero, target):
        following = target if use_real_labels else dynamics.step(current['state'], action, current['xi'])
        now, future, goal, halo = control.reference(current, residual)
        proposal = dict(current=now, future=future, goal=goal, halo=halo, residual=residual,
                        raw_residual=residual, action=action, zero_action=zero)
        end, reward, raw, *_ = control.advance(current, proposal, following)
        return end, reward_vector(raw, reward)
    rows = []
    for t in range(horizon):
        args = (z, pairs['all_action'][:, t].flatten(0, 1), pairs['all_residual'][:, t].flatten(0, 1),
                pairs['all_zero_action'][:, t].flatten(0, 1), pairs['all_next'][:, t].flatten(0, 1))
        z, scores = checkpoint(step, *args, use_reentrant=False) if checkpoint_steps else step(*args)
        rows.append(scores.reshape(groups, 4, -1))
    return torch.stack(rows, 1)


def reward_response_loss(dynamics, control, pairs, horizon=32, *, scale=.001, checkpoint_steps=False):
    """只更新WM的稳健累计奖励差监督；低于真实重复噪声的标签不赋予确定方向。"""
    predicted = score_response_sequences(dynamics, control, pairs, horizon, checkpoint_steps=checkpoint_steps)
    actual = pairs['all_rewards'][:, :horizon]
    discount = .99**torch.arange(horizon, device=actual.device, dtype=actual.dtype)
    a, p = ((v*discount[None, :, None, None]).cumsum(1) for v in (actual, predicted))
    delta_a, delta_p = (v[:, :, 2:]-v[:, :, :1] for v in (a, p))
    noise = torch.maximum(10*(a[:, :, 1]-a[:, :, 0]).abs(), a.new_tensor(1e-6))
    valid = pairs['reliable'][:, :horizon] & pairs['trajectory_valid'][:, :horizon]
    mask = valid[:, :, None, None] & (delta_a.abs() > noise[:, :, None])
    costs = torch.nn.functional.huber_loss((delta_p-delta_a)/scale, torch.zeros_like(delta_p), reduction='none')
    loss = (costs*mask).sum()/mask.sum().clamp_min(1)
    fields = {}
    for i, name in enumerate(REWARD_NAMES):
        m = mask[..., i]
        fields[name] = dict(valid_deltas=int(m.sum()),
            delta_mse=float((((delta_p[..., i]-delta_a[..., i]).square()*m).sum()/m.sum().clamp_min(1)).detach()),
            per_step_prediction_rmse=float((predicted[..., i]-actual[..., i])[valid].detach().square().mean().sqrt()) if bool(valid.any()) else None,
            opposite=int(((delta_p[..., i]*delta_a[..., i] < 0) & m & (delta_p[..., i].abs()>1e-8)).sum()))
    endpoints = {}
    for h in (8, 16, 32, 64):
        if h <= horizon:
            endpoints[str(h)] = dict(actual_delta=delta_a[:, h-1].detach().cpu().tolist(),
                predicted_delta=delta_p[:, h-1].detach().cpu().tolist(), identifiable=mask[:, h-1].cpu().tolist())
    return loss, dict(reward_response_loss=float(loss.detach()), reward_response_scale=scale,
        reward_response_fields=fields, reward_response_endpoints=endpoints,
        reward_semantics='original_six_and_exact_full_objective_dt_once_actual_reference_action_previous_action')
