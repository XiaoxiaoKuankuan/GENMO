"""当前Adapter经投影和冻结Actor产生的完整真实参考响应监督。

按实际H8/H16/H32/H64长度分桶保存正负响应，短窗口不补齐，长窗口不截成8步。
监督标签只有训练集真实状态、实际动作和物理条件；接触切换、重复性不足、终止及
动作变化过小的步骤均有可靠mask。位置/姿态/关节速度、根角速度及现有SM FK的
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
            'contact_switch', 'repeated_states', 'trajectory_valid'}


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
        eligible = [(h, b, b['reliable'][:, :horizon].all(-1).nonzero().flatten())
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
                reliable_at={str(t): int(b['reliable'][:, :t].all(-1).sum()) for t in (8, 16, 32, 64) if t <= h},
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


def consistency_loss(dynamics, pairs, horizon=8, *, kin, scales=None, weights=None, checkpoint_steps=False):
    from torch.utils.checkpoint import checkpoint
    scales, weights = dict(GROUP_SCALES if scales is None else scales), dict(GROUP_WEIGHTS if weights is None else weights)
    if set(scales) != set(GROUP_SCALES) or set(weights) != set(GROUP_WEIGHTS) or min(scales.values()) <= 0:
        raise ValueError('各响应字段必须使用完整、独立且为正的归一化尺度')
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
            error = (predicted[key]-actual[key])/scales[key]/action_size.clamp_min(1e-4)[:, None]
            errors[key].append(error.square().mean(-1))
    mask = pairs['reliable'][:, :horizon]
    losses = {k: (torch.stack(v, 1)*mask).sum()/mask.sum().clamp_min(1) for k, v in errors.items()}
    loss = sum(weights[k]*v for k, v in losses.items())
    return loss, dict(reference_response_valid_steps=int(mask.sum()), reference_response_pairs=len(origin),
        reference_response_horizon=horizon, reference_response_reliable_endpoint=int(mask[:, -1].sum()),
        response_group_losses={k: float(v.detach()) for k, v in losses.items()},
        response_group_scales=scales, response_group_weights=weights)


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
