"""当前Adapter经投影和冻结Actor可实现方向的真实参考响应监督。

Replay保存同一真实起点的正负参考产生的动作序列、执行状态、物理条件和可靠mask，
不保存完整歌曲，不用想象状态冒充标签。可靠mask排除初态不匹配、重复不稳定、
接触切换、终止以及动作差过小的方向。WM同时预测两条真实动作驱动轨迹，以原起点
局部坐标比较根位移、SO(3)姿态、关节位置/速度响应；这只更新独立WM，不更新Actor。
H16/H32方向报告与单步导数、闭环参数更新验收分开，任何旧模型报告不能自动放行。
"""

import torch
from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_log


def local_difference(origin, plus, minus):
    p0, R0, *_ = unflatten_state(origin)
    p1, R1, _, _, q1, qd1 = unflatten_state(plus)
    p2, R2, _, _, q2, qd2 = unflatten_state(minus)
    dp = (R0.transpose(-1, -2)@(p1-p2)[..., None]).squeeze(-1)
    dr = so3_log(R0.transpose(-1, -2)@R1)-so3_log(R0.transpose(-1, -2)@R2)
    return torch.cat((dp, dr, q1-q2, qd1-qd2), -1)


def response_scale(state):
    return state.new_tensor([.05]*3+[.1]*3+[.1]*21+[2.]*21)


class ReferenceResponseReplay:
    def __init__(self, capacity=8192):
        self.capacity, self.data = capacity, {}
        self.total_pairs = 0
        self.rejected = {}

    def add(self, data):
        if data.get('origin') != 'current_policy_real_reference_response' or data.get('split') != 'train':
            raise ValueError('参考响应监督只允许当前策略训练歌曲真实配对')
        for key, value in data['rejected'].items():
            self.rejected[key] = self.rejected.get(key, 0)+int(value)
        self.total_pairs += len(data['state'])
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
                self.data[key] = torch.cat((self.data[key], value))[-self.capacity:] if key in self.data else value[-self.capacity:]

    def sample(self, count, *, horizon=8):
        if not self.data:
            return None
        eligible = self.data['reliable'][:, :horizon].any(-1).nonzero().flatten()
        if not len(eligible):
            return None
        ids = eligible[torch.randint(len(eligible), (count,), device=eligible.device)]
        return {k: v[ids] for k, v in self.data.items()}

    def state_dict(self):
        return dict(data=self.data, total_pairs=self.total_pairs, rejected=self.rejected, capacity=self.capacity)

    def load_state_dict(self, state, device):
        self.capacity, self.total_pairs, self.rejected = state['capacity'], state['total_pairs'], state['rejected']
        self.data = {k: v.to(device) for k, v in state['data'].items()}


def consistency_loss(dynamics, pairs, horizon=8):
    origin = pairs['state']
    plus, minus = origin, origin
    errors = []
    scale = response_scale(origin)
    for t in range(horizon):
        plus = dynamics.step(plus, pairs['plus_action'][:, t], pairs['xi'])
        minus = dynamics.step(minus, pairs['minus_action'][:, t], pairs['xi'])
        actual = local_difference(origin, pairs['plus_next'][:, t], pairs['minus_next'][:, t])/scale
        predicted = local_difference(origin, plus, minus)/scale
        # 以真实动作差归一化；无动作响应的方向由采集器拒绝，不能用极小分母制造伪梯度。
        action_size = (pairs['plus_action'][:, :t+1]-pairs['minus_action'][:, :t+1]).square().mean((1, 2)).sqrt()
        error = ((predicted-actual)/action_size.clamp_min(1e-4)[:, None]).square().mean(-1)
        errors.append(error)
    mask = pairs['reliable'][:, :horizon]
    loss = (torch.stack(errors, 1)*mask).sum()/mask.sum().clamp_min(1)
    return loss, dict(reference_response_valid_steps=int(mask.sum()), reference_response_pairs=len(origin))


@torch.no_grad()
def validate_response(dynamics, pairs, horizons=(1, 4, 8, 16, 32)):
    from .audit import _state_error
    from .update_diagnostics import parameter_identity
    origin = pairs['state']
    plus, minus = origin, origin
    result = dict(wm_version=dynamics.model_version, wm_sha256=parameter_identity(dynamics), horizons={})
    scale = response_scale(origin)
    for t in range(max(horizons)):
        plus = dynamics.step(plus, pairs['plus_action'][:, t], pairs['xi'])
        minus = dynamics.step(minus, pairs['minus_action'][:, t], pairs['xi'])
        if t+1 not in horizons:
            continue
        actual = local_difference(origin, pairs['plus_next'][:, t], pairs['minus_next'][:, t])/scale
        predicted = local_difference(origin, plus, minus)/scale
        dot = (actual*predicted).sum(-1)
        norm = actual.norm(dim=-1)*predicted.norm(dim=-1)
        cosine = dot/norm.clamp_min(1e-12)
        relative = (actual-predicted).norm(dim=-1)/actual.norm(dim=-1).clamp_min(1e-6)
        valid = pairs['reliable'][:, :t+1].all(-1)
        result['horizons'][str(t+1)] = dict(reliable_pairs=int(valid.sum()), total_pairs=len(valid),
            cosine=cosine.cpu().tolist(), relative_error=relative.cpu().tolist(), reliable=valid.cpu().tolist(),
            plus_state_error=_state_error(plus, pairs['plus_next'][:, t]),
            minus_state_error=_state_error(minus, pairs['minus_next'][:, t]),
            high_snr_reverse_count=int(((cosine < 0) & valid & (actual.norm(dim=-1) > 1e-4)).sum()),
            contact_switch=pairs['contact_switch'][:, :t+1].any(-1).cpu().tolist())
    return result
