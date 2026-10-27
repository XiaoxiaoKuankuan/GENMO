"""Stage2-A原始GENMO精度目标及固定训练标尺，不改动SuperMimic六项奖励。

真实或想象执行状态与只读名义参考比较：身体姿态SO(3)、关节位置/速度及逐步局部
根位移；身体角速度只保留误差诊断，不再增加重复的Huber奖励。身体姿态使用原始
世界姿态，不沿实际机器人重新锚定yaw。标尺只从训练歌曲零Adapter误差拟合一次；
删除重复项后其余比例保持原值，不自动放大其它损失。Huber避免指数
饱和。动作偏离与参考导数成本独立相加，不再添加原SM已有的action-rate惩罚。
世界根位置误差保留作监控，根XY通道关闭时默认不以它驱动优化。所有新项只乘一次dt。
"""

import torch
from supermimic.rotations import so3_log, so3_exp

ERROR_NAMES = ('body_orientation', 'joint_position', 'joint_velocity', 'body_angular_velocity', 'root_local_displacement')
NAMES = ('body_orientation', 'joint_position', 'joint_velocity', 'root_local_displacement')
RATIOS = (.30, .25, .20, .10)
FLOORS = (.02, .01, .1, .001)


def errors(control, z, state, nominal, sm):
    body = so3_log(sm['tracked_body_R']@nominal.body_R.transpose(-1, -2)).norm(dim=-1)
    q = state[:, 13:34]-nominal.q
    qd = state[:, 34:]-nominal.qd
    bw = (sm['tracked_body_w']-nominal.body_w).norm(dim=-1)
    previous = control.bank.ref(z['song'], z['frame']).root_p
    dp = (state[:, :3]-z['state'][:, :3])-(nominal.root_p-previous)
    local = (z['heading'].transpose(-1, -2)@dp[..., None]).squeeze(-1)
    return dict(zip(ERROR_NAMES, (body, q, qd, bw, local)))


def calibrate(error_rows):
    """只能传入已声明的训练集真实零Adapter分布；每项固定绝对误差P75及物理下限。"""
    return {name: max(float(torch.quantile(error_rows[name].detach().abs().float().flatten(), .75)), floor)
            for name, floor in zip(NAMES, FLOORS)}


class PrecisionObjective:
    def __init__(self, scales=None, *, weight=1., action_weight=.01, velocity_weight=.002, acceleration_weight=.0002):
        self.scales = dict(zip(NAMES, FLOORS)) if scales is None else {name: scales[name] for name in NAMES}
        self.weight = weight
        self.action_weight, self.velocity_weight, self.acceleration_weight = action_weight, velocity_weight, acceleration_weight
        if set(self.scales) != set(NAMES) or any(v <= 0 for v in self.scales.values()):
            raise ValueError('精度标尺必须是训练集固定正数且完整包含五项')

    def __call__(self, control, z, proposal, state, nominal, sm):
        values = errors(control, z, state, nominal, sm)
        raw = {'absolute_body_angular_velocity': values['body_angular_velocity'].abs().mean(-1)}
        weighted = {}
        for name, ratio in zip(NAMES, RATIOS):
            error = values[name]/self.scales[name]
            cost = torch.nn.functional.huber_loss(error, torch.zeros_like(error), reduction='none').mean(-1)
            raw['precision_'+name] = cost
            raw['absolute_'+name] = values[name].abs().mean(-1)
            weighted['precision_'+name] = -.02*self.weight*ratio*cost
        action = (proposal['action']-proposal['zero_action']).square().mean(-1)
        d = torch.cat((z['left'][:, None], z['anchor'][:, None], proposal['residual'], proposal['halo'][:, None]), 1)
        v = d.diff(dim=1)/.02
        frames = (z['frame'][:, None]+control.bank.offset13).clamp_min(0)
        rotation = control.bank.pose(z['song'][:, None], frames)[1]
        corrected = rotation@so3_exp(d[..., 3:6])
        angular = so3_log(corrected[:, 1:]@corrected[:, :-1].transpose(-1, -2))/.02
        nominal_angular = so3_log(rotation[:, 1:]@rotation[:, :-1].transpose(-1, -2))/.02
        v = torch.cat((v[..., :3], angular-nominal_angular, v[..., 6:]), -1)
        a = v.diff(dim=1)/.02
        budget = control.projection
        velocity = (v/budget.velocity).square().mean((1, 2))
        acceleration = (a/budget.acceleration).square().mean((1, 2))
        raw.update(action_deviation=action, added_reference_velocity=velocity, added_reference_acceleration=acceleration)
        weighted.update(action_deviation=-.02*self.action_weight*action,
                        added_reference_velocity=-.02*self.velocity_weight*velocity,
                        added_reference_acceleration=-.02*self.acceleration_weight*acceleration)
        return raw, weighted


def calibrate_gradient_weights(control, dynamics, starts, *, microbatch=32, horizon=32):
    """用训练歌曲真实零策略起点预先固定两档权重，不以评价结果调参或更新任何网络。"""
    from .vector_control import select_state, detach_state
    from .vector_learning import GroupNormalizedCritic, rollout
    from .dynamics import frozen_parameters
    from .update_diagnostics import parameter_identity
    if not bool(torch.isin(starts['song'], control.bank.by_split['train']).all()):
        raise ValueError('精度梯度标定只能使用训练歌曲')
    parameters = tuple(control.adapter.parameters())
    critic = GroupNormalizedCritic(2835, control.spec).to(starts['state'].device)
    names = ('sm', 'precision')
    gradients = {k: [torch.zeros_like(p) for p in parameters] for k in names}
    values = dict.fromkeys(names, 0.)
    total = len(starts['state'])
    with dynamics.adapter_phase(), frozen_parameters(critic):
        for offset in range(0, total, microbatch):
            z = detach_state(select_state(starts, slice(offset, offset+microbatch)))
            result = rollout(control, dynamics, critic, z, horizon, terminal_value=False, checkpoint_steps=True)
            terms = dict(sm=result['weighted']['sm'], precision=sum(v for k, v in result['weighted'].items() if k.startswith('precision_')))
            for index, name in enumerate(names):
                term = terms[name].sum()/total
                grad = torch.autograd.grad(term, parameters, retain_graph=index == 0, allow_unused=True)
                for accumulator, value in zip(gradients[name], grad):
                    if value is not None:
                        accumulator.add_(value.detach())
                values[name] += float(term.detach())
    norms = {k: float(sum(v.square().sum() for v in g).sqrt()) for k, g in gradients.items()}
    if min(norms.values()) <= 1e-9:
        raise ValueError('训练集执行梯度不足，无法标定精度权重')
    base = control.precision.weight*norms['sm']/norms['precision']
    return dict(samples=total, songs=int(starts['song'].unique().numel()), horizon=horizon, gradient_norms=norms,
        values=values, low_weight=.5*base, high_weight=base, target_gradient_ratios=[.5, 1.],
        wm_sha256=parameter_identity(dynamics), wm_version=dynamics.model_version,
        adapter_sha256=parameter_identity(control.adapter), scales=control.precision.scales,
        accepted_updates=0, selection='predeclared_train_gradient_norm_ratio_not_evaluation_search',
        boundary='gradient_scale_calibration_does_not_certify_WM_direction_on_all_songs')
