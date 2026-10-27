"""Stage2-A原始GENMO精度目标及固定训练标尺，不改动SuperMimic六项奖励。

真实或想象执行状态与只读名义参考比较：身体姿态SO(3)、关节位置/速度、身体角速度、
逐控制步局部根位移。身体姿态使用原始世界姿态，不沿实际机器人重新锚定yaw。
标尺只从训练歌曲零Adapter真实误差拟合一次，五项内部比例固定；Huber避免指数
饱和。动作偏离与参考导数成本独立相加，不再添加原SM已有的action-rate惩罚。
世界根位置误差保留作监控，根XY通道关闭时默认不以它驱动优化。所有新项只乘一次dt。
"""

import torch
from supermimic.rotations import so3_log, so3_exp

NAMES = ('body_orientation', 'joint_position', 'joint_velocity', 'body_angular_velocity', 'root_local_displacement')
RATIOS = (.30, .25, .20, .15, .10)
FLOORS = (.02, .01, .1, .1, .001)


def errors(control, z, state, nominal, sm):
    body = so3_log(sm['tracked_body_R']@nominal.body_R.transpose(-1, -2)).norm(dim=-1)
    q = state[:, 13:34]-nominal.q
    qd = state[:, 34:]-nominal.qd
    bw = (sm['tracked_body_w']-nominal.body_w).norm(dim=-1)
    previous = control.bank.ref(z['song'], z['frame']).root_p
    dp = (state[:, :3]-z['state'][:, :3])-(nominal.root_p-previous)
    local = (z['heading'].transpose(-1, -2)@dp[..., None]).squeeze(-1)
    return dict(zip(NAMES, (body, q, qd, bw, local)))


def calibrate(error_rows):
    """只能传入已声明的训练集真实零Adapter分布；每项固定绝对误差P75及物理下限。"""
    return {name: max(float(torch.quantile(error_rows[name].detach().abs().float().flatten(), .75)), floor)
            for name, floor in zip(NAMES, FLOORS)}


class PrecisionObjective:
    def __init__(self, scales=None, *, weight=1., action_weight=.01, velocity_weight=.002, acceleration_weight=.0002):
        self.scales = dict(zip(NAMES, FLOORS)) if scales is None else dict(scales)
        self.weight = weight
        self.action_weight, self.velocity_weight, self.acceleration_weight = action_weight, velocity_weight, acceleration_weight
        if set(self.scales) != set(NAMES) or any(v <= 0 for v in self.scales.values()):
            raise ValueError('精度标尺必须是训练集固定正数且完整包含五项')

    def __call__(self, control, z, proposal, state, nominal, sm):
        values = errors(control, z, state, nominal, sm)
        raw, weighted = {}, {}
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
