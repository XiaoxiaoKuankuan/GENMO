"""Stage2-A原始GENMO精度目标及固定训练标尺，不改动SuperMimic六项奖励。

真实或想象执行状态与只读名义参考比较：身体姿态SO(3)、关节位置/速度及逐步局部
根位移；身体角速度只保留误差诊断，不再增加重复的Huber奖励。身体姿态使用原始
世界姿态，不沿实际机器人重新锚定yaw。标尺只从训练歌曲零Adapter误差拟合一次；
删除重复项后其余比例保持原值，不自动放大其它损失。Huber避免指数饱和。
执行精度之外可显式配置根姿态软成本：相对名义的角速度偏差，以及绝对时刻对齐的
前后计划姿态差。两项只进入奖励，不改变参考、不截断梯度、不作运行门禁；没有
角加速度成本或硬预算，也不惩罚偏离基线控制动作。默认关闭以保留旧评价定义。
世界根位置误差保留作监控，根XY通道关闭时默认不以它驱动优化。所有新项只乘一次dt。
"""

import torch
from supermimic.rotations import so3_exp, so3_log

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


def root_reference_costs(residual, anchor, previous_plan, valid, nominal_rotation,
                         velocity_scale=.3, continuity_scale=.05):
    """只读十帧参考的根姿态软成本；不修改输出，不求解QP/SQP。

    根角速度用当前锚点、十帧及真实名义halo的SO(3)前向差分，减去同区间名义
    角速度，故原舞蹈转身不受额外惩罚。连续性按新前九帧/旧后九帧相同绝对时刻
    的旋转测地差计算；首计划无旧窗口时为零。两种scale仅是损失单位，不是限制。
    所有计划/锚点保留梯度；仅使用已知名义参考，不访问未来实际执行状态。
    """
    rotation = residual[..., 3:6]
    halo = 2*rotation[:, -1:]-rotation[:, -2:-1]
    plan = torch.cat((anchor[:, None, 3:6], rotation, halo), 1)
    nominal = nominal_rotation[:, 1:]
    corrected = nominal@so3_exp(plan)
    w = so3_log(corrected[:, 1:]@corrected[:, :-1].transpose(-1, -2))/.02
    w0 = so3_log(nominal[:, 1:]@nominal[:, :-1].transpose(-1, -2))/.02
    mismatch = so3_log(so3_exp(previous_plan[:, 1:, 3:6]).transpose(-1, -2)
                      @so3_exp(rotation[:, :9]))
    huber = lambda v: torch.nn.functional.huber_loss(v, torch.zeros_like(v), reduction='none').mean((1, 2))
    return dict(root_reference_velocity=huber((w-w0)/velocity_scale),
                root_reference_continuity=huber(mismatch/continuity_scale)*valid.to(residual))


class PrecisionObjective:
    def __init__(self, scales=None, *, weight=1., root_velocity_weight=0.,
                 root_continuity_weight=0., root_velocity_scale=.3, root_continuity_scale=.05):
        self.scales = dict(zip(NAMES, FLOORS)) if scales is None else {name: scales[name] for name in NAMES}
        self.weight = weight
        self.root_velocity_weight, self.root_continuity_weight = root_velocity_weight, root_continuity_weight
        self.root_velocity_scale, self.root_continuity_scale = root_velocity_scale, root_continuity_scale
        if min(root_velocity_weight, root_continuity_weight) < 0 or min(root_velocity_scale, root_continuity_scale) <= 0:
            raise ValueError('根姿态软成本权重须非负、归一化尺度须为正；这些数值不是硬约束')
        if set(self.scales) != set(NAMES) or any(v <= 0 for v in self.scales.values()):
            raise ValueError('精度标尺必须是训练集固定正数且完整包含四项')

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
        if self.root_velocity_weight or self.root_continuity_weight:
            rotation = proposal.get('nominal_rotation')
            if rotation is None:  # 离线真实奖励重放没有缓存该字段，按相同歌曲/绝对帧恢复。
                frames = (z['frame'][:, None]+control.bank.offset13).clamp_min(0)
                rotation = control.bank.pose(z['song'][:, None], frames)[1]
            costs = root_reference_costs(proposal['residual'], z['anchor'], z['plan'], z['plan_valid'],
                                        rotation, self.root_velocity_scale, self.root_continuity_scale)
            raw.update(costs)
            weighted['root_reference_velocity'] = -.02*self.root_velocity_weight*costs['root_reference_velocity']
            weighted['root_reference_continuity'] = -.02*self.root_continuity_weight*costs['root_reference_continuity']
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
