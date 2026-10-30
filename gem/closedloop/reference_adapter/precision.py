"""Stage2-A原始GENMO精度目标及固定训练标尺，不改动SuperMimic六项奖励。

真实或想象执行状态与只读名义参考比较：身体姿态SO(3)、关节位置/速度及逐步局部
根位移；身体角速度只保留误差诊断，不再增加重复的Huber奖励。身体姿态使用原始
世界姿态，不沿实际机器人重新锚定yaw。标尺只从训练歌曲零Adapter误差拟合一次；
删除重复项后其余比例保持原值，不自动放大其它损失。Huber避免指数饱和。
可显式启用名义heading系绝对根XY Huber，以及相对名义新增的关节/根姿态速度、
加速度和重叠软成本。导数直接复用Actor Goal的中心差分，包含历史锚点和真实halo。
这些成本只改变奖励，不改变参考、不截断梯度、不引入硬预算。旧配置默认保持原值；
新权重由训练数据梯度标定。根位置真值只用于训练目标，不成为可部署观测。
所有新增奖励只乘一次dt；新六项连续性与旧两项根姿态成本互斥，避免重复惩罚。
执行jerk与名义支撑足端速度成本单独配置；根Tracking可加训练专用bonus，原六项
Tracking报告仍用共同权重，以免把改变计分权重造成的分数增长冒充物理改善。
"""

import torch
from supermimic.rotations import so3_exp, so3_log, yaw_mat

ERROR_NAMES = ('body_orientation', 'joint_position', 'joint_velocity', 'body_angular_velocity', 'root_local_displacement')
NAMES = ('body_orientation', 'joint_position', 'joint_velocity', 'root_local_displacement')
RATIOS = (.30, .25, .20, .10)
FLOORS = (.02, .01, .1, .001)
REFERENCE_NAMES = ('joint_velocity', 'joint_acceleration', 'root_velocity',
                   'root_acceleration', 'joint_overlap', 'root_overlap')
REFERENCE_FLOORS = dict(zip(REFERENCE_NAMES, (.1, 1., .05, .5, .005, .002)))


def goal_derivatives(rotation, joints, frames):
    """Actor Goal唯一的中心差分定义：输入真实名义支持t-1至t+11，输出t至t+10。

    起始帧由调用方钳到实际存在的第0帧，分母使用实际时间跨度，不外推缺失参考。
    根角速度为世界系SO(3)差分；软成本与Goal构造共享本函数，避免两套速度定义。
    """
    span = ((frames[:, 2:]-frames[:, :-2]).to(joints)*.02)[..., None]
    return ((joints[:, 2:]-joints[:, :-2])/span,
            so3_log(rotation[:, 2:]@rotation[:, :-2].transpose(-1, -2))/span)


def absolute_root_xy(state, nominal, scale=.2):
    """执行后t+1的真实/WM状态与同一时刻原GENMO参考比较，不读修正参考。

    只绕名义yaw转入heading坐标系；该训练特权误差不加入Adapter或Actor观测。
    Huber在归一化误差超过1后具有常数斜率，避免指数Tracking的大误差饱和。
    """
    world = state[:, :3]-nominal.root_p
    local = (yaw_mat(nominal.root_R).transpose(-1, -2)@world[..., None]).squeeze(-1)
    xy = local[:, :2]
    cost = torch.nn.functional.huber_loss(xy/scale, torch.zeros_like(xy), reduction='none').mean(-1)
    return cost, dict(root_xy_huber=cost, root_xy_squared_m=world[:, :2].square().sum(-1),
        root_xy_heading_x_m=xy[:, 0], root_xy_heading_y_m=xy[:, 1],
        root_xy_world_x_m=world[:, 0], root_xy_world_y_m=world[:, 1])


def reference_continuity_values(control, z, proposal):
    """直接读取实际Goal的qdot/w，惩罚相对名义轨迹新增的速度、加速度和计划跳变。

    t时刻速度依赖已提交t-1与新t+1，t+1速度依赖已提交t与新t+2；相邻Goal速度
    差因此覆盖前两帧接缝。末端t+10速度读取真实名义t+11及2*d10-d9的残差halo，
    t+9/t+10速度差覆盖halo连接。加速度是这些同一定义速度的差/0.02，不另造速度。
    新前九帧与旧后九帧对应相同绝对时刻；首次计划没有旧重叠，成本为0。
    这里只计算可微成本，不修改任何参考或控制动作，不进行QP/SQP求解。
    """
    frames = (z['frame'][:, None]+control.bank.offset13).clamp_min(0)
    rotation, q = proposal.get('nominal_rotation'), proposal.get('nominal_q')
    if rotation is None or q is None:
        _, rotation, q = control.bank.pose(z['song'][:, None], frames)
    qd0, w0 = goal_derivatives(rotation, q, frames)
    current, future = proposal['current'], proposal['future']
    qd = torch.cat((current.qd[:, None], future.qd), 1)
    w = torch.cat((current.root_w[:, None], future.root_w), 1)
    extra_qd, extra_w = qd-qd0, w-w0
    # 根旋转重叠使用同绝对时刻的群差；名义旋转相同，可在残差群中精确消去。
    valid = z['plan_valid'][:, None, None].to(q)
    r = proposal['residual']
    overlap_q = (r[:, :9, 6:]-z['plan'][:, 1:, 6:])*valid
    overlap_R = so3_log(so3_exp(z['plan'][:, 1:, 3:6]).transpose(-1, -2)
                          @so3_exp(r[:, :9, 3:6]))*valid
    return dict(joint_velocity=extra_qd, joint_acceleration=extra_qd.diff(dim=1)/.02,
                root_velocity=extra_w, root_acceleration=extra_w.diff(dim=1)/.02,
                joint_overlap=overlap_q, root_overlap=overlap_R)


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
                 root_continuity_weight=0., root_velocity_scale=.3, root_continuity_scale=.05,
                 root_xy_weight=0., root_xy_scale=.2, soft_reference_weights=None, soft_reference_scales=None,
                 root_tracking_bonus_weight=0., execution_jerk_weight=0., execution_jerk_scale=1000.,
                 stance_foot_weight=0., stance_foot_scale=.1, stance_height_band=.04,
                 stance_height_tau=.01, stance_speed_scale=.2):
        self.scales = dict(zip(NAMES, FLOORS)) if scales is None else {name: scales[name] for name in NAMES}
        self.weight = weight
        self.root_velocity_weight, self.root_continuity_weight = root_velocity_weight, root_continuity_weight
        self.root_velocity_scale, self.root_continuity_scale = root_velocity_scale, root_continuity_scale
        self.root_xy_weight, self.root_xy_scale = root_xy_weight, root_xy_scale
        self.soft_reference_weights = dict(soft_reference_weights or {})
        self.soft_reference_scales = dict(REFERENCE_FLOORS if soft_reference_scales is None else soft_reference_scales)
        self.root_tracking_bonus_weight = root_tracking_bonus_weight
        self.execution_jerk_weight, self.execution_jerk_scale = execution_jerk_weight, execution_jerk_scale
        self.stance_foot_weight, self.stance_foot_scale = stance_foot_weight, stance_foot_scale
        self.stance_height_band, self.stance_height_tau = stance_height_band, stance_height_tau
        self.stance_speed_scale = stance_speed_scale
        self._stance_bank = None
        if min(root_tracking_bonus_weight, execution_jerk_weight, stance_foot_weight) < 0 or min(
                execution_jerk_scale, stance_foot_scale, stance_height_band, stance_height_tau, stance_speed_scale) <= 0:
            raise ValueError('执行稳定性及根bonus权重须非负，固定物理标尺须为正')
        if root_xy_weight < 0 or root_xy_scale <= 0:
            raise ValueError('绝对根XY权重须非负，物理尺度须为正')
        if set(self.soft_reference_weights)-set(REFERENCE_NAMES) or any(v < 0 for v in self.soft_reference_weights.values()):
            raise ValueError('软参考权重只能包含六个明确的非负成本')
        if set(self.soft_reference_scales) != set(REFERENCE_NAMES) or any(v <= 0 for v in self.soft_reference_scales.values()):
            raise ValueError('软参考标尺须是训练集固定的六项正数')
        if any(self.soft_reference_weights.values()) and (root_velocity_weight or root_continuity_weight):
            raise ValueError('新六项连续性成本已包含根速度和重叠，不能重复叠加旧两项成本')
        if min(root_velocity_weight, root_continuity_weight) < 0 or min(root_velocity_scale, root_continuity_scale) <= 0:
            raise ValueError('根姿态软成本权重须非负、归一化尺度须为正；这些数值不是硬约束')
        if set(self.scales) != set(NAMES) or any(v <= 0 for v in self.scales.values()):
            raise ValueError('精度标尺必须是训练集固定正数且完整包含四项')

    def __call__(self, control, z, proposal, state, nominal, sm):
        values = errors(control, z, state, nominal, sm)
        raw = {'absolute_body_angular_velocity': values['body_angular_velocity'].abs().mean(-1)}
        weighted = {}
        if self.root_tracking_bonus_weight:
            raw['root_tracking_bonus'] = sm['root_pos']
            weighted['root_tracking_bonus'] = .02*self.root_tracking_bonus_weight*sm['root_pos']
        if self.execution_jerk_weight:
            from .execution_costs import execution_jerk
            cost, rms = execution_jerk(state, z['state'], z['execution_previous_qd'],
                                       z['execution_jerk_valid'], self.execution_jerk_scale)
            raw.update(execution_jerk_cost=cost, execution_jerk_rms=rms,
                       execution_jerk_valid=z['execution_jerk_valid'].to(state))
            weighted['execution_jerk'] = -.02*self.execution_jerk_weight*cost
        if self.stance_foot_weight:
            from .execution_costs import stance_foot_velocity
            self.prepare_stance(control)
            feet = self._stance_feet
            cost, diagnostics = stance_foot_velocity(sm['tracked_body_v'][:, feet], nominal.body_p[:, feet],
                nominal.body_v[:, feet], self._stance_floor[z['song']], scale=self.stance_foot_scale,
                height_band=self.stance_height_band, height_tau=self.stance_height_tau,
                speed_scale=self.stance_speed_scale)
            raw.update(diagnostics, stance_foot_cost=cost)
            weighted['stance_foot_velocity'] = -.02*self.stance_foot_weight*cost
        root_cost, root_raw = absolute_root_xy(state, nominal, self.root_xy_scale)
        raw.update(root_raw)
        if self.root_xy_weight:
            # 覆盖旧禁用的三维平方root_non_saturating项；新配置只有这一个根辅助目标。
            raw['root_non_saturating'] = root_cost
            weighted['root_non_saturating'] = -.02*self.root_xy_weight*root_cost
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
        if any(self.soft_reference_weights.values()):
            features = reference_continuity_values(control, z, proposal)
            for name, value in features.items():
                normalized = value/self.soft_reference_scales[name]
                cost = torch.nn.functional.huber_loss(normalized, torch.zeros_like(normalized), reduction='none').mean((1, 2))
                key = 'reference_soft_'+name
                raw[key] = cost
                raw[key+'_rms'] = value.square().mean((1, 2)).sqrt()
                weighted[key] = -.02*self.soft_reference_weights.get(name, 0.)*cost
            for name in ('joint_velocity', 'root_velocity', 'joint_acceleration', 'root_acceleration'):
                raw['reference_seam_'+name+'_rms'] = features[name][:, :1].square().mean((1, 2)).sqrt()
                raw['reference_halo_'+name+'_rms'] = features[name][:, -1:].square().mean((1, 2)).sqrt()
        return raw, weighted

    def prepare_stance(self, control):
        """每份只读名义库只准备一次相位基准；不读取任何未来执行状态或仿真接触。"""
        if self._stance_bank is control.bank:
            return
        names = tuple(control.bundle.spec.tracked_bodies)
        self._stance_feet = [names.index(n) for n in ('l_ankle_roll_link', 'r_ankle_roll_link')]
        with torch.no_grad():
            self._stance_floor = torch.stack([torch.quantile(
                control.bank.reference.body_p[offset:offset+length, self._stance_feet, 2], .02, dim=0)
                for offset, length in zip(control.bank.offsets.tolist(), control.bank.lengths.tolist())])
        self._stance_bank = control.bank


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
