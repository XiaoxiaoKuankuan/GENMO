"""十帧绝对残差到SM完整参考的可微重建。

名义30Hz缓存始终只读；左支持和当前姿态继承已提交物理姿态，未来十点允许重规划。
位置在固定episode航向基叠加，姿态右乘SO(3)，关节按RobotSpec名称映射。第11点仅用
d11=2*d10-d9作用于真实存在的名义点；原SM中心差分/FK/Goal公式统一重建所有速度。
"""

from dataclasses import dataclass
from collections import OrderedDict

import torch

from supermimic.bridge.reference import build_reference, current_future
from supermimic.rotations import quat_to_mat, mat_to_quat, so3_exp, so3_log
from gem.closedloop.mb_control.contracts import ReferenceExhausted
from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
from .contracts import Pose, ResidualPlan, CONTROL_TICKS


_NOMINAL_CACHE = OrderedDict()


def make_nominal(world_qpos, joint_names, spec, *, start_tick=0, version=0, contact=None):
    names = tuple(joint_names)
    if len(set(names)) != 21 or set(names) != set(spec.joint_names):
        raise ValueError("名义参考关节名称必须与RobotSpec一一对应")
    ticks = start_tick + torch.arange(world_qpos.shape[1], device=world_qpos.device) * 20
    contact = world_qpos.new_zeros((*world_qpos.shape[:2], 2)) if contact is None else contact
    return ReferenceBuffer(ticks, world_qpos.detach().clone(), contact.detach().clone(),
                           torch.full_like(ticks, version), None,
                           tuple(names.index(n) for n in spec.joint_names), version)


def nominal_window(nominal, tick, kin):
    if tick < nominal.lower or tick + 132 > nominal.upper:
        raise ReferenceExhausted("当前+十未来点+真实第11halo支持不足")
    if nominal.world_qpos.requires_grad:
        ticks = tick + torch.arange(11, device=nominal.ticks.device) * CONTROL_TICKS
        return current_future(build_reference(nominal.pose_at, ticks, nominal.lower, nominal.upper, kin))
    # 名义轨迹冻结且完整已知：复用原ReferenceBuffer向量化FK缓存。
    # Replay中只读tensor的detach视图共享存储；持有源tensor防止地址复用，版本防原地变更。
    # 修正参考及执行图不进入此缓存，Adapter/Actor/WM的输入梯度不受影响。
    source = nominal.world_qpos
    key = (source.untyped_storage().data_ptr(), source.storage_offset(), tuple(source.shape), source.device,
           source.dtype, source._version, nominal.lower, nominal.upper, nominal.to_sm, id(kin))
    if key not in _NOMINAL_CACHE:
        from dataclasses import replace
        _NOMINAL_CACHE[key] = replace(nominal, kin=kin, prepared_reference=None, prepared_source=None)
        if len(_NOMINAL_CACHE) > 32:
            _NOMINAL_CACHE.popitem(last=False)
    _NOMINAL_CACHE.move_to_end(key)
    return _NOMINAL_CACHE[key].read_current_future(tick)


def nominal_pose(nominal, ticks):
    p, quat, q = nominal.pose_at(ticks)
    return Pose(p, quat_to_mat(quat), q)


def pose_residual(pose, nominal, heading):
    return torch.cat(((heading.transpose(-1, -2) @ (pose.p - nominal.p).unsqueeze(-1)).squeeze(-1),
                      so3_log(nominal.R.transpose(-1, -2) @ pose.R), pose.q - nominal.q), -1)


@dataclass(frozen=True)
class CorrectedReference:
    current: object
    future: object
    goal: torch.Tensor
    plan: ResidualPlan
    halo_residual: torch.Tensor


def reconstruct(z, residual, kin):
    if residual.shape != (len(z.executed_state), 10, 27):
        raise ValueError("Adapter必须恰好输出[B,10,27]，halo不占网络输出")
    nominal = z.nominal_reference
    if z.tick + 132 > nominal.upper:
        raise ReferenceExhausted("不得外推缺失名义第11点")
    ticks = (z.tick + torch.arange(-1, 12, device=residual.device) * CONTROL_TICKS).clamp_min(nominal.lower)
    base = nominal_pose(nominal, ticks)
    halo = 2 * residual[:, -1:] - residual[:, -2:-1]
    d = torch.cat((residual.new_zeros(len(residual), 2, 27), residual, halo), 1)
    p = base.p + (z.heading[:, None] @ d[..., :3, None]).squeeze(-1)
    R = base.R @ so3_exp(d[..., 3:6])
    q = base.q + d[..., 6:]
    # 已提交的当前/左姿态属于上一次计划，名义版本更换也不得回写。
    p = torch.cat((z.left_pose.p[:, None], z.current_pose.p[:, None], p[:, 2:]), 1)
    R = torch.cat((z.left_pose.R[:, None], z.current_pose.R[:, None], R[:, 2:]), 1)
    q = torch.cat((z.left_pose.q[:, None], z.current_pose.q[:, None], q[:, 2:]), 1)
    quat = mat_to_quat(R)

    def pose_at(query):
        indices = ((query - z.tick) // CONTROL_TICKS + 1).long()
        return p[:, indices], quat[:, indices], q[:, indices]

    current, future, goal = current_future(build_reference(pose_at, ticks[1:12], nominal.lower, nominal.upper, kin))
    if goal.shape[-1] != 570:
        raise ValueError("halo被错误计入Goal或参考维度错误")
    plan = ResidualPlan(ticks[2:12], residual, Pose(p[:, 2:12], R[:, 2:12], q[:, 2:12]),
                        torch.ones(residual.shape[:2], dtype=torch.bool, device=residual.device), nominal.version)
    return CorrectedReference(current, future, goal, plan, halo[:, 0])


def overlap_cost(new, old, bounds, heading):
    if old is None:
        return new.residual.new_zeros(len(new.residual))
    matched = new.ticks[:, None] == old.ticks[None]
    pairs = matched.nonzero()
    if not len(pairs):
        return new.residual.new_zeros(len(new.residual))
    i, j = pairs.unbind(-1)
    mask = new.valid[:, i] & old.valid[:, j]
    if new.nominal_version == old.nominal_version:
        error = new.residual[:, i] - old.residual[:, j]
    else:
        a, b = new.corrected_pose.select((slice(None), i)), old.corrected_pose.select((slice(None), j))
        error = pose_residual(a, b, heading[:, None])
    error = (error / bounds).square().mean(-1)
    return (error * mask).sum(-1) / mask.sum(-1).clamp_min(1)
