"""音乐动作保持的物理空间稳健损失，避免直接对 qpos30 混合编码做距离。

执行损失按根部位置/姿态/速度、根部坐标系身体位置、身体姿态和速度、关节角/速度
分别以物理尺度归一化，使用 pseudo-Huber 保持大误差下的非饱和导数。
局部教师先与学生共同解码为世界 qpos28，再通过同一 SM FK 比较；只对可修改帧
计入位置和姿态损失，时间差分保留跨已知前缀边界的导数，已知坐标绝不成为优化变量。
固定音乐教师和同请求局部教师承担不同职责，不能用后者替代固定音乐时间线。
"""

import torch
from supermimic.mdp import unflatten_state
from supermimic.rotations import quat_to_mat, so3_log


DEFAULT_SCALES = {"root_position": .3, "root_orientation": .4, "root_velocity": 1.,
    "body_position": .15, "body_orientation": .3, "body_velocity": 1.,
    "joint_position": .35, "joint_velocity": 2., "joint_acceleration": 20.}


def robust(value, scale):
    return torch.sqrt(1+(value/scale).square())-1


def execution_loss(state, reference, kin, scales=None):
    scales = dict(DEFAULT_SCALES, **(scales or {}))
    p, R, v, w, q, qd = unflatten_state(state)
    ref = reference
    bp, bR, bv, _ = kin.forward(p, R, v, w, q, qd, all_bodies=True)
    tp, tR, tv, _ = kin.forward(ref.root_p, ref.root_R, ref.root_v, ref.root_w,
                                ref.q, ref.qd, all_bodies=True)
    local = (R.transpose(-1, -2)[:, None]@(bp-p[:, None])[..., None]).squeeze(-1)
    target_local = (ref.root_R.transpose(-1, -2)[:, None]@(tp-ref.root_p[:, None])[..., None]).squeeze(-1)
    errors = {"root_position": p-ref.root_p, "root_orientation": so3_log(ref.root_R.transpose(-1, -2)@R),
        "root_velocity": v-ref.root_v, "body_position": local-target_local,
        "body_orientation": so3_log(tR.transpose(-1, -2)@bR), "body_velocity": bv-tv,
        "joint_position": q-ref.q, "joint_velocity": qd-ref.qd}
    groups = {k: robust(x, scales[k]).flatten(1).mean(-1) for k,x in errors.items()}
    return torch.stack(list(groups.values())).mean(0), groups


def reference_prior(candidate, teacher, adapter, *, scales=None, global_reference=None):
    scales = dict(DEFAULT_SCALES, **(scales or {}))
    a, b = candidate.world_qpos, teacher.world_qpos.detach()
    prefix = candidate.ticket.prefix_frames
    if a.shape != b.shape or prefix < 1 or prefix >= a.shape[1]:
        raise ValueError("动作保持需要相同60帧物理轨迹和合法保护前缀")
    free = torch.arange(a.shape[1], device=a.device) >= prefix
    a = torch.cat((a[:, :prefix].detach(), a[:, prefix:]), 1)
    def state_and_bodies(x):
        R = quat_to_mat(x[..., 3:7]); q = x[..., 7:][..., adapter.to_sm]
        zeros = torch.zeros_like(x[..., :3])
        bodies = adapter.runtime.kin.forward(x[..., :3], R, zeros, zeros, q,
                                             torch.zeros_like(q), all_bodies=True)[0]
        local = (R.transpose(-1,-2)[:, :, None]@(bodies-x[..., :3][:, :, None])[..., None]).squeeze(-1)
        return R, q, local
    Ra, qa, ba = state_and_bodies(a)
    Rb, qb, bb = state_and_bodies(b)
    errors = {"root_position": a[..., :3]-b[..., :3],
        "root_orientation": so3_log(Rb.transpose(-1,-2)@Ra), "joint_position": qa-qb,
        "body_position": ba-bb}
    groups = {k: robust(v[:, free], scales[k]).mean() for k,v in errors.items()}
    groups["root_velocity"] = robust(torch.diff(a[..., :3]-b[..., :3], dim=1)[:, free[1:]]*30,
                                      scales["root_velocity"]).mean()
    groups["joint_velocity"] = robust(torch.diff(qa-qb, dim=1)[:, free[1:]]*30,
                                       scales["joint_velocity"]).mean()
    groups["body_velocity"] = robust(torch.diff(ba-bb, dim=1)[:, free[1:]]*30,
                                      scales["body_velocity"]).mean()
    groups["joint_acceleration"] = robust(torch.diff(qa-qb, n=2, dim=1)[:, free[2:]]*900,
                                           scales["joint_acceleration"]).mean()
    # 独立时间线保留整曲原动作；这里只报告偏差，执行主损失直接使用该目标。
    if global_reference is not None:
        ticks = candidate.ticket.request_tick+torch.arange(60, device=a.device)*20
        fixed = global_reference.source_at(ticks).detach()
        groups["fixed_timeline_joint_deviation_rms"] = (a[:, free, 7:]-fixed[:, free, 7:]).square().mean().sqrt()
    loss = torch.stack([v for k,v in groups.items() if not k.endswith("deviation_rms")]).mean()
    return loss, groups
