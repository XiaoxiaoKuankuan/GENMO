"""真实物理与WM配对轨迹的逐分量误差报告，不把一个pass当成精度说明。

输入是同时间戳的state55序列；按根位置、SO(3)姿态、速度、关节与权威FK足部计算
逐控制时刻的误差，再给出RMSE、逐帧误差P95、末帧误差和首次越过预声明阈值的时刻。
阈值只用于标记偏离，不自动授权训练。运动标签是运动学描述；真实接触事件必须另从
Newton求解器读取，不能把足高、GENMO contact2或该标签冒充真实碰撞证据。
"""

import torch
from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_log


def component_errors(predicted, real, kin):
    p, R, v, w, q, qd = unflatten_state(predicted)
    rp, rR, rv, rw, rq, rqd = unflatten_state(real)
    errors = {"root_xy_m": p[..., :2]-rp[..., :2], "root_height_m": p[..., 2:3]-rp[..., 2:3],
              "root_orientation_rad": so3_log(rR.transpose(-1, -2) @ R).norm(dim=-1, keepdim=True),
              "root_linear_velocity_m_s": v-rv, "root_angular_velocity_rad_s": w-rw,
              "joint_position_rad": q-rq, "joint_velocity_rad_s": qd-rqd}
    if kin is not None:
        bodies = [kin.forward(*(x.to(kin.axes.device) for x in unflatten_state(s)), all_bodies=True)[0]
                  for s in (predicted, real)]
        feet = [kin.body_names.index(n) for n in ("l_ankle_roll_link", "r_ankle_roll_link")]
        delta = bodies[0][:, feet]-bodies[1][:, feet]
        errors.update(foot_position_m=delta.flatten(1), foot_height_m=delta[..., 2])
    return {k: v.reshape(len(predicted), -1) for k, v in errors.items()}


def motion_labels(states):
    _, _, v, w, _, qd = unflatten_state(states)
    labels = []
    for index in range(len(states)):
        row = []
        if float(v[index, :2].norm()) > .15:
            row.append("root_translation")
        if abs(float(w[index, 2])) > .25:
            row.append("turning")
        if float(qd[index].abs().max()) > 1.:
            row.append("active_joints")
        labels.append(row or ["low_velocity"])
    return labels


def trajectory_metrics(predicted, real, kin, *, limits=None, start_tick=0):
    if len(predicted) != len(real) or not len(real):
        raise ValueError("配对误差需要等长非空真实轨迹")
    labels, result = motion_labels(real), {}
    for name, delta in component_errors(predicted, real, kin).items():
        frame = delta.square().mean(-1).sqrt()
        limit = (limits or {}).get(name)
        crossed = (frame > limit).nonzero().flatten() if limit is not None else []
        first = int(crossed[0]) if len(crossed) else None
        result[name] = {"rmse": float(delta.square().mean().sqrt()),
                       "p95_frame_error": float(torch.quantile(frame, .95)),
                       "endpoint_error": float(frame[-1]), "deviation_threshold": limit,
                       "first_deviation_tick": start_tick + (first+1)*12 if first is not None else None,
                       "first_deviation_seconds": (first+1)*.02 if first is not None else None,
                       "first_deviation_motion": labels[first] if first is not None else None}
    return result
