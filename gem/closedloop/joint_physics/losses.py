"""同一数据动作的物理执行误差，固定世界坐标且按有效折扣质量归一化。

根位姿只比较一次；身体位置/姿态/速度转到各自根坐标，避免根yaw重复计权。FK、旋转
全部调用SM权威实现，使用明确物理尺度的pseudo-Huber。实际执行关节限位独立于原监督
中生成参考限位；Action Rate为零，没有Stability Reward或生成教师项。
"""

import torch

from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_log
from gem.closedloop.mb_control.dance_losses import robust


def execution_terms(state, reference, kin, cfg):
    p, r, v, w, q, qd = unflatten_state(state.float())
    t = reference
    bp, br, bv, _ = kin.forward(p, r, v, w, q, qd, all_bodies=True)
    tp, tr, tv, _ = kin.forward(t.root_p, t.root_R, t.root_v, t.root_w, t.q, t.qd, all_bodies=True)
    rt, tt = r.transpose(-1, -2)[:, None], t.root_R.transpose(-1, -2)[:, None]
    local = lambda rotation, value: (rotation @ value[..., None]).squeeze(-1)
    body_r, target_r = rt @ br, tt @ tr
    errors = {"root_position": p - t.root_p, "root_orientation": so3_log(t.root_R.transpose(-1, -2) @ r),
        "joint_position": q-t.q, "joint_velocity": qd-t.qd,
        "body_position": local(rt, bp-p[:, None])-local(tt, tp-t.root_p[:, None]),
        "body_orientation": so3_log(target_r.transpose(-1, -2) @ body_r),
        "body_velocity": local(rt, bv-v[:, None])-local(tt, tv-t.root_v[:, None])}
    terms = {key: robust(error, cfg["scales"][key]).flatten(1).mean(-1) for key, error in errors.items()}
    total = sum(terms[key]*cfg["weights"][key] for key in terms)
    violation = torch.relu(kin.q_lo-q) + torch.relu(q-kin.q_hi)
    safety = robust(violation, .05).mean(-1)
    return total, safety, terms


def normalized_window(values, discount):
    if not values or not 0 < discount <= 1:
        raise ValueError("执行Loss需要非空有效窗口和(0,1]折扣")
    stacked = torch.stack(values)
    weights = stacked.new_tensor([discount**i for i in range(len(values))])
    return (stacked*weights[:, None]).sum(0)/weights.sum()
