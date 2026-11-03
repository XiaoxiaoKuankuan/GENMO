"""同一数据动作的物理执行误差，固定世界坐标且按有效折扣质量归一化。

根位姿只比较一次；身体位置/姿态/速度转到各自根坐标，避免根yaw重复计权。FK、旋转
全部调用SM权威实现，使用明确物理尺度的pseudo-Huber。实际执行关节限位独立于原监督
中生成参考限位；可显式启用机械上下限及安全余量的平方铰链惩罚，均值与最严重关节
共同计权，不裁剪预测状态、不截断输入梯度。有限惩罚不能数学保证永不越界；真实Newton
硬检查独立保留。Action Rate为零，没有Stability Reward或生成教师项。
"""

import torch

from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_log
from gem.closedloop.mb_control.dance_losses import robust


def joint_limit_constraint(q, kin, cfg):
    """按权威SM关节顺序计算机械越界和安全余量，最坏关节不会被全关节均值稀释。"""
    violation = torch.relu(kin.q_lo-q) + torch.relu(q-kin.q_hi)
    options = cfg.get('joint_limit_constraint', {})
    if options.get('enabled', False):
        margin = options['margin_rad']
        residual = torch.relu(kin.q_lo+margin-q) + torch.relu(q-(kin.q_hi-margin))
        squared = (residual/options['scale_rad']).square()
        penalty = squared.mean(-1)+squared.amax(-1)
    else:
        penalty = q.sum(-1)*0.
    return penalty, violation


def execution_limit_diagnostics(states, kin, cfg):
    """只对已接受WM步骤统计；返回GPU聚合量供日志间隔全rank归约，不保存大状态链。"""
    q = states.detach().float()[..., 13:34].reshape(-1, 21)
    penalty, excess = joint_limit_constraint(q, kin, cfg)
    violated = excess > 1e-4
    # sums: 窗口数、状态步数、越界步数、越界窗口数、惩罚步总和、逐关节越界步数。
    sums = torch.cat((q.new_tensor([1., len(q)]), violated.any(-1).sum().reshape(1).float(),
                      violated.any().reshape(1).float(), penalty.sum().reshape(1), violated.sum(0).float()))
    return sums, excess.amax(0)


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
    penalty, violation = joint_limit_constraint(q, kin, cfg)
    constraint = cfg.get('joint_limit_constraint', {})
    total = total+constraint.get('weight', 0.)*penalty
    terms['joint_limit_constraint'] = penalty
    terms["joint_limit_violation_rad"] = violation.mean(-1)
    safety = state.new_zeros(state.shape[0])
    return total, safety, terms


def normalized_window(values, discount):
    if not values or not 0 < discount <= 1:
        raise ValueError("执行Loss需要非空有效窗口和(0,1]折扣")
    stacked = torch.stack(values)
    weights = stacked.new_tensor([discount**i for i in range(len(values))])
    return (stacked*weights[:, None]).sum(0)/weights.sum()
