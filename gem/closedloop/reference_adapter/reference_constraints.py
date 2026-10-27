"""十帧独立提案的因果二阶参考约束，处理已提交锚点、接缝和真实名义halo。

本模块不是网络层：网络仍独立输出10×27个绝对名义残差。执行前用两个一阶
稳定滤波器的串联限制残差变化率；内部一阶状态由上一已提交的两个锚点精确恢复，
所以既不修改过去，也不引入额外观测或未来真实状态。每个未来提案都独立参与结果，
没有端点插值。对界限B、系数beta，残差速度逐分量不超过2B*beta/dt，残差加速度
不超过4B*beta²/dt²。根旋转在旋转向量球中作凸组合，然后仍以SO(3)复合；报告
实际SO(3)角速度/角加速度，不能把旋转向量导数冒充角速度。

第11点仍严格为2*d10-d9，只提供导数：其残差速度等于最后一段，残差加速度为零。
名义轨迹自身的导数单独计入总参考上界；本模块不声称能修复原名义轨迹的违约。
"""

import torch
from supermimic.rotations import so3_exp, so3_log


def constrain_plan(proposal, left, anchor, beta):
    """可微、线性且不detach的稳定约束；beta=1为原始独立十帧提案。"""
    if not 0 < beta <= 1:
        raise ValueError('参考滤波beta必须位于(0,1]')
    rho = 1-beta
    stage = (anchor-rho*left)/beta
    value = anchor
    result = []
    for target in proposal.unbind(1):
        stage = rho*stage+beta*target
        value = rho*value+beta*stage
        result.append(value)
    return torch.stack(result, 1)


def reference_kinematics(p, R, q, d, heading, dt=.02):
    """13个姿态覆盖t-1..t+11；返回包含锚点接缝和halo的真实离散导数。"""
    corrected_p = p+(heading[:, None]@d[..., :3, None]).squeeze(-1)
    corrected_R = R@so3_exp(d[..., 3:6])
    corrected_q = q+d[..., 6:]
    v = corrected_p.diff(dim=1)/dt
    w = so3_log(corrected_R[:, 1:]@corrected_R[:, :-1].transpose(-1, -2))/dt
    qd = corrected_q.diff(dim=1)/dt
    return dict(q=corrected_q[:, 1:-1], qd=qd, qdd=qd.diff(dim=1)/dt,
                root_v=v, root_a=v.diff(dim=1)/dt, root_w=w, root_alpha=w.diff(dim=1)/dt)


def quantiles(value):
    value = value.detach().float().reshape(-1).abs()
    q = torch.quantile(value, value.new_tensor([.5, .95, .99]))
    return dict(p50=float(q[0]), p95=float(q[1]), p99=float(q[2]), maximum=float(value.max()))
