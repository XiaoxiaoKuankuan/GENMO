"""预览电机目标限幅的执行器合同测试，不模拟Newton或改变正式训练。

用非零default_q、逐环境xi偏置和不同alpha验证反算动作经过实际同款公式后确实得到
机械上下限内的目标，覆盖正/负饱和、未饱和逐位保留以及输入不被修改。这里仅验证
输出限幅，不把它当作真实关节无越界、整曲成功或硬件安全证明。
"""

from types import SimpleNamespace

import torch

from gem.closedloop.joint_physics.preview_limits import clamp_motor_action


def test_motor_target_limits_use_actual_scale_and_offset():
    env = SimpleNamespace(default_q=torch.linspace(-.02, .02, 21),
        alpha=torch.linspace(.1, .4, 21), xi=torch.zeros(2, 25))
    env.xi[0, 1:22] = .007
    env.xi[1, 1:22] = -.009
    action = torch.zeros(2, 21); action[0, 0] = 10; action[1, 1] = -10
    original = action.clone(); original_xi = env.xi.clone()
    lo, hi = torch.full((21,), -.17), torch.full((21,), .17)
    executed, requested, target, saturated = clamp_motor_action(action, env, lo, hi)
    reconstructed = env.default_q+env.xi[:, 1:22]+env.alpha*executed
    torch.testing.assert_close(reconstructed, target, rtol=0, atol=2e-8)
    assert (target >= lo).all() and (target <= hi).all()
    assert saturated.sum() == 2 and saturated[0, 0] and saturated[1, 1]
    assert requested[0, 0] > hi[0] and requested[1, 1] < lo[1]
    assert target[0, 0] == hi[0] and target[1, 1] == lo[1]
    torch.testing.assert_close(action, original, rtol=0, atol=0)
    torch.testing.assert_close(env.xi, original_xi, rtol=0, atol=0)
