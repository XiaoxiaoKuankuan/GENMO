"""验证新606维底层控制器的参考Adapter数学及冻结边界。

覆盖零残差不改变原目标、十帧独立输出和中心差分关系、SO(3)零点梯度、
Actor权重/RunningNorm冻结但输入梯度连通。使用真实40k接口的小网络夹具，
正式40k权重另由训练启动的ONNX对照与真实环境零Adapter检查验证。
"""

import torch

from supermimic.policy.nets import Policy, PolicyCfg
from gem.closedloop.noetix_adapter.model import correct_goal, ReferencePolicy, INPUT_SIZE, MUSIC_SIZE


def inputs(batch=3):
    poses = torch.zeros(batch, 12, 28)
    poses[..., 2] = .7
    poses[..., 3] = 1
    goal = torch.zeros(batch, 10, 60)
    goal[..., 44] = -1
    goal[..., 54] = 1
    goal[..., 57] = 1
    anchor = torch.tensor([1., 0., 0., 1., 0., 0.]).expand(batch, -1)
    return poses, torch.cat((goal.flatten(1), anchor), -1)


def test_zero_reference_preserves_native_goal():
    poses, goal = inputs()
    out = correct_goal(goal, poses, torch.zeros(3, 10, 27))
    torch.testing.assert_close(out, goal, atol=0, rtol=0)


def test_independent_future_frames_and_derivative_support():
    poses, goal = inputs()
    d = torch.zeros(3, 10, 27)
    d[:, 9, 6] = .04
    out = correct_goal(goal, poses, d)[:, :600].reshape(3, 10, 60)
    # 未来第10点不占Actor当前0..9窗口，但作为第9点右支持产生1 rad/s速度。
    torch.testing.assert_close(out[:, 9, 21], torch.ones(3))
    torch.testing.assert_close(out[:, :, 0], torch.zeros(3, 10))
    d[:, 2, 7] = .01
    out = correct_goal(goal, poses, d)[:, :600].reshape(3, 10, 60)
    torch.testing.assert_close(out[:, 3, 1], torch.full((3,), .01))
    torch.testing.assert_close(out[:, 2, 1], torch.zeros(3))


def test_zero_rotation_has_finite_nonzero_gradients():
    poses, goal = inputs()
    residual = torch.zeros(3, 10, 27, requires_grad=True)
    out = correct_goal(goal, poses, residual)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.isfinite(residual.grad).all()
    assert residual.grad[..., 3:6].abs().sum() > 0
    assert residual.grad[:, -1].abs().sum() > 0


def test_optimizer_only_changes_adapter():
    torch.manual_seed(42)
    pc = PolicyCfg(encoder_hidden=(32, 16), decoder_hidden=(32, 16), num_joints=21,
                   goal_dim=606, goal_frame_dim=60, proprio_anchor_ori=True,
                   goal_root_height=False, future_frames=10)
    actor = Policy(pc)
    before = {k: v.clone() for k,v in actor.state_dict().items()}
    model = ReferencePolicy(actor, [f'joint{i}' for i in range(21)])
    poses, goal = inputs()
    packed = torch.cat((goal[:, :600], poses.flatten(1), torch.zeros(3, MUSIC_SIZE), goal[:, -6:]), -1)
    obs = torch.randn(3, 750)
    torch.testing.assert_close(model(obs, packed), actor(obs, goal), atol=0, rtol=0)
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-5)
    (model(obs, packed) * torch.randn(3, 21)).sum().backward()
    assert model.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in actor.parameters())
    opt.step()
    model.update_input_norms(obs, packed)
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert INPUT_SIZE == 1948
