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


def test_zero_reference_preserves_arbitrary_quaternions_and_rotation_gradients():
    torch.manual_seed(24000)
    poses, goal = inputs(batch=128)
    quat = torch.randn(128, 12, 4)
    # 包括任意旋转、四元数正负号及实际数据的微小非单位长度误差。
    poses[..., 3:7] = quat / quat.norm(dim=-1, keepdim=True) * 1.000001
    goal = torch.randn_like(goal)
    residual = torch.zeros(128, 10, 27, requires_grad=True)
    out = correct_goal(goal, poses, residual)
    torch.testing.assert_close(out, goal, atol=0, rtol=0)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.isfinite(residual.grad).all()
    assert residual.grad[..., 3:6].abs().sum() > 0


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


def test_current_source_loads_40k_configuration_without_retired_sampler_field():
    from gem.closedloop.noetix_adapter.training import make_config
    settings = dict(run_name='test', root='/tmp', seed=42, library='/tmp/train.pt',
                    horizon=64, adapter_lr=5e-6, target_updates=30000)
    saved = {'env': {'sampler': {'patience_iterations': 5000}}, 'robot': 'bumi4340'}
    cfg = make_config(saved, settings)
    assert cfg.env.sampler.retire_enabled is False
    assert cfg.policy_trainer.use_critic is False
    assert cfg.policy_trainer.horizon == 64
    assert cfg.policy_trainer.lr == 5e-6
    assert saved['env']['sampler']['patience_iterations'] == 5000


def test_critic_sees_nominal_goal_and_state_anchor_gradient():
    from gem.closedloop.noetix_adapter.critic import nominal_goal
    packed = torch.randn(3, 2, 1534, requires_grad=True)
    goal = nominal_goal(packed)
    assert goal.shape == (3, 2, 606)
    goal.sum().backward()
    assert torch.equal(packed.grad[..., :600], torch.ones_like(packed.grad[..., :600]))
    assert torch.equal(packed.grad[..., -6:], torch.ones_like(packed.grad[..., -6:]))
    assert torch.count_nonzero(packed.grad[..., 600:-6]) == 0


def test_real_td_terminal_never_bootstraps():
    from gem.closedloop.noetix_adapter.critic import td_target
    reward = torch.tensor([.1, -.2])
    target = td_target(reward, torch.tensor([0., 1.]), torch.tensor([4., 100.]), .99)
    torch.testing.assert_close(target, torch.tensor([4.06, -.2]))


def test_adapter_decay_keeps_critic_optimizer_lr_independent():
    from types import SimpleNamespace
    from gem.closedloop.noetix_adapter.critic import AdapterPolicyTrainer
    actor = torch.nn.Parameter(torch.ones(()))
    critic = torch.nn.Parameter(torch.ones(()))
    trainer = SimpleNamespace(opt=torch.optim.Adam([actor], lr=1e-5),
                              critic_opt=torch.optim.Adam([critic], lr=3e-5))
    AdapterPolicyTrainer.set_lr(trainer, 2.5e-6)
    assert trainer.opt.param_groups[0]['lr'] == 2.5e-6
    assert trainer.critic_opt.param_groups[0]['lr'] == 3e-5


def test_critic_configuration_and_fifty_thousand_adapter_budget():
    from gem.closedloop.noetix_adapter.training import make_config
    cfg = make_config({'robot': 'bumi4340'}, dict(run_name='test', root='/tmp', seed=42,
        library='/tmp/train.pt', horizon=64, adapter_lr=1e-5, use_critic=True,
        critic_lr=3e-5, critic_epochs=2, critic_minibatches=4, wm_warmup=500, target_updates=50000))
    assert cfg.policy_trainer.use_critic is True
    assert cfg.policy_trainer.critic_lr == 3e-5
    assert cfg.policy_trainer.critic_epochs * cfg.policy_trainer.critic_minibatches == 8
    assert cfg.schedule.total_iterations == 50500
    assert cfg.env.perturbations.push_enabled is False
    assert cfg.env.perturbations.kp_range == (1., 1.)


def test_accelerated_base_keeps_saved_physics_and_extended_adapter_reference():
    """新基座的物理扰动/奖励必须保留，紧凑参考预计算不能丢掉音乐和姿态输入。"""
    from gem.closedloop.noetix_adapter.training import make_config
    from supermimic.config import TrainConfig
    if not hasattr(TrainConfig().policy_trainer, 'prepare_references'):
        import pytest
        pytest.skip('本测试针对含参考预计算的加速版源码')
    saved = dict(robot='bumi4340', env=dict(self_collision_weight=-7., joint_limit_weight=-2.,
        action_rate_weight=-.2, survival_weight=4., joint_velocity_weight=-.0005,
        mass_pd_context=True, perturbations=dict(torso_mass_range=[.8, 1.2],
            kp_range=[.7, 1.2], kd_range=[.7, 1.2], push_enabled=True)),
        policy_trainer=dict(prepare_references=True, reference_compile='default', skip_frozen_prepass=True))
    cfg = make_config(saved, dict(run_name='test', root='/tmp', seed=42, library='/tmp/train.pt',
        horizon=64, adapter_lr=5e-6, use_critic=True, critic_lr=1e-5, wm_warmup=500, target_updates=50000))
    assert cfg.env.perturbations.push_enabled and cfg.env.perturbations.kd_range == [.7, 1.2]
    assert (cfg.env.self_collision_weight, cfg.env.joint_limit_weight, cfg.env.action_rate_weight) == (-7., -2., -.2)
    assert cfg.env.survival_weight == 4. and cfg.env.mass_pd_context
    assert not cfg.policy_trainer.prepare_references and cfg.policy_trainer.reference_compile is None
    assert cfg.policy_trainer.skip_frozen_prepass and cfg.policy_trainer.use_critic
    assert not cfg.defer_metrics and not cfg.async_checkpoint
    assert saved['policy_trainer']['prepare_references']
