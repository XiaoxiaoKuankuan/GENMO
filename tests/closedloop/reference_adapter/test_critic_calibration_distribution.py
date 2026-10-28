"""Critic 真实回报边界与 WM 归一化函数等价的数值回归。

这里检查真实 MC 不跨 reset、不把截断当终止，未裁剪的 Critic 高估不能被上界
保护掩盖，以及真实 SM 权重迁移前后的预测/输入导数。合成连续状态仅用于数值
合同，不替代真实仿真评价，不产生任何正式训练输出或模型收益结论。
"""

import pytest
import torch

from gem.closedloop.reference_adapter.critic_calibration import (
    real_mc_suffixes, discounted_value_upper_bound, bounded_terminal_value, physical_calibration,
    terminal_value_allowed, preserve_critic_input_function, real_anchor_loss)
from gem.closedloop.reference_adapter.wm_distribution import (
    rebase_wm_normalization, distribution_shift, restore_rebased_normalization_policy)


def test_mc_requires_real_terminal_continuity_and_never_crosses_reset():
    reward = torch.ones(5, 2)
    episode = torch.tensor([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]])
    frame = torch.tensor([[8, 0], [9, 1], [0, 4], [1, 5], [2, 6]])
    terminal = torch.tensor([[0, 0], [1, 0], [0, 0], [0, 0], [0, 1]], dtype=torch.bool)
    target, valid = real_mc_suffixes(reward, episode, frame, terminal, torch.ones_like(terminal), gamma=.5)
    assert valid[:, 0].tolist() == [True, True, False, False, False]
    assert valid[:, 1].tolist() == [False, False, True, True, True]
    torch.testing.assert_close(target[:2, 0], torch.tensor([1.5, 1.]))
    torch.testing.assert_close(target[2:, 1], torch.tensor([1.75, 1.5, 1.]))


def test_value_bound_and_raw_calibration_reject_bootstrap_inflation():
    assert discounted_value_upper_bound() == pytest.approx(10)
    assert discounted_value_upper_bound(remaining_steps=1) == pytest.approx(.1)
    raw = torch.tensor([5., 15.], requires_grad=True)
    bounded_terminal_value(raw).sum().backward()
    torch.testing.assert_close(raw.grad, torch.tensor([1., 0.]))
    actual = torch.linspace(2., 8., 128)
    songs = torch.arange(128)//64
    bad = physical_calibration(actual+10, actual, songs, split='validation', train_return_mean=5.)
    good = physical_calibration(actual, actual, songs, split='validation', train_return_mean=5.)
    assert not terminal_value_allowed(True, bad, real_anchor_updates=10)
    assert not terminal_value_allowed(True, good, real_anchor_updates=0)
    assert not terminal_value_allowed(False, good, real_anchor_updates=10)
    assert terminal_value_allowed(True, good, real_anchor_updates=10)


def test_critic_normalizer_change_preserves_values_and_gradients():
    from supermimic.networks import Critic
    torch.manual_seed(31)
    critic = Critic(7).double()
    x = torch.randn(16, 7, dtype=torch.float64, requires_grad=True)
    old = critic(x)
    old_grad = torch.autograd.grad(old.sum(), x)[0]
    preserve_critic_input_function(critic, torch.randn(7, dtype=torch.float64),
                                  torch.rand(7, dtype=torch.float64)+.1, 1024)
    new = critic(x)
    torch.testing.assert_close(new, old, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(new.sum(), x)[0], old_grad, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match='train'):
        real_anchor_loss(critic, x, old.detach(), split='validation')


def test_wm_normalization_migration_preserves_physical_function_and_action_gradient(system):
    from supermimic.mdp import unflatten_state, normalize_xi
    engine, z = system
    dynamics = engine.dynamics
    state, xi = z.executed_state.expand(4, -1), z.xi.expand(4, -1)
    action = torch.randn(4, 21).requires_grad_()
    snapshot = engine.step.bundle.frozen_snapshot()
    old = dynamics.step(state, action, xi)
    old_gradient = torch.autograd.grad(old.square().sum(), action)[0]
    inputs = dynamics.model.inputs(unflatten_state(state), action.detach(), normalize_xi(xi))
    # 非退化训练统计模拟不同分布，不改目标状态定义；增大样本只为稳定测试数值。
    inputs = inputs.repeat(64, 1)+torch.randn(256, inputs.shape[-1])*.3
    raw = dynamics.model.targets(unflatten_state(state), unflatten_state(old.detach()))
    targets = raw.repeat(64, 1)+torch.randn(256, raw.shape[-1])*.03
    optimizer = torch.optim.Adam(dynamics.model.net.parameters(), lr=3e-5)
    for parameter in dynamics.model.net.parameters():
        optimizer.state[parameter] = dict(step=torch.tensor(1.), exp_avg=torch.zeros_like(parameter),
                                         exp_avg_sq=torch.ones_like(parameter))
    untouched = dynamics.model.net[3].weight
    untouched_state = optimizer.state[untouched]
    report = rebase_wm_normalization(dynamics, inputs, targets, split='train', optimizer=optimizer)
    assert report['optimizer_tensors_reinitialized'] == 4
    assert optimizer.state[untouched] is untouched_state
    new = dynamics.step(state, action, xi)
    torch.testing.assert_close(new, old, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.autograd.grad(new.square().sum(), action)[0], old_gradient,
                               atol=2e-5, rtol=1e-4)
    assert report['old_counts']['input'] >= 2e8
    assert dynamics.model.in_norm.count == 256
    engine.step.bundle.assert_unchanged(snapshot)
    with pytest.raises(ValueError, match='train'):
        rebase_wm_normalization(dynamics, inputs, targets, split='validation')
    diag = distribution_shift(dynamics.model.in_norm, inputs)
    assert diag['running_statistics_frozen'] and diag['mean_shift_in_old_std_max'] < 1e-5
    dynamics.model.in_norm.max_count = 2e8
    dynamics.model.out_norm.max_count = 2e8
    restore_rebased_normalization_policy(dynamics, report)
    assert dynamics.model.in_norm.max_count == dynamics.model.in_norm.count == 256
