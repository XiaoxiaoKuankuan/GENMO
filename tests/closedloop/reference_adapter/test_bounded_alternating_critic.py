"""本轮根奖励权重与有界交替Critic的CPU语义回归。

使用真实冻结SM权重核对H64全图BPTT、可微价值上界、真实终止的TD标签和Critic
冻结输入梯度。真实标签夹具只用于核验公式，不能声称完成物理MC校准；正式物理
采集、首个更新及保存/恢复另在Server-1独立目录验证，不触碰旧checkpoint。
"""

import copy
import torch

from test_alternating import setup
from gem.closedloop.reference_adapter.critic import AlternatingCritic
from gem.closedloop.reference_adapter.alternating_learning import critic_update, policy_update


def test_value_bound_preserves_input_gradient_and_checkpoint():
    torch.manual_seed(42)
    critic = AlternatingCritic(12, 12.4).double()
    x = torch.randn(2, 12, dtype=torch.double, requires_grad=True)
    assert critic(x).abs().max() < 1e-6
    with torch.no_grad():
        critic.net[-1].weight.normal_(std=.02)
    assert torch.autograd.gradcheck(critic, (x,), fast_mode=True)
    grad, = torch.autograd.grad(critic(x).sum(), x)
    assert grad.isfinite().all() and grad.norm() > 0
    restored = AlternatingCritic(12, 1.).double()
    restored.load_state_dict(copy.deepcopy(critic.state_dict()))
    torch.testing.assert_close(restored(x), critic(x), atol=0, rtol=0)
    with torch.no_grad():
        critic.net[-1].weight.zero_()
        critic.net[-1].bias.fill_(100.)
    assert (critic(x) <= critic.value_upper_bound).all()
    with torch.no_grad():
        critic.net[-1].bias.fill_(-100.)
    assert (critic(x) < -99.).all(), '真实奖励没有已证明的负下界，不能对负价值强行截断'


def test_real_TD_anchor_uses_true_terminal_and_detached_labels():
    critic = AlternatingCritic(12, 12.4)
    with torch.no_grad():
        critic.net[-1].bias.fill_(2.)
    x, xn = torch.randn(4, 12), torch.randn(4, 12)
    rewards = torch.tensor([.1, .2, -.5, -.2])
    done = torch.tensor([False, False, True, True])
    with torch.no_grad():
        expected = rewards+.99*(~done)*critic(xn)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    report = critic_update(critic, optimizer, None,
        real=dict(features=x, next_features=xn, rewards=rewards, done=done), epochs=1, batch_size=2)
    assert report['critic_optimizer_steps'] == 2
    assert report['critic_real_terminals'] == 2
    assert abs(report['critic_real_target_mean']-float(expected.mean())) < 1e-6
    assert report['critic_imagined_mse'] is None
    assert report['critic_target_source'] == 'real_TD0_not_MC'


def test_H64_critic_BPTT_and_combined_update_freeze_other_networks(system):
    engine, control, z = setup(system, count=1)
    critic = AlternatingCritic(2835, 12.4)
    with torch.no_grad():
        critic.net[-1].weight.normal_(std=.001)
        critic.norm.update(control.critic_features(z))
    before = copy.deepcopy(critic.state_dict())
    optimizer = torch.optim.Adam(control.adapter.parameters(), lr=1e-5)
    report, trajectory = policy_update(control, engine.dynamics, critic, optimizer, z, 64,
                                       terminal_value=True, microbatch=1)
    assert report['optimizer_step_completed'] and report['parameter_delta'] > 0
    assert trajectory['features'].shape == (65, 1, 2835)
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert all(p.grad is None for p in critic.parameters())
    assert all(p.grad is None for p in engine.dynamics.parameters())
    real = dict(features=trajectory['features'][0], next_features=trajectory['features'][1],
                rewards=torch.tensor([.08]), done=torch.tensor([False]))
    report = critic_update(critic, torch.optim.Adam(critic.parameters(), lr=1e-4), trajectory,
                           real=real, epochs=1, batch_size=16)
    assert report['critic_optimizer_steps'] == 4 and report['critic_real_samples'] == 1
    assert report['critic_value_max'] <= 12.4+1e-6
