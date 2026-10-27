"""正式SM交替循环的CPU语义回归，不启动仿真、方向审计或pilot。

使用真实冻结SM权重核对H8损失尺度及梯度、WM统计继承/Actor隔离、探索动作与
Replay/历史一致、Critic开关和H+1 GAE目标。合成状态标签仅证明算法实现与原版
公式一致，不宣称物理质量改善；真实更新计数由正式服务器训练日志给出。
"""
import copy
from dataclasses import replace
import pytest
import torch
from test_vector import vector
from gem.closedloop.reference_adapter.music_reward import AudioTrack
from gem.closedloop.reference_adapter.alternating_learning import (
    enable_wm_normalization, update_wm_normalization, wm_loss, policy_update, critic_update)
from gem.closedloop.reference_adapter.critic import lambda_returns
from gem.closedloop.reference_adapter.vector_collector import VectorCollector
from gem.closedloop.reference_adapter.vector_replay import VectorReplay
from supermimic.networks import Critic


def setup(system, count=2):
    engine, old = system
    old = replace(old, audio=AudioTrack.from_features(old.audio.features[:, :-10]))
    engine, _, control, z = vector((engine, old), count=count)
    control.sm_soft_termination_only = True
    return engine, control, z


def test_wm_loss_matches_original_time_sum_and_gradients(system):
    engine, control, z = setup(system)
    dynamics = engine.dynamics
    with torch.no_grad():
        states, targets, actions = [], [], []
        s = z['state']
        for _ in range(8):
            states.append(s)
            a = torch.zeros(2, 21); actions.append(a)
            s = dynamics.step(s, a, z['xi'])
            targets.append(s.clone())
    targets = torch.stack(targets, 1)
    targets[..., 13:34] += .001
    w = dict(origin='real', state=torch.stack(states, 1), next_state=targets,
             action=torch.stack(actions, 1), xi=z['xi'][:, None].expand(-1, 8, -1), mask=torch.ones(2, 8))
    actual, _ = wm_loss(dynamics, w)
    mean_loss, _ = dynamics.supervised_loss(w)
    torch.testing.assert_close(actual, mean_loss*8)
    p = tuple(dynamics.model.net.parameters())
    a = torch.autograd.grad(actual, p)
    b = torch.autograd.grad(mean_loss*8, p)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
        assert torch.isfinite(x).all()
    assert sum(x.square().sum() for x in a) > 0


def test_normalization_inherits_weights_and_updates_independent_WM(system):
    engine, control, z = setup(system)
    snapshot = control.bundle.frozen_snapshot()
    old = {k: v.clone() for k, v in engine.dynamics.state_dict().items()}
    enable_wm_normalization(engine.dynamics)
    for k, v in engine.dynamics.state_dict().items():
        torch.testing.assert_close(v, old[k], rtol=0, atol=0)
    with torch.no_grad():
        proposal = control.propose(z)
        next_state = engine.dynamics.step(z['state'], proposal['action'], z['xi'])
    norms = (engine.dynamics.model.in_norm, engine.dynamics.model.out_norm)
    previous = [float(n.count) for n in norms]
    # 原RunningNorm已有上限时保持冻结；本测试不清空count。
    update_wm_normalization(engine.dynamics, dict(start=z, proposal=proposal, end=dict(state=next_state)))
    for n, before in zip(norms, previous):
        assert float(n.count) == (before+len(next_state) if before < n.max_count else before)
    control.bundle.assert_unchanged(snapshot)


def test_noisy_action_is_the_replay_and_history_action(system):
    engine, control, z = setup(system)
    class Env:
        N = 2
        def step_physics(self, action):
            self.applied = action.clone()
            return z['state'].clone()
    env = Env()
    collector = VectorCollector(env, control)
    collector.state = z
    replay = VectorReplay(100, 2, 'cpu', compute_mc=False)
    row = collector.step(replay, action_noise=.1, auto_reset=False)
    actual = row['proposal']['action']
    assert not torch.equal(actual, row['proposal']['policy_action'])
    torch.testing.assert_close(env.applied, actual)
    torch.testing.assert_close(replay.data['applied_action'][0], actual)
    torch.testing.assert_close(row['end']['action'], actual)
    from supermimic.motion import select_reference
    expected = control.runtime.observation(row['end']['state'], actual, z['xi'],
        select_reference(row['proposal']['future'], (slice(None), 0)))
    torch.testing.assert_close(row['end']['history'][:, -1], expected)
    restored = VectorReplay(100, 2, 'cpu', compute_mc=False)
    restored.load_state_dict(replay.state_dict())
    assert not restored.compute_mc
    torch.testing.assert_close(restored.data['applied_action'][0], actual)


@pytest.mark.parametrize('enabled', [False, True])
def test_policy_freezes_WM_Actor_and_optional_critic(system, enabled):
    engine, control, z = setup(system)
    critic = Critic(2835)
    critic.requires_grad_(enabled)
    optimizer = torch.optim.Adam(control.adapter.parameters(), lr=3e-5, betas=(.9, .95))
    wm = {k: v.clone() for k, v in engine.dynamics.state_dict().items()}
    actor = control.bundle.frozen_snapshot()
    calls = []
    report, trajectory = policy_update(control, engine.dynamics, critic, optimizer, z, 8,
        terminal_value=enabled, microbatch=1, diagnostics=True, on_step=lambda r: calls.append(r.copy()))
    assert len(calls) == 1 and report['optimizer_step_completed']
    assert report['adapter_gradient_norm'] > 0 and report['parameter_delta'] > 0
    control.bundle.assert_unchanged(actor)
    for k, v in engine.dynamics.state_dict().items():
        torch.testing.assert_close(v, wm[k], atol=0, rtol=0)
    assert all(p.grad is None for p in engine.dynamics.parameters())
    assert all(p.grad is None for p in critic.parameters())
    if not enabled:
        assert trajectory is None and report['terminal'] == 0
        assert critic_update(critic, None, None)['critic_optimizer_steps'] == 0
    else:
        assert trajectory['features'].shape == (9, 2, 2835)
        assert not trajectory['features'].requires_grad
        original = copy.deepcopy(critic)
        original.norm.update(trajectory['features'])
        with torch.no_grad():
            expected = lambda_returns(trajectory['rewards'], trajectory['continuation'], original(trajectory['features']))
        opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
        v = critic_update(critic, opt, trajectory, epochs=1, minibatches=2)
        assert v['critic_optimizer_steps'] == 2
        assert v['critic_norm_count'] == 18
        assert v['critic_target_mean'] == pytest.approx(float(expected.mean()), abs=1e-6)


def test_formal_soft_termination_not_hard_failure_mask(system):
    engine, control, z = setup(system)
    original = control.runtime.termination
    control.runtime.termination = lambda state, ref, soft=False: (
        state.new_full((len(state),), .2) if soft else torch.ones(len(state), dtype=torch.bool))
    try:
        proposal = control.propose(z)
        _, _, _, _, continuation, done, failed = control.advance(z, proposal, z['state'])
        assert done.all() and failed.all()
        torch.testing.assert_close(continuation, torch.full((2,), .8))
        control.sm_soft_termination_only = False
        assert not control.advance(z, proposal, z['state'])[4].any()
    finally:
        control.runtime.termination = original
