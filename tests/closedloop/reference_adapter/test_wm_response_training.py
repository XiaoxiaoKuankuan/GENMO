"""真实SM权重上的WM监督数值回归，不将合成标签冒充Newton物理证据。

验证一步状态损失确实更新原WM.net、权重控制梯度，四条件奖励重放与原控制目标完全
同义且只乘一次dt，接触切换被保留而真实终止/时间不连续仍被拒绝。标签由数值夹具
构造，正式泛化和H32方向仍由独立真实物理对照确定。
"""
import copy
import torch
from test_repair import constrained_system
from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
from gem.closedloop.reference_adapter.precision import PrecisionObjective
from gem.closedloop.reference_adapter.reference_response import (
    ResponseRecorder, ReferenceResponseReplay, score_response_sequences, reward_response_loss)


def test_real_state_one_step_has_weighted_WM_gradient(system):
    engine, control, z = constrained_system(system, count=2)
    s = z['state'][:, None].expand(-1, 8, -1).clone()
    a = s.new_zeros(2, 8, 21)
    xi = z['xi'][:, None].expand(-1, 8, -1)
    with torch.no_grad():
        target = engine.dynamics.step(s.flatten(0, 1), a.flatten(0, 1), xi.flatten(0, 1)).reshape_as(s)
    target[..., 34:] += .1
    target[..., 10:13] += .01
    window = dict(origin='real', state=s, action=a, xi=xi, next_state=target,
                  mask=torch.ones(2, 8, dtype=torch.bool))
    loss, report = engine.dynamics.one_step_loss(window)
    params = tuple(engine.dynamics.model.net.parameters())
    grad = torch.autograd.grad(loss, params, retain_graph=True)
    scaled = torch.autograd.grad(3*loss, params)
    assert sum(g.square().sum() for g in grad) > 0
    for g, h in zip(grad, scaled):
        assert torch.isfinite(g).all()
        torch.testing.assert_close(h, g*3, rtol=1e-4, atol=1e-7)
    assert report['one_step_valid_transitions'] == 16
    assert all(p.grad is None for p in control.adapter.parameters())


def recorded_fixture(system, horizon=8):
    engine, control, z = constrained_system(system, count=4)
    z = {k: v[:1].expand(4, *v.shape[1:]).clone() for k, v in z.items()}
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds)
    control.precision = PrecisionObjective()
    control.soft_joint_limit_cost = False
    control.soft_imagination_continuation = False
    control.weights = dict(control.weights, reference_limit=0.)
    source = {k: v[:1].clone() for k, v in z.items()}
    rec = ResponseRecorder(z, 1, torch.zeros(1))
    with torch.no_grad():
        for t in range(horizon):
            offset = z['state'].new_zeros(4, 10, 27)
            offset[2, :, 6] = .001
            offset[3, :, 6] = -.001
            proposal = control.propose(z, residual_offset=offset)
            state = engine.dynamics.step(z['state'], proposal['action'], z['xi'])
            end, reward, raw, _, _, _, _ = control.advance(z, proposal, state)
            row = dict(start=z, end=end, proposal=proposal, reward=reward, raw=raw,
                done=torch.zeros(4, dtype=torch.bool), contacts=(torch.full((4, 2), t>=2), None))
            rec.append(row)
            z = end
    return engine, control, rec, source


def test_contact_change_kept_and_exact_reward_replay(system):
    engine, control, rec, source = recorded_fixture(system)
    pairs = rec.finish(source, 0, torch.tensor([8]), 0, 'train')
    assert pairs['contact_switch'].any()
    assert pairs['reliable'].all()
    assert not pairs['legacy_reliable'][:, -1].any()
    replay = ReferenceResponseReplay(); replay.add(pairs)
    assert replay.sample(2, horizon=8) is not None
    with torch.no_grad():
        actual = score_response_sequences(engine.dynamics, control, pairs, 8, use_real_labels=True)
    torch.testing.assert_close(actual, pairs['all_rewards'], rtol=0, atol=2e-7)
    pairs['all_rewards'][:, :, 2:, 3] -= .0002
    loss, report = reward_response_loss(engine.dynamics, control, pairs, 8, checkpoint_steps=True)
    grads = torch.autograd.grad(loss, tuple(engine.dynamics.model.net.parameters()))
    assert torch.isfinite(loss) and sum(g.square().sum() for g in grads)>0
    assert report['reward_response_fields']['body_angvel']['valid_deltas']>0
    assert all(p.grad is None for p in control.adapter.parameters())
    pairs['trajectory_valid'][:, 4:] = False
    replay = ReferenceResponseReplay(); replay.add(pairs)
    assert replay.sample(2, horizon=8) is None
