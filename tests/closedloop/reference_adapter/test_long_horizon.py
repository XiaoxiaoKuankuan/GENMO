"""H32/H64响应与BPTT数值回归，使用真实SM冻结权重但不冒充真实物理验证。

检查不同长度真实响应标签的储存合同、FK身体角速度来源、原SM奖励不变及删除重复
Huber项。比较有无梯度检查点的H32/H64完整Adapter梯度，验证节省显存没有截断历史、
残差或执行状态计算图；所有合成标签仅用于数值测试，正式方向仍需独立真实仿真。
"""

import torch
import pytest
from gem.closedloop.reference_adapter.reference_response import ReferenceResponseReplay, response_groups, consistency_loss, validate_response
from gem.closedloop.reference_adapter.precision import PrecisionObjective, NAMES, RATIOS
from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
from gem.closedloop.reference_adapter.vector_learning import rollout
from test_repair import constrained_system


def test_true_response_lengths_and_capacity_restore():
    replay = ReferenceResponseReplay(capacity=7)
    for h in (8, 16, 32, 64):
        value = torch.arange(h).float()[None, :, None].expand(2, -1, 21)+h*100
        replay.add(dict(origin='current_policy_real_reference_response', split='train', rejected={},
            state=torch.zeros(2, 55), song=torch.full((2,), h), plus_action=value, minus_action=-value,
            reliable=torch.ones(2, h, dtype=torch.bool)))
    assert sum(len(b['state']) for b in replay.buckets.values()) == 7
    assert replay.total_pairs == 8
    for h in (8, 16, 32, 64):
        sampled = replay.sample(64, horizon=h)
        assert sampled['plus_action'].shape == (64, h, 21)
        assert (sampled['song'] >= h).all()
        torch.testing.assert_close(sampled['plus_action'][:, -1, 0], (sampled['song']*100+h-1).float())
    assert replay.sample(1, horizon=65) is None
    restored = ReferenceResponseReplay()
    restored.load_state_dict(replay.state_dict(), 'cpu')
    for h, bucket in replay.buckets.items():
        for k, v in bucket.items():
            torch.testing.assert_close(v, restored.buckets[h][k], rtol=0, atol=0)


def test_body_w_is_FK_supervision_not_an_extra_reward(system):
    _, control, z = constrained_system(system, count=2)
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds)
    control.precision = PrecisionObjective()
    control.soft_joint_limit_cost = False
    control.weights = dict(control.weights, reference_limit=0.)
    proposal = control.propose(z)
    nxt = z['state'].clone(); nxt[:, 10:13] += .2; nxt[:, 34:] += .1; nxt.requires_grad_()
    _, _, raw, terms, *_ = control.advance(z, proposal, nxt)
    assert 'precision_body_angular_velocity' not in terms
    assert 'absolute_body_angular_velocity' in raw
    assert set(k.removeprefix('precision_') for k in terms if k.startswith('precision_')) == set(NAMES)
    assert RATIOS == (.30, .25, .20, .10)
    groups = response_groups(z['state'], nxt, z['state'], control.runtime.kin)
    gradient, = torch.autograd.grad(groups['body_angular_velocity'].square().sum(), nxt)
    assert gradient[:, 10:13].norm() > 0 and gradient[:, 34:].norm() > 0
    expected = raw['track']+.02*control.runtime.mdp.cfg.w_action_rate*raw['action_rate']
    torch.testing.assert_close(terms['sm'], expected, atol=0, rtol=0)


def test_long_response_does_not_count_single_reliable_step_as_H32():
    replay = ReferenceResponseReplay()
    mask = torch.ones(2, 64, dtype=torch.bool); mask[0, 1:] = False; mask[1, 32:] = False
    replay.add(dict(origin='current_policy_real_reference_response', split='train', rejected={},
        state=torch.zeros(2, 55), song=torch.arange(2), plus_action=torch.zeros(2, 64, 21),
        minus_action=torch.zeros(2, 64, 21), reliable=mask, intervention_steps=torch.tensor([1, 32])))
    assert replay.sample(32, horizon=32)['song'].eq(1).all()
    assert replay.sample(1, horizon=64) is None
    assert replay.coverage()['windows']['64']['reliable_at']['32'] == 1


def test_response_metrics_exclude_states_after_real_termination(system):
    engine, control, z = constrained_system(system, count=2)
    state, rows = z['state'], []
    action = torch.zeros(2, 16, 21)
    with torch.no_grad():
        for t in range(16):
            state = engine.dynamics.step(state, action[:, t], z['xi'])
            rows.append(state)
    target = torch.stack(rows, 1)
    valid = torch.ones(2, 16, dtype=torch.bool)
    valid[1, 4:] = False
    # 模拟第二条轨迹结束后已属于其它episode的标签，不能进入误差或可靠对统计。
    target[1, 4:, :3] += 1000
    target[1, 4:, 10:13] += 1000
    target[1, 4:, 34:] += 1000
    pairs = dict(state=z['state'], xi=z['xi'], plus_action=action, minus_action=action,
        plus_next=target, minus_next=target, reliable=valid.clone(), trajectory_valid=valid,
        contact_switch=torch.zeros_like(valid))
    report = validate_response(engine.dynamics, pairs, horizons=(16,), kin=control.runtime.kin)['horizons']['16']
    assert report['continuous_real_pairs'] == report['reliable_pairs'] == 1
    assert report['plus_state_error']['joint_velocity_rad_s'] == 0
    assert report['plus_state_error']['root_position_m'] == 0
    assert all(field['physical_response_rmse'] == 0 for field in report['fields'].values())
    pairs['trajectory_valid'][:] = False
    pairs['reliable'][:] = False
    report = validate_response(engine.dynamics, pairs, horizons=(16,), kin=control.runtime.kin)['horizons']['16']
    assert report['continuous_real_pairs'] == report['reliable_pairs'] == 0
    assert report['plus_state_error'] is None
    assert all(field['physical_response_rmse'] is None for field in report['fields'].values())


@pytest.mark.parametrize('horizon', [32, 64])
def test_checkpoint_keeps_complete_long_BPTT_gradient(system, horizon):
    engine, control, z = constrained_system(system, count=1)
    control.constraint_beta = 1.
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds)
    control.precision = PrecisionObjective()
    control.soft_joint_limit_cost = False
    control.weights = dict(control.weights, reference_limit=0.)
    control.soft_imagination_continuation = False
    parameters = tuple(control.adapter.parameters())
    results = []
    frozen = engine.step.bundle.frozen_snapshot()
    with engine.dynamics.adapter_phase():
        for checkpointed in (False, True):
            result = rollout(control, engine.dynamics, None, z, horizon, terminal_value=False,
                             checkpoint_steps=checkpointed)
            gradients = torch.autograd.grad(result['objective'].sum(), parameters)
            results.append((result['objective'].detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)
        assert torch.isfinite(b).all()
    assert sum(g.square().sum() for g in results[1][1]) > 0
    assert all(p.grad is None for p in engine.dynamics.parameters())
    engine.step.bundle.assert_unchanged(frozen)


def test_long_velocity_response_backprop_without_new_heads(system):
    engine, control, z = constrained_system(system, count=1)
    plus, minus = z['state'], z['state']
    rows_p, rows_m = [], []
    ap = torch.zeros(1, 16, 21)+.02; am = -ap
    with torch.no_grad():
        for t in range(16):
            plus = engine.dynamics.step(plus, ap[:, t], z['xi'])
            minus = engine.dynamics.step(minus, am[:, t], z['xi'])
            rows_p.append(plus); rows_m.append(minus)
    pairs = dict(state=z['state'], xi=z['xi'], plus_action=ap, minus_action=am,
        plus_next=torch.stack(rows_p, 1), minus_next=torch.stack(rows_m, 1),
        reliable=torch.ones(1, 16, dtype=torch.bool))
    pairs['plus_next'][:, :, 10:13] += .01
    names = tuple(engine.dynamics.state_dict())
    loss, report = consistency_loss(engine.dynamics, pairs, 16, kin=control.runtime.kin, checkpoint_steps=True)
    gradients = torch.autograd.grad(loss, tuple(engine.dynamics.model.net.parameters()))
    assert loss > 0 and report['response_group_losses']['root_angular_velocity'] > 0
    assert report['response_group_losses']['body_angular_velocity'] > 0
    assert sum(g.square().sum() for g in gradients) > 0
    assert names == tuple(engine.dynamics.state_dict())
    with pytest.raises(ValueError, match='缺少真实'):
        consistency_loss(engine.dynamics, pairs, 32, kin=control.runtime.kin)
