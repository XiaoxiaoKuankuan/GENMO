"""Stage2-A投影、精度和冻结网络的可复现数值测试；不冒充真实物理方向证据。

比较独立SciPy凸优化解，验证活跃约束隐式导数、历史接缝梯度、零输入严格恒等及
可行提案保真。使用仓库实际SM checkpoint测试完整Goal570、冻结Actor和原六项；
精度项检查执行状态梯度及世界根XY平移不泄漏到Adapter。物理Oracle必须另行执行。
"""

from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import minimize, LinearConstraint
from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection, constrain_plan
from gem.closedloop.reference_adapter.precision import PrecisionObjective
from test_repair import constrained_system


def projector(dtype=torch.double):
    return ReferenceProjection(SimpleNamespace(joint_names=['joint']*21),
        torch.tensor([.03]*3+[.05]*3+[.1]*21, dtype=dtype), iterations=1200)


def test_projection_is_nearest_feasible_plan_and_preserves_unsaturated_proposals():
    torch.manual_seed(123)
    p = projector()
    y = torch.randn(2, 10, 27, dtype=torch.double)*.03
    z = torch.zeros(2, 27, dtype=torch.double)
    x = p(y, z, z)
    target = y[0, :, 6].numpy()
    limits = np.array([.1]*10+[.02]*10+[.004]*10)
    answer = minimize(lambda a: .5*((a-target)**2).sum(), target, jac=lambda a: a-target,
        constraints=LinearConstraint(p.matrix.numpy(), -limits, limits), method='SLSQP',
        options=dict(ftol=1e-14, maxiter=1000))
    assert answer.success
    np.testing.assert_allclose(x[0, :, 6].numpy(), answer.x, atol=2e-6)
    feasible = y*.0001
    torch.testing.assert_close(p(feasible, z, z), feasible, rtol=0, atol=1e-20)
    zero = torch.zeros_like(y, requires_grad=True)
    result = p(zero, z, z)
    assert result.count_nonzero() == 0
    torch.testing.assert_close(torch.autograd.grad(result.sum(), zero)[0], torch.ones_like(y))
    # 新投影对可执行小提案不抹掉99.75%的首帧响应。
    old = constrain_plan(feasible, z, z, .05)
    torch.testing.assert_close(old[:, 0], feasible[:, 0]*.0025)


def test_projection_backward_includes_committed_history():
    torch.manual_seed(19)
    p = projector()
    y = (torch.randn(1, 10, 27, dtype=torch.double)*.025).requires_grad_()
    # 避开数个活跃面恰好相交的不可微点；该处只有广义导数，中央差分不唯一。
    left = (torch.randn(1, 27, dtype=torch.double)*.0001).requires_grad_()
    anchor = (left.detach()+torch.randn(1, 27, dtype=torch.double)*.00005).requires_grad_()
    assert torch.autograd.gradcheck(p, (y, left, anchor), fast_mode=True, atol=3e-4, rtol=3e-3)


def test_projection_receding_seam_and_real_halo():
    torch.manual_seed(4)
    p = projector()
    left = torch.zeros(2, 27, dtype=torch.double)
    anchor = left.clone()
    for _ in range(20):
        raw = (torch.rand(2, 10, 27, dtype=torch.double)*2-1)*p.bounds
        plan = p(raw, left, anchor)
        d = torch.cat((left[:, None], anchor[:, None], plan, 2*plan[:, -1:]-plan[:, -2:-1]), 1)
        v = d.diff(dim=1)/.02
        a = v.diff(dim=1)/.02
        assert float((v.abs()-p.velocity).max()) < 2e-4
        assert float((a.abs()-p.acceleration).max()) < .01
        torch.testing.assert_close(v[:, -1], v[:, -2], atol=1e-12, rtol=0)
        left, anchor = anchor, plan[:, 0]


def test_new_projection_real_SM_zero_identity_and_execution_gradients(system):
    engine, control, z = constrained_system(system)
    control.constraint_beta = 1
    baseline = control.propose(z)
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds, iterations=200)
    bounded = control.propose(z)
    for field in ('goal', 'action', 'residual'):
        torch.testing.assert_close(baseline[field], bounded[field], atol=0, rtol=0)
    assert control.constraint_certificate(z, bounded)['feasible'].all()
    control.precision = PrecisionObjective()
    proposal = control.propose(z)
    next_state = z['state'].detach().clone().requires_grad_()
    _, _, raw, terms, *_ = control.advance(z, proposal, next_state)
    grad, = torch.autograd.grad(sum(v.sum() for k, v in terms.items() if k.startswith('precision_')), next_state)
    assert torch.isfinite(grad).all() and grad[:, 13:].norm() > 0
    old = control.runtime.reward(next_state, proposal['action'], z['action'], control.bank.ref(z['song'], z['frame']+1))[0]
    torch.testing.assert_close(terms['sm'], old, atol=0, rtol=0)
    shifted = dict(z, state=z['state'].clone())
    shifted['state'][:, :2] += 10
    torch.testing.assert_close(control.packed(control.fields(shifted)), control.packed(control.fields(z)), atol=0, rtol=0)


def test_disabled_soft_joint_limits_preserve_six_tracking_and_action_rate(system):
    _, control, z = constrained_system(system)
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds)
    proposal = control.propose(z)
    nxt = z['state'].clone()
    nxt[:, 13:34] = control.runtime.mdp.soft_hi+.01
    nxt.requires_grad_()
    enabled = control.advance(z, proposal, nxt)
    control.soft_joint_limit_cost = False
    control.weights = dict(control.weights, reference_limit=0.)
    disabled = control.advance(z, proposal, nxt)
    raw, terms = disabled[2:4]
    expected = raw['track']+.02*control.runtime.mdp.cfg.w_action_rate*raw['action_rate']
    torch.testing.assert_close(terms['sm'], expected, atol=0, rtol=0)
    assert terms['reference_limit'].count_nonzero() == 0
    torch.testing.assert_close(enabled[3]['sm']-terms['sm'],
        .02*control.runtime.mdp.cfg.w_joint_limit*raw['joint_limit'], atol=1e-8, rtol=1e-6)
    for key in ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori'):
        torch.testing.assert_close(raw[key], enabled[2][key], atol=0, rtol=0)
    actual, = torch.autograd.grad(terms['sm'].sum(), nxt, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected.sum(), nxt)
    torch.testing.assert_close(actual, expected_gradient, atol=0, rtol=0)


def test_SO3_constraints_use_true_composed_angular_acceleration():
    from supermimic.rotations import so3_exp
    p = projector()
    time = torch.arange(13, dtype=torch.double)*.02
    nominal = so3_exp(torch.stack((.2*torch.sin(time*5), .1*torch.cos(time*7), time*2), -1))[None]
    proposal = torch.zeros(1, 10, 27, dtype=torch.double)
    proposal[..., 3:6] = torch.tensor([.008, -.007, .006])
    left = torch.zeros(1, 27, dtype=torch.double)
    raw = p(proposal, left, left)
    assert not p.certificate(raw, left, left, nominal)['feasible'].all()
    corrected = p(proposal, left, left, nominal)
    report = p.certificate(corrected, left, left, nominal)
    assert report['feasible'].all(), report
    assert float(report['angular_acceleration_excess'].max()) < .0025
    # 比较执行的整个数值算法的导数，不以有限梯度本身充当方向证据。
    proposal.requires_grad_()
    value = p(proposal, left, left, nominal)
    probe = torch.linspace(.3, 1., value.numel(), dtype=torch.double).reshape_as(value)
    grad, = torch.autograd.grad((value*probe).sum(), proposal)
    direction = torch.zeros_like(proposal); direction[..., 3] = .01
    eps = 1e-4
    finite = ((p(proposal.detach()+eps*direction, left, left, nominal)
              -p(proposal.detach()-eps*direction, left, left, nominal))*probe).sum()/(2*eps)
    torch.testing.assert_close(finite, (grad*direction).sum(), atol=2e-4, rtol=.02)


def test_response_labels_do_not_cross_validation_boundary():
    from gem.closedloop.reference_adapter.reference_response import ReferenceResponseReplay
    import pytest
    replay = ReferenceResponseReplay()
    with pytest.raises(ValueError):
        replay.add(dict(origin='current_policy_real_reference_response', split='validation'))
    with pytest.raises(ValueError):
        replay.add(dict(origin='imagined', split='train'))


def test_WM_BPTT_precision_preserves_frozen_parameters(system):
    from gem.closedloop.reference_adapter.vector_learning import GroupNormalizedCritic, adapter_update
    engine, control, z = constrained_system(system, count=2)
    control.constraint_beta = 1
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds)
    control.precision = PrecisionObjective()
    critic = GroupNormalizedCritic(2835, control.spec)
    optimizer = torch.optim.Adam(control.adapter.parameters(), lr=3e-5)
    frozen = engine.step.bundle.frozen_snapshot()
    old_wm = {k: v.clone() for k, v in engine.dynamics.state_dict().items()}
    report, _ = adapter_update(control, engine.dynamics, critic, optimizer, z, 8, terminal_value=False, microbatch=2)
    assert report['execution_gradient_norm'] > 0
    assert report['gradient_precision_joint_position'] > 0
    assert all(p.grad is None for p in engine.dynamics.parameters())
    for k, v in old_wm.items():
        torch.testing.assert_close(engine.dynamics.state_dict()[k], v, atol=0, rtol=0)
    engine.step.bundle.assert_unchanged(frozen)


def test_response_replay_mixes_long_audit_and_H8_candidate_without_fake_padding():
    from gem.closedloop.reference_adapter.reference_response import ReferenceResponseReplay
    replay = ReferenceResponseReplay()
    for length in (32, 8):
        value = torch.arange(2*length*21).reshape(2, length, 21).float()
        replay.add(dict(origin='current_policy_real_reference_response', split='train', rejected={},
            state=torch.zeros(2, 55), plus_action=value, minus_action=-value,
            reliable=torch.ones(2, length, dtype=torch.bool)))
    assert replay.data['plus_action'].shape == (4, 8, 21)
    assert replay.total_pairs == 4
    assert replay.sample(32)['reliable'].shape == (32, 8)
    clone = ReferenceResponseReplay()
    clone.load_state_dict(replay.state_dict(), 'cpu')
    torch.testing.assert_close(clone.data['plus_action'], replay.data['plus_action'], atol=0, rtol=0)


def test_real_committed_SO3_seam_regression():
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent/'fixtures/so3_committed_seam.json').read_text())
    p = projector()
    raw = torch.zeros(1, 10, 27, dtype=torch.double)
    left, anchor = torch.zeros(1, 27, dtype=torch.double), torch.zeros(1, 27, dtype=torch.double)
    raw[..., 3:6] = torch.tensor(fixture['raw'], dtype=torch.double)
    left[..., 3:6] = torch.tensor(fixture['left'], dtype=torch.double)
    anchor[..., 3:6] = torch.tensor(fixture['anchor'], dtype=torch.double)
    rotation = torch.tensor(fixture['rotation'], dtype=torch.double)[None]
    projected = p(raw, left, anchor, rotation)
    certificate = p.certificate(projected, left, anchor, rotation)
    assert certificate['feasible'].all(), certificate
    assert float(certificate['angular_acceleration_excess'].max()) < .0025


def test_real_SO3_nearly_parallel_active_constraints():
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent/'fixtures/so3_nearly_parallel_constraints.json').read_text())
    for dtype in (torch.float32, torch.float64):
        p = projector(dtype)
        raw = torch.zeros(6, 10, 27, dtype=dtype)
        left = torch.zeros(6, 27, dtype=dtype)
        anchor = torch.zeros_like(left)
        raw[..., 3:6] = torch.tensor([c['raw'] for c in fixture['cases']], dtype=dtype)
        rotation = torch.tensor([c['rotation'] for c in fixture['cases']], dtype=dtype)
        result = p(raw, left, anchor, rotation)
        certificate = p.certificate(result, left, anchor, rotation)
        assert certificate['feasible'].all(), certificate


def test_collector_accounting_role_does_not_change_reset_task_split():
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    class Control:
        bank = SimpleNamespace(device='cpu')
        def propose(self, z, **kwargs):
            return dict(action=torch.zeros(2, 21))
        def constraint_certificate(self, z, proposal):
            return None
        def advance(self, z, proposal, nxt):
            done = torch.tensor([True, False])
            return z, torch.zeros(2), {}, {}, (~done).float(), done, done
    collector = VectorCollector(SimpleNamespace(N=2, step_physics=lambda a: torch.zeros(2, 55)), Control())
    collector.state = dict(state=torch.zeros(2, 55))
    resets = []
    collector.reset = lambda ids, **kwargs: resets.append((ids.tolist(), kwargs['split']))
    collector.step(zero=True, split='train_safety', reset_split='train')
    assert resets == [([0], 'train')]
    assert collector.real_by_role == {'train_safety': 2}


def test_real_SO3_nonlinear_convergence_under_two_budgets():
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent/'fixtures/so3_nonlinear_convergence.json').read_text())
    for case in fixture['cases']:
        p = ReferenceProjection(SimpleNamespace(joint_names=['joint']*21), projector().bounds,
            rotation_velocity=.3*case['budget'], rotation_acceleration=3*case['budget'])
        raw = torch.zeros(1, 10, 27, dtype=torch.double)
        left, anchor = torch.zeros(1, 27, dtype=torch.double), torch.zeros(1, 27, dtype=torch.double)
        raw[..., 3:6] = torch.tensor(case['raw'], dtype=torch.double)
        left[..., 3:6] = torch.tensor(case['left'], dtype=torch.double)
        anchor[..., 3:6] = torch.tensor(case['anchor'], dtype=torch.double)
        rotation = torch.tensor(case['rotation'], dtype=torch.double)[None]
        result = p(raw, left, anchor, rotation)
        certificate = p.certificate(result, left, anchor, rotation)
        assert certificate['feasible'].all(), certificate


def test_real_affine_and_IPM_convergence_with_actual_RobotSpec(assets):
    import json
    from pathlib import Path
    from gem.closedloop.reference_adapter.policy import ReferenceAdapter
    bundle, _ = assets
    for name in ('failed_affine_QP_tolerance', 'failed_SO3_IPM_gpu', 'failed_SO3_final_merit', 'failed_affine_conservative'):
        case = json.loads((Path(__file__).parent/'fixtures'/f'{name}.json').read_text())
        p = ReferenceProjection(bundle.spec, ReferenceAdapter(bundle.spec).bounds, **case.get('projection', {}))
        raw, left, anchor, rotation = (torch.tensor(case[k], dtype=torch.float32) for k in ('raw', 'left', 'anchor', 'rotation'))
        result = p(raw, left, anchor, rotation)
        certificate = p.certificate(result, left, anchor, rotation)
        assert certificate['feasible'].all(), certificate
