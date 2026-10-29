"""无硬投影根姿态补偿的数值、梯度与奖励一致性回归。

验证直接270维有界输出不会被软成本改写；以真实冻结SuperMimic权重检查零策略
Goal/动作恒等、非零参考到Actor/WM的梯度，以及原六项Tracking不变。SO(3)软成本
单独检查名义旋转抵消、同绝对帧重叠、当前接缝/halo和历史梯度的有限差分。
这些CPU回归不是实际物理收益证明；正式训练必须另存零Adapter配对物理评价。
"""

import torch

from supermimic.rotations import so3_exp
from gem.closedloop.reference_adapter.application import direct_reference_config
from gem.closedloop.reference_adapter.precision import PrecisionObjective, root_reference_costs
from test_direct_reference import direct_system


def test_soft_root_gradients_include_history_and_nominal():
    torch.manual_seed(73)
    residual = (torch.randn(2, 10, 27, dtype=torch.double)*.002).requires_grad_()
    anchor = (torch.randn(2, 27, dtype=torch.double)*.002).requires_grad_()
    old = (torch.randn(2, 10, 27, dtype=torch.double)*.002).requires_grad_()
    nominal = (torch.randn(2, 13, 3, dtype=torch.double)*.05).requires_grad_()
    def cost(r, a, p, n):
        values = root_reference_costs(r, a, p, torch.ones(2, dtype=torch.bool), so3_exp(n))
        return torch.stack(tuple(values.values()))
    assert torch.autograd.gradcheck(cost, (residual, anchor, old, nominal), fast_mode=True, atol=2e-5, rtol=2e-3)


def test_zero_and_constant_rotation_dont_penalize_nominal_dance():
    torch.manual_seed(5)
    n = so3_exp(torch.randn(2, 13, 3, dtype=torch.double)*.15)
    zero = torch.zeros(2, 10, 27, dtype=torch.double, requires_grad=True)
    a = torch.zeros(2, 27, dtype=torch.double)
    values = root_reference_costs(zero, a, zero, torch.ones(2, dtype=torch.bool), n)
    assert sum(values.values()).abs().max() < 1e-24
    assert torch.autograd.grad(sum(v.sum() for v in values.values()), zero)[0].isfinite().all()
    constant = zero.detach().clone()
    constant[..., 3:6] = torch.tensor([.01, -.02, .01])
    a[:, 3:6] = constant[:, 0, 3:6]
    values = root_reference_costs(constant, a, constant, torch.ones(2, dtype=torch.bool), n)
    assert values['root_reference_velocity'].max() < 1e-24


def test_overlap_uses_matching_times_and_seam_and_halo_have_gradients():
    old = torch.randn(1, 10, 27, dtype=torch.double)*.001
    r = torch.cat((old[:, 1:], old[:, -1:]+.001), 1).requires_grad_()
    anchor = old[:, 0].clone().requires_grad_()
    nominal = torch.eye(3, dtype=torch.double).expand(1, 13, 3, 3)
    values = root_reference_costs(r, anchor, old, torch.ones(1, dtype=torch.bool), nominal)
    assert values['root_reference_continuity'].abs().max() < 1e-24
    gr, ga = torch.autograd.grad(values['root_reference_velocity'].sum(), (r, anchor))
    assert gr[:, 0, 3:6].norm() > 0 and gr[:, -1, 3:6].norm() > 0 and ga[:, 3:6].norm() > 0
    values = root_reference_costs(r, anchor, old, torch.zeros(1, dtype=torch.bool), nominal)
    assert values['root_reference_continuity'].item() == 0


def test_soft_cost_does_not_project_or_change_SM_and_has_dt_once(system, monkeypatch):
    engine, _, control, z = direct_system(system)
    control.adapter.disable_root_xy = True
    zero = control.propose(z)
    control.precision = PrecisionObjective(root_velocity_weight=.002, root_continuity_weight=.001)
    same = control.propose(z)
    for key in ('goal', 'action', 'residual'):
        torch.testing.assert_close(zero[key], same[key], atol=0, rtol=0)
    from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
    def forbidden(*a, **kw):
        raise AssertionError('直接参考不能调用硬投影')
    monkeypatch.setattr(ReferenceProjection, 'forward', forbidden)
    with torch.no_grad():
        control.adapter.net[-1].bias.normal_(std=.1)
    frozen = control.bundle.frozen_snapshot()
    with engine.dynamics.adapter_phase():
        proposal = control.propose(z)
        torch.testing.assert_close(proposal['raw_residual'], proposal['residual'], atol=0, rtol=0)
        nxt = engine.dynamics.step(z['state'], proposal['action'], z['xi'])
        _, reward, raw, terms, *_ = control.advance(z, proposal, nxt)
        torch.testing.assert_close(terms['root_reference_velocity'], -.02*.002*raw['root_reference_velocity'])
        grad, = torch.autograd.grad(reward.sum(), control.adapter.net[-1].bias)
        assert grad.isfinite().all() and grad.norm() > 0
        control.precision = PrecisionObjective()
        _, _, off_raw, off_terms, *_ = control.advance(z, proposal, nxt)
        torch.testing.assert_close(off_terms['sm'], terms['sm'], atol=0, rtol=0)
        for name in ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori'):
            torch.testing.assert_close(raw[name], off_raw[name], atol=0, rtol=0)
    control.bundle.assert_unchanged(frozen)


def test_explicit_soft_fields_survive_config_without_restoring_old_hard_fields():
    cfg = direct_reference_config(dict(alternating=dict(projection={'rotation_acceleration': 3.},
        precision=dict(weight=.4, root_velocity_weight=.002, root_continuity_weight=.001,
                       root_velocity_scale=.3, root_continuity_scale=.05, acceleration_weight=100.))))
    assert 'projection' not in cfg['alternating']
    assert 'acceleration_weight' not in cfg['alternating']['precision']
    assert cfg['alternating']['precision']['root_velocity_weight'] == .002
    assert cfg['alternating']['objective_version'] == 'sm_music_precision_soft_root_v8'
