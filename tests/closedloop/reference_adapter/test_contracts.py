"""270输出、SO(3)、十帧Goal、名义只读与可观测输入的合同验收。

真实权重/FK用于接口parity，合成名义参考用于覆盖所有坐标与时间边界；这些测试不
执行物理。零Adapter对齐比较全部ReferenceBatch字段、Actor输出和原SM奖励，而非
仅比较位置或形状。过去/当前不可回写，未来允许变化。
"""

from dataclasses import replace

import pytest
import torch

from supermimic.bridge.reference import build_reference, current_future
from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_exp, so3_log
from gem.closedloop.reference_adapter.reference import reconstruct, nominal_window, overlap_cost
from gem.closedloop.reference_adapter.features import adapter_fields


def test_structure_zero_parity_all_reference_fields(system):
    engine, z = system
    adapter, bundle, runtime = engine.step.adapter, engine.step.bundle, engine.step.runtime
    linear = [m for m in adapter.modules() if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in linear] == [512, 512, 256, 270]
    nominal_before = z.nominal_reference.world_qpos.clone()
    for _ in range(3):
        proposal = engine.step.propose(z)
        original = build_reference(z.nominal_reference.pose_at, z.tick+torch.arange(11)*12,
                                   z.nominal_reference.lower, z.nominal_reference.upper, runtime.kin)
        now, future, goal = current_future(original)
        for a, b in zip(proposal.corrected_reference.current, now):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        for a, b in zip(proposal.corrected_reference.future, future):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(proposal.corrected_reference.goal, goal, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(proposal.action, bundle.act(z.control_history, goal), atol=3e-5, rtol=3e-5)
        s1 = engine.dynamics.step(z.executed_state, proposal.action, z.xi)
        row = engine.step.advance(z, proposal, s1, origin="imagined")
        expected, _ = runtime.reward(s1, proposal.action, z.previous_action, proposal.nominal_next)
        torch.testing.assert_close(row.weighted["sm"], expected, atol=0, rtol=0)
        terms = row.raw
        track = sum(terms[n] for n in ("body_pos", "body_ori", "body_linvel", "body_angvel"))+.5*(terms["root_pos"]+terms["root_ori"])
        torch.testing.assert_close(track*.02, terms["track"])
        z = row.end
    assert torch.equal(nominal_before, z.nominal_reference.world_qpos)


def test_ten_independent_frames_halo_and_gradients(system):
    engine, z = system
    d = (torch.randn(1, 10, 27)*.001).requires_grad_()
    corrected = reconstruct(z, d, engine.step.runtime.kin)
    assert corrected.goal.shape == (1, 570)
    assert corrected.plan.ticks.tolist() == list(range(12, 121, 12))
    torch.testing.assert_close(corrected.halo_residual, 2*d[:, -1]-d[:, -2])
    assert torch.equal(corrected.current.root_p, z.current_pose.p)
    gradient, = torch.autograd.grad((corrected.goal*torch.randn_like(corrected.goal)).sum(), d)
    assert torch.isfinite(gradient).all() and (gradient.abs().sum(-1) > 0).all()
    from gem.closedloop.mb_control.contracts import ReferenceExhausted
    with pytest.raises(ReferenceExhausted):
        reconstruct(replace(z, tick=z.nominal_reference.upper-120), d, engine.step.runtime.kin)


def test_overlap_absolute_tick_and_committed_anchor(system):
    engine, z = system
    d = torch.arange(10)[None, :, None].expand(1, 10, 27)*.001
    old = reconstruct(z, d, engine.step.runtime.kin)
    z1 = replace(z, tick=12, previous_plan=old.plan, left_pose=z.current_pose,
                 current_pose=old.plan.corrected_pose.select((slice(None), 0)))
    new_d = torch.cat((d[:, 1:], d[:, -1:]), 1)
    new = reconstruct(z1, new_d, engine.step.runtime.kin)
    assert overlap_cost(new.plan, old.plan, engine.step.adapter.bounds, z.heading).item() == 0
    assert torch.equal(new.current.q, old.future.q[:, 0])
    changed = reconstruct(z1, new_d+.001, engine.step.runtime.kin)
    assert torch.equal(changed.current.q, old.future.q[:, 0])
    assert not torch.equal(changed.future.q, new.future.q)
    # 基准版本不同但修正物理姿态相同，必须比较物理姿态，不能盲比残差。
    same_pose = replace(new.plan, residual=new.plan.residual+.3, nominal_version=123)
    assert overlap_cost(same_pose, old.plan, engine.step.adapter.bounds, z.heading).item() < 1e-8


def test_input_privilege_exclusion_and_history_semantics(system):
    engine, z = system
    current, _, goal = nominal_window(z.nominal_reference, z.tick, engine.step.runtime.kin)
    fields = adapter_fields(z, engine.step.adapter, engine.step.bundle, current, goal)
    changed = z.executed_state.clone()
    changed[:, :3] += 100
    changed[:, 7:10] += 50
    other = adapter_fields(replace(z, executed_state=changed, xi=z.xi+10), engine.step.adapter, engine.step.bundle, current, goal)
    for name in fields:
        assert torch.equal(fields[name], other[name]), name
    with pytest.raises(ValueError):
        engine.step.adapter(fields | {"xi": z.xi})
    engine.step.adapter.net[-1].bias.data[6] = .1
    proposal = engine.step.propose(z)
    row = engine.step.advance(z, proposal, engine.dynamics.step(z.executed_state, proposal.action, z.xi), origin="imagined")
    assert torch.equal(row.end.control_history[:, :-1], z.control_history[:, 1:])
    expected = engine.step.runtime.observation(row.end.executed_state, row.action, z.xi,
        __import__('supermimic.motion', fromlist=['select_reference']).select_reference(proposal.corrected_reference.future, (slice(None), 0)))
    torch.testing.assert_close(row.end.control_history[:, -1], expected)


def test_so3_zero_and_small_rotation_gradcheck():
    for value in (torch.zeros(2, 3, dtype=torch.double), torch.randn(2, 3, dtype=torch.double)*.02):
        value.requires_grad_()
        assert torch.autograd.gradcheck(lambda x: so3_log(so3_exp(x)), value, eps=1e-6, atol=2e-5, rtol=2e-3)
        torch.testing.assert_close(so3_exp(value).transpose(-1, -2) @ so3_exp(value), torch.eye(3, dtype=value.dtype).expand(2, 3, 3), atol=1e-9, rtol=1e-9)
