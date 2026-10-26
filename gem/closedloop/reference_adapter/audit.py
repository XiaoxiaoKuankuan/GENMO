"""Adapter当前分布的数值与真实物理配对审计。

数值审计使用原SM权重检查零参考、执行项/music-only输入导数；物理审计显式reset后
从同一初态重放，每个H分别跑8/16步及正负扰动，核对重复性、逐分量模型误差和方向。
接触分支变化或差分低于噪声底时标记不确定，不能视为通过；历史H16报告不被导入。
"""

from dataclasses import replace
import hashlib

import torch

from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_log
from .dynamics import frozen_parameters
from .reference import nominal_window, reconstruct


def tensor_identity(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def numerical_audit(engine, z):
    bundle, runtime = engine.step.bundle, engine.step.runtime
    before = bundle.frozen_snapshot()
    nominal = nominal_window(z.nominal_reference, z.tick, runtime.kin)
    corrected = reconstruct(z, z.executed_state.new_zeros(1, 10, 27), runtime.kin)
    errors = {"current_"+name: float((a-b).abs().max()) for name, a, b in zip(nominal[0]._fields, nominal[0], corrected.current)}
    errors.update({"future_"+name: float((a-b).abs().max()) for name, a, b in zip(nominal[1]._fields, nominal[1], corrected.future)})
    errors["goal"] = float((nominal[2]-corrected.goal).abs().max())
    errors["actor"] = float((bundle.act(z.control_history, nominal[2])-bundle.act(z.control_history, corrected.goal)).abs().max())
    results = []
    old_horizon = engine.bptt_steps
    try:
        for horizon in (8, 16):
            engine.bptt_steps = horizon
            with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
                trajectory = engine.rollout(z)
                loss = trajectory.rows[-1].end.executed_state[:, 34:].square().mean()
                bias = engine.step.adapter.net[-1].bias
                state_grad, = torch.autograd.grad(loss, bias, retain_graph=True)
                music = sum(r.raw["music"].sum() for r in trajectory.rows)
                music_grad, = torch.autograd.grad(music, bias, allow_unused=True)
                results.append(dict(horizon=horizon, executed_model_steps=len(trajectory.rows), boundary=trajectory.boundary,
                                    execution_gradient=float(state_grad.norm()), execution_finite=bool(torch.isfinite(state_grad).all()),
                                    music_gradient=float(music_grad.norm()) if music_grad is not None else 0.,
                                    music_valid_centers=sum(int(r.raw["music_valid"].sum()) for r in trajectory.rows)))
    finally:
        engine.bptt_steps = old_horizon
    bundle.assert_unchanged(before)
    return dict(evidence="actual_weights_numerical_not_physics", zero_max_abs=errors, horizons=results,
                frozen_unchanged=True, physical_quality_passed=False)


def _state_error(a, b):
    A, B = unflatten_state(a), unflatten_state(b)
    return dict(root_position_m=float((A[0]-B[0]).norm(dim=-1).max()),
                root_rotation_rad=float(so3_log(A[1].transpose(-1, -2) @ B[1]).norm(dim=-1).max()),
                joint_position_rad=float((A[4]-B[4]).square().mean().sqrt()),
                joint_velocity_rad_s=float((A[5]-B[5]).square().mean().sqrt()))


def physical_audit(collector, engine, initial, nominal, audio, metadata, criteria, *, start_steps=0):
    required = {"repeatability_max_abs", "root_position_m", "root_rotation_rad", "joint_position_rad",
                "joint_velocity_rad_s", "direction_noise_floor", "residual_epsilon", "action_epsilon"}
    if not required <= set(criteria) or any(criteria[k] <= 0 for k in required):
        raise ValueError("物理验收需要事前误差/噪声及物理残差和动作扰动尺度")
    old_horizon, results = engine.bptt_steps, []
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def collect(horizon, intervention=None):
        with torch.no_grad():
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            collector.reset(initial, nominal, audio, **metadata)
            if start_steps:
                prefix = collector.collect(start_steps)
                if len(prefix) != start_steps or collector.terminal:
                    return [], []
            rows, contacts = [], []
            for i in range(horizon):
                part = collector.collect(1, intervention=intervention if i == 0 else None)
                if not part:
                    break
                rows.extend(part)
                contacts.append(collector.env.contact_events())
                if collector.terminal:
                    break
            return rows, contacts

    def task_return(rows):
        return sum(engine.gamma**i*(r.weighted["sm"]+r.weighted["music"]) for i, r in enumerate(rows)).sum()

    try:
        for horizon in (8, 16):
            base, contacts = collect(horizon)
            repeat, repeated_contacts = collect(horizon)
            if len(base) != horizon or len(repeat) != horizon:
                results.append(dict(horizon=horizon, passed=False, reason="real_episode_ended_before_horizon"))
                continue
            initial_difference = float((base[0].start.executed_state-repeat[0].start.executed_state).abs().max())
            repeat_error = max(float((a.end.executed_state-b.end.executed_state).abs().max()) for a, b in zip(base, repeat))
            signature = lambda cs: [c["pair_signature"] for c in cs]
            repeat_ok = max(repeat_error, initial_difference) <= criteria["repeatability_max_abs"] and signature(contacts) == signature(repeated_contacts)
            noise = max(criteria["direction_noise_floor"], abs(float(task_return(base)-task_return(repeat)))*5)
            with torch.no_grad():
                state, predictions = base[0].start.executed_state, []
                for row in base:
                    state = engine.dynamics.step(state, row.action, row.start.xi)
                    predictions.append(state)
                errors = _state_error(torch.cat(predictions), torch.cat([r.end.executed_state for r in base]))
            engine.bptt_steps = horizon
            directions = []
            # 直接扰动本次部署的十帧残差和第一步Actor控制，覆盖两个执行接口。
            # 不局限最后bias子空间，另外报告全部Adapter参数的任务梯度。
            for kind, shape, epsilon in (("residual_offset", (1, 10, 27), criteria["residual_epsilon"]),
                                         ("action_offset", (1, 21), criteria["action_epsilon"])):
                offset = initial.new_zeros(shape, requires_grad=True)
                with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
                    trajectory = engine.rollout(base[0].start, intervention={kind: offset})
                    value = task_return(trajectory.rows)
                    gradient, = torch.autograd.grad(value, offset, retain_graph=True)
                    parameters = tuple(engine.step.adapter.parameters())
                    pg = torch.autograd.grad(value, parameters, allow_unused=True)
                    parameter_norm = float(sum((g.square().sum() for g in pg if g is not None), initial.new_zeros(())).sqrt())
                norm = gradient.norm()
                if not torch.isfinite(gradient).all() or norm <= 1e-12:
                    directions.append(dict(kind=kind, passed=False, reason="zero_or_nonfinite_execution_gradient"))
                    continue
                direction = gradient/norm
                plus, cp = collect(horizon, {kind: epsilon*direction})
                minus, cm = collect(horizon, {kind: -epsilon*direction})
                if len(plus) != horizon or len(minus) != horizon:
                    directions.append(dict(kind=kind, passed=False, reason="perturbation_caused_real_terminal"))
                    continue
                delta = float(task_return(plus)-task_return(minus))
                contact_change = signature(cp) != signature(cm)
                action_rms = float((plus[0].action-minus[0].action).square().mean().sqrt())
                residual_delta = plus[0].end.previous_plan.residual-minus[0].end.previous_plan.residual
                direction_ok = delta > noise and not contact_change and len(trajectory.rows) == horizon and action_rms > 0
                directions.append(dict(kind=kind, passed=direction_ok, physical_central_derivative=delta/(2*epsilon),
                    model_directional_derivative=float(norm), all_adapter_parameter_gradient_norm=parameter_norm,
                    physical_return_difference=delta, noise_floor=noise, signal_to_noise=abs(delta)/noise,
                    contact_branch_changed=contact_change, action_rms_difference=action_rms,
                    residual_root_rms=float(residual_delta[..., :6].square().mean().sqrt()),
                    residual_joint_rms=float(residual_delta[..., 6:].square().mean().sqrt()),
                    direction_status="uncertain_contact_change" if contact_change else "uncertain_below_noise" if abs(delta) <= noise else "positive" if delta > 0 else "negative"))
            results.append(dict(horizon=horizon, passed=repeat_ok and all(errors[k] <= criteria[k] for k in errors)
                                and all(d["passed"] for d in directions),
                                initial_max_abs=initial_difference, repeatability_max_abs=repeat_error,
                                model_errors=errors, directions=directions, base_return=float(task_return(base))))
    finally:
        engine.bptt_steps = old_horizon
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        collector.terminal = True
    return dict(passed=bool(results) and all(r["passed"] for r in results), cases=results, criteria=criteria, start_steps=start_steps,
                adapter_tensor_sha256=tensor_identity(engine.step.adapter), dynamics_tensor_sha256=tensor_identity(engine.dynamics),
                model_version=engine.dynamics.model_version, evidence="current_adapter_distribution_real_reset_paired_8_then_16")
