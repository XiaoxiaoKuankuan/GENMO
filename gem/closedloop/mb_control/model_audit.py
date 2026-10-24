"""冻结WM的真实闭环预测、候选排序与物理梯度方向审计。

每个候选从显式reset重新执行同样输入/噪声/外生延迟，不靠写回中途qpos冒充求解器
恢复。已知参考与Stage1生成参考分别报告；平滑关节微扰比较模型导数、真实差分和
优化后真实收益，另做同reset重复性检查。审计物理生成可离线暂停，绝不作实时性能证据。
"""

from dataclasses import replace
import itertools

import torch

from supermimic.mdp import flatten_state, unflatten_state
from supermimic.rotations import so3_log
from .returns import actor_return
from .execution_audit import state_probe
from .validation import state_envelope


def state_errors(predicted, real, kin=None):
    p, R, v, w, q, qd = unflatten_state(predicted)
    rp, rR, rv, rw, rq, rqd = unflatten_state(real)
    result = {"root_xy_m": (p[..., :2] - rp[..., :2]).square().mean().sqrt(),
            "root_height_m": (p[..., 2] - rp[..., 2]).square().mean().sqrt(),
            "root_orientation_rad": so3_log(rR.transpose(-1, -2) @ R).square().sum(-1).mean().sqrt(),
            "root_linear_velocity_m_s": (v - rv).square().mean().sqrt(),
            "root_angular_velocity_rad_s": (w - rw).square().mean().sqrt(),
            "joint_position_rad": (q - rq).square().mean().sqrt(),
            "joint_velocity_rad_s": (qd - rqd).square().mean().sqrt()}
    if kin is not None:
        dev = kin.axes.device
        predicted_bodies = kin.forward(*(x.to(dev) for x in unflatten_state(predicted)), all_bodies=True)[0]
        real_bodies = kin.forward(*(x.to(dev) for x in unflatten_state(real)), all_bodies=True)[0]
        feet = [kin.body_names.index(n) for n in ("l_ankle_roll_link", "r_ankle_roll_link")]
        result["foot_position_m"] = (predicted_bodies[:, feet] - real_bodies[:, feet]).square().mean().sqrt()
        result["foot_height_m"] = (predicted_bodies[:, feet, 2] - real_bodies[:, feet, 2]).square().mean().sqrt()
    return result


class PerturbedGenerator:
    def __init__(self, actor, joint_index, amplitude):
        self.actor, self.joint_index, self.amplitude = actor, joint_index, amplitude

    def generate(self, state, ticket, context, noise):
        plan = self.actor.generate(state, ticket, context, noise)
        frame = torch.arange(60, device=plan.world_qpos.device).to(plan.world_qpos)
        phase = ((frame - ticket.prefix_frames) / (59 - ticket.prefix_frames)).clamp(0, 1)
        envelope = (torch.sin(torch.pi * phase) ** 2)[None, :, None]
        coordinate = torch.nn.functional.one_hot(torch.as_tensor(7 + self.joint_index, device=frame.device), 28).to(plan.world_qpos)
        return replace(plan, world_qpos=plan.world_qpos + self.amplitude * envelope * coordinate)


def audit_generated_reference(engine, collector, music, initial, *, noise_seed, latencies,
                              joint_indices, epsilon=.01, criteria=None):
    """返回原始证据；只有显式阈值满足且多次方向支持才填supported，不默认判好。"""
    base_actor, old_limit = engine.actor, engine.max_model_steps
    engine.max_model_steps = None  # 这是测量所需时域的审计，不是越界训练。
    generator = torch.Generator(device=initial.device).manual_seed(noise_seed)
    noises = [torch.randn(1, 60, 30, device=initial.device, generator=generator) for _ in range(engine.decisions)]

    def run(physical, joint, amplitude):
        z = collector.reset(music, initial)
        engine.actor = PerturbedGenerator(base_actor, joint, amplitude)
        trace, rewards = [], []
        def record(state, reward):
            trace.append(state.state.detach().cpu())
            rewards.append(float(reward.detach().mean()))
        need_gradient = isinstance(amplitude, torch.Tensor) and amplitude.requires_grad
        with torch.set_grad_enabled(not physical and need_gradient):
            intervals = engine.rollout(z, noises, latencies, transition_fn=collector.env.step_physics if physical else None,
                                       on_step=record)
            objective = actor_return(intervals, torch.zeros_like(intervals[-1].reward)).mean()
        return objective, torch.cat(trace) if trace else torch.empty(0, 55), intervals, rewards

    try:
        model_base, model_trace, model_intervals, _ = run(False, joint_indices[0], 0.)
        physical_base, real_trace, real_intervals, real_rewards = run(True, joint_indices[0], 0.)
        repeated, repeated_trace, _, _ = run(True, joint_indices[0], 0.)
        length = min(len(model_trace), len(real_trace))
        horizons = {}
        dependency_steps = max((max(x.event.get("longest_model_dependency_steps", 0),
                                    x.event.get("maximum_forecast_dependency_steps", 0)) for x in model_intervals), default=0)
        requested_horizons = criteria.get("horizons", [5, 10, 15, 20, 32]) if criteria else [5, 10, 15, 20, 32]
        for horizon in sorted(set([*requested_horizons, len(model_trace), dependency_steps])):
            horizons[str(horizon)] = ({k: float(v) for k, v in state_errors(model_trace[:horizon], real_trace[:horizon], collector.adapter.runtime.kin).items()}
                                      if horizon <= length else {"status": "insufficient_physical_horizon"})
        candidates = []
        for joint in joint_indices:
            amplitude = torch.zeros((), device=initial.device, requires_grad=True)
            objective, _, probe_intervals, _ = run(False, joint, amplitude)
            gradient, = torch.autograd.grad(objective, amplitude, retain_graph=True)
            execution_gradient, = torch.autograd.grad(state_probe(probe_intervals[-1].end.state), amplitude)
            changes = []
            for amount in (-epsilon, epsilon, epsilon * float(gradient.sign())):
                mr, _, _, _ = run(False, joint, amount)
                pr, _, intervals, _ = run(True, joint, amount)
                changes.append({"amplitude_rad": amount, "model_return": float(mr.detach()),
                                "physical_return": float(pr), "physical_end": intervals[-1].reason,
                                "physical_control_steps": sum(x.steps for x in intervals),
                                "physical_state_probe": float(state_probe(intervals[-1].end.state))})
            physical_slope = (changes[1]["physical_return"] - changes[0]["physical_return"]) / (2 * epsilon)
            state_slope = (changes[1]["physical_state_probe"] - changes[0]["physical_state_probe"]) / (2 * epsilon)
            equal_horizons = all(c["physical_control_steps"] == len(model_trace) for c in changes[:2])
            candidates.append({"joint_index": joint, "model_autograd": float(gradient),
                               "physical_centered_difference": physical_slope,
                               "direction_agrees": bool(gradient * physical_slope > 0), "candidates": changes,
                               "state_only_model_autograd": float(execution_gradient),
                               "state_only_physical_difference": state_slope,
                               "state_only_direction_agrees": equal_horizons and bool(execution_gradient * state_slope > 0),
                               "optimized_physical_delta": changes[2]["physical_return"] - float(physical_base)})
        repeat_n = min(len(real_trace), len(repeated_trace))
        repeat_error = float((real_trace[:repeat_n] - repeated_trace[:repeat_n]).abs().max()) if repeat_n else None
        prediction_passed = criteria is not None and all(
            "status" not in values and all(values[name] <= bound for name, bound in criteria["state_rmse_limits"].items())
            for values in horizons.values())
        direction_fraction = sum(c["direction_agrees"] for c in candidates) / len(candidates)
        state_direction_fraction = sum(c["state_only_direction_agrees"] for c in candidates) / len(candidates)
        direction_passed = (criteria is not None and len(candidates) >= criteria["minimum_directions"] and
                            direction_fraction >= criteria["minimum_direction_fraction"] and
                            state_direction_fraction >= criteria.get("minimum_state_direction_fraction", 2/3) and
                            sum(c["optimized_physical_delta"] for c in candidates) / len(candidates) >= criteria["minimum_optimized_delta"])
        repeat_passed = criteria is not None and repeat_error is not None and repeat_error <= criteria["reset_replay_max_abs"] and len(real_trace) == len(repeated_trace)
        paired = list(itertools.chain.from_iterable(c["candidates"][:2] for c in candidates))
        rank_pairs = [(a, b) for a, b in itertools.combinations(paired, 2)
                      if a["model_return"] != b["model_return"] and a["physical_return"] != b["physical_return"]]
        rank_agreement = (sum((a["model_return"] - b["model_return"]) *
                              (a["physical_return"] - b["physical_return"]) > 0 for a, b in rank_pairs) / len(rank_pairs)) if rank_pairs else None
        return {"evidence": "offline_reset_and_replay_real_physics_not_realtime", "horizons": horizons,
                "model_base_return": float(model_base.detach()), "physical_base_return": float(physical_base),
                "physical_repeat_return": float(repeated), "physical_reward_dt": real_rewards,
                "physical_end": real_intervals[-1].reason, "model_end": model_intervals[-1].reason,
                "main_control_steps": len(model_trace), "required_dependency_steps": dependency_steps,
                "real_state_envelope": state_envelope(torch.cat((initial.detach().cpu(), real_trace))),
                "forecast_branches": model_intervals[-1].event.get("forecast_branches", []),
                "candidates": candidates, "direction_fraction": direction_fraction,
                "state_only_direction_fraction": state_direction_fraction,
                "candidate_pairwise_rank_agreement": rank_agreement, "reset_repeat_max_abs": repeat_error,
                "prediction_passed": bool(prediction_passed), "physical_direction_passed": bool(direction_passed),
                "reset_replay_repeatability_passed": bool(repeat_passed),
                "supported_control_steps": length if prediction_passed and direction_passed and repeat_passed else 0,
                "criteria": criteria, "source_hash_gate": False,
                "contact_force_or_solver_state": "state55不包含接触力或求解器状态，不伪造该项预测"}
    finally:
        engine.actor, engine.max_model_steps = base_actor, old_limit


def audit_known_reference(collector, music, initial, reference, horizons=(5, 10, 15, 20, 32), criteria=None):
    """分离固定真实动作的动力学与固定参考下SM闭环；均从真实参考首状态显式复位。"""
    horizons = sorted(set(horizons))
    z = collector.reset(music, initial, initial_reference=reference)
    ref0, _, _ = z.reference.read_current_future(0)
    matched_initial = flatten_state((ref0.root_p, ref0.root_R, ref0.root_v, ref0.root_w, ref0.q, ref0.qd))
    traces, real_actions = {}, []
    with torch.no_grad():
        for regime in ("real", "fixed_real_actions", "fixed_reference_closed_loop"):
            z = collector.reset(music, matched_initial, initial_reference=reference)
            trace = []
            for step in range(max(horizons)):
                if regime == "fixed_real_actions" and step >= len(real_actions):
                    break
                z = collector.adapter.refresh_current_observation(z)
                _, _, goal = z.reference.read_current_future(z.tick)
                a = real_actions[step] if regime == "fixed_real_actions" else collector.adapter.bundle.act(z.sm_history, goal)
                state = (collector.env.step_physics(a) if regime == "real" else
                         collector.adapter.bundle.step_model(z.state, a, z.xi))
                z = collector.adapter.advance(z, state, a, origin="real" if regime == "real" else "imagined")
                trace.append(state.cpu())
                if regime == "real":
                    real_actions.append(a.detach().clone())
                    current, _, _ = z.reference.read_current_future(z.tick)
                    if bool(collector.adapter.runtime.termination(state, current).any()):
                        break
            traces[regime] = torch.cat(trace)
    result = {}
    for regime in ("fixed_real_actions", "fixed_reference_closed_loop"):
        prediction, real = traces[regime], traces["real"]
        errors = {}
        for horizon in horizons:
            if min(len(prediction), len(real)) < horizon:
                errors[str(horizon)] = {"status": "insufficient_physical_horizon"}
            else:
                errors[str(horizon)] = {k: float(v) for k, v in state_errors(prediction[:horizon], real[:horizon], collector.adapter.runtime.kin).items()}
        result[regime] = errors
    passed = criteria is not None and all("status" not in value and all(value[k] <= bound for k, bound in criteria["state_rmse_limits"].items())
                                         for regime in result.values() for value in regime.values())
    return {"regimes": result, "passed": bool(passed), "horizons": horizons,
            "real_state_envelope": state_envelope(traces["real"]),
            "real_control_steps": len(traces["real"]), "initialization": "reference_first_full_state_no_future_real_state"}
