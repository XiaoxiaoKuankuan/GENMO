"""真实首次接管状态的短时WM物理方向审计，输出可追溯的原始反事实配对证据。

先在Newton中以当前GENMO真实异步滚动，保存请求与首次消费前状态；每次反事实从
episode初态重置，以原已提交参考驱动同一SM反馈闭环，恢复接触/执行器前史，
核对anchor恢复误差，不写回中途state55或以固定动作长时重放冒充相同anchor。
在同一真实anchor比较H=8/16/24/32、根/腿/臂多尺度扰动的模型导数和真实差分。
真实接触对与重放噪声单独记录；低于噪声的方向标不可判定。原请求ForecastContext
另重放旧参考分支检验，不能用这些未来实测状态作为GENMO条件。各时域独立验收，
固定动作93步压力测试没有在这里被删改，也不用于替代短时闭环实证。
"""

from dataclasses import replace
import json
from pathlib import Path

import torch

from .application import task_inputs, seed_all, task_identity
from .checkpoint import pack_value
from .imagination import PhysicalAnchorImagination
from .audit_metrics import trajectory_metrics
from .forecast import forecast_rollout
from .returns import actor_return
from .validation import state_envelope, combine_envelopes


def contact_signature(event):
    return tuple(tuple(pair) for pair in event["pair_signature"])


def perturb_plan(plan, coordinate, amplitude):
    frame = torch.arange(60, device=plan.world_qpos.device).to(plan.world_qpos)
    # 保护前缀严格零扰动，之后六个源帧平滑接入；全样本使用预声明同一形状。
    phase = ((frame-plan.ticket.prefix_frames)/6).clamp(0, 1)
    envelope = (.5-.5*torch.cos(torch.pi*phase))[None, :, None]
    vector = torch.nn.functional.one_hot(frame.new_tensor(coordinate, dtype=torch.long), 28).to(frame)
    return replace(plan, world_qpos=plan.world_qpos+amplitude*envelope*vector)


def phase_labels(sample, event, previous_event, adapter):
    from supermimic.mdp import unflatten_state
    s = sample.state
    _, _, velocity, angular, _, joint_velocity = unflatten_state(s.state)
    left, right = event["left_foot"], event["right_foot"]
    labels = []
    if left and right:
        labels.append("double_support")
    if left != right:
        labels.append("single_support")
        if float(velocity[:, :2].norm()) > .08 or float(joint_velocity[:, :12].abs().max()) > .5:
            labels.append("stepping")
    if previous_event is not None and ((left and not previous_event["left_foot"]) or (right and not previous_event["right_foot"])):
        labels.append("landing")
    if float(angular[:, 2].abs().max()) > .2:
        labels.append("turning")
    arm_indices = [i for i, n in enumerate(adapter.bundle.spec.joint_names) if "arm" in n or "elbow" in n]
    if float(joint_velocity[:, arm_indices].abs().max()) > .7:
        labels.append("active_upper_body")
    current, _, _ = s.reference.read_current_future(s.tick)
    if float((s.state[:, :3]-current.root_p).norm()) > .03 or float((s.state[:, 13:34]-current.q).square().mean().sqrt()) > .04:
        labels.append("tracking_deviation")
    return labels


def audit_short(collector, tasks, output, cfg, criteria, *, device):
    output = Path(output)
    horizons = criteria["horizons"]
    if horizons != [8, 16, 24, 32]:
        raise ValueError("短时审计必须分别测量8/16/24/32，不外推时域")
    (output/"declared_criteria.json").write_text(json.dumps(criteria, ensure_ascii=False, indent=2))
    collector.anchor_horizon = max(horizons)
    collector.record_controls = True
    collector.record_contacts = True
    before = collector.adapter.bundle.frozen_snapshot()
    all_cases, envelopes, phases_seen = [], [], set()
    joints = collector.actor.stage1.endecoder.kinematics.joint_order
    directions = [("root_x", 0, criteria["root_epsilons_m"]),
                  ("root_z", 2, criteria["root_epsilons_m"]),
                  ("leg", 7+joints.index("l_knee_pitch_joint"), criteria["joint_epsilons_rad"]),
                  ("arm", 7+joints.index("l_arm_roll_joint"), criteria["joint_epsilons_rad"])]
    for task in tasks:
        seed = task.get("seeds", [42])[0]
        seed_all(seed)
        music, initial = task_inputs(task, collector.adapter, device)
        collector.reset(music, initial)
        controls, samples, events_by_tick = [], [], {}
        while not collector.terminal and collector.state.tick < int(criteria["collection_seconds"]*600):
            collector.collect(1)
            block = collector.take_controls()
            controls.extend(block)
            events_by_tick.update({x.end.tick: x.event["contacts"] for x in block})
            while collector.ready_anchors:
                samples.append(collector.ready_anchors.popleft())
        if not samples:
            raise RuntimeError("没有完成H=32真实控制轨迹的首次接管样本")
        envelopes.append(state_envelope(torch.cat([x.start.state for x in controls])))
        selected = []
        for sample in samples:
            event = events_by_tick.get(sample.state.tick)
            if event is None:
                continue
            labels = phase_labels(sample, event, events_by_tick.get(sample.state.tick-12), collector.adapter)
            # 优先新阶段，再均匀取样。所有候选及选择原因保留，不能看方向结果挑样本。
            if not selected or (set(labels)-phases_seen):
                selected.append((sample, labels))
                phases_seen.update(labels)
            if len(selected) >= criteria["anchors_per_task"]:
                break
        for sample in samples[::max(1, len(samples)//criteria["anchors_per_task"])]:
            if len(selected) >= criteria["anchors_per_task"]:
                break
            if any(sample is old for old, _ in selected):
                continue
            event = events_by_tick.get(sample.state.tick)
            if event is not None:
                labels = phase_labels(sample, event, events_by_tick.get(sample.state.tick-12), collector.adapter)
                selected.append((sample, labels)); phases_seen.update(labels)
        actions = {x.start.tick: x.end.previous_action for x in controls}
        controls_by_tick = {x.start.tick: x for x in controls}

        def restore_prefix(tick):
            collector.env.reset(torch.arange(1, device=initial.device), initial)
            state = controls[0].start.detached()
            with torch.no_grad():
                for t in range(0, tick, 12):
                    # 长前缀不能开环重放固定动作：极小求解器噪声会破坏恢复起点。
                    # 重放同一已记录参考时间轴，SM仍按真实反馈计算动作，不使用未来真值。
                    state = replace(state, reference=controls_by_tick[t].start.reference)
                    state = collector.adapter.refresh_current_observation(state)
                    _, _, goal = state.reference.read_current_future(t)
                    action = collector.adapter.bundle.act(state.sm_history, goal)
                    next_state = collector.env.step_physics(action)
                    state = collector.adapter.advance(state, next_state, action, origin="real")
            expected = controls_by_tick[tick].start.state
            error = float((state.state-expected).abs().max())
            if error > criteria["reset_replay_max_abs"]:
                (output/"prefix_restore_failure.json").write_text(json.dumps({"task": task["id"],
                    "tick": tick, "state_max_abs": error, "method": "same_reference_closed_loop_reset_replay",
                    "not_world_model_evidence": True}, indent=2))
                raise RuntimeError(f"物理前缀恢复偏差{error}，不能把不配对的anchor当作WM误差")

        for anchor_index, (sample, labels) in enumerate(selected):
            request = sample.generation.request
            saved = output/f"anchor_{len(all_cases):04d}.pt"
            torch.save({"sample": pack_value(sample), "initial": initial.cpu(),
                        "prefix_actions": torch.cat([actions[t] for t in range(0, sample.state.tick, 12)]).cpu(),
                        "task": task, "seed": seed, "phases": labels}, saved)
            restore_prefix(request.ticket.request_tick)
            predicted_states, physical_states = [], []
            with torch.no_grad():
                predicted = forecast_rollout(request.snapshot, collector.adapter, request.ticket.first_influence_tick,
                    on_step=lambda z: predicted_states.append(z.state))
                physical = forecast_rollout(request.snapshot, collector.adapter, request.ticket.first_influence_tick,
                    transition_fn=collector.env.step_physics, on_step=lambda z: physical_states.append(z.state))
            forecast_stats = trajectory_metrics(torch.cat(predicted_states), torch.cat(physical_states),
                collector.adapter.runtime.kin, limits=criteria["state_rmse_limits"], start_tick=request.ticket.request_tick)
            forecast_ok = all(v["rmse"] <= criteria["state_rmse_limits"][k] for k,v in forecast_stats.items())
            case = {"task": task["id"], "seed": seed, "anchor_tick": sample.state.tick, "phases": labels,
                    "request_tick": request.ticket.request_tick, "actor_version": request.actor_version,
                    "actual_ready_tick": sample.generation.actual_ready_tick, "commit_tick": sample.generation.commit_tick,
                    "consumed_source_ticks": sample.consumed_source_ticks, "previously_consumed_versions": sample.previously_consumed_versions,
                    "no_previous_consumption_verified": True, "artifact": str(saved),
                    "prefix_restore": "same_reference_closed_loop_reset_replay",
                    "forecast_prediction": {"passed": forecast_ok, "statistics": forecast_stats,
                        "recorded_context_max_abs": float((predicted.context.features-sample.generation.forecast.features).abs().max()),
                        "condition_error_rms": float((predicted.context.features-physical.context.features).square().mean().sqrt())}, "horizons": {}}
            for horizon in horizons:
                engine = PhysicalAnchorImagination(collector.actor, collector.adapter, collector.scheduler, collector.rewards,
                    horizon_steps=horizon, gamma=collector.gamma, forecast_enabled=True)

                def run(physical_run, coordinate=0, amplitude=0.):
                    if physical_run:
                        restore_prefix(sample.state.tick)
                    trace, contact = [], []
                    def record(z, reward):
                        trace.append(z.state.detach().cpu())
                        if physical_run:
                            contact.append(collector.env.contact_events())
                    with torch.set_grad_enabled(not physical_run and isinstance(amplitude, torch.Tensor) and amplitude.requires_grad):
                        interval, _, _ = engine.rollout_anchor(sample,
                            plan_override=perturb_plan(sample.generation.plan, coordinate, amplitude),
                            transition_fn=collector.env.step_physics if physical_run else None, on_step=record)
                        objective = actor_return([interval], torch.zeros_like(interval.reward)).mean()
                    return objective, torch.cat(trace), contact, interval.steps

                model, mt, _, _ = run(False)
                real, rt, rc, rs = run(True)
                repeated, repeated_states, repeat_contacts, repeated_steps = run(True)
                length = min(len(mt), len(rt))
                statistics = trajectory_metrics(mt[:length], rt[:length], collector.adapter.runtime.kin,
                    limits=criteria["state_rmse_limits"], start_tick=sample.state.tick)
                repeat_delta = abs(float(real-repeated))
                reset_error = float((rt[:min(len(rt), len(repeated_states))]-repeated_states[:min(len(rt), len(repeated_states))]).abs().max())
                floor = max(criteria["minimum_resolvable_return_delta"], repeat_delta*criteria["noise_multiplier"])
                directions_report = []
                for name, coordinate, epsilons in directions:
                    amplitude = torch.zeros((), device=device, requires_grad=True)
                    objective, _, _, _ = run(False, coordinate, amplitude)
                    gradient, = torch.autograd.grad(objective, amplitude)
                    for eps in epsilons:
                        minus, _, _, _ = run(False, coordinate, -eps)
                        plus, _, _, _ = run(False, coordinate, eps)
                        negative, _, nc, ns = run(True, coordinate, -eps)
                        positive, _, pc, ps = run(True, coordinate, eps)
                        model_fd = float((plus-minus)/(2*eps))
                        physical_delta = float(positive-negative)
                        slope = physical_delta/(2*eps)
                        finite = bool(torch.isfinite(gradient))
                        fd_ok = finite and abs(float(gradient)-model_fd) <= criteria["autograd_fd_atol"] + criteria["autograd_fd_rtol"]*abs(model_fd)
                        changed = [contact_signature(x) for x in pc] != [contact_signature(x) for x in nc]
                        determinate = abs(physical_delta) > floor and rs == ns == ps == horizon
                        delta = float((positive if float(gradient) > 0 else negative)-real)
                        row = {"direction": name, "coordinate": coordinate, "epsilon": eps,
                               "unit": "m" if name.startswith("root") else "rad",
                               "model_autograd": float(gradient), "model_centered_difference": model_fd,
                               "autograd_fd_passed": fd_ok, "physical_centered_difference": slope,
                               "physical_return_delta_plus_minus": physical_delta, "noise_floor": floor,
                               "status": "determinate" if determinate else "indeterminate_noise_or_early_termination",
                               "direction_agrees": float(gradient)*slope > 0 if determinate else None,
                               "optimized_physical_delta": delta, "contact_events_changed": changed,
                               "physical_positive_contacts": [c["pair_signature"] for c in pc],
                               "physical_negative_contacts": [c["pair_signature"] for c in nc]}
                        directions_report.append(row)
                case["horizons"][str(horizon)] = {"statistics": statistics,
                    "prediction_passed": rs == horizon and all(v["rmse"] <= criteria["state_rmse_limits"][k] for k,v in statistics.items()),
                    "model_return": float(model), "physical_return": float(real), "physical_repeat_return": float(repeated),
                    "reset_repeat_state_max_abs": reset_error, "reset_repeat_return_delta": repeat_delta,
                    "reset_repeat_passed": rs == repeated_steps == horizon and reset_error <= criteria["reset_replay_max_abs"],
                    "base_contacts": [c["pair_signature"] for c in rc], "directions": directions_report}
                with (output/"short_progress.jsonl").open("a") as stream:
                    stream.write(json.dumps({"task": task["id"], "anchor_tick": sample.state.tick, "horizon": horizon,
                        "prediction": case["horizons"][str(horizon)]["prediction_passed"],
                        "determinate": sum(r["status"] == "determinate" for r in directions_report),
                        "agree": sum(r["direction_agrees"] is True for r in directions_report)}, allow_nan=False)+"\n")
                print(json.dumps({"event": "short_physics_anchor", "task": task["id"], "tick": sample.state.tick, "horizon": horizon}), flush=True)
            all_cases.append(case)
            with (output/"short_cases.jsonl").open("a") as stream:
                stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False)+"\n")
    collector.adapter.bundle.assert_unchanged(before)
    report = summarize_short(all_cases, criteria)
    report.update(schema="physical_anchor_short_bptt.v1", training_method="physical_anchor_short_bptt",
        cases=all_cases, frozen_sha256=collector.adapter.bundle.metadata["sha256"], frozen_unchanged=True,
        forecast_enabled=True, source_hash_gate=False, validated_state_envelope=combine_envelopes(envelopes),
        physics_determinism=cfg.get("execution", {}).get("physics_determinism", "not_guaranteed"),
        coverage={"physics": "nominal_flat_newton_mujoco_warp", "tasks": [
            {k: t.get(k) for k in ("id", "source", "split", "distribution_id")} for t in tasks]},
        task_identities={t["id"]: task_identity(t) for t in tasks}, criteria=criteria)
    (output/"short_physics_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


def numerical_convergence(rows, criteria):
    """区分局部导数与有限幅度弦斜率；物理方向仍检查全部预声明尺度。

    最小步长必须满足原误差阈值；若大步长不满足，还要求缩小步长时误差递减。
    小于原绝对误差阈值的波动视为已处于原数值容差内，不能用更大误差放宽小步检查。
    """
    result = {}
    for name in sorted({r["direction"] for r in rows}):
        sequence = sorted([r for r in rows if r["direction"] == name], key=lambda r: -r["epsilon"])
        errors = [abs(r["model_autograd"]-r["model_centered_difference"]) for r in sequence]
        decreasing = all(b <= max(a, criteria["autograd_fd_atol"]) for a,b in zip(errors, errors[1:]))
        result[name] = {"epsilons": [r["epsilon"] for r in sequence], "absolute_errors": errors,
            "all_scales_within_tolerance": all(r["autograd_fd_passed"] for r in sequence),
            "finest_scale_passed": sequence[-1]["autograd_fd_passed"], "error_converges": decreasing,
            "passed": sequence[-1]["autograd_fd_passed"] and decreasing}
    return result


def summarize_short(cases, criteria, *, numerical_rule="all_scales"):
    phases = set().union(*(set(c["phases"]) for c in cases)) if cases else set()
    coverage_ok = set(criteria["required_phases"]) <= phases
    result, supported = {}, []
    for h in criteria["horizons"]:
        entries = [c["horizons"][str(h)] for c in cases]
        rows = [r for x in entries for r in x["directions"]]
        groups = {}
        for group in ("root_x", "root_z", "leg", "arm"):
            determinate = [r for r in rows if r["direction"] == group and r["status"] == "determinate"]
            agrees = sum(r["direction_agrees"] for r in determinate)
            delta = sum(r["optimized_physical_delta"] for r in determinate)/len(determinate) if determinate else None
            fraction = agrees/len(determinate) if determinate else None
            groups[group] = {"determinate": len(determinate), "agreement_fraction": fraction,
                "mean_optimized_physical_delta": delta,
                "contact_changed": sum(r["contact_events_changed"] for r in determinate),
                "passed": len(determinate) >= criteria["minimum_determinate_per_group"] and
                    fraction >= criteria["minimum_direction_fraction"] and delta >= criteria["minimum_optimized_delta"]}
        prediction = bool(entries) and all(x["prediction_passed"] for x in entries)
        repeatable = bool(entries) and all(x["reset_repeat_passed"] for x in entries)
        original_numerical = bool(rows) and all(r["autograd_fd_passed"] for r in rows)
        convergence = [numerical_convergence(x["directions"], criteria) for x in entries]
        numerical = (original_numerical if numerical_rule == "all_scales" else
                     bool(entries) and all(g["passed"] for entry in convergence for g in entry.values()))
        direction = all(g["passed"] for g in groups.values())
        passed = prediction and repeatable and numerical and direction and coverage_ok
        if passed:
            supported.append(h)
        result[str(h)] = {"prediction_passed": prediction, "repeatability_passed": repeatable,
            "autograd_fd_passed": numerical, "original_all_scales_numerical_passed": original_numerical,
            "numerical_convergence": convergence, "direction_passed": direction, "groups": groups, "passed": passed}
    forecast = bool(cases) and all(c["forecast_prediction"]["passed"] for c in cases)
    choices = [h for h in supported if h in (16, 24, 32)]
    return {"horizons": result, "supported_horizons": supported, "numerical_derivative_rule": numerical_rule,
            "selected_horizon_steps": min(choices) if choices and forecast else None,
            "forecast_prediction_passed": forecast, "phase_coverage_passed": coverage_ok,
            "phases_observed": sorted(phases), "missing_phases": sorted(set(criteria["required_phases"])-phases),
            "legal_real_anchors": len(cases), "passed": bool(choices) and forecast and coverage_ok}
