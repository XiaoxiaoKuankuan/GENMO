"""重建、纯噪声生成、连续真实执行与导数方向的分离评价。

重建固定同源GT、k和epsilon；自由生成只传十个合法条件且运行完整DDIM，不以任意数据舞蹈
逐帧评分；闭环复用原异步PhysicalCollector/Scheduler，从真实历史持续接管。Newton扰动从
episode初态重放旧控制前缀，不回写中途state55冒充求解器恢复。GT接口审计与学生预测分开，
每个H的新方向/精度报告独立保存，失败不覆盖也不自动放宽状态支持。
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.closedloop.stage1_dataset import collate_stage1_training_samples
from gem.closedloop.training import batch_to_device
from gem.closedloop.mb_control.contracts import ReferenceExhausted
from gem.closedloop.mb_control.validation import state_envelope
from gem.closedloop.mb_control.model_audit import state_errors
from .checkpoint import read_checkpoint, atomic_save
from .config import build_actor, seed_all
from .contracts import PhysicsRejected, source_key, execution_contract, digest
from .data import SourceCatalog, SidecarStore
from .execution import execute_window
from .prepare_rollouts import asset_contract
from .reference import attach_sidecars, candidate_reference


def load_actor(cfg, checkpoint, device):
    actor, loss = build_actor(cfg, device)
    if checkpoint is not None:
        value = read_checkpoint(checkpoint)
        if value["config"]["stage1"]["model"] != cfg["stage1"]["model"]:
            raise ValueError("评价架构必须与本run新checkpoint一致")
        actor.load_state_dict(value["actor"], strict=True)
    actor.eval()
    return actor, loss


def sample_seed(meta, seed):
    return (int(hashlib.sha256(source_key(meta).encode()).hexdigest()[:12], 16)+seed) % (2**63-1)


def evaluation_windows(cfg, actor, adapter, *, split, structural_search_frames=0):
    catalog = SourceCatalog(cfg, split)
    horizon = max(cfg["physics"]["horizon"], *cfg["audit"]["horizons"]) if structural_search_frames else cfg["physics"]["horizon"]
    store = SidecarStore(cfg["data"]["sidecar"], split, asset_contract(cfg, adapter),
                        sample_contract=cfg["data"]["sample_contract"])
    available_sources = 0
    for dataset, row in catalog.sources():
        if available_sources >= cfg["evaluation"]["max_sources"]:
            break
        metadata = dataset.get_window(row, 0)["meta"]
        try:
            episode = store.episode(metadata)
        except PhysicsRejected:
            continue
        available_sources += 1
        last_frame = min(episode["source_frames"]-61, (len(episode["state"])-1)*12//20 - 30)
        if last_frame < 0:
            yield episode, None, None, None, "insufficient_real_trajectory"
            continue
        for fraction in (.15, .5, .85):
            frame = int(last_frame*fraction)//3*3
            offsets = [0] + [sign*delta for delta in range(3, structural_search_frames+1, 3) for sign in (1, -1)]
            window, reason = None, "insufficient_structural_support"
            for offset in offsets:
                selected_frame = frame+offset
                if not 0 <= selected_frame <= last_frame:
                    continue
                batch = batch_to_device(collate_stage1_training_samples([dataset.get_window(row, selected_frame)]), adapter.runtime.kin.device)
                batch, contexts, windows, reasons = attach_sidecars(batch, store, adapter, actor.stage1.endecoder.codec, horizon)
                window, reason = windows[0], reasons[0]
                if window is not None:
                    # 在模型预测前，只按固定60源帧/右速度halo检查时域几何，绝不按Loss或方向挑选。
                    required = window.start.tick + (window.horizon-1)*12 + 132
                    if structural_search_frames and required > window.request.tick+59*20:
                        window, reason = None, "candidate_right_halo_insufficient"
                    else:
                        window.provenance["audit_anchor_selection"] = {"requested_frame": frame,
                            "selected_frame": selected_frame, "search_frames": structural_search_frames,
                            "criterion": "real_time_prefix_and_halo_only_before_prediction"}
                        break
            yield episode, batch, contexts, window, reason


@torch.no_grad()
def replay_to_start(env, episode, window):
    device = window.start.state.device
    env.reset(torch.arange(1, device=device), episode["initial_state"].to(device)[None])
    for action in episode["action"][:window.start.tick//12]:
        env.step_physics(action.to(device)[None])
    observed = env.observe()
    return float((observed["state"]-window.start.state).abs().max())


def audit_gradients(cfg, checkpoint, output, adapter, env, interface_probe=False):
    if checkpoint is None and not interface_probe:
        raise ValueError("学生审计需要新流程checkpoint；GT接口测试必须显式--interface-probe")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"input_manifest.json").write_text(json.dumps({"schema": "genmo.joint_physics.audit_inputs.v1", "config": cfg}, ensure_ascii=False, indent=2))
    actor, _ = load_actor(cfg, checkpoint, adapter.runtime.kin.device)
    frozen = adapter.bundle.frozen_snapshot()
    assets = asset_contract(cfg, adapter)
    rows, real_states, real_actions, source_ids = [], [], [], set()
    from .trainer import grad_summary
    for episode, batch, contexts, window, reason in evaluation_windows(cfg, actor, adapter, split="train",
            structural_search_frames=cfg["audit"].get("structural_search_frames", 0)):
        source_ids.add(episode["source_key"])
        real_states.append(episode["state"])
        real_actions.append(episode["action"])
        if window is None:
            rows.append({"source": episode["source_key"], "rejection": reason})
            continue
        generator = torch.Generator().manual_seed(sample_seed(batch["meta"][0], cfg["evaluation"]["seed"]))
        noise = torch.randn(1, 60, 30, generator=generator).to(window.start.state)
        forward = actor.denoise_supervised(batch, contexts, timesteps=torch.tensor([50], device=noise.device), noise=noise,
                                           drop_music_mask=torch.zeros(1, dtype=torch.bool, device=noise.device))
        prediction = (actor.stage1.endecoder.normalize(batch["target_qpos30"]).detach().requires_grad_()
                      if interface_probe else forward["pred_x_start"])
        for horizon in cfg["audit"]["horizons"]:
            selected = replace(window, horizon=horizon)
            for joint in (10, 17):
                row = {"source": episode["source_key"], "frame": batch["meta"][0]["decision_frame"],
                       "anchor_selection": window.provenance.get("audit_anchor_selection"),
                       "horizon": horizon, "joint_genmo_feature": joint, "prediction_origin": "gt_interface_probe" if interface_probe else "student_denoised_x0"}
                try:
                    result = execute_window(selected, prediction, forward["static_conf_logits"], actor, adapter, cfg["execution_loss"])
                    gradient, = torch.autograd.grad(result["loss"], prediction, retain_graph=True)
                    direction = torch.zeros_like(prediction)
                    start = window.ticket.prefix_frames
                    direction[:, start:60, joint] = torch.sin(torch.linspace(0, torch.pi, 60-start, device=prediction.device))**2
                    eps = float(cfg["audit"]["epsilon"])
                    automatic = float((gradient*direction).sum())
                    values, physical, repeat = [], [], []
                    with torch.no_grad():
                        for sign in (1, -1):
                            perturbation = prediction+sign*eps*direction
                            model = execute_window(selected, perturbation, forward["static_conf_logits"], actor, adapter, cfg["execution_loss"])
                            values.append(float(model["loss"]))
                            if env is not None:
                                repeat.append(replay_to_start(env, episode, selected))
                                physical.append(execute_window(selected, perturbation, forward["static_conf_logits"], actor, adapter,
                                                               cfg["execution_loss"], transition=env.step_physics))
                    fd = (values[0]-values[1])/(2*eps)
                    row.update(execution_loss=float(result["loss"].detach()), x0_gradient_norm=float(gradient.norm()),
                               autograd=automatic, finite_difference=fd,
                               fd_pass=abs(fd-automatic) <= cfg["audit"]["fd_atol"]+cfg["audit"]["fd_rtol"]*abs(fd),
                               coverage=result["coverage"])
                    if not interface_probe:
                        row["execution_parameter_gradients"] = grad_summary(result["loss"], actor)
                    if physical:
                        pd = (float(physical[0]["loss"])-float(physical[1]["loss"]))/ (2*eps)
                        # ±中心的平均用于局部预测误差；不冒充真实完整反事实轨迹。
                        error = state_errors(result["states"].flatten(0, 1),
                            ((physical[0]["states"]+physical[1]["states"])/2).flatten(0, 1))
                        row.update(newton_derivative=pd, same_start_max_abs=max(repeat),
                            newton_direction_pass=abs(pd) > cfg["audit"]["direction_min_abs"] and automatic*pd > 0,
                            model_errors={k: float(v) for k,v in error.items()},
                            model_accuracy_pass=all(float(error[k]) <= threshold for k,threshold in cfg["audit"]["state_rmse_limits"].items()))
                        real_states.extend(p["states"].detach().cpu().flatten(0, 1) for p in physical)
                        real_actions.extend(p["actions"].detach().cpu().flatten(0, 1) for p in physical)
                except PhysicsRejected as error:
                    row["rejection"] = error.reason
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    adapter.bundle.assert_unchanged(frozen)
    horizons, supported = {}, []
    for horizon in cfg["audit"]["horizons"]:
        selected = [r for r in rows if r.get("horizon") == horizon and "rejection" not in r]
        direction = sum(r.get("newton_direction_pass", False) for r in selected)
        passed = (env is not None and len(selected) >= cfg["audit"]["minimum_probes"] and
            direction/max(1,len(selected)) >= cfg["audit"]["minimum_direction_fraction"] and
            all(r["fd_pass"] and r.get("model_accuracy_pass", False) and
                r.get("same_start_max_abs", 1.) <= cfg["audit"]["same_start_atol"] for r in selected))
        horizons[str(horizon)] = {"probes": len(selected), "direction_agreement": direction, "passed": passed}
        if passed:
            supported.append(horizon)
    report = {"schema": "genmo.joint_physics.validation.v1", "assets": assets, "source_ids": sorted(source_ids),
        "objective": execution_contract(cfg),
        "prediction_origin": "gt_interface_probe_not_student_success" if interface_probe else "student_denoised_x0",
        "rows": rows, "horizons": horizons, "supported_horizons": supported, "frozen_unchanged": True,
        "criteria": cfg["audit"], "state_envelope": state_envelope(torch.cat(real_states)),
        "action_min": torch.cat(real_actions).amin(0).tolist(), "action_max": torch.cat(real_actions).amax(0).tolist(),
        "evidence": "newton_replayed_prefix_paired_perturbations" if env is not None else "autograd_and_finite_difference_only"}
    forecast = validate_forecasts(cfg, adapter)
    report.update(forecast_rows=forecast, forecast_passed=bool(forecast) and all(r["passed"] for r in forecast))
    (output/"validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


@torch.no_grad()
def validate_forecasts(cfg, adapter):
    """只用已提交旧计划预测，再与未进入条件的实测未来状态比较；不重跑/改写物理。"""
    from types import SimpleNamespace
    from gem.robots.bumi.endecoder import BumiEndecoder
    from .config import ROOT
    options = dict(cfg["stage1"]["endecoder"])
    options["kinematics_path"] = str(ROOT/options["kinematics_path"])
    holder = SimpleNamespace(stage1=SimpleNamespace(endecoder=BumiEndecoder(**options)))
    rows = []
    for episode, batch, contexts, window, reason in evaluation_windows(cfg, holder, adapter, split="train",
            structural_search_frames=cfg["audit"].get("structural_search_frames", 0)):
        if window is None:
            continue
        requests = episode["offline_requests"]
        i = batch["meta"][0]["decision_frame"]//3
        end = int(requests["state_offsets"][i+1])
        predicted = requests["predicted_states"][end-1:end].to(window.start.state)
        errors = state_errors(predicted, window.start.state)
        rows.append({"source": episode["source_key"], "request_tick": window.request.tick, "forecast_tick": window.start.tick,
            "steps": (window.start.tick-window.request.tick)//12, "errors": {k:float(v) for k,v in errors.items()},
            "passed": all(float(errors[k]) <= bound for k,bound in cfg["audit"]["state_rmse_limits"].items())})
    return rows


def complete_validation(cfg, raw_path, output, adapter):
    """对已有原始测量补齐合同；必须读取该次运行的输入manifest，不能猜测旧报告目标。"""
    raw_path, output = Path(raw_path), Path(output)
    report = json.loads(raw_path.read_text())
    recorded = json.loads((raw_path.parent/"input_manifest.json").read_text())["config"]
    if report["schema"] != "genmo.joint_physics.validation.v1" or report["assets"] != asset_contract(cfg, adapter):
        raise ValueError("不能给旧Stage2或不同资产报告补合同")
    if execution_contract(recorded) != execution_contract(cfg) or recorded["data"] != cfg["data"] or report["criteria"] != cfg["audit"]:
        raise ValueError("已实际记录的目标/输入/阈值不同，需要重新实际测量")
    forecasts = validate_forecasts(cfg, adapter)
    report.update(objective=execution_contract(recorded), forecast_rows=forecasts,
        forecast_passed=bool(forecasts) and all(r["passed"] for r in forecasts),
        raw_report=str(raw_path), raw_report_sha256=digest(raw_path),
        completion_scope="原始Newton结果不变；绑定该次已记录配置并追加sidecar真实前瞻误差")
    output.mkdir(parents=True, exist_ok=False)
    (output/"validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


@torch.no_grad()
def execute_generated(window, reference, adapter, env, steps):
    z = replace(window.start, reference=reference)
    records, rewards = [], []
    reason = "short_window_budget"
    for _ in range(steps):
        try:
            z = adapter.refresh_current_observation(z)
            _, _, goal = z.reference.read_current_future(z.tick)
            action = adapter.bundle.act(z.sm_history, goal)
            state = env.step_physics(action)
            previous_action = z.previous_action
            z = adapter.advance(z, state, action, origin="real")
            target, _, _ = z.reference.read_current_future(z.tick, 0)
            reward, _ = adapter.runtime.reward(state, action, previous_action, target)
            records.append(state.cpu())
            rewards.append(float(reward))
            if adapter.runtime.termination(state, target).any():
                reason = "physical_failure"
                break
        except ReferenceExhausted:
            reason = "reference_exhausted"
            break
    return {"reason": reason, "steps": len(records), "tracking_reward_mean": sum(rewards)/max(1,len(rewards))}, records


class TrackingMonitor:
    """仅服务真实评价的SM原指标，不参与联合训练目标。"""
    def __init__(self, adapter):
        self.adapter, self.config = adapter, {}

    def __call__(self, state, action, previous, reference, **unused):
        reward, components = self.adapter.runtime.reward(state, action, previous, reference)
        return reward, {"sm_reward": reward, **components}


def closed_loop_source(cfg, actor, adapter, env, dataset, row):
    from supermimic.mdp import flatten_state
    from gem.closedloop.mb_control.collector import PhysicalCollector
    from gem.closedloop.mb_control.scheduler import Scheduler
    from gem.closedloop.mb_control.evaluation import motion_metrics
    from supermimic.rotations import quat_to_mat
    sequence = dataset.reader.load_aligned_sequence(dataset.rows[row])
    # 只用初态；后续未知GT动作不交给生成器/参考缓存。启动参考由collector保持初态。
    q = sequence["qpos"][0].to(env.device)
    ground = dataset._ground_supervision(sequence)["ground_height_world_m"]
    p = q[:3] - q.new_tensor([0., 0., ground])
    initial = flatten_state((p[None], quat_to_mat(q[3:7])[None], q.new_zeros(1,3), q.new_zeros(1,3),
                             q[7:][list(adapter.to_sm)][None], q.new_zeros(1,21)))
    music = sequence["music"].to(env.device)[None]
    scheduler = Scheduler(cfg["evaluation"]["latency_ticks"], request_mode="fixed_interval",
                          request_interval_ticks=cfg["evaluation"]["request_interval_ticks"],
                          latency_margin_ticks=cfg["evaluation"].get("latency_margin_ticks",0))
    collector = PhysicalCollector(env, actor, adapter, scheduler, TrackingMonitor(adapter), record_trajectory=True)
    reason = "not_started"
    try:
        collector.reset(music, initial)
        timing_mode = cfg["evaluation"].get("timing_mode", "wall_clock_async")
        if timing_mode == "fixed_delay_replay":
            delay = cfg["evaluation"]["fixed_delivery_ticks"]
            if delay % 12:
                raise ValueError("固定交付事件必须在50Hz控制网格")
            collector.timing_mode = timing_mode
            collector.delay_events = {str(t): {"delivered":True,"actual_ready_tick":t+delay,
                "delivery_tick":t+delay,"commit_tick":t+delay} for t in range(0,music.shape[1]*20,60)}
        elif timing_mode != "wall_clock_async":
            raise ValueError("闭环必须明确固定交付或实际墙钟模式")
        warmup = None
        if cfg["evaluation"].get("warmup_inference", True):
            from gem.closedloop.mb_control.compilation import warmup_inference
            # 只预热数值计算；不推进或重置真实状态，不更改权重、延迟预算或保护前缀。
            warmup = warmup_inference(actor, adapter, scheduler, collector.state.detached())
        started = time.perf_counter()
        while not collector.terminal:
            intervals = collector.collect(1)
            reason = intervals[-1].reason
            limit = cfg["evaluation"]["max_control_steps"]
            if limit is not None and collector.state.tick//12 >= limit:
                reason = "explicit_control_budget"
                break
        result = {"source": dataset.rows[row]["sample_id"], "reason": reason,
            "complete_song": reason == "music_end", "physical_seconds": collector.state.tick/600,
            "music_seconds": music.shape[1]/30, "wall_seconds": time.perf_counter()-started,
            "commits": sum(e["accepted"] for e in collector.events), "events": collector.events,
            "cancelled_requests": collector.cancelled_requests, "control_deadline_overruns": collector.overruns,
            "requests": collector.request_distribution, "numerical_warmup": warmup,
            "bootstrap_latency_ticks": cfg["evaluation"]["latency_ticks"],
            "startup_hold_source_seconds": 59/30, "startup_executable_seconds": 1.74,
            "startup_hold_counted_as_dance_gain": False,
            "inference_backend": cfg["runtime"].get("inference_backend","eager"),
            "request_interval_seconds": cfg["evaluation"]["request_interval_ticks"]/600,
            "motion": motion_metrics(collector.trajectory, music), "timing_mode": collector.timing_mode}
        return result, collector.trajectory
    finally:
        collector.worker.close()


def evaluate(cfg, checkpoint, output, adapter, env, mode):
    if checkpoint is None:
        raise ValueError("评价必须显式提供新流程checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    actor, losses = load_actor(cfg, checkpoint, env.device)
    from .inference import configure
    configure(actor,cfg["runtime"].get("inference_backend","eager"))
    checkpoint_step = read_checkpoint(checkpoint)["global_step"]
    frozen = adapter.bundle.frozen_snapshot()
    rows = []
    seed_all(cfg["evaluation"]["seed"])
    if mode == "closed_loop":
        catalog = SourceCatalog(cfg, cfg["evaluation"]["split"])
        for i, (dataset, row) in enumerate(catalog.sources()):
            if i >= cfg["evaluation"]["max_sources"]:
                break
            result, trajectory = closed_loop_source(cfg, actor, adapter, env, dataset, row)
            atomic_save(trajectory, output/f"closed_loop_{i:04d}.pt")
            rows.append(result)
    else:
        for episode, batch, contexts, window, reason in evaluation_windows(cfg, actor, adapter, split=cfg["evaluation"]["split"]):
            if window is None:
                rows.append({"source": episode["source_key"], "rejection": reason})
                continue
            g = torch.Generator().manual_seed(sample_seed(batch["meta"][0], cfg["evaluation"]["seed"]))
            noise = torch.randn(1, 60, 30, generator=g).to(env.device)
            if mode == "reconstruction":
                for k in cfg["evaluation"]["reconstruction_timesteps"]:
                    with torch.no_grad():
                        out = actor.denoise_supervised(batch, contexts, torch.tensor([k], device=env.device), noise,
                            torch.zeros(1, dtype=torch.bool, device=env.device))
                        loss, _ = losses(batch, out["pred_x_start"], out["static_conf_logits"], checkpoint_step)
                        row = {"source": episode["source_key"], "frame": batch["meta"][0]["decision_frame"], "k": k,
                               "noise_seed": g.initial_seed(), "data_loss": float(loss)}
                        try:
                            row["same_start_max_abs"] = replay_to_start(env, episode, window)
                            result = execute_window(window, out["pred_x_start"], out["static_conf_logits"], actor, adapter,
                                                    cfg["execution_loss"], transition=env.step_physics)
                            row["real_execution_loss"] = float(result["loss"])
                            row["coverage"] = result["coverage"]
                            row["physical_failure_steps"] = result["failure_steps"]
                        except PhysicsRejected as error:
                            row["rejection"] = error.reason
                    rows.append(row)
            else:
                conditions = {k: batch[k] for k in STAGE1_CONDITION_KEYS}
                generated = []
                with torch.no_grad():
                    for seed_offset in (0, 1):
                        g.manual_seed(g.initial_seed()+seed_offset)
                        out = actor.sample_differentiable(conditions, contexts[0], torch.randn(1,60,30,generator=g).to(env.device))
                        generated.append(out["qpos30"])
                        row = {"source": episode["source_key"], "frame": batch["meta"][0]["decision_frame"],
                               "noise_seed": g.initial_seed(), "sampler": "pure_noise_ddim20_eta0_cfg2.5",
                               "joint_range_mean_rad": float(out["qpos30"][...,9:].std(dim=1).mean())}
                        try:
                            prediction = actor.stage1.endecoder.normalize(out["qpos30"])
                            reference = candidate_reference(window, prediction, out["contact_logits"], actor.stage1.endecoder.codec, actor.stage1.endecoder)
                            row["same_start_max_abs"] = replay_to_start(env, episode, window)
                            metrics, states = execute_generated(window, reference, adapter, env, window.horizon)
                            row["real_physics"] = metrics
                            atomic_save({"generated_qpos30": out["qpos30"].cpu(), "physical_states": states}, output/f"free_{len(rows):04d}.pt")
                        except PhysicsRejected as error:
                            row["rejection"] = error.reason
                        rows.append(row)
                    rows[-1]["within_model_seed_diversity_rms"] = float((generated[0]-generated[1]).square().mean().sqrt())
                    silent = dict(conditions, music_valid=torch.zeros_like(conditions["music_valid"]))
                    without = actor.sample_differentiable(silent, contexts[0], noise)
                    rows[-1]["music_condition_sensitivity_rms"] = float((generated[0]-without["qpos30"]).square().mean().sqrt())
    adapter.bundle.assert_unchanged(frozen)
    result = {"mode": mode, "checkpoint": str(checkpoint), "rows": rows, "frozen_unchanged": True,
              "checkpoint_step": checkpoint_step, "evaluation_config": cfg["evaluation"],
              "scope": "complete_song_attempts" if mode == "closed_loop" else "independent_local_windows",
              "evidence": "real_newton", "method_improvement_claim": False}
    (output/"evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result
