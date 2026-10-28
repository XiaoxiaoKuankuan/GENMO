"""新Stage2显式命令行入口，默认有界，导入本模块没有GPU/Isaac副作用。

先读用户配置及实际资产，再按需加载SM桥接和Isaac。提供短测训练、真实全曲评价、
WM物理方向审计和纯数值梯度审计；任何运行必须由用户显式执行命令。新输出目录
独占创建，续训也写新目录；源码版本仅落manifest，不作恢复门禁。
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time


def main(kind):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--torch-device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--max-actor-updates", type=int, default=5000)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--actor-checkpoint")
    parser.add_argument("--rpc-directory")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--collector-id", type=int, default=0)
    parser.add_argument("--initial-actor", action="store_true",
                        help="仅评价：用明确的未更新GENMO运行完整Forecast路径并导出权重身份")
    parser.add_argument("--mode", choices=("stage1", "full", "no_forecast", "calibration"), default="stage1" if kind == "evaluate" else "full")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--criteria", help="WM审计的显式JSON数值/方向验收阈值")
    parser.add_argument("--diagnostic-horizons", type=int, nargs="+", default=[5, 10, 15, 20, 32, 45, 90, 93, 102],
                        help="仅dynamics故障定位使用；不能用该局部报告解锁正式训练")
    parser.add_argument("--perturbation-epsilon", type=float, default=.01,
                        help="真实方向中心差分的关节扰动幅度；改幅度的复查不能覆盖原验收报告")
    parser.add_argument("--latency-scenario", choices=("sample", "measured_worst"), default="sample",
                        help="P3可按实测延迟上界构造全部决策的保守时域；不更改延迟样本")
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--bootstrap-latency-ticks", type=int, help="首轮部署测量可显式给候选预算；报告保留未标定状态")
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    initial_config = OmegaConf.load(preliminary.config)
    sys.path.insert(0, str(Path(initial_config.supermimic.repo).expanduser().resolve()))
    if kind != "gradients":
        from isaaclab.app import add_launcher_args, launch_simulation
        add_launcher_args(parser)
    args = parser.parse_args()
    if args.initial_actor and (kind != "evaluate" or args.actor_checkpoint or args.mode not in ("full", "no_forecast")):
        raise ValueError("initial-actor只用于没有训练checkpoint的更新前完整条件物理基线")
    import numpy as np
    import torch
    from .application import (read_config, require_file, build_components, seed_all, read_tasks, task_inputs,
                              behavior_config, provenance, SupervisionStream, LatencyProfile, build_stage1_losses, task_identity)
    from .collector import PhysicalCollector
    from .evaluation import evaluate_tasks
    from .trainer import Stage2Trainer
    from .replay import RecentReplay
    from .checkpoint import save_checkpoint, load_checkpoint, SCHEMA
    cfg = read_config(args.config)
    if args.run_mode == "formal" and kind != "train":
        raise ValueError("formal只用于显式训练命令")
    if args.run_mode == "formal":
        if args.max_actor_updates < 1 or not cfg.get("formal"):
            raise ValueError("正式模式需要正数accepted目标和完整formal配置")
        cfg["training"]["require_critic_quality"] = True
    cfg["model"]["forecast_enabled"] = args.mode in ("full", "calibration")
    if args.bootstrap_latency_ticks is not None:
        cfg["clock"]["bootstrap_latency_ticks"] = args.bootstrap_latency_ticks
    seed_all(cfg["training"]["seed"])
    tasks = read_tasks(cfg["tasks"])
    profile = LatencyProfile(cfg["latency_profile"], cfg["training"]["seed"] + 1) if cfg["latency_profile"] else None
    if cfg["clock"]["bootstrap_latency_ticks"] is None and profile is not None:
        cfg["clock"]["bootstrap_latency_ticks"] = int(np.quantile(profile.samples, .9, method="higher"))
    if cfg["clock"]["bootstrap_latency_ticks"] is None:
        raise ValueError("需要新延迟profile或显式首轮测量预算，不能继承旧控制器延迟")
    if kind == "train":
        if args.mode in ("stage1", "calibration"):
            raise ValueError("Stage1 baseline不启动优化器")
        if args.run_mode == "smoke" and not 1 <= args.updates <= cfg["execution"]["maximum_explicit_updates"]:
            raise ValueError("本入口仅执行配置允许的显式有界更新")
        if not 1 <= cfg["training"]["actor_microbatch"] <= 8 or cfg["execution"]["physical_batch"] != 1:
            raise ValueError("每环境物理batch1，BPTT微批1–8逐样本梯度累积")
        if profile is None:
            raise ValueError("训练需要独立实测部署延迟分布")
        report = json.loads(require_file(cfg["imagination"]["physical_validation_report"], "P3物理方向报告").read_text())
        if cfg["imagination"].get("mode") == "physical_anchor_short_bptt":
            from .validation import select_short_horizon
            select_short_horizon(cfg, report)
        else:
            for key in ("prediction_passed", "physical_direction_passed", "reset_replay_repeatability_passed", "known_reference_passed"):
                if report.get(key) is not True:
                    raise ValueError(f"P3尚未支持训练: {key}")
            steps = cfg["imagination"]["validated_model_steps"]
            if not isinstance(steps, int) or steps < 1 or steps > report["supported_control_steps"]:
                raise ValueError("训练WM时域必须受实际P3证据支持")
        from .validation import validate_coverage
        validate_coverage(tasks, report)
        if report.get("forecast_enabled") != cfg["model"]["forecast_enabled"]:
            raise ValueError("P3报告与本次前瞻条件设置不一致")
    actor, adapter, scheduler, rewards, critic, engine, original = build_components(cfg, args.torch_device)
    if kind == "train":
        report_frozen = report.get("frozen_sha256")
        if report_frozen != adapter.bundle.metadata["sha256"]:
            raise ValueError("P3报告不属于当前冻结资产")
        if profile.metadata.get("forecast_enabled") != cfg["model"]["forecast_enabled"] or profile.metadata.get("frozen_sha256") != adapter.bundle.metadata["sha256"]:
            raise ValueError("部署延迟必须来自同一冻结资产及同样的前瞻路径，不能借用无前瞻baseline延迟")
    if args.actor_checkpoint:
        ck = torch.load(require_file(args.actor_checkpoint, "Stage2 Actor"), map_location="cpu", weights_only=False)
        inference_only=ck.get("schema")=="genmo.stage2.inference_export.v1" and kind=="evaluate"
        if (ck.get("schema") != SCHEMA and not inference_only) or ck["frozen_sha256"] != adapter.bundle.metadata["sha256"]:
            raise ValueError("评估必须使用同一SM资产的新Stage2 checkpoint")
        model_contract=ck["model_contract"] if inference_only else ck["behavior_config"]["model"]
        if args.mode in ("stage1", "calibration") or model_contract["forecast_enabled"] != cfg["model"]["forecast_enabled"]:
            raise ValueError("条件消融必须匹配训练条件，不能只在测试时删除未来条件")
        actor.load_state_dict(ck["actor"], strict=True)
    elif kind == "evaluate" and args.mode not in ("stage1", "calibration") and not args.initial_actor:
        raise ValueError("完整/无前瞻训练模型评估需要明确Actor checkpoint")
    output = Path(args.output).expanduser().resolve()
    if args.run_mode == "formal":
        if output.exists() and (output / "training.jsonl").exists() and not (args.resume or args.resume_latest):
            raise ValueError("正式目录已有训练，必须明确resume，不能覆盖")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
    source_info = provenance(cfg)
    manifest = output / (f"resume_manifest_{time.time_ns()}.json" if (output / "manifest.json").exists() else "manifest.json")
    manifest.write_text(json.dumps(source_info, ensure_ascii=False, indent=2))
    if kind == "evaluate":
        from .evaluation import evaluation_contract
        contract = evaluation_contract(cfg, tasks, adapter.bundle.metadata["sha256"])
        (output/"evaluation_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2))
        if args.initial_actor:
            from supermimic.bridge.bundle import atomic_save
            # 推理身份资产不包含优化器，不能冒充已开始训练或可恢复训练checkpoint。
            atomic_save({"schema": "genmo.evaluation.initial_actor.v1", "actor": actor.state_dict(),
                         "accepted_actor_updates": 0, "contract": contract}, output/"evaluation_actor.pt")
    if cfg["model"].get("compile_networks", False):
        from .compilation import warmup_inference
        from .gradient_audit import tensor_fixture
        music, initial = task_inputs(tasks[0], adapter, args.torch_device)
        warmup = warmup_inference(actor, adapter, scheduler, tensor_fixture(adapter, music, initial))
        warmup_path = output / (f"compile_warmup_{time.time_ns()}.json" if (output / "compile_warmup.json").exists()
                                else "compile_warmup.json")
        warmup_path.write_text(json.dumps(warmup, ensure_ascii=False, indent=2))
    if kind == "gradients":
        from .gradient_audit import tensor_fixture, audit_gradients
        music, initial = task_inputs(tasks[0], adapter, args.torch_device)
        z = tensor_fixture(adapter, music, initial)
        latencies = profile.sample(engine.decisions) if profile else [scheduler.bootstrap_latency_ticks] * engine.decisions
        result = audit_gradients(actor, critic, engine, z, latencies, finite_difference=args.finite_difference)
        from .execution_audit import audit_execution_paths
        supervision = SupervisionStream(cfg["stage1"]["data_config"], cfg["training"]["supervision_batch"],
                                         cfg["training"]["seed"], args.torch_device)
        result["execution_paths"] = audit_execution_paths(actor, critic, engine, z,
            [torch.randn(1, 60, 30, device=args.torch_device) for _ in range(engine.decisions)], latencies,
            supervised_batch=supervision.next(), stage1_losses=build_stage1_losses(actor.stage1, original).to(args.torch_device),
            finite_difference=args.finite_difference)
        (output / "gradient_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps(result, ensure_ascii=False))
        return
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    sim_cfg = make_sim(args.torch_device, args)
    if cfg.get("execution", {}).get("physics_determinism"):
        from isaaclab.utils import replace as config_replace
        # SM观测来自实际刚体/关节状态；不使用MuJoCo内部sensor计算。
        # Newton确定性模式要求关闭该独立计算，真实碰撞数据仍从solver.contact读取。
        solver = config_replace(sim_cfg.physics.solver_cfg, disable_sensors=True)
        sim_cfg = config_replace(sim_cfg, physics=config_replace(sim_cfg.physics,
            class_type=None, solver_cfg=solver, deterministic_mode=cfg["execution"]["physics_determinism"]))
    activity = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    profiler = torch.profiler.profile(activities=activity, profile_memory=True) if args.profile else nullcontext()
    with launch_simulation(sim_cfg, args), profiler as measured:
        env = StreamingPhysicalEnv(sim_cfg, 1, adapter.bundle)
        # Newton首次step触发编译/图捕获。显式在episode前完成，不计入部署生成延迟。
        warm_started = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                env.step_physics(torch.zeros(1, 21, device=args.torch_device))
        torch.cuda.synchronize()
        (output / "physics_startup.json").write_text(json.dumps({"warmup_control_steps": 3,
            "startup_seconds": time.perf_counter() - warm_started, "episode_reset_after_warmup": True}))
        collector = PhysicalCollector(env, actor, adapter, scheduler, rewards, forecast_enabled=cfg["model"]["forecast_enabled"],
                                      gamma=cfg["critic"]["gamma_per_control"], record_trajectory=kind in ("evaluate", "census"))
        collector.record_contact_metrics = kind in ("evaluate", "census")
        if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1":
            collector.noise_seed=cfg["teacher"].get("online_seed",42)
        # 固定事件仅用于离线评价；正式采集仍使用真实墙钟异步，不能由评价设置改写。
        collector.timing_mode=("wall_clock_async" if kind in ("train","collector") else
            cfg.get("evaluation",{}).get("timing_mode","wall_clock_async"))
        try:
            if kind == "census":
                from .stage1_census import run_worker
                run_worker(collector, cfg, output, args.torch_device)
            elif kind == "online_teacher":
                from .online_teacher import build_online_timelines
                build_online_timelines(collector,engine.teacher,tasks,cfg,output,args.torch_device)
            elif kind == "evaluate":
                # 推理已加载checkpoint中的实际权重，事件元数据也须携带同一个
                # 有效更新版本；保留参考版本自己的编号，不改物理/采样/调度。
                collector.actor_version = (int(ck["accepted_actor_updates"] if inference_only else ck["counters"]["accepted_actor_updates"]) if args.actor_checkpoint else 0)
                evaluate_tasks(collector, tasks, output, mode=args.mode, device=args.torch_device, max_seconds=args.max_seconds)
            elif kind == "collector":
                from .parallel_collection import run_collector_service
                run_collector_service(args, cfg, collector)
            elif kind == "drift":
                from .drift_physical import run_drift
                run_drift(collector,actor,critic,engine,original,cfg,tasks,output,args.torch_device)
            elif kind == "critic":
                from .critic_warmup import choose_warmup_tasks, warmup_critic, save_warmup
                losses = build_stage1_losses(actor.stage1, original).to(args.torch_device)
                trainer = Stage2Trainer(actor, critic, engine, losses, cfg["training"])
                selected = choose_warmup_tasks(tasks, cfg["formal"]["critic_warmup"])
                validation = read_tasks(cfg["formal"]["critic_validation_tasks"])
                warmup_critic(trainer, collector, selected, validation, cfg["formal"]["critic_warmup"], output,
                              device=args.torch_device)
                save_warmup(output/"critic_ready.pt", trainer, cfg)
            elif kind == "mc":
                from .critic_warmup import collect_mc
                settings = cfg["mc_collection"]
                _, episodes, steps = collect_mc(collector,tasks,seed=settings["seed"],device=args.torch_device,
                    max_control_steps=cfg["formal"]["critic_warmup"]["max_collection_steps"],output=output,
                    split=settings["split"],episode_offset=settings["episode_offset"])
                torch.save({"actor":actor.state_dict(),"episodes":episodes,"physical_control_steps":steps,
                    "split":settings["split"],"contract":{k:cfg[k] for k in ("model","clock","rewards","critic")},
                    "teacher_identity":engine.teacher.identity,"frozen_sha256":adapter.bundle.metadata["sha256"],
                    "physics_determinism":cfg["execution"].get("physics_determinism")},output/"mc_complete.pt")
            elif kind == "teacher_audit":
                from .teacher_audit import audit_teacher
                audit_teacher(collector,actor,critic,engine,original,cfg,tasks,output,args.torch_device)
            elif kind == "music_repair":
                from .music_repair_audit import run_music_repair_audit
                run_music_repair_audit(collector,actor,critic,engine,original,cfg,tasks,output,args.torch_device)
            elif kind == "short":
                from .short_audit import audit_short
                if not args.criteria:
                    raise ValueError("短时物理验收必须预先声明完整阈值与扰动尺度")
                audit_short(collector, tasks, output, cfg, json.loads(Path(args.criteria).read_text()), device=args.torch_device)
            elif kind == "collapse":
                from .collapse_audit import audit_collapse
                audit_collapse(collector,tasks,cfg,output,args.torch_device)
            elif kind == "dynamics":
                from .model_audit import audit_known_reference
                criteria = json.loads(Path(args.criteria).read_text()) if args.criteria else None
                before = adapter.bundle.frozen_snapshot()
                cases = []
                if not args.diagnostic_horizons or min(args.diagnostic_horizons) < 1:
                    raise ValueError("诊断时域必须为正整数")
                for task in tasks:
                    for seed in task.get("seeds", [42]):
                        seed_all(seed)
                        music, initial = task_inputs(task, adapter, args.torch_device)
                        reference = torch.as_tensor(np.load(require_file(task["known_reference_world_qpos28"],
                            "已知参考qpos28"), allow_pickle=False), device=args.torch_device, dtype=torch.float32)[None]
                        item = audit_known_reference(collector, music, initial, reference,
                            horizons=args.diagnostic_horizons, criteria=criteria)
                        item.update(task=task["id"], seed=seed)
                        cases.append(item)
                        with (output / "dynamics_cases.jsonl").open("a") as stream:
                            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                        print(json.dumps({"event": "dynamics_case", "task": task["id"],
                            "known_reference_passed": item["passed"]}, ensure_ascii=False), flush=True)
                adapter.bundle.assert_unchanged(before)
                (output / "dynamics_diagnostic.json").write_text(json.dumps({
                    "scope": "known_reference_diagnostics_only_not_formal_acceptance",
                    "frozen_sha256": adapter.bundle.metadata["sha256"], "frozen_unchanged": True,
                    "supported_control_steps": 0, "cases": cases}, ensure_ascii=False, indent=2, allow_nan=False))
            elif kind == "model":
                from .model_audit import audit_generated_reference, audit_known_reference
                criteria = json.loads(Path(args.criteria).read_text()) if args.criteria else None
                if criteria is not None:
                    required = ("minimum_directions", "minimum_direction_fraction", "minimum_optimized_delta", "reset_replay_max_abs")
                    if any(not isinstance(criteria.get(k), (int, float)) for k in required) or not criteria.get("state_rmse_limits"):
                        raise ValueError("验收阈值必须显式填入数值；null模板不能判定通过")
                    if any(not isinstance(v, (int, float)) or v < 0 for v in criteria["state_rmse_limits"].values()):
                        raise ValueError("状态误差阈值必须显式为非负数值")
                names = ["l_knee_pitch_joint", "r_knee_pitch_joint", "l_arm_roll_joint"]
                indices = [actor.stage1.endecoder.kinematics.joint_order.index(n) for n in names]
                cases = []
                for task in tasks:
                    for seed in task.get("seeds", [42]):
                        seed_all(seed)
                        music, initial = task_inputs(task, adapter, args.torch_device)
                        latencies = profile.sample(engine.decisions) if profile else [scheduler.bootstrap_latency_ticks] * engine.decisions
                        if args.latency_scenario == "measured_worst":
                            if profile is None:
                                raise ValueError("实测最坏延迟验收必须已有完整部署profile")
                            latencies = [max(profile.samples)] * engine.decisions
                        item = audit_generated_reference(engine, collector, music, initial, noise_seed=seed,
                                                         latencies=latencies, joint_indices=indices, criteria=criteria,
                                                         epsilon=args.perturbation_epsilon)
                        item.update(task=task["id"], seed=seed, known_reference_passed=False)
                        known = task.get("known_reference_world_qpos28")
                        if known:
                            array = np.load(require_file(known, "已知参考qpos28"), allow_pickle=False)
                            reference = torch.as_tensor(array, device=args.torch_device, dtype=torch.float32)[None]
                            horizons = sorted(set([5, 10, 15, 20, 32, item["required_dependency_steps"]]))
                            item["known_reference"] = audit_known_reference(collector, music, initial, reference, horizons=horizons, criteria=criteria)
                            item["known_reference_passed"] = item["known_reference"]["passed"]
                        cases.append(item)
                        with (output / "model_physics_cases.jsonl").open("a") as stream:
                            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                        print(json.dumps({"event": "model_physics_case", "task": task["id"], "seed": seed,
                            "prediction_passed": item["prediction_passed"], "physical_direction_passed": item["physical_direction_passed"],
                            "known_reference_passed": item["known_reference_passed"],
                            "required_dependency_steps": item["required_dependency_steps"]}, ensure_ascii=False), flush=True)
                keys = ("prediction_passed", "physical_direction_passed", "reset_replay_repeatability_passed", "known_reference_passed")
                result = {k: all(item[k] for item in cases) for k in keys}
                from .validation import combine_envelopes
                envelopes = [item["real_state_envelope"] for item in cases]
                envelopes.extend(item["known_reference"]["real_state_envelope"] for item in cases if "known_reference" in item)
                result.update(cases=cases, frozen_sha256=adapter.bundle.metadata["sha256"], task_ids=[t["id"] for t in tasks],
                              coverage={"physics": "nominal_flat_newton_mujoco_warp",
                                        "tasks": [{k:t.get(k) for k in ("id", "source", "split", "distribution_id")} for t in tasks]},
                              validated_state_envelope=combine_envelopes(envelopes),
                              task_identities={t["id"]: task_identity(t) for t in tasks},
                              forecast_enabled=cfg["model"]["forecast_enabled"],
                              latency_scenario=args.latency_scenario,
                              scope="selected_tasks_seeds_and_nominal_physics_only",
                              supported_control_steps=min(item["supported_control_steps"] for item in cases) if all(result.values()) else 0)
                (output / "model_physics_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            elif kind == "train":
                losses = build_stage1_losses(actor.stage1, original).to(args.torch_device)
                trainer = Stage2Trainer(actor, critic, engine, losses, cfg["training"])
                if args.run_mode == "formal":
                    from .formal import run_formal
                    run_formal(args, cfg, output, trainer, collector, profile, tasks, report, source_info)
                    return
                short = cfg["imagination"].get("mode") == "physical_anchor_short_bptt"
                if short:
                    collector.anchor_horizon = engine.horizon_steps
                    collector.record_controls = True
                    collector.record_contacts = True
                    if cfg["formal"].get("critic_warmup_checkpoint") and not args.resume:
                        from .critic_warmup import load_warmup
                        load_warmup(cfg["formal"]["critic_warmup_checkpoint"], trainer, cfg)
                replay = RecentReplay(cfg["training"]["replay_capacity"], cfg["training"]["seed"],
                                      balanced=cfg["training"].get("balanced_motion_sampling",False))
                supervised = SupervisionStream(cfg["stage1"]["data_config"], cfg["training"]["supervision_batch"],
                                               cfg["training"]["seed"], args.torch_device)
                behavior = behavior_config(cfg, actor, adapter)
                task_cursor = 0
                if args.resume:
                    state = load_checkpoint(args.resume, trainer, replay, behavior_config=behavior, device=args.torch_device)
                    supervised.load_state_dict(state["supervision"])
                    profile.rng.setstate(state["latency_rng"])
                    task_cursor = state["task_cursor"]
                for _ in range(args.updates):
                    if collector.state is None or collector.terminal:
                        task = tasks[task_cursor % len(tasks)]
                        task_cursor += 1
                        music, initial = task_inputs(task, adapter, args.torch_device)
                        collector.reset(music, initial)
                    collector.actor_version = trainer.accepted_actor_updates
                    physical = collector.collect(cfg["training"]["collect_decisions"], replay=replay)
                    if short:
                        while replay.eligible_count(trainer.accepted_actor_updates) < cfg["training"]["actor_microbatch"] and not collector.terminal:
                            collector.collect(1, replay=replay)
                        physical = collector.take_controls()
                        states = replay.sample(cfg["training"]["actor_microbatch"], actor_version=trainer.accepted_actor_updates)
                        result = trainer.update_anchors(states, supervised.next(), physical_intervals=physical)
                    else:
                        states = replay.sample(cfg["training"]["actor_microbatch"])
                        latencies = [profile.sample(engine.decisions) for _ in states]
                        result = trainer.update(states, supervised.next(), latencies, physical_intervals=physical)
                    with (output / "training.jsonl").open("a") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    save_checkpoint(output / f"stage2_{trainer.update_index:06d}.pt", trainer, replay,
                                    behavior_config=behavior, runtime_state={"supervision": supervised.state_dict(),
                                    "latency_rng": profile.rng.getstate(), "task_cursor": task_cursor,
                                    "collector_restart": "new_episode_required_no_solver_snapshot"}, provenance=source_info)
                    print(json.dumps(result, ensure_ascii=False))
        finally:
            collector.close()
    if args.profile:
        (output / "performance.txt").write_text(measured.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
        measured.export_chrome_trace(str(output / "performance_trace.json"))
        (output / "gpu_memory.json").write_text(json.dumps({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                                            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}))
