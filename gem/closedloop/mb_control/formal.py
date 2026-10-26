"""按有效执行反馈更新计数的正式Stage2运行器及独立物理评价管理。

入口先通过真实WM/梯度/基线合同，固定GENMO采集完整MC并验证独立Critic，然后才
优化GENMO。Actor更新只在采集请求全部收束后发生；每环境episode跨采集轮保持。
评价使用独立进程和固定音乐/初态/噪声种子，不修改训练对象，模型回报不用于挑选
物理best。完整checkpoint含两个优化器、回放、RNG、数据与任务游标；恢复从新物理
episode开始。质量变化仅记录和展示，不因退化退出；资源预算及数值错误仍留证据。
"""

import json
import os
import random
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf

from .application import task_inputs, read_tasks, behavior_config, SupervisionStream
from .checkpoint import save_checkpoint, load_checkpoint, capture_rng, restore_rng
from .critic_warmup import warmup_critic, load_warmup, choose_warmup_tasks
from .replay import RecentReplay
from .validation import ModelSupport


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


def checkpoint_link(directory, name, target):
    destination = directory / name
    temporary = directory / (name + ".tmp")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target.name)
    os.replace(temporary, destination)


def paired_quality(baseline, current, criteria, tasks=None):
    before = {(r["task"], r["seed"]): r for r in baseline["physical_results"]}
    after = {(r["task"], r["seed"]): r for r in current["physical_results"]}
    if before.keys() != after.keys():
        raise ValueError("真实评价音乐及随机种子必须配对")
    rows = list(after.values())
    completion = sum(r["complete"] for r in rows) / len(rows)
    old_completion = sum(r["complete"] for r in before.values()) / len(before)
    duration_ratio = sum(r["physical_seconds"] for r in rows) / max(1e-9, sum(r["physical_seconds"] for r in before.values()))
    def mean(values, key):
        xs = [r["motion"].get(key) for r in values if r["motion"].get(key) is not None]
        return float(np.mean(xs)) if xs else None
    activity, old_activity = mean(rows, "joint_range_mean_rad"), mean(before.values(), "joint_range_mean_rad")
    beat, old_beat = mean(rows, "beat_alignment"), mean(before.values(), "beat_alignment")
    activity_ratio = activity/max(old_activity, 1e-9) if activity is not None and old_activity is not None else None
    beat_drop = old_beat-beat if old_beat is not None and beat is not None else None
    degraded = (completion < old_completion-criteria["completion_drop"] or duration_ratio < criteria["minimum_duration_ratio"] or
                activity_ratio is None or activity_ratio < criteria["minimum_activity_ratio"] or
                (beat_drop is not None and beat_drop > criteria["maximum_beat_drop"]))
    score = float(np.mean([r["physical_metrics_mean"].get("upper_eight_tracking", 0.) for r in rows]))
    old_score = float(np.mean([r["physical_metrics_mean"].get("upper_eight_tracking", 0.) for r in before.values()]))
    metadata = {t["id"]: t for t in tasks or []}
    comparisons = []
    for key, new in after.items():
        old = before[key]
        if old.get("timing_mode","wall_clock_async")!=new.get("timing_mode","wall_clock_async"):
            raise ValueError("固定外生事件回放与真实墙钟异步评价不能作为同一种配对比较")
        if old.get("input_identity")!=new.get("input_identity") or old.get("noise_seed")!=new.get("noise_seed"):
            raise ValueError("物理比较的音乐/初态资产或DDIM噪声身份不一致")
        def values(row):
            metrics = row["physical_metrics_mean"]
            return {"tracking": metrics["upper_eight_tracking"],
                    "original_music_execution_loss":metrics.get("dance_loss"),
                    "current_reference_tracking_loss":metrics.get("current_reference_loss"),
                    "physical_cost":metrics.get("physical_cost"),
                    "root_position_rmse_m": float(np.sqrt(metrics["error/anchor_pos"])),
                    "root_xy_rmse_m": float(np.sqrt(metrics["root_xy_error_sq"])),
                    "root_orientation_rmse_rad": float(np.sqrt(metrics["error/anchor_ori"])),
                    "joint_range_mean_rad": row["motion"]["joint_range_mean_rad"],
                    "joint_speed_rms_rad_s": row["motion"]["joint_speed_rms_rad_s"],
                    "beat_alignment": row["motion"].get("beat_alignment")}
        old_values, new_values = values(old), values(new)
        from .dance_evaluation import common_window,paired_delivery_context
        meta = metadata.get(key[0], new)
        comparisons.append({"task": key[0], "seed": key[1], "source": meta.get("source"), "split": meta.get("split"),
            "baseline_complete": old["complete"], "current_complete": new["complete"],
            "baseline_reason": old["reason"], "current_reason": new["reason"],
            "baseline": old_values, "current": new_values,
            "delta": {k: new_values[k]-v if v is not None and new_values[k] is not None else None for k,v in old_values.items()},
            "baseline_contacts": old.get("contacts"), "current_contacts": new.get("contacts"),
            "delivery_pairing":paired_delivery_context(old,new),
            "common_valid_window":common_window(old,new),"baseline_music_content":old.get("music_content"),
            "current_music_content":new.get("music_content"),
            "contact_pair_available": old.get("contacts") is not None and new.get("contacts") is not None})
    source_split = []
    for group in sorted({(str(r["source"]), str(r["split"])) for r in comparisons}):
        selected = [r for r in comparisons if (str(r["source"]), str(r["split"])) == group]
        source_split.append({"source": group[0], "split": group[1], "episodes": len(selected),
            "baseline_completed": sum(r["baseline_complete"] for r in selected),
            "current_completed": sum(r["current_complete"] for r in selected),
            "mean_tracking_delta": float(np.mean([r["delta"]["tracking"] for r in selected])),
            "mean_root_position_rmse_delta_m": float(np.mean([r["delta"]["root_position_rmse_m"] for r in selected]))})
    teacher_losses=[r["physical_metrics_mean"]["dance_loss"] for r in rows if "dance_loss" in r["physical_metrics_mean"]]
    baseline_teacher=[r["physical_metrics_mean"]["dance_loss"] for r in before.values() if "dance_loss" in r["physical_metrics_mean"]]
    content_reasons=[]
    if teacher_losses and baseline_teacher and np.mean(teacher_losses)>=np.mean(baseline_teacher):
        content_reasons.append({"task":None,"reason":"fixed_music_objective_not_improved_over_original_stage1"})
    if any(r.get("music_content") for r in rows):
        for key,new in after.items():
            old=before[key]
            if (old["complete"] and not new["complete"]) or new["physical_seconds"]+1e-6<old["physical_seconds"]:
                content_reasons.append({"task":key[0],"reason":"shorter_or_new_failed_episode_cannot_improve_selection"})
            for group in ("upper","lower"):
                prior=old.get("music_content",{}).get(group,{});now=new.get("music_content",{}).get(group,{})
                for metric in ("actual_range_rad","actual_speed_rad_s"):
                    if prior.get(metric,0.)>.001 and now.get(metric,0.)<criteria["minimum_activity_ratio"]*prior[metric]:
                        content_reasons.append({"task":key[0],"reason":group+"_"+metric+"_degraded"})
                old_phase=prior.get("velocity_phase_cosine");phase=now.get("velocity_phase_cosine")
                if old_phase is not None and phase is not None and phase<old_phase-criteria["maximum_beat_drop"]:
                    content_reasons.append({"task":key[0],"reason":group+"_motion_phase_degraded"})
    return {"completion_rate": completion, "baseline_completion_rate": old_completion, "duration_ratio": duration_ratio,
            "music_content_selection_eligible":not content_reasons,"music_content_selection_reasons":content_reasons,
            "physical_music_execution_loss":float(np.mean(teacher_losses)) if teacher_losses else None,
            "baseline_physical_music_execution_loss":float(np.mean(baseline_teacher)) if baseline_teacher else None,
            "activity_ratio": activity_ratio, "beat_alignment_drop": beat_drop, "beat_available": beat_drop is not None,
            "physical_tracking_score": score, "baseline_physical_tracking_score": old_score,
            "physical_tracking_delta": score-old_score, "paired_tasks": comparisons, "source_split": source_split,
            "degraded": degraded, "quality_exit_enabled": False, "model_return_used": False}


def run_evaluation_processes(cfg, config_path, checkpoint, path, model_mode, environment, *, cancel_requested=None):
    """固定任务按完整歌曲分到独立GPU，保持每个机器人的物理和请求时序独立。"""
    gpus = cfg["formal"].get("evaluation_gpus", [cfg["formal"]["evaluation_gpu"]])
    if len(gpus) == 1:
        command = [sys.executable, "-u", "tools/evaluate_stage2_supermimic.py", "--config", str(config_path),
                   "--mode", model_mode, "--output", str(path), "--actor-checkpoint", str(checkpoint)]
        with path.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(command, env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpus[0])),
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if cancel_requested and cancel_requested():
                        raise InterruptedError("收到停止请求，结束独立评价并回到完整检查点边界")
                    time.sleep(.2)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=20)
        return
    if not 1 <= len(gpus) <= 8 or len(set(gpus)) != len(gpus):
        raise ValueError("独立固定评价支持1至8张不同GPU")
    path.mkdir(exist_ok=False)
    tasks = read_tasks(cfg["formal"]["evaluation_tasks"])
    gpus = gpus[:len(tasks)]
    workers, logs, manifest = [], [], []
    try:
        for index, gpu in enumerate(gpus):
            selected = tasks[index::len(gpus)]
            task_file, config_file = path/f"tasks_{index}.json", path/f"config_{index}.yaml"
            write_json(task_file, selected)
            OmegaConf.save(OmegaConf.create(dict(cfg, tasks=str(task_file))), config_file)
            log = (path/f"shard_{index}.log").open("w")
            logs.append(log)
            command = [sys.executable, "-u", "tools/evaluate_stage2_supermimic.py", "--config", str(config_file),
                "--mode", model_mode, "--output", str(path/f"shard_{index}"), "--actor-checkpoint", str(checkpoint)]
            process = subprocess.Popen(command, env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpu)),
                                       stdout=log, stderr=subprocess.STDOUT)
            workers.append(process)
            manifest.append({"gpu": gpu, "pid": process.pid, "tasks": [t["id"] for t in selected], "command": command})
        write_json(path/"processes.json", manifest)
        while any(p.poll() is None for p in workers):
            if cancel_requested and cancel_requested():
                raise InterruptedError("收到停止请求，结束独立评价并回到完整检查点边界")
            if any(p.poll() not in (None, 0) for p in workers):
                raise RuntimeError(f"独立物理评价子进程失败，原日志保留在{path}")
            time.sleep(.2)
        if any(p.returncode for p in workers):
            raise RuntimeError("独立物理评价未正常结束")
        summaries = [json.loads((path/f"shard_{i}"/"physical_summary.json").read_text()) for i in range(len(gpus))]
        rows = [r for s in summaries for r in s["physical_results"]]
        expected = {(t["id"], seed) for t in tasks for seed in t.get("seeds", [42])}
        if len(rows) != len(expected) or {(r["task"], r["seed"]) for r in rows} != expected:
            raise RuntimeError("并行物理评价遗漏或重复了固定任务/种子")
        summary = {"mode": model_mode, "episodes": len(rows), "completed": sum(r["complete"] for r in rows),
            "physical_results": rows, "diversity": {k:v for s in summaries for k,v in s["diversity"].items()},
            "model_reward": None, "quality_improvement_claim": False, "physical_processes": manifest}
        write_json(path/"physical_summary.json", summary)
        with (path/"physical_results.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    finally:
        for process in workers:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
        for log in logs:
            log.close()


def _evaluate_checkpoint_mode(cfg, output, checkpoint, *, update, directory, cancel_requested=None):
    directory.mkdir(parents=True,exist_ok=True)
    eval_cfg = dict(cfg, tasks=cfg["formal"]["evaluation_tasks"])
    config_path = directory / "fixed_evaluation.yaml"
    if not config_path.exists():
        OmegaConf.save(OmegaConf.create(eval_cfg), config_path)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["formal"]["evaluation_gpu"]), PYTHONDONTWRITEBYTECODE="1")
    outputs = {"initial": directory / "initial_genmo_baseline", "full": directory / f"actor_{update:06d}"}
    if cfg["formal"].get("initial_physical_baseline"):
        from .evaluation import evaluation_contract
        initial_path = Path(cfg["formal"]["initial_physical_baseline"])
        identity = torch.load(initial_path/"evaluation_actor.pt", map_location="cpu", weights_only=False)
        baseline_actor = torch.load(output/"checkpoints"/"initial_actor.pt", map_location="cpu", weights_only=False)
        contract = evaluation_contract(cfg, read_tasks(cfg["formal"]["evaluation_tasks"]), baseline_actor["frozen_sha256"])
        if identity.get("schema") != "genmo.evaluation.initial_actor.v1" or identity.get("accepted_actor_updates") != 0 or identity["contract"] != contract:
            raise ValueError("预先评价的更新前基线与本次物理、任务或条件合同不一致")
        if identity["actor"].keys() != baseline_actor["actor"].keys() or any(
                not torch.equal(value, baseline_actor["actor"][name]) for name, value in identity["actor"].items()):
            raise ValueError("预先评价的基线不属于本次更新前GENMO张量")
        if not (initial_path/"physical_summary.json").is_file():
            raise ValueError("预先评价尚未完整结束，不能使用部分结果作为固定基线")
        outputs["initial"] = initial_path
    for mode, path in outputs.items():
        if (path / "physical_summary.json").is_file():
            continue
        # 半次失败评价保留现场；重试写新目录，不覆盖原始失败证据。
        if path.exists():
            path = path.with_name(path.name + f"_retry_{time.time_ns()}")
            outputs[mode] = path
        # 配对基线也运行相同Forecast/DDIM/异步路径，仅GENMO参数取更新前版本。
        # 原Stage1无前瞻全曲能力由独立preflight报告验收，不能混入这里比较训练收益。
        model_mode = "full" if cfg["model"]["forecast_enabled"] else "no_forecast"
        model_checkpoint = output/"checkpoints"/"initial_actor.pt" if mode == "initial" else checkpoint
        run_evaluation_processes(cfg, config_path, model_checkpoint, path, model_mode, environment,
                                 cancel_requested=cancel_requested)
    baseline = json.loads((outputs["initial"] / "physical_summary.json").read_text())
    current = json.loads((outputs["full"] / "physical_summary.json").read_text())
    quality = paired_quality(baseline, current, cfg["formal"]["degradation"], read_tasks(cfg["formal"]["evaluation_tasks"]))
    write_json(directory / f"paired_{update:06d}.json", quality)
    return quality


def evaluate_checkpoint(cfg, output, checkpoint, *, update, cancel_requested=None):
    """固定事件主对照与墙钟异步独立保存，禁止混合均值选best。"""
    modes=cfg["formal"].get("evaluation_timing_modes")
    if not modes:
        return _evaluate_checkpoint_mode(cfg,output,checkpoint,update=update,directory=output/"evaluations",
                                        cancel_requested=cancel_requested)
    if modes != ["fixed_delay_replay","wall_clock_async"]:
        raise ValueError("双模式评价需以固定外生事件为主对照，并独立保留真实墙钟结果")
    qualities={}
    for mode in modes:
        per_mode=dict(cfg,evaluation=dict(cfg.get("evaluation",{}),timing_mode=mode),
            formal=dict(cfg["formal"],initial_physical_baseline=cfg["formal"].get("initial_physical_baselines",{}).get(mode)))
        qualities[mode]=_evaluate_checkpoint_mode(per_mode,output,checkpoint,update=update,
            directory=output/"evaluations"/mode,cancel_requested=cancel_requested)
    primary=dict(qualities["fixed_delay_replay"],selection_timing_mode="fixed_delay_replay",
        wall_clock_async_quality=qualities["wall_clock_async"])
    write_json(output/"evaluations"/f"paired_{update:06d}.json",primary)
    return primary


def run_formal(args, cfg, output, trainer, collector, profile, tasks, report, source_info):
    from torch.utils.tensorboard import SummaryWriter
    settings = cfg["formal"]
    short = cfg["imagination"].get("mode") == "physical_anchor_short_bptt"
    if short:
        collector.anchor_horizon = trainer.imagination.horizon_steps
        collector.record_controls = True
        collector.record_contacts = True
    for name in ("logs", "checkpoints", "evaluations", "diagnostics", "tensorboard"):
        (output/name).mkdir(exist_ok=True)
    lock = output / "training.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    writer = SummaryWriter(str(output/"tensorboard"))
    stopping = []
    previous_handlers = {s: signal.signal(s, lambda signum, frame: stopping.append(signum)) for s in (signal.SIGTERM, signal.SIGINT)}
    replay = RecentReplay(cfg["training"]["replay_capacity"], cfg["training"]["seed"],
                          balanced=cfg["training"].get("balanced_motion_sampling", False))
    supervised = SupervisionStream(cfg["stage1"]["data_config"], cfg["training"]["supervision_batch"], cfg["training"]["seed"], args.torch_device)
    behavior = behavior_config(cfg, trainer.actor, trainer.imagination.adapter)
    task_cursor, control_steps, total_wall_seconds = 0, 0, 0.
    best_score, degradation_count, last_evaluated = None, 0, 0
    remotes, restored_worker_rng = [], {}
    published_version = -1
    started = time.perf_counter()
    checkpoint_dir = output/"checkpoints"

    def snapshot(tag=None):
        path = checkpoint_dir / (tag or f"actor_{trainer.accepted_actor_updates:06d}.pt")
        if path.exists():
            path = path.with_stem(path.stem + f"_state_{time.time_ns()}")
        save_checkpoint(path, trainer, replay, behavior_config=behavior,
            runtime_state={"supervision": supervised.state_dict(), "latency_rng": profile.rng.getstate(),
                "task_cursor": task_cursor, "physical_control_steps": control_steps,
                "total_wall_seconds": total_wall_seconds + time.perf_counter()-started,
                "best_score": best_score, "degradation_count": degradation_count, "last_evaluated": last_evaluated,
                "physical_worker_rng": {str(w.collector_id): w.rng for w in remotes},
                "collector_restart": "new_episode_required_no_solver_snapshot"}, provenance=source_info)
        checkpoint_link(checkpoint_dir, "latest.pt", path)
        return path

    def run_evaluation(n):
        nonlocal best_score, degradation_count, last_evaluated
        if collector.worker.future is not None or any(w.request_id is not None for w in remotes):
            raise RuntimeError("采集尚未收束，不允许评价复用GPU")
        checkpoint = snapshot()
        rng = capture_rng()
        try:
            quality = evaluate_checkpoint(cfg, output, checkpoint, update=n, cancel_requested=lambda: bool(stopping))
        except Exception as error:
            # 独立评价的负结果、子进程故障或主动中断都不能杀死已合法更新的Actor。
            # 原始日志与失败现场保留，下一评价周期继续尝试；不把失败当成合格best。
            write_json(output/"diagnostics"/f"evaluation_failure_{n}_{time.time_ns()}.json",
                       {"error": repr(error), "traceback": traceback.format_exc(),
                        "accepted_actor_updates": n, "training_continues": not bool(stopping)})
            last_evaluated = n
            print(json.dumps({"event": "evaluation_incomplete_training_continues", "update": n,
                              "error": repr(error)}), flush=True)
            return
        finally:
            restore_rng(rng)
        last_evaluated = n
        degradation_count = degradation_count+1 if quality["degraded"] else 0
        score = (-quality["physical_music_execution_loss"] if cfg.get("teacher",{}).get("enabled")
                 else quality["physical_tracking_score"])
        if best_score is None and cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1":
            best_score=-quality["baseline_physical_music_execution_loss"]
            checkpoint_link(checkpoint_dir,"best.pt",checkpoint_dir/"initial_actor.pt")
        if not quality["degraded"] and quality.get("music_content_selection_eligible",True) and (best_score is None or score > best_score):
            best_score = score
            checkpoint_link(checkpoint_dir, "best.pt", checkpoint)
        snapshot()
        # 2026-10-26 用户要求：保留完整评价和连续退化记录，评价结果不终止训练。

    try:
        resume = args.resume or (str(checkpoint_dir/"latest.pt") if args.resume_latest else None)
        if resume:
            state = load_checkpoint(resume, trainer, replay, behavior_config=behavior, device=args.torch_device)
            supervised.load_state_dict(state["supervision"])
            profile.rng.setstate(state["latency_rng"])
            task_cursor, control_steps = state["task_cursor"], state["physical_control_steps"]
            total_wall_seconds = state["total_wall_seconds"]
            best_score, degradation_count, last_evaluated = state["best_score"], state["degradation_count"], state["last_evaluated"]
            restored_worker_rng = state.get("physical_worker_rng", {})
        defer_critic=cfg["training"].get("defer_critic",False)
        if defer_critic and cfg["training"].get("terminal_value_weight",0.)!=0:raise ValueError("未验证新Critic不得进入Actor")
        if not defer_critic and not trainer.critic_ready and settings.get("critic_warmup_checkpoint"):
            quality = load_warmup(settings["critic_warmup_checkpoint"], trainer, cfg)
            control_steps += quality["physical_control_steps"]
            snapshot("critic_ready_imported.pt")
        if not defer_critic and not trainer.critic_ready:
            validation = read_tasks(settings["critic_validation_tasks"])
            groups = {}
            for task in tasks:
                groups.setdefault(task["source"], []).append(task)
            randomizer = random.Random(settings["critic_warmup"]["seed"])
            for values in groups.values():
                randomizer.shuffle(values)
            chosen = []
            while len(chosen) < settings["critic_warmup"]["train_episodes"] and any(groups.values()):
                for source in sorted(groups):
                    if groups[source] and len(chosen) < settings["critic_warmup"]["train_episodes"]:
                        chosen.append(groups[source].pop())
            quality = warmup_critic(trainer, collector, chosen, validation, settings["critic_warmup"],
                                    output/"diagnostics"/f"critic_warmup_{time.time_ns()}", device=args.torch_device)
            control_steps += quality["physical_control_steps"]
            snapshot("critic_ready.pt")
        worker_gpus = cfg.get("execution", {}).get("physical_worker_gpus", [])
        if worker_gpus:
            if not short or len(worker_gpus) > 7 or len(set(worker_gpus)) != len(worker_gpus):
                raise ValueError("并行支持短时真实anchor及最多八个独立物理采集器")
            from .parallel_collection import RemotePhysicalCollector
            directory = output/"collectors"
            directory.mkdir(exist_ok=True)
            worker_cfg = directory/"resolved_config.yaml"
            OmegaConf.save(OmegaConf.create(cfg), worker_cfg)
            for index, gpu in enumerate(worker_gpus, 1):
                remote = RemotePhysicalCollector(cfg, worker_cfg, directory/f"worker_{index}_{time.time_ns()}",
                    gpu, index, restored_rng=restored_worker_rng.get(str(index)))
                remotes.append(remote)
            for remote in remotes:
                remote.wait_ready(trainer.imagination.adapter.bundle.metadata["sha256"])
            print(json.dumps({"event": "parallel_physical_collectors_ready", "environments": 1+len(remotes),
                "worker_pids": [w.process.pid for w in remotes]}), flush=True)
        from .validation import training_support_envelope
        support_envelope=training_support_envelope(cfg,report)
        write_json(output/"state_support_contract.json",support_envelope)
        trainer.imagination.support_guard = ModelSupport(support_envelope, device=args.torch_device)
        if not (checkpoint_dir/"initial_actor.pt").is_file():
            if trainer.accepted_actor_updates != 0:
                raise ValueError("缺少更新前Actor，不能用已训练权重伪造配对基线")
            snapshot("initial_actor.pt")
        if trainer.accepted_actor_updates > last_evaluated and ((trainer.accepted_actor_updates == 1 and settings.get("evaluate_first_update", True)) or trainer.accepted_actor_updates % settings["evaluate_every"] == 0):
            run_evaluation(trainer.accepted_actor_updates)
        invalid_streak = 0
        while trainer.accepted_actor_updates < args.max_actor_updates:
            elapsed = total_wall_seconds + time.perf_counter()-started
            from .parallel_layout import resource_stop_reason
            budget_reason = resource_stop_reason(settings, control_steps, elapsed)
            if stopping or budget_reason:
                write_json(output/"status.json", {"status": "stopped_at_signal_or_resource_budget", "signals": stopping,
                                                   "resource_stop_reason": budget_reason,
                                                   "accepted_actor_updates": trainer.accepted_actor_updates})
                snapshot()
                return
            if collector.state is None or collector.terminal:
                task = tasks[task_cursor % len(tasks)]
                task_cursor += 1
                music, initial = task_inputs(task, collector.adapter, args.torch_device)
                collector.reset(music, initial)
            collector.actor_version = trainer.accepted_actor_updates
            collect_started = time.perf_counter()
            if remotes:
                from .parallel_collection import publish_tensor
                weights_path = output/"collectors"/"current_actor.pt"
                if published_version != trainer.accepted_actor_updates:
                    publish_tensor({"schema": "genmo.collection.actor.v1", "actor": trainer.actor.state_dict(),
                        "actor_version": trainer.accepted_actor_updates,
                        "frozen_sha256": trainer.imagination.adapter.bundle.metadata["sha256"]}, weights_path)
                    published_version = trainer.accepted_actor_updates
                for remote in remotes:
                    task = None
                    if remote.needs_task:
                        task = tasks[task_cursor % len(tasks)]
                        task_cursor += 1
                    remote.submit(weights_path, trainer.accepted_actor_updates, cfg["training"]["collect_decisions"], task)
            physical = collector.collect(cfg["training"]["collect_decisions"], replay=replay)
            control_steps += sum(x.steps for x in physical)
            if collector.worker.future is not None:
                raise RuntimeError("采集Actor仍在生成，禁止并发修改其权重")
            if short:
                physical = collector.take_controls()
                sample_funnel=collector.take_distribution()
                physical_groups = [physical]
                per_collector = [{"collector_id": 0, "physical_steps": len(physical), "tick": collector.state.tick}]
                remote_timings = []
                for remote in remotes:
                    response = remote.receive(trainer.imagination.adapter, args.torch_device)
                    if response["actor_version"] != trainer.accepted_actor_updates:
                        raise RuntimeError("并行采集响应不是本轮当前Actor")
                    for anchor in response["anchors"]:
                        replay.add(anchor)
                    physical_groups.append(list(response["controls"]))
                    for key,values in response.get("sample_funnel",{}).items():sample_funnel.setdefault(key,[]).extend(values)
                    control_steps += response["physical_steps"]
                    remote_timings.append(response["seconds"])
                    per_collector.append({"collector_id": remote.collector_id,
                        "physical_steps": response["physical_steps"], "anchors": len(response["anchors"]),
                        "tick": response["tick"], "seconds": response["seconds"]})
                physical = [step for group in physical_groups for step in group]
                if cfg["training"].get("output_trust_region",{}).get("enabled"):
                    sample_funnel["BC_support_classification"]=replay.classify_support(trainer.imagination,trainer.accepted_actor_updates)
                    from .sample_distribution import coverage_summary
                    sample_funnel["legal_pool_coverage"]=coverage_summary([r for r in sample_funnel["BC_support_classification"] if r["legal"]])
                    with (output/"sampling_funnel.jsonl").open("a") as stream:
                        stream.write(json.dumps({"actor_version":trainer.accepted_actor_updates,**sample_funnel},ensure_ascii=False)+"\n")
                if replay.eligible_count(trainer.accepted_actor_updates) < cfg["training"]["actor_microbatch"]:
                    # 正在积累H个真实执行步，不把尚未完成的anchor计作训练失败或收益。
                    print(json.dumps({"event": "waiting_for_real_anchor", "physical_control_steps": control_steps,
                                      "eligible_anchors": replay.eligible_count(trainer.accepted_actor_updates),
                                      "required_anchors": cfg["training"]["actor_microbatch"],
                                      "actor_version": trainer.accepted_actor_updates}), flush=True)
                    write_json(output/"status.json", {"status": "collecting_real_anchors", "pid": os.getpid(),
                        **trainer.counters(), "physical_control_steps": control_steps,
                        "physical_collectors": 1+len(remotes), "per_collector": per_collector})
                    continue
                states = replay.sample(cfg["training"]["actor_microbatch"], actor_version=trainer.accepted_actor_updates)
                candidate_distribution = None
                if cfg.get("teacher", {}).get("enabled"):
                    from .sample_distribution import describe_sample
                    candidate_distribution = [describe_sample(x,trainer.imagination.adapter) for x in replay.items
                        if x.generation.request.actor_version == trainer.accepted_actor_updates]
            else:
                states = replay.sample(cfg["training"]["actor_microbatch"])
                latencies = [profile.sample(trainer.imagination.decisions) for _ in states]
            collect_seconds = time.perf_counter()-collect_started
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
            row = (trainer.update_anchors(states, supervised.next(), physical_intervals=physical,
                                          physical_interval_groups=physical_groups) if short else
                   trainer.update(states, supervised.next(), latencies, physical_intervals=physical))
            gpu_end.record()
            gpu_end.synchronize()
            if short and candidate_distribution is not None:
                row["real_candidate_distribution"] = candidate_distribution
                row["sample_funnel"] = sample_funnel
            row.update(physical_control_steps=control_steps, collection_wall_seconds=collect_seconds,
                       physical_collectors=1+len(remotes), remote_collection_seconds=remote_timings if short else [],
                       per_collector=per_collector if short else [],
                       drawn_collector_ids=[sample.collector_id for sample in states] if short else [],
                       collection_policy_version=trainer.accepted_actor_updates-int(row["accepted_execution_feedback_update"]),
                       cuda_event_interval_ms=gpu_start.elapsed_time(gpu_end),
                       cuda_event_note="事件间隔含CPU调度间隙，不冒充纯GPU内核活跃时间",
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       accepted_updates_per_hour=trainer.accepted_actor_updates/max(1e-9, elapsed)*3600)
            with (output/"training.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            for key,value in row.items():
                if isinstance(value, (int,float)):
                    writer.add_scalar(key, value, trainer.update_index)
            writer.flush()
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            write_json(output/"status.json", {"status": "running", "pid": os.getpid(), **trainer.counters(),
                                               "physical_control_steps": control_steps,
                                               "physical_collectors": 1+len(remotes),
                                               "quality_exit_enabled": False})
            accepted = row["accepted_execution_feedback_update"]
            # 新实验拒绝候选意味着继续采集，不把音乐/输出验收失败变成质量退化退出。
            finite_feedback_rejection=("output_trust_region" in row and row.get("valid_execution_rollouts",0)>0 and
                "nonfinite_candidate" not in row.get("rejection_reasons",[]))
            invalid_streak = 0 if accepted or finite_feedback_rejection else invalid_streak+1
            report_every = settings.get("maximum_invalid_streak", 20) or 20
            if invalid_streak and invalid_streak % max(1, report_every) == 0:
                print(json.dumps({"event": "invalid_update_streak_continuing_collection",
                    "invalid_streak": invalid_streak, "accepted_actor_updates": trainer.accepted_actor_updates,
                    "quality_exit_enabled": False}), flush=True)
            n = trainer.accepted_actor_updates
            if accepted and (n == 1 or n % settings["checkpoint_every"] == 0):
                checkpoint = snapshot()
            if accepted and ((n == 1 and settings.get("evaluate_first_update", True)) or n % settings["evaluate_every"] == 0) and n > last_evaluated:
                run_evaluation(n)
        snapshot()
        write_json(output/"status.json", {"status": "target_reached", "accepted_actor_updates": trainer.accepted_actor_updates})
    except BaseException as error:
        write_json(output/"diagnostics"/f"failure_{time.time_ns()}.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "counters": trainer.counters(), "physical_control_steps": control_steps, "automatic_quality_restart": False})
        write_json(output/"status.json", {"status": "failed", "error": repr(error), "accepted_actor_updates": trainer.accepted_actor_updates})
        # 仅保留之前已完成的完整latest；数值错误后的部分优化状态不冒充可恢复checkpoint。
        raise
    finally:
        for remote in remotes:
            remote.close()
        writer.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock.unlink(missing_ok=True)
