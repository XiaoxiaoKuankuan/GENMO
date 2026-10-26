"""按有效执行反馈更新计数的正式Stage2运行器及独立物理评价管理。

入口先通过真实WM/梯度/基线合同，固定GENMO采集完整MC并验证独立Critic，然后才
优化GENMO。Actor更新只在采集请求全部收束后发生；每环境episode跨采集轮保持。
评价使用独立进程和固定音乐/初态/噪声种子，不修改训练对象，模型回报不用于挑选
物理best。完整checkpoint含两个优化器、回放、RNG、数据与任务游标；恢复从新物理
episode开始。资源预算、质量退化、数值错误均留证据退出，不无限重启质量失败。
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


def paired_quality(baseline, current, criteria):
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
    return {"completion_rate": completion, "baseline_completion_rate": old_completion, "duration_ratio": duration_ratio,
            "activity_ratio": activity_ratio, "beat_alignment_drop": beat_drop, "beat_available": beat_drop is not None,
            "physical_tracking_score": score, "degraded": degraded, "model_return_used": False}


def evaluate_checkpoint(cfg, output, checkpoint, *, update):
    directory = output / "evaluations"
    eval_cfg = dict(cfg, tasks=cfg["formal"]["evaluation_tasks"])
    config_path = directory / "fixed_evaluation.yaml"
    if not config_path.exists():
        OmegaConf.save(OmegaConf.create(eval_cfg), config_path)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["formal"]["evaluation_gpu"]), PYTHONDONTWRITEBYTECODE="1")
    outputs = {"initial": directory / "initial_genmo_baseline", "full": directory / f"actor_{update:06d}"}
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
        command = [sys.executable, "-u", "tools/evaluate_stage2_supermimic.py", "--config", str(config_path),
                   "--mode", model_mode, "--output", str(path)]
        model_checkpoint = output/"checkpoints"/"initial_actor.pt" if mode == "initial" else checkpoint
        command += ["--actor-checkpoint", str(model_checkpoint)]
        with (directory / (path.name+".log")).open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    baseline = json.loads((outputs["initial"] / "physical_summary.json").read_text())
    current = json.loads((outputs["full"] / "physical_summary.json").read_text())
    quality = paired_quality(baseline, current, cfg["formal"]["degradation"])
    write_json(directory / f"paired_{update:06d}.json", quality)
    return quality


def run_formal(args, cfg, output, trainer, collector, profile, tasks, report, source_info):
    from torch.utils.tensorboard import SummaryWriter
    settings = cfg["formal"]
    short = cfg["imagination"].get("mode") == "physical_anchor_short_bptt"
    if short:
        collector.anchor_horizon = trainer.imagination.horizon_steps
        collector.record_controls = True
    for name in ("logs", "checkpoints", "evaluations", "diagnostics", "tensorboard"):
        (output/name).mkdir(exist_ok=True)
    lock = output / "training.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    writer = SummaryWriter(str(output/"tensorboard"))
    stopping = []
    previous_handlers = {s: signal.signal(s, lambda signum, frame: stopping.append(signum)) for s in (signal.SIGTERM, signal.SIGINT)}
    replay = RecentReplay(cfg["training"]["replay_capacity"], cfg["training"]["seed"])
    supervised = SupervisionStream(cfg["stage1"]["data_config"], cfg["training"]["supervision_batch"], cfg["training"]["seed"], args.torch_device)
    behavior = behavior_config(cfg, trainer.actor, trainer.imagination.adapter)
    task_cursor, control_steps, total_wall_seconds = 0, 0, 0.
    best_score, degradation_count, last_evaluated = None, 0, 0
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
                "collector_restart": "new_episode_required_no_solver_snapshot"}, provenance=source_info)
        checkpoint_link(checkpoint_dir, "latest.pt", path)
        return path

    def run_evaluation(n):
        nonlocal best_score, degradation_count, last_evaluated
        checkpoint = snapshot()
        rng = capture_rng()
        try:
            quality = evaluate_checkpoint(cfg, output, checkpoint, update=n)
        finally:
            restore_rng(rng)
        last_evaluated = n
        degradation_count = degradation_count+1 if quality["degraded"] else 0
        score = quality["physical_tracking_score"]
        if not quality["degraded"] and (best_score is None or score > best_score):
            best_score = score
            checkpoint_link(checkpoint_dir, "best.pt", checkpoint)
        snapshot()
        if degradation_count >= settings["degradation"]["consecutive_evaluations"]:
            raise RuntimeError("独立真实物理连续评价退化，按预设规则暂停")

    try:
        resume = args.resume or (str(checkpoint_dir/"latest.pt") if args.resume_latest else None)
        if resume:
            state = load_checkpoint(resume, trainer, replay, behavior_config=behavior, device=args.torch_device)
            supervised.load_state_dict(state["supervision"])
            profile.rng.setstate(state["latency_rng"])
            task_cursor, control_steps = state["task_cursor"], state["physical_control_steps"]
            total_wall_seconds = state["total_wall_seconds"]
            best_score, degradation_count, last_evaluated = state["best_score"], state["degradation_count"], state["last_evaluated"]
        if not trainer.critic_ready and settings.get("critic_warmup_checkpoint"):
            quality = load_warmup(settings["critic_warmup_checkpoint"], trainer, cfg)
            control_steps += quality["physical_control_steps"]
            snapshot("critic_ready_imported.pt")
        if not trainer.critic_ready:
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
        trainer.imagination.support_guard = ModelSupport(report["validated_state_envelope"], device=args.torch_device)
        if not (checkpoint_dir/"initial_actor.pt").is_file():
            if trainer.accepted_actor_updates != 0:
                raise ValueError("缺少更新前Actor，不能用已训练权重伪造配对基线")
            snapshot("initial_actor.pt")
        if trainer.accepted_actor_updates > last_evaluated and (trainer.accepted_actor_updates == 1 or trainer.accepted_actor_updates % settings["evaluate_every"] == 0):
            run_evaluation(trainer.accepted_actor_updates)
        invalid_streak = 0
        while trainer.accepted_actor_updates < args.max_actor_updates:
            elapsed = total_wall_seconds + time.perf_counter()-started
            if stopping or control_steps >= settings["max_control_steps"] or elapsed >= settings["max_wall_hours"]*3600:
                write_json(output/"status.json", {"status": "stopped_at_signal_or_resource_budget", "signals": stopping,
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
            physical = collector.collect(cfg["training"]["collect_decisions"], replay=replay)
            control_steps += sum(x.steps for x in physical)
            collect_seconds = time.perf_counter()-collect_started
            if collector.worker.future is not None:
                raise RuntimeError("采集Actor仍在生成，禁止并发修改其权重")
            if short:
                physical = collector.take_controls()
                if replay.eligible_count(trainer.accepted_actor_updates) == 0:
                    # 正在积累H个真实执行步，不把尚未完成的anchor计作训练失败或收益。
                    print(json.dumps({"event": "waiting_for_real_anchor", "physical_control_steps": control_steps,
                                      "actor_version": trainer.accepted_actor_updates}), flush=True)
                    continue
                states = replay.sample(cfg["training"]["actor_microbatch"], actor_version=trainer.accepted_actor_updates)
            else:
                states = replay.sample(cfg["training"]["actor_microbatch"])
                latencies = [profile.sample(trainer.imagination.decisions) for _ in states]
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
            row = (trainer.update_anchors(states, supervised.next(), physical_intervals=physical) if short else
                   trainer.update(states, supervised.next(), latencies, physical_intervals=physical))
            gpu_end.record()
            gpu_end.synchronize()
            row.update(physical_control_steps=control_steps, collection_wall_seconds=collect_seconds,
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
                                               "physical_control_steps": control_steps})
            accepted = row["accepted_execution_feedback_update"]
            invalid_streak = 0 if accepted else invalid_streak+1
            if invalid_streak >= settings["maximum_invalid_streak"]:
                raise RuntimeError("连续没有合法执行反馈更新，暂停保留证据")
            n = trainer.accepted_actor_updates
            if accepted and (n == 1 or n % settings["checkpoint_every"] == 0):
                checkpoint = snapshot()
            if accepted and (n == 1 or n % settings["evaluate_every"] == 0) and n > last_evaluated:
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
        writer.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock.unlink(missing_ok=True)
