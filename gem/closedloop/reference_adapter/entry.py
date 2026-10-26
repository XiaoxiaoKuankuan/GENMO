"""独立train/evaluate/audit/prepare命令行入口。

默认有界且必须显式执行；只读help和self-test不初始化物理。真实路径先绑定资产，
独占新输出目录，然后复用SM原StreamingPhysicalEnv，按完整episode采集再更新/审计，
不因采集分段reset。恢复从新物理episode开始；正式训练必须通过真实门槛，按绝对accepted目标和分层审计执行。
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main(kind):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/closedloop/stage2_supermimic_adapter.yaml")
    parser.add_argument("--output", required=True, help="必须为不存在的新目录，恢复也使用新目录")
    parser.add_argument("--torch-device", default="cpu" if kind == "audit" else "cuda:0")
    parser.add_argument("--tasks")
    parser.add_argument("--music-baseline")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--variant", choices=("zero", "no_music", "music"), default="zero" if kind == "evaluate" else "music")
    parser.add_argument("--updates", type=int, default=1, help="有界测试的绝对目标，正式/恢复使用target-updates")
    parser.add_argument("--target-updates", type=int, help="绝对accepted更新目标，恢复不追加预算")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--physical", action="store_true", help="audit显式运行8/16步真实物理审计")
    parser.add_argument("--self-test", action="store_true", help="audit运行CPU合同测试，无物理或优化任务")
    parser.add_argument("--calibrate-music", action="store_true", help="train仅采集零Adapter真实速度，保存固定s0")
    parser.add_argument("--max-steps", type=int, help="仅评价允许的行政截断，不宣称歌曲完成")
    parser.add_argument("--compare", nargs=3, metavar=("ZERO_JSON", "NO_MUSIC_JSON", "MUSIC_JSON"))
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    cfg0 = OmegaConf.load(preliminary.config)
    sys.path.insert(0, str(Path(cfg0.supermimic.repo).expanduser().resolve()))
    physical = kind in ("train", "evaluate", "prepare") or preliminary.physical
    if preliminary.self_test or preliminary.compare:
        physical = False
    if physical:
        from isaaclab.app import add_launcher_args, launch_simulation
        add_launcher_args(parser)
    args = parser.parse_args()
    if args.resume and args.checkpoint:
        raise ValueError("resume与checkpoint只能指定一个完整模型")
    from .application import read_config, read_tasks, require_file, seed_all, build_engine, task_assets, raw_task, behavior_config
    cfg = read_config(args.config)
    if args.tasks:
        cfg["tasks"] = args.tasks
    if args.music_baseline:
        cfg["music_baseline"] = args.music_baseline
    if args.variant == "no_music":
        cfg["rewards"]["music"] = 0.
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"不覆盖任何已有模型或日志目录: {output}")
    if args.self_test:
        if kind != "audit":
            raise ValueError("self-test仅用于audit")
        output.mkdir(parents=True)
        import shutil
        temp = output/"pytest_tmp"
        command = [sys.executable, "-B", "-m", "pytest", "tests/closedloop/reference_adapter", "-q", "-s", "-p", "no:cacheprovider", "--basetemp="+str(temp)]
        env = os.environ | dict(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                SUPERMIMIC_REPO=cfg["supermimic"]["repo"], SM_BRIDGE_CHECKPOINT=cfg["supermimic"]["checkpoint"],
                                SM_ROBOT_ASSET_DIR=cfg["supermimic"]["robot_asset_dir"])
        with (output/"pytest.log").open("x") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if temp.exists() and temp.resolve().parent == output:
            shutil.rmtree(temp)
        write_json(output/"result.json", dict(command=command, exit_code=result.returncode, physical_run=False,
                   music_reward="causal_event_matching_v2", physics_quality="not_run"))
        return result.returncode
    if args.compare:
        from .evaluation import validate_comparison
        reports = {name: json.loads(Path(path).read_text()) for name, path in zip(("zero", "no_music", "music"), args.compare)}
        comparison = validate_comparison(reports)
        output.mkdir(parents=True)
        write_json(output/"comparison.json", comparison)
        return 0
    target_updates = args.target_updates if args.target_updates is not None else args.updates
    if (args.formal or args.resume) and args.target_updates is None:
        raise ValueError("正式训练/恢复必须显式指定绝对--target-updates")
    if kind == "train" and not args.calibrate_music:
        if args.variant == "zero" or target_updates < 1:
            raise ValueError("零Adapter只评价/标定；更新次数必须为正")
        if not args.formal and target_updates > cfg["training"]["maximum_bounded_updates"]:
            raise ValueError("超出有界更新预算；长训必须显式--formal")

    tasks = read_tasks(cfg["tasks"], preparing=kind == "prepare")
    if args.formal and {t["split"] for t in tasks} != {"train", "validation", "audit", "evaluation"}:
        raise ValueError("正式训练需song/audio SHA互斥的train/validation/audit/evaluation四组")
    seed_all(cfg["training"]["seed"])
    engine = build_engine(cfg, args.torch_device, calibrating=args.calibrate_music or kind == "prepare")
    from .replay import RealReplay
    from .trainer import AdapterTrainer
    from .checkpoint import load_checkpoint, save_checkpoint
    replay = RealReplay(cfg["training"]["replay_capacity"], cfg["training"]["seed"])
    trainer = AdapterTrainer(engine, replay, cfg["training"])
    resume = args.resume or args.checkpoint
    restored = load_checkpoint(resume, trainer, behavior=behavior_config(cfg), device=args.torch_device) if resume else None
    if kind == "evaluate" and args.variant != "zero" and not resume:
        raise ValueError("有/无音乐Adapter评价必须加载各自真实训练checkpoint")
    criteria = None
    if args.physical or (kind == "train" and not args.calibrate_music):
        criteria = json.loads(require_file(cfg["physical_criteria"], "physical_criteria").read_text())
    if kind == "prepare":
        for field in ("checkpoint", "stats", "kinematics"):
            require_file(cfg["genmo"][field], "genmo."+field)
        if cfg["genmo"]["bootstrap_latency_ticks"] is None:
            raise ValueError("准备缓存必须显式提供已测/候选GENMO延迟tick并记录，不猜测")
    output.mkdir(parents=True)
    source = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    write_json(output/"manifest.json", dict(config=cfg, tasks=tasks, genmo_source_commit=source,
                physical_run=physical, mode=kind, variant=args.variant, resume=resume,
                physics_resume=restored["physics_resume"] if restored else None))
    if kind == "audit" and not physical:
        from .audit import numerical_audit
        from .collector import initial_state
        results = []
        for task in tasks:
            initial, nominal, audio, _ = task_assets(task, engine, args.torch_device)
            xi = initial.new_zeros(1, 25)
            xi[:, 0] = .75
            z = initial_state(dict(state=initial, xi=xi, previous_action=initial.new_zeros(1, 21)), nominal, audio, engine.step,
                              episode_id="numerical", song_id=task["song_id"], split=task["split"])
            results.append(dict(task=task["id"], report=numerical_audit(engine, z)))
        write_json(output/"numerical_audit.json", results)
        return 0
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from .collector import PhysicalCollector
    import torch
    from .evaluation import summarize
    from .audit import physical_audit
    sim_cfg = make_sim(args.torch_device, args)
    with launch_simulation(sim_cfg, args):
        env = StreamingPhysicalEnv(sim_cfg, 1, engine.step.bundle)
        collector = PhysicalCollector(env, engine.step)
        if restored:
            collector.steps = restored["collector_steps"]
            collector.reset_count = restored["collector_resets"]
        # JIT/图捕获预热只发生在第一episode之前，不计入部署路径时间。
        with torch.no_grad():
            for _ in range(3):
                env.step_physics(torch.zeros(1, 21, device=args.torch_device))
        serial = (restored.get("cursor") or {}).get("episodes", 0) if restored else 0

        def assets(task):
            nonlocal serial
            serial += 1
            seed_all(task["seed"])
            initial, nominal, audio, contract = task_assets(task, engine, args.torch_device)
            metadata = dict(episode_id=f"{task['id']}_{uuid.uuid4().hex}", song_id=task["song_id"], split=task["split"])
            return initial, nominal, audio, contract, metadata

        def collect_episode(task, *, zero, add_replay=True, max_steps=None):
            initial, nominal, audio, contract, metadata = assets(task)
            collector.policy_version = trainer.update_index
            collector.reset(initial, nominal, audio, **metadata)
            rows, contacts = [], []
            while not collector.terminal:
                part = collector.collect(1, replay if add_replay else None, zero=zero)
                if not part:
                    raise RuntimeError("固定缓存采集无进展，不伪造参考")
                rows.extend(part)
                contacts.append(env.contact_events())
                if max_steps is not None and len(rows) >= max_steps:
                    break
            trainer.real_steps += len(rows) if add_replay else 0
            report = summarize(rows, contacts=contacts, runtime=engine.step.runtime)
            report.update(task=task["id"], song=task["song_id"], split=task["split"], seed=task["seed"])
            if max_steps is not None and not collector.terminal:
                report["reason"] = "administrative_truncation"
                report["supported_reference_complete"] = report["full_music_complete"] = False
            with (output/"physical_episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+"\n")
            return rows, report, contract

        def gate(*, small=False):
            reports = []
            audit_tasks = [t for t in tasks if t["split"] == "audit"]
            if not audit_tasks:
                raise ValueError("真实方向审计需要独立audit歌曲")
            for task in (audit_tasks[:1] if small else audit_tasks):
                initial, nominal, audio, _, metadata = assets(task)
                for start_steps in ([0] if small else task.get("audit_start_steps", cfg["training"]["physical_audit_start_steps"])):
                    reports.append(dict(task=task["id"], **physical_audit(collector, engine, initial, nominal, audio, metadata, criteria, start_steps=start_steps)))
            result = dict(passed=all(r["passed"] for r in reports), tasks=reports, measured_unix=time.time(),
                          evidence="real_reset_paired_directions", audit_scope="small" if small else "full")
            write_json(output/f"physics_model{engine.dynamics.model_version}_adapter{trainer.update_index}_{time.time_ns()}.json", result)
            return result

        if kind == "prepare":
            from .nominal import prepare_cache
            prepared = []
            for task in tasks:
                seed_all(task["seed"])
                music, initial = raw_task(task, args.torch_device)
                target = output/f"nominal_{len(prepared):04d}.pt"
                contract = prepare_cache(env, engine.step.bundle, cfg["genmo"], task, music, initial, target,
                                         bootstrap_latency_ticks=cfg["genmo"]["bootstrap_latency_ticks"])
                prepared.append(task | {"nominal_reference": str(target), "generation_contract": contract})
            write_json(output/"tasks.json", prepared)
        elif args.calibrate_music:
            from .music_reward import calibrate_s0
            from supermimic.bridge.bundle import atomic_save, file_digest
            samples, provenance = [], []
            for task in tasks:
                if task["split"] == "train":
                    rows, _, contract = collect_episode(task, zero=True, add_replay=False)
                    samples.extend(r.end.executed_state[:, 34:] for r in rows)
                    provenance.append(dict(task=task["id"], contract=contract, nominal_sha256=file_digest(task["nominal_reference"])))
            if not samples:
                raise ValueError("标定缺少训练集真实零Adapterepisode")
            atomic_save(dict(source="zero_adapter_real_training_episodes", s0=calibrate_s0(torch.cat(samples)).cpu(),
                             joint_names=engine.step.bundle.spec.joint_names, samples=sum(len(x) for x in samples), tasks=provenance), output/"music_baseline.pt")
        elif kind == "evaluate":
            episodes, contracts = [], []
            for task in tasks:
                _, report, contract = collect_episode(task, zero=args.variant == "zero", add_replay=False, max_steps=args.max_steps)
                episodes.append(report)
                contracts.append(dict(id=task["id"], contract=contract, robot_contract=engine.step.bundle.metadata["robot"],
                                      observation_noise=cfg["observation_noise"], music_s0=engine.step.rewards.music.s0.cpu().tolist()))
            write_json(output/"evaluation.json", dict(variant=args.variant, task_contract=contracts, episodes=episodes,
                       budget=dict(updates=trainer.update_index, real_steps=trainer.real_steps),
                       physical_quality_passed=None, physical_quality_status="measured_without_acceptance_comparison",
                       measured_unix=time.time()))
        elif kind == "audit":
            result = gate()
            write_json(output/"physical_audit.json", result)
            return 0 if result["passed"] else 2
        else:
            training_tasks = [t for t in tasks if t["split"] == "train"]
            validation_tasks = [t for t in tasks if t["split"] == "validation"]
            evaluation_tasks = [t for t in tasks if t["split"] == "evaluation"]
            cursor = (restored.get("cursor") or {}).get("task_cursor", 0) if restored else 0
            baseline = (restored.get("cursor") or {}).get("evaluation_baseline") if restored else None
            started_training = time.perf_counter()

            def save(label):
                save_checkpoint(output/f"{label}_{trainer.update_index:06d}_{time.time_ns()}.pt", trainer, collector=collector,
                    behavior=behavior_config(cfg), provenance={"genmo_source_commit": source},
                    cursor={"episodes": serial, "task_cursor": cursor, "target_updates": target_updates,
                            "evaluation_baseline": baseline})

            def evaluate_fixed(zero=False, small=False):
                selected = evaluation_tasks[:1] if small else evaluation_tasks
                if not selected:
                    raise ValueError("缺少独立固定真实评价歌曲")
                return [collect_episode(t, zero=zero, add_replay=False)[1] for t in selected]

            try:
                if not trainer.critic_ready:
                    for task in training_tasks+validation_tasks:
                        collect_episode(task, zero=True)
                    for warmup_round in range(cfg["training"].get("critic_warmup_rounds", 5)):
                        report = trainer.warmup_real(cfg["training"]["critic_warmup_epochs"])
                        write_json(output/f"critic_warmup_{warmup_round:02d}.json", report)
                        if trainer.critic_ready:
                            break
                    if not trainer.critic_ready:
                        raise RuntimeError(f"Critic真实MC预热门槛失败: {trainer.warmup_report}")
                # 独立WM先真实监督到验证合格；物理方向仍必须随后独立通过。
                for warmup in range(cfg["training"].get("wm_warmup_steps", 200)):
                    if trainer.validate_wm()["quality_gate"]["passed"]:
                        break
                    trainer.dynamics.fit(replay.sample_windows(cfg["training"]["batch_size"], 8), trainer.wm_optimizer)
                wm_preflight = trainer.validate_wm()
                write_json(output/"wm_preflight.json", wm_preflight)
                if not wm_preflight["quality_gate"]["passed"]:
                    raise RuntimeError(f"WM真实1/8/16步预热门槛失败: {wm_preflight}")
                trainer.register_physical_gate(gate())
                if baseline is None:
                    baseline = evaluate_fixed(zero=True)
                    write_json(output/"zero_physical_baseline.json", baseline)
                from .evaluation import quality_comparison
                while trainer.update_index < target_updates:
                    # 单环境方案只在真episode边界优化/审计。只轮转必要歌曲，不每轮重跑全表。
                    collected = 0
                    while collected < cfg["training"]["collect_steps_per_round"]:
                        task = training_tasks[cursor % len(training_tasks)]
                        cursor += 1
                        rows, _, _ = collect_episode(task, zero=trainer.update_index == 0)
                        collected += len(rows)
                    # 当前分布独立真实WM与MC校验，验证episode不复用训练歌曲。
                    collect_episode(validation_tasks[(cursor-1) % len(validation_tasks)], zero=trainer.update_index == 0)
                    report = trainer.update()
                    index = trainer.update_index
                    small_due = index == 1 or index % cfg["training"].get("small_audit_every", 25) == 0
                    full_due = index % cfg["training"].get("full_audit_every", 100) == 0
                    if small_due or full_due:
                        current_gate = gate(small=not full_due)
                        measured = evaluate_fixed(small=not full_due)
                        comparison = quality_comparison(baseline, measured, cfg["training"]["physical_monitor"])
                        report["post_update_physical"] = dict(direction=current_gate, comparison=comparison, episodes=measured)
                        if not current_gate["passed"] or not comparison["passed"]:
                            trainer.rollback_last_update()
                            raise RuntimeError(f"候选Adapter真实配对质量失败，回滚拒绝计数: {comparison}; direction={current_gate['passed']}")
                        trainer.register_physical_gate(current_gate)
                    report.update(real_environment_count=1, collector_resets=collector.reset_count,
                                  accepted_updates_per_hour=3600*trainer.update_index/max(time.perf_counter()-started_training, 1e-6),
                                  target_updates=target_updates, continuity="updates_only_after_true_episode_end")
                    with (output/"training.jsonl").open("a") as stream:
                        stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+"\n")
                        stream.flush()
                    if index == 1 or index % cfg["training"].get("checkpoint_every", 25) == 0:
                        save("accepted")
                    print(json.dumps({k: report[k] for k in ("accepted_adapter_updates", "model_version", "real_steps", "update_wall_seconds")}), flush=True)
                save("completed")
                write_json(output/"completed.json", dict(accepted_adapter_updates=trainer.update_index, target_updates=target_updates))
            except (RuntimeError, FloatingPointError, ValueError) as error:
                save("stopped")
                write_json(output/"stopped.json", dict(reason=str(error), accepted_adapter_updates=trainer.update_index,
                           attempted_updates=trainer.attempted_updates, physical_gate=trainer.physical_gate))
                return 2
    return 0
