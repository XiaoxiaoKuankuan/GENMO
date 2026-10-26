"""独立train/evaluate/audit/prepare命令行入口。

默认有界且必须显式执行；只读help和self-test不初始化物理。真实路径先绑定资产，
独占新输出目录，然后复用SM原StreamingPhysicalEnv，按完整episode采集再更新/审计，
不因采集分段reset。恢复学习状态时明确新建物理episode，不启动旧Trainer或正式长训。
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
    parser.add_argument("--updates", type=int, default=1)
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
                   music_proxy_sharp_valley_generalization="known_counterexample_not_accepted"))
        return result.returncode
    if args.compare:
        from .evaluation import validate_comparison
        reports = {name: json.loads(Path(path).read_text()) for name, path in zip(("zero", "no_music", "music"), args.compare)}
        comparison = validate_comparison(reports)
        output.mkdir(parents=True)
        write_json(output/"comparison.json", comparison)
        return 0
    if kind == "train" and not args.calibrate_music:
        if args.variant == "zero" or args.updates < 1:
            raise ValueError("零Adapter只评价/标定；更新次数必须为正")
        if not args.formal and args.updates > cfg["training"]["maximum_bounded_updates"]:
            raise ValueError("超出有界更新预算；长训必须显式--formal")
        if not cfg["training"]["physical_audit_each_update"]:
            raise ValueError("真实训练不能绕过当前Adapter分布8/16步物理核验")
    tasks = read_tasks(cfg["tasks"], preparing=kind == "prepare")
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

        def gate():
            reports = []
            for task in tasks:
                initial, nominal, audio, _, metadata = assets(task)
                for start_steps in task.get("audit_start_steps", cfg["training"]["physical_audit_start_steps"]):
                    reports.append(dict(task=task["id"], **physical_audit(collector, engine, initial, nominal, audio, metadata, criteria, start_steps=start_steps)))
            result = dict(passed=all(r["passed"] for r in reports), tasks=reports)
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
                       budget=dict(updates=trainer.update_index, real_steps=trainer.real_steps), physical_quality_passed=False))
        elif kind == "audit":
            result = gate()
            write_json(output/"physical_audit.json", result)
            return 0 if result["passed"] else 2
        else:
            if not trainer.critic_ready:
                if {t["split"] for t in tasks} != {"train", "validation"}:
                    raise ValueError("价值预热与WM验证需要歌曲互斥的train/validation任务")
                for task in tasks:
                    collect_episode(task, zero=True)
                write_json(output/"critic_warmup.json", trainer.warmup_real(cfg["training"]["critic_warmup_epochs"]))
            for _ in range(args.updates):
                # 分段收集不reset；本入口在完整episode结束后才更新/使用同一物理环境审计。
                collected = 0
                while collected < cfg["training"]["collect_steps_per_round"]:
                    for task in tasks:
                        rows, _, _ = collect_episode(task, zero=trainer.update_index == 0)
                        collected += len(rows)
                try:
                    report = trainer.update(model_gate=gate)
                    current_gate = gate()
                    report["post_update_physics_passed"] = current_gate["passed"]
                    # 当前策略独立真实验证回报，不沿用旧策略/旧模型value targets。
                    validation_episodes = [collect_episode(t, zero=False, add_replay=True)[0] for t in tasks if t["split"] == "validation"]
                    report["real_critic_calibration"] = trainer.check_real_calibration(validation_episodes)
                    with (output/"training.jsonl").open("a") as stream:
                        stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+"\n")
                    save_checkpoint(output/f"update_{trainer.update_index:06d}.pt", trainer, collector=collector,
                                    behavior=behavior_config(cfg), provenance={"genmo_source_commit": source}, cursor={"episodes": serial})
                    if not current_gate["passed"]:
                        write_json(output/"stopped.json", dict(reason="new_adapter_distribution_physics_not_passed", update=trainer.update_index))
                        return 2
                except (RuntimeError, FloatingPointError) as error:
                    save_checkpoint(output/f"stopped_{time.time_ns()}.pt", trainer, collector=collector,
                                    behavior=behavior_config(cfg), provenance={"genmo_source_commit": source}, cursor={"episodes": serial})
                    write_json(output/"stopped.json", dict(reason=str(error), update=trainer.update_index))
                    return 2
    return 0
