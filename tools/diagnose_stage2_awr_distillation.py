"""Server-2有界自蒸馏定位：旧单分支、相同预算CFG修复、受控更新三组。

从已停止运行的首轮Stage1 Replay只读提取四来源等数完整Episode的Actor条件/参考，
三组均从原s595000初始化、使用单位权重、同顺序/扩散噪声/真实BC流。旧奖励与优势
只保留来源身份，不参与优化；这些历史记录绝不写入新奖励Replay。第一二组只改变
CFG监督算式，第三组显式缩小LR并加入原模型两分支约束，用于有界候选验收。

固定真实val条件和种子检查更新前后生成质量；可另行用evaluate操作对指定候选与
Stage1重新执行完整32任务，使用删除活动乘数后的同一奖励。保留所有失败与模型，
不自动启动Critic/AWR或长期训练，不以少量探针通过宣称五轮自蒸馏稳定。
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "evaluate"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/closedloop/stage2_awr.yaml")
    parser.add_argument("--episodes-per-source", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--candidate", choices=("legacy", "guided", "bounded"), default="bounded")
    args = parser.parse_args()
    if not 1 <= args.episodes_per_source <= 16 or not 1 <= args.max_steps <= 200:
        parser.error("Bounded diagnosis permits 1–16 Episodes/source and at most 200 steps/arm")
    import torch
    import torch.distributed as dist
    import numpy as np
    from gem.closedloop.awr.config import load_config, physical_config, preflight
    from gem.closedloop.awr.distributed import Distributed
    from gem.closedloop.awr.io import atomic_json, cpu_copy, digest, read_tree, write_tree
    from gem.closedloop.awr.records import AWRCompleteEpisode, TRAINING_METADATA_KEYS
    from gem.closedloop.awr.dataset import ReplayDataset
    from gem.closedloop.awr.trainer import ActorTrainer
    from gem.closedloop.awr.checkpoint import Checkpoints, stop_requested
    from gem.closedloop.awr.sampler import AWRSampler
    from gem.closedloop.awr.metrics import stage1_validation, parameter_changes
    from gem.closedloop.common.supervision import load_actor, Stage1Supervision
    from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
    from gem.closedloop.stage1_dataset import collate_stage1_training_samples
    from gem.closedloop.training import batch_to_device
    from tools.validate_stage2_awr_generation import clip_metrics
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8:
        parser.error("Use exactly eight ranks")
    dist.init_process_group("gloo", timeout=timedelta(hours=2))
    group = None
    started = time.perf_counter()
    try:
        cfg = load_config(args.config); cfg["paths"]["genmo_repo"] = str(ROOT)
        d = Distributed(f"cuda:{rank}")
        audit = d.root(lambda: preflight(cfg))
        cfg["awr"]["configuration_assets"] = audit["configuration_assets"]
        cfg["awr"]["source_commits"] = {k: v["commit"] for k, v in audit["repositories"].items()}
        source, output = args.source_run.resolve(), args.output.resolve()
        state = json.loads((source/"state.json").read_text())
        if state["status"] not in ("stopped", "complete", "user_interrupted"):
            raise ValueError("Source training must be stopped before diagnosis")
        torch.cuda.set_device(rank); torch.set_num_threads(4); torch.manual_seed(812)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        group = dist.new_group(backend="nccl", timeout=timedelta(hours=1)); d.tensor_group = group
        runtime = physical_config(cfg, run_id=output.name, rank=rank, device=d.device)
        actor, train, report = d.local(lambda: load_actor(runtime)); d.sync_model(actor)
        original = cpu_copy(actor.state_dict()); fingerprint = d.assert_models_equal(actor)
        identity = dict(sources=audit["repositories"], config=digest(cfg), source_run=str(source),
                        source_request=state["request_sha256"], original_actor=fingerprint)
        if args.operation == "train":
            d.root(lambda: output.mkdir(parents=True, exist_ok=False))
            d.root(lambda: atomic_json(output/"identity.json", identity))
            d.root(lambda: atomic_json(output/"preflight.json", audit))
        else:
            if json.loads((output/"identity.json").read_text()) != identity:
                raise ValueError("Diagnosis/evaluation source or configuration changed")
        checkpoints = Checkpoints(output/"checkpoints", identity)
        policy = AWRSampler(actor)
        def probe(name, cases):
            actor.eval(); result = []
            for index in range(rank, len(cases), 8):
                case = cases[index]
                batch = batch_to_device(case["batch"], d.device)
                for seed in (1729, 2718):
                    with torch.no_grad():
                        pred = policy.sample({k: batch[k] for k in STAGE1_CONDITION_KEYS}, seeds=[seed])
                        metrics = clip_metrics(batch, pred, actor.endecoder.normalize(batch["target_qpos30"]))
                        mask = (batch["target_qpos30_valid"] & ~batch["known_qpos30_mask"])[..., 9:]
                        values = pred["qpos30"][..., 9:]
                        for order, label in ((1, "speed"), (2, "acceleration"), (3, "jerk")):
                            values = torch.diff(values, dim=1)*30
                            mask = mask[:, 1:] & mask[:, :-1]
                            metrics[label] = float(values[mask].double().square().mean().sqrt())
                    result.append(dict(case=index, source=case["source"], seed=seed, metrics=metrics,
                                       qpos30=pred["qpos30"].cpu(), mask=(batch["target_qpos30_valid"] &
                                       ~batch["known_qpos30_mask"]).cpu()))
            values = [v for items in d.gather(result) for v in items]
            d.root(lambda: write_tree(output/"probes"/name, values))
            return values
        if args.operation == "train":
            def select():
                snap = json.loads((source/"replay/snapshots"/(state["artifacts"]["macro000_replay"]+".json")).read_text())
                allowed = set(snap["episode_ids"])
                with sqlite3.connect("file:"+str(source/"replay/index.sqlite")+"?mode=ro", uri=True) as db:
                    rows = db.execute("SELECT id,source,sha FROM episodes ORDER BY seq").fetchall()
                groups = {}
                for key, origin, sha in rows:
                    if key in allowed:
                        groups.setdefault(origin, []).append(dict(id=key, source=origin, sha=sha))
                if len(groups) != 4 or any(len(v) < args.episodes_per_source for v in groups.values()):
                    raise ValueError("Need four sources with enough audited Stage1 Episodes")
                return [r for key in sorted(groups) for r in groups[key][:args.episodes_per_source]]
            selected = d.root(select)
            d.root(lambda: atomic_json(output/"episodes.json", selected))
            def read_episodes():
                episodes = []
                for entry in selected:
                    path = source/"replay/episodes"/entry["id"]
                    if json.loads((path/"manifest.json").read_text())["sha256"] != entry["sha"]:
                        raise ValueError("Historical Replay identity mismatch")
                    ep = AWRCompleteEpisode.from_dict(read_tree(path, episode_metadata_keys=TRAINING_METADATA_KEYS))
                    ep.training_view = True; ep.validate()
                    if any(r.identity["actor_fingerprint"] != fingerprint for r in ep.rows):
                        raise ValueError("Self-distillation source must be the same frozen Stage1 teacher")
                    episodes.append(ep)
                return episodes
            episodes = d.local(read_episodes)
            data = ReplayDataset(episodes, unit_weights=True)
            def validation_cases():
                bc = Stage1Supervision(runtime, actor, train, source_step=report["source_global_step"], split="val")
                cases = []; gen = torch.Generator().manual_seed(90317)
                with bc._rng_scope(actor):
                    for dataset in bc.datasets:
                        for index in torch.randperm(len(dataset), generator=gen)[:4].tolist():
                            batch = collate_stage1_training_samples([dataset[index]])
                            if batch["meta"][0]["split"] != "val":
                                raise ValueError("Generation retention cases must be held out")
                            cases.append(dict(source=dataset.dataset_name, batch=cpu_copy(batch)))
                return cases
            cases = d.root(validation_cases)
            d.root(lambda: write_tree(output/"validation_cases", cases))
            before = probe("stage1", cases)
            base = copy.deepcopy(cfg["awr"]["actor"])
            base.update(max_optimizer_steps=args.max_steps, reference_weight=0.)
            coefficient = None; summaries = {}
            for name in ("legacy", "guided", "bounded"):
                if d.root(lambda: stop_requested(output)):
                    return 0
                actor.load_state_dict(original); torch.manual_seed(42+rank*100003)
                c = copy.deepcopy(base)
                if name == "legacy":
                    c.update(denoising_guidance_scale=1., replay_eval_mode=False)
                if name == "bounded":
                    c.update(lr=1e-6, weight_decay=0., reference_weight=1.)
                bc = Stage1Supervision(runtime, actor, train, source_step=report["source_global_step"])
                trainer = ActorTrainer(actor, data, bc, c, d, seed=42, bc_weight=coefficient)
                if name in ("legacy", "bounded"):
                    calibration = trainer.calibrate(8, .1 if name == "legacy" else .3)
                    d.root(lambda: atomic_json(output/(name+"_calibration.json"), calibration))
                    if name == "legacy": coefficient = trainer.bc_weight
                def log(current, entry):
                    if rank == 0:
                        with (output/(name+"_steps.jsonl")).open("a") as stream:
                            stream.write(json.dumps(entry)+"\n")
                        if entry["step"] % 10 == 0: print(name, entry["step"], flush=True)
                result = trainer.run(on_step=log, should_stop=lambda: stop_requested(output))
                reference = d.root(lambda: checkpoints.save(name, trainer.state_dict(), progress=dict(method=name)))
                d.root(lambda: atomic_json(output/(name+"_checkpoint.json"), reference))
                if not result["complete"]: return 0
                results = d.gather(result)
                changes = d.root(lambda: parameter_changes(actor, original))
                after = probe(name, cases)
                def summarize():
                    pairs = {(v["case"], v["seed"]): v for v in before}
                    metrics = {}
                    for key in ("speed", "acceleration", "jerk", "true_contact_bce", "normalized_unknown_qpos30_mse"):
                        left = np.mean([v["metrics"][key] for v in before])
                        right = np.mean([v["metrics"][key] for v in after])
                        metrics[key] = dict(baseline=float(left), current=float(right), ratio=float(right/max(left, 1e-12)))
                    changes_joint = []
                    for row in after:
                        old = pairs[row["case"], row["seed"]]
                        mask = row["mask"][..., 9:]
                        changes_joint.append(float((row["qpos30"][..., 9:]-old["qpos30"][..., 9:])[mask].double().square().mean().sqrt()))
                    joint_change = float(np.mean(changes_joint))
                    gates = dict(motion_bounded=all(.9 <= metrics[k]["ratio"] <= 1.1 for k in ("speed", "acceleration", "jerk")),
                                 joint_change_bounded=joint_change <= .03,
                                 true_contact_retained=metrics["true_contact_bce"]["ratio"] <= 1.05)
                    return dict(config=c, coefficient=trainer.bc_weight, samples=len(data), episodes=len(episodes),
                        training=results, parameters=changes, metrics=metrics, joint_change_mean_rad=joint_change,
                        gates=gates, generation_probe_passed=all(gates.values()),
                        checkpoint=reference, physical_retention="not_yet_evaluated")
                summary = d.root(summarize); summaries[name] = summary
                d.root(lambda: atomic_json(output/(name+"_summary.json"), summary))
                del trainer, bc
            d.root(lambda: atomic_json(output/"training_summary.json", dict(arms=summaries, seconds=time.perf_counter()-started)))
        else:
            from gem.closedloop.common.full_dataset import FullMusicCatalog
            from gem.closedloop.online_conditions import OnlineConditionBuilder
            from gem.closedloop.awr.world import calibrate
            from gem.closedloop.awr.evaluation import evaluate
            from gem.closedloop.awr.retention import retention_comparison
            summary = json.loads((output/(args.candidate+"_summary.json")).read_text())
            # 评价失败探针也允许保留真实反例，但绝不据此自动批准AWR。
            catalog = d.local(lambda: FullMusicCatalog(cfg["paths"]["data_root"]))
            audit_data = json.loads(Path(state["artifacts"]["data_audit"]).read_text())
            d.local(lambda: catalog.apply_audit(audit_data))
            plan = json.loads(Path(state["artifacts"]["evaluation_fixed"]).read_text())
            builder = OnlineConditionBuilder(actor.endecoder.codec, history_steps=50, motion_frames=60)
            profile_path = output/"deployment_profile.json"
            if profile_path.exists():
                profile = json.loads(profile_path.read_text())
            else:
                profile = calibrate(runtime, policy, builder, catalog, plan["tasks"], output/"calibration", d, fingerprint)
                d.root(lambda: atomic_json(profile_path, profile))
            runtime["timing"]["deployment_profile"] = profile
            physics = digest(dict(config=cfg, profile=profile, sources=audit["repositories"]))
            reports = {}
            for name in ("stage1", args.candidate):
                path = output/(name+"_physical.json")
                if path.exists(): reports[name] = json.loads(path.read_text()); continue
                actor.load_state_dict(original if name == "stage1" else checkpoints.load(summary["checkpoint"])["payload"]["actor"])
                physical = evaluate(runtime, policy, builder, catalog, plan, output/"physical"/name, d,
                    fingerprint=d.assert_models_equal(actor), physics_identity=physics,
                    progress_path=output/"evaluation_progress"/name, should_stop=lambda: stop_requested(output))
                d.root(lambda: atomic_json(path, physical)); reports[name] = physical
            retention = retention_comparison(reports["stage1"], reports[args.candidate], cfg["awr"]["retention_gate"])
            retention["generation_probe_passed"] = summary["generation_probe_passed"]
            retention["accepted"] = retention["accepted"] and summary["generation_probe_passed"]
            d.root(lambda: atomic_json(output/(args.candidate+"_physical_retention.json"), retention))
        return 0
    finally:
        if group is not None: dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
