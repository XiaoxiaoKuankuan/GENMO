"""Server-2有界自蒸馏定位：旧单分支、仅关闭Dropout、CFG修复、BC单独更新、受控更新五组。

从已停止运行的首轮Stage1 Replay只读提取四来源等数完整Episode的Actor条件/参考，
五组均从原s595000初始化、使用单位权重、同顺序/扩散噪声/真实BC流。旧奖励与优势
只保留来源身份，不参与优化；这些历史记录绝不写入新奖励Replay。第一二组只改变网络Dropout，第二三组只改变
CFG监督算式，第四组只训练原BC，第五组显式缩小LR并加入原模型两分支约束，用于有界候选验收。

固定真实val条件和种子检查更新前后生成质量；可另行用evaluate操作对指定候选与
Stage1重新执行完整32任务，使用删除活动乘数后的同一奖励。保留所有失败与模型，
train/evaluate不启动Critic/AWR或长期训练，不以少量探针通过宣称五轮自蒸馏稳定。
可显式选择单个对照分支、固定BC系数及教师权重；每次写独立实验参数与数据身份，
用于只改变一个因素的后续复验，不重写先前失败候选。
对照可显式复用已测部署profile，校验采样器/资产/来源SHA与测量数组；不把重新
测时导致的物理变化算成模型变化，训练与评价必须传同一profile。
refit操作只读复用已停止的新奖励运行的全部至多64集、原快照顺序、MC与固定交叉标签，
不新增训练采集。先执行单位权重生成和完整物理保留门禁，通过后才允许相同配置的
AWR对照；每组最多45步，不恢复旧正式训练。失败候选、原数据和原时钟始终保留。
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


def refit_snapshot_entries(snapshot, rows):
    """保持原快照完整顺序，禁止默默抽样、重复或补足不同Episode。"""
    ids = snapshot["episode_ids"]
    if not 1 <= len(ids) <= 64 or len(set(ids)) != len(ids):
        raise ValueError("Refit requires 1–64 unique complete Episodes")
    lookup = {key: dict(id=key, source=source, sha=sha) for key, source, sha in rows}
    if not set(ids).issubset(lookup):
        raise ValueError("Refit snapshot contains missing Episodes")
    return [lookup[key] for key in ids]


def validate_refit_contract(contract, config, sampler):
    """重用奖励标签必须保持资产、时钟、终止和采样定义；原采集提交身份不改写。"""
    expected = {key: config[key] for key in ("assets", "reward", "environment", "termination", "timing")}
    expected.update(configuration_assets=config["awr"]["configuration_assets"], sampler=sampler,
                    horizon_frames=60, schema="genmo.awr.replay_physics.v1")
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Refit Replay contract changed: {key}")


def require_unweighted_refit(generation, physical):
    """任何未通过或缺失的保留证据都禁止进入优势加权训练。"""
    if generation.get("generation_probe_passed") is not True or physical.get("accepted") is not True:
        raise ValueError("Unweighted generation and physical retention must pass before AWR")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "evaluate", "refit"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deployment-profile", type=Path, help="复用已经SHA绑定的真实部署测量JSON；训练/评价须一致")
    parser.add_argument("--episodes-per-source", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--candidate", choices=("legacy", "conditional_eval", "guided", "bc_only", "bounded"), default="bounded")
    parser.add_argument("--bounded-lr", type=float, default=1e-7)
    parser.add_argument("--reference-weight", type=float, default=10.)
    parser.add_argument("--bc-weight", type=float)
    parser.add_argument("--methods", nargs="+", choices=("legacy", "conditional_eval", "guided", "bc_only", "bounded"),
                        default=["legacy", "conditional_eval", "guided", "bc_only", "bounded"])
    args = parser.parse_args()
    if not 0 < args.bounded_lr <= 1e-5 or not 1 <= args.episodes_per_source <= 16 or not 1 <= args.max_steps <= 200:
        parser.error("Bounded diagnosis permits 1–16 Episodes/source and at most 200 steps/arm")
    import torch
    import torch.distributed as dist
    import numpy as np
    import math
    if not math.isfinite(args.reference_weight) or args.reference_weight < 0:
        parser.error("Reference weight must be finite and nonnegative")
    if args.bc_weight is not None and (not math.isfinite(args.bc_weight) or args.bc_weight <= 0):
        parser.error("BC weight must be finite and positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate diagnostic arms are not allowed")
    if args.operation == "refit":
        if args.max_steps > 45 or args.reference_weight <= 0:
            parser.error("Refit requires a positive teacher constraint and at most 45 steps/arm")
        args.methods = ["bounded", "awr"]
    if args.operation == "train" and args.bc_weight is None and args.methods[0] != "legacy":
        parser.error("Start with legacy calibration or provide an explicit fixed --bc-weight")
    from gem.closedloop.awr.config import load_config, physical_config, preflight, validate_measured_deployment_profile
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
        source, output = args.source_run.resolve(), args.output.resolve()
        state = json.loads((source/"state.json").read_text())
        if state["status"] not in ("stopped", "complete", "user_interrupted", "retention_rejected", "physical_rejected"):
            raise ValueError("Source training must be stopped before diagnosis")
        cfg = load_config(source/"request.json" if args.operation == "refit" else args.config)
        cfg["paths"]["genmo_repo"] = str(ROOT)
        if args.operation == "refit":
            cfg["timing"]["deployment_profile"] = validate_measured_deployment_profile(
                state["config"]["timing"]["deployment_profile"], cfg)
            if cfg["timing"]["deployment_profile"] is None:
                raise ValueError("Refit must reuse its source's measured physical clock")
            if args.bc_weight is None:
                args.bc_weight = json.loads(Path(state["artifacts"]["macro000_bc_calibration"]).read_text())["coefficient"]
        if args.deployment_profile is not None:
            profile = validate_measured_deployment_profile(
                json.loads(args.deployment_profile.read_text()), cfg)
            if args.operation == "refit" and profile != cfg["timing"]["deployment_profile"]:
                raise ValueError("Refit cannot change its source's physical clock")
            cfg["timing"]["deployment_profile"] = profile
        d = Distributed(f"cuda:{rank}")
        audit = d.root(lambda: preflight(cfg))
        cfg["awr"]["configuration_assets"] = audit["configuration_assets"]
        cfg["awr"]["source_commits"] = {k: v["commit"] for k, v in audit["repositories"].items()}
        torch.cuda.set_device(rank); torch.set_num_threads(4); torch.manual_seed(812)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        group = dist.new_group(backend="nccl", timeout=timedelta(hours=1)); d.tensor_group = group
        runtime = physical_config(cfg, run_id=output.name, rank=rank, device=d.device)
        actor, train, report = d.local(lambda: load_actor(runtime)); d.sync_model(actor)
        original = cpu_copy(actor.state_dict()); fingerprint = d.assert_models_equal(actor)
        identity = dict(sources=audit["repositories"], config=digest(cfg), source_run=str(source),
                        source_request=state["request_sha256"], original_actor=fingerprint)
        if args.operation in ("train", "refit"):
            d.root(lambda: output.mkdir(parents=True, exist_ok=False))
            d.root(lambda: atomic_json(output/"identity.json", identity))
            d.root(lambda: atomic_json(output/"preflight.json", audit))
            d.root(lambda: atomic_json(output/"experiment.json", dict(methods=args.methods, bounded_lr=args.bounded_lr,
                reference_weight=args.reference_weight, bc_weight=args.bc_weight, max_steps=args.max_steps,
                episodes_per_source=args.episodes_per_source, operation=args.operation,
                new_collection_episodes=0, formal_training_restarted=False)))
            d.root(lambda: atomic_json(output/"resolved_config.json", cfg))
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
        def physical_pair(candidate, summary):
            from gem.closedloop.common.full_dataset import FullMusicCatalog
            from gem.closedloop.online_conditions import OnlineConditionBuilder
            from gem.closedloop.awr.world import calibrate
            from gem.closedloop.awr.evaluation import evaluate
            from gem.closedloop.awr.retention import retention_comparison
            # 评价失败探针也允许保留真实反例，但绝不据此自动批准AWR。
            catalog = d.local(lambda: FullMusicCatalog(cfg["paths"]["data_root"]))
            audit_data = json.loads(Path(state["artifacts"]["data_audit"]).read_text())
            d.local(lambda: catalog.apply_audit(audit_data))
            plan = json.loads(Path(state["artifacts"]["evaluation_fixed"]).read_text())
            builder = OnlineConditionBuilder(actor.endecoder.codec, history_steps=50, motion_frames=60)
            profile_path = output/"deployment_profile.json"
            fixed_profile = cfg["timing"]["deployment_profile"]
            if profile_path.exists():
                profile = json.loads(profile_path.read_text())
                if fixed_profile is not None and profile != fixed_profile:
                    raise ValueError("Evaluation must retain the explicitly bound deployment profile")
            elif fixed_profile is not None:
                profile = fixed_profile
                d.root(lambda: atomic_json(profile_path, profile))
            else:
                profile = calibrate(runtime, policy, builder, catalog, plan["tasks"], output/"calibration", d, fingerprint)
                d.root(lambda: atomic_json(profile_path, profile))
            runtime["timing"]["deployment_profile"] = profile
            physics = digest(dict(config=cfg, profile=profile, sources=audit["repositories"]))
            reports = {}
            for name in ("stage1", candidate):
                path = output/(name+"_physical.json")
                if path.exists(): reports[name] = json.loads(path.read_text()); continue
                d.root(lambda: atomic_json(output/"evaluation_state.json", dict(status="evaluating", method=name)))
                actor.load_state_dict(original if name == "stage1" else checkpoints.load(summary["checkpoint"])["payload"]["actor"])
                physical = evaluate(runtime, policy, builder, catalog, plan, output/"physical"/name, d,
                    fingerprint=d.assert_models_equal(actor), physics_identity=physics,
                    progress_path=output/"evaluation_progress"/name, should_stop=lambda: stop_requested(output))
                d.root(lambda: atomic_json(path, physical)); reports[name] = physical
            retention = retention_comparison(reports["stage1"], reports[candidate], cfg["awr"]["retention_gate"])
            retention["generation_probe_passed"] = summary["generation_probe_passed"]
            retention["accepted"] = retention["accepted"] and summary["generation_probe_passed"]
            d.root(lambda: atomic_json(output/(candidate+"_physical_retention.json"), retention))
            return reports, retention
        if args.operation in ("train", "refit"):
            def select():
                snap = json.loads((source/"replay/snapshots"/(state["artifacts"]["macro000_replay"]+".json")).read_text())
                allowed = set(snap["episode_ids"])
                with sqlite3.connect("file:"+str(source/"replay/index.sqlite")+"?mode=ro", uri=True) as db:
                    rows = db.execute("SELECT id,source,sha FROM episodes ORDER BY seq").fetchall()
                if args.operation == "refit":
                    return refit_snapshot_entries(snap, rows)
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
                    if args.operation == "refit":
                        validate_refit_contract(ep.contract, cfg, policy.kernel_config)
                    if any(r.identity["actor_fingerprint"] != fingerprint for r in ep.rows):
                        raise ValueError("Self-distillation source must be the same frozen Stage1 teacher")
                    episodes.append(ep)
                return episodes
            episodes = d.local(read_episodes)
            labels = None
            if args.operation == "refit":
                from gem.closedloop.awr.advantages import AdvantageLabelSet
                labels = AdvantageLabelSet.load(state["artifacts"]["macro000_labels"])
                labeled = ReplayDataset(episodes, labels)
                d.root(lambda: atomic_json(output/"source_labels.json", dict(
                    path=state["artifacts"]["macro000_labels"], identity=labels.identity,
                    statistics=labels.statistics, replay_snapshot=state["artifacts"]["macro000_replay"],
                    sample_ids_digest=digest(labeled.sample_ids))))
                del labeled
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
            coefficient = args.bc_weight; summaries = {}; unit_retention = None
            for name in args.methods:
                if d.root(lambda: stop_requested(output)):
                    return 0
                if name == "awr":
                    require_unweighted_refit(summaries["bounded"], unit_retention or {})
                actor.load_state_dict(original); torch.manual_seed(cfg["awr"]["seed"]+rank*100003)
                data = ReplayDataset(episodes, labels, unit_weights=name != "awr")
                c = copy.deepcopy(base)
                if name in ("legacy", "conditional_eval"):
                    c.update(denoising_guidance_scale=1., replay_eval_mode=name != "legacy")
                if name == "bc_only":
                    c.update(replay_coefficient=0.)
                if name in ("bounded", "awr"):
                    c.update(lr=args.bounded_lr, weight_decay=0., reference_weight=args.reference_weight)
                bc = Stage1Supervision(runtime, actor, train, source_step=report["source_global_step"])
                trainer = ActorTrainer(actor, data, bc, c, d, seed=cfg["awr"]["seed"], bc_weight=coefficient)
                if name == "legacy" and coefficient is None:
                    calibration = trainer.calibrate(8, .1)
                    d.root(lambda: atomic_json(output/(name+"_calibration.json"), calibration))
                    coefficient = trainer.bc_weight
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
                        checkpoint=reference, physical_retention="not_yet_evaluated",
                        data_identity=digest(data.sample_ids), optimizer_seed=trainer.seed,
                        labels_identity=labels.identity if labels is not None else None)
                summary = d.root(summarize); summaries[name] = summary
                d.root(lambda: atomic_json(output/(name+"_summary.json"), summary))
                del trainer, bc
                if args.operation == "refit":
                    if name == "bounded" and not summary["generation_probe_passed"]:
                        d.root(lambda: atomic_json(output/"refit_state.json", dict(
                            status="generation_retention_rejected", awr_steps=0, new_collection_episodes=0)))
                        return 0
                    reports, retention = physical_pair(name, summary)
                    if name == "bounded":
                        unit_retention = retention
                        if not retention["accepted"]:
                            d.root(lambda: atomic_json(output/"refit_state.json", dict(
                                status="physical_retention_rejected", awr_steps=0, new_collection_episodes=0)))
                            return 0
                    else:
                        from gem.closedloop.awr.evaluation import paired_comparison
                        unweighted = json.loads((output/"bounded_physical.json").read_text())
                        versus_base = paired_comparison(reports["stage1"], reports[name])
                        versus_unit = paired_comparison(unweighted, reports[name])
                        d.root(lambda: atomic_json(output/"awr_comparison.json", dict(
                            versus_stage1=versus_base, versus_self_distillation=versus_unit)))
                        d.root(lambda: atomic_json(output/"refit_state.json", dict(status="complete",
                            physical_quality_passed=versus_base["acceptable"],
                            advantage_effect_supported=versus_unit["improvement_supported"],
                            awr_steps=results[0]["steps"], new_collection_episodes=0,
                            formal_training_restarted=False)))
            d.root(lambda: atomic_json(output/"training_summary.json", dict(arms=summaries, seconds=time.perf_counter()-started)))
        else:
            summary = json.loads((output/(args.candidate+"_summary.json")).read_text())
            physical_pair(args.candidate, summary)
        return 0
    finally:
        if group is not None: dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
