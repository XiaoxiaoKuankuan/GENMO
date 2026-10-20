"""比较冻结Stage1、自蒸馏和AWR在固定原验证条件下的生成保留能力。

从原Stage1 val Dataset取得真实条件、qpos30、Contact及完整物理监督metadata，
先原样存成独立验证分片，再让三份Actor使用相同初始噪声执行DDIM20/eta0/CFG2.5。
报告原Stage1完整生成损失、未知坐标误差、真实Contact BCE、关节活动度及相对
Stage1的输出变化。此处是真实验证标签，不把Replay生成Contact当标签，也不把
小批固定生成探针冒称完整验证集或物理学习效果。

仅在指定运行完成且GPU空闲后进行单GPU只读推理，沿用原运行提交和资产身份，
不更新模型/优化器/运行状态，不写Replay或checkpoint。工具可独立归档并显式
导入原GENMO工作树；输出目录必须全新，保存条件、种子、模型指纹和工具摘要。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def clip_metrics(batch, result, normalized_target, baseline=None):
    """只在真实有效未知坐标及真实Contact标签上比较，不把已知前缀算作生成能力。"""
    import torch
    from torch.nn import functional as F
    mask = batch["target_qpos30_valid"] & ~batch["known_qpos30_mask"]
    def mean_masked(value, valid):
        if not valid.any():
            raise ValueError("Generation probe has no valid supervised coordinates")
        return torch.where(valid, value, 0.).double().sum()/valid.sum()
    physical = result["qpos30"]
    normalized_mse = mean_masked((result["normalized"]-normalized_target).square(), mask)
    joint_mask = mask[..., 9:]
    joint_mse = mean_masked((physical[..., 9:]-batch["target_qpos30"][..., 9:]).square(), joint_mask)
    contact_mask = batch["target_contact_valid"]
    contact_bce = mean_masked(F.binary_cross_entropy_with_logits(result["contact_logits"],
        batch["target_contact"].float(), reduction="none"), contact_mask)
    difference_mask = joint_mask[:, 1:] & joint_mask[:, :-1]
    joint_velocity = (physical[:, 1:, 9:]-physical[:, :-1, 9:])*30.
    velocity_rms = mean_masked(joint_velocity.square(), difference_mask).sqrt()
    baseline = physical if baseline is None else baseline
    change = mean_masked((physical[..., 9:]-baseline[..., 9:]).square(), joint_mask).sqrt()
    values = dict(normalized_unknown_qpos30_mse=float(normalized_mse),
        unknown_joint_rmse_rad=float(joint_mse.sqrt()), true_contact_bce=float(contact_bce),
        unknown_joint_velocity_rms_rad_s=float(velocity_rms), joint_change_from_stage1_rms_rad=float(change))
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("Nonfinite fixed-condition generation metric")
    known = batch["known_qpos30_mask"]
    if not torch.equal(physical[known], batch["known_qpos30"][known]):
        raise ValueError("Fixed generation changed its known physical prefix")
    return values


def main():
    tool_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips-per-source", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2718])
    parser.add_argument("--selection-seed", type=int, default=90317)
    parser.add_argument("--macro", type=int, default=0)
    args = parser.parse_args()
    root, repo, output = (p.resolve() for p in (args.run_dir, args.runtime_repo, args.output_dir))
    if args.clips_per_source < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Need positive clip count and unique generation seeds")
    if output.exists() or any(output.is_relative_to(root/name) for name in ("replay", "checkpoints")):
        parser.error("Use a new independent acceptance directory, not Replay/checkpoints")
    state = json.loads((root/"state.json").read_text())
    if state["status"] != "complete":
        parser.error("Complete the physical run before this separate retention probe")
    config = json.loads((root/"request.json").read_text())
    if repo != Path(config["paths"]["genmo_repo"]).resolve():
        parser.error("Use the original run's GENMO worktree")
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from gem.closedloop.awr.config import preflight, physical_config
    from gem.closedloop.awr.checkpoint import Checkpoints
    from gem.closedloop.awr.io import atomic_json, cpu_copy, digest, write_tree
    from gem.closedloop.common.evaluation import _model_fingerprint
    from gem.closedloop.common.supervision import load_actor, Stage1Supervision
    from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
    from gem.closedloop.stage1_dataset import collate_stage1_training_samples
    from gem.closedloop.training import batch_to_device
    audit = preflight(config)
    if state["source_commits"] != {k: v["commit"] for k, v in audit["repositories"].items()}:
        raise ValueError("Retention probe source differs from the physical experiment")
    if state["configuration_assets"] != audit["configuration_assets"]:
        raise ValueError("Frozen control configuration differs")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    runtime = physical_config(state["config"], run_id=root.name, rank=0, device="cuda:0")
    actor, train_config, load_report = load_actor(runtime)
    actor.eval()
    baseline_state = cpu_copy(actor.state_dict())
    bc = Stage1Supervision(runtime, actor, train_config,
        source_step=load_report["source_global_step"], split="val")
    cases = []
    generator = torch.Generator().manual_seed(args.selection_seed)
    with bc._rng_scope(actor):
        for dataset in bc.datasets:
            order = torch.randperm(len(dataset), generator=generator)[:args.clips_per_source].tolist()
            for index in order:
                batch = collate_stage1_training_samples([dataset[index]])
                if batch["meta"][0]["split"] != "val":
                    raise ValueError("Retention probe requires actual Stage1 validation labels")
                cases.append(dict(source=dataset.dataset_name, dataset_index=index, batch=cpu_copy(batch)))
    output.mkdir(parents=True, exist_ok=False)
    case_identity = write_tree(output/"fixed_validation_cases", cases)
    checkpoints = Checkpoints(root/"checkpoints",
        dict(request=state["request_sha256"], sources=state["source_commits"]))
    prefix = f"macro{args.macro:03d}"
    methods = ["stage1", "self_distillation", "awr"]
    reference = {
        method: state["artifacts"][prefix+"_"+method+"_final"] for method in methods[1:]}
    reports, baseline_outputs = {}, {}
    started = time.perf_counter()
    with torch.no_grad():
        for method in methods:
            payload = baseline_state if method == "stage1" else checkpoints.load(reference[method])["payload"]["actor"]
            actor.load_state_dict(payload, strict=True)
            actor.eval()
            rows, arrays = [], []
            fingerprint = _model_fingerprint(actor)
            for index, case in enumerate(cases):
                batch = batch_to_device(case["batch"], "cuda:0")
                context = {key: batch[key] for key in STAGE1_CONDITION_KEYS}
                normalized_target = actor.endecoder.normalize(batch["target_qpos30"])
                for seed in args.seeds:
                    rng = torch.Generator().manual_seed(seed+index*100003)
                    noise = torch.randn(batch["target_qpos30"].shape, generator=rng)
                    begin = time.perf_counter()
                    result = actor.sample(context, steps=20, guidance_scale=2.5, noise=noise)
                    torch.cuda.synchronize()
                    sampling_seconds = time.perf_counter()-begin
                    key = (index, seed)
                    if method == "stage1":
                        baseline_outputs[key] = result["qpos30"].detach().cpu()
                    metrics = clip_metrics(batch, result, normalized_target,
                                           baseline_outputs[key].to("cuda:0"))
                    loss, details = bc.losses(batch, result["normalized"], result["contact_logits"],
                                             global_step=load_report["source_global_step"])
                    metrics["full_stage1_generated_loss"] = float(loss)
                    if not math.isfinite(metrics["full_stage1_generated_loss"]):
                        raise FloatingPointError("Nonfinite original Stage1 generation loss")
                    rows.append(dict(case=index, source=case["source"], seed=seed,
                        actual_noise_seed=seed+index*100003, sampling_seconds=sampling_seconds,
                        metrics=metrics, details={k: float(v) for k, v in details.items()}))
                    arrays.append(dict(case=index, seed=seed, qpos30=result["qpos30"].cpu(),
                                       contact=result["contact"].cpu(), initial_noise=noise))
            array_identity = write_tree(output/(method+"_outputs"), arrays)
            by_source = {}
            for source in sorted({row["source"] for row in rows}):
                selected = [r for r in rows if r["source"] == source]
                by_source[source] = dict(count=len(selected), metrics={
                    key: float(np.mean([r["metrics"][key] for r in selected]))
                    for key in selected[0]["metrics"]})
            balanced = {key: float(np.mean([v["metrics"][key] for v in by_source.values()]))
                        for key in rows[0]["metrics"]}
            reports[method] = dict(actor_fingerprint=fingerprint, output_sha256=array_identity,
                checkpoint=reference.get(method), source_balanced_metrics=balanced,
                by_source=by_source, cases=rows)
    report = dict(schema="genmo.awr.fixed_stage1_generation_retention.v1",
        source_commits=state["source_commits"], source_run=str(root),
        request_sha256=state["request_sha256"], actual_cases=len(cases), seeds=args.seeds,
        condition_sha256=case_identity, selection_seed=args.selection_seed,
        original_stage1_ground_truth_used=True, full_validation_set=False,
        physical_interaction_or_optimizer_updates=0, methods=reports,
        scope="fixed_subset_of_original_stage1_val_generation_not_physical_improvement",
        seconds=time.perf_counter()-started,
        total_seconds_from_main=time.perf_counter()-tool_started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated("cuda:0"),
        peak_gpu_memory_scope="single_gpu_offline_process_pytorch_allocator",
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        numpy_version=np.__version__,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output/"report.json", report)
    print(json.dumps(dict(report=str(output/"report.json"),
        source_balanced_metrics={k: v["source_balanced_metrics"] for k, v in reports.items()}),
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
