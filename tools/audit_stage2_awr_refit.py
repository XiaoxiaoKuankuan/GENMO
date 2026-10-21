"""有限自蒸馏复验结束后的只读Stage1完整监督验收。

显式导入原运行冻结工作树，核验源码/资产/检查点身份并等待GPU全部空闲；不更新
模型、优化器、Replay或原运行状态。沿用八个逻辑rank的独立验证采样种子，每个
rank四批真实val样本，在单GPU串行比较Stage1、原教师1000候选、新无权重及AWR。
每次比较后恢复验证随机状态，保留真实Contact、FK等原Stage1完整Loss。

额外将原教师1000候选放到新复验相同的验证条件和采样种子上，填补跨数据/跨探针
不能直接比较的证据缺口。旧候选只是诊断，不回写新Replay、不重算优势。输出必须
全新，保存模型引用、工具摘要、逐批损失和原始生成探针；不宣称有限验证代表长期保留。
本工具可从已提交仓库复制到独立验收目录，以免热更新原运行源码。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, root, output = (p.resolve() for p in (args.runtime_repo, args.run_dir, args.output))
    if output.exists() or any(output.is_relative_to(root/p) for p in ("replay", "checkpoints")):
        parser.error("Use a new independent verification output")
    lifecycle = json.loads(Path(str(root)+".exit.json").read_text())
    if lifecycle["exit_code"] != 0:
        parser.error("Wait for the bounded run to finish before independent GPU validation")
    config = json.loads((root/"resolved_config.json").read_text())
    if Path(config["paths"]["genmo_repo"]).resolve() != repo:
        parser.error("Use the immutable original runtime repository")
    sys.path.insert(0, str(repo))
    import torch
    from gem.closedloop.awr.checkpoint import Checkpoints
    from gem.closedloop.awr.config import preflight, physical_config
    from gem.closedloop.awr.io import atomic_json, cpu_copy, digest, file_sha256, read_tree, write_tree
    from gem.closedloop.awr.metrics import stage1_validation
    from gem.closedloop.awr.sampler import AWRSampler
    from gem.closedloop.common.evaluation import _model_fingerprint
    from gem.closedloop.common.supervision import load_actor, Stage1Supervision
    from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
    from gem.closedloop.training import batch_to_device
    from tools.validate_stage2_awr_generation import clip_metrics
    audit = preflight(config)
    identity = json.loads((root/"identity.json").read_text())
    if digest(config) != identity["config"] or audit["configuration_assets"] != config["awr"]["configuration_assets"]:
        raise ValueError("Refit configuration or frozen physical control assets changed")
    if audit["repositories"] != identity["sources"]:
        raise ValueError("Refit source repository changed")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    runtime = physical_config(config, run_id=root.name, rank=0, device=device)
    actor, train, report = load_actor(runtime)
    original = cpu_copy(actor.state_dict())
    if _model_fingerprint(actor) != identity["original_actor"]:
        raise ValueError("Original Stage1 fingerprint differs")
    source = Path(identity["source_run"])
    source_state = json.loads((source/"state.json").read_text())
    old_ref = source_state["artifacts"]["macro000_self_distillation_final"]
    states = {"stage1": original, "teacher1000": Checkpoints(source/"checkpoints", old_ref["identity"]).load(old_ref)["payload"]["actor"]}
    references = {"stage1": identity["original_actor"], "teacher1000": old_ref}
    for name in ("bounded", "awr"):
        path = root/(name+"_checkpoint.json")
        if path.exists():
            reference = json.loads(path.read_text())
            states[name] = Checkpoints(root/"checkpoints", identity).load(reference)["payload"]["actor"]
            references[name] = reference
    output.mkdir(parents=True, exist_ok=False)
    losses = {name: [] for name in states}
    for rank in range(8):
        local = physical_config(config, run_id=root.name, rank=rank, device=device)
        bc = Stage1Supervision(local, actor, train, source_step=report["source_global_step"], split="val")
        seed = config["awr"]["seed"]+700001+rank
        for name, weights in states.items():
            actor.load_state_dict(weights)
            values = stage1_validation(actor, bc, batches=4,
                batch_size=config["awr"]["actor"]["bc_microbatch"], device=device, seed=seed)
            losses[name].extend(values)
        del bc
        print("validated logical rank", rank, flush=True)
    results = {name: dict(stage1_full_loss=sum(v)/len(v), batches=len(v), per_batch=v)
               for name, v in losses.items()}
    actor.load_state_dict(states["teacher1000"]); actor.eval()
    policy = AWRSampler(actor)
    probes = []
    for index, case in enumerate(read_tree(root/"validation_cases")):
        batch = batch_to_device(case["batch"], device)
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
            probes.append(dict(case=index, source=case["source"], seed=seed, metrics=metrics,
                qpos30=pred["qpos30"].cpu(), mask=(batch["target_qpos30_valid"] & ~batch["known_qpos30_mask"]).cpu()))
    write_tree(output/"teacher1000_probes", probes)
    result = dict(schema="genmo.awr.refit_stage1_validation.v1", run=str(root), runtime_sources=audit["repositories"],
        tool_sha256=file_sha256(__file__), checkpoints=references, split="val",
        logical_rank_seeds=[config["awr"]["seed"]+700001+i for i in range(8)],
        execution="single_gpu_serial_reproduction_of_eight_logical_rank_validation_streams",
        sample_count_per_model=32*config["awr"]["actor"]["bc_microbatch"],
        fixed_validation_randomness=True, optimizer_steps=0, physical_tasks=0,
        teacher1000_probe_count=len(probes), losses=results, seconds=time.perf_counter()-started)
    atomic_json(output/"report.json", result)
    print(json.dumps(dict(losses={k:v["stage1_full_loss"] for k,v in results.items()}, seconds=result["seconds"])))


if __name__ == "__main__":
    main()
