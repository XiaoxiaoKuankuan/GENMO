"""对已完成AWR运行进行有限八卡前反向计时，不执行优化器更新。

从同一不可变Replay快照读取足够组成一个真实全局Batch的Episode，恢复已训练AWR
Actor及固定优势权重，使用原Stage1真标签BC。分别同步计时单扩散时间前反向、BC
前反向、梯度SUM和裁剪；一次预热后默认测3批，最多8批。前后核验Actor指纹不变，
不修改训练状态、标签、Replay或checkpoint，不采集新物理数据。
仅在原运行完成且八卡空闲时运行。报告是有限离线测量，不能冒充原整轮BC精确累计
耗时，也不能替代整轮含失败尝试/I/O/评价的成本。归档工具显式导入原运行源码。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def timed_backward(trainer, indices, step, synchronize):
    """只累计和归约梯度；优化器step与scheduler.step都不在此路径。"""
    import torch
    trainer.optimizer.zero_grad(set_to_none=True)
    synchronize()
    start = time.perf_counter()
    awr_loss = trainer.backward_awr(indices, step=step)
    synchronize()
    awr_done = time.perf_counter()
    bc_loss = trainer.backward_bc(trainer.bc_weight)
    synchronize()
    bc_done = time.perf_counter()
    trainer.distributed.gradients(trainer.actor)
    synchronize()
    reduced = time.perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in trainer.actor.parameters() if p.requires_grad],
        trainer.config.get("gradient_clip", 1.), error_if_nonfinite=True)
    synchronize()
    finished = time.perf_counter()
    return dict(awr_forward_backward_seconds=awr_done-start,
        stage1_bc_forward_backward_seconds=bc_done-awr_done,
        gradient_sum_seconds=reduced-bc_done, clip_seconds=finished-reduced,
        total_without_optimizer_seconds=finished-start, awr_loss=awr_loss, bc_loss=bc_loss,
        preclip_gradient_norm=float(norm), optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    root, repo, output = (p.resolve() for p in (args.run_dir, args.runtime_repo, args.output_dir))
    if (not 1 <= args.steps <= 8 or output.exists()
            or any(output.is_relative_to(root/name) for name in ("replay", "checkpoints"))):
        parser.error("Use 1–8 measured batches and a new output directory")
    state = json.loads((root/"state.json").read_text())
    config = json.loads((root/"request.json").read_text())
    if state["status"] != "complete" or Path(config["paths"]["genmo_repo"]).resolve() != repo:
        parser.error("Require a completed run and its original runtime worktree")
    sys.path.insert(0, str(repo))
    import torch
    import torch.distributed as dist
    from gem.closedloop.awr.config import preflight, physical_config
    from gem.closedloop.awr.distributed import Distributed
    from gem.closedloop.awr.checkpoint import Checkpoints
    from gem.closedloop.awr.advantages import AdvantageLabelSet
    from gem.closedloop.awr.io import read_tree, atomic_json, digest
    from gem.closedloop.awr.records import AWRCompleteEpisode, TRAINING_METADATA_KEYS
    from gem.closedloop.awr.dataset import ReplayDataset
    from gem.closedloop.awr.trainer import ActorTrainer
    from gem.closedloop.common.supervision import load_actor, Stage1Supervision
    rank = int(os.environ["RANK"])
    if (int(os.environ["WORLD_SIZE"]) != 8 or int(os.environ["LOCAL_RANK"]) != rank
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7"):
        parser.error("Use the approved single-node eight-GPU torchrun environment")
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    group = None
    started = time.perf_counter()
    try:
        d = Distributed(f"cuda:{rank}")
        evidence = d.root(lambda: preflight(config))
        if state["source_commits"] != {k: v["commit"] for k, v in evidence["repositories"].items()}:
            raise ValueError("Profiling source differs from the completed run")
        if state["configuration_assets"] != evidence["configuration_assets"]:
            raise ValueError("Frozen control configuration differs")
        d.root(lambda: output.mkdir(parents=True, exist_ok=False))
        torch.cuda.set_device(rank)
        torch.set_num_threads(config["runtime"]["torch_threads"])
        torch.manual_seed(91821+rank)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        group = dist.new_group(backend="nccl", timeout=timedelta(minutes=30))
        d.tensor_group = group
        runtime = physical_config(state["config"], run_id=root.name, rank=rank, device=d.device)
        actor, train, loaded = d.local(lambda: load_actor(runtime))
        key = f"macro{state['macro']-1:03d}"
        reference = state["artifacts"][key+"_awr_final"]
        checkpoint = Checkpoints(root/"checkpoints",
            dict(request=state["request_sha256"], sources=state["source_commits"]))
        actor.load_state_dict(checkpoint.load(reference)["payload"]["actor"])
        fingerprint = d.assert_models_equal(actor)
        evaluation = json.loads(Path(state["artifacts"][key+"_awr_fixed"]).read_text())
        if fingerprint != evaluation["actor_fingerprint"]:
            raise ValueError("Profiling Actor differs from physically evaluated Actor")
        snapshot_id = state["artifacts"][key+"_replay"]
        snapshot = json.loads((root/"replay/snapshots"/(snapshot_id+".json")).read_text())
        def load_subset():
            episodes, samples = [], 0
            for identifier in snapshot["episode_ids"]:
                episode = AWRCompleteEpisode.from_dict(read_tree(root/"replay/episodes"/identifier,
                    episode_metadata_keys=TRAINING_METADATA_KEYS))
                episode.training_view = True
                if episode.episode_id != identifier:
                    raise ValueError("Replay snapshot/episode identity mismatch")
                episodes.append(episode)
                samples += sum(row.actor_valid for row in episode.rows)
                if samples >= config["awr"]["actor"]["global_batch"]:
                    return episodes
            raise ValueError("Not enough real Actor samples for the configured global batch")
        dataset = ReplayDataset(d.local(load_subset), unit_weights=True)
        labels = AdvantageLabelSet.load(state["artifacts"][key+"_labels"])
        lookup = {sample: index for index, sample in enumerate(labels.sample_ids)}
        indices = torch.tensor([lookup[sample] for sample in dataset.sample_ids])
        if not torch.equal(labels.returns[indices], dataset.returns):
            raise ValueError("Fixed labels differ from actual Replay MC")
        dataset.weights = labels.weights[indices].clone()
        coefficient = json.loads(Path(state["artifacts"][key+"_bc_calibration"]).read_text())["coefficient"]
        bc = Stage1Supervision(runtime, actor, train, source_step=loaded["source_global_step"])
        trainer = ActorTrainer(actor, dataset, bc, config["awr"]["actor"], d, seed=91821,
                               bc_weight=coefficient)
        ids = list(range(config["awr"]["actor"]["global_batch"]))
        actor.train()
        sync = lambda: torch.cuda.synchronize(d.device)
        timed_backward(trainer, ids, -1, sync)
        rows = [timed_backward(trainer, ids, i, sync) for i in range(args.steps)]
        trainer.optimizer.zero_grad(set_to_none=True)
        if d.assert_models_equal(actor) != fingerprint or trainer.steps or trainer.optimizer.state:
            raise ValueError("Read-only backward profiling unexpectedly updated Actor/optimizer")
        ranks = d.gather(rows)
        peaks = d.gather(torch.cuda.max_memory_allocated(d.device))
        fields = [key for key in rows[0] if key.endswith("_seconds")]
        maxima = [{key: max(rank[i][key] for rank in ranks) for key in fields} for i in range(args.steps)]
        report = dict(schema="genmo.awr.bounded_backward_profile.v1", source_run=str(root),
            source_commits=state["source_commits"], checkpoint=reference, actor_fingerprint=fingerprint,
            replay_snapshot=snapshot_id, sample_ids_sha256=digest([dataset.sample_ids[i] for i in ids]),
            profiled_source_samples=dict(Counter(dataset.sources[i] for i in ids)),
            actor_config=config["awr"]["actor"], bc_weight=coefficient, warmup_batches=1,
            measured_batches=args.steps, ranks=ranks, maximum_rank_seconds_per_batch=maxima,
            mean_maximum_rank_seconds={key: sum(row[key] for row in maxima)/args.steps for key in fields},
            physical_interaction_steps=0, optimizer_updates=0, actor_parameters_unchanged=True,
            scope="bounded_offline_backward_profile_not_whole_macro_training_cost",
            peak_allocator_bytes_by_rank=peaks, seconds=time.perf_counter()-started,
            tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        d.root(lambda: atomic_json(output/"report.json", report))
        if rank == 0:
            print(json.dumps(report["mean_maximum_rank_seconds"]), flush=True)
        return 0
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
