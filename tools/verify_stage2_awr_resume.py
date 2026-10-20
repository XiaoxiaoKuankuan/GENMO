"""用真实Replay、原Stage1 BC和八卡NCCL验证完整优化步断点续算。

仅在正式运行完成且GPU空闲后执行：取三条已审核Replay，固定真实优势权重，从最终
AWR权重复制出验收对象，连续运行两步，并在第一步原子保存全部rank RNG/BC状态；
另建Trainer恢复该断点再运行第二步，逐元素比较Actor、AdamW矩、Scheduler及随机流。
全局AWR三条/BC一条自然包含空rank。总计三个诊断优化步，仅作用于独立验收对象；
结束恢复原Actor指纹，不修改正式checkpoint、训练步数、标签或Replay，不增加物理
交互。这里的微型续算证明恢复算术，不能被当作新一轮学习或真实物理中断恢复。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import timedelta


def verify_resume(actor, dataset, bc_factory, config, distributed, output):
    import torch
    from gem.closedloop.awr.checkpoint import Checkpoints, capture_rng
    from gem.closedloop.awr.trainer import ActorTrainer
    from gem.closedloop.awr.io import cpu_copy, digest
    from verify_stage2_awr_checkpoint import assert_tree_equal
    d = distributed
    initial = cpu_copy(actor.state_dict())
    fingerprint = d.assert_models_equal(actor)
    c = dict(config, epochs=2, global_batch=3, bc_global_batch=1,
             microbatch=1, bc_microbatch=1, warmup_steps=1)
    if len(dataset) != 3:
        raise ValueError("This finite acceptance requires exactly three real Replay samples")
    checkpoint = Checkpoints(output, dict(purpose="isolated_ddp_resume_acceptance",
        actor=fingerprint, data=digest(dataset.sample_ids), config=c, world_size=d.world_size))
    torch.manual_seed(91821+d.rank*100003)
    trainer = ActorTrainer(actor, dataset, bc_factory(), c, d, seed=91821, bc_weight=.01)
    saved = []
    def first_step(current, _):
        if current.steps == 1:
            ranks = d.gather(dict(bc=current.bc.state_dict(), rng=capture_rng(d.device)))
            payload = current.state_dict() if d.rank == 0 else None
            if payload is not None:
                payload["rank_states"] = ranks
            saved.append(d.root(lambda: checkpoint.save("step_one", payload,
                progress=dict(phase="diagnostic_actor", step=1))))
    continuous = trainer.run(on_step=first_step)
    expected = trainer.state_dict()
    expected_rng = capture_rng(d.device)
    expected_fingerprint = d.assert_models_equal(actor)
    second = ActorTrainer(actor, dataset, bc_factory(), c, d, seed=91821, bc_weight=.01)
    payload = d.local(lambda: checkpoint.load(saved[0])["payload"])
    second.load_state_dict(payload, rank_state=payload["rank_states"][d.rank])
    resumed = second.run()
    actual = second.state_dict()
    def compare():
        for key in ("actor", "optimizer", "scheduler", "bc", "bc_weight",
                    "epoch", "cursor", "steps", "data_identity", "total_steps", "seed", "config"):
            assert_tree_equal(actual[key], expected[key])
        assert_tree_equal(capture_rng(d.device), expected_rng)
        return True
    d.local(compare)
    if d.assert_models_equal(actor) != expected_fingerprint:
        raise ValueError("Restored distributed update differs from continuous update")
    actor.load_state_dict(initial)
    if d.assert_models_equal(actor) != fingerprint:
        raise ValueError("Diagnostic Actor did not restore the original final weights")
    return dict(status="passed", world_size=d.world_size, continuous_steps=continuous["steps"],
        resumed_total_steps=resumed["steps"], diagnostic_optimizer_updates=3,
        formal_run_optimizer_updates_added=0, physical_controls=0, global_awr_samples=3,
        global_bc_samples=1, sample_ids=dataset.sample_ids,
        continuous_and_resumed_actor=expected_fingerprint, original_actor_restored=fingerprint,
        actor_optimizer_scheduler_bc_rng_exact=True, diagnostic_checkpoint=saved[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, repo, output = (p.resolve() for p in (args.run_dir, args.runtime_repo, args.output_dir))
    if output.exists() or any(output.is_relative_to(root/n) for n in ("replay", "checkpoints")):
        parser.error("Use a new independent acceptance directory")
    before = (root/"state.json").read_bytes()
    state, config = json.loads(before), json.loads((root/"request.json").read_text())
    if state["status"] != "complete" or Path(config["paths"]["genmo_repo"]).resolve() != repo:
        parser.error("Require completed physical run and original runtime worktree")
    sys.path.insert(0, str(repo))
    import torch
    import torch.distributed as dist
    from gem.closedloop.awr.config import preflight, physical_config
    from gem.closedloop.awr.distributed import Distributed
    from gem.closedloop.awr.checkpoint import Checkpoints
    from gem.closedloop.awr.records import AWRCompleteEpisode, TRAINING_METADATA_KEYS
    from gem.closedloop.awr.dataset import ReplayDataset
    from gem.closedloop.awr.advantages import AdvantageLabelSet
    from gem.closedloop.awr.io import atomic_json, read_tree
    from gem.closedloop.common.supervision import load_actor, Stage1Supervision
    rank = int(os.environ["RANK"])
    if (int(os.environ["WORLD_SIZE"]) != 8 or int(os.environ["LOCAL_RANK"]) != rank
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7"):
        parser.error("Use single-node eight-GPU torchrun")
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    group = None
    started = time.perf_counter()
    try:
        d = Distributed(f"cuda:{rank}")
        audit = d.root(lambda: preflight(config))
        if state["source_commits"] != {k: v["commit"] for k, v in audit["repositories"].items()}:
            raise ValueError("Resume acceptance changed original source identity")
        if state["configuration_assets"] != audit["configuration_assets"]:
            raise ValueError("Frozen control configuration changed")
        d.root(lambda: output.mkdir(parents=True, exist_ok=False))
        torch.cuda.set_device(rank)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        group = dist.new_group(backend="nccl", timeout=timedelta(minutes=30))
        d.tensor_group = group
        runtime = physical_config(state["config"], run_id=root.name, rank=rank, device=d.device)
        actor, train, loaded = d.local(lambda: load_actor(runtime))
        prefix = f"macro{state['macro']-1:03d}"
        ref = state["artifacts"][prefix+"_awr_final"]
        cp = Checkpoints(root/"checkpoints", dict(request=state["request_sha256"], sources=state["source_commits"]))
        actor.load_state_dict(cp.load(ref)["payload"]["actor"])
        evaluation = json.loads(Path(state["artifacts"][prefix+"_awr_fixed"]).read_text())
        if d.assert_models_equal(actor) != evaluation["actor_fingerprint"]:
            raise ValueError("Input Actor differs from physically evaluated Actor")
        snapshot_id = state["artifacts"][prefix+"_replay"]
        snapshot = json.loads((root/"replay/snapshots"/(snapshot_id+".json")).read_text())
        def subset():
            episodes = []
            for identifier in snapshot["episode_ids"]:
                episode = AWRCompleteEpisode.from_dict(read_tree(root/"replay/episodes"/identifier,
                    episode_metadata_keys=TRAINING_METADATA_KEYS))
                episode.training_view = True
                if episode.episode_id != identifier:
                    raise ValueError("Snapshot Episode identity changed")
                episodes.append(episode)
                if sum(sum(row.actor_valid for row in ep.rows) for ep in episodes) >= 3:
                    break
            data = ReplayDataset(episodes, unit_weights=True)
            data.rows, data.sample_ids, data.sources = data.rows[:3], data.sample_ids[:3], data.sources[:3]
            data.returns = data.returns[:3]
            labels = AdvantageLabelSet.load(state["artifacts"][prefix+"_labels"])
            lookup = {name: i for i, name in enumerate(labels.sample_ids)}
            ids = torch.tensor([lookup[name] for name in data.sample_ids])
            if not torch.equal(labels.returns[ids], data.returns):
                raise ValueError("Fixed labels differ from actual Replay MC")
            data.weights = labels.weights[ids].clone()
            return data
        dataset = d.local(subset)
        result = verify_resume(actor, dataset,
            lambda: Stage1Supervision(runtime, actor, train, source_step=loaded["source_global_step"]),
            config["awr"]["actor"], d, output/"diagnostic_checkpoints")
        if (root/"state.json").read_bytes() != before:
            raise ValueError("Official training state changed during independent acceptance")
        peaks = d.gather(torch.cuda.max_memory_allocated(d.device))
        result.update(source_run=str(root), source_commits=state["source_commits"],
            snapshot=snapshot_id, original_checkpoint=ref, backend="nccl", real_replay=True,
            real_stage1_bc=True, peak_memory_bytes=peaks, seconds=time.perf_counter()-started,
            tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            component_helper_sha256=hashlib.sha256(Path(__file__).with_name(
                "verify_stage2_awr_checkpoint.py").read_bytes()).hexdigest())
        d.root(lambda: atomic_json(output/"report.json", result))
        if rank == 0:
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
