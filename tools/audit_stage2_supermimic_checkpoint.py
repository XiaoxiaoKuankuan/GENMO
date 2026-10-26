"""读取已经发生有效GENMO更新的完整检查点，实际装配模型并验证恢复数据。

本工具使用正式配置、原Stage1与固定SM资产，通过训练同一个load_checkpoint入口
恢复GENMO、Upper Critic、两个AdamW、真实回放、随机数及监督数据游标，并逐张量
对比保存内容；再验证恢复后下一批原配对监督数据可重复。不会调用优化器、初始化
物理求解器或修改原正式目录，输出写到明确的独立验收目录，只在服务器2运行。
该报告证明完整状态加载，不能代替新episode的实际继续优化，不能冒充训练收益；
物理求解器明确不在检查点内，异常后采集必须从新episode开始。源码版本不作门槛。
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def equal_tree(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sys.path.insert(0, cfg["supermimic"]["repo"])
    from gem.closedloop.mb_control.application import build_components, build_stage1_losses, behavior_config, SupervisionStream, LatencyProfile
    from gem.closedloop.mb_control.checkpoint import load_checkpoint, pack_value, capture_rng, restore_rng, SCHEMA
    from gem.closedloop.mb_control.replay import RecentReplay
    from gem.closedloop.mb_control.trainer import Stage2Trainer
    path = Path(args.checkpoint).resolve()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != SCHEMA or saved["counters"]["accepted_actor_updates"] < 1:
        raise ValueError("需要真实有效更新后的完整训练checkpoint，不接受推理导出或仅Critic预热")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    actor, adapter, scheduler, rewards, critic, engine, original = build_components(cfg, args.device)
    trainer = Stage2Trainer(actor, critic, engine, build_stage1_losses(actor.stage1, original).to(args.device), cfg["training"])
    replay = RecentReplay(cfg["training"]["replay_capacity"], cfg["training"]["seed"])
    restored = load_checkpoint(path, trainer, replay,
        behavior_config=behavior_config(cfg, actor, adapter), device=args.device)
    checks = {
        "actor_tensor_identity": equal_tree(actor.state_dict(), saved["actor"]),
        "critic_tensor_identity": equal_tree(critic.state_dict(), saved["critic"]),
        "genmo_optimizer_identity": equal_tree(trainer.actor_optimizer.state_dict(), saved["genmo_optimizer"]),
        "critic_optimizer_identity": equal_tree(trainer.critic_optimizer.state_dict(), saved["upper_critic_optimizer"]),
        "counters_identity": equal_tree(trainer.counters(), saved["counters"]),
        "global_rng_identity": equal_tree(capture_rng(), saved["rng"]),
        "requires_new_physical_episode": restored.pop("requires_new_physical_episode") is True,
        "runtime_and_worker_rng_identity": equal_tree(restored, saved["runtime_state"]),
        "solver_snapshot_not_claimed": saved["physical_solver_state_saved"] is False,
    }
    replay_saved = replay.state_dict()
    memo = {}
    replay_saved["items"] = [pack_value(value, memo) for value in replay_saved["items"]]
    checks["real_replay_identity"] = equal_tree(replay_saved, saved["replay"])
    for sample in replay.items:
        sample.validate(len(sample.transitions))
    stream = SupervisionStream(cfg["stage1"]["data_config"], cfg["training"]["supervision_batch"], cfg["training"]["seed"], args.device)
    stream.load_state_dict(restored["supervision"])
    checks["supervision_cursor_identity"] = equal_tree(stream.state_dict(), restored["supervision"])
    before_batch = capture_rng()
    first_batch = stream.next()
    restore_rng(before_batch)
    stream.load_state_dict(restored["supervision"])
    checks["next_original_supervision_batch_reproducible"] = equal_tree(first_batch, stream.next())
    profile = LatencyProfile(cfg["latency_profile"], cfg["training"]["seed"]+1)
    profile.rng.setstate(restored["latency_rng"])
    checks["latency_rng_identity"] = equal_tree(profile.rng.getstate(), restored["latency_rng"])
    adapter.bundle.assert_unchanged(trainer.frozen_identity)
    checks["frozen_parameters_and_statistics_unchanged"] = True
    optimizer_states = {name: {"parameters_with_moments": len(saved[key]["state"]),
        "step_min": min(float(v["step"]) for v in saved[key]["state"].values()),
        "step_max": max(float(v["step"]) for v in saved[key]["state"].values())}
        for name, key in (("genmo", "genmo_optimizer"), ("critic", "upper_critic_optimizer"))}
    report = {"passed": all(checks.values()), "checks": checks, "checkpoint": str(path),
        "saved_accepted_actor_updates": trainer.accepted_actor_updates, "new_actor_updates_in_this_audit": 0,
        "optimizers": optimizer_states, "real_replay_samples": len(replay.items),
        "physical_worker_rng_count": len(restored.get("physical_worker_rng", {})),
        "source_hash_gate": False, "evidence": "actual_complete_state_load_not_resumed_physics_or_training_gain"}
    (output/"checkpoint_restore.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise RuntimeError("完整checkpoint恢复核对失败，保留报告且不改原训练")


if __name__ == "__main__":
    main()
