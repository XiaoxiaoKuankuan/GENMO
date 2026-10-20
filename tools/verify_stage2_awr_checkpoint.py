"""只读核验真实AWR最终断点的CPU组件恢复及物理评价绑定。

从指定运行的原源码装配Stage1 Actor，恢复B/C最终权重、AdamW矩和学习率调度状态，
逐元素核对恢复结果，并检查已保存八rank的CPU/BC随机流可恢复、CUDA随机状态存在。
模型指纹必须与固定及全量音乐评价一致。此工具不推进优化器、调度器、数据采样或物理
环境，不改运行状态；它是组件恢复验收，不冒充再次执行八卡续训或GPU随机流测试。
运行须已完成，输出必须全新，保留原断点SHA、工具SHA和实际源码身份。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time


def assert_tree_equal(actual, expected):
    import numpy as np
    import torch
    if torch.is_tensor(expected):
        if not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError("Restored tensor differs from checkpoint")
    elif isinstance(expected, np.ndarray):
        if not np.array_equal(actual, expected):
            raise ValueError("Restored array differs from checkpoint")
    elif isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Restored mapping keys differ")
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError("Restored sequence length differs")
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)
    elif actual != expected:
        raise ValueError("Restored scalar differs")


def restore_components(actor, payload):
    """恢复真正优化器及调度器，但绝不执行step；比较全部参数与内部矩。"""
    import torch
    config, total = payload["config"], payload["total_steps"]
    actor.load_state_dict(payload["actor"], strict=True)
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
        lr=config["lr"], weight_decay=config.get("weight_decay", .01))
    warmup = config.get("warmup_steps", 100)
    def scale(step):
        if step < warmup:
            return (step+1)/max(warmup, 1)
        phase = min(1., (step-warmup)/max(total-warmup-1, 1))
        floor = config.get("min_lr_ratio", .1)
        return floor+(1-floor)*(1+math.cos(math.pi*phase))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    assert_tree_equal(actor.state_dict(), payload["actor"])
    assert_tree_equal(optimizer.state_dict(), payload["optimizer"])
    assert_tree_equal(scheduler.state_dict(), payload["scheduler"])
    steps = []
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite restored AdamW state")
        steps.append(float(state["step"]))
    if not steps or payload["steps"] != total or payload["cursor"] != 0:
        raise ValueError("Final checkpoint is not a completed configured Actor training")
    return dict(optimizer_state_entries=len(steps), adam_step_min=min(steps), adam_step_max=max(steps),
        scheduler_last_epoch=scheduler.last_epoch, learning_rates=scheduler.get_last_lr(),
        epoch=payload["epoch"], cursor=payload["cursor"], steps=payload["steps"],
        data_identity=payload["data_identity"], bc_weight=payload["bc_weight"],
        restored_tensors_exact=True, optimizer_updates=0)


def check_rank_random_states(states):
    import numpy as np
    import torch
    if len(states) != 8:
        raise ValueError("Expected all eight rank random states")
    for rank in states:
        for node, keys in ((rank["rng"], ("torch", "numpy", "python", "cuda")),
                           (rank["bc"], ("torch_rng", "numpy_rng", "python_rng", "cuda_rng"))):
            torch.Generator().set_state(node[keys[0]])
            np.random.RandomState().set_state(node[keys[1]])
            random.Random().setstate(node[keys[2]])
            cuda = node[keys[3]]
            if not torch.is_tensor(cuda) or cuda.dtype != torch.uint8 or cuda.numel() == 0:
                raise ValueError("Missing stored CUDA random state")
        torch.Generator().set_state(rank["bc"]["generator"])
    return dict(rank_states=len(states), cpu_random_states_restored=True,
                cuda_random_states_present=True, cuda_random_states_executed=False)


def pairing_state(payload):
    """B/C必须有相同样本顺序身份、预算、BC系数及最终各rank BC随机流。"""
    return {**{k: payload[k] for k in
               ("data_identity", "total_steps", "steps", "config", "seed", "bc_weight")},
            "rank_bc_states": [rank["bc"] for rank in payload["rank_states"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, repo, output = (p.resolve() for p in (args.run_dir, args.runtime_repo, args.output))
    if output.exists() or any(output.is_relative_to(root/n) for n in ("replay", "checkpoints")):
        parser.error("Use a new independent acceptance report")
    before = (root/"state.json").read_bytes()
    state = json.loads(before)
    config = json.loads((root/"request.json").read_text())
    if state["status"] != "complete" or Path(config["paths"]["genmo_repo"]).resolve() != repo:
        parser.error("Require a completed run and its original runtime worktree")
    sys.path.insert(0, str(repo))
    import torch
    from gem.closedloop.awr.config import physical_config, preflight
    from gem.closedloop.awr.checkpoint import Checkpoints
    from gem.closedloop.common.supervision import load_actor
    from gem.closedloop.common.evaluation import _model_fingerprint
    torch.set_num_threads(4)
    started = time.perf_counter()
    audit = preflight(config, check_gpus=False)
    if state["source_commits"] != {k: v["commit"] for k, v in audit["repositories"].items()}:
        raise ValueError("Checkpoint verification must use original running sources")
    if state["configuration_assets"] != audit["configuration_assets"]:
        raise ValueError("Frozen physical configuration changed")
    actor, _, _ = load_actor(physical_config(state["config"], run_id=root.name, device="cpu"))
    checkpoints = Checkpoints(root/"checkpoints",
        dict(request=state["request_sha256"], sources=state["source_commits"]))
    prefix, reports = f"macro{state['macro']-1:03d}", {}
    paired = None
    for method in ("self_distillation", "awr"):
        reference = state["artifacts"].get(prefix+"_"+method+"_final")
        if reference is None:
            continue
        payload = checkpoints.load(reference)["payload"]
        if paired is None:
            paired = pairing_state(payload)
        else:
            assert_tree_equal(pairing_state(payload), paired)
        report = restore_components(actor, payload)
        fingerprint = _model_fingerprint(actor)
        evaluations = {}
        for plan in ("fixed", "all"):
            key = prefix+"_"+method+"_"+plan
            if key in state["artifacts"]:
                evaluation = json.loads(Path(state["artifacts"][key]).read_text())
                if evaluation["actor_fingerprint"] != fingerprint:
                    raise ValueError("Restored Actor differs from physically evaluated Actor")
                evaluations[plan] = evaluation["plan_identity"]
        if "fixed" not in evaluations:
            raise ValueError("No fixed physical evaluation binds the restored Actor")
        reports[method] = dict(report, checkpoint=reference,
            restored_actor_fingerprint=fingerprint, evaluation_fingerprint_matches=True,
            evaluation_plans=evaluations, random_states=check_rank_random_states(payload["rank_states"]))
    if (root/"state.json").read_bytes() != before:
        raise ValueError("Read-only verification observed a run state change")
    report = dict(schema="genmo.awr.real_checkpoint_components.v1", source_run=str(root),
        source_commits=state["source_commits"], request_sha256=state["request_sha256"],
        methods=reports, seconds=time.perf_counter()-started, optimizer_updates=0,
        physical_controls=0, state_unchanged=True,
        paired_data_budget_bc_random_streams_exact=len(reports) == 2,
        scope="cpu_component_restore_not_new_distributed_training",
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
