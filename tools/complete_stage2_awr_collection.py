"""在明确追加预算下补足AWR完整Episode，保留原任务与全部失败证据。

此入口仅修复尚未开始Critic/Actor更新的采集数量缺口。原固定计划、request、源码
提交、冻结资产、物理配置和已有Replay保持原身份；额外候选使用原来源采样器及
初始化机制，单独保存不可变计划、用户授权说明、工具代码摘要和每批尝试预留记录。
每批最多发出当前缺口数量；首次决策后的真实失败仍是有效完整Episode，不按成功
筛数据。预热失败没有虚构动作或MC，原始快照继续保留。中断批次已预留的任务也
占用尝试预算，恢复不反复重试同一失败直到成功。

工具可归档在运行acceptance目录，显式导入原运行提交的独立GENMO工作树；不替换
正在运行的源码，不关闭源码身份检查。必须等原采集进程退出、八卡空闲后启动。
最终使用原Replay校验器重新核验所有分片，发布精确数量的快照及采集修订报告，
之后可在同进程继续原五阶段实现；正常resume也可读取此已审计的阶段产物。
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def next_wave(candidates, reserved_ids, completed, target, limit):
    """预留计数是物理尝试的保守上界，永远不超额发任务。"""
    if not 0 <= completed <= target or not 0 < limit == len(candidates):
        raise ValueError("Invalid complete Episode target or extra-attempt limit")
    ids = [t["task_id"] for t in candidates]
    if len(set(ids)) != len(ids) or len(set(reserved_ids)) != len(reserved_ids):
        raise ValueError("Duplicate supplemental task/reservation")
    if not set(reserved_ids) <= set(ids):
        raise ValueError("Reservation was not in the immutable supplemental plan")
    count = min(target-completed, limit-len(reserved_ids))
    return [t for t in candidates if t["task_id"] not in set(reserved_ids)][:count]


def initial_coverage(plan, outcomes, completed_ids):
    """原配额必须全部执行；只允许真实预热失败形成待补数量。"""
    expected = {t["task_id"] for t in plan["tasks"]}
    if len(expected) != len(plan["tasks"]) or plan["plan_sha256"] != sha(plan["tasks"]):
        raise ValueError("Original task plan identity changed")
    selected = [v for v in outcomes if v["task_id"] in expected]
    counts = Counter(v["task_id"] for v in selected)
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise ValueError("Original tasks must finish exactly once before supplemental collection")
    for value in selected:
        published = value["task_id"] in completed_ids
        if bool(value["complete"]) != published:
            raise ValueError("Original outcome and complete Episode publication differ")
        if not published and not (value["reason"] == "physical_warmup_failure"
                                  and value.get("physical_failure") is True):
            raise ValueError("Missing/administrative task requires normal recovery, not replacement")
    return selected


def read_outcomes(root, macro):
    return [v for path in sorted((root/f"macro{macro:03d}"/"collection").glob(
        "*/rank*/execution/outcomes.json")) for v in json.loads(path.read_text())]


def receipts(root, macro):
    values = [json.loads(path.read_text())
              for path in (root/"replay"/"publications").glob("*.json")]
    values = [v for v in values if v["macro"] == macro]
    tasks = [v["task_id"] for v in values]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Two physical Episodes claim the same task")
    return values


def immutable_json(path, value):
    from gem.closedloop.awr.io import atomic_json
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refuse changing immutable supplemental evidence: {path}")
    else:
        atomic_json(path, value)


def supplemental_candidates(run, original, limit):
    from gem.closedloop.awr.tasks import training_tasks
    config = copy.deepcopy(run.request)
    if config["awr"]["profile"] != "pilot":
        raise ValueError("Supplemental sampling is for the pilot source sampler, not exploration pairs")
    macro = run.state["macro"]
    config["awr"]["seed"] += 9000001
    config["awr"]["initial_episodes" if macro == 0 else "new_episodes"] = limit
    raw = training_tasks(run.catalog, run.folds, config, run.d.world_size, macro)["tasks"]
    per_rank, numbered = Counter(), []
    for task in raw:
        task["supplemental_origin_plan"] = original["plan_sha256"]
        task["task_id"] = sha({k: v for k, v in task.items()
                              if k not in ("task_id", "noise_index_start")})
        task["noise_index_start"] = int(sha([task["task_id"], "generation"])[:12], 16)
        numbered.append((per_rank[task["rank"]], task["rank"], task))
        per_rank[task["rank"]] += 1
    # 同一批先覆盖各rank，不改变已采样的音乐、初态或种子。
    return [t for _, _, t in sorted(numbered, key=lambda item: item[:2])]


def complete_collection(run, args, tool_identity):
    from gem.closedloop.awr.collection import CompleteCollector
    from gem.closedloop.awr.io import atomic_json, read_tree
    from gem.closedloop.awr.world import open_world, close_isolated_world, budget_for_tasks
    from gem.closedloop.awr.checkpoint import stop_requested

    macro, d = run.state["macro"], run.d
    prefix = f"macro{macro:03d}"
    replay_key, approval_key = prefix+"_replay", prefix+"_collection_amendment"
    if replay_key in run.state["artifacts"]:
        if approval_key not in run.state["artifacts"]:
            raise ValueError("Collection was already sealed without this supplemental protocol")
        return True
    if any(k.startswith(prefix) and any(s in k for s in (
        "_labels", "_critic_progress", "_critics", "_awr_", "_self_distillation_"))
           for k in run.state["artifacts"]):
        raise ValueError("Never replace the data of a Critic/Actor stage that has already started")
    original = json.loads(Path(run.state["artifacts"][prefix+"_tasks"]).read_text())
    target = len(original["tasks"])
    folder = run.root/prefix/"supplement"
    d.root(lambda: folder.mkdir(parents=True, exist_ok=True))
    run.publish(phase="collect", status="running")
    run.config["awr"]["actor_version"] = macro
    run.runtime_config["awr"]["actor_version"] = macro
    if macro != 0:
        previous = run.checkpoints.load(run.state["artifacts"][f"macro{macro-1:03d}_awr_final"])
        run.actor.load_state_dict(previous["payload"]["actor"])
    fingerprint = d.assert_models_equal(run.actor)
    # 中断可能发生在数组原子发布与回执之间，先恢复回执再决定数量，避免超采。
    d.root(lambda: run._recover_publications(macro))

    def plan_or_create():
        path = folder/"plan.json"
        current = receipts(run.root, macro)
        outcomes = read_outcomes(run.root, macro)
        base = initial_coverage(original, outcomes, {v["task_id"] for v in current})
        identity = dict(original_plan_sha256=original["plan_sha256"], target_episodes=target,
            max_extra_attempts=args.max_extra_attempts, authorization=args.authorization,
            request_sha256=run.state["request_sha256"], source_commits=run.state["source_commits"],
            tool=tool_identity, actor_fingerprint=fingerprint)
        if path.exists():
            plan = json.loads(path.read_text())
            if plan["identity"] != identity or plan["plan_sha256"] != sha(plan["tasks"]):
                raise ValueError("Supplemental plan/authorization/tool identity changed on resume")
            return plan
        if sum(t["complete"] for t in base) >= target:
            raise ValueError("No original collection deficit requires extra attempts")
        candidates = supplemental_candidates(run, original, args.max_extra_attempts)
        plan = dict(schema="genmo.awr.supplemental_plan.v1", identity=identity,
                    tasks=candidates, plan_sha256=sha(candidates), created_unix=time.time())
        immutable_json(folder/"before_state.json", run.state)
        immutable_json(path, plan)
        return plan

    plan = d.root(plan_or_create)
    started = time.perf_counter()
    while True:
        def reserve():
            current = receipts(run.root, macro)
            reservations = [json.loads(p.read_text()) for p in sorted(folder.glob("wave*.json"))]
            used = [task for wave in reservations for task in wave["task_ids"]]
            chosen = next_wave(plan["tasks"], used, len(current), target, args.max_extra_attempts)
            if not chosen:
                return None
            wave = dict(index=len(reservations), attempt="supplement_"+uuid.uuid4().hex[:8],
                        task_ids=[t["task_id"] for t in chosen], tasks=chosen,
                        complete_before=len(current), reserved_unix=time.time(),
                        plan_sha256=plan["plan_sha256"], tool=tool_identity)
            immutable_json(folder/f"wave{wave['index']:03d}.json", wave)
            return wave
        if d.root(lambda: stop_requested(run.root)):
            run.publish(status="stopped", phase="collect")
            return False
        wave = d.root(reserve)
        if wave is None:
            break
        local_tasks = [t for t in wave["tasks"] if t["rank"] == d.rank]
        path = run.root/prefix/"collection"/wave["attempt"]/f"rank{d.rank:02d}"
        handle, driver = None, None
        wave_started = time.perf_counter()
        def local():
            nonlocal handle, driver
            if not local_tasks:
                return dict(tasks=[], seconds=0.)
            slots = min(run.config["runtime"]["num_envs"], len(local_tasks))
            handle = open_world(run.runtime_config, path/"world", d, num_envs=slots)
            driver = CompleteCollector(run.runtime_config, run.policy, run.builder, handle.world,
                budget_for_tasks(local_tasks, slots), path/"execution", num_envs=slots,
                actor_fingerprint=fingerprint)
            def publish(episode):
                value = run.replay.publish(episode)
                value.update(task_id=episode.task["task_id"], macro=macro,
                             published_unix_ns=time.time_ns())
                atomic_json(run.root/"replay"/"publications"/(episode.episode_id+".json"), value)
                return value
            return driver.collect(local_tasks, run.catalog, run.contract, on_episode=publish,
                                  should_stop=lambda: stop_requested(run.root))
        try:
            report = d.local(local)
        finally:
            if driver: driver.close()
            if handle: close_isolated_world(handle)
        reports = d.gather(report)
        run.json_artifact(prefix+"_collection_"+wave["attempt"],
            lambda: dict(ranks=reports, seconds=time.perf_counter()-wave_started,
                         supplemental_plan_sha256=plan["plan_sha256"], tool=tool_identity))
    if d.root(lambda: stop_requested(run.root)):
        run.publish(status="stopped", phase="collect")
        return False

    def seal():
        # 原生产路径重新验证每条Episode的SHA、连续性、完整MC、资产与留出隔离。
        completed = run._recover_publications(macro)
        current = receipts(run.root, macro)
        outcomes = read_outcomes(run.root, macro)
        original_outcomes = initial_coverage(original, outcomes, set(completed))
        reservations = [json.loads(p.read_text()) for p in sorted(folder.glob("wave*.json"))]
        reserved = [task for wave in reservations for task in wave["task_ids"]]
        allowed = {t["task_id"] for t in original["tasks"]} | set(reserved)
        if len(reserved) > args.max_extra_attempts or not set(completed) <= allowed:
            raise ValueError("Published data exceeds the authorized supplemental plan")
        if len(completed) != target:
            report = dict(completed=len(completed), target=target, reserved_extra_attempts=len(reserved),
                          max_extra_attempts=args.max_extra_attempts, full_outcomes=outcomes)
            immutable_json(folder/"exhausted.json", report)
            raise RuntimeError(f"Extra-attempt budget exhausted: {len(completed)}/{target}; failures retained")
        by_task = {v["task_id"]: v for v in outcomes}
        if len(by_task) != len(outcomes) or not set(by_task) <= allowed:
            raise ValueError("Duplicate or unplanned physical outcomes")
        for value in current:
            if value["task_id"] not in by_task or not by_task[value["task_id"]]["complete"]:
                raise ValueError("Complete Episode has no authoritative successful publication outcome")
        # 失败不是只留一个计数：核验每个预热失败原始二进制快照仍在且有真实终止。
        failure_evidence = []
        for outcome_path in sorted((run.root/prefix/"collection").glob("*/rank*/execution/outcomes.json")):
            for outcome in json.loads(outcome_path.read_text()):
                if outcome["reason"] != "physical_warmup_failure":
                    continue
                path = outcome_path.parent/"excluded"/outcome["task_id"]
                value = read_tree(path)
                snapshot = value["snapshot"]
                if not snapshot["terminated"] or snapshot["truncated"]:
                    raise ValueError("Warmup exclusion lacks a true physical termination")
                failure_evidence.append(dict(task_id=outcome["task_id"], path=str(path),
                    manifest_sha256=json.loads((path/"manifest.json").read_text())["sha256"]))
        snapshot = run.replay.snapshot()
        if macro == 0 and len(snapshot.episode_ids) != target:
            raise ValueError("First-macro FIFO snapshot lost accepted complete Episodes")
        report = dict(schema="genmo.awr.collection_amendment.v1", identity=plan["identity"],
            supplemental_plan_sha256=plan["plan_sha256"], original_tasks=len(original["tasks"]),
            original_complete=sum(t["complete"] for t in original_outcomes),
            reserved_extra_attempts=len(reserved),
            confirmed_extra_attempts=sum(t["task_id"] in set(reserved) for t in outcomes),
            total_confirmed_attempts=len(outcomes), complete_episodes=len(completed),
            excluded_tasks=sum(not t["complete"] for t in outcomes),
            reasons=dict(Counter(t["reason"] for t in outcomes)), failure_evidence=failure_evidence,
            unused_or_interrupted_reservations=sorted(set(reserved)-set(by_task)),
            full_outcomes=outcomes, replay_snapshot=snapshot.to_dict(),
            current_invocation_seconds=time.perf_counter()-started,
            original_request_and_task_plan_unchanged=True, physics_source_commits_unchanged=True)
        immutable_json(folder/"acceptance.json", report)
        return snapshot.to_dict()
    snapshot = d.root(seal)
    run.artifact(approval_key, str(folder/"acceptance.json"))
    run.artifact(replay_key, snapshot["snapshot_id"])
    run.publish(phase="collection_complete")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--runtime-repo", required=True, type=Path)
    parser.add_argument("--max-extra-attempts", required=True, type=int)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--tool-commit", required=True)
    parser.add_argument("--continue-training", action="store_true")
    args = parser.parse_args()
    if (args.max_extra_attempts < 1 or not args.authorization.strip()
            or len(args.tool_commit) != 40
            or any(c not in "0123456789abcdef" for c in args.tool_commit)):
        parser.error("A positive explicit attempt limit and authorization record are required")
    root, runtime = args.run_dir.resolve(), args.runtime_repo.resolve()
    sys.path.insert(0, str(runtime))
    config = json.loads((root/"request.json").read_text())
    if Path(config["paths"]["genmo_repo"]).resolve() != runtime:
        parser.error("Use the original run's GENMO worktree, with its original source commit")
    rank, world, local = [int(os.environ.get(k, "-1")) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if world != 8 or rank != local or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7":
        parser.error("Use the same single-node eight-GPU torchrun environment as the original run")
    import torch
    import torch.distributed as dist
    from gem.closedloop.awr.config import preflight
    from gem.closedloop.awr.distributed import Distributed
    from gem.closedloop.awr.resources import GPUResourceMonitor
    from gem.closedloop.awr.runtime import AWRRun
    dist.init_process_group("gloo", timeout=timedelta(hours=2))
    group, monitor, run, lock = None, None, None, None
    try:
        d = Distributed(device=f"cuda:{local}")
        audit = d.root(lambda: preflight(config))
        def acquire():
            nonlocal lock
            path = root/"acceptance"/"collection_completion.lock"
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = path.open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        d.root(acquire)
        torch.cuda.set_device(local)
        torch.set_num_threads(config["runtime"]["torch_threads"])
        torch.manual_seed(config["awr"]["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        group = dist.new_group(backend="nccl", timeout=timedelta(hours=1))
        d.tensor_group = group
        if float(d.sum(torch.tensor(rank+1., device=d.device))) != 36.:
            raise RuntimeError("Eight-GPU SUM validation failed")
        os.environ.setdefault("ISAACLAB_PATH", config["runtime"]["isaaclab_root"])
        libraries = config["runtime"]["sysroot_library"]
        os.environ["LD_LIBRARY_PATH"] = libraries+os.pathsep+os.environ.get("LD_LIBRARY_PATH", "")
        run = AWRRun(config, root, d, audit, resume=True)
        if rank == 0:
            monitor = GPUResourceMonitor(root).start()
        run.prepare()
        tool = dict(git_commit=args.tool_commit, file=str(Path(__file__).resolve()),
                    sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if complete_collection(run, args, tool) and args.continue_training:
            run.replay.close()
            run.replay = None
            return run.run()
        return 0
    finally:
        if run is not None and run.replay is not None:
            run.replay.close()
        if monitor is not None:
            monitor.close()
        if lock is not None:
            lock.close()
        if group is not None:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
