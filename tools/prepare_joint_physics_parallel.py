"""多GPU真实sidecar预处理协调器：CPU核验清单，独立Newton worker，原子汇总进度。

协调器在CPU加载冻结SM/WM资产和原数据目录，核验已发布分片与exclude_sidecars，
把尚未处理的source按确定顺序轮流分配给GPU。每GPU仅启动一个批量Newton进程，
调用原prepare入口；环境、冻结Actor/WM、时间与终止算法全部由既有模块承担。
整个worker生命周期持有根目录preparation_lock，防止旧单进程或第二协调器重复写入。
每个worker独立保存worker_XX分片，由已有merge_worker持锁原子追加主索引。

启动前及实际spawn前检查所选GPU的compute进程；发现占用即拒绝，不停止现有任务。
来源清单和分配持久保存，恢复必须保留worker数量；只重做尚未发布的source，失败不无限
重启。progress.json按已经完整发布的episode统计source/s、真实控制步/s、有效Future
请求/s、H32窗口与ETA，保留失败/提前终止。summary.json明确完整覆盖或benchmark边界。
默认取消旧小样本/控制步预算并处理完整catalog；显式--benchmark才允许有界预算。
本工具不修改GENMO网络、WM权重或训练Loss，不创建训练任务，也不宣称缓存完成证明质量。
"""

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST_SCHEMA = "genmo.joint_physics.batched_preparation.v1"


def atomic_json(path, value, *, replace=True):
    path = Path(path)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            temporary.replace(path)
        else:
            os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_gpus(value):
    gpus = value.split(",")
    if not gpus or any(not item.isdigit() for item in gpus) or len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus需要互不重复的物理GPU编号，例如0,1,2,3,4,5,6,7")
    return gpus


def require_idle_gpus(gpus, *, runner=subprocess.run):
    """以GPU UUID关联compute进程，避免CUDA_VISIBLE_DEVICES重映射混淆物理卡号。"""
    def query(argument):
        result = runner(["nvidia-smi", argument, "--format=csv,noheader,nounits"],
                        check=True, text=True, capture_output=True, timeout=20)
        return [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    inventory = {row[0].strip(): row[1].strip() for row in query("--query-gpu=index,uuid")}
    missing = set(gpus)-inventory.keys()
    if missing:
        raise RuntimeError(f"找不到GPU编号: {sorted(missing)}")
    selected = {inventory[gpu]: gpu for gpu in gpus}
    occupied = []
    for row in query("--query-compute-apps=gpu_uuid,pid,process_name"):
        gpu_uuid, pid, name = (part.strip() for part in row[:3])
        if gpu_uuid in selected:
            occupied.append({"gpu": selected[gpu_uuid], "pid": pid, "process": name})
    if occupied:
        raise RuntimeError("所选GPU有现有compute任务，拒绝抢占: "+json.dumps(occupied, ensure_ascii=False))
    return {gpu: inventory[gpu] for gpu in gpus}


def source_keys(catalog, prepare_options):
    from gem.closedloop.joint_physics.prepare_rollouts import preparation_sources
    result = []
    for dataset, row in preparation_sources(catalog, prepare_options):
        result.append(json.dumps([dataset.dataset_name, dataset.split, dataset.rows[row]["sample_id"]],
                                 ensure_ascii=False, separators=(",", ":")))
    if len(set(result)) != len(result):
        raise ValueError("原数据catalog存在重复source身份")
    return result


def assigned_sources(keys, skip, world_size, max_sources=None):
    todo = [key for key in keys if key not in skip]
    if max_sources is not None:
        todo = todo[:max_sources]
    # 原清单按四来源交错，直接8卡stride会固定每rank的数据集；逐块轮换接收rank。
    assignments = {str(rank): [] for rank in range(world_size)}
    for index, key in enumerate(todo):
        block, position = divmod(index, world_size)
        assignments[str((position+block) % world_size)].append(key)
    return assignments


def episode_counts(index, keys=None):
    episodes = [item for item in index.get("episodes", []) if keys is None or item["source_key"] in keys]
    result = {"sources": len(episodes), "control_steps": 0, "valid_future_requests": 0,
              "legal_h32_windows": 0, "termination_reasons": {}}
    for item in episodes:
        result["control_steps"] += int(item["control_steps"])
        preprocessing = item.get("offline_preprocessing", {})
        result["valid_future_requests"] += int(preprocessing.get("valid_requests", 0))
        result["legal_h32_windows"] += int(preprocessing.get("legal_windows", {}).get("32", 0))
    result["termination_reasons"] = dict(Counter(item["termination_reason"] for item in episodes))
    result["source_or_reference_end_count"] = result["termination_reasons"].get("source_or_reference_end", 0)
    result["explicit_control_budget_count"] = result["termination_reasons"].get("explicit_control_budget", 0)
    result["failed_termination_count"] = sum(value for reason, value in result["termination_reasons"].items()
                                              if reason not in ("source_or_reference_end", "explicit_control_budget"))
    return result


def progress_payload(index, manifest, baseline_keys, elapsed, processes):
    target = {key for values in manifest["assignments"].values() for key in values}
    published = set(index["sources"])
    complete = target & published
    current = episode_counts(index, complete-baseline_keys)
    total = episode_counts(index, complete)
    rate = current["sources"]/max(elapsed, 1e-9)
    return {"status": "running", "wall_seconds": elapsed, "target_sources": len(target),
            "completed_sources": len(complete), "remaining_sources": len(target-complete),
            "session": current, "total": total, "source_per_second": rate,
            "control_steps_per_second": current["control_steps"]/max(elapsed, 1e-9),
            "future_requests_per_second": current["valid_future_requests"]/max(elapsed, 1e-9),
            "eta_seconds": (len(target-complete)/rate if rate else None),
            "workers": [{"rank": rank, "pid": p.pid, "returncode": p.poll()} for rank, p in processes.items()],
            "updated_unix_seconds": time.time()}


def records_from_verified_shards(index, directory, validate_request):
    """旧转换v2索引只有前瞻摘要；从已核验真实分片还原统计，每个shard只加载一次。"""
    import torch
    grouped = {}
    for source, location in index["sources"].items():
        shard = location["shard"]
        if shard not in index["shards"]:
            raise ValueError(f"source引用未核验分片: {shard}")
        grouped.setdefault(shard, []).append((source, int(location["episode"])))
    records = []
    for shard, locations in grouped.items():
        payload = torch.load(Path(directory)/shard, map_location="cpu", weights_only=False, mmap=True)
        for source, offset in locations:
            if not 0 <= offset < len(payload):
                raise ValueError(f"排除缓存episode索引越界: {source}")
            episode = payload[offset]
            if episode["source_key"] != source or episode["split"] != index["split"]:
                raise ValueError(f"排除缓存真实episode身份/split错误: {source}")
            steps = int(episode["control_steps"])
            if steps < 0 or len(episode["action"]) != steps:
                raise ValueError(f"排除缓存真实控制步数与动作长度不一致: {source}")
            requests = episode["offline_requests"]
            if len(requests["valid"]) != len(requests["frames"]):
                raise ValueError(f"排除缓存请求valid/frames长度不同: {source}")
            records.append({"source_key": source, "control_steps": steps,
                "termination_reason": episode["termination_reason"],
                "offline_preprocessing": {"valid_requests": int(requests["valid"].sum()),
                    "legal_windows": {"32": sum(validate_request(episode, i, 32)[1] is None
                                                 for i in range(len(requests["frames"])))}}})
        del payload
    return records


def verify_excluded(cfg, split, expected):
    from gem.closedloop.joint_physics.data import SidecarStore
    from gem.closedloop.joint_physics.contracts import digest
    from gem.closedloop.joint_physics.requests import static_request
    keys, directories, episodes = set(), [], []
    for name in cfg["prepare"].get("exclude_sidecars", []):
        directory = Path(name).resolve()
        store = SidecarStore(str(directory), split, expected["assets"], sample_contract=expected["sample_contract"])
        for shard, sha in store.index["shards"].items():
            if digest(directory/shard) != sha:
                raise ValueError(f"排除缓存已发布分片损坏: {directory/shard}")
        if keys & set(store.index["sources"]):
            raise ValueError("exclude_sidecars之间存在重复source，禁止重复统计或混合训练")
        keys.update(store.index["sources"])
        # 转换生成的旧v2 index.episodes仅有offline摘要，不能缺字段就按0统计。
        # 数据真实性来自不可变分片中的episode，H32沿用训练的同一静态时间支持检查。
        episodes.extend(records_from_verified_shards(store.index, directory, static_request))
        directories.append(str(directory))
    return keys, directories, episode_counts({"episodes": episodes})


def initialize(cfg, args, output, adapter, catalog):
    from gem.closedloop.joint_physics.batched_prepare import empty_index, verify_index
    from gem.closedloop.joint_physics.prepare_rollouts import publish_index
    expected = empty_index(cfg, catalog, adapter, args.split)
    index_path = output/"index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else expected
    verify_index(index, expected, output)
    excluded, exclude_directories, excluded_counts = verify_excluded(cfg, args.split, expected)
    if set(index["sources"]) & excluded:
        raise ValueError("主索引与exclude_sidecars重复保存source，不能继续扩充")
    keys = source_keys(catalog, cfg["prepare"])
    manifest_path = output/"batch_manifest.json"
    identity = {key: expected[key] for key in ("schema", "split", "assets", "dataset", "sample_contract")}
    budget = {"benchmark": args.benchmark, "max_sources": args.max_sources,
              "max_control_steps": args.max_control_steps}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema") != MANIFEST_SCHEMA or manifest["world_size"] != len(args.gpus)
                or manifest["identity"] != identity or manifest["catalog_source_keys"] != keys
                or set(manifest["excluded_source_keys"]) != excluded or manifest["budget"] != budget):
            raise ValueError("续处理清单/worker数量/数据资产/预算不同，禁止重新分配或覆盖；使用原参数或独立目录")
    else:
        if (output/"batch_config.json").exists():
            raise ValueError("发现无manifest的batch_config，保留现场并核对中断状态")
        manifest = {"schema": MANIFEST_SCHEMA, "world_size": len(args.gpus), "identity": identity,
                    "catalog_source_keys": keys, "excluded_source_keys": sorted(excluded),
                    "exclude_sidecars": exclude_directories, "excluded_counts": excluded_counts,
                    "baseline_source_keys": sorted(index["sources"]),
                    "assignments": assigned_sources(keys, set(index["sources"])|excluded, len(args.gpus), args.max_sources),
                    "budget": budget, "created_unix_seconds": time.time()}
        if not index_path.exists():
            publish_index(output, index)
        atomic_json(manifest_path, manifest, replace=False)
    config_path = output/"batch_config.json"
    if config_path.exists():
        persisted = json.loads(config_path.read_text())
        if persisted["prepare"] != cfg["prepare"]:
            raise ValueError("续处理prepare选项与持久配置不同，禁止静默改变数据合同")
    else:
        atomic_json(config_path, cfg, replace=False)
    return manifest, index, config_path


def launch_and_wait(args, output, manifest, index, config_path):
    from gem.closedloop.joint_physics.batched_prepare import verify_index
    baseline = set(index["sources"])
    gpu_identity = require_idle_gpus(args.gpus)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    started, children, handles, launches = time.perf_counter(), {}, [], []
    launch_path = output/f"launch_{session_id}.json"
    session = {"session_id": session_id, "gpu_identity": gpu_identity, "num_envs": args.num_envs,
               "world_size": len(args.gpus), "started_unix_seconds": time.time(), "workers": launches}
    try:
        for rank, gpu in enumerate(args.gpus):
            worker_dir = output/f"worker_{rank:02d}"
            worker_dir.mkdir(exist_ok=True)
            log_path = worker_dir/f"stdout_{session_id}.log"
            handle = log_path.open("x")
            handles.append(handle)
            command = [sys.executable, "tools/prepare_joint_physics_data.py", "--config", str(config_path),
                       "--split", args.split, "--output", str(output), "--torch-device", "cuda:0",
                       "--num-envs", str(args.num_envs), "--worker-rank", str(rank),
                       "--worker-count", str(len(args.gpus)), "--chunk-steps", str(args.chunk_steps),
                       "--forecast-batch-size", str(args.forecast_batch_size)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
            # 只创建本次拥有的子进程；任何其它现有GPU任务都不在控制范围内。
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            children[rank] = process
            launches.append({"rank": rank, "gpu": gpu, "pid": process.pid, "command": command, "stdout": str(log_path)})
            atomic_json(launch_path, session)
        while True:
            live = any(process.poll() is None for process in children.values())
            index = json.loads((output/"index.json").read_text())
            progress = progress_payload(index, manifest, baseline, time.perf_counter()-started, children)
            failed = {rank: process.returncode for rank, process in children.items() if process.returncode not in (None, 0)}
            if failed:
                progress["status"] = "running_with_failed_workers" if live else "failed"
                progress["failed_workers"] = failed
            atomic_json(output/"progress.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            if not live:
                break
            time.sleep(args.poll_seconds)
        assigned = {key for values in manifest["assignments"].values() for key in values}
        missing = sorted(assigned-set(index["sources"]))
        expected = manifest["identity"]
        verify_index(index, expected, output)
        unexpected = set(index["sources"])-set(manifest["baseline_source_keys"])-assigned
        if unexpected:
            raise ValueError("协调器运行中主索引出现未分配source")
        covered = set(index["sources"])|set(manifest["excluded_source_keys"])
        catalog = set(manifest["catalog_source_keys"])
        worker_summaries = {}
        for rank, process in children.items():
            worker_summary_path = output/f"worker_{rank:02d}"/"summary.json"
            if process.returncode == 0 and worker_summary_path.exists():
                worker_summaries[str(rank)] = json.loads(worker_summary_path.read_text())
        process_wall = time.perf_counter()-args.coordinator_started
        summary = {**progress, "status": "failed" if failed or missing else "completed",
                   "manifest": str(output/"batch_manifest.json"), "launch": str(launch_path),
                   "num_envs": args.num_envs, "gpus": args.gpus, "benchmark": args.benchmark,
                   "catalog_sources": len(catalog), "covered_catalog_sources": len(catalog & covered),
                   "uncovered_catalog_sources": len(catalog-covered), "missing_assigned_sources": missing,
                   "all_published": episode_counts(index),
                   "all_published_legal_h32": episode_counts(index)["legal_h32_windows"],
                   "excluded": manifest.get("excluded_counts", {}),
                   "excluded_legal_h32": manifest.get("excluded_counts", {}).get("legal_h32_windows", 0),
                   "frozen_unchanged": len(worker_summaries) == len(children) and all(
                       value.get("frozen_unchanged", False) for value in worker_summaries.values()),
                   "worker_summaries": worker_summaries, "process_wall_seconds": process_wall,
                   "process_source_per_second": progress["session"]["sources"]/max(process_wall, 1e-9),
                   "process_control_steps_per_second": progress["session"]["control_steps"]/max(process_wall, 1e-9),
                   "process_future_requests_per_second": progress["session"]["valid_future_requests"]/max(process_wall, 1e-9),
                   "complete_index_verified": True}
        if (not args.benchmark and catalog-covered) or not summary["frozen_unchanged"]:
            summary["status"] = "failed"
        atomic_json(output/"summary.json", summary)
        atomic_json(output/"progress.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0 if summary["status"] == "completed" else 1
    except BaseException as error:
        # 只回收本协调器启动且仍存活的子进程，已发布原子分片完整保留供显式恢复。
        for process in children.values():
            if process.poll() is None:
                process.terminate()
        for process in children.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        atomic_json(output/"coordinator_failure.json", {"session_id": session_id, "error": repr(error),
                    "workers": [{"rank": rank, "pid": p.pid, "returncode": p.returncode} for rank, p in children.items()]})
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    coordinator_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--num-envs", type=int, choices=(256, 512, 1024), default=512)
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--forecast-batch-size", type=int, default=512)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--initialize-only", action="store_true", help="仅CPU核验资产、4287来源及清单；不查询或占用GPU，不启动worker")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--max-control-steps", type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.)
    args = parser.parse_args()
    args.coordinator_started = coordinator_started
    if ((args.max_sources is not None or args.max_control_steps is not None) and not args.benchmark):
        parser.error("--max-sources/--max-control-steps仅允许显式--benchmark；正式预处理默认全库完整轨迹")
    if args.benchmark and args.max_sources is None:
        parser.error("benchmark必须明确--max-sources，并使用独立目录")
    if any(value is not None and value <= 0 for value in (args.max_sources, args.max_control_steps,
                                                        args.chunk_steps, args.forecast_batch_size, args.poll_seconds)):
        parser.error("预算、batch和检查间隔必须为正数")
    if not args.initialize_only:
        require_idle_gpus(args.gpus)
    # 协调器绝不建立CUDA上下文；worker显式覆盖其独立CUDA_VISIBLE_DEVICES。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from gem.closedloop.joint_physics.config import read_config, build_adapter
    cfg = read_config(args.config, args.set)
    cfg["prepare"].update(max_sources=None, max_sources_per_dataset=None,
                          max_control_steps=args.max_control_steps, resume=True)
    import torch
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    adapter = build_adapter(cfg, "cpu")
    from gem.closedloop.joint_physics.data import SourceCatalog
    from gem.closedloop.joint_physics.prepare_rollouts import preparation_lock
    catalog = SourceCatalog(cfg, args.split)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with preparation_lock(output):
        manifest, index, config_path = initialize(cfg, args, output, adapter, catalog)
        if args.initialize_only:
            print(json.dumps({"status": "initialized_only", "output": str(output),
                "catalog_sources": len(manifest["catalog_source_keys"]),
                "previously_published_sources": len(index["sources"]),
                "excluded_sources": len(manifest["excluded_source_keys"]),
                "assigned_sources": sum(len(items) for items in manifest["assignments"].values()),
                "world_size": len(args.gpus), "cuda_context_created": False,
                "workers_started": 0, "manifest": str(output/"batch_manifest.json")}, ensure_ascii=False), flush=True)
            return 0
        return launch_and_wait(args, output, manifest, index, config_path)


if __name__ == "__main__":
    raise SystemExit(main())
