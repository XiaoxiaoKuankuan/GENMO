"""单GPU真实批量sidecar规模选择：256/512/1024环境完整采集、前瞻、写盘计时。

默认每组相同1024个未缓存source、最多300个50Hz控制步（6秒），逐组调用正式多GPU
协调器的单GPU模式。每组仍完成真实Newton采集、批量SM/WM前瞻、v2分片落盘和SHA核验，
进程端到端时间包含Python/资产/环境初始化，不以只测MLP前向的时间充当预处理吞吐。
各组独立目录，失败/solver溢出组完整保留日志并排除选型；来源清单必须与第一组一致。
只按成功组每秒有效Future请求数选择规模，并报告H32、真实控制步、采集wall和显存。
benchmark产生的6秒截断缓存不注册到正式训练sidecar，不代表整曲效果或全库预处理完成。
本工具不会停止现有GPU任务，也不会自动启动训练；忙卡立即拒绝，所有正式处理另行启动。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_joint_physics_parallel import atomic_json, parse_gpus, require_idle_gpus


def select_result(results):
    eligible = [item for item in results if item.get("eligible", False)]
    if not eligible:
        return None
    # 吞吐相同优先较小环境数，避免无收益的显存/求解器容量增加。
    return max(eligible, key=lambda item: (item["process_future_requests_per_second"], -item["num_envs"]))


def summarize_trial(directory, num_envs, returncode, process_wall, expected_sources):
    directory = Path(directory)
    summary_path, manifest_path = directory/"summary.json", directory/"batch_manifest.json"
    result = {"num_envs": num_envs, "returncode": returncode, "process_wall_seconds": process_wall,
              "output": str(directory), "summary": str(summary_path), "eligible": False}
    sources = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        sources = manifest["assignments"]["0"]
        result["same_sources_as_first_group"] = expected_sources is None or sources == expected_sources
    if not summary_path.exists():
        result["rejection_reason"] = "missing_pipeline_summary"
        return result, sources
    summary = json.loads(summary_path.read_text())
    counts = summary.get("session", {})
    result.update(sources=counts.get("sources", 0), control_steps=counts.get("control_steps", 0),
                  valid_future_requests=counts.get("valid_future_requests", 0),
                  legal_h32_windows=counts.get("legal_h32_windows", 0),
                  termination_reasons=counts.get("termination_reasons", {}),
                  source_per_second=counts.get("sources", 0)/max(process_wall, 1e-9),
                  control_steps_per_second=counts.get("control_steps", 0)/max(process_wall, 1e-9),
                  process_future_requests_per_second=counts.get("valid_future_requests", 0)/max(process_wall, 1e-9),
                  child_process_wall_seconds=summary.get("process_wall_seconds"),
                  complete_index_verified=summary.get("complete_index_verified", False))
    workers = summary.get("worker_summaries", {})
    result["worker_collection_wall_seconds"] = {rank: value.get("wall_seconds") for rank, value in workers.items()}
    result["peak_memory_bytes"] = max((value.get("peak_memory_bytes", 0) for value in workers.values()), default=0)
    result["solver_health"] = {rank: value.get("solver_health") for rank, value in workers.items()}
    if returncode:
        result["rejection_reason"] = "pipeline_process_failed_including_solver_overflow"
    elif summary.get("status") != "completed" or not summary.get("frozen_unchanged") or not result["complete_index_verified"]:
        result["rejection_reason"] = "pipeline_or_frozen_asset_verification_failed"
    elif not result.get("same_sources_as_first_group", False):
        result["rejection_reason"] = "source_manifest_differs_between_groups"
    elif not result["valid_future_requests"] or not result["legal_h32_windows"]:
        result["rejection_reason"] = "no_valid_future_requests_or_h32_windows"
    else:
        result["eligible"] = True
    return result, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="必须为新的独立benchmark目录，不得使用正式sidecar目录")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--sizes", type=int, nargs="+", choices=(256, 512, 1024), default=[256, 512, 1024])
    parser.add_argument("--max-sources", type=int, default=1024)
    parser.add_argument("--max-control-steps", type=int, default=300)
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--forecast-batch-size", type=int, default=512)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    gpus = parse_gpus(args.gpu)
    if len(gpus) != 1 or len(set(args.sizes)) != len(args.sizes):
        parser.error("规模benchmark仅用一张物理GPU，每个规模只运行一次")
    if min(args.max_sources, args.max_control_steps, args.chunk_steps, args.forecast_batch_size) <= 0:
        parser.error("benchmark各项预算必须为正整数")
    require_idle_gpus(gpus)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg_path = str(Path(args.config).expanduser().resolve())
    started = time.perf_counter()
    common = {"created_at": datetime.now().isoformat(), "config": cfg_path, "gpu": args.gpu,
              "sizes": args.sizes, "max_sources": args.max_sources, "max_control_steps": args.max_control_steps,
              "control_hz": 50, "maximum_simulated_seconds_per_source": args.max_control_steps/50,
              "benchmark_only_not_formal_training_cache": True}
    atomic_json(output/"benchmark_config.json", common, replace=False)
    expected_sources, results = None, []
    for size in args.sizes:
        # 前组正常退出仍重新检查，避免外部任务恰好占用后继续抢卡。
        require_idle_gpus(gpus)
        trial = output/f"envs_{size}"
        command = [sys.executable, "tools/prepare_joint_physics_parallel.py", "--config", cfg_path,
                   "--output", str(trial), "--split", "train", "--gpus", args.gpu, "--num-envs", str(size),
                   "--benchmark", "--max-sources", str(args.max_sources), "--max-control-steps", str(args.max_control_steps),
                   "--chunk-steps", str(args.chunk_steps), "--forecast-batch-size", str(args.forecast_batch_size)]
        for override in args.set:
            command.extend(["--set", override])
        log_path = output/f"envs_{size}.coordinator.log"
        before = time.perf_counter()
        with log_path.open("x") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter()-before
        result, sources = summarize_trial(trial, size, process.returncode, elapsed, expected_sources)
        result.update(stdout=str(log_path), command=command)
        if expected_sources is None and sources is not None:
            expected_sources = sources
        results.append(result)
        atomic_json(output/"progress.json", {**common, "completed_trials": len(results), "results": results})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    selected = select_result(results)
    summary = {**common, "results": results, "status": "selected" if selected else "no_eligible_scale",
               "selected_num_envs": selected["num_envs"] if selected else None,
               "selection_metric": "valid_future_requests / observed_complete_process_wall_seconds",
               "selected_process_future_requests_per_second": selected["process_future_requests_per_second"] if selected else None,
               "wall_seconds": time.perf_counter()-started,
               "source_keys": expected_sources, "formal_preprocessing_started": False}
    atomic_json(output/"summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "source_keys"}, ensure_ascii=False), flush=True)
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
