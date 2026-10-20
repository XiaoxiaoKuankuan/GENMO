"""只读汇总AWR运行的训练、物理评价、成本与数据身份。

输入是独立运行目录的state.json及已提交JSON产物，输出紧凑JSON供验收和比较。
不启动GPU、不加载模型或大型Replay、不修改训练状态，也不重新选择最佳模型。
中途运行仅报告已发布证据，缺失阶段保持缺失；物理效果直接引用成对评价判定，
绝不将优化步数、价值拟合或损失下降解释为学习成功。可从新代码读取旧运行报告，
报告中的源码身份始终来自实际运行state，而非本汇总脚本所在提交。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def select(value, keys):
    return {key: value[key] for key in keys if key in value}


def summarize(run_dir):
    root = Path(run_dir).resolve()
    state = json.loads((root / "state.json").read_text())
    result = dict(schema="genmo.awr.acceptance_summary.v1", run_dir=str(root),
        **select(state, ("status", "phase", "macro", "created_unix", "finished_unix",
                         "source_commits", "configuration_assets", "request_sha256",
                         "latest", "best_saved", "best_observed")))
    result["elapsed_wall_seconds"] = state.get("finished_unix", time.time()) - state["created_unix"]
    result["elapsed_scope"] = "run_creation_to_finish_or_snapshot_includes_initialization_and_io"
    result["evaluations"], result["comparisons"], result["training"] = {}, {}, {}
    result["critics"], result["collection"], result["diagnostics"], result["evidence"] = {}, {}, {}, {}
    for name, path in state["artifacts"].items():
        if not isinstance(path, str):
            continue
        target = Path(path)
        if target.suffix != ".json":
            continue
        data = target.read_bytes()
        value = json.loads(data)
        result["evidence"][name] = dict(path=str(target), sha256=hashlib.sha256(data).hexdigest())
        if value.get("schema") == "genmo.awr.physical_evaluation.v1":
            overall = value["aggregate"]["overall"]
            result["evaluations"][name] = dict(
                **select(value, ("actor_fingerprint", "plan_identity", "physics_identity",
                                 "sampler_identity", "source_balanced_reward_per_second",
                                 "failure_reasons", "seconds", "fresh_physics_world")),
                **select(overall, ("episode_count", "music_completion_count", "physical_failure_count",
                                   "infrastructure_failure_count", "administrative_truncation_count",
                                   "executed_seconds", "executed_control_steps", "decision_count",
                                   "reward_sum", "reward_per_executed_second")),
                activity=overall["activity"], latency_seconds=overall["latency_seconds"],
                physical_metrics=value["physical_metrics"]["overall"],
                by_source={source: select(v, ("episode_count", "music_completion_count",
                    "physical_failure_count", "executed_seconds", "reward_sum", "reward_per_executed_second"))
                    for source, v in value["aggregate"]["by_source"].items()})
        elif "balanced_return_delta" in value:
            result["comparisons"][name] = {k: v for k, v in value.items() if k != "paired"}
        elif name.endswith("_training") and "ranks" in value:
            result["training"][name] = value
        elif name.endswith("_critic_report"):
            result["critics"][name] = dict(
                **select(value, ("selected", "seconds", "critic_seconds", "advantage_labeling_seconds",
                                 "fold_identity", "crossfit_identity", "value_semantics")),
                folds=[dict(fold=f["fold"], counts=f["counts"],
                    group_counts={k: len(v) for k, v in f["groups"].items()},
                    mc=select(f["mc"], ("epochs", "seconds", "validation")),
                    fvi=[select(v, ("epochs", "seconds", "validation", "frozen_target_identity"))
                         for v in f["fvi"]]) for f in value["folds"]])
        elif name.endswith("_collection_amendment"):
            result["collection"][name] = {k: v for k, v in value.items() if k != "full_outcomes"}
        elif "_collection_" in name and "ranks" in value:
            outcomes = [task for rank in value["ranks"] for task in rank["tasks"]]
            complete = [task for task in outcomes if task["complete"]]
            decisions = sum(t["states"] for t in complete)
            result["collection"][name] = dict(seconds=value["seconds"],
                complete_episodes=len(complete), excluded_tasks=len(outcomes)-len(complete),
                failure_reasons=dict(Counter(t["reason"] for t in outcomes)),
                transitions=decisions, actor_samples=sum(t["actor_samples"] for t in complete),
                physical_seconds=sum(t["executed_seconds"] for t in complete),
                valid_replans_per_wall_second=sum(t["actor_samples"] for t in complete)/value["seconds"],
                rank_seconds=[v["seconds"] for v in value["ranks"]])
            batches = [b for rank in value["ranks"] for b in rank.get("generation_batches", [])]
            if batches:
                generation_times = [b["generation_seconds"] for b in batches]
                result["collection"][name]["generation"] = dict(
                    batches=len(batches), effective_rows=sum(b["effective_rows"] for b in batches),
                    sum_over_ranks_seconds=sum(generation_times),
                    mean_batch_seconds=sum(generation_times)/len(batches),
                    max_batch_seconds=max(generation_times),
                    scope="batch_wall_latency_sum_over_parallel_ranks_not_whole_run_wall_time")
            result["collection"][name]["world_timing_by_rank"] = [
                select(rank.get("timing", {}), ("active_control_steps", "scene_control_steps",
                    "allocated_physics_steps", "physx_step_wall_seconds", "gmt_wall_seconds",
                    "reward_math_seconds", "diagnostics_seconds", "world_rpc_seconds"))
                for rank in value["ranks"]]
            result["collection"][name]["budget_charged_by_rank"] = [
                rank.get("actual_budget", {}).get("used", {}) for rank in value["ranks"]]
        elif name.endswith(("_advantage_distribution", "_bc_calibration", "_bc_validation",
                            "_parameter_changes", "_replay_cost")):
            result["diagnostics"][name] = value
        elif name.endswith("_exploration"):
            pairs = [p for g in value["groups"] for p in g["pairs"] if p["same_conditions_exact"]]
            result["diagnostics"][name] = dict(
                exact_condition_pairs=len(pairs),
                insufficient_reference_diversity=value["insufficient_reference_diversity"],
                same_condition_joint_rms_rad=[p["joint_rms_rad"] for p in pairs],
                grouped_complete_returns=[dict(group_id=g["group_id"], returns=g["complete_returns"],
                    failures=g["failures"]) for g in value["groups"]])
    index = root / "replay" / "index.sqlite"
    if index.exists():
        with sqlite3.connect("file:" + str(index) + "?mode=ro", uri=True) as database:
            row = database.execute("SELECT COUNT(*),COALESCE(SUM(transitions),0),"
                "COALESCE(SUM(samples),0),COALESCE(SUM(bytes),0),COALESCE(SUM(active),0) FROM episodes").fetchone()
        result["replay"] = dict(zip(("registered_episodes", "transitions", "actor_samples",
                                    "retained_bytes", "active_episodes"), row))
    result["gpu_resources"] = [json.loads(path.read_text())
                              for path in sorted((root / "resources").glob("gpu_*.summary.json"))]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="可选新JSON文件；不覆盖已有证据")
    args = parser.parse_args()
    text = json.dumps(summarize(args.run_dir), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
