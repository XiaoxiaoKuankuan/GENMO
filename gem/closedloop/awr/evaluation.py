"""AWR完整音乐评价、来源均衡音乐组置信区间与模型发布门槛。

直接使用公共真实物理评价流程，所有失败保留。报告绑定Actor、任务计划、冻结资产
和新建世界身份；用音乐组聚类重采样而非把同歌不同种子当独立观察。训练Loss及
Critic指标不参与物理成功判定，未通过门槛时best_saved继续指向Stage1。
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import time
import numpy as np
import torch

from gem.closedloop.common.evaluation import evaluation_episode_flow, aggregate_evaluation, _model_fingerprint
from gem.closedloop.common.world_flow import backend_operation
from gem.closedloop.dppo.evaluation_world import assign_tasks, balanced_rank_tasks
from .collection import CompleteCollector
from .world import open_world, close_isolated_world, budget_for_tasks
from .checkpoint import capture_rng, restore_rng
from .io import atomic_json, write_tree, read_tree, digest
from .physical_metrics import PhysicalMetrics, summarize_metrics


def paired_comparison(baseline, current, *, bootstrap=2000, seed=42):
    if baseline["plan_identity"] != current["plan_identity"]:
        raise ValueError("Paired evaluation tasks changed")
    for key in ("physics_identity", "sampler_identity"):
        if baseline[key] != current[key]:
            raise ValueError(f"Paired evaluation changed {key}")
    b, c = ({e["task_id"]: e for e in report["task_summaries"]} for report in (baseline, current))
    if set(b) != set(c):
        raise ValueError("Paired evaluation omitted tasks")
    groups = {}
    new_failures, deltas = [], []
    for task in sorted(b):
        left, right = b[task], c[task]
        if right["physical_failure"] and not left["physical_failure"]:
            new_failures.append(task)
        delta = right["reward_sum"]-left["reward_sum"]
        groups.setdefault(left["source"], {}).setdefault(left["group_id"], []).append(delta)
        deltas.append(dict(task_id=task, reward_delta=delta,
                           seconds_delta=right["seconds"]-left["seconds"]))
    grouped = {s: np.array([np.mean(v) for v in values.values()]) for s, values in groups.items()}
    rng = np.random.default_rng(seed)
    means = [float(np.mean([rng.choice(v, size=len(v), replace=True).mean() for v in grouped.values()]))
             for _ in range(bootstrap)]
    point = float(np.mean([v.mean() for v in grouped.values()]))
    ci = np.quantile(means, [.025, .975]).tolist()
    left, right = baseline["aggregate"]["overall"], current["aggregate"]["overall"]
    ba = left["activity"]["actual_activity_rad_s"]["mean"]
    ca = right["activity"]["actual_activity_rad_s"]["mean"]
    activity = ca/ba if ba is not None and ca is not None and ba > 0 else None
    gates = dict(completion_not_decreased=right["music_completion_count"] >= left["music_completion_count"],
                 failure_count_not_increased=right["physical_failure_count"] <= left["physical_failure_count"],
                 no_new_paired_failure=not new_failures,
                 duration_not_decreased=right["executed_seconds"] >= left["executed_seconds"]-1e-8,
                 activity_retained=activity is not None and activity >= .8,
                 complete_return_improved=point > 0)
    return dict(paired= deltas, balanced_return_delta=point, balanced_return_95ci=ci,
                confidence_unit="source_balanced_music_group_cluster",
                activity_ratio=activity, new_paired_failures=new_failures, gates=gates,
                acceptable=all(gates.values()), improvement_supported=all(gates.values()) and ci[0] > 0)


def evaluate(config, policy, builder, catalog, plan, path, distributed, *, fingerprint, physics_identity):
    config = copy.deepcopy(config)
    # 原完整评价合同要求每个真实控制步的分项，不能沿用训练的稀疏展示级别。
    config["runtime"].update(evidence_level="full", evaluation_metrics_only=False)
    path = Path(path)
    local_tasks = balanced_rank_tasks(plan, distributed.world_size)[distributed.rank]
    local_path = path/f"rank{distributed.rank:02d}"
    slots = config["evaluation"]["envs_per_rank"]
    rng = capture_rng(distributed.device)
    handle = None; driver = None
    started = time.perf_counter()
    references = []
    def run_local():
        nonlocal handle, driver
        handle = open_world(config, local_path/"world", distributed, num_envs=slots)
        driver = CompleteCollector(config, policy, builder, handle.world, budget_for_tasks(local_tasks, slots),
                                   local_path/"execution", num_envs=slots, actor_fingerprint=fingerprint)
        queues = assign_tasks(local_tasks, slots)
        errors = []
        def flow(slot):
            env = driver.collector.states[slot].resource.env
            for index, task in queues[slot]:
                diagnostics = PhysicalMetrics()
                episode, error = yield from evaluation_episode_flow(env, task, index, plan, catalog=catalog,
                    limit=None, deterministic=True, transition_sink=diagnostics.observe, progress=None)
                episode["awr_physical_metrics"] = diagnostics.result()
                target = local_path/"episodes"/task["task_id"]
                sha = write_tree(target, episode)
                references.append(dict(path=str(target), sha256=sha, task_id=task["task_id"]))
                if error:
                    errors.append(str(error)); break
            yield from backend_operation("retire")
        with torch.no_grad():
            driver.collector._drive({i: flow(i) for i in range(slots)})
        if errors:
            raise RuntimeError("Invalid physical evaluation: "+" | ".join(errors))
    try:
        distributed.local(run_local)
    finally:
        if driver:
            driver.close()
        if handle:
            close_isolated_world(handle)
        restore_rng(rng, distributed.device)
    all_refs = [ref for rank in distributed.gather(references) for ref in rank]
    def publish():
        task_map = {t["task_id"]: t for t in plan["tasks"]}
        if {r["task_id"] for r in all_refs} != set(task_map) or len(all_refs) != len(task_map):
            raise ValueError("Evaluation failed to account for every fixed task")
        episodes = [read_tree(r["path"]) for r in all_refs]
        if any(not e["transition_valid"] or e["truncated"] for e in episodes):
            raise ValueError("Incomplete/invalid evaluation cannot select a checkpoint")
        summaries = [dict(task_id=e["task_id"], source=e["dataset"],
                          group_id=task_map[e["task_id"]]["sample"]["group_id"],
                          seed=e["seed"], reward_sum=e["reward_sum"], physical_failure=e["physical_failure"],
                          end_reason=e["end_reason"],
                          seconds=sum(d["executed_control_steps"]*.02 for d in e["decisions"]),
                          generated=e["generated_reference_diagnostic"], motion=e["motion_diagnostic"])
                     for e in episodes]
        aggregate = aggregate_evaluation(episodes)
        source_rates = [v["reward_per_executed_second"] for v in aggregate["by_source"].values()]
        report = dict(schema="genmo.awr.physical_evaluation.v1", actor_fingerprint=fingerprint,
                      physics_identity=physics_identity, sampler_identity=digest(policy.kernel_config),
                      plan_identity=plan["plan_sha256"], task_summaries=summaries, episode_files=all_refs,
                      aggregate=aggregate, failure_reasons=dict(Counter(e["end_reason"] for e in episodes if e["physical_failure"])),
                      physical_metrics=dict(overall=summarize_metrics(episodes),
                          by_source={source: summarize_metrics([e for e in episodes if e["dataset"] == source])
                                     for source in aggregate["by_source"]}),
                      source_balanced_reward_per_second=float(np.mean(source_rates)),
                      seconds=time.perf_counter()-started, fresh_physics_world=True)
        atomic_json(path/"report.json", report)
        return report
    report = distributed.root(publish)
    if distributed.assert_models_equal(policy.actor) != fingerprint:
        raise ValueError("Evaluation modified Actor weights or buffers")
    return report
