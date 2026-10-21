"""原Stage1训练音乐的固定物理评价清单，不使用验证集或测试集音乐。

从真实train清单按来源、group及音频去重选歌，保留原完整歌曲、固定种子和站立预热
评价流程。任务身份明确标train，报告明确不做泛化性结论；评价数据本身不自动写入
Replay。此处不改变旧DPPO的独立验证计划，不把train伪装成val来绕过接口。
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from gem.closedloop.common.evaluation import build_evaluation_tasks, _digest
from gem.closedloop.common.full_dataset import SOURCES


def build_train_plan(catalog, *, samples_per_source, seeds, selection_seed):
    plan = build_evaluation_tasks(catalog, split="train", seeds=seeds, selection_seed=selection_seed,
                                  episode_seconds=None, training_music_evaluation=True)
    rng = np.random.default_rng(selection_seed)
    heldout = [s for split in ("val", "test") for pool in catalog.samples.get(split, {}).values() for s in pool]
    excluded_groups = {s["group_id"] for s in heldout}
    excluded_audio = {s["row"]["source_audio_sha256"] for s in heldout}
    selected, counts, seen_groups, seen_audio = set(), {}, set(), set()
    for source in sorted(SOURCES):
        groups = {}
        for sample in catalog.samples["train"][source]:
            if sample["group_id"] in excluded_groups or sample["row"]["source_audio_sha256"] in excluded_audio:
                raise ValueError("Train evaluation catalog overlaps validation/test music identity")
            groups.setdefault(sample["group_id"], []).append(sample)
        pool = []
        local_audio = set()
        for group, samples in sorted(groups.items()):
            sample = min(samples, key=lambda s: (-s["row"]["num_frames"], s["row"]["sample_id"]))
            audio = sample["row"]["source_audio_sha256"]
            if group not in seen_groups and audio not in seen_audio | local_audio:
                pool.append(sample); local_audio.add(audio)
        count = (len(pool) if samples_per_source == "all_unique" else
                 samples_per_source[source] if isinstance(samples_per_source, Mapping) else samples_per_source)
        if type(count) is not int or not 0 < count <= len(pool):
            raise ValueError(f"Insufficient unique training music for {source}")
        counts[source] = count
        for i in sorted(rng.choice(len(pool), count, replace=False).tolist()):
            sample = pool[i]
            selected.add((source, str(sample["row"]["sample_id"])))
            seen_groups.add(sample["group_id"]); seen_audio.add(sample["row"]["source_audio_sha256"])
    plan["tasks"] = [t for t in plan["tasks"] if (t["dataset"], t["sample_id"]) in selected]
    plan.update(task_count=len(plan["tasks"]), selected_sample_count=len(selected),
                selected_unique_group_count=len(seen_groups), selected_unique_audio_count=len(seen_audio),
                selection_rule="fixed_source_balanced_original_training_music",
                training_evaluation=dict(schema="genmo.awr.train_music_evaluation.v1", source_sample_counts=counts),
                acceptance_metric="source_balanced_complete_task_return")
    plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan
