"""完整物理任务的固定计划与训练/验证音乐隔离。

正式先导复用原来源概率、rank共享时间分层和配对/站立初始化。小型闭环专门从每来源
的两折训练/内部验证各选一音乐组，再固定起点和初态变化生成种子，用于真实探索诊断。
所有计划先持久化；恢复重做未完成task，不能重新抽样替换失败任务。
"""
from __future__ import annotations

import copy
import math
import numpy as np
from gem.closedloop.common.full_dataset import FullMusicSampler, SOURCES
from gem.closedloop.common.time_sampling import StratifiedTimePool, MOTION_RESET_CONTRACT
from gem.closedloop.common.evaluation_plan import build_balanced_plan
from .folds import build_folds
from .io import digest


def folds_for_catalog(catalog, seed=42):
    samples = [dict(split="train", group_id=s["group_id"], audio_sha256=s["row"]["source_audio_sha256"], source=source)
               for source, pool in catalog.samples["train"].items() for s in pool]
    return build_folds(samples, seed=seed)


def _task(sample, start, seed, initialization, rank, index, macro):
    value = dict(sample=copy.deepcopy(sample), dataset=sample["dataset"],
                 sample_id=sample["row"]["sample_id"], group_id=sample["group_id"],
                 music_start_frame=int(start), seed=int(seed), initialization=initialization,
                 full_music_seconds=sample["row"]["num_frames"]/30.,
                 remaining_music_seconds=(sample["row"]["num_frames"]-start)/30.,
                 rank=rank, macro_iteration=macro, planned_index=index)
    value["task_id"] = digest(value)
    value["noise_index_start"] = int(digest([value["task_id"], "generation"])[:12], 16)
    return value


def training_tasks(catalog, folds, config, world_size, macro=0):
    c, a = config["dataset"], config["awr"]
    seed = a["seed"]+macro*10007
    tasks = []
    if a["profile"] == "smoke":
        rng = np.random.default_rng(seed)
        for source in SOURCES:
            for fold in (0, 1):
                for validation in (False, True):
                    pool = [s for s in catalog.samples["train"][source]
                            if folds["groups"][s["group_id"]]["fold"] == fold and
                            folds["groups"][s["group_id"]]["validation"] == validation]
                    chosen = min(pool, key=lambda s: digest([seed, s["group_id"], s["row"]["sample_id"]]))
                    standing = rng.random() < c["standing_start_probability"]
                    maximum = max(0, chosen["row"]["num_frames"]-math.ceil(c["minimum_remaining_seconds"]*30))
                    start = 0 if standing else int(rng.integers(maximum+1))
                    init = "standing" if standing else MOTION_RESET_CONTRACT
                    group_index = len(tasks)//len(a["smoke"]["seeds"])
                    for generation_seed in a["smoke"]["seeds"]:
                        task = _task(chosen, start, seed, init, group_index % world_size, len(tasks), macro)
                        task["generation_seed"] = generation_seed
                        task["task_id"] = digest(task)
                        task["noise_index_start"] = int(digest([chosen["group_id"], generation_seed])[:12], 16)
                        tasks.append(task)
        if len(tasks) != a["initial_episodes"]:
            raise ValueError("Smoke plan must be four sources x four groups x four seeds")
    else:
        count = a["initial_episodes"] if macro == 0 else a["new_episodes"]
        for rank in range(world_size):
            pool = StratifiedTimePool(catalog, seed=seed, rank=rank, world_size=world_size,
                minimum_remaining_seconds=c["minimum_remaining_seconds"], bin_seconds=c["time_bin_seconds"],
                standing_start_probability=c["standing_start_probability"])
            sampler = FullMusicSampler(catalog, seed=seed+rank*1009,
                minimum_remaining_seconds=c["minimum_remaining_seconds"], random_start=True,
                source_probabilities=c["source_probabilities"], time_pool=pool)
            for _ in range(len(range(rank, count, world_size))):
                item = sampler.next_task()
                tasks.append(_task(item["sample"], item["music_start_frame"], seed+len(tasks),
                                   item["sampling"]["initialization"], rank, len(tasks), macro))
    return dict(schema="genmo.awr.complete_task_plan.v1", tasks=tasks, macro_iteration=macro,
                fold_identity=folds["sha256"], plan_sha256=digest(tasks))


def evaluation_plan(catalog, config, *, all_unique=False):
    c = config["evaluation"]
    return build_balanced_plan(catalog, samples_per_source="all_unique" if all_unique else c["samples_per_source"],
                              seeds=c["seeds"], seconds=None, selection_seed=c["selection_seed"],
                              start_mode="beginning", unique_music=True, split="val")
