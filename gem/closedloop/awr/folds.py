"""按音乐连通组固定的双Critic交叉拟合分组。

输入来自完整train清单，group相同或audio相同的条目先合并为不可拆分连通分量。
按来源计数贪心平衡两折，确定性hash负责排序与打破平局；每折内部再留出音乐组
用于Early Stopping。映射在采集前发布并跨宏迭代复用，不随Replay内容重分折，
避免历史拟合过的音乐后来进入同一Critic的标注折。外部val/test不参与构造。
"""
from __future__ import annotations

import math
from .io import digest


def build_folds(samples, *, seed=42, validation_fraction=.2):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    samples = list(samples)
    for item in samples:
        if item["split"] != "train":
            raise ValueError("Cross-fit folds may only use training music")
        union("g:" + item["group_id"], "a:" + item["audio_sha256"])
    components = {}
    for item in samples:
        component = components.setdefault(find("g:" + item["group_id"]),
                                           dict(groups=set(), audio=set(), sources=set()))
        component["groups"].add(item["group_id"])
        component["audio"].add(item["audio_sha256"])
        component["sources"].add(item["source"])
    counts = [{}, {}]
    assigned = {}
    for key in sorted(components, key=lambda k: digest([seed, k])):
        item = components[key]
        scores = [sum(counts[f].get(s, 0) for s in item["sources"]) for f in (0, 1)]
        fold = min(range(2), key=lambda f: (scores[f], sum(counts[f].values()), f))
        assigned[key] = fold
        for source in item["sources"]:
            counts[fold][source] = counts[fold].get(source, 0) + 1
    validation = set()
    for fold in (0, 1):
        for source in sorted(counts[fold]):
            eligible = [key for key, item in components.items()
                        if assigned[key] == fold and source in item["sources"]]
            if len(eligible) < 2:
                raise ValueError(f"Need >=2 music groups per source/fold: {source}/{fold}")
            eligible.sort(key=lambda k: digest([seed, "inner_validation", k]))
            count = min(len(eligible)-1, max(1, math.ceil(len(eligible)*validation_fraction)))
            validation.update(eligible[:count])
    for fold in (0, 1):
        for source in counts[fold]:
            if not any(assigned[k] == fold and source in v["sources"] and k not in validation
                       for k, v in components.items()):
                raise ValueError("Cross-source connected groups left no inner training group")
    groups = {}
    for key, item in components.items():
        for group in item["groups"]:
            groups[group] = dict(fold=assigned[key], validation=key in validation,
                                 component=key)
    result = dict(schema="genmo.awr.music_crossfit.v1", seed=seed, groups=groups,
                  components={k: {name: sorted(values) for name, values in v.items()}
                              for k, v in components.items()})
    result["sha256"] = digest(result)
    return result


def partition_episodes(episodes, fold_map, fold):
    if digest({k: v for k, v in fold_map.items() if k != "sha256"}) != fold_map["sha256"]:
        raise ValueError("Cross-fit fold map identity changed")
    train, validation, label = [], [], []
    for episode in episodes:
        item = fold_map["groups"][episode.group_id]
        destination = label if item["fold"] != fold else validation if item["validation"] else train
        destination.append(episode)
    if not train or not validation or not label:
        raise ValueError("Replay must cover inner train/validation and opposite labeling folds")
    return train, validation, label
