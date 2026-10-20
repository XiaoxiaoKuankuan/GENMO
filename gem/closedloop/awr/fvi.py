"""音乐组交叉拟合及冻结上一轮价值的N-step FVI调度。

每一轮先在完整训练Episode上一次性生成detach目标，再执行下一轮拟合。内部验证只
使用真实MC并决定停止；两折FVI都改善MSE且偏差不增时才选择FVI，否则整轮使用MC。
标签始终等于真实MC减另一音乐折的Value，包含Critic训练/标注身份和统计证据。
"""
from __future__ import annotations

import time
import torch

from .advantages import label_advantages
from .critic import CriticFit, assert_independent, flatten, predict
from .folds import partition_episodes
from .io import digest
from .returns import nstep_targets


def fit_crossfit(actor, episodes, fold_map, config, distributed, *, seed=42, checkpoint=None, resume=None):
    started = time.perf_counter()
    fits = [CriticFit(actor, config, distributed, seed=seed+i*1009) for i in range(2)]
    assert_independent(fits[0].model, fits[1].model)
    reports, mc_states, fvi_states, partitions = [], [], [], []
    if resume:
        if resume["fold_identity"] != fold_map["sha256"] or resume["episode_ids"] != [e.episode_id for e in episodes]:
            raise ValueError("Critic phase resume differs from fixed fold/Replay identity")
        for fit, state in zip(fits, resume["critics"]):
            fit.load_state_dict(state)
        reports, mc_states, fvi_states = resume["reports"], resume["mc_states"], resume["fvi_states"]
    def save(fold_index, next_fvi, current):
        if checkpoint:
            checkpoint("critic_fit_boundary", dict(
                fold_identity=fold_map["sha256"], episode_ids=[e.episode_id for e in episodes],
                critics=[f.state_dict() for f in fits], reports=reports,
                mc_states=mc_states, fvi_states=fvi_states,
                fold_index=fold_index, next_fvi=next_fvi, current=current))
    for index, fit in enumerate(fits):
        fold = ("A", "B")[index]
        part = dict(zip(("train", "validation", "label"),
                        partition_episodes(episodes, fold_map, index)))
        train, validation, label = part["train"], part["validation"], part["label"]
        partitions.append(part)
        if resume and index < resume["fold_index"]:
            continue
        continuing = resume and index == resume["fold_index"] and resume["current"] is not None
        report = resume["current"] if continuing else dict(fold=fold, mc=fit.fit(train, validation), fvi=[],
                      groups={key: sorted({e.group_id for e in value}) for key, value in part.items()},
                      counts={key: dict(episodes=len(value), states=sum(len(e.rows) for e in value),
                                        music_groups=len({e.group_id for e in value}),
                                        source_fractions={s: sum(e.source == s for e in value)/len(value)
                                                          for s in sorted({e.source for e in value})},
                                        sources={s: sum(e.source == s for e in value)
                                                 for s in sorted({e.source for e in value})})
                              for key, value in part.items()})
        if not continuing:
            mc_states.append(fit.state_dict())
            save(index, 0, report)
        begin_fvi = resume["next_fvi"] if continuing else 0
        for iteration in range(begin_fvi, config.get("fvi_iterations", 3) if config.get("mode", "mc_fvi") == "mc_fvi" else 0):
            rows, _ = flatten(train)
            previous = predict(fit.model, rows, distributed, config["microbatch"])
            target_identity = digest(dict(fold=fold, iteration=iteration, values=previous.tolist(),
                                          episodes=[e.episode_id for e in train]))
            cursor, targets = 0, []
            for episode in train:
                count = len(episode.rows)
                targets.append(nstep_targets(episode, previous[cursor:cursor+count],
                                             config.get("nstep", 16), config.get("gamma_upper", .99)))
                cursor += count
            fvi = fit.fit(train, validation, targets=torch.cat(targets),
                          epochs=config.get("fvi_epochs", 16), phase=f"fvi_{iteration+1}")
            fvi["frozen_target_identity"] = target_identity
            report["fvi"].append(fvi)
            save(index, iteration+1, report)
        fvi_states.append(fit.state_dict())
        reports.append(report)
        save(index+1, 0, None)
    labeling_started = time.perf_counter()
    use_fvi = all(r["fvi"] and
                  r["fvi"][-1]["validation"]["overall"]["mse"] < r["mc"]["validation"]["overall"]["mse"] and
                  abs(r["fvi"][-1]["validation"]["overall"]["bias"]) <= abs(r["mc"]["validation"]["overall"]["bias"])
                  for r in reports)
    chosen = fvi_states if use_fvi else mc_states
    ids, returns, values, sources, provenance = [], [], [], [], []
    for index, (fit, part) in enumerate(zip(fits, partitions)):
        fit.load_state_dict(chosen[index])
        rows, target = flatten(part["label"])
        prediction = predict(fit.model, rows, distributed, config["microbatch"])
        cursor = 0
        for episode in part["label"]:
            for j, row in enumerate(episode.rows):
                if row.actor_valid:
                    ids.append(row.sample_id); returns.append(target[cursor+j])
                    values.append(prediction[cursor+j]); sources.append(episode.source)
            cursor += len(episode.rows)
        provenance.append(dict(critic=("A", "B")[index],
                               trained_groups=sorted({e.group_id for e in part["train"]}),
                               labeled_groups=sorted({e.group_id for e in part["label"]}),
                               labeled_episodes=[e.episode_id for e in part["label"]]))
    report = dict(schema="genmo.awr.crossfit.v1", selected="mc_fvi" if use_fvi else "mc_only",
                  folds=reports, provenance=provenance, seconds=time.perf_counter()-started,
                  fold_identity=fold_map["sha256"], value_semantics="replay_behavior_mixture_baseline")
    report["crossfit_identity"] = digest({k: v for k, v in report.items() if k != "seconds"})
    labels = label_advantages(ids, torch.stack(returns), torch.stack(values), sources=sources,
                              beta=config.get("beta", 1.), weight_cap=config.get("weight_cap", 20.),
                              weight_mean_normalize=config.get("weight_mean_normalize", True),
                              identity=dict(crossfit=report["crossfit_identity"], folds=fold_map["sha256"]))
    report["critic_seconds"] = labeling_started-started
    report["advantage_labeling_seconds"] = time.perf_counter()-labeling_started
    report["seconds"] = time.perf_counter()-started
    return labels, report, dict(selected=[f.state_dict() for f in fits], mc=mc_states, fvi=fvi_states)
