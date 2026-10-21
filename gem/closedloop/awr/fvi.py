"""音乐组交叉拟合及冻结上一轮价值的N-step FVI调度。

每一轮先在完整训练Episode上一次性生成detach目标，再执行下一轮拟合。内部验证只
使用真实MC并决定停止；两折FVI都改善MSE且偏差不增时才选择FVI，否则整轮使用MC。
冻结同一交叉Critic的全部当前/后继Value，原MC标签仍为真实MC减Value；可选Actor
N-step、参考信用与候选融合只改变标签层，不把FVI训练目标冒充MC回报。
"""
from __future__ import annotations

import time
import torch

from .advantages import label_episode_advantages
from .critic import CriticFit, assert_independent, flatten, predict, fvi_selection
from .folds import partition_episodes
from .io import digest
from .returns import nstep_targets


def fit_crossfit(actor, episodes, fold_map, config, distributed, *, seed=42, checkpoint=None, resume=None,
                 label_config=None, candidate_config=None, replay_snapshot=None, candidate_plan=None):
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
    selection = [fvi_selection(r["mc"]["validation"], r["fvi"][-1]["validation"],
                              config.get("validation_selection", "overall")) if r["fvi"]
                 else dict(accepted=False, reasons=["mc_only_requested"]) for r in reports]
    use_fvi = all(r["accepted"] for r in selection)
    chosen = fvi_states if use_fvi else mc_states
    value_by_episode, provenance, fingerprints = {}, [], []
    for index, (fit, part) in enumerate(zip(fits, partitions)):
        fit.load_state_dict(chosen[index])
        from gem.closedloop.common.evaluation import _model_fingerprint
        fingerprints.append(_model_fingerprint(fit.model))
        rows, target = flatten(part["label"])
        prediction = predict(fit.model, rows, distributed, config["microbatch"])
        cursor = 0
        for episode in part["label"]:
            value_by_episode[episode.episode_id] = prediction[cursor:cursor+len(episode.rows)].detach().cpu().double()
            cursor += len(episode.rows)
        provenance.append(dict(critic=("A", "B")[index],
                               trained_groups=sorted({e.group_id for e in part["train"]}),
                               labeled_groups=sorted({e.group_id for e in part["label"]}),
                               labeled_episodes=[e.episode_id for e in part["label"]]))
    report = dict(schema="genmo.awr.crossfit.v1", selected="mc_fvi" if use_fvi else "mc_only",
                  selection_diagnostics=selection,
                  folds=reports, provenance=provenance, seconds=time.perf_counter()-started,
                  fold_identity=fold_map["sha256"], selected_critic_fingerprints=fingerprints,
                  value_semantics="replay_behavior_mixture_baseline")
    report["crossfit_identity"] = digest({k: v for k, v in report.items() if k != "seconds"})
    labels, signal_report = label_episode_advantages(episodes, value_by_episode,
                              config=label_config, candidate_config=candidate_config,
                              candidate_plan=candidate_plan,
                              beta=config.get("beta", 1.), weight_cap=config.get("weight_cap", 20.),
                              weight_mean_normalize=config.get("weight_mean_normalize", True),
                              gamma_upper=config.get("gamma_upper", .99),
                              identity=dict(crossfit=report["crossfit_identity"], folds=fold_map["sha256"],
                                  critic_fingerprints=fingerprints, replay_snapshot=replay_snapshot,
                                  episode_ids=[e.episode_id for e in episodes]))
    report["actor_signal_audit"] = signal_report
    report["critic_seconds"] = labeling_started-started
    report["advantage_labeling_seconds"] = time.perf_counter()-labeling_started
    report["seconds"] = time.perf_counter()-started
    return labels, report, dict(selected=[f.state_dict() for f in fits], mc=mc_states, fvi=fvi_states,
                              crossfit_values=value_by_episode, replay_snapshot=replay_snapshot)
