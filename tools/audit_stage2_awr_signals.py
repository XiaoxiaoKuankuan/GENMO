"""同一真实Replay和冻结交叉Critic下的多优势模式审计，不训练或重新执行物理。

核验运行快照、Episode分片和Critic checkpoint，再比较MC、4/8/16步、两种参考信用、
alpha=0/0.5/0.7及组内混合。各模式生成独立标签及ESS/来源权重/真实误差诊断，旧标签
与物理账本不修改。只有新系统已保存每集全部交叉Value时才可运行，不能用Actor更新
后的Critic代替原标签Value。输出必须是新目录；分布诊断不等同于Actor物理改善。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(root, output, macro=0):
    from gem.closedloop.awr.io import digest, read_tree, atomic_json
    from gem.closedloop.awr.records import AWRCompleteEpisode, TRAINING_METADATA_KEYS
    from gem.closedloop.awr.checkpoint import Checkpoints
    from gem.closedloop.awr.advantages import label_episode_advantages
    from gem.closedloop.awr.returns import nstep_advantages
    from gem.closedloop.awr.reference_credit import credit_advantages
    started = time.perf_counter(); root = Path(root).resolve(); output = Path(output).resolve()
    state = json.loads((root/"state.json").read_text()); prefix = f"macro{macro:03d}"
    snapshot = json.loads((root/"replay/snapshots"/(state["artifacts"][prefix+"_replay"]+".json")).read_text())
    entries, episodes = [], []
    for key in snapshot["episode_ids"]:
        path = root/"replay/episodes"/key
        manifest = json.loads((path/"manifest.json").read_text())
        entries.append([key, manifest["sha256"]])
        e = AWRCompleteEpisode.from_dict(read_tree(path, episode_metadata_keys=TRAINING_METADATA_KEYS))
        if e.episode_id != key or digest(e.contract) != snapshot["contract_sha256"]:
            raise ValueError("Replay snapshot Episode or physical identity changed")
        episodes.append(e)
    if digest(dict(contract=snapshot["contract_sha256"], episodes=entries)) != snapshot["snapshot_id"]:
        raise ValueError("Replay snapshot SHA changed")
    ck = Checkpoints(root/"checkpoints", dict(request=state["request_sha256"], sources=state["source_commits"]))
    critics = ck.load(state["artifacts"][prefix+"_critics"])["payload"]
    if critics.get("replay_snapshot") != snapshot["snapshot_id"] or "crossfit_values" not in critics:
        raise ValueError("No frozen complete crossfit values for this snapshot; never refit silently")
    values = critics["crossfit_values"]
    label_report = json.loads(Path(state["artifacts"][prefix+"_critic_report"]).read_text())
    tasks = json.loads(Path(state["artifacts"][prefix+"_tasks"]).read_text())["tasks"]
    variants = {"mc": dict(mode="mc"), "nstep4": dict(mode="nstep", nstep=4),
        "nstep8": dict(mode="nstep", nstep=8), "nstep16": dict(mode="nstep", nstep=16),
        "credit_current": dict(mode="credit_nstep", credit_mode="current_only"),
        "credit_lookahead": dict(mode="credit_nstep", credit_mode="current_plus_lookahead"),
        "hybrid0": dict(mode="hybrid", alpha=0.), "hybrid05": dict(mode="hybrid", alpha=.5),
        "hybrid07": dict(mode="hybrid", alpha=.7), "grouped": dict(mode="mc"),
        "group_hybrid": dict(mode="group_hybrid")}
    output.mkdir(parents=True, exist_ok=False)
    stats = {}
    for name, patch in variants.items():
        signal = dict(state["config"]["awr"]["advantage"], **patch)
        group = dict(state["config"]["awr"]["candidate_sampling"],
            advantage_mode="grouped" if name == "grouped" else "hybrid" if name == "group_hybrid" else "standard")
        labels, report = label_episode_advantages(episodes, values, config=signal, candidate_config=group,
            candidate_plan=tasks, identity=dict(replay_snapshot=snapshot["snapshot_id"],
                crossfit=label_report["crossfit_identity"], folds=label_report["fold_identity"],
                critic_checkpoint=state["artifacts"][prefix+"_critics"]["sha256"],
                critic_fingerprints=label_report["selected_critic_fingerprints"]))
        labels.save(output/name)
        atomic_json(output/(name+".statistics.json"), labels.statistics)
        stats[name] = dict(label_id=labels.identity["label_id"], weight=labels.statistics["weight"],
            ess=labels.statistics["effective_sample_size"], ess_fraction=labels.statistics["effective_sample_fraction"],
            credit_valid=labels.statistics["credit_valid_samples"], group_valid=labels.statistics["group_valid_samples"],
            by_source=labels.statistics["by_source"], within_condition_weight_quality=labels.statistics["within_condition_weight_quality"])
    errors = []
    for e in episodes:
        for n in (4, 8, 16, "full_mc"):
            local = credit_advantages(e, values[e.episode_id], torch.ones(len(e.rows), len(e.rows)), nstep=n)
            expected = nstep_advantages(e, values[e.episode_id], n)
            errors.append(float((local-expected).abs().max()))
    report = dict(schema="genmo.awr.real_signal_comparison.v1", source_run=str(root),
        replay_snapshot=snapshot["snapshot_id"], source_commits=state["source_commits"],
        episodes=len(episodes), actor_samples=sum(r.actor_valid for e in episodes for r in e.rows),
        full_credit_equals_nstep_max_error=max(errors), variants=stats, seconds=time.perf_counter()-started,
        actor_optimizer_updates=0, new_physical_controls=0, learning_improvement_claim=False)
    atomic_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--macro", type=int, default=0)
    args = parser.parse_args()
    report = audit(args.run_dir, args.output, args.macro)
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
