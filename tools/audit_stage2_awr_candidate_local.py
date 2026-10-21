"""只读核验同状态候选、真实消费局部优势及固定监督份额的实验链路。

输入为已经发布Replay快照与冻结标签的AWR运行，逐Episode核对合同和内容摘要，
从保存的原始组回报重新计算首次组优势和指数权重，报告初态最大差、2/4秒排序、
实际自由参考暴露、来源/组监督质量及Critic失败分层选择。不得修改旧标签、奖励、
原模型或运行state；没有调用优化器/仿真，不把标签排序正确冒充物理学习收益。
所有输出使用全新文件；实验失败、缺失组和低覆盖继续保留在报告中。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(root):
    from gem.closedloop.awr.io import read_tree, digest, file_sha256
    from gem.closedloop.awr.records import AWRCompleteEpisode, TRAINING_METADATA_KEYS
    from gem.closedloop.awr.advantages import AdvantageLabelSet, candidate_ranking_audit
    from gem.closedloop.awr.dataset import ReplayDataset
    from gem.closedloop.awr.objective import candidate_objective_weights
    from gem.closedloop.awr.reference_credit import reference_exposure
    from gem.closedloop.awr.resources import prepare_file_handles
    root = Path(root).resolve(); started = time.perf_counter()
    original = (root/"state.json").read_bytes()
    state = json.loads(original); prefix = "macro000"
    config = state["config"]
    prepare_file_handles(config["awr"]["replay_capacity"])
    snapshot_id = state["artifacts"][prefix+"_replay"]
    snapshot = json.loads((root/"replay/snapshots"/(snapshot_id+".json")).read_text())
    episodes, entries = [], []
    for name in snapshot["episode_ids"]:
        folder = root/"replay/episodes"/name
        episode = AWRCompleteEpisode.from_dict(read_tree(folder, episode_metadata_keys=TRAINING_METADATA_KEYS))
        episode.training_view = True
        if episode.episode_id != name or digest(episode.contract) != snapshot["contract_sha256"]:
            raise ValueError("Replay physical identity changed")
        episodes.append(episode)
        entries.append([name, json.loads((folder/"manifest.json").read_text())["sha256"]])
    if digest(dict(contract=snapshot["contract_sha256"], episodes=entries)) != snapshot_id:
        raise ValueError("Immutable Replay snapshot differs")
    labels = AdvantageLabelSet.load(state["artifacts"][prefix+"_labels"])
    data = ReplayDataset(episodes, labels, expected_snapshot=snapshot_id,
        expected_signal=config["awr"]["advantage"], expected_candidates=config["awr"]["candidate_sampling"])
    actor = config["awr"]["actor"]
    coefficients = candidate_objective_weights(data.sources, data.group_valid,
        fraction=actor.get("candidate_objective_fraction"), source_balance=actor.get("source_balance", "none"))
    combined = coefficients*data.weights
    weight = labels.identity["weight_config"]
    raw = torch.exp(torch.minimum(labels.final_advantage/weight["beta"],
                                 torch.tensor(np.log(weight["weight_cap"]), dtype=torch.float64)))
    expected = raw/raw.mean() if weight["weight_mean_normalize"] else raw
    torch.testing.assert_close(labels.weights, expected, atol=1e-12, rtol=1e-12)
    index = {s:i for i,s in enumerate(labels.sample_ids)}
    rows = {r.sample_id:r for e in episodes for r in e.rows}
    groups = []
    for g in labels.statistics["candidate_groups"]:
        summary = dict(candidate_group_id=g["candidate_group_id"], valid=g["valid"], reasons=g["reasons"])
        if not g["valid"]:
            groups.append(summary); continue
        ids = [index[m["sample_id"]] for m in g["members"]]
        scores = np.asarray([v["return_value"] for v in g["windows"][format(float(g["score_window_seconds"]), "g")]])
        np.testing.assert_allclose(labels.group_advantage[ids].numpy(), scores-scores.mean(), atol=1e-12, rtol=1e-12)
        if config["awr"]["candidate_sampling"]["advantage_mode"] == "grouped":
            sigma = labels.identity["component_scales"]["group"]["std"]
            np.testing.assert_allclose(labels.final_advantage[ids].numpy(), (scores-scores.mean())/sigma,
                                       atol=1e-12, rtol=1e-12)
        members = [rows[m["sample_id"]] for m in g["members"]]
        pair_rms = []
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                mask = members[i].free_mask & members[j].free_mask
                mask[:, :9] = False
                difference = members[i].target_qpos30-members[j].target_qpos30
                pair_rms.append(float(difference[mask].double().square().mean().sqrt()))
        summary.update(initial_state_maximum_differences={k:max(c["maximum_differences"][k] for c in g["initial_comparisons"])
                       for k in g["initial_comparisons"][0]["maximum_differences"]},
            joint_reference_pair_rms_rad=pair_rms,
            crossfit_first_value_range=float(labels.values[ids].max()-labels.values[ids].min()),
            windows=g["windows"], selected_weights=labels.weights[ids].tolist())
        groups.append(summary)
    exposure, excluded, complete_exposure = [], 0., 0.
    signal = config["awr"]["advantage"]
    for episode in episodes:
        result = reference_exposure(episode, mode=signal["credit_mode"],
            current_weight=signal["current_weight"], lookahead_weight=signal["lookahead_weight"],
            coordinate_scope=signal.get("coordinate_scope", "all_committed"))
        for row, plan in zip(episode.rows, result["plans"]):
            excluded += plan["excluded_known_coordinate_seconds"]
            complete_exposure += plan["full_plan_exposure_seconds"]
            if plan["first_current_tick"] is not None:
                exposure.append((plan["first_current_tick"]-row.control_tick_begin)/600)
    critic = json.loads(Path(state["artifacts"][prefix+"_critic_report"]).read_text())
    # 采集/训练可能继续发布自己的state；本工具只证明源文件未被本进程写入，不要求其静止。
    return dict(schema="genmo.awr.candidate_local_audit.v1", source_run=str(root),
        source_commits=state["source_commits"], tool_sha256=file_sha256(__file__), replay_snapshot=snapshot_id,
        labels_manifest=file_sha256(Path(state["artifacts"][prefix+"_labels"])/"manifest.json"),
        complete_episodes=len(episodes), physical_failures=sum(bool(e.rows[-1].metadata.get("physical_failure")) for e in episodes),
        actor_samples=len(data), verified_candidate_samples=int(data.group_valid.sum()),
        groups=groups, valid_groups=sum(g["valid"] for g in groups), total_groups=len(groups),
        ranking=candidate_ranking_audit(labels, labels.statistics["candidate_groups"]),
        coefficient_mass_before_advantage=float(coefficients[data.group_valid].sum()/coefficients.sum()),
        coefficient_mass_after_advantage=float(combined[data.group_valid].sum()/combined.sum()),
        effective_sample_size_after_all_fixed_factors=float(combined.sum().square()/combined.square().sum()),
        current_reference_delay_mean_seconds=float(np.mean(exposure)) if exposure else None,
        excluded_known_coordinate_exposure_fraction=excluded/complete_exposure if complete_exposure else None,
        critic_selected=critic["selected"], critic_selection=critic.get("selection_diagnostics"),
        critic_validation=[dict(fold=f["fold"], mc=f["mc"]["validation"],
                               fvi=f["fvi"][-1]["validation"] if f["fvi"] else None) for f in critic["folds"]],
        seconds=time.perf_counter()-started, optimizer_updates=0, new_physical_controls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Use a new output file")
    report = audit(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:report[k] for k in ("complete_episodes", "actor_samples", "verified_candidate_samples",
        "valid_groups", "total_groups", "critic_selected", "seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
