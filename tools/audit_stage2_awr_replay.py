"""独立只读核验真实AWR完整Episode的目标、提交和Monte Carlo回报。

以原子发布回执选择Episode，核验完整manifest和所有数组SHA，再检查物理时间、
条件/Mask、原始生成qpos30与实际提交自由坐标。MC用逐50Hz控制步的独立Python
反向累加重算，不调用训练使用的MC函数；零控制步事件单独处理且不再乘dt。
审计可以读取正在采集运行中已经完整发布的分片，明确报告抽样数量和全部发布数量，
不访问未完成Episode、不写回Replay、不改变训练或物理进程。输出禁止覆盖旧证据。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gem.closedloop.awr.collection import reference_audit
from gem.closedloop.awr.io import read_tree
from gem.closedloop.awr.records import AWRCompleteEpisode, TRAINING_METADATA_KEYS


def independent_mc(rows, gamma_upper=.99):
    """按每个真实控制奖励逆推，与区间R/Gamma实现保持独立。"""
    gamma, tail, values = gamma_upper ** (1/25), 0., []
    for row in reversed(rows):
        event = float(row.metadata.get("event_reward", 0.))
        if row.executed_control_steps == 0:
            tail += event
        else:
            for index in range(row.executed_control_steps-1, -1, -1):
                reward = float(row.rewards[index])
                if index == row.executed_control_steps-1:
                    reward += event
                tail = reward + gamma*tail
        values.append(tail)
    return np.asarray(values[::-1], dtype=np.float64)


def audit(run_dir, *, max_episodes=None):
    root = Path(run_dir).resolve()
    state = json.loads((root/"state.json").read_text())
    publications = [json.loads(path.read_text())
                    for path in (root/"replay"/"publications").glob("*.json")]
    publications.sort(key=lambda p: (p["published_unix_ns"], p["episode_id"]))
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("Audit sample limit must be positive")
    selected = publications if max_episodes is None else publications[:max_episodes]
    if not selected:
        raise ValueError("No atomically published complete Episodes to audit")
    if len({p["episode_id"] for p in publications}) != len(publications):
        raise ValueError("Duplicate published Episode identity")
    started = time.perf_counter()
    reports = []
    for publication in selected:
        folder = root/"replay"/"episodes"/publication["episode_id"]
        manifest = json.loads((folder/"manifest.json").read_text())
        if manifest["sha256"] != publication["sha256"]:
            raise ValueError("Publication and immutable array identity differ")
        value = read_tree(folder, episode_metadata_keys=TRAINING_METADATA_KEYS | {"generated", "published"})
        episode = AWRCompleteEpisode.from_dict(value)
        episode.training_view = True
        if episode.episode_id != publication["episode_id"] or episode.task["task_id"] != publication["task_id"]:
            raise ValueError("Episode/task identity differs from publication")
        maximum_target_error = 0.
        for row in episode.rows:
            generated = row.metadata["generated"]
            target = torch.as_tensor(np.asarray(generated["qpos30"]))
            error = float((row.target_qpos30.double()-target.double()).abs().max())
            maximum_target_error = max(maximum_target_error, error)
            if error != 0:
                raise ValueError("Replay target differs from original generated qpos30")
            evidence = reference_audit(generated, row.metadata["published"])
            if row.actor_valid and not evidence["committed"]:
                raise ValueError("Uncommitted reference entered Actor supervision")
            if row.execution["reward_interval_ticks"] != [row.control_tick_begin, row.control_tick_end]:
                raise ValueError("Reward interval differs from executed control interval")
        expected = independent_mc(episode.rows)
        actual = episode.returns.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-12)
        reports.append(dict(episode_id=episode.episode_id, source=episode.source,
            states=len(episode.rows), actor_samples=sum(row.actor_valid for row in episode.rows),
            controls=sum(row.executed_control_steps for row in episode.rows),
            reason=episode.rows[-1].reason, mc_max_absolute_error=float(np.max(np.abs(expected-actual))),
            generated_target_max_absolute_error=maximum_target_error,
            full_manifest_and_all_array_sha_verified=True))
    return dict(schema="genmo.awr.independent_replay_audit.v1", run_dir=str(root),
        source_commits=state["source_commits"], published_at_snapshot=len(publications),
        audited_episodes=len(reports), all_published_audited=len(selected)==len(publications),
        independent_mc_formula="per_control_reverse_sum_gamma_0.99_pow_1_over_25",
        reward_multiplied_by_dt_again=False, seconds=time.perf_counter()-started,
        outcomes=dict(Counter(row["reason"] for row in reports)), episodes=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir, max_episodes=args.max_episodes)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
