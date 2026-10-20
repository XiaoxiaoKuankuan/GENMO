"""真实探索、参数模块变化和固定条件输出的诊断。

只比较同音乐起点与初始条件的首个决策，不把后续已经分叉的状态冒充相同条件。
qpos30不同物理量分别列出Root/Joint RMS，Contact只作概率诊断；参数变化只证明
优化发生，物理改善必须来自独立完整音乐评价。
"""
from __future__ import annotations

import itertools
import math
import numpy as np
import torch
from .io import digest


def exploration(episodes):
    grouped = {}
    for e in episodes:
        key = (e.group_id, e.task["music_start_frame"], e.task["initialization"])
        grouped.setdefault(key, []).append(e)
    reports = []
    for key, values in grouped.items():
        if len(values) < 2:
            continue
        pairs = []
        for a, b in itertools.combinations(values, 2):
            x, y = a.rows[0], b.rows[0]
            exact = all(torch.equal(x.context[k], y.context[k]) for k in x.context)
            # PhysX浮点差异单独报告；只有确实同条件的pair才作为固定条件多样性证据。
            difference = (x.target_qpos30-y.target_qpos30).double()
            pairs.append(dict(same_conditions_exact=exact,
                history_rms=float((x.context["proprio_history"]-y.context["proprio_history"]).double().square().mean().sqrt()),
                qpos30_rms=float(difference.square().mean().sqrt()),
                root_representation_rms=float(difference[:, :9].square().mean().sqrt()),
                joint_rms_rad=float(difference[:, 9:].square().mean().sqrt()),
                contact_probability_rms=float((x.contact-y.contact).double().square().mean().sqrt())))
        returns = [float(e.returns[0]) for e in values]
        reports.append(dict(group_id=key[0], music_start_frame=key[1], initialization=key[2],
                            episodes=len(values), pairs=pairs, complete_returns=returns,
                            return_std=float(np.std(returns)),
                            failures=sum(e.rows[-1].metadata["physical_failure"] for e in values)))
    fixed = [p for r in reports for p in r["pairs"] if p["same_conditions_exact"]]
    insufficient = bool(fixed) and max(p["qpos30_rms"] for p in fixed) < 1e-6
    return dict(groups=reports, exact_condition_pairs=len(fixed), insufficient_reference_diversity=insufficient,
                caveat="Root qpos30 combines representation coordinates; joint RMS is in radians")


def parameter_changes(actor, previous):
    values = {}
    for name, parameter in actor.named_parameters():
        module = name.split(".")[0]
        delta = parameter.detach().cpu().double()-previous[name].double()
        row = values.setdefault(module, dict(squared_l2=0., changed_elements=0, elements=0))
        row["squared_l2"] += float(delta.square().sum())
        row["changed_elements"] += int(torch.count_nonzero(delta))
        row["elements"] += delta.numel()
    for value in values.values():
        value["l2"] = math.sqrt(value.pop("squared_l2"))
    return values


@torch.no_grad()
def fixed_outputs(policy, rows, *, seed=1729):
    selected = rows[:min(4, len(rows))]
    context = {k: torch.cat([r.context[k] for r in selected]) for k in selected[0].context}
    result = policy.sample(context, seeds=[seed+i for i in range(len(selected))])
    return dict(sample_ids=[r.sample_id for r in selected], seeds=[seed+i for i in range(len(selected))],
                qpos30=result["qpos30"].cpu(), contact=result["contact"].cpu())
