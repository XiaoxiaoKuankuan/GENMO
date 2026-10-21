"""交叉拟合MC、局部N-step、实际参考暴露及同条件组内优势的冻结标签。

真实完整MC减去另一音乐折Critic给出的Value，只除整轮总体标准差，不减均值。
权重在FP64 log空间截顶并检查下溢；可选整轮均值归一化属于明确工程选项，
其最终上界为原上限除原均值。标签与Replay/Critic/折身份一起冻结，Mini-batch
只读取标签，不重新归一化，不按权重再抽样，也不把FVI target当MC。新模式保留各
原始分量、独立标准差、fallback及真实物理诊断，先按配置融合再进行指数加权。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from .io import digest, read_tree, write_tree


def distribution(values):
    x = torch.as_tensor(values).double()
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Distribution requires finite nonempty values")
    q = torch.quantile(x, torch.tensor([.5, .95, .99], dtype=torch.float64))
    return dict(count=x.numel(), mean=x.mean().item(), std=x.std(correction=0).item(),
                p50=q[0].item(), p95=q[1].item(), p99=q[2].item(),
                minimum=x.min().item(), maximum=x.max().item())


@dataclass
class AdvantageLabelSet:
    sample_ids: list[str]
    returns: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    normalized: torch.Tensor
    raw_weights: torch.Tensor
    weights: torch.Tensor
    identity: dict
    statistics: dict
    mc_advantage: torch.Tensor | None = None
    nstep_advantage: torch.Tensor | None = None
    credit_advantage: torch.Tensor | None = None
    group_advantage: torch.Tensor | None = None
    final_advantage: torch.Tensor | None = None
    final_weight: torch.Tensor | None = None
    credit_valid: torch.Tensor | None = None
    credit_exposure: torch.Tensor | None = None
    group_valid: torch.Tensor | None = None
    source: list[str] | None = None
    label_mode: str = "mc"

    def save(self, path):
        return write_tree(path, vars(self))

    @classmethod
    def load(cls, path):
        return cls(**read_tree(path))


def label_advantages(sample_ids, returns, values, *, sources, identity,
                     beta=1., weight_cap=20., epsilon=1e-8, weight_mean_normalize=True,
                     final_normalized=None, allow_degenerate=False):
    g, v = (torch.as_tensor(x).detach().cpu().double() for x in (returns, values))
    if (g.ndim != 1 or g.shape != v.shape or len(sample_ids) != g.numel()
            or len(set(sample_ids)) != len(sample_ids) or len(sources) != len(sample_ids)):
        raise ValueError("AWR labels must match unique replay samples")
    if not torch.isfinite(g).all() or not torch.isfinite(v).all():
        raise FloatingPointError("Nonfinite true MC or cross-fitted Value")
    if not all(math.isfinite(x) and x > 0 for x in (beta, weight_cap, epsilon)):
        raise ValueError("Invalid AWR weight configuration")
    a = g - v
    sigma = a.std(correction=0)
    if not torch.isfinite(sigma) or sigma < epsilon and final_normalized is None:
        raise ValueError("Advantage signal has near-zero variance; do not publish Actor")
    normalized = (a / sigma.clamp_min(epsilon) if final_normalized is None else
                  torch.as_tensor(final_normalized).detach().cpu().double())
    if normalized.shape != a.shape or not torch.isfinite(normalized).all():
        raise FloatingPointError("Nonfinite or mismatched final normalized advantage")
    log_weights = torch.minimum(normalized / beta, torch.tensor(math.log(weight_cap), dtype=torch.float64))
    if (log_weights < math.log(torch.finfo(torch.float64).tiny)).any():
        raise FloatingPointError("AWR exponential underflow; inspect value bias/advantage scale")
    raw = torch.where(normalized/beta >= math.log(weight_cap),
                      torch.full_like(log_weights, weight_cap), log_weights.exp())
    mean = raw.mean()
    if not allow_degenerate and raw.std(correction=0) <= torch.finfo(torch.float64).eps*mean:
        raise ValueError("Degenerate AWR weights contain no relative advantage signal")
    weights = raw / mean if weight_mean_normalize else raw.clone()
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise FloatingPointError("AWR weights are not representable positive numbers")
    config = dict(beta=beta, weight_cap=weight_cap, epsilon=epsilon,
                  weight_mean_normalize=weight_mean_normalize, std_correction=0)
    stats = dict(advantage=distribution(a), normalized_advantage=distribution(normalized),
                 raw_weight=distribution(raw), weight=distribution(weights), sigma=sigma.item(),
                 cap_fraction=(normalized / beta >= math.log(weight_cap)).double().mean().item(),
                 positive_fraction=(a > 0).double().mean().item(),
                 negative_fraction=(a < 0).double().mean().item(),
                 weight_upper_bound=weight_cap / mean.item() if weight_mean_normalize else weight_cap,
                 by_source={})
    stats.update(effective_sample_size=float(weights.sum().square()/weights.square().sum()),
                 effective_sample_fraction=float(weights.sum().square()/weights.square().sum()/len(weights)),
                 high_weight_threshold=2., high_weight_fraction=float((weights >= 2.).double().mean()),
                 degenerate_relative_weights=bool(raw.std(correction=0) <= torch.finfo(torch.float64).eps*mean))
    for source in sorted(set(sources)):
        mask = torch.tensor([s == source for s in sources])
        stats["by_source"][source] = dict(advantage=distribution(a[mask]),
            weight=distribution(weights[mask]), weight_mass_fraction=float(weights[mask].sum()/weights.sum()),
            effective_sample_size=float(weights[mask].sum().square()/weights[mask].square().sum()),
            high_weight_fraction=float((weights[mask] >= 2.).double().mean()))
    binding = dict(identity, weight_config=config)
    binding["label_id"] = digest(dict(binding, sample_ids=sample_ids,
                                      advantages=a.tolist(), weights=weights.tolist()))
    return AdvantageLabelSet(list(sample_ids), g, v, a, normalized, raw, weights, binding, stats)


def signal_config(config=None, candidate_config=None):
    """解析显式标签模式；旧MC算法没有偷偷改变公式，其他模式各有冻结身份。"""
    c = dict(mode="mc", nstep=4, credit_mode="current_only", current_weight=.5, lookahead_weight=.5,
             alpha=.5, group_alpha=.5, credit_fallback="nstep", epsilon=1e-8)
    c.update(config or {})
    group = dict(candidate_config or {})
    group.setdefault("advantage_mode", "standard")
    if c["mode"] not in ("mc", "nstep", "credit_nstep", "hybrid", "group_hybrid"):
        raise ValueError("Unknown Actor advantage label mode")
    if c["nstep"] not in (4, 8, 16, "full_mc") or c["credit_fallback"] not in ("mc", "nstep"):
        raise ValueError("Invalid Actor horizon or credit fallback")
    if (c["credit_mode"] not in ("current_only", "current_plus_lookahead")
            or not all(math.isfinite(float(c[k])) and 0 <= c[k] <= 1 for k in ("alpha", "group_alpha"))
            or c["epsilon"] <= 0 or not math.isfinite(c["epsilon"])):
        raise ValueError("Invalid advantage mixing configuration")
    if group["advantage_mode"] not in ("standard", "grouped", "hybrid"):
        raise ValueError("Candidate advantage mode must be standard, grouped or hybrid")
    if c["mode"] == "group_hybrid" and (not group.get("enabled") or group["advantage_mode"] != "hybrid"):
        raise ValueError("group_hybrid requires enabled hybrid candidates")
    return c, group


def _normalize_component(values, mask, epsilon):
    x = values[mask]
    if not torch.isfinite(x).all():
        raise FloatingPointError("Nonfinite advantage component")
    sigma = float(x.std(correction=0)) if len(x) else 0.
    # 缺失/零方差分量不能通过除极小数制造巨大指数权重；明确记录无信息并由调用者回退。
    available = sigma >= epsilon
    result = values/sigma if available else torch.zeros_like(values)
    return result, dict(valid_count=len(x), std=sigma, available=available, subtract_mean=False)


def physical_weight_diagnostics(weights, sources, diagnostics):
    """同来源与整体分层统计；跨音乐难度相关性只作诊断，不声称因果。"""
    w = weights.double().numpy()
    result = {}
    for source in [None]+sorted(set(sources)):
        ids = [i for i, s in enumerate(sources) if source is None or s == source]
        local_w = w[ids]; threshold = float(np.quantile(local_w, .75))
        high = [i for i in ids if w[i] >= threshold]
        low = [i for i in ids if w[i] < threshold]
        metrics = sorted({k for i in ids for k in diagnostics[i]})
        entry = dict(high_weight_definition="within_source_weight_top_quartile", threshold=threshold,
                     samples=len(ids), metrics={})
        for name in metrics:
            valid = [i for i in ids if diagnostics[i].get(name) is not None]
            if not valid:
                continue
            x = np.array([diagnostics[i][name] for i in valid], dtype=np.float64)
            ww = w[valid]
            def mean(indices):
                v = [float(diagnostics[i][name]) for i in indices if diagnostics[i].get(name) is not None]
                return float(np.mean(v)) if v else None
            entry["metrics"][name] = dict(count=len(valid), mean=float(x.mean()),
                weighted_mean=float(np.dot(x, ww)/ww.sum()), high_weight_mean=mean(high), low_weight_mean=mean(low),
                correlation=float(np.corrcoef(ww, x)[0, 1]) if len(x) > 1 and min(ww.std(), x.std()) > 1e-12 else None)
        result["overall" if source is None else source] = entry
    return dict(by_source=result, causal_claim=False,
                scope="physical_interval_errors_and_complete_episode_failure_diagnostic_not_reward_redefinition")


def label_episode_advantages(episodes, value_by_episode, *, identity, config=None, candidate_config=None,
                             beta=1., weight_cap=20., weight_mean_normalize=True, gamma_upper=.99, candidate_plan=None):
    """每集所有状态使用同一未训练该音乐组的Critic，然后生成一次性冻结Actor标签。"""
    from .returns import nstep_advantages
    from .reference_credit import reference_exposure, credit_advantages
    from .candidates import group_advantages
    c, group = signal_config(config, candidate_config)
    grouped, group_reports = group_advantages(episodes, group, gamma_upper=gamma_upper, planned_tasks=candidate_plan)
    ids, sources, returns, values, ns, credit, cv, ce, ga, gv, diagnostics = ([] for _ in range(11))
    credit_reports = []
    for episode in episodes:
        if episode.episode_id not in value_by_episode:
            raise ValueError("Missing cross-fitted whole-Episode values")
        v = value_by_episode[episode.episode_id]
        n = nstep_advantages(episode, v, c["nstep"], gamma_upper)
        exposure = reference_exposure(episode, mode=c["credit_mode"],
                    current_weight=c["current_weight"], lookahead_weight=c["lookahead_weight"])
        local = credit_advantages(episode, v, exposure["matrix"], nstep=c["nstep"], gamma_upper=gamma_upper)
        credit_reports.append(dict(episode_id=episode.episode_id, valid=exposure["valid"],
            reason=exposure.get("reason"), ledger_sha256=exposure.get("ledger_sha256"), plans=exposure["plans"]))
        ledger = episode.reference_ledger or {}
        for j, row in enumerate(episode.rows):
            if not row.actor_valid:
                continue
            ids.append(row.sample_id); sources.append(episode.source)
            returns.append(episode.returns[j]); values.append(v[j]); ns.append(n[j]); credit.append(local[j])
            stop = len(episode.rows) if c["nstep"] == "full_mc" else min(len(episode.rows), j+c["nstep"])
            cv.append(exposure["valid"] and bool(exposure["matrix"][j, j:stop].sum() > 0))
            ce.append(exposure["exposure"][j]); ga.append(grouped.get(row.sample_id, 0.)); gv.append(row.sample_id in grouped)
            diagnostic = dict(physical_failure=float(episode.rows[-1].metadata.get("physical_failure", False)),
                episode_seconds=sum(r.executed_control_steps for r in episode.rows)*.02,
                interval_seconds=row.executed_control_steps*.02)
            if ledger.get("available"):
                mask = np.asarray(ledger["interval_indices"]) == j
                if mask.any():
                    diagnostic.update(zip(ledger["error_names"], np.asarray(ledger["tracking_errors"])[mask].mean(axis=0).tolist()))
            diagnostics.append(diagnostic)
    g, v, n, a_credit, a_group, exposure_seconds = [torch.as_tensor(x).double() for x in (returns, values, ns, credit, ga, ce)]
    valid_credit, valid_group = torch.tensor(cv, dtype=torch.bool), torch.tensor(gv, dtype=torch.bool)
    a_mc = g-v
    components, scales = {}, {}
    for name, raw, mask in (("mc", a_mc, torch.ones(len(ids), dtype=torch.bool)),
                             ("nstep", n, torch.ones(len(ids), dtype=torch.bool)),
                             ("credit", a_credit, valid_credit), ("group", a_group, valid_group)):
        components[name], scales[name] = _normalize_component(raw, mask, c["epsilon"])
    fallback = components[c["credit_fallback"]]
    usable_credit = valid_credit & scales["credit"]["available"]
    local = torch.where(usable_credit, components["credit"], fallback)
    if c["mode"] == "mc":
        final = components["mc"].clone()
    elif c["mode"] == "nstep":
        final = components["nstep"].clone()
    elif c["mode"] == "credit_nstep":
        final = local.clone()
    else:
        final = (1-c["alpha"])*components["mc"]+c["alpha"]*local
    group_mask = valid_group & scales["group"]["available"]
    if group["advantage_mode"] == "grouped":
        final[group_mask] = components["group"][group_mask]
    elif group["advantage_mode"] == "hybrid":
        final[group_mask] = ((1-c["group_alpha"])*final+c["group_alpha"]*components["group"])[group_mask]
    binding = dict(identity, schema="genmo.awr.advantage_labels.v2", signal_config=c, candidate_config=group,
                   gamma_upper=gamma_upper, gamma_control=gamma_upper**(1/25),
                   reward_identities=sorted({digest(e.contract.get("reward")) for e in episodes}),
                   sampler_identities=sorted({digest(e.contract.get("sampler")) for e in episodes}),
                   group_audit_identities=[r["identity"] for r in group_reports],
                   credit_ledger_identities=[r["ledger_sha256"] for r in credit_reports],
                   component_scales=scales)
    from .candidates import state_fingerprint
    binding["component_identity"] = state_fingerprint(dict(mc=a_mc, nstep=n, credit=a_credit,
        group=a_group, credit_valid=valid_credit, group_valid=valid_group, exposure=exposure_seconds, final=final))
    label = label_advantages(ids, g, v, sources=sources, identity=binding, beta=beta,
            weight_cap=weight_cap, weight_mean_normalize=weight_mean_normalize,
            final_normalized=final, allow_degenerate=True)
    label.mc_advantage, label.nstep_advantage = a_mc, n
    label.credit_advantage, label.group_advantage = a_credit, a_group
    label.final_advantage, label.final_weight = final.clone(), label.weights.clone()
    label.credit_valid, label.credit_exposure, label.group_valid = valid_credit, exposure_seconds, valid_group
    label.source, label.label_mode = sources, c["mode"]+":"+group["advantage_mode"]
    weight_lookup = dict(zip(ids, label.weights.tolist()))
    group_quality = []
    for report in group_reports:
        if not report["valid"]:
            continue
        key = format(float(report["score_window_seconds"]), "g")
        group_quality.append(dict(candidate_group_id=report["candidate_group_id"],
            comparison="verified_same_first_state_fixed_window",
            candidates=[dict(member, weight=weight_lookup[member["sample_id"]], **score)
                        for member, score in zip(report["members"], report["windows"][key])]))
    label.statistics.update(component_scales=scales, credit_valid_samples=int(valid_credit.sum()),
        group_valid_samples=int(valid_group.sum()), credit_fallback_samples=int((~usable_credit).sum()),
        label_mode=label.label_mode, final_advantage=distribution(final),
        physics=physical_weight_diagnostics(label.weights, sources, diagnostics),
        candidate_groups=group_reports, within_condition_weight_quality=group_quality)
    return label, dict(credit=credit_reports, candidate_groups=group_reports,
                       approximation="reference_exposure_not_unbiased_policy_gradient")
