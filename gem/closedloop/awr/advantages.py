"""交叉拟合优势与整轮固定的优势权重。

真实完整MC减去另一音乐折Critic给出的Value，只除整轮总体标准差，不减均值。
权重在FP64 log空间截顶并检查下溢；可选整轮均值归一化属于明确工程选项，
其最终上界为原上限除原均值。标签与Replay/Critic/折身份一起冻结，Mini-batch
只读取标签，不重新归一化，不按权重再抽样，也不把FVI target当MC。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
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

    def save(self, path):
        return write_tree(path, vars(self))

    @classmethod
    def load(cls, path):
        return cls(**read_tree(path))


def label_advantages(sample_ids, returns, values, *, sources, identity,
                     beta=1., weight_cap=20., epsilon=1e-8, weight_mean_normalize=True):
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
    if not torch.isfinite(sigma) or sigma < epsilon:
        raise ValueError("Advantage signal has near-zero variance; do not publish Actor")
    normalized = a / sigma.clamp_min(epsilon)
    log_weights = torch.minimum(normalized / beta, torch.tensor(math.log(weight_cap), dtype=torch.float64))
    if (log_weights < math.log(torch.finfo(torch.float64).tiny)).any():
        raise FloatingPointError("AWR exponential underflow; inspect value bias/advantage scale")
    raw = torch.where(normalized/beta >= math.log(weight_cap),
                      torch.full_like(log_weights, weight_cap), log_weights.exp())
    mean = raw.mean()
    if raw.std(correction=0) <= torch.finfo(torch.float64).eps*mean:
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
    for source in sorted(set(sources)):
        mask = torch.tensor([s == source for s in sources])
        stats["by_source"][source] = dict(advantage=distribution(a[mask]),
                                          weight=distribution(weights[mask]))
    binding = dict(identity, weight_config=config)
    binding["label_id"] = digest(dict(binding, sample_ids=sample_ids,
                                      advantages=a.tolist(), weights=weights.tolist()))
    return AdvantageLabelSet(list(sample_ids), g, v, a, normalized, raw, weights, binding, stats)
