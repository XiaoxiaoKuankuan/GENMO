"""双Critic的完整MC回归、独立验证和同步数据并行拟合。

复用公共非对称网络，不改变Actor。两个网络从原始Stage1独立复制，只训练各自折的
训练音乐；同折留出音乐决定停止，另一折才用于Actor标签。每个全局Batch按实际
样本数归一化，空rank仍参与SUM。验证始终比较真实MC，不把Bootstrap目标冒充真值。
"""
from __future__ import annotations

import copy
import time
import torch

from gem.closedloop.common.pretrained_critic import PretrainedConditionCritic
from .dataset import critic_batch
from .io import cpu_copy, digest


def value_metrics(prediction, target):
    p, y = torch.as_tensor(prediction).double(), torch.as_tensor(target).double()
    if p.shape != y.shape or not p.numel() or not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("Value metrics require aligned finite predictions and MC")
    error = p-y
    variance = y.var(unbiased=False)
    return dict(count=y.numel(), mse=float(error.square().mean()), mae=float(error.abs().mean()),
                bias=float(error.mean()), explained_variance=(
                    float(1-error.var(unbiased=False)/variance) if variance > 1e-12 else None))


def flatten(episodes):
    rows, values = [], []
    for episode in episodes:
        episode.validate()
        if episode.returns is None:
            raise ValueError("Critic requires audited MC")
        rows.extend(episode.rows)
        values.extend(episode.returns.unbind())
    return rows, torch.stack(values).double()


def regression_weights(episodes, mode="none"):
    """先平衡来源，再在各来源内等权自然结束/失败；不删失败、不使用回报优先采样。"""
    from collections import Counter
    keys = [(e.source, bool(e.rows[-1].metadata.get("physical_failure")))
            for e in episodes for _ in e.rows]
    if mode == "none":
        return torch.ones(len(keys), dtype=torch.float64)
    if mode != "source_episode_type":
        raise ValueError("Unknown Critic regression balance")
    counts = Counter(keys)
    sources = {s for s, _ in counts}
    types = {s: sum(key[0] == s for key in counts) for s in sources}
    return torch.tensor([len(keys)/(len(sources)*types[s]*counts[(s, f)]) for s, f in keys], dtype=torch.float64)


def validation_score(report, mode="overall"):
    """独立验证仍对真实MC计算；分层评分避免多数成功/长歌掩盖失败段。"""
    if mode == "overall":
        return report["overall"]["mse"]
    if mode != "source_episode_type":
        raise ValueError("Unknown Critic validation selection")
    groups = report["source_episode_types"]
    return sum(sum(v["mse"] for v in types.values())/len(types) for types in groups.values())/len(groups)


def fvi_selection(mc, fitted, mode="overall"):
    """只用预定独立验证选MC/FVI，不参考Actor评价或标注折回报。"""
    reasons = []
    if validation_score(fitted, mode) >= validation_score(mc, mode):
        reasons.append("validation_mse_not_improved")
    if abs(fitted["overall"]["bias"]) > abs(mc["overall"]["bias"]):
        reasons.append("overall_bias_increased")
    if mode == "source_episode_type":
        for name, old in mc["episode_types"].items():
            if fitted["episode_types"][name]["mse"] > old["mse"]:
                reasons.append(name+"_mse_increased")
    return dict(accepted=not reasons, reasons=reasons, criterion=mode,
                mc_score=validation_score(mc, mode), fvi_score=validation_score(fitted, mode))


@torch.no_grad()
def predict(model, rows, distributed, batch_size=128):
    """先按确定顺序分rank，按原序合并；预测量完全detach。"""
    model.eval()
    local = []
    indices = list(range(distributed.rank, len(rows), distributed.world_size))
    for offset in range(0, len(indices), batch_size):
        ids = indices[offset:offset+batch_size]
        context, remaining = critic_batch([rows[i] for i in ids], distributed.device)
        values = model(context, remaining).detach().cpu().double()
        local.extend(zip(ids, values.tolist()))
    merged = [item for rank in distributed.gather(local) for item in rank]
    if sorted(i for i, _ in merged) != list(range(len(rows))):
        raise RuntimeError("Distributed Critic predictions lost or repeated states")
    return torch.tensor([v for _, v in sorted(merged)], dtype=torch.float64)


def evaluate(model, episodes, distributed, batch_size=128):
    rows, target = flatten(episodes)
    prediction = predict(model, rows, distributed, batch_size)
    by_source, by_type, by_joint = {}, {}, {}
    cursor = 0
    for episode in episodes:
        ids = list(range(cursor, cursor+len(episode.rows)))
        by_source.setdefault(episode.source, []).extend(ids)
        kind = "physical_failure" if episode.rows[-1].metadata.get("physical_failure") else "natural_end"
        by_type.setdefault(kind, []).extend(ids)
        by_joint.setdefault(episode.source, {}).setdefault(kind, []).extend(ids)
        cursor += len(episode.rows)
    return dict(overall=value_metrics(prediction, target),
                sources={k: value_metrics(prediction[v], target[v]) for k, v in by_source.items()},
                episode_types={k: value_metrics(prediction[v], target[v]) for k, v in by_type.items()},
                source_episode_types={s: {k: value_metrics(prediction[v], target[v]) for k, v in types.items()}
                                      for s, types in by_joint.items()})


class CriticFit:
    def __init__(self, actor, config, distributed, *, seed):
        self.config, self.distributed = config, distributed
        devices = [distributed.device] if distributed.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.model = PretrainedConditionCritic(actor).to(distributed.device)
        distributed.sync_model(self.model)
        self.optimizer = torch.optim.AdamW([
            dict(params=[p for n, p in self.model.named_parameters()
                         if p.requires_grad and n.startswith("history_encoder.")],
                 lr=config["history_lr"]),
            dict(params=[p for n, p in self.model.named_parameters()
                         if p.requires_grad and not n.startswith("history_encoder.")],
                 lr=config["lr"]),
        ], weight_decay=config.get("weight_decay", .01))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.generator = torch.Generator().manual_seed(seed+1)
        self.fit_steps = 0

    def state_dict(self):
        return dict(model=cpu_copy(self.model.state_dict()), optimizer=cpu_copy(self.optimizer.state_dict()),
                    scheduler=self.scheduler.state_dict(), generator=self.generator.get_state(),
                    fit_steps=self.fit_steps)

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.generator.set_state(state["generator"])
        self.fit_steps = state["fit_steps"]

    def fit(self, train, validation, *, targets=None, epochs=None, phase="mc"):
        rows, mc = flatten(train)
        targets = mc if targets is None else targets.detach().cpu().double()
        if targets.shape != mc.shape or not torch.isfinite(targets).all():
            raise ValueError("Fitted targets must cover exactly the training states")
        d, c = self.distributed, self.config
        weights = regression_weights(train, c.get("regression_balance", "none"))
        batch_size, micro = int(c["global_batch"]), int(c["microbatch"])
        if min(batch_size, micro) < 1:
            raise ValueError("Positive Critic batch sizes required")
        started = time.perf_counter()
        best, best_state, stale, history = float("inf"), None, 0, []
        for epoch in range(epochs or c["mc_epochs"]):
            order = torch.randperm(len(rows), generator=self.generator).tolist()
            total_loss, samples = 0., 0
            self.model.train()
            for start in range(0, len(order), batch_size):
                global_ids = order[start:start+batch_size]
                local_ids = global_ids[d.rank::d.world_size]
                self.optimizer.zero_grad(set_to_none=True)
                local_loss = 0.
                for offset in range(0, len(local_ids), micro):
                    ids = local_ids[offset:offset+micro]
                    context, remaining = critic_batch([rows[i] for i in ids], d.device)
                    prediction = self.model(context, remaining)
                    loss = ((prediction.double()-targets[ids].to(d.device)).square()
                            * weights[ids].to(d.device)).sum()/len(global_ids)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite fitted Value loss")
                    loss.backward()
                    local_loss += float(loss.detach())
                d.gradients(self.model)
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], c.get("gradient_clip", 1.),
                    error_if_nonfinite=True)
                self.optimizer.step()
                self.scheduler.step()
                self.fit_steps += 1
                total_loss += float(d.sum(torch.tensor(local_loss, dtype=torch.float64, device=d.device)))*len(global_ids)
                samples += len(global_ids)
            report = evaluate(self.model, validation, d, micro)
            history.append(dict(epoch=epoch+1, train_mse=total_loss/samples, validation=report,
                                gradient_norm=float(norm), optimizer_steps=self.fit_steps))
            score = validation_score(report, c.get("validation_selection", "overall"))
            if score < best-c.get("min_delta", 0.):
                best, stale, best_state = score, 0, self.state_dict()
            else:
                stale += 1
            if stale >= c.get("patience", 3):
                break
        self.load_state_dict(best_state)
        d.assert_models_equal(self.model)
        return dict(phase=phase, epochs=len(history), history=history,
                    regression_balance=c.get("regression_balance", "none"),
                    validation_selection=c.get("validation_selection", "overall"),
                    validation=evaluate(self.model, validation, d, micro),
                    seconds=time.perf_counter()-started,
                    train_episode_ids=[e.episode_id for e in train],
                    validation_episode_ids=[e.episode_id for e in validation])


def assert_independent(first, second):
    left = {p.data_ptr() for p in first.parameters()}
    if any(p.data_ptr() in left for p in second.parameters()):
        raise RuntimeError("Critic A/B share parameter storage")
    for model in (first, second):
        for encoder in model.encoder_modules():
            if any(p.requires_grad for p in encoder.parameters()):
                raise RuntimeError("Frozen Critic conditioning became trainable")
