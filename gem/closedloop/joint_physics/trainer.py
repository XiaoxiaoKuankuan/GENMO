"""从随机初始化开始的单优化器联合训练，支持同步DDP和逐样本物理拒绝。

完整1000步扩散监督作用于常规batch，执行BPTT只用保留音乐的随机子批。DDP按全局有效
物理样本数归一化，各rank含零有效样本时仍执行同一backward。保存数据、噪声与窗口游标，
课程从零推进；诊断间隔才计算两条损失的参数梯度分解，日常仅测小型x0执行梯度。
真实学生无合法参考时如实记录physics_updates=0，绝不用GT替换学生预测。
"""

from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import build_actor, build_adapter, seed_all
from .contracts import PhysicsRejected, digest, execution_contract
from .curriculum import physics_selection, physics_weights
from .data import JointDataStream, SidecarStore
from .checkpoint import restore_trainer, save_trainer


def grad_summary(loss, actor, *, retain_graph=True):
    named = [(name, parameter) for name, parameter in actor.named_parameters() if parameter.requires_grad]
    values = torch.autograd.grad(loss, [p for _, p in named], retain_graph=retain_graph, allow_unused=True)
    result = {}
    for group in ("future_encoder", "stage1.denoiser", "stage1.history_encoder", "stage1.music_embedder"):
        terms = [g.float().square().sum() for (name, _), g in zip(named, values) if name.startswith(group) and g is not None]
        result[group] = float(torch.stack(terms).sum().sqrt()) if terms else 0.
    return result


def distributed_execution_scale(local_count, device, world_size):
    """DDP平均梯度与全局有效样本均值的精确换算；零样本rank仍参加collective。"""
    count = torch.tensor(float(local_count), device=device)
    if world_size > 1:
        dist.all_reduce(count)
    return count, world_size/count.clamp_min(1)


class JointTrainer:
    def __init__(self, cfg, device, *, rank=0, world_size=1, resume=None):
        self.cfg, self.device, self.rank, self.world_size = cfg, torch.device(device), rank, world_size
        seed_all(cfg["training"]["seed"])
        self.actor, self.data_loss = build_actor(cfg, device)
        self.stream = JointDataStream(cfg, rank=rank, world_size=world_size)
        self.adapter = build_adapter(cfg, device) if cfg["physics"]["enabled"] or cfg["data"]["sidecar"] else None
        from .prepare_rollouts import asset_contract
        from .execution import ValidatedSupport
        self.assets = asset_contract(cfg, self.adapter) if self.adapter is not None else {
            "stats_sha256": digest(cfg["stage1"]["endecoder"]["stats_path"]),
            "fk_sha256": self.actor.stage1.endecoder.kinematics.kinematics_sha256}
        self.frozen = self.adapter.bundle.frozen_snapshot() if self.adapter is not None else None
        self.store = SidecarStore(cfg["data"]["sidecar"], "train", self.assets,
            sample_contract=cfg["data"]["sample_contract"]) if cfg["data"]["sidecar"] else None
        report = cfg["physics"].get("validation_report")
        self.support = ValidatedSupport(report, self.assets, cfg["physics"]["horizon"], device, execution_contract(cfg)) if report else None
        self.noise_rng = torch.Generator().manual_seed(cfg["training"]["seed"]+1000+rank)
        self.window_rng = np.random.default_rng(cfg["training"]["seed"]+2000+rank)
        self.optimizer = torch.optim.AdamW((p for p in self.actor.parameters() if p.requires_grad), lr=cfg["training"]["learning_rate"],
                                           weight_decay=cfg["training"]["weight_decay"])
        tc = cfg["training"]
        def lr(step):
            if step < tc["lr_warmup_steps"]:
                return (step+1)/max(1, tc["lr_warmup_steps"])
            progress = min(1., (step-tc["lr_warmup_steps"])/max(1, tc["max_steps"]-tc["lr_warmup_steps"]))
            return tc["min_lr_ratio"]+(1-tc["min_lr_ratio"])*.5*(1+math.cos(math.pi*progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr)
        self.step = 0
        self.counters = {"supervised_updates": 0, "physics_updates": 0, "supervised_samples": 0, "physics_samples": 0}
        self.contract = {"assets": self.assets, "dataset": self.stream.catalog.identity,
            "sidecar": self.store.identity if self.store else None, "model": cfg["stage1"]["model"],
            "loss": cfg["stage1"]["loss"], "sampling": cfg["sampling"], "physics": cfg["physics"],
            "execution_loss": cfg["execution_loss"], "training": cfg["training"],
            "data_sample_contract": cfg["data"]["sample_contract"],
            "validation_sha256": digest(report) if report else None}
        if resume:
            restore_trainer(self, resume)
        self.network = DistributedDataParallel(self.actor, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                                              find_unused_parameters=True) if world_size > 1 else self.actor

    def update(self):
        from .reference import attach_sidecars
        from .execution import execute_window
        started = time.perf_counter()
        self.actor.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = self.stream.next(self.cfg["training"]["batch_size"], self.device)
        batch, contexts, windows, missing = attach_sidecars(batch, self.store, self.adapter,
            self.actor.stage1.endecoder.codec, self.cfg["physics"]["horizon"],
            support_guard=self.support.state_guard if self.support else None)
        b = len(batch["decision_time"])
        timesteps = torch.randint(1000, (b,), generator=self.noise_rng).to(self.device)
        noise = torch.randn(b, 60, 30, generator=self.noise_rng).to(self.device)
        drop = (torch.rand(b, generator=self.noise_rng) < self.actor.stage1.music_mask_prob).to(self.device)
        out = self.network(batch, contexts, timesteps=timesteps, noise=noise, drop_music_mask=drop)
        bc, components = self.data_loss(batch, out["pred_x_start"], out["static_conf_logits"], self.step)
        weight, safety_weight = physics_weights(self.step, self.cfg["physics"])
        eligible, snr = physics_selection(out, self.actor.stage1.train_diffusion, self.cfg["physics"])
        rejection, accepted, coverage = Counter(), [], []
        indices = self.window_rng.permutation(b).tolist()
        for i in indices:
            if not self.cfg["physics"]["enabled"]:
                rejection["physics_disabled"] += 1
            elif not weight and not safety_weight:
                rejection["curriculum_data_dominant"] += 1
            elif bool(drop[i]):
                rejection["music_dropped"] += 1
            elif not bool(eligible[i]):
                rejection["snr_excluded"] += 1
            elif windows[i] is None:
                rejection[missing[i]] += 1
            elif self.support is None:
                rejection["new_newton_validation_missing"] += 1
            elif windows[i].source_id not in self.support.source_ids:
                rejection["source_not_in_new_validation"] += 1
            elif len(accepted) >= self.cfg["physics"]["subbatch"]:
                rejection["physics_subbatch_budget"] += 1
            else:
                try:
                    result = execute_window(windows[i], out["pred_x_start"][i:i+1], out["static_conf_logits"][i:i+1],
                        self.actor, self.adapter, self.cfg["execution_loss"], support=self.support)
                    derivative, = torch.autograd.grad(result["loss"], out["pred_x_start"], retain_graph=True)
                    if not torch.isfinite(derivative).all() or not bool(derivative.norm() > 0):
                        raise PhysicsRejected("zero_or_nonfinite_execution_gradient")
                    result["x0_gradient_norm"] = float(derivative.norm())
                    accepted.append(result)
                    coverage.append(result["coverage"])
                except PhysicsRejected as error:
                    rejection[error.reason] += 1
        count, scale = distributed_execution_scale(len(accepted), self.device, self.world_size)
        zero = out["pred_x_start"].sum()*0.
        local_exec = sum((r["loss"] for r in accepted), zero)
        local_safety = sum((r["safety"] for r in accepted), zero)
        # DDP平均rank梯度，因此乘world_size后恰为全局有效样本均值。
        total = bc + scale*(weight*local_exec + safety_weight*local_safety)
        finite = torch.tensor(int(torch.isfinite(total).all()), device=self.device)
        if self.world_size > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError("数据或执行总损失非有限，各rank同步拒绝更新")
        diagnostics = {}
        if self.step % self.cfg["training"]["diagnostics_every"] == 0 and self.world_size == 1:
            diagnostics["data_gradient"] = grad_summary(bc, self.actor)
            diagnostics["execution_gradient"] = grad_summary(local_exec, self.actor)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["training"]["grad_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError("GENMO梯度非有限，未执行优化")
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self.counters["supervised_updates"] += 1
        self.counters["supervised_samples"] += b*self.world_size
        self.counters["physics_updates"] += int(count > 0)
        self.counters["physics_samples"] += int(count)
        reduced = torch.tensor([float(bc.detach()), float(local_exec.detach()), float(local_safety.detach())], device=self.device)
        if self.world_size > 1:
            dist.all_reduce(reduced)
            reasons = [None]*self.world_size
            dist.all_gather_object(reasons, dict(rejection))
            rejection = sum((Counter(r) for r in reasons), Counter())
        return {"step": self.step, **self.counters, "data_loss": float(reduced[0]/self.world_size),
            "execution_loss": float(reduced[1]/count.clamp_min(1)), "execution_safety": float(reduced[2]/count.clamp_min(1)),
            "physics_valid_rate": float(count)/(b*self.world_size), "lambda_exec": weight, "lambda_safety": safety_weight,
            "rejections": dict(rejection), "snr_range": [float(snr.min()), float(snr.max())],
            "timesteps": timesteps.tolist(), "sample_ids": [(m["sample_id"], m["decision_frame"]) for m in batch["meta"]],
            "music_dropout_mask": drop.tolist(), "physics_noise_eligible": int(eligible.sum()),
            "history_sources": [m["proprio_source"] for m in batch["meta"]],
            "data_terms": {k: float(v.detach().mean()) for k, v in components.items() if isinstance(v, torch.Tensor)},
            "gradient_norm": float(norm), **diagnostics, "coverage": coverage,
            "execution_x0_gradient_norm": [r["x0_gradient_norm"] for r in accepted],
            "predicted_failure_steps": [r["failure_steps"] for r in accepted],
            "wall_seconds": time.perf_counter()-started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0}

    def run(self, output, stop_step):
        output = Path(output)
        if self.rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            (output/"config.json").write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2))
        if self.world_size > 1:
            dist.barrier()
        try:
            while self.step < stop_step:
                result = self.update()
                if self.rank == 0:
                    with (output/"metrics.jsonl").open("a") as file:
                        file.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
                    print(json.dumps({k: result[k] for k in ("step", "data_loss", "execution_loss", "physics_updates", "rejections", "wall_seconds")}, ensure_ascii=False), flush=True)
                if self.step % self.cfg["training"]["checkpoint_every"] == 0 or self.step == stop_step:
                    save_trainer(self, output/f"joint_{self.step:08d}.pt")
        finally:
            if self.adapter is not None:
                self.adapter.bundle.assert_unchanged(self.frozen)
        if self.rank == 0:
            summary = {"global_step": self.step, **self.counters, "frozen_unchanged": True,
                       "status": "joint_student_updates_observed" if self.counters["physics_updates"] else "supervision_started_only"}
            (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return self.counters
