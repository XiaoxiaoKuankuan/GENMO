"""优势加权单时间步扩散训练、真实Stage1 BC及可恢复同步更新。

Replay每Epoch均匀遍历一次，先逐样本Mask归约再乘固定权重；每条样本只执行一个
重新抽取的扩散时间。AWR与BC使用各自真实全局分母、独立随机流，在SUM梯度后裁剪。
校准只诊断梯度并恢复所有RNG，不更新参数；宏迭代内固定BC系数。这里不调用采样链，
不构造Contact伪标签，不把参数变化当作物理收益。
"""
from __future__ import annotations

import copy
import math
import time
import torch

from .checkpoint import capture_rng, restore_rng
from .io import cpu_copy, digest
from .losses import weighted_denoising


def gradient_snapshot(model):
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}


def gradient_report(first, second=None):
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in first.values()))
    result = dict(norm=norm, modules={})
    for name, g in first.items():
        module = name.split(".")[0]
        result["modules"][module] = result["modules"].get(module, 0.)+float(g.double().square().sum())
    result["modules"] = {k: math.sqrt(v) for k, v in result["modules"].items()}
    if second is not None:
        other = math.sqrt(sum(float(g.double().square().sum()) for g in second.values()))
        dot = sum(float(first[n].double().flatten().dot(second[n].double().flatten()))
                  for n in first.keys() & second.keys())
        result.update(other_norm=other, cosine=dot/(norm*other) if norm*other > 0 else None)
    return result


class ActorTrainer:
    def __init__(self, actor, dataset, bc, config, distributed, *, seed=42, bc_weight=None):
        self.actor, self.dataset, self.bc = actor, dataset, bc
        self.config, self.distributed, self.seed = config, distributed, seed
        self.bc_weight = bc_weight
        self.optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
                                          lr=config["lr"], weight_decay=config.get("weight_decay", .01))
        self.total_steps = config["epochs"]*math.ceil(len(dataset)/config["global_batch"])
        warmup = config.get("warmup_steps", 100)
        def scale(step):
            if step < warmup:
                return (step+1)/max(warmup, 1)
            phase = min(1., (step-warmup)/max(self.total_steps-warmup-1, 1))
            return config.get("min_lr_ratio", .1)+(1-config.get("min_lr_ratio", .1))*(1+math.cos(math.pi*phase))/2
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scale)
        self.epoch, self.cursor, self.steps = 0, 0, 0
        self.history = []

    def state_dict(self):
        return dict(actor=cpu_copy(self.actor.state_dict()), optimizer=cpu_copy(self.optimizer.state_dict()),
                    scheduler=self.scheduler.state_dict(), bc=self.bc.state_dict(),
                    bc_weight=self.bc_weight, epoch=self.epoch, cursor=self.cursor, steps=self.steps,
                    rng=capture_rng(self.distributed.device), history=self.history,
                    data_identity=digest(self.dataset.sample_ids), total_steps=self.total_steps,
                    seed=self.seed, config=self.config)

    def load_state_dict(self, state, *, rank_state=None):
        if (state["data_identity"] != digest(self.dataset.sample_ids) or state["total_steps"] != self.total_steps
                or state["seed"] != self.seed or state["config"] != self.config):
            raise ValueError("Actor optimizer/data/budget identity differs on resume")
        self.actor.load_state_dict(state["actor"]); self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        local = rank_state or state
        self.bc.load_state_dict(local["bc"]); restore_rng(local["rng"], self.distributed.device)
        self.bc_weight, self.epoch, self.cursor, self.steps = (state[k] for k in
                                                           ("bc_weight", "epoch", "cursor", "steps"))
        self.history = state["history"]

    def _noise(self, indices, *, step):
        # 与方法、rank、microbatch无关；B/C相同样本次序得到完全相同t与epsilon。
        noise, times = [], []
        for index in indices:
            value = int(digest([self.seed, step, self.dataset.sample_ids[index]])[:15], 16)
            rng = torch.Generator().manual_seed(value)
            times.append(torch.randint(self.actor.diffusion_steps, (), generator=rng))
            noise.append(torch.randn(self.dataset.rows[index].target_qpos30.shape, generator=rng))
        return torch.stack(times).to(self.distributed.device), torch.stack(noise).to(self.distributed.device)

    def backward_awr(self, indices, *, step):
        d, c = self.distributed, self.config
        local = indices[d.rank::d.world_size]
        total = 0.
        for offset in range(0, len(local), c["microbatch"]):
            ids = local[offset:offset+c["microbatch"]]
            batch = self.dataset.batch(ids, d.device)
            t, noise = self._noise(ids, step=step)
            out = self.actor.denoising_forward(batch["conditions"], batch["target"], batch["target_valid"],
                                               timesteps=t, noise=noise, music_dropout=False)
            loss, _ = weighted_denoising(out["pred_x_start"], out["target_x_start"], batch["free_mask"],
                                         batch["weights"], global_count=len(indices))
            loss.backward(); total += float(loss.detach())
        return float(d.sum(torch.tensor(total, dtype=torch.float64, device=d.device)))

    def backward_bc(self, coefficient):
        d, c = self.distributed, self.config
        count = len(range(d.rank, c["bc_global_batch"], d.world_size))
        total = 0.
        for start in range(0, count, c["bc_microbatch"]):
            losses = self.bc.loss(self.actor, min(c["bc_microbatch"], count-start))
            loss = losses.double().sum()/c["bc_global_batch"]
            (coefficient*loss).backward(); total += float(loss.detach())
        return float(d.sum(torch.tensor(total, dtype=torch.float64, device=d.device)))

    def calibrate(self, batches=8, ratio=.1):
        original_rng = capture_rng(self.distributed.device)
        bc_state = copy.deepcopy(self.bc.state_dict())
        reports = []
        self.actor.train()
        try:
            order = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.seed)).tolist()
            for batch in range(batches):
                start = (batch*self.config["global_batch"]) % len(order)
                ids = order[start:start+self.config["global_batch"]]
                self.optimizer.zero_grad(set_to_none=True)
                awr_loss = self.backward_awr(ids, step=batch)
                self.distributed.gradients(self.actor)
                first = gradient_snapshot(self.actor)
                self.optimizer.zero_grad(set_to_none=True)
                bc_loss = self.backward_bc(1.)
                self.distributed.gradients(self.actor)
                second = gradient_snapshot(self.actor)
                report = gradient_report(first, second)
                if min(report["norm"], report["other_norm"]) <= 0:
                    raise ValueError("Cannot calibrate BC from zero/nonfinite real gradients")
                report.update(awr_loss=awr_loss, bc_loss=bc_loss, bc_modules=gradient_report(second)["modules"])
                reports.append(report)
            self.bc_weight = ratio*sum(r["norm"] for r in reports)/sum(r["other_norm"] for r in reports)
            if not math.isfinite(self.bc_weight) or self.bc_weight <= 0:
                raise ValueError("Invalid fixed BC coefficient")
            return dict(batches=reports, coefficient=self.bc_weight, target_gradient_ratio=ratio,
                        optimizer_updates=0, rng_restored=True)
        finally:
            self.bc.load_state_dict(bc_state); restore_rng(original_rng, self.distributed.device)
            self.optimizer.zero_grad(set_to_none=True); self.actor.eval()

    def run(self, *, on_step=None, should_stop=lambda: False):
        if self.bc_weight is None or self.bc_weight <= 0:
            raise ValueError("A fixed BC coefficient is required before Actor updates")
        started = time.perf_counter()
        c, d = self.config, self.distributed
        self.actor.train()
        try:
            while self.epoch < c["epochs"]:
                epoch_start = time.perf_counter()
                order = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(
                    self.seed+self.epoch)).tolist()
                while self.cursor < len(order):
                    if d.root(should_stop):
                        return dict(complete=False, steps=self.steps, seconds=time.perf_counter()-started)
                    ids = order[self.cursor:self.cursor+c["global_batch"]]
                    step_start = time.perf_counter()
                    self.optimizer.zero_grad(set_to_none=True)
                    awr_loss = self.backward_awr(ids, step=self.steps)
                    awr_done = time.perf_counter()
                    bc_loss = self.backward_bc(self.bc_weight)
                    d.gradients(self.actor)
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.actor.parameters() if p.requires_grad], c.get("gradient_clip", 1.),
                        error_if_nonfinite=True)
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    self.optimizer.step(); self.scheduler.step()
                    self.steps += 1; self.cursor += len(ids); self.bc.bc_update_steps += 1
                    entry = dict(step=self.steps, epoch=self.epoch, samples=len(ids), awr_loss=awr_loss,
                                 bc_loss=bc_loss, bc_weight=self.bc_weight, total_loss=awr_loss+self.bc_weight*bc_loss,
                                 gradient_norm=float(norm), clipped_gradient_norm=min(float(norm), c.get("gradient_clip", 1.)),
                                 lr=learning_rate, awr_seconds=awr_done-step_start,
                                 bc_and_update_seconds=time.perf_counter()-awr_done,
                                 seconds=time.perf_counter()-step_start)
                    self.history.append(entry)
                    if on_step:
                        on_step(self, entry)
                self.history[-1]["epoch_seconds"] = time.perf_counter()-epoch_start
                self.epoch += 1; self.cursor = 0
            d.assert_models_equal(self.actor)
            return dict(complete=True, steps=self.steps, seconds=time.perf_counter()-started,
                        replay_samples=len(self.dataset), physical_samples_duplicated=False,
                        replay_read_seconds=self.dataset.read_seconds,
                        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(d.device) if d.device.type == "cuda" else 0)
        finally:
            self.actor.eval()
