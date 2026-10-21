"""优势加权单时间步扩散训练、真实Stage1 BC及可恢复同步更新。

Replay每Epoch均匀遍历一次，先逐样本Mask归约再乘固定权重；每条样本只执行一个
重新抽取的扩散时间。AWR与BC使用各自真实全局分母、独立随机流，在SUM梯度后裁剪。
校准只诊断梯度并恢复所有RNG，不更新参数；宏迭代内固定BC系数。这里不调用采样链，
不构造Contact伪标签，不把参数变化当作物理收益。
Replay监督与部署共用CFG两分支组合。当前signals.v2明确关闭冻结教师且不分配教师
网络；旧一致性实现仅供历史诊断复现。真实BC保持独立，不伪造Contact监督。
bc_enabled=false时不创建BC采样器、不校准/前反向、不推进BC数据游标；Contact无伪标签。
时间与来源权重是显式目标配置，在整轮固定，不改变AWR标签或microbatch分母。
"""
from __future__ import annotations

import copy
import math
import time
import torch

from .checkpoint import capture_rng, restore_rng
from .io import cpu_copy, digest
from .losses import weighted_denoising, per_sample_denoising
from .metrics import actor_module_name
from .objective import timestep_weights, source_weights


def gradient_snapshot(model):
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}


def gradient_report(first, second=None):
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in first.values()))
    result = dict(norm=norm, modules={})
    for name, g in first.items():
        module = actor_module_name(name)
        result["modules"][module] = result["modules"].get(module, 0.)+float(g.double().square().sum())
    result["modules"] = {k: math.sqrt(v) for k, v in result["modules"].items()}
    if second is not None:
        other = math.sqrt(sum(float(g.double().square().sum()) for g in second.values()))
        dot = sum(float(first[n].double().flatten().dot(second[n].double().flatten()))
                  for n in first.keys() & second.keys())
        result.update(other_norm=other, cosine=dot/(norm*other) if norm*other > 0 else None)
    return result


def choose_bc_coefficient(calibration, fixed=None):
    """保留真实梯度诊断并支持显式固定BC系数，不以自蒸馏物理质量阻止AWR。"""
    result = copy.deepcopy(calibration)
    if fixed is not None:
        if isinstance(fixed, bool) or not math.isfinite(fixed) or fixed <= 0:
            raise ValueError("Fixed BC coefficient must be finite and positive")
        result["gradient_calibrated_coefficient"] = result["coefficient"]
        result["coefficient"] = float(fixed)
        result["coefficient_source"] = "explicit_fixed_config"
    else:
        result["coefficient_source"] = "current_macro_gradient_calibration"
    result["actual_gradient_ratio"] = result["coefficient"]*sum(
        row["other_norm"] for row in result["batches"])/sum(row["norm"] for row in result["batches"])
    return result


class ActorTrainer:
    def __init__(self, actor, dataset, bc, config, distributed, *, seed=42, bc_weight=None):
        self.actor, self.dataset, self.bc = actor, dataset, bc
        self.config, self.distributed, self.seed = config, distributed, seed
        self.bc_enabled = config.get("bc_enabled", True)
        if type(self.bc_enabled) is not bool or (self.bc_enabled and bc is None) or (not self.bc_enabled and bc is not None):
            raise ValueError("BC sampler presence must match explicit bc_enabled")
        if not self.bc_enabled and bc_weight not in (None, 0.):
            raise ValueError("Disabled BC cannot retain a nonzero coefficient")
        if not self.bc_enabled and config.get("bc_global_batch", 0) != 0:
            raise ValueError("Disabled BC requires zero global batch")
        self.bc_weight = bc_weight if self.bc_enabled else 0.
        self.time_weights = timestep_weights(actor.train_diffusion.alphas_cumprod,
                                             config.get("timestep_weighting"))
        self.source_weights = source_weights(dataset.sources, config.get("source_balance", "none"))
        self.objective_identity = digest(dict(timestep_weights=self.time_weights.tolist(),
            source_weights=self.source_weights.tolist(), advantage_weights=dataset.weights.tolist(),
            bc_enabled=self.bc_enabled))
        self.reference_weight = float(config.get("reference_weight", 0.))
        if not math.isfinite(self.reference_weight) or self.reference_weight < 0:
            raise ValueError("Invalid frozen-reference coefficient")
        self.reference_actor = (copy.deepcopy(actor).eval().requires_grad_(False)
                                if self.reference_weight else None)
        self.last_reference_loss = 0.
        self.optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
                                          lr=config["lr"], weight_decay=config.get("weight_decay", .01))
        self.total_steps = config["epochs"]*math.ceil(len(dataset)/config["global_batch"])
        if config.get("max_optimizer_steps") is not None:
            if type(config["max_optimizer_steps"]) is not int or config["max_optimizer_steps"] < 1:
                raise ValueError("max_optimizer_steps must be positive")
            self.total_steps = min(self.total_steps, config["max_optimizer_steps"])
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
                    scheduler=self.scheduler.state_dict(), bc=self.bc.state_dict() if self.bc_enabled else None,
                    bc_weight=self.bc_weight, epoch=self.epoch, cursor=self.cursor, steps=self.steps,
                    rng=capture_rng(self.distributed.device), history=self.history,
                    data_identity=digest(self.dataset.sample_ids), total_steps=self.total_steps,
                    seed=self.seed, config=self.config, objective_identity=self.objective_identity,
                    reference_actor=(cpu_copy(self.reference_actor.state_dict())
                                     if self.reference_actor is not None else None))

    def load_state_dict(self, state, *, rank_state=None):
        if (state["data_identity"] != digest(self.dataset.sample_ids) or state["total_steps"] != self.total_steps
                or state["seed"] != self.seed or state["config"] != self.config
                or state.get("objective_identity") != self.objective_identity):
            raise ValueError("Actor optimizer/data/budget identity differs on resume")
        self.actor.load_state_dict(state["actor"]); self.optimizer.load_state_dict(state["optimizer"])
        if self.reference_actor is not None:
            if state.get("reference_actor") is None:
                raise ValueError("Frozen reference missing from checkpoint")
            self.reference_actor.load_state_dict(state["reference_actor"])
        self.scheduler.load_state_dict(state["scheduler"])
        local = rank_state or state
        if self.bc_enabled:
            self.bc.load_state_dict(local["bc"])
        elif local.get("bc") is not None or state["bc_weight"] != 0.:
            raise ValueError("Disabled BC checkpoint must contain no sampler state or coefficient")
        restore_rng(local["rng"], self.distributed.device)
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
        total, reference_total, unscaled_total = 0., 0., 0.
        for offset in range(0, len(local), c["microbatch"]):
            ids = local[offset:offset+c["microbatch"]]
            batch = self.dataset.batch(ids, d.device)
            t, noise = self._noise(ids, step=step)
            # Replay监督对应部署去噪器；eval只关闭Dropout，梯度仍然开启。
            mode = self.actor.training
            if c.get("replay_eval_mode", True):
                self.actor.eval()
            try:
                kwargs = dict(timesteps=t, noise=noise, music_dropout=False,
                              guidance_scale=c.get("denoising_guidance_scale", 2.5))
                out = self.actor.denoising_forward(batch["conditions"], batch["target"],
                                                   batch["target_valid"], **kwargs)
                teacher = None
                if self.reference_actor is not None:
                    with torch.no_grad():
                        teacher = self.reference_actor.denoising_forward(batch["conditions"], batch["target"],
                                                                         batch["target_valid"], **kwargs)
            finally:
                self.actor.train(mode)
            time_factor = self.time_weights[t.detach().cpu()].to(d.device)
            source_factor = self.source_weights[ids].to(d.device)
            loss, per_sample = weighted_denoising(out["pred_x_start"], out["target_x_start"], batch["free_mask"],
                batch["weights"]*time_factor*source_factor, global_count=len(indices))
            unscaled_total += float((per_sample.detach().double()*batch["weights"]).sum()/len(indices))
            objective = loss*c.get("replay_coefficient", 1.)
            if teacher is not None:
                # 两分支各自锚定，避免CFG输出相同但分支互相抵消；不使用Contact伪标签。
                anchor = sum(per_sample_denoising(out[key], teacher[key], batch["free_mask"])
                             for key in ("conditional_x_start", "unconditional_x_start"))*.5
                anchor = anchor.double().sum()/len(indices)
                objective = objective+self.reference_weight*anchor
                reference_total += float(anchor.detach())
            objective.backward(); total += float(loss.detach())
        self.last_reference_loss = float(d.sum(torch.tensor(reference_total, dtype=torch.float64, device=d.device)))
        self.last_unscaled_loss = float(d.sum(torch.tensor(unscaled_total, dtype=torch.float64, device=d.device)))
        return float(d.sum(torch.tensor(total, dtype=torch.float64, device=d.device)))

    def backward_bc(self, coefficient):
        if not self.bc_enabled:
            if coefficient != 0.:
                raise ValueError("Disabled BC coefficient must be zero")
            return 0.
        d, c = self.distributed, self.config
        count = len(range(d.rank, c["bc_global_batch"], d.world_size))
        total = 0.
        for start in range(0, count, c["bc_microbatch"]):
            losses = self.bc.loss(self.actor, min(c["bc_microbatch"], count-start))
            loss = losses.double().sum()/c["bc_global_batch"]
            (coefficient*loss).backward(); total += float(loss.detach())
        return float(d.sum(torch.tensor(total, dtype=torch.float64, device=d.device)))

    def calibrate(self, batches=8, ratio=.1):
        if not self.bc_enabled:
            self.bc_weight = 0.
            return dict(batches=[], coefficient=0., coefficient_source="bc_disabled",
                        target_gradient_ratio=0., actual_gradient_ratio=0., optimizer_updates=0,
                        bc_forward_backward_calls=0, rng_restored=True)
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
        if self.bc_enabled and (self.bc_weight is None or self.bc_weight <= 0):
            raise ValueError("A fixed BC coefficient is required before Actor updates")
        started = time.perf_counter()
        c, d = self.config, self.distributed
        self.actor.train()
        try:
            while self.epoch < c["epochs"] and self.steps < self.total_steps:
                epoch_start = time.perf_counter()
                order = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(
                    self.seed+self.epoch)).tolist()
                while self.cursor < len(order) and self.steps < self.total_steps:
                    if d.root(should_stop):
                        return dict(complete=False, steps=self.steps, seconds=time.perf_counter()-started)
                    ids = order[self.cursor:self.cursor+c["global_batch"]]
                    step_start = time.perf_counter()
                    self.optimizer.zero_grad(set_to_none=True)
                    awr_loss = self.backward_awr(ids, step=self.steps)
                    awr_done = time.perf_counter()
                    bc_loss = self.backward_bc(self.bc_weight)
                    bc_done = time.perf_counter()
                    d.gradients(self.actor)
                    if d.device.type == "cuda":
                        torch.cuda.synchronize(d.device)
                    reduction_done = time.perf_counter()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.actor.parameters() if p.requires_grad], c.get("gradient_clip", 1.),
                        error_if_nonfinite=True)
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    self.optimizer.step(); self.scheduler.step()
                    if d.device.type == "cuda":
                        torch.cuda.synchronize(d.device)
                    update_done = time.perf_counter()
                    self.steps += 1; self.cursor += len(ids)
                    if self.bc_enabled:
                        self.bc.bc_update_steps += 1
                    entry = dict(step=self.steps, epoch=self.epoch, samples=len(ids), awr_loss=awr_loss,
                                 unscaled_awr_loss=self.last_unscaled_loss,
                                 bc_loss=bc_loss, bc_weight=self.bc_weight, bc_enabled=self.bc_enabled,
                                 reference_loss=self.last_reference_loss, reference_weight=self.reference_weight,
                                 total_loss=c.get("replay_coefficient", 1.)*awr_loss+self.bc_weight*bc_loss+self.reference_weight*self.last_reference_loss,
                                 gradient_norm=float(norm), clipped_gradient_norm=min(float(norm), c.get("gradient_clip", 1.)),
                                 lr=learning_rate, awr_seconds=awr_done-step_start,
                                 bc_forward_backward_seconds=bc_done-awr_done,
                                 gradient_reduction_seconds=reduction_done-bc_done,
                                 clip_optimizer_seconds=update_done-reduction_done,
                                 bc_and_update_seconds=update_done-awr_done,
                                 seconds=update_done-step_start)
                    self.history.append(entry)
                    if on_step:
                        on_step(self, entry)
                self.history[-1]["epoch_seconds"] = time.perf_counter()-epoch_start
                self.history[-1]["epoch_complete"] = self.cursor == len(order)
                if self.cursor == len(order):
                    self.epoch += 1; self.cursor = 0
            d.assert_models_equal(self.actor)
            return dict(complete=True, steps=self.steps, seconds=time.perf_counter()-started,
                        bc_enabled=self.bc_enabled, bc_optimizer_steps=self.steps if self.bc_enabled else 0,
                        objective_identity=self.objective_identity,
                        timestep_weighting=dict(config=c.get("timestep_weighting", {"mode": "uniform"}),
                            mean=float(self.time_weights.mean()), minimum=float(self.time_weights.min()),
                            maximum=float(self.time_weights.max()), table=self.time_weights.tolist()),
                        source_balance=c.get("source_balance", "none"),
                        source_objective={s:dict(samples=self.dataset.sources.count(s),
                            multiplier=float(self.source_weights[self.dataset.sources.index(s)]))
                            for s in sorted(set(self.dataset.sources))},
                        complete_epochs=self.epoch, partial_epoch_samples=self.cursor,
                        replay_samples=len(self.dataset), physical_samples_duplicated=False,
                        epoch_seconds={str(e["epoch"]): e["epoch_seconds"]
                                       for e in self.history if "epoch_seconds" in e},
                        awr_forward_backward_seconds=sum(e["awr_seconds"] for e in self.history),
                        bc_reduction_update_seconds=sum(e["bc_and_update_seconds"] for e in self.history),
                        bc_forward_backward_seconds=sum(e["bc_forward_backward_seconds"] for e in self.history),
                        gradient_reduction_seconds=sum(e["gradient_reduction_seconds"] for e in self.history),
                        clip_optimizer_seconds=sum(e["clip_optimizer_seconds"] for e in self.history),
                        optimizer_step_seconds_mean=sum(e["seconds"] for e in self.history)/self.steps,
                        replay_read_seconds=self.dataset.read_seconds,
                        peak_gpu_memory_scope="process_start_through_this_actor_training",
                        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(d.device) if d.device.type == "cuda" else 0)
        finally:
            self.actor.eval()
