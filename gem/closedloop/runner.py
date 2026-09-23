"""Stage1 的持久训练执行器：真实 DDP、混合精度、周期验证和完整断点恢复。

本模块复用已有 Actor、Dataset、逐坐标损失与条件采样，不改变 GRU 历史、MLP 前缀、
qpos30/contact2 或 120帧布局。torchrun 的每个进程绑定 LOCAL_RANK，通过 DDP.forward
执行反传；加权采样按全局抽样位置分片，确定性 draw key 固定裁剪，支持 worker 预取后
精确恢复。优化步是 global_step 的唯一计数单位，梯度累计使用 no_sync，FP32 物理损失
保持原实现，BF16 不使用 GradScaler，FP16 则保存/恢复 scaler。

只有 rank0 写 JSONL、TensorBoard、配置、报告及编号 checkpoint。checkpoint 保存模型、
AdamW、调度器、AMP（若用）、各 rank RNG 与实际消费的采样游标，并绑定数据指纹和训练
关键配置；旧音乐权重仅用于显式 warm start，绝不冒充 resume。验证使用各库固定子集、
固定随机种子，只向 sample 传条件并检查每步及最终物理前缀；验证结束恢复训练 RNG。
从同一 checkpoint 恢复后若在下次保存前中断，可以再次使用同一命令：输出会话自动增加
attempt 后缀，保留前次配置、日志和验证报告；存在更晚完整 checkpoint 时仍拒绝回退。
所有循环有明确 max_steps；本模块不会创建后台任务或自动启动其他实验。
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from gem.closedloop.checkpoint import load_training_checkpoint, save_training_checkpoint
from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.closedloop.stage1_dataset import (
    BumiClosedLoopStage1Dataset,
    collate_stage1_training_samples,
)
from gem.closedloop.training import (
    _loss_values,
    batch_to_device,
    build_stage1_actor,
    build_stage1_loader,
    build_stage1_losses,
    initialize_stage1_weights,
    load_stage1_data_config,
    repository_path,
)
from gem.robots.bumi.kinematics import sha256_file


def plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [torch.cuda.get_rng_state(device)] if device.type == "cuda" else [],
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if len(state["cuda"]) != 1:
            raise ValueError("Expected one local-device CUDA RNG state per rank")
        torch.cuda.set_rng_state(state["cuda"][0], device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def distributed_context(config):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(str(config.runtime.device))
    if device.type == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("Requested local CUDA device is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=datetime.timedelta(seconds=300),
        )
    return rank, world, device


def all_objects(value, world):
    if world == 1:
        return [value]
    gathered = [None] * world
    dist.all_gather_object(gathered, value)
    return gathered


def fingerprint_data(data):
    entries = {}
    for split, datasets in data.datasets.items():
        for name, entry in datasets.items():
            root = Path(str(entry.root))
            entries[f"{split}/{name}"] = {
                "manifest": sha256_file(root / "manifests" / f"{entry.split}.jsonl"),
                "info": sha256_file(root / "meta/dataset_info.json"),
                "options": {
                    k: v for k, v in plain(entry).items() if k not in {"root", "kinematics_path"}
                },
            }
    contents = {"datasets": entries, "sample_contract": plain(data.sample_contract)}
    return hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()


def make_scheduler(optimizer, options):
    total = int(options.total_steps)
    warmup = int(options.warmup_steps)
    minimum = float(options.min_lr_ratio)
    if not 0 <= warmup < total or not 0 <= minimum <= 1:
        raise ValueError("Invalid warmup/cosine scheduler configuration")

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def runtime_contract(config, data, world):
    return {
        "world_size": world,
        "batch_size": int(config.data_loader.batch_size),
        "gradient_accumulation": int(config.trainer.gradient_accumulation),
        "precision": str(config.trainer.precision),
        "data_fingerprint": fingerprint_data(data),
        "sampling": {
            "contract": "global_weighted_rank_stride_deterministic_draw.v1",
            "seed": int(config.runtime.seed),
            "epoch_size_per_rank": int(config.data_loader.samples_per_epoch),
            "weights": plain(data.train_sampling_reference),
        },
        "training_contract": {
            "learning_rate": float(config.train.learning_rate),
            "weight_decay": float(config.train.weight_decay),
            "grad_clip_norm": float(config.train.grad_clip_norm),
            "scheduler": plain(config.scheduler),
            "loss": plain(config.loss),
            "loss_weights_file_sha256": sha256_file(repository_path(config.loss.weights_from)),
        },
    }


def build_monitor_loaders(data, options):
    """每个来源都有固定覆盖，不让ConcatDataset前几个batch代表四库。"""
    result = {}
    for name, entry in data.datasets[str(options.split)].items():
        arguments = OmegaConf.to_container(entry, resolve=True)
        arguments.pop("_target_")
        arguments.update(random_decision=False, duration_aware_sampling=False)
        dataset = BumiClosedLoopStage1Dataset(**arguments)
        count = min(len(dataset), int(options.samples_per_source))
        if count < 1:
            raise ValueError(f"Empty validation source {name}")
        indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
        result[name] = DataLoader(
            Subset(dataset, indices),
            batch_size=1,
            num_workers=0,
            collate_fn=collate_stage1_training_samples,
            generator=torch.Generator().manual_seed(int(options.seed)),
        )
    return result


@torch.no_grad()
def monitor(actor, losses, loaders, options, device, step):
    rng = capture_rng(device)
    was_training = actor.training
    actor.eval()
    losses.eval()
    seed_all(int(options.seed))
    reports = {}
    try:
        for name, loader in loaders.items():
            rows = []
            for source in loader:
                batch = batch_to_device(source, device)
                conditions = {key: batch[key] for key in STAGE1_CONDITION_KEYS}
                generated = actor.sample(
                    conditions,
                    steps=int(options.sample_steps),
                    guidance_scale=float(options.guidance_scale),
                    return_trace=True,
                )
                for key in ("qpos30", "contact", "qpos", "normalized", "contact_logits"):
                    if not torch.isfinite(generated[key]).all():
                        raise FloatingPointError(f"nonfinite validation output: {key}")
                known = batch["known_qpos30_mask"]
                error = (
                    float((generated["qpos30"][known] - batch["known_qpos30"][known]).abs().max())
                    if known.any()
                    else 0.0
                )
                if error != 0:
                    raise AssertionError("physical prefix changed during sampling")
                expected = actor.endecoder.normalize(batch["known_qpos30"])
                trace_error = max(
                    (float((x[known] - expected[known]).abs().max()) if known.any() else 0.0)
                    for x in generated["trace"]
                )
                if trace_error != 0:
                    raise AssertionError(
                        "known coordinates changed in intermediate diffusion steps"
                    )
                if (
                    generated["qpos30"].shape != batch["known_qpos30"].shape
                    or generated["contact"].shape[-1] != 2
                ):
                    raise AssertionError("sampling output contract changed")
                loss, values = losses(
                    batch, generated["normalized"], generated["contact_logits"], global_step=step
                )
                rows.append(
                    {
                        "known_max_abs_error": error,
                        "trace_known_max_abs_error": trace_error,
                        "known_coordinates": int(known.sum()),
                        "qpos30_shape": list(generated["qpos30"].shape),
                        "contact_shape": list(generated["contact"].shape),
                        **_loss_values(loss, values),
                    }
                )
            reports[name] = rows
    finally:
        restore_rng(rng, device)
        actor.train(was_training)
        losses.train(was_training)
    return {
        "step": step,
        "sources": reports,
        "sampling_received_target_fields": False,
        "fixed_seed": int(options.seed),
        "completed_samples": sum(len(x) for x in reports.values()),
    }


def model_digest(actor):
    digest = hashlib.sha256()
    for name, value in actor.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_output_session(output, session, start_step, resume, contract):
    """核验输出归属并选取未用会话名，让保存前崩溃的 resume 可安全重试。"""
    output = Path(output)
    if resume:
        later = [p for p in (output / "checkpoints").glob("s*.pt") if int(p.stem[1:]) > start_step]
        if later:
            raise ValueError(
                "Resume would roll back newer complete checkpoints; use a new output directory"
            )
        previous = output / "run_contract.json"
        if previous.exists() and json.loads(previous.read_text()) != contract:
            raise ValueError("Existing output directory belongs to a different training contract")

    def occupied(candidate):
        # 配置通常最先落盘；也检查其他产物，防止中断或手动归档后覆盖旧报告。
        return (
            (output / f"config_{candidate}.yaml").exists()
            or (output / "reports" / f"{candidate}.json").exists()
            or (output / "reports" / f"resume_{candidate}.json").exists()
            or (output / "validation" / candidate).exists()
            or any((output / "tensorboard").glob(f"events.*.{candidate}"))
        )

    candidate, attempt = session, 1
    while occupied(candidate):
        if not resume:
            raise FileExistsError(f"Training session already exists: {candidate}")
        attempt += 1
        candidate = f"{session}_attempt{attempt:03d}"
    return candidate


def run_persistent(config, output_dir):
    """执行一个有明确终点的同步训练/验证会话；异常时释放分布式资源。"""
    rank, world, device = distributed_context(config)
    writer = None
    try:
        output = Path(output_dir)
        resume = config.get("resume_checkpoint")
        if resume and (config.get("warm_start_checkpoint") or config.get("stage1_checkpoint")):
            raise ValueError("resume and weights-only initialization are mutually exclusive")
        if world > 1 and not output.is_absolute():
            raise ValueError("DDP requires one absolute shared output directory")
        if rank == 0:
            if output.exists() and any(output.iterdir()) and not resume:
                raise FileExistsError(
                    f"new run requires a new empty experiment directory: {output}"
                )
            output.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        torch.set_num_threads(int(config.runtime.get("cpu_threads", 1)))
        seed_all(int(config.runtime.seed))
        data = load_stage1_data_config(config)
        actor = build_stage1_actor(config, data)
        if resume:
            weight_report = None
        else:
            if (
                config.mode == "train"
                and config.trainer.get("require_warm_start", False)
                and not config.get("warm_start_checkpoint")
            ):
                raise ValueError("Formal Stage1 requires explicit music weights-only warm start")
            source = config.get("warm_start_checkpoint")
            expected_sha = config.get("warm_start_checkpoint_sha256")
            if source and expected_sha and sha256_file(repository_path(source)) != expected_sha:
                raise ValueError("warm-start checkpoint SHA256 mismatch")
            weight_report = initialize_stage1_weights(actor, config)
            if source:
                weight_report["source_checkpoint_sha256"] = expected_sha or sha256_file(
                    repository_path(source)
                )
                weight_report["source_checkpoint_path"] = str(repository_path(source))
        actor.to(device)
        losses = build_stage1_losses(actor, config).to(device)
        loaders = build_monitor_loaders(data, config.validation) if rank == 0 else None
        if config.mode == "validate":
            if resume:
                raise ValueError("Validation uses explicit stage1_checkpoint weights, not resume")
            report = (
                monitor(actor, losses, loaders, config.validation, device, 0) if rank == 0 else {}
            )
            if rank == 0:
                write_json(output / "validation.json", report)
            return report
        if config.mode != "train":
            raise ValueError("mode must be train or validate")
        contract = runtime_contract(config, data, world)
        batch_size = contract["batch_size"]
        accumulation = contract["gradient_accumulation"]
        epoch_size = int(config.data_loader.samples_per_epoch)
        if batch_size <= 0 or accumulation <= 0 or epoch_size % (batch_size * accumulation):
            raise ValueError("samples_per_epoch must be divisible by per-rank batch * accumulation")
        precision = contract["precision"]
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        if precision != "fp32" and device.type != "cuda":
            raise ValueError("Mixed precision runner requires CUDA")
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=float(config.train.learning_rate),
            weight_decay=float(config.train.weight_decay),
        )
        scheduler = make_scheduler(optimizer, config.scheduler)
        step, epoch, offset = 0, 0, 0
        restore_report = None
        if resume:
            restored = load_training_checkpoint(
                actor,
                repository_path(resume),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                expected_runtime=contract,
            )
            step = int(restored["global_step"])
            rank_state = restored["runtime_state"]["rank_states"][rank]
            epoch, offset = rank_state["sampler_epoch"], rank_state["sampler_offset"]
            weight_report = restored["warm_start_report"]
            if config.trainer.get("require_warm_start", False):
                if not weight_report or weight_report.get("mode") != "weights_only_warm_start":
                    raise ValueError("Formal resume requires recorded music warm-start lineage")
                if weight_report.get("source_checkpoint_sha256") != config.get(
                    "warm_start_checkpoint_sha256"
                ):
                    raise ValueError("Formal resume initial music checkpoint lineage SHA mismatch")
            for saved in restored["runtime_state"]["rank_states"]:
                if (saved["sampler_epoch"], saved["sampler_offset"]) != (epoch, offset):
                    raise ValueError("Resume rank sampling cursors disagree")
            if (
                offset > epoch_size
                or offset % (batch_size * accumulation)
                or epoch * epoch_size + offset != step * batch_size * accumulation
            ):
                raise ValueError(
                    "Resume sampling cursor does not match completed optimizer updates"
                )
            restore_report = {
                k: v
                for k, v in restored.items()
                if k not in {"runtime_state", "loaded", "warm_start_report"}
            }
        start_step = step
        maximum = int(config.train.max_steps)
        if maximum <= step or maximum > int(config.scheduler.total_steps):
            raise ValueError(
                "max_steps must exceed restored global_step and not exceed scheduler total_steps"
            )
        model = (
            DistributedDataParallel(
                actor,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if world > 1
            else actor
        )
        loader = build_stage1_loader(
            data,
            split="train",
            batch_size=batch_size,
            num_workers=int(config.data_loader.num_workers),
            seed=int(config.runtime.seed),
            samples_per_epoch=epoch_size,
            rank=rank,
            world_size=world,
            resumable=True,
            pin_memory=device.type == "cuda",
        )
        if offset == epoch_size:
            epoch, offset = epoch + 1, 0
        loader.sampler.set_epoch(epoch, offset)
        iterator = iter(loader)
        if resume:
            restore_report.update(
                rng_restored=True, sampler_restored=True, sampler_epoch=epoch, sampler_offset=offset
            )
        session = f"from_s{start_step:06d}_to_s{maximum:06d}"
        if rank == 0:
            session = check_output_session(output, session, start_step, resume, contract)
            if not (output / "run_contract.json").exists():
                write_json(output / "run_contract.json", contract)
            if not (output / "weight_loading_report.json").exists():
                write_json(output / "weight_loading_report.json", weight_report)
            if restore_report:
                restore_report["session"] = session
                write_json(output / "reports" / f"resume_{session}.json", restore_report)
            OmegaConf.save(config, output / f"config_{session}.yaml", resolve=True)
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(
                str(output / "tensorboard"),
                purge_step=start_step + 1 if resume else None,
                filename_suffix=f".{session}",
            )
        if resume:
            restore_rng(rank_state["rng"], device)
        else:
            seed_all(int(config.runtime.seed) + rank)
        actor.train()
        losses.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        step_reports, validations = [], []
        last_checkpoint = None
        loop_start = time.perf_counter()
        while step < maximum:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            examples = []
            lr_used = optimizer.param_groups[0]["lr"]
            for micro in range(accumulation):
                if offset == epoch_size:
                    epoch, offset = epoch + 1, 0
                    loader.sampler.set_epoch(epoch, 0)
                    iterator = iter(loader)
                batch = batch_to_device(next(iterator), device)
                examples.extend(
                    f"{m.get('dataset_id', '')}:{m.get('sample_id')}:{m.get('decision_frame')}"
                    for m in batch["meta"]
                )
                sync = (
                    model.no_sync()
                    if world > 1 and micro < accumulation - 1
                    else contextlib.nullcontext()
                )
                with sync:
                    with torch.autocast(
                        device_type=device.type, dtype=dtype, enabled=precision != "fp32"
                    ):
                        prediction = model(batch)
                        loss, values = losses(
                            batch,
                            prediction["pred_x_start"],
                            prediction["static_conf_logits"],
                            global_step=step,
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    scaled = loss / accumulation
                    (scaler.scale(scaled) if scaler else scaled).backward()
                for key, value in _loss_values(loss, values).items():
                    accumulated[key] = accumulated.get(key, 0.0) + value / accumulation
                offset += batch_size
            if scaler:
                scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), float(config.train.grad_clip_norm), error_if_nonfinite=True
            )
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - begin
            local = {
                "rank": rank,
                "step": step,
                "gradient_norm": float(norm),
                "seconds": duration,
                "sample_signature": hashlib.sha256("|".join(examples).encode()).hexdigest(),
                "sampler_epoch": epoch,
                "sampler_offset": offset,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0.0,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20
                if device.type == "cuda"
                else 0.0,
                **accumulated,
            }
            ranks = all_objects(local, world)
            metrics = {k: sum(r[k] for r in ranks) / world for k in accumulated}
            row = {
                "session": session,
                "step": step,
                "learning_rate": lr_used,
                "gradient_norm": max(r["gradient_norm"] for r in ranks),
                "step_seconds": max(r["seconds"] for r in ranks),
                "ranks": ranks,
                **metrics,
            }
            if rank == 0:
                with (output / "train_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                for key in ("loss", "gradient_norm", "learning_rate", "step_seconds"):
                    writer.add_scalar(f"train/{key}", row[key], step)
                if step % int(config.trainer.log_every_steps) == 0 or step == maximum:
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in [
                                    "step",
                                    "loss",
                                    "gradient_norm",
                                    "learning_rate",
                                    "step_seconds",
                                ]
                            }
                        ),
                        flush=True,
                    )
            step_reports.append(
                {
                    k: row[k]
                    for k in ["step", "loss", "gradient_norm", "learning_rate", "step_seconds"]
                }
            )
            do_validation = step % int(config.trainer.validate_every_steps) == 0 or step == maximum
            if do_validation:
                validation = (
                    monitor(actor, losses, loaders, config.validation, device, step)
                    if rank == 0
                    else None
                )
                if rank == 0:
                    write_json(output / "validation" / session / f"s{step:06d}.json", validation)
                    validations.append(validation)
                    for name, samples in validation["sources"].items():
                        writer.add_scalar(
                            f"val/{name}/loss", sum(r["loss"] for r in samples) / len(samples), step
                        )
                if world > 1:
                    dist.barrier()
            if step % int(config.trainer.save_every_steps) == 0 or step == maximum:
                if step == maximum:
                    digests = all_objects(model_digest(actor), world)
                    if len(set(digests)) != 1:
                        raise AssertionError(
                            "DDP models differ before final checkpoint publication"
                        )
                rank_state = {
                    "rank": rank,
                    "rng": capture_rng(device),
                    "sampler_epoch": epoch,
                    "sampler_offset": offset,
                }
                states = all_objects(rank_state, world)
                last_checkpoint = output / "checkpoints" / f"s{step:06d}.pt"
                if rank == 0:
                    save_training_checkpoint(
                        actor,
                        last_checkpoint,
                        config=plain(config),
                        global_step=step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        runtime_state={**contract, "rank_states": states, "completed_steps": step},
                        warm_start_report=weight_report,
                    )
                    write_json(
                        output / "latest.json",
                        {"global_step": step, "checkpoint": str(last_checkpoint)},
                    )
                    writer.flush()
                if world > 1:
                    dist.barrier()
        identities = all_objects(
            {
                "rank": rank,
                "state_sha256": model_digest(actor),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0.0,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20
                if device.type == "cuda"
                else 0.0,
            },
            world,
        )
        if len({r["state_sha256"] for r in identities}) != 1:
            raise AssertionError("DDP rank model states diverged")
        report = {
            "status": "completed",
            "session": session,
            "start_step": start_step,
            "global_step": step,
            "executed_optimizer_steps": step - start_step,
            "world_size": world,
            "batch_size_per_rank": batch_size,
            "gradient_accumulation": accumulation,
            "effective_global_batch": world * batch_size * accumulation,
            "precision": precision,
            "steps": step_reports,
            "wall_seconds_including_validation_checkpoint": time.perf_counter() - loop_start,
            "ranks": identities,
            "rank_weights_identical": True,
            "resume": restore_report,
            "warm_start_mode": weight_report["mode"],
            "last_checkpoint": str(last_checkpoint),
            "validation_steps": [v["step"] for v in validations],
        }
        if rank == 0:
            write_json(output / "reports" / f"{session}.json", report)
        return report if rank == 0 else {"rank": rank, "status": "completed"}
    finally:
        if writer:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
