"""MotionMillion 续训的固定验证、动作留档与带退化保护的轻量早停。

每个 DDP rank 分担固定验证 cohort，使用训练缓存中的 T5 文本和确定性中心裁剪。
样本增强、扩散噪声与生成种子均固定，检查前后恢复 Python/NumPy/Torch RNG 和模型
训练模式。复用 GEM.prepare_batch/train_step/validation_step 以及现有 SMPL FK，
不加载第二份生成模型，不改变训练损失、数据采样或优化器状态。

选模量为固定噪声下的去噪验证损失；32 个动作各四个 seed 先在动作内平均，再以
动作配对 bootstrap 判断改善。只有损失可靠改善且相对起点的多样性/脚滑代理指标
不退化，才更新 best；连续三次不满足即请求所有 rank 正常结束。它不是官方 FID、
文本检索或动力学评测，报告显式记录该边界。

训练记录保留 history.json 和 cohort.json；动作仅保留 baseline.npz 与 latest.npz。
写入使用同目录临时文件和原子替换，失败删除本次临时文件并保留旧交付。完整 callback
状态随 checkpoint 保存，因此阶段内恢复不会重置 baseline、最佳值或 patience。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gem.utils.lr_scheduler import ContinuationWarmupCosineLR


@contextmanager
def isolated_rng(seed, device):
    """只操作当前 rank 的 CUDA RNG，不给其他 GPU 创建上下文。"""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def atomic_artifact(path, payload, *, npz=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if npz:
                np.savez_compressed(stream, **payload)
            else:
                stream.write(
                    (
                        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                    ).encode()
                )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def motion_proxies(joints, fps=30.0):
    """Y-up FK 的近地低垂速脚滑代理；不推断真实接触力或真实地面高度。"""
    joints = np.asarray(joints, dtype=np.float64)
    if (
        joints.ndim != 3
        or joints.shape[1:] != (22, 3)
        or len(joints) < 3
        or not np.isfinite(joints).all()
    ):
        raise ValueError("质量监控要求有限 [F>=3,22,3] 世界关节")
    feet = joints[:, [10, 11]]
    velocity = np.diff(feet, axis=0) * fps
    ground = np.quantile(feet[..., 1], 0.05)
    contact = (feet[:-1, :, 1] < ground + 0.05) & (np.abs(velocity[..., 1]) < 0.15)
    slide = np.linalg.norm(velocity[..., [0, 2]], axis=-1)
    local = joints - joints[:, [0]]
    return {
        "slide_sum": float(slide[contact].sum()),
        "contact_count": int(contact.sum()),
        "contact_total": int(contact.size),
        "acceleration_mps2": float(
            np.linalg.norm(np.diff(local, n=2, axis=0) * fps**2, axis=-1).mean()
        ),
    }


def quality_decision(
    history,
    report,
    *,
    patience,
    min_relative_improvement,
    min_diversity_ratio,
    max_slide_ratio,
    slide_tolerance,
):
    """纯函数：配对 bootstrap、基线退化保护及连续检查的早停判定。"""
    losses = np.asarray(report["per_motion_loss"], dtype=float)
    if not np.isfinite(losses).all() or not all(
        math.isfinite(report[k]) for k in ("loss", "diversity_m", "foot_slide_proxy_mps")
    ):
        raise ValueError("质量指标包含非有限值")
    if not history:
        return {
            "accepted": True,
            "bad_checks": 0,
            "stop": False,
            "improvement_ci_low": 0.0,
            "guards_pass": True,
        }
    baseline = history[0]
    best = next(row for row in reversed(history) if row["accepted"])
    reference = np.asarray(best["per_motion_loss"], dtype=float)
    if reference.shape != losses.shape:
        raise ValueError("固定 cohort 大小改变")
    delta = reference - losses
    rng = np.random.default_rng(20260916)
    bootstrap = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
    ci_low = float(np.quantile(bootstrap, 0.025))
    guards = (
        report["diversity_m"] >= baseline["diversity_m"] * min_diversity_ratio
        and report["foot_slide_proxy_mps"]
        <= baseline["foot_slide_proxy_mps"] * max_slide_ratio + slide_tolerance
        and report["contact_fraction"] >= baseline["contact_fraction"] * 0.5
    )
    accepted = bool(
        guards and ci_low > 0 and delta.mean() >= abs(reference.mean()) * min_relative_improvement
    )
    bad = 0 if accepted else int(history[-1]["bad_checks"]) + 1
    return {
        "accepted": accepted,
        "bad_checks": bad,
        "stop": bad >= patience,
        "improvement_ci_low": ci_low,
        "guards_pass": bool(guards),
    }


class MotionMillionQualityMonitor(pl.Callback):
    def __init__(
        self,
        output_dir,
        dataset_cfg,
        collate_cfg,
        cohort_size=32,
        seeds=(20260916, 20260917, 20260918, 20260919),
        every_n_steps=5000,
        patience=3,
        min_relative_improvement=0.002,
        min_diversity_ratio=0.9,
        max_slide_ratio=1.1,
        slide_tolerance=0.005,
    ):
        super().__init__()
        if cohort_size < 2 or len(set(seeds)) < 2 or every_n_steps <= 0:
            raise ValueError("监控需要至少两个动作、两个不同 seed 和正检查间隔")
        self.output_dir = Path(output_dir)
        self.dataset_cfg = dataset_cfg
        self.collate_cfg = collate_cfg
        self.cohort_size = int(cohort_size)
        self.seeds = [int(seed) for seed in seeds]
        self.every_n_steps = int(every_n_steps)
        self.decision_cfg = dict(
            patience=patience,
            min_relative_improvement=min_relative_improvement,
            min_diversity_ratio=min_diversity_ratio,
            max_slide_ratio=max_slide_ratio,
            slide_tolerance=slide_tolerance,
        )
        self.history = []
        self.cohort_fingerprint = None
        self.samples = []

    def state_dict(self):
        return {"history": self.history, "cohort_fingerprint": self.cohort_fingerprint}

    def load_state_dict(self, state_dict):
        self.history = state_dict["history"]
        self.cohort_fingerprint = state_dict["cohort_fingerprint"]

    @staticmethod
    def _gather(value):
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, value)
            return gathered
        return [value]

    def on_train_start(self, trainer, pl_module):
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        if not isinstance(scheduler, ContinuationWarmupCosineLR) or not scheduler.restored:
            raise RuntimeError("固定续训监控必须在完整 checkpoint 恢复和阶段调度器之后启动")
        if trainer.global_step != scheduler.last_epoch:
            raise RuntimeError("恢复 global_step 与 scheduler.last_epoch 不一致")
        optimizer = trainer.optimizers[0]
        if not optimizer.state:
            raise RuntimeError("AdamW 动量为空，拒绝作为续训启动")
        first = next(iter(optimizer.state.values()))
        audit = {
            "global_step": trainer.global_step,
            "stage": scheduler.stage_signature,
            "lr": scheduler.get_last_lr(),
            "optimizer_state_count": len(optimizer.state),
            "adam_step": float(first["step"]),
            "exp_avg_head_sum": float(first["exp_avg"].flatten()[:32].sum()),
            "exp_avg_sq_head_sum": float(first["exp_avg_sq"].flatten()[:32].sum()),
        }
        if trainer.is_global_zero:
            print("[Continuation restored] " + json.dumps(audit), flush=True)
            atomic_artifact(self.output_dir / "resume_audit.json", audit)
        error = None
        try:
            cfg = OmegaConf.to_container(self.dataset_cfg, resolve=True)
            cfg.update(limit_size=None, random_crop=False, shard_cache_size=1)
            with isolated_rng(self.seeds[0], pl_module.device):
                dataset = instantiate(cfg)
                indices = sorted(
                    np.random.default_rng(self.seeds[0])
                    .choice(len(dataset), self.cohort_size, replace=False)
                    .tolist()
                )
                contract = {
                    "indices": indices,
                    "seeds": self.seeds,
                    "frames": dataset.motion_frames,
                    "dataset_cfg": cfg,
                    "collate_cfg": OmegaConf.to_container(self.collate_cfg, resolve=True),
                    "diffusion": OmegaConf.to_container(
                        pl_module.model_cfg.diffusion, resolve=True
                    ),
                    "decision": self.decision_cfg,
                    "every_n_steps": self.every_n_steps,
                    "manifests": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (dataset.motion_manifest_path, dataset.embedding_manifest_path)
                    },
                }
                fingerprint = hashlib.sha256(
                    json.dumps(contract, sort_keys=True).encode()
                ).hexdigest()
                if self.cohort_fingerprint not in (None, fingerprint):
                    raise ValueError("恢复质量监控的 cohort/生成配置/判定口径发生变化")
                self.cohort_fingerprint = fingerprint
                for ordinal, index in enumerate(indices):
                    if ordinal % trainer.world_size == trainer.global_rank:
                        with isolated_rng(self.seeds[0] + index, pl_module.device):
                            self.samples.append((ordinal, index, dataset[index]))
                if trainer.is_global_zero:
                    atomic_artifact(
                        self.output_dir / "cohort.json",
                        {
                            **contract,
                            "fingerprint": fingerprint,
                            "boundary": "固定验证损失与运动学代理监控，不是官方 FID/文本语义/动力学评测",
                        },
                    )
                del dataset
        except Exception as exc:
            error = repr(exc)
        errors = self._gather(error)
        if any(errors):
            raise RuntimeError(f"质量 cohort 初始化失败: {errors}")
        if not self.history or self.history[-1]["step"] != trainer.global_step:
            self._evaluate(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps == 0 and (
            not self.history or self.history[-1]["step"] != trainer.global_step
        ):
            self._evaluate(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if not self.history or self.history[-1]["step"] != trainer.global_step:
            self._evaluate(trainer, pl_module)

    def _evaluate(self, trainer, model):
        from gem.datamodule.mocap_trainX_testY import collate_fn
        from tools.eval.motionmillion_smpl_to_272 import compute_world_joints

        rows, error = [], None
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for ordinal, index, sample in self.samples:
                    generated, losses, proxies, artifacts = [], [], [], {}
                    for seed_index, seed in enumerate(self.seeds):
                        sample_seed = (seed + index * 1009) % (2**31 - 1)
                        with (
                            isolated_rng(sample_seed, model.device),
                            trainer.precision_plugin.forward_context(),
                        ):
                            batch = collate_fn(
                                [copy.deepcopy(sample)], mode="val", collate_cfg=self.collate_cfg
                            )
                            batch = model.transfer_batch_to_device(batch, model.device, 0)
                            loss_batch = copy.deepcopy(batch)
                            model.prepare_batch(loss_batch, "diffusion")
                            loss = model.train_step(loss_batch, ordinal, "diffusion")["loss"]
                            losses.append(float(loss.detach()))
                        with (
                            isolated_rng(sample_seed, model.device),
                            trainer.precision_plugin.forward_context(),
                        ):
                            output = model.validation_step(batch, ordinal)
                        length = int(sample["length"])
                        params = {
                            key: value[:length].float()
                            for key, value in output["pred_body_params_global"].items()
                        }
                        joints = compute_world_joints(params, endecoder=model.endecoder).numpy()
                        generated.append(joints - joints[:, [0]])
                        proxies.append(motion_proxies(joints))
                        for key, value in params.items():
                            artifacts[f"motion_{ordinal:03d}_seed_{seed_index}_{key}"] = (
                                value.detach().cpu().numpy()
                            )
                        del loss_batch, output, batch
                    diversity = np.mean(
                        [
                            np.linalg.norm(left - right, axis=-1).mean()
                            for left, right in combinations(generated, 2)
                        ]
                    )
                    rows.append(
                        {
                            "ordinal": ordinal,
                            "motion_id": sample["meta"]["motion_id"],
                            "caption": sample["caption"],
                            "length": int(sample["length"]),
                            "loss": float(np.mean(losses)),
                            "diversity_m": float(diversity),
                            "proxies": proxies,
                            "artifacts": artifacts,
                        }
                    )
        except Exception as exc:
            error = repr(exc)
        finally:
            model.train(was_training)
        gathered = self._gather({"rows": rows, "error": error})
        if any(item["error"] for item in gathered):
            raise RuntimeError(f"固定质量监控失败: {[item['error'] for item in gathered]}")
        rows = sorted(
            [row for item in gathered for row in item["rows"]], key=lambda row: row["ordinal"]
        )
        if [row["ordinal"] for row in rows] != list(range(self.cohort_size)):
            raise RuntimeError("DDP 固定验证样本遗漏或重复")
        proxies = [proxy for row in rows for proxy in row["proxies"]]
        contacts = sum(proxy["contact_count"] for proxy in proxies)
        if contacts == 0:
            raise RuntimeError("全部生成动作缺少可观测近地接触，拒绝把脚滑记为零")
        report = {
            "step": int(trainer.global_step),
            "loss": float(np.mean([row["loss"] for row in rows])),
            "per_motion_loss": [row["loss"] for row in rows],
            "diversity_m": float(np.mean([row["diversity_m"] for row in rows])),
            "foot_slide_proxy_mps": sum(proxy["slide_sum"] for proxy in proxies) / contacts,
            "contact_fraction": contacts / sum(proxy["contact_total"] for proxy in proxies),
            "acceleration_mps2": float(np.mean([proxy["acceleration_mps2"] for proxy in proxies])),
        }
        report.update(quality_decision(self.history, report, **self.decision_cfg))
        write_error = None
        if trainer.is_global_zero:
            try:
                packed = {key: value for row in rows for key, value in row["artifacts"].items()}
                packed["metadata_json"] = np.asarray(
                    json.dumps(
                        {
                            "step": report["step"],
                            "fps": 30,
                            "seeds": self.seeds,
                            "cohort_fingerprint": self.cohort_fingerprint,
                            "samples": [
                                {
                                    key: row[key]
                                    for key in ("ordinal", "motion_id", "caption", "length")
                                }
                                for row in rows
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
                if not self.history:
                    atomic_artifact(self.output_dir / "baseline.npz", packed, npz=True)
                atomic_artifact(self.output_dir / "latest.npz", packed, npz=True)
                atomic_artifact(self.output_dir / "history.json", self.history + [report])
                trainer.logger.log_metrics(
                    {
                        f"quality/{key}": value
                        for key, value in report.items()
                        if isinstance(value, (int, float, bool)) and key != "step"
                    },
                    step=trainer.global_step,
                )
                print(
                    "[Fixed quality] "
                    + json.dumps(
                        {key: value for key, value in report.items() if key != "per_motion_loss"}
                    ),
                    flush=True,
                )
            except Exception as exc:
                write_error = repr(exc)
        if any(self._gather(write_error)):
            raise RuntimeError("质量交付保存失败，旧文件仍可检查")
        self.history.append(report)
        trainer.should_stop = trainer.should_stop or report["stop"]
