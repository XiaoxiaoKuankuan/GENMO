"""与 Stage 1 Actor 分离的小规模监督训练、验证和数据装配。

本模块直接复用第 3 步 ``BumiClosedLoopStage1Dataset`` 和 collate/validator，显式加载
四来源 YAML 后仅在内存覆盖 H/P 配置。训练采样对每个来源给定总概率，再在该来源已由
Dataset 展开的 duration-aware 索引内采样；不改变 split、provenance、关节顺序或坐标 anchor。
音乐 MLP、存在标记投影和 ``NetworkEncoderRoPE`` 保持旧 music-only 的结构与参数名字，
Actor 自身不依赖训练器，未来 Stage 2 可直接复用。

入口默认只运行有步数上限的小规模验证/训练；这里不构造 GMT、Critic、DPPO，也不修改
30→50 Hz 速度派生链。proprio 使用配置明确指定的物理尺度，而非数据估计统计或 GMT 的
69 维 normalizer。新建 AdamW 始终从第 0 步开始，旧 checkpoint 仅作权重初始化。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from gem.closedloop.checkpoint import load_stage1_checkpoint, warm_start_weights
from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.closedloop.stage1_dataset import (
    BumiClosedLoopStage1Dataset,
    collate_stage1_training_samples,
)
from gem.robots.bumi.endecoder import BumiEndecoder
from gem.robots.bumi.kinematics import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def load_stage1_config(path: str | Path, overrides: Iterable[str] = ()) -> DictConfig:
    """读取独立 Stage 1 配置及显式 CLI 覆盖，避免触碰旧 Hydra 训练入口。"""

    path = repository_path(path)
    config = OmegaConf.load(path)
    if config.get("base_config"):
        base = OmegaConf.load(path.parent / str(config.pop("base_config")))
        config = OmegaConf.merge(base, config)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    if config.get("config_version") != "genmo.bumi_closedloop.stage1_run.v1":
        raise ValueError("Stage 1 run config version mismatch")
    return config


def load_stage1_data_config(config: Mapping[str, Any]) -> DictConfig:
    """读取已有四库配置；H/P 只作用于当前运行的内存副本。"""

    result = OmegaConf.load(repository_path(config["dataset_config"]))
    if config.get("sample_contract"):
        result = OmegaConf.merge(result, {"sample_contract": config["sample_contract"]})
    return result


def build_stage1_actor(config: Mapping[str, Any], data_config: DictConfig) -> nn.Module:
    """仅实例化旧主干/音乐模块与新 Actor，不构造完整旧 GEM/Lightning。"""

    from timm.models.vision_transformer import Mlp

    from gem.closedloop.actor import Stage1Actor
    from gem.network.gem_denoiser import NetworkEncoderRoPE

    options = dict(config["model"])
    latent_dim = int(options["latent_dim"])
    dropout = float(options.get("dropout", 0.1))
    endecoder_options = OmegaConf.to_container(OmegaConf.create(config["endecoder"]), resolve=True)
    if not endecoder_options.get("stats_path"):
        endecoder_options["stats_path"] = str(data_config.qpos30_stats.path)
    if not endecoder_options.get("kinematics_path"):
        endecoder_options["kinematics_path"] = str(data_config.dataset_defaults.kinematics_path)
    endecoder = BumiEndecoder(**endecoder_options)
    for split, entries in data_config.datasets.items():
        for name, entry in entries.items():
            asset_path = repository_path(str(entry.kinematics_path))
            if sha256_file(asset_path) != endecoder.kinematics.kinematics_sha256:
                raise ValueError(
                    f"Stage 1 actor/dataset kinematics mismatch at {split}/{name}: {asset_path}"
                )
    denoiser = NetworkEncoderRoPE(
        output_dim=30,
        xt_dim=30,
        max_len=120,
        njoints=30,
        pred_cam_dim=0,
        static_conf_dim=2,
        avgbeta=False,
        encode_text=False,
        use_text_pos_enc=False,
        input_remove_global=False,
        input_remove_condition=False,
        latent_dim=latent_dim,
        num_layers=int(options["num_layers"]),
        num_heads=int(options["num_heads"]),
        mlp_ratio=float(options.get("mlp_ratio", 4.0)),
        dropout=dropout,
        args={"motion_backend": "bumi"},
    )
    music_embedder = Mlp(35, hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout)
    exists = None
    if bool(options.get("use_cond_exists_as_input", True)):
        exists = nn.Sequential(
            nn.Linear(latent_dim + 1, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        nn.init.zeros_(exists[-1].weight)
        nn.init.zeros_(exists[-1].bias)
    return Stage1Actor(
        endecoder=endecoder,
        denoiser=denoiser,
        music_embedder=music_embedder,
        cond_exists_embedder=exists,
        history_steps=int(data_config.sample_contract.history_steps),
        history_hidden_dim=int(options.get("history_hidden_dim", 128)),
        proprio_scales=tuple(float(value) for value in options.get("proprio_scales", (1, 1, 1, 1))),
        music_mask_prob=float(options.get("music_mask_prob", 0.1)),
        noise_schedule=str(options.get("noise_schedule", "cosine")),
        diffusion_steps=int(options.get("diffusion_steps", 1000)),
    )


def build_stage1_losses(actor: nn.Module, config: Mapping[str, Any]) -> nn.Module:
    """以既有 v5 权重为基础构造独立逐坐标 mask 损失适配器。"""

    from gem.closedloop.losses import Stage1BumiLosses

    loss_config = config["loss"]
    legacy = OmegaConf.load(repository_path(loss_config["weights_from"]))
    weights = dict(legacy.args.weights)
    weights.update(dict(loss_config.get("weights", {})))
    return Stage1BumiLosses(actor.endecoder, weights, **dict(loss_config.get("options", {})))


def build_stage1_loader(
    data_config: DictConfig,
    *,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    samples_per_epoch: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    resumable: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """直接实例化第 3 步 Dataset，训练保留四来源相对采样权重。"""

    if split not in data_config.datasets:
        raise ValueError(f"Stage 1 dataset config has no split {split!r}")
    datasets = []
    for entry in data_config.datasets[split].values():
        options = OmegaConf.to_container(entry, resolve=True)
        if (
            options.pop("_target_", None)
            != "gem.closedloop.stage1_dataset.BumiClosedLoopStage1Dataset"
        ):
            raise ValueError("Stage 1 only accepts the existing closed-loop Dataset")
        dataset = BumiClosedLoopStage1Dataset(**options)
        ground = dataset.reader.dataset_info.get("ground_semantics")
        if ground not in {
            "legacy_body_origin_min_zero",
            "gmr_foot_sole_ground_zero_v1",
            "robot_retargeter_floor_zero_v1",
            "umr_foot_sole_ground_zero_v1",
            "mixed_floor_zero_fk_contact_v2",
        }:
            raise ValueError(
                f"Stage 1 FK/contact losses require supported ground provenance; "
                f"got {ground!r} for {dataset.dataset_name}. "
                "Legacy data must carry full-sequence ground supervision in existing meta."
            )
        if len(dataset) == 0:
            raise ValueError(f"empty Stage 1 dataset: {dataset.dataset_name}/{split}")
        datasets.append(dataset)
    if not datasets:
        raise ValueError(f"empty Stage 1 split: {split}")
    merged = ConcatDataset(datasets)
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if split == "train":
        reference = data_config.get("train_sampling_reference", {})
        sample_weights = []
        for dataset in datasets:
            source_weight = float(reference.get(dataset.dataset_name, 1.0))
            if not math.isfinite(source_weight) or source_weight <= 0.0:
                raise ValueError("dataset sampling weights must be finite and positive")
            sample_weights.extend([source_weight / len(dataset)] * len(dataset))
        draws = len(merged) if samples_per_epoch is None else int(samples_per_epoch)
        if draws <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if resumable:
            from gem.closedloop.sampling import (
                DeterministicDrawDataset,
                ResumableDistributedWeightedSampler,
            )

            sampler = ResumableDistributedWeightedSampler(
                sample_weights,
                draws,
                seed,
                rank=rank,
                world_size=world_size,
                emit_draw_keys=True,
            )
            merged = DeterministicDrawDataset(merged, seed=seed)
        else:
            if world_size != 1 or rank != 0:
                raise ValueError("Distributed training requires resumable sharded sampling")
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=draws, replacement=True, generator=generator
            )
    return DataLoader(
        merged,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_stage1_training_samples,
        generator=generator,
        drop_last=False,
        pin_memory=pin_memory,
    )


def batch_to_device(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    """仅移动张量，保留已有 provenance/meta 的结构与内容。"""

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _loss_values(loss: torch.Tensor, values: Mapping[str, Any]) -> dict[str, float]:
    result = {"loss": float(loss.detach())}
    for name, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[name] = float(value.detach())
        elif isinstance(value, (float, int)):
            result[name] = float(value)
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Stage 1 loss report contains NaN or Inf")
    return result


def initialize_stage1_weights(actor: nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    """旧权重 warm start、新接口权重验证加载、从头初始化三者明确区分。"""

    legacy = config.get("warm_start_checkpoint")
    stage1 = config.get("stage1_checkpoint")
    if legacy and stage1:
        raise ValueError("choose either warm_start_checkpoint or stage1_checkpoint")
    if legacy:
        assets = config.get("warm_start_source_assets")
        return warm_start_weights(
            actor,
            repository_path(legacy),
            source_assets=assets,
            source_model_config=config.get("warm_start_source_model_config"),
        )
    if stage1:
        return load_stage1_checkpoint(actor, repository_path(stage1))
    return {
        "mode": "random_initialization",
        "global_step_restored": False,
        "optimizer_restored": False,
    }


def train_stage1_steps(
    actor: nn.Module,
    losses: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    max_steps: int,
    learning_rate: float = 1.0e-5,
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
) -> dict[str, Any]:
    """执行明确上限的监督更新；每次调用创建新 optimizer，step 恒从零开始。"""

    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    device = next(actor.parameters()).device
    actor.train()
    losses.train()
    optimizer = torch.optim.AdamW(
        (value for value in actor.parameters() if value.requires_grad),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    reports = []
    iterator = iter(loader)
    for step in range(int(max_steps)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("Stage 1 training loader is empty") from exc
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = actor.training_forward(batch)
        loss, values = losses(
            batch, prediction["pred_x_start"], prediction["static_conf_logits"], global_step=step
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Stage 1 training loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), float(grad_clip_norm), error_if_nonfinite=True
        )
        optimizer.step()
        reports.append(
            {"step": step, "gradient_norm": float(gradient_norm), **_loss_values(loss, values)}
        )
    return {"optimizer_start_step": 0, "completed_steps": int(max_steps), "steps": reports}


@torch.no_grad()
def validate_stage1_batches(
    actor: nn.Module,
    losses: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    max_batches: int = 2,
    sample_steps: int = 10,
    guidance_scale: float = 1.0,
) -> dict[str, Any]:
    """条件采样后才使用标签计算验证损失，监督字段不传入 sample。"""

    if int(max_batches) <= 0:
        raise ValueError("max_batches must be positive")
    actor.eval()
    losses.eval()
    device = next(actor.parameters()).device
    reports = []
    for index, source in enumerate(loader):
        if index >= int(max_batches):
            break
        batch = batch_to_device(source, device)
        conditions = {key: batch[key] for key in STAGE1_CONDITION_KEYS}
        generated = actor.sample(
            conditions, steps=int(sample_steps), guidance_scale=float(guidance_scale)
        )
        loss, values = losses(
            batch, generated["normalized"], generated["contact_logits"], global_step=0
        )
        known = batch["known_qpos30_mask"]
        known_max_abs_error = (
            float((generated["qpos30"][known] - batch["known_qpos30"][known]).abs().max())
            if bool(known.any())
            else 0.0
        )
        reports.append(
            {
                "batch": index,
                "known_max_abs_error": known_max_abs_error,
                **_loss_values(loss, values),
            }
        )
    if not reports:
        raise ValueError("Stage 1 validation loader is empty")
    return {
        "completed_batches": len(reports),
        "batches": reports,
        "sampling_received_target_fields": False,
    }
