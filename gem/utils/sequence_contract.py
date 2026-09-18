"""MotionMillion 序列训练契约及配置检查。

把数据 release 身份与训练取样策略分开：旧 checkpoint 缺少此契约时仍是历史路径，
不能自动解释成完整动作模型。A0 的真实动作是 60—300 帧，300 只是 batch padding；
局部注意力范围、有效元素 loss reduction 一并进入 checkpoint。此模块不加载模型、
数据或 GPU，供训练入口、推理和 CPU 配置测试共同使用。
"""

from __future__ import annotations

from collections.abc import Mapping

from omegaconf import OmegaConf


def normalize_sequence_contract(value):
    """严格解析显式契约；None 专门表示没有新字段的旧模型。"""
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError("genmo_sequence_contract 必须是字典")
    required = {
        "schema_version",
        "sequence_mode",
        "min_frames",
        "max_frames",
        "pad_to_frames",
        "fps",
        "attention_mode",
        "attention_max_len",
        "loss_reduction",
        "train_caption_sampling",
        "val_caption_sampling",
    }
    if set(value) != required:
        raise ValueError(f"sequence contract 字段不匹配: {set(value) ^ required}")
    result = dict(value)
    for key in ("schema_version", "min_frames", "max_frames", "pad_to_frames", "attention_max_len"):
        if type(result[key]) is not int:
            raise ValueError(f"sequence contract {key} 必须为整数")
    if result["schema_version"] != 1 or result["sequence_mode"] not in {"full", "crop"}:
        raise ValueError("不支持的 sequence contract 版本或模式")
    if (result["min_frames"], result["max_frames"], result["fps"]) != (60, 300, 30):
        raise ValueError("MotionMillion 源动作契约必须为 60—300 帧、30 FPS")
    expected_pad = 300 if result["sequence_mode"] == "full" else 120
    if result["pad_to_frames"] != expected_pad:
        raise ValueError(f"{result['sequence_mode']} 模式 padding 必须为 {expected_pad}")
    if result["attention_mode"] not in {"legacy", "valid_length"}:
        raise ValueError("未知 attention_mode")
    if result["attention_max_len"] <= 0:
        raise ValueError("attention_max_len 必须为正数")
    if result["loss_reduction"] not in {"legacy", "valid_per_sample"}:
        raise ValueError("未知 loss_reduction")
    if result["sequence_mode"] == "full" and (
        result["attention_mode"] != "valid_length" or result["loss_reduction"] != "valid_per_sample"
    ):
        raise ValueError("full 模式必须使用有效长度注意力及 valid_per_sample reduction")
    if (result["train_caption_sampling"], result["val_caption_sampling"]) != ("random", "first"):
        raise ValueError("A0 caption 策略必须为 train=random / val=first")
    return result


def validate_resume_contract(saved, current):
    """完整恢复不能跨取样/监督契约；weights-only 不调用此检查。"""
    saved = normalize_sequence_contract(saved)
    current = normalize_sequence_contract(current)
    if saved != current:
        raise ValueError(
            "禁止跨序列训练契约完整 resume：旧 crop/full、attention 或 loss reduction 不一致。"
            "如需迁移权重，请用 pretrain_ckpt 做 weights-only 初始化新实验。"
        )


def validate_generation_length(contract, num_frames, fps=30):
    contract = normalize_sequence_contract(contract)
    if contract is not None and contract["sequence_mode"] == "full":
        if not contract["min_frames"] <= num_frames <= contract["max_frames"]:
            raise ValueError("fullseq checkpoint 的 num_frames 必须在 [60, 300]")
        if fps != contract["fps"]:
            raise ValueError("fullseq checkpoint 固定 30 FPS，不能改变动作时间尺度")


def validate_sequence_experiment(cfg):
    """在建立 DataModule/CUDA 前检查新实验，旧配置无契约则原样兼容。"""
    raw = OmegaConf.select(cfg, "model.model_cfg.sequence_contract")
    contract = normalize_sequence_contract(raw)
    if contract is None:
        return None
    is_bumi = OmegaConf.select(cfg, "model.model_cfg.motion_backend") == "bumi"
    denoiser = cfg.network.model_cfg.denoiser
    if (
        denoiser.max_len != contract["attention_max_len"]
        or denoiser.attention_mode != contract["attention_mode"]
    ):
        raise ValueError("denoiser 与序列 attention 契约不一致")
    if cfg.pipeline.args.loss_reduction != contract["loss_reduction"]:
        raise ValueError("pipeline loss reduction 与契约不一致")
    if cfg.data.dataset_opts.max_motion_frames != contract["pad_to_frames"]:
        raise ValueError("DataModule padding 与契约不一致")
    for split, configs in (("train", cfg.train_datasets), ("val", cfg.test_datasets)):
        if len(configs) != 1 and not is_bumi:
            raise ValueError("A0 仅支持单一 MotionMillion 数据集")
        if not configs:
            raise ValueError("序列实验需要非空 train/val 配置")
        for ds in configs.values():
            if (ds.sequence_mode, ds.pad_to_frames, ds.caption_sampling) != (
                contract["sequence_mode"], contract["pad_to_frames"], contract[f"{split}_caption_sampling"],
            ):
                raise ValueError(f"{split} Dataset 与 sequence contract 不一致")
            if contract["sequence_mode"] == "full" and ds.get("random_crop") is not None:
                raise ValueError("full 模式不接受 random_crop，使用独立 caption_sampling")
    if cfg.pl_trainer.use_distributed_sampler or not cfg.data.shard_aware_sampling.enabled:
        raise ValueError("必须保留 shard-aware sampler，禁止 Lightning 二次分片")
    if cfg.model.model_cfg.text_encoder.max_text_len != 150 or denoiser.encoded_text_dim != 1024:
        raise ValueError("A0 必须保持 T5 150-token / 1024D")
    if denoiser.output_dim != (30 if is_bumi else 151) or list(cfg.pipeline.args.in_attr):
        raise ValueError("A0 必须保持 text-only，并匹配后端运动维度（SMPL151/BUMI30）")
    if is_bumi:
        if (denoiser.xt_dim, denoiser.static_conf_dim, denoiser.pred_cam_dim) != (30, 2, 0):
            raise ValueError("BUMI 必须30D运动、2D接触、无相机头")
        if not denoiser.encode_text or denoiser.text_mask_prob != 0.1 or cfg.endecoder.sequence_mode != "full":
            raise ValueError("BUMI 必须完整动作、文本交叉注意力及单处0.1 CFG dropout")
        if cfg.pretrain_ckpt is not None or cfg.ckpt_path is not None:
            raise ValueError("首版 BUMI 文本只允许从零训练或同契约 resume")
        if not all(0 <= int(cfg.training_budget[k]) < int(cfg.training_budget.max_steps)
                   for k in ("warmup_steps", "auxiliary_warmup_steps")):
            raise ValueError("短程预算必须同时调整学习率和机器人辅助项warmup")
    if cfg.pipeline.args.get("physics_losses", {}).get("enabled", False):
        raise ValueError("A0 不增加额外 physics_losses")
    total = int(cfg.pl_trainer.max_steps)
    schedule = cfg.scheduler.scheduler
    if (
        total <= 0
        or int(schedule.total_steps) != total
        or not 0 <= int(schedule.warmup_steps) < total
    ):
        raise ValueError("max_steps / scheduler.total_steps / warmup_steps 不一致")
    if int(cfg.pl_trainer.max_epochs) != -1:
        raise ValueError("A0 必须 max_epochs=-1，由 max_steps 控制预算")
    if cfg.resume_mode is not None and (cfg.pretrain_ckpt is not None or cfg.ckpt_path is not None):
        raise ValueError("完整 resume 与 weights-only 初始化不能同时设置")
    return contract
