# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI 文本模型独立的 Lightning 公共训练层。

从原 GEM 基类提取优化器、文本条件、训练日志、序列身份与 checkpoint 恢复逻辑。
保持 pipeline/endecoder 属性名和参数键不变，不构建 SMPL、视觉或音乐条件模块；
机器人编解码、损失和验证仍由 BumiGEM 与 BumiTextGEM 实现。
"""

from __future__ import annotations

import os

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from transformers import T5EncoderModel, T5Tokenizer

from gem.runtime.text_condition import prepare_text_attention_mask
from gem.utils.pylogger import Log
from gem.utils.tools import Timer


class TextTrainingBase(pl.LightningModule):
    """纯文本扩散的优化、日志和 checkpoint 公共层，保持已有参数路径。"""

    def __init__(
        self, pipeline, optimizer=None, scheduler=None, model_cfg=None, ignored_weights_prefix=None
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.endecoder = self.pipeline.endecoder
        self.optimizer = instantiate(optimizer, _partial_=True)
        self.model_cfg = model_cfg
        from gem.utils.sequence_contract import normalize_sequence_contract

        self.sequence_contract = normalize_sequence_contract(model_cfg.get("sequence_contract"))
        self.scheduler = scheduler
        self.train_modes = list(model_cfg.get("train_modes", ["diffusion"]))
        self.motion_backend = str(model_cfg.get("motion_backend", "bumi"))
        if (
            self.motion_backend != "bumi"
            or self.train_modes != ["diffusion"]
            or self.pipeline.args.in_attr
        ):
            raise ValueError("本分支只支持无逐帧条件的 BUMI 文本扩散训练")
        self.body_model_type = "bumi"
        self.body_model = None
        self.ignored_weights_prefix = (
            ignored_weights_prefix
            if ignored_weights_prefix is not None
            else [
                "pipeline.endecoder",
                "pipeline.denoiser3d.endecoder",
                "endecoder",
                "body_model",
                "feature_extractor",
            ]
        )
        self.test_step = self.predict_step = self.validation_step
        self.timing = os.environ.get("DEBUG_TIMING", "FALSE") == "TRUE"
        text_cfg = model_cfg.get("text_encoder")
        if text_cfg is None:
            raise ValueError("文本训练必须配置文本编码器")
        self.max_text_len = int(text_cfg.max_text_len)
        self.use_text_encoder = self.text_condition_enabled = True
        self.denoiser_uses_text = bool(self.pipeline.denoiser3d.denoiser.encode_text)
        self.text_encoder = self.tokenizer = None
        if text_cfg.get("load_llm", False):
            encoder, self.tokenizer = self.load_and_freeze_llm(text_cfg.llm_version)
            self.text_encoder = [encoder.cuda()]
        self.latent_dim = self.pipeline.args_denoiser3d.get("latent_dim", 512)
        self.normalizer_stats = {}
        for key, path in model_cfg.get("norm_attr_stats", {}).items():
            self.normalizer_stats[key] = torch.load(path, map_location="cpu", weights_only=False)

    def load_and_freeze_llm(self, llm_version):
        tokenizer = T5Tokenizer.from_pretrained(llm_version)
        model = T5EncoderModel.from_pretrained(llm_version)
        # Freeze llm weights
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, tokenizer

    def encode_text(self, raw_text, has_text=None, *, return_attention_mask=False):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        if self.tokenizer is None or self.text_encoder is None:
            batch_size = len(raw_text)
            max_text_len = getattr(self, "max_text_len", 16)
            text_dim = 1024
            denoiser = getattr(getattr(self.pipeline, "denoiser3d", None), "denoiser", None)
            if denoiser is not None and hasattr(denoiser, "encoded_text_dim"):
                text_dim = denoiser.encoded_text_dim
            encoded_text = torch.zeros(
                (batch_size, max_text_len, text_dim),
                device=device,
                dtype=torch.float32,
            )
            if has_text is not None:
                no_text = ~has_text.to(device)
                encoded_text[no_text] = 0
            attention_mask = torch.zeros(
                (batch_size, max_text_len), device=device, dtype=torch.bool
            )
            if has_text is not None:
                attention_mask[~no_text, 0] = True
                attention_mask[no_text, 0] = True
            else:
                attention_mask[:, 0] = True
            if return_attention_mask:
                return encoded_text, attention_mask
            return encoded_text
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                max_text_len = self.max_text_len

                encoded = self.tokenizer(
                    raw_text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_text_len,
                    truncation=True,
                )
                # We expect all the processing is done in GPU.
                input_ids = encoded.input_ids.to(device)
                attn_mask = encoded.attention_mask.to(device)

                with torch.no_grad():
                    output = self.text_encoder[0](input_ids=input_ids, attention_mask=attn_mask)
                    encoded_text = output.last_hidden_state.detach()

                encoded_text = encoded_text[:, :max_text_len]
                attn_mask = attn_mask[:, :max_text_len]
                encoded_text *= attn_mask.unsqueeze(-1)
                # for bnum in range(encoded_text.shape[0]):
                #     nvalid_elem = attn_mask[bnum].sum().item()
                #     encoded_text[bnum][nvalid_elem:] = 0
        if has_text is not None:
            no_text = ~has_text.to(encoded_text.device)
            encoded_text[no_text] = 0
            # 无文本兼容样本保留一个全零有效 token，避免全 padding 注意力。
            attn_mask[no_text] = 0
            attn_mask[no_text, 0] = 1
        attention_mask = attn_mask.bool()
        if return_attention_mask:
            return encoded_text, attention_mask
        return encoded_text

    def attach_text_condition(self, target_batch, source_batch=None):
        """把预计算或在线 T5 条件及其 padding mask 放到目标 batch。"""
        source_batch = target_batch if source_batch is None else source_batch
        if not self.text_condition_enabled:
            return
        device = target_batch.get("device")
        if device is None:
            device = target_batch["target_x"].device
        if "text_embed" in source_batch:
            encoded_text = source_batch["text_embed"].to(device=device, dtype=torch.float32)
            attention_mask = prepare_text_attention_mask(
                encoded_text,
                source_batch.get("text_attention_mask"),
                has_text=source_batch.get("has_text"),
            )
        else:
            encoded_text, attention_mask = self.encode_text(
                source_batch["caption"],
                source_batch.get("has_text"),
                return_attention_mask=True,
            )
        target_batch["encoded_text"] = encoded_text
        target_batch["text_attention_mask"] = attention_mask

    def training_step(self, batch, batch_idx):
        def append_mode_to_loss(outputs, mode, suffix=""):
            if suffix != "":
                suffix = f"_{suffix}"
            for k in list(outputs.keys()):
                if "_loss" in k or k in {"loss"}:
                    outputs[f"Loss_{mode}{suffix}/{k}"] = outputs.pop(k)
                elif k.endswith("_metric"):
                    outputs[f"Metric_{mode}{suffix}/{k}"] = outputs.pop(k)
            return outputs

        outputs = {"loss": 0}

        with Timer("train_step", enabled=self.timing):
            for mode in self.train_modes:
                self.prepare_batch(batch, mode)  # set "obs" (2d keypoints)
                outputs_mode = self.train_step(batch, batch_idx, mode=mode)
                outputs["loss"] += outputs_mode["loss"]
                append_mode_to_loss(outputs_mode, mode)
                outputs.update(outputs_mode)
                if mode == "regression" and "diffusion" in self.train_modes:
                    batch["regression_outputs"] = outputs_mode.copy()
                # batch[f"{mode}_condition"] = outputs_mode[f"{mode}_condition"]

        # Log
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "sync_dist": True,
            "batch_size": outputs["batch_size"],
        }
        self.log("train/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"{k}", v, **log_kwargs)
            elif "_metric" in k:
                reduce_fx = "sum" if "_count_metric" in k else "mean"
                self.log(
                    f"{k}",
                    v,
                    # 文本 CFG 的实际丢弃比例需要逐 step 可见，其余历史指标
                    # 继续保持只按 epoch 聚合，避免改变既有实验的日志行为。
                    on_step=k.endswith("text_cfg_dropout_metric"),
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=outputs["batch_size"],
                    reduce_fx=reduce_fx,
                )

        return outputs

    def on_before_optimizer_step(self, optimizer):
        """为纯文本实验记录裁剪前的全模型 L2 梯度范数。"""
        del optimizer
        in_attr = self.pipeline.args.get("in_attr", None)
        encode_text = bool(self.pipeline.denoiser3d.denoiser.encode_text)
        if in_attr is None or len(in_attr) != 0 or not encode_text:
            return
        parameter_norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        if parameter_norms:
            total_norm = torch.stack(parameter_norms).norm(2)
            self.log(
                "train/gradient_norm_2",
                total_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

    def train_step(self, batch, batch_idx, mode):
        batch = batch.copy()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.detach().clone()

        cond_mask_cfg = self.model_cfg.get("condition_mask", {})
        batch = self.create_condition_mask(batch, cond_mask_cfg, mode, train=True)

        # Forward and get loss
        outputs = self.pipeline.forward(
            batch,
            train=True,
            global_step=self.trainer.global_step,
            mode=mode,
            normalizer_stats=self.normalizer_stats,
        )
        if "text_cfg_dropout_mask" in batch:
            outputs["text_cfg_dropout_metric"] = batch["text_cfg_dropout_mask"].float().mean()
        outputs["batch_size"] = batch["B"]
        return outputs

    def configure_optimizers(self):
        params = []
        for _, v in self.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler is None or self.scheduler["scheduler"] is None:
            return optimizer

        scheduler = dict(self.scheduler)
        scheduler["scheduler"] = instantiate(scheduler["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler]

    def load_state_dict(self, state_dict, strict=True):
        """Filter intentionally dropped prefixes when loading checkpoints."""
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not any(k.startswith(prefix) for prefix in self.ignored_weights_prefix)
        }
        incompatible = super().load_state_dict(filtered_state_dict, strict=False)

        real_missing = [
            k
            for k in incompatible.missing_keys
            if not any(k.startswith(prefix) for prefix in self.ignored_weights_prefix)
        ]
        real_unexpected = [
            k
            for k in incompatible.unexpected_keys
            if not any(k.startswith(prefix) for prefix in self.ignored_weights_prefix)
        ]

        if real_missing:
            Log.warn(f"Missing keys: {real_missing}")
        if real_unexpected:
            Log.warn(f"Unexpected keys: {real_unexpected}")
        if strict and (real_missing or real_unexpected):
            raise RuntimeError(
                f"Error(s) in loading state_dict: missing={real_missing}, unexpected={real_unexpected}"
            )
        return incompatible

    def on_load_checkpoint(self, checkpoint) -> None:
        from gem.utils.sequence_contract import validate_resume_contract

        validate_resume_contract(
            checkpoint.get("genmo_sequence_contract"), getattr(self, "sequence_contract", None)
        )
        self._resume_data_identity = checkpoint.get("genmo_data_identity")

    def _sequence_data_identity(self):
        trainer = getattr(self, "_trainer", None)
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None
        return {
            split: [
                getattr(ds, "data_identity", None) for ds in getattr(datamodule, f"{split}sets", [])
            ]
            for split in ("train", "val")
        }

    def on_fit_start(self) -> None:
        saved = getattr(self, "_resume_data_identity", None)
        if saved is not None and saved != self._sequence_data_identity():
            raise ValueError("完整 resume 的 MotionMillion 数据身份与当前release不一致")

    def on_save_checkpoint(self, checkpoint) -> None:
        if getattr(self, "sequence_contract", None) is not None:
            checkpoint["genmo_sequence_contract"] = dict(self.sequence_contract)
            checkpoint["genmo_data_identity"] = self._sequence_data_identity()
        # 新 checkpoint 显式携带文本长度契约；旧 checkpoint 没有该字段时，
        # 推理端按历史默认 50 token 兼容解析。
        if self.text_condition_enabled:
            diffusion_model = self.pipeline.denoiser3d
            denoiser = getattr(diffusion_model, "denoiser", diffusion_model)
            checkpoint["genmo_text_contract"] = {
                "schema_version": 1,
                "max_text_len": int(self.max_text_len),
                "encoded_text_dim": int(getattr(denoiser, "encoded_text_dim", 1024)),
                "text_only": len(self.pipeline.args.in_attr) == 0,
                "pipeline_in_attr": list(self.pipeline.args.in_attr),
            }
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)
