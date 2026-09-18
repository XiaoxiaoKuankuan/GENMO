"""BUMI 完整序列文本生成训练入口。

机器人编解码、FK、优化和验证复用 BumiGEM；文本通过 GEM 的 T5 条件接口进入
共享 Transformer，逐帧条件置零，不伪造音乐。完整恢复额外核对机器人资产和损失。
本模块属于训练侧；无 checkpoint 部署由 gem.runtime.bumi_text_runtime 实现。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from gem.bumi_gem import BumiGEM
from gem.runtime.bumi_text_contract import (
    SCHEMA,
    REPRESENTATION,
    inspect_payload,
    sha256_file,
    validate_contract,
)
from gem.robots.bumi.postprocess import lock_bumi_foot_contacts


class BumiTextGEM(BumiGEM):
    condition_type = "text"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.pipeline.args.in_attr
            or not self.text_condition_enabled
            or not self.denoiser_uses_text
        ):
            raise ValueError("BUMI 文本要求空逐帧条件和启用 T5/cross-attention")
        if self.max_text_len != 150 or self.pipeline.denoiser3d.denoiser.encoded_text_dim != 1024:
            raise ValueError("BUMI 文本需要150-token/1024D")
        if self.sequence_contract is None or self.endecoder.sequence_mode != "full":
            raise ValueError("BUMI 文本需要完整序列契约")
        stats = json.loads(Path(self.endecoder.stats_path).read_text())
        if stats.get("data_kind") != "bumi_text_fullseq" or stats.get("split") != "train":
            raise ValueError("必须使用 BUMI 文本 train 有效元素统计量")
        self.stats_identity = stats

    def on_fit_start(self):
        super().on_fit_start()
        datasets = self.trainer.datamodule.trainsets
        identities = [dataset.data_identity for dataset in datasets]
        if any(
            identity["manifest_sha256"] != self.stats_identity["data_identity"]["manifest_sha256"]
            or identity["kinematics_sha256"] != self.endecoder.kinematics.kinematics_sha256
            for identity in identities
        ):
            raise ValueError("stats/训练release/运动学身份不一致")
        selection = self.stats_identity.get("dataset")
        expected = {"motionmillion", "humanml3d"} if selection is None else {selection}
        if {identity["dataset"] for identity in identities} != expected:
            raise ValueError("联合和单集实验必须采用对应train统计量")

    def create_condition_mask(
        self, batch, cond_mask_cfg=None, mode=None, train=False, first_k_frames=None
    ):
        if first_k_frames is not None and first_k_frames != batch["L"]:
            raise ValueError("完整文本动作禁止 first_k_frames 裁剪")
        zeros = batch["target_x"].new_zeros(batch["B"], batch["L"], self.latent_dim)
        batch.update(f_cond=zeros, f_uncond=zeros.clone(), f_empty=zeros.clone())
        batch["motion"] = batch["target_x"] * batch["mask"]["valid"][..., None]
        return batch

    def training_contract(self):
        end = self.endecoder
        kin = end.kinematics
        return validate_contract(
            dict(
                schema=SCHEMA,
                motion_backend="bumi",
                condition="text",
                representation=REPRESENTATION,
                feature_dim=30,
                contact_dim=2,
                qpos_dim=28,
                fps=30,
                quaternion_convention="wxyz",
                max_text_len=150,
                encoded_text_dim=1024,
                joint_names=list(kin.joint_order),
                source_mjcf_sha256=kin.source_mjcf_sha256,
                sequence=dict(self.sequence_contract),
                loss_reduction="valid_per_sample",
                loss_contract=self.pipeline.args.loss_contract,
                loss_config=OmegaConf.to_container(self.pipeline.args, resolve=True),
                assets={
                    "kinematics": {
                        "path": str(kin.kinematics_path),
                        "sha256": kin.kinematics_sha256,
                    },
                    "stats": {"path": end.stats_path, "sha256": sha256_file(end.stats_path)},
                },
            )
        )

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["bumi_text_contract"] = self.training_contract()
        checkpoint["bumi_text_denoiser_config"] = OmegaConf.to_container(
            self.pipeline.args_denoiser3d.model_cfg.denoiser, resolve=True
        )
        checkpoint["bumi_text_diffusion_config"] = OmegaConf.to_container(
            self.pipeline.args_denoiser3d.model_cfg.diffusion, resolve=True
        )

    def on_load_checkpoint(self, checkpoint):
        saved = inspect_payload(checkpoint)
        current = self.training_contract()

        # 资产可以搬家，完整恢复只允许相同内容；其余训练参数必须精确一致。
        def identity(value):
            value = copy.deepcopy(value)
            for asset in value["assets"].values():
                asset.pop("path", None)
            value["loss_config"].pop("endecoder_opt", None)
            return value

        if identity(saved) != identity(current):
            raise ValueError("禁止跨 BUMI 文本训练/资产/损失契约完整 resume")
        super().on_load_checkpoint(checkpoint)

    def load_pretrained_model(self, ckpt_path):
        raise ValueError('首版BUMI文本不提供weights-only初始化；从零训练或使用同契约resume_mode完整恢复')

    @torch.no_grad()
    def predict(self, data, static_cam=False, postproc=False):
        from gem.utils.sequence_contract import validate_generation_length

        frames = int(data["num_frames"])
        validate_generation_length(self.sequence_contract, frames)
        device = next(self.parameters()).device
        valid = torch.ones(1, frames, dtype=torch.bool, device=device)
        batch = dict(
            B=1,
            L=frames,
            device=device,
            length=torch.tensor([frames], device=device),
            target_x=torch.zeros(1, frames, 30, device=device),
            mask={"valid": valid},
            sample_indices_dict=self.endecoder.obs_indices_dict,
            caption=[data.get("caption", "")],
            has_text=torch.ones(1, dtype=torch.bool, device=device),
            world_anchor={"root_xy": [0.0, 0.0], "yaw": 0.0},
        )
        if "text_embed" in data:
            batch["text_embed"] = data["text_embed"].reshape(1, 150, 1024)
            batch["text_attention_mask"] = data["text_attention_mask"].reshape(1, 150)
        self.attach_text_condition(batch)
        outputs = self.pipeline(self.create_condition_mask(batch), train=False)
        raw = outputs["pred_qpos"][0, :frames]
        contact = outputs["pred_foot_contact_logits"][0, :frames]
        qpos = (
            lock_bumi_foot_contacts(
                raw, contact, self.endecoder.kinematics, contact_is_logits=True
            ).qpos
            if postproc
            else raw
        )
        return dict(qpos=qpos, qpos_raw=raw, pred_foot_contact_logits=contact, fps=30)
