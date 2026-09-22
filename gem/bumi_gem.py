"""BUMI 文本训练的机器人公共层。

负责 qpos30 编解码、接触监督、运动学验证以及机器人 checkpoint 表示检查。
继承独立文本训练基类，保持参数键和保存契约；不导入原 SMPL GEM 或音乐迁移器。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from gem.robots.bumi.feature_codec import BUMI_FEATURE_DIM, BUMI_REPRESENTATION_CONTRACT_VERSION
from gem.robots.bumi.metrics import compute_bumi_kinematic_metrics
from gem.text_training import TextTrainingBase


def reorder_mujoco_joints_to_gmt(
    qpos_mujoco: torch.Tensor,
    mujoco_joint_names: Sequence[str],
    gmt_joint_names: Sequence[str],
) -> torch.Tensor:
    """Explicit publication-boundary reorder; model-internal order never changes."""

    if qpos_mujoco.shape[-1] != 28:
        raise ValueError(f"BUMI qpos must end in 28 values, got {qpos_mujoco.shape}")
    source = tuple(map(str, mujoco_joint_names))
    target = tuple(map(str, gmt_joint_names))
    if len(source) != 21 or len(target) != 21:
        raise ValueError("Both MuJoCo and GMT joint orders must contain exactly 21 names")
    if len(set(source)) != 21 or len(set(target)) != 21 or set(source) != set(target):
        raise ValueError(
            "GMT reorder requires two duplicate-free joint orders with identical names"
        )
    source_index = {name: index for index, name in enumerate(source)}
    permutation = torch.tensor(
        [source_index[name] for name in target], dtype=torch.long, device=qpos_mujoco.device
    )
    return torch.cat(
        (qpos_mujoco[..., :7], qpos_mujoco[..., 7:].index_select(-1, permutation)), dim=-1
    )


class BumiGEM(TextTrainingBase):
    condition_type = "text"

    @staticmethod
    def _validate_representation_checkpoint(checkpoint: Mapping[str, Any]) -> None:
        actual = checkpoint.get("bumi_representation_contract_version")
        if actual != BUMI_REPRESENTATION_CONTRACT_VERSION:
            raise RuntimeError(
                "BUMI checkpoint representation mismatch: expected "
                f"{BUMI_REPRESENTATION_CONTRACT_VERSION!r}, got {actual!r}. "
                "旧 93D checkpoint 不能当作 qpos30 权重继续加载；请使用 qpos30 统计量"
                "重新训练，本分支不提供旧人体权重迁移入口。"
            )
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeError("native BUMI qpos30 checkpoint is missing state_dict")
        final_shapes = [
            tuple(value.shape)
            for key, value in state.items()
            if str(key).endswith("denoiser.final_layer.fc2.weight")
            and isinstance(value, torch.Tensor)
        ]
        contact_shapes = [
            tuple(value.shape)
            for key, value in state.items()
            if str(key).endswith("static_conf_head.fc2.weight") and isinstance(value, torch.Tensor)
        ]
        if len(final_shapes) != 1 or final_shapes[0][0] != BUMI_FEATURE_DIM:
            raise RuntimeError(
                f"native BUMI checkpoint must contain one qpos30 output head, got {final_shapes}"
            )
        if len(contact_shapes) != 1 or contact_shapes[0][0] != 2:
            raise RuntimeError(
                f"native BUMI checkpoint must contain one 2D contact head, got {contact_shapes}"
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["bumi_representation_contract_version"] = BUMI_REPRESENTATION_CONTRACT_VERSION

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self._validate_representation_checkpoint(checkpoint)

    def prepare_batch(self, batch: dict[str, Any], mode: str) -> None:
        if mode != "diffusion":
            raise ValueError(f"BumiGEM only supports diffusion, got mode={mode!r}")
        encoded = self.endecoder.encode_with_aux(batch)
        valid = batch["mask"]["valid"].bool()
        if tuple(valid.shape) != tuple(encoded.normalized_features.shape[:-1]):
            raise ValueError(
                f"BUMI valid mask {valid.shape} does not match motion "
                f"{encoded.normalized_features.shape}"
            )
        batch["target_x"] = encoded.normalized_features
        batch["target_x_mask"] = valid[..., None].expand_as(encoded.normalized_features)
        batch["target_physical_features"] = encoded.physical_features
        batch["target_qpos_canonical"] = encoded.canonical_qpos
        batch["target_body_link_pos_root"] = encoded.target_body_link_pos_root
        batch["target_foot_contact"] = encoded.target_foot_contact
        batch["target_foot_contact_mask"] = encoded.target_foot_contact_mask
        batch["target_contact_ground_height"] = encoded.target_contact_ground_height
        batch["canonical_anchor"] = encoded.anchor_metadata
        batch["sample_indices_dict"] = self.endecoder.obs_indices_dict
        batch["device"] = encoded.normalized_features.device
        batch["B"], batch["L"] = encoded.normalized_features.shape[:2]
        self.attach_text_condition(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation(batch, "default", batch_idx, dataloader_idx)

    def validation(self, batch, test_mode, batch_idx, dataloader_idx=0):
        del batch_idx
        self.prepare_batch(batch, "diffusion")
        batch["target_x"] = torch.zeros_like(batch["target_x"])
        batch = self.create_condition_mask(batch, cond_mask_cfg=None, mode=None, train=False)
        outputs = self.pipeline.forward(batch, train=False, test_mode=test_mode)
        outputs["target_qpos_canonical"] = batch["target_qpos_canonical"]
        outputs["target_body_link_pos_root"] = batch["target_body_link_pos_root"]
        metrics = compute_bumi_kinematic_metrics(
            outputs["pred_qpos_canonical"],
            self.endecoder.kinematics,
            target_qpos=batch["target_qpos_canonical"],
            valid_mask=batch["mask"]["valid"],
            target_contact=batch["target_foot_contact"],
            pred_contact_logits=outputs.get("pred_foot_contact_logits"),
            music_beats=batch.get("music_beats"),
            fps=30,
            ground_height=(
                batch["target_contact_ground_height"].to(outputs["pred_qpos_canonical"])
                - self.endecoder.kinematics.default_qpos[2].to(outputs["pred_qpos_canonical"])
            ),
        )
        report_names = (
            "joint_angle_mae_rad",
            "root_trajectory_error_m",
            "fk_body_position_error_m",
            "joint_limit_violation_rate",
            "minimum_joint_margin_rad",
            "foot_penetration_mean_m",
            "foot_penetration_max_m",
            "foot_sliding_mean_mps",
            "foot_sliding_p95_mps",
            "foot_sliding_max_mps",
            "root_height_min_m",
            "root_tilt_mean_rad",
            "root_tilt_max_rad",
            "joint_velocity_p95_radps",
            "joint_velocity_max_radps",
            "joint_acceleration_p95_radps2",
            "joint_acceleration_max_radps2",
            "joint_jerk_p95_radps3",
            "joint_jerk_max_radps3",
            "root_linear_velocity_p95_mps",
            "root_linear_velocity_max_mps",
            "root_angular_velocity_p95_radps",
            "root_angular_velocity_max_radps",
            "beat_alignment_mean_distance_s",
            "beat_alignment_score",
        )
        dataset_name = str(batch["meta"][0].get("dataset_id", f"loader{dataloader_idx}"))
        for name in report_names:
            if name in metrics:
                self.log(
                    f"val/{dataset_name}/{name}",
                    metrics[name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=int(batch["B"]),
                    add_dataloader_idx=False,
                )
        outputs["kinematic_metrics"] = metrics
        return outputs
