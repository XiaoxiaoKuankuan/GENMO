"""Lightning module for the independent BUMI-native music-only backend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from gem.gem import GEM
from gem.robots.bumi.metrics import compute_bumi_kinematic_metrics
from gem.utils.bumi_checkpoint_adapter import adapt_smpl_music_checkpoint_to_bumi
from gem.utils.pylogger import Log


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
        raise ValueError("GMT reorder requires two duplicate-free joint orders with identical names")
    source_index = {name: index for index, name in enumerate(source)}
    permutation = torch.tensor(
        [source_index[name] for name in target], dtype=torch.long, device=qpos_mujoco.device
    )
    return torch.cat((qpos_mujoco[..., :7], qpos_mujoco[..., 7:].index_select(-1, permutation)), dim=-1)


class BumiMusicGEM(GEM):
    """Reuse GEM optimization/CFG infrastructure while bypassing every SMPL path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.motion_backend != "bumi":
            raise ValueError(
                f"BumiMusicGEM requires model_cfg.motion_backend='bumi', got {self.motion_backend!r}"
            )
        if self.body_model is not None:
            raise RuntimeError("BumiMusicGEM must not instantiate an SMPL body model")
        if self.train_modes != ["diffusion"]:
            raise ValueError(
                f"BumiMusicGEM supports only train_modes=['diffusion'], got {self.train_modes}"
            )
        if list(self.pipeline.args.in_attr) != ["encoded_music"]:
            raise ValueError("BumiMusicGEM accepts only the encoded_music condition")
        if self.text_condition_enabled or self.denoiser_uses_text:
            raise ValueError("BumiMusicGEM must disable text encoding in model and denoiser")
        self.checkpoint_adaptation_report: dict[str, Any] | None = None

    def prepare_batch(self, batch: dict[str, Any], mode: str) -> None:
        if mode != "diffusion":
            raise ValueError(f"BumiMusicGEM only supports diffusion, got mode={mode!r}")
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
        batch["target_body_link_pos_local"] = encoded.target_body_link_pos_local
        batch["target_foot_contact"] = encoded.target_foot_contact
        batch["target_foot_contact_mask"] = encoded.target_foot_contact_mask
        batch["canonical_anchor"] = encoded.anchor_metadata
        batch["sample_indices_dict"] = self.endecoder.obs_indices_dict
        batch["device"] = encoded.normalized_features.device
        batch["B"], batch["L"] = encoded.normalized_features.shape[:2]
        batch["condition_mask"] = {
            "has_music_mask": batch["mask"]["has_music_mask"].bool() & valid
        }

    def create_condition_mask(
        self,
        batch: dict[str, Any],
        cond_mask_cfg,
        mode,
        train: bool,
        first_k_frames: int | None = None,
    ) -> dict[str, Any]:
        del cond_mask_cfg, mode
        batch_size, full_length = int(batch["B"]), int(batch["L"])
        end = full_length if first_k_frames is None else min(full_length, int(first_k_frames))
        device = batch["device"]
        music = batch.get("music_embed")
        if not isinstance(music, torch.Tensor) or tuple(music.shape[:2]) != (
            batch_size,
            full_length,
        ):
            raise ValueError(
                f"BumiMusicGEM requires music_embed [B,T,35], got "
                f"{getattr(music, 'shape', None)} for B={batch_size}, T={full_length}"
            )
        if music.shape[-1] != 35 or not bool(torch.isfinite(music).all()):
            raise ValueError(f"BUMI music_embed must be finite [B,T,35], got {music.shape}")
        valid = batch["mask"]["valid"][:, :end].bool()
        has_music = batch["condition_mask"]["has_music_mask"][:, :end].bool() & valid
        embedded = self.music_embedder(music[:, :end])
        conditional = embedded * has_music[..., None].to(embedded)
        unconditional = torch.zeros_like(conditional)
        empty = torch.zeros_like(conditional)
        if self.model_cfg.get("use_cond_exists_as_input", False):
            exists_embedder = self.cond_exists_embedder["encoded_music"]
            conditional = exists_embedder(
                torch.cat((conditional, has_music[..., None].to(conditional)), dim=-1)
            )
            no_music = torch.zeros_like(has_music[..., None], dtype=conditional.dtype)
            unconditional = exists_embedder(torch.cat((unconditional, no_music), dim=-1))
            empty = exists_embedder(torch.cat((empty, no_music), dim=-1))
        dropout = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if train and self.music_mask_prob > 0.0:
            dropout = torch.rand(batch_size, device=device) < float(self.music_mask_prob)
            conditional = torch.where(
                dropout[:, None, None], unconditional, conditional
            )
        batch["music_dropout_mask"] = dropout
        batch["f_cond"] = conditional
        batch["f_uncond"] = unconditional
        batch["f_empty"] = empty
        length = batch["length"].to(device=device).long().clamp(max=end)
        batch["length"] = length
        batch["motion"] = batch["target_x"][:, :end] * valid[..., None].to(
            batch["target_x"]
        )
        return batch

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation(batch, "default", batch_idx, dataloader_idx)

    def validation(self, batch, test_mode, batch_idx, dataloader_idx=0):
        del batch_idx
        self.prepare_batch(batch, "diffusion")
        batch["target_x"] = torch.zeros_like(batch["target_x"])
        batch = self.create_condition_mask(
            batch, cond_mask_cfg=None, mode=None, train=False
        )
        outputs = self.pipeline.forward(batch, train=False, test_mode=test_mode)
        outputs["target_qpos_canonical"] = batch["target_qpos_canonical"]
        outputs["target_body_link_pos_local"] = batch["target_body_link_pos_local"]
        metrics = compute_bumi_kinematic_metrics(
            outputs["pred_qpos_canonical"],
            self.endecoder.kinematics,
            target_qpos=batch["target_qpos_canonical"],
            valid_mask=batch["mask"]["valid"],
            music_beats=batch.get("music_beats"),
            fps=30,
        )
        report_names = (
            "joint_angle_mae_rad",
            "root_trajectory_error_m",
            "fk_body_position_error_m",
            "joint_limit_violation_rate",
            "minimum_joint_margin_rad",
            "joint_velocity_p95_radps",
            "joint_acceleration_p95_radps2",
            "joint_jerk_p95_radps3",
            "root_linear_velocity_p95_mps",
            "root_angular_velocity_p95_radps",
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

    @torch.no_grad()
    def predict(
        self,
        data: Mapping[str, Any],
        static_cam: bool = False,
        postproc: bool = False,
    ) -> dict[str, Any]:
        del static_cam, postproc
        music = data.get("music_embed")
        if not isinstance(music, torch.Tensor):
            raise KeyError("BumiMusicGEM.predict requires data['music_embed'] Tensor[T,35]")
        if music.ndim == 3 and music.shape[0] == 1:
            music = music[0]
        if music.ndim != 2 or music.shape[1] != 35 or music.shape[0] <= 0:
            raise ValueError(f"music_embed must have shape [T,35], got {music.shape}")
        if not bool(torch.isfinite(music).all()):
            raise ValueError("music_embed contains NaN or Inf")
        device = next(self.parameters()).device
        music = music.to(device=device, dtype=torch.float32)
        sequence_frames = int(music.shape[0])
        raw_length = data.get("length", sequence_frames)
        if isinstance(raw_length, torch.Tensor):
            length_value = int(raw_length.reshape(-1)[0].item())
        else:
            length_value = int(raw_length)
        if not 1 <= length_value <= sequence_frames:
            raise ValueError(
                f"predict length must be in [1,{sequence_frames}], got {length_value}"
            )
        valid = torch.arange(sequence_frames, device=device) < length_value
        raw_has_music = data.get("has_music_mask")
        if raw_has_music is None:
            has_music = valid.clone()
        else:
            has_music = torch.as_tensor(raw_has_music, device=device).bool()
            if has_music.ndim == 2 and has_music.shape[0] == 1:
                has_music = has_music[0]
            if tuple(has_music.shape) != (sequence_frames,):
                raise ValueError(
                    f"has_music_mask must have shape [{sequence_frames}], got {has_music.shape}"
                )
            has_music = has_music & valid
        batch: dict[str, Any] = {
            "B": 1,
            "L": sequence_frames,
            "device": device,
            "length": torch.tensor([length_value], dtype=torch.long, device=device),
            "music_embed": music.unsqueeze(0),
            "target_x": torch.zeros((1, sequence_frames, 93), device=device),
            "sample_indices_dict": self.endecoder.obs_indices_dict,
            "mask": {
                "valid": valid.unsqueeze(0),
                "has_music_mask": has_music.unsqueeze(0),
            },
            "condition_mask": {"has_music_mask": has_music.unsqueeze(0)},
        }
        if data.get("world_anchor") is not None:
            batch["world_anchor"] = data["world_anchor"]
        batch = self.create_condition_mask(
            batch, cond_mask_cfg=None, mode=None, train=False
        )
        outputs = self.pipeline.forward(batch, train=False, test_mode="default")
        qpos = outputs["pred_qpos"][0, :length_value]
        canonical = outputs["pred_qpos_canonical"][0, :length_value]
        result = {
            "qpos": qpos,
            "qpos_canonical": canonical,
            "fps": 30,
            "robot_name": "bumi",
            "joint_names": list(self.endecoder.kinematics.joint_order),
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "feature_dim": 93,
            "anchor_mode": self.endecoder.anchor_mode,
            "music_path": str(data.get("music_path", "")),
            "world_anchor_applied": data.get("world_anchor") is not None,
            "net_outputs": outputs,
        }
        contact_logits = outputs.get("pred_foot_contact_logits")
        if isinstance(contact_logits, torch.Tensor):
            result["pred_foot_contact_logits"] = contact_logits[0, :length_value]
        return result

    def load_pretrained_model(self, ckpt_path):
        adapter = self.model_cfg.get("checkpoint_adapter", None)
        if adapter in (None, "null", "none"):
            return super().load_pretrained_model(ckpt_path)
        if adapter != "smpl_music_to_bumi":
            raise ValueError(f"Unknown BUMI checkpoint_adapter={adapter!r}")
        Log.info(f"[BUMI CKPT Adapter] Loading SMPL music checkpoint: {ckpt_path}")
        checkpoint, report = adapt_smpl_music_checkpoint_to_bumi(self, ckpt_path)
        self.checkpoint_adaptation_report = report
        return checkpoint

    def on_fit_start(self) -> None:
        if self.checkpoint_adaptation_report is None:
            return
        trainer = self.trainer
        if not getattr(trainer, "is_global_zero", True):
            return
        run_dir = getattr(trainer, "log_dir", None) or getattr(
            trainer, "default_root_dir", "."
        )
        path = Path(run_dir) / "checkpoint_adaptation_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.checkpoint_adaptation_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Log.info(f"[BUMI CKPT Adapter] Wrote report: {path}")


__all__ = ["BumiMusicGEM", "reorder_mujoco_joints_to_gmt"]
