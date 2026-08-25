"""BUMI qpos30 表示的归一化、编解码、FK 几何和足底接触监督桥接。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .contacts import BumiFootContactTargets, derive_bumi_foot_contact
from .feature_codec import (
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_QUATERNION_CONVENTION,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
    BumiMotionFeatureCodec,
)
from .kinematics import BumiKinematics, resolve_asset_path

STATS_CONTRACT_VERSION = "genmo.bumi_qpos30_stats.v3"


@dataclass(frozen=True)
class BumiEncodedMotion:
    normalized_features: torch.Tensor
    physical_features: torch.Tensor
    canonical_qpos: torch.Tensor
    target_body_link_pos_root: torch.Tensor
    anchor_metadata: dict[str, torch.Tensor | str]
    target_foot_contact: torch.Tensor
    target_foot_contact_mask: torch.Tensor
    target_contact_ground_height: torch.Tensor

    @property
    def body_link_pos_root(self) -> torch.Tensor:
        return self.target_body_link_pos_root


def _canonical_feature_slices(value: Any) -> dict[str, tuple[int, int]]:
    if not isinstance(value, dict):
        raise ValueError("BUMI stats feature_slices must be an object")
    result: dict[str, tuple[int, int]] = {}
    for name, expected in BUMI_FEATURE_SLICES.items():
        raw = value.get(name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"BUMI stats feature_slices.{name} must contain [start, end]")
        result[name] = (int(raw[0]), int(raw[1]))
        if result[name] != expected:
            raise ValueError(
                f"BUMI stats feature_slices.{name} must be {expected}, got {result[name]}"
            )
    if set(value) != set(BUMI_FEATURE_SLICES):
        raise ValueError(
            "BUMI stats feature_slices must contain exactly "
            f"{sorted(BUMI_FEATURE_SLICES)}, got {sorted(value)}"
        )
    return result


class BumiEndecoder(nn.Module):
    """在 BUMI qpos28、归一化 qpos30 特征和权威 FK link 几何间转换。"""

    def __init__(
        self,
        kinematics_path: str | Path,
        stats_path: str | Path,
        feat_dim: int = BUMI_FEATURE_DIM,
        rotation_representation: str = "rot6d",
        anchor_mode: str = BUMI_ANCHOR_MODE,
        clip_std_min: float = 0.01,
        allow_placeholder_stats: bool = False,
        contact_height_threshold: float = 0.035,
        contact_velocity_threshold: float = 0.15,
        contact_exit_height_threshold: float = 0.055,
        contact_exit_velocity_threshold: float = 0.25,
        contact_ground_quantile: float = 0.02,
        contact_min_frames: int = 2,
        enable_contact_targets: bool = True,
        fps: int = 30,
    ) -> None:
        super().__init__()
        if int(feat_dim) != BUMI_FEATURE_DIM:
            raise ValueError(f"BumiEndecoder feat_dim must be {BUMI_FEATURE_DIM}, got {feat_dim}")
        if rotation_representation != "rot6d":
            raise ValueError(
                f"BumiEndecoder rotation_representation must be 'rot6d', got {rotation_representation!r}"
            )
        if anchor_mode != BUMI_ANCHOR_MODE:
            raise ValueError(
                f"BumiEndecoder anchor_mode must be {BUMI_ANCHOR_MODE!r}, got {anchor_mode!r}"
            )
        if not 0.0 < float(clip_std_min) <= 1.0:
            raise ValueError("clip_std_min must be in (0, 1]")
        if int(fps) != 30:
            raise ValueError(f"BUMI music-native training is fixed at 30 FPS, got {fps}")
        self.kinematics = BumiKinematics(kinematics_path)
        self.codec = BumiMotionFeatureCodec(self.kinematics)
        self.feat_dim = BUMI_FEATURE_DIM
        self.representation_contract_version = BUMI_REPRESENTATION_CONTRACT_VERSION
        self.rotation_representation = "rot6d"
        self.anchor_mode = BUMI_ANCHOR_MODE
        self.fps = 30
        self.contact_height_threshold = float(contact_height_threshold)
        self.contact_velocity_threshold = float(contact_velocity_threshold)
        self.contact_exit_height_threshold = float(contact_exit_height_threshold)
        self.contact_exit_velocity_threshold = float(contact_exit_velocity_threshold)
        self.contact_ground_quantile = float(contact_ground_quantile)
        self.contact_min_frames = int(contact_min_frames)
        self.enable_contact_targets = bool(enable_contact_targets)
        self.clip_std_min = float(clip_std_min)
        self.stats_path = str(resolve_asset_path(stats_path)) if str(stats_path).strip() else ""
        if not self.stats_path:
            raise ValueError(
                "BUMI stats_path is required and must point to train-split qpos30 statistics"
            )
        stats_file = Path(self.stats_path)
        if not stats_file.is_file():
            raise FileNotFoundError(
                f"BUMI stats JSON does not exist: {stats_file}. Compute it from the real train "
                "split with tools/data/bumi/compute_bumi_30d_stats.py"
            )
        try:
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid BUMI stats JSON {stats_file}: {exc}") from exc
        self._validate_stats(stats, stats_file, allow_placeholder_stats)
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32)
        if mean.shape != (BUMI_FEATURE_DIM,) or std.shape != (BUMI_FEATURE_DIM,):
            raise ValueError(
                f"BUMI stats {stats_file} must contain mean/std [{BUMI_FEATURE_DIM}], "
                f"got {mean.shape}/{std.shape}"
            )
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError(f"BUMI stats {stats_file} contains NaN or Inf")
        if bool((std < 0).any()):
            raise ValueError(f"BUMI stats {stats_file} contains a negative std")
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std.clamp_min(float(clip_std_min)), persistent=False)
        self.obs_indices_dict: dict[str, tuple[int, int]] | None = None
        self.build_obs_indices_dict()

    def _validate_stats(self, stats: Any, path: Path, allow_placeholder_stats: bool) -> None:
        if not isinstance(stats, dict):
            raise ValueError(f"BUMI stats must be a JSON object: {path}")
        if bool(stats.get("is_placeholder", False)) and not allow_placeholder_stats:
            raise ValueError(
                f"BUMI stats {path} is marked is_placeholder=true. Formal training refuses "
                "placeholder/identity statistics unless allow_placeholder_stats=true is explicit."
            )
        expected = {
            "contract_version": STATS_CONTRACT_VERSION,
            "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
            "robot_name": "bumi",
            "feature_dim": BUMI_FEATURE_DIM,
            "anchor_mode": BUMI_ANCHOR_MODE,
            "quaternion_convention": BUMI_QUATERNION_CONVENTION,
            "training_clip_std_min": self.clip_std_min,
        }
        for key, expected_value in expected.items():
            if stats.get(key) != expected_value:
                raise ValueError(
                    f"BUMI stats {path}: {key} must be {expected_value!r}, got {stats.get(key)!r}"
                )
        _canonical_feature_slices(stats.get("feature_slices"))
        if tuple(map(str, stats.get("joint_names", ()))) != self.kinematics.joint_order:
            raise ValueError(
                f"BUMI stats {path} joint_names do not exactly match kinematics joint_order"
            )
        if stats.get("kinematics_sha256") != self.kinematics.kinematics_sha256:
            raise ValueError(
                f"BUMI stats {path} was computed with kinematics_sha256="
                f"{stats.get('kinematics_sha256')!r}, but loaded kinematics is "
                f"{self.kinematics.kinematics_sha256!r}"
            )

    def normalize(self, value: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        if value.shape[-1] != BUMI_FEATURE_DIM:
            raise ValueError(
                f"BUMI normalization expects [...,{BUMI_FEATURE_DIM}], got {value.shape}"
            )
        return (value - self.mean.to(value)) / self.std.to(value)

    def denormalize(self, value: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        if value.shape[-1] != BUMI_FEATURE_DIM:
            raise ValueError(
                f"BUMI denormalization expects [...,{BUMI_FEATURE_DIM}], got {value.shape}"
            )
        return value * self.std.to(value) + self.mean.to(value)

    def encode(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        return self.encode_with_aux(inputs).normalized_features

    def encode_with_aux(self, inputs: Mapping[str, Any]) -> BumiEncodedMotion:
        qpos = inputs.get("qpos")
        if not isinstance(qpos, torch.Tensor):
            raise KeyError("BumiEndecoder.encode_with_aux requires inputs['qpos'] tensor")
        encoded = self.codec.encode(qpos)
        normalized = self.normalize(encoded.physical_features)
        valid = self._valid_mask(inputs, qpos)
        if self.enable_contact_targets:
            derived_targets = self.infer_foot_contact_targets(
                encoded.normalized_world_qpos,
                valid,
                estimate_ground_mask=self._estimate_ground_mask(inputs, qpos),
                fk={
                    "body_pos_w": encoded.body_pos_w,
                    "body_quat_w": encoded.body_quat_w,
                },
            )
            derived_contact = derived_targets.contact
            target_contact, contact_mask = self._merge_contact_labels(
                inputs, derived_contact, valid
            )
            contact_ground_height = derived_targets.ground_height
        else:
            target_contact = torch.zeros(
                (*valid.shape, 2), dtype=encoded.physical_features.dtype, device=valid.device
            )
            contact_mask = torch.zeros_like(target_contact, dtype=torch.bool)
            contact_ground_height = torch.zeros(
                qpos.shape[:-2], dtype=encoded.physical_features.dtype, device=qpos.device
            )
        anchor = encoded.anchor
        return BumiEncodedMotion(
            normalized_features=normalized,
            physical_features=encoded.physical_features,
            canonical_qpos=encoded.canonical_qpos,
            target_body_link_pos_root=encoded.body_link_pos_root,
            anchor_metadata={
                "anchor_mode": BUMI_ANCHOR_MODE,
                "position_w": anchor.position_w,
                "heading_quat_wxyz": anchor.heading_quat_wxyz,
                "heading_inverse_quat_wxyz": anchor.heading_inverse_quat_wxyz,
                "yaw": anchor.yaw,
                "default_root_height": anchor.default_root_height,
            },
            target_foot_contact=target_contact,
            target_foot_contact_mask=contact_mask,
            target_contact_ground_height=contact_ground_height,
        )

    @staticmethod
    def _valid_mask(inputs: Mapping[str, Any], qpos: torch.Tensor) -> torch.Tensor:
        masks = inputs.get("mask")
        valid = masks.get("valid") if isinstance(masks, Mapping) else None
        if valid is None:
            valid = torch.ones(qpos.shape[:-1], dtype=torch.bool, device=qpos.device)
        if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != tuple(qpos.shape[:-1]):
            raise ValueError(
                f"inputs['mask']['valid'] must have shape {qpos.shape[:-1]}, "
                f"got {getattr(valid, 'shape', None)}"
            )
        return valid.bool()

    def infer_foot_contact(
        self,
        qpos: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        velocity_threshold: float | None = None,
        estimate_ground_mask: torch.Tensor | None = None,
        fk: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """用权威 FK 鞋底代理、迟滞阈值和最短持续时间生成左右接触。"""

        return self.infer_foot_contact_targets(
            qpos,
            valid_mask,
            velocity_threshold=velocity_threshold,
            estimate_ground_mask=estimate_ground_mask,
            fk=fk,
        ).contact

    def infer_foot_contact_targets(
        self,
        qpos: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        velocity_threshold: float | None = None,
        estimate_ground_mask: torch.Tensor | None = None,
        fk: Mapping[str, torch.Tensor] | None = None,
    ) -> BumiFootContactTargets:
        """返回接触标签及其 FK 速度、足底高度和逐样本地面诊断。"""

        return derive_bumi_foot_contact(
            qpos,
            self.kinematics,
            valid_mask=valid_mask,
            fps=self.fps,
            ground_height=0.0,
            estimate_ground_mask=estimate_ground_mask,
            ground_quantile=self.contact_ground_quantile,
            enter_height=self.contact_height_threshold,
            exit_height=self.contact_exit_height_threshold,
            enter_speed=(
                self.contact_velocity_threshold
                if velocity_threshold is None
                else float(velocity_threshold)
            ),
            exit_speed=max(
                self.contact_exit_velocity_threshold,
                self.contact_velocity_threshold
                if velocity_threshold is None
                else float(velocity_threshold),
            ),
            min_contact_frames=self.contact_min_frames,
            fk=fk,
        )

    @staticmethod
    def _estimate_ground_mask(inputs: Mapping[str, Any], qpos: torch.Tensor) -> torch.Tensor | None:
        """历史 body-origin 数据估地面；GMR 足底归零数据严格使用 Z=0。"""

        metadata = inputs.get("meta")
        if metadata is None:
            return None
        if isinstance(metadata, Mapping):
            metadata = [metadata]
        if not isinstance(metadata, (list, tuple)):
            raise ValueError("BUMI batch meta must be a mapping or a sequence of mappings")
        batch_shape = qpos.shape[:-2]
        expected_samples = 1
        for size in batch_shape:
            expected_samples *= int(size)
        if len(metadata) != expected_samples:
            raise ValueError(
                f"BUMI batch meta has {len(metadata)} samples, expected {expected_samples}"
            )
        modes: list[bool] = []
        for item in metadata:
            if not isinstance(item, Mapping):
                raise ValueError("every BUMI batch meta item must be a mapping")
            semantics = str(item.get("ground_semantics", ""))
            if semantics == "legacy_body_origin_min_zero":
                modes.append(True)
            elif semantics == "gmr_foot_sole_ground_zero_v1":
                modes.append(False)
            else:
                raise ValueError(f"unsupported BUMI ground_semantics={semantics!r}")
        return torch.tensor(modes, dtype=torch.bool, device=qpos.device).reshape(batch_shape)

    @staticmethod
    def _merge_contact_labels(
        inputs: Mapping[str, Any], derived: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        supplied = inputs.get("foot_contact")
        if supplied is None:
            return derived, valid[..., None].expand_as(derived)
        if not isinstance(supplied, torch.Tensor) or tuple(supplied.shape) != tuple(derived.shape):
            raise ValueError(
                f"foot_contact must have shape {tuple(derived.shape)}, "
                f"got {getattr(supplied, 'shape', None)}"
            )
        if not bool(torch.isfinite(supplied).all()):
            raise ValueError("foot_contact contains NaN or Inf")
        supplied = supplied.to(derived).clamp(0.0, 1.0)
        available = inputs.get("foot_contact_available")
        if available is None:
            available_mask = valid
        else:
            if not isinstance(available, torch.Tensor) or tuple(available.shape) != tuple(
                valid.shape
            ):
                raise ValueError(
                    f"foot_contact_available must have shape {tuple(valid.shape)}, "
                    f"got {getattr(available, 'shape', None)}"
                )
            available_mask = available.bool() & valid
        target = torch.where(available_mask[..., None], supplied, derived)
        # Missing dataset labels are replaced by a deterministic GT-qpos FK label,
        # so every valid frame remains supervised without pretending zero is GT.
        return target, valid[..., None].expand_as(target)

    def decode(self, normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        physical = self.denormalize(normalized)
        components = self.codec.split_features(physical)
        rot_slice = BUMI_FEATURE_SLICES["root_rot_local"]
        return {
            "root_delta_xy_heading": components.root_delta_xy_heading,
            "root_height_offset": components.root_height_offset,
            "root_rot_local_6d": physical[..., rot_slice[0] : rot_slice[1]],
            "root_rot_local_quat": components.root_rot_local_quat,
            "joint_dof": components.joint_dof,
            "physical_features": physical,
        }

    def compose_qpos(
        self,
        decode_dict: Mapping[str, torch.Tensor],
        world_anchor: Mapping[str, Any] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        required = (
            "root_delta_xy_heading",
            "root_height_offset",
            "root_rot_local_quat",
            "joint_dof",
        )
        missing = [key for key in required if key not in decode_dict]
        if missing:
            raise KeyError(f"BUMI decode_dict is missing {missing}")
        root_position = self.codec.rollout_root_position(
            decode_dict["root_delta_xy_heading"],
            decode_dict["root_height_offset"],
            decode_dict["root_rot_local_quat"],
        )
        canonical = torch.cat(
            (
                root_position,
                decode_dict["root_rot_local_quat"],
                decode_dict["joint_dof"],
            ),
            dim=-1,
        )
        canonical = self.codec.normalize_qpos_sequence(canonical)
        if world_anchor is None:
            return canonical
        return self.codec.apply_world_anchor(canonical, world_anchor)

    def authoritative_body_link_positions_root(self, canonical_qpos: torch.Tensor) -> torch.Tensor:
        fk = self.kinematics.forward_kinematics(canonical_qpos)
        return self.codec.body_positions_in_root_frame(
            canonical_qpos[..., :3],
            canonical_qpos[..., 3:7],
            fk["body_pos_w"][..., 1:, :],
        )

    def build_obs_indices_dict(self) -> None:
        self.obs_indices_dict = dict(BUMI_FEATURE_SLICES)

    def get_obs_indices(self, name: str) -> tuple[int, int]:
        if self.obs_indices_dict is None:
            self.build_obs_indices_dict()
        if name not in self.obs_indices_dict:
            raise KeyError(f"Unknown BUMI feature group {name!r}")
        return self.obs_indices_dict[name]

    def get_motion_dim(self) -> int:
        return BUMI_FEATURE_DIM

    def get_static_gt(
        self, inputs: Mapping[str, Any], vel_thr: float | None = None
    ) -> torch.Tensor:
        qpos = inputs.get("qpos")
        if not isinstance(qpos, torch.Tensor):
            raise KeyError("BUMI contact supervision requires inputs['qpos']")
        valid = self._valid_mask(inputs, qpos)
        derived = self.infer_foot_contact(
            qpos,
            valid,
            velocity_threshold=None if vel_thr is None else float(vel_thr),
            estimate_ground_mask=self._estimate_ground_mask(inputs, qpos),
        )
        return self._merge_contact_labels(inputs, derived, valid)[0]


__all__ = ["BumiEncodedMotion", "BumiEndecoder", "STATS_CONTRACT_VERSION"]
