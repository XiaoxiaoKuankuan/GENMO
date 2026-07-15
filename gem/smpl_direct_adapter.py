# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""SMPL-X 22-joint landmarks -> explicit joint-center/anatomical targets.

This is the GEM2 path.  It deliberately does not reuse the legacy GEM1/Xsens-
style segment names: positions are named SMPL joint centers while rotations are
anatomical frames derived from neighbouring joint geometry.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - NumPy-only tests do not need Torch.
    torch = None


TARGET_NAMES = (
    "SMPL_Pelvis",
    "SMPL_Chest",
    "SMPL_LeftHip",
    "SMPL_RightHip",
    "SMPL_LeftKnee",
    "SMPL_RightKnee",
    "SMPL_LeftAnkle",
    "SMPL_RightAnkle",
    "SMPL_LeftShoulder",
    "SMPL_RightShoulder",
    "SMPL_LeftElbow",
    "SMPL_RightElbow",
    "SMPL_LeftWrist",
    "SMPL_RightWrist",
)

TARGET_JOINT_INDICES = {
    "SMPL_Pelvis": 0,
    "SMPL_Chest": 9,
    "SMPL_LeftHip": 1,
    "SMPL_RightHip": 2,
    "SMPL_LeftKnee": 4,
    "SMPL_RightKnee": 5,
    "SMPL_LeftAnkle": 7,
    "SMPL_RightAnkle": 8,
    "SMPL_LeftShoulder": 16,
    "SMPL_RightShoulder": 17,
    "SMPL_LeftElbow": 18,
    "SMPL_RightElbow": 19,
    "SMPL_LeftWrist": 20,
    "SMPL_RightWrist": 21,
}

HIERARCHY = (
    ("SMPL_Pelvis", "SMPL_Chest"),
    ("SMPL_Pelvis", "SMPL_LeftHip"),
    ("SMPL_LeftHip", "SMPL_LeftKnee"),
    ("SMPL_LeftKnee", "SMPL_LeftAnkle"),
    ("SMPL_Pelvis", "SMPL_RightHip"),
    ("SMPL_RightHip", "SMPL_RightKnee"),
    ("SMPL_RightKnee", "SMPL_RightAnkle"),
    ("SMPL_Chest", "SMPL_LeftShoulder"),
    ("SMPL_LeftShoulder", "SMPL_LeftElbow"),
    ("SMPL_LeftElbow", "SMPL_LeftWrist"),
    ("SMPL_Chest", "SMPL_RightShoulder"),
    ("SMPL_RightShoulder", "SMPL_RightElbow"),
    ("SMPL_RightElbow", "SMPL_RightWrist"),
)

AXIS_CONVERT_AY_TO_ZUP = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class TargetPose:
    """One named SMPL joint-center position and its anatomical frame."""

    position_zup: np.ndarray
    rotation_zup: np.ndarray


@dataclass(frozen=True)
class SMPLDirectFrame:
    """One complete GEM2 adapter result."""

    frame_id: int
    timestamp_ns: int
    joints_zup: np.ndarray
    raw_targets: dict[str, TargetPose]
    scaled_targets: dict[str, TargetPose]
    vertical_mode: str
    contact_mask: tuple[bool, bool]


class SMPLDirectAdapter:
    """Stateful SMPL joint-center adapter for the independent GEM2 protocol."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        heading_source: str | None = None,
        forward_sign: int | None = None,
        vertical_mode: str | None = None,
        user_yaw_deg: float = 0.0,
        global_scale: float = 1.0,
    ) -> None:
        self.config = copy.deepcopy(config)
        if self.config.get("schema") != "smpl_direct_joint_center_v1":
            raise ValueError("adapter schema must be smpl_direct_joint_center_v1")
        if tuple(self.config["target_order"]) != TARGET_NAMES:
            raise ValueError("target_order must contain the canonical 14 GEM2 names")
        configured_indices = {
            str(name): int(index) for name, index in self.config["target_joint_indices"].items()
        }
        if configured_indices != TARGET_JOINT_INDICES:
            raise ValueError("target_joint_indices must match the SMPL-X 22-joint table")
        if tuple(tuple(edge) for edge in self.config["hierarchy"]) != HIERARCHY:
            raise ValueError("hierarchy must be the joint-center GEM2 hierarchy")

        self.heading_source = heading_source or self.config.get("heading_source", "joints")
        if self.heading_source not in {"joints", "pelvis"}:
            raise ValueError("heading_source must be joints or pelvis")
        self.forward_sign = int(
            self.config.get("forward_sign", 1) if forward_sign is None else forward_sign
        )
        if self.forward_sign not in {-1, 1}:
            raise ValueError("forward_sign must be +1 or -1")
        self.vertical_mode = vertical_mode or self.config.get("vertical_mode", "foot_lock")
        if self.vertical_mode not in {"gem", "foot_lock", "contact"}:
            raise ValueError("vertical_mode must be gem, foot_lock, or contact")
        if not math.isfinite(global_scale) or global_scale <= 0.0:
            raise ValueError("global_scale must be finite and > 0")

        self.global_scale = float(global_scale)
        self.root_translation_scale = float(self.config["root_translation_scale"])
        self.edge_scales = {
            str(name): float(value) for name, value in self.config["edge_scales"].items()
        }
        if set(self.edge_scales) != {f"{p}->{c}" for p, c in HIERARCHY}:
            raise ValueError("edge_scales must contain all 13 joint-center edges")
        if any(not math.isfinite(value) or value <= 0 for value in self.edge_scales.values()):
            raise ValueError("all edge scales must be finite and > 0")

        self.contact = self.config.get("contact", {})
        self._user_yaw = self._rot_z(math.radians(float(user_yaw_deg)))
        self.reset()

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> SMPLDirectAdapter:
        return cls(json.loads(Path(path).read_text()), **kwargs)

    def reset(self) -> None:
        self._heading_rotation: np.ndarray | None = None
        self._initial_pelvis_xy: np.ndarray | None = None
        self._initial_ground_z: float | None = None
        self._ground_z: float | None = None
        self._previous_frames: dict[str, np.ndarray] = {}
        self._previous_forward: np.ndarray | None = None
        self._previous_foot_z: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._contact_mask = np.asarray((True, True), dtype=bool)
        self._previous_root_z: float | None = None
        self._previous_raw_pelvis_z: float | None = None
        self._flight_root_z: float | None = None
        self._flight_pelvis_z: float | None = None

    @staticmethod
    def _rot_z(angle: float) -> np.ndarray:
        cosine, sine = math.cos(angle), math.sin(angle)
        return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))

    @staticmethod
    def _numpy(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        if torch is not None and isinstance(value, torch.Tensor):
            array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        else:
            array = np.asarray(value, dtype=np.float64)
        if array.shape != shape:
            raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return array

    @staticmethod
    def _normalize(vector: np.ndarray, epsilon: float = 1e-8) -> np.ndarray | None:
        norm = float(np.linalg.norm(vector))
        return None if norm < epsilon else vector / norm

    @staticmethod
    def _proper_rotation(rotation: np.ndarray) -> bool:
        return bool(
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=1e-5)
        )

    def _fallback(self, name: str, rotations: np.ndarray, joint: int) -> np.ndarray:
        previous = self._previous_frames.get(name)
        if previous is not None:
            return previous.copy()
        fallback = rotations[joint]
        if not self._proper_rotation(fallback):
            raise ValueError(f"fallback joint rotation {joint} is not proper SO(3)")
        return fallback.copy()

    def _store_frame(self, name: str, rotation: np.ndarray) -> np.ndarray:
        if not self._proper_rotation(rotation):
            raise ValueError(f"derived frame for {name} is not proper SO(3)")
        self._previous_frames[name] = rotation.copy()
        return rotation

    def _frame_zy(
        self,
        name: str,
        z_raw: np.ndarray,
        y_raw: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        """Build pelvis/chest R=[x,y,z] from z and projected y."""
        z_axis = self._normalize(z_raw)
        if z_axis is None:
            return self._store_frame(name, fallback)
        y_axis = self._normalize(y_raw - z_axis * np.dot(y_raw, z_axis))
        if y_axis is None:
            return self._store_frame(name, fallback)
        previous = self._previous_frames.get(name)
        if previous is not None and np.dot(y_axis, previous[:, 1]) < 0.0:
            y_axis = -y_axis
        x_axis = self._normalize(np.cross(y_axis, z_axis))
        if x_axis is None:
            return self._store_frame(name, fallback)
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        return self._store_frame(name, np.column_stack((x_axis, y_axis, z_axis)))

    def _frame_zx(
        self,
        name: str,
        z_raw: np.ndarray,
        x_raw: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        """Build long-axis limb R=[x,y,z] from z and projected forward x."""
        z_axis = self._normalize(z_raw)
        if z_axis is None:
            return self._store_frame(name, fallback)
        x_axis = self._normalize(x_raw - z_axis * np.dot(x_raw, z_axis))
        if x_axis is None:
            return self._store_frame(name, fallback)
        previous = self._previous_frames.get(name)
        if previous is not None and np.dot(x_axis, previous[:, 0]) < 0.0:
            x_axis = -x_axis
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        if y_axis is None:
            return self._store_frame(name, fallback)
        x_axis = self._normalize(np.cross(y_axis, z_axis))
        return self._store_frame(name, np.column_stack((x_axis, y_axis, z_axis)))

    def _frame_foot(
        self,
        name: str,
        x_raw: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        x_axis = self._normalize(x_raw)
        if x_axis is None:
            return self._store_frame(name, fallback)
        world_up = np.asarray((0.0, 0.0, 1.0))
        z_axis = self._normalize(world_up - x_axis * np.dot(world_up, x_axis))
        if z_axis is None:
            return self._store_frame(name, fallback)
        previous = self._previous_frames.get(name)
        if previous is not None and np.dot(z_axis, previous[:, 2]) < 0.0:
            z_axis = -z_axis
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        if y_axis is None:
            return self._store_frame(name, fallback)
        z_axis = self._normalize(np.cross(x_axis, y_axis))
        return self._store_frame(name, np.column_stack((x_axis, y_axis, z_axis)))

    def _heading_forward(self, joints: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        if self.heading_source == "pelvis":
            candidate = rotations[0, :, 0].copy()
        else:
            up_axis = self._normalize(joints[9] - joints[0])
            if up_axis is None:
                up_axis = np.asarray((0.0, 0.0, 1.0))
            left_raw = (joints[1] - joints[2]) + (joints[16] - joints[17])
            left_axis = self._normalize(left_raw - up_axis * np.dot(left_raw, up_axis))
            candidate = (
                None if left_axis is None else self.forward_sign * np.cross(left_axis, up_axis)
            )
        if candidate is not None:
            candidate = np.asarray(candidate, dtype=np.float64)
            candidate[2] = 0.0
        forward = None if candidate is None else self._normalize(candidate)
        if forward is None:
            forward = self._previous_forward
        if forward is None:
            raise ValueError("cannot derive SMPL body heading")
        self._previous_forward = forward.copy()
        return forward

    def _normalize_vertical(
        self, joints: np.ndarray, timestamp_ns: int
    ) -> tuple[np.ndarray, tuple[bool, bool]]:
        pelvis = joints[0]
        if self._initial_pelvis_xy is None:
            self._initial_pelvis_xy = pelvis[:2].copy()
        root_xy = pelvis[:2] - self._initial_pelvis_xy
        feet = joints[np.asarray((10, 11))]
        if self._initial_ground_z is None:
            self._initial_ground_z = float(np.median(feet[:, 2]))
        if self._ground_z is None:
            self._ground_z = self._initial_ground_z

        if self.vertical_mode == "gem":
            origin = np.asarray(
                (self._initial_pelvis_xy[0], self._initial_pelvis_xy[1], self._initial_ground_z)
            )
            return joints - origin, (False, False)

        joints_relative = joints - pelvis
        candidates = -joints_relative[np.asarray((10, 11)), 2]
        contact_mask = np.asarray((True, True), dtype=bool)
        if self.vertical_mode == "foot_lock":
            root_z = float(np.median(candidates))
        else:
            dt = 1.0 / 30.0
            if self._previous_timestamp_ns is not None:
                dt = max((timestamp_ns - self._previous_timestamp_ns) * 1e-9, 1e-3)
            velocities = (
                np.zeros(2)
                if self._previous_foot_z is None
                else (feet[:, 2] - self._previous_foot_z) / dt
            )
            enter_height = float(self.contact.get("enter_height_m", 0.035))
            exit_height = float(self.contact.get("exit_height_m", 0.080))
            enter_velocity = float(self.contact.get("enter_velocity_mps", 0.25))
            exit_velocity = float(self.contact.get("exit_velocity_mps", 0.45))
            heights = np.abs(feet[:, 2] - self._ground_z)
            contact_mask = self._contact_mask.copy()
            contact_mask[contact_mask] &= (heights[contact_mask] <= exit_height) & (
                np.abs(velocities[contact_mask]) <= exit_velocity
            )
            inactive = ~contact_mask
            contact_mask[inactive] = (heights[inactive] <= enter_height) & (
                np.abs(velocities[inactive]) <= enter_velocity
            )

            if np.any(contact_mask):
                alpha = float(self.contact.get("ground_ema_alpha", 0.10))
                support_ground = float(np.median(feet[contact_mask, 2]))
                self._ground_z += alpha * (support_ground - self._ground_z)
                root_z = float(np.median(candidates[contact_mask]))
                self._flight_root_z = None
                self._flight_pelvis_z = None
            else:
                if self._flight_root_z is None:
                    self._flight_root_z = (
                        float(np.median(candidates))
                        if self._previous_root_z is None
                        else self._previous_root_z
                    )
                    self._flight_pelvis_z = (
                        float(pelvis[2])
                        if self._previous_raw_pelvis_z is None
                        else self._previous_raw_pelvis_z
                    )
                root_z = self._flight_root_z + float(pelvis[2] - self._flight_pelvis_z)

        root_output = np.asarray((root_xy[0], root_xy[1], root_z))
        normalized = joints_relative + root_output
        self._previous_foot_z = feet[:, 2].copy()
        self._previous_timestamp_ns = timestamp_ns
        self._contact_mask = contact_mask.copy()
        self._previous_root_z = float(root_z)
        self._previous_raw_pelvis_z = float(pelvis[2])
        return normalized, (bool(contact_mask[0]), bool(contact_mask[1]))

    def _derive_targets(self, joints: np.ndarray, rotations: np.ndarray) -> dict[str, TargetPose]:
        pelvis_fallback = self._fallback("SMPL_Pelvis", rotations, 0)
        pelvis_rotation = self._frame_zy(
            "SMPL_Pelvis",
            joints[9] - joints[0],
            joints[1] - joints[2],
            pelvis_fallback,
        )
        chest_fallback = self._fallback("SMPL_Chest", rotations, 9)
        chest_rotation = self._frame_zy(
            "SMPL_Chest",
            joints[12] - joints[0],
            joints[16] - joints[17],
            chest_fallback,
        )

        frames: dict[str, np.ndarray] = {
            "SMPL_Pelvis": pelvis_rotation,
            "SMPL_Chest": chest_rotation,
        }
        limb_definitions = (
            ("SMPL_LeftHip", 1, 4, 0, pelvis_rotation[:, 0]),
            ("SMPL_RightHip", 2, 5, 0, pelvis_rotation[:, 0]),
            ("SMPL_LeftKnee", 4, 7, 4, pelvis_rotation[:, 0]),
            ("SMPL_RightKnee", 5, 8, 5, pelvis_rotation[:, 0]),
            ("SMPL_LeftShoulder", 16, 18, 16, chest_rotation[:, 0]),
            ("SMPL_RightShoulder", 17, 19, 17, chest_rotation[:, 0]),
            ("SMPL_LeftElbow", 18, 20, 18, chest_rotation[:, 0]),
            ("SMPL_RightElbow", 19, 21, 19, chest_rotation[:, 0]),
        )
        for name, proximal, distal, fallback_joint, reference in limb_definitions:
            frames[name] = self._frame_zx(
                name,
                joints[distal] - joints[proximal],
                reference,
                self._fallback(name, rotations, fallback_joint),
            )

        frames["SMPL_LeftAnkle"] = self._frame_foot(
            "SMPL_LeftAnkle",
            joints[10] - joints[7],
            self._fallback("SMPL_LeftAnkle", rotations, 7),
        )
        frames["SMPL_RightAnkle"] = self._frame_foot(
            "SMPL_RightAnkle",
            joints[11] - joints[8],
            self._fallback("SMPL_RightAnkle", rotations, 8),
        )
        frames["SMPL_LeftWrist"] = frames["SMPL_LeftElbow"].copy()
        frames["SMPL_RightWrist"] = frames["SMPL_RightElbow"].copy()
        self._previous_frames["SMPL_LeftWrist"] = frames["SMPL_LeftWrist"].copy()
        self._previous_frames["SMPL_RightWrist"] = frames["SMPL_RightWrist"].copy()

        return {
            name: TargetPose(joints[TARGET_JOINT_INDICES[name]].copy(), frames[name].copy())
            for name in TARGET_NAMES
        }

    def _scale_hierarchy(self, raw: dict[str, TargetPose]) -> dict[str, TargetPose]:
        root = "SMPL_Pelvis"
        scaled: dict[str, TargetPose] = {
            root: TargetPose(
                raw[root].position_zup * self.root_translation_scale * self.global_scale,
                raw[root].rotation_zup.copy(),
            )
        }
        for parent, child in HIERARCHY:
            edge = f"{parent}->{child}"
            scale = self.edge_scales[edge] * self.global_scale
            position = scaled[parent].position_zup + scale * (
                raw[child].position_zup - raw[parent].position_zup
            )
            scaled[child] = TargetPose(position, raw[child].rotation_zup.copy())
        return {name: scaled[name] for name in TARGET_NAMES}

    def adapt(
        self,
        joints_ay: Any,
        joint_rots_ay: Any,
        *,
        frame_id: int = 0,
        timestamp_ns: int | None = None,
    ) -> SMPLDirectFrame:
        joints_ay_np = self._numpy(joints_ay, (22, 3), "joints_ay")
        rotations_ay_np = self._numpy(joint_rots_ay, (22, 3, 3), "joint_rots_ay")
        stamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        if stamp < 0:
            raise ValueError("timestamp_ns must be non-negative")

        joints = np.einsum("ij,nj->ni", AXIS_CONVERT_AY_TO_ZUP, joints_ay_np)
        rotations = np.einsum(
            "ij,njk,kl->nil",
            AXIS_CONVERT_AY_TO_ZUP,
            rotations_ay_np,
            AXIS_CONVERT_AY_TO_ZUP.T,
        )
        forward = self._heading_forward(joints, rotations)
        if self._heading_rotation is None:
            yaw = math.atan2(float(forward[1]), float(forward[0]))
            self._heading_rotation = self._user_yaw @ self._rot_z(-yaw)
        joints = np.einsum("ij,nj->ni", self._heading_rotation, joints)
        rotations = np.einsum("ij,njk->nik", self._heading_rotation, rotations)

        normalized_joints, contact_mask = self._normalize_vertical(joints, stamp)
        raw_targets = self._derive_targets(normalized_joints, rotations)
        scaled_targets = self._scale_hierarchy(raw_targets)
        return SMPLDirectFrame(
            frame_id=int(frame_id),
            timestamp_ns=stamp,
            joints_zup=normalized_joints,
            raw_targets=raw_targets,
            scaled_targets=scaled_targets,
            vertical_mode=self.vertical_mode,
            contact_mask=contact_mask,
        )
