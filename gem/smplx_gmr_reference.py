# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Original-GMR-style SMPL-X joint targets for the independent SMP1 protocol.

Positions are the selected SMPL-X joint centers.  Rotations are the global FK
joint rotations, not anatomical frames reconstructed from joint geometry.
GENMO uses AY/Y-up world coordinates, so a world-only AY -> Z-up transform is
applied to positions and global rotations while preserving the SMPL-X local
joint-frame convention expected by the original GMR configuration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - NumPy-only tests do not require Torch.
    torch = None


TARGET_NAMES = (
    "pelvis",
    "spine3",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

TARGET_JOINT_INDICES = {
    "pelvis": 0,
    "spine3": 9,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_foot": 10,
    "right_foot": 11,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

HIERARCHY = (
    ("pelvis", "spine3"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
    ("spine3", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("spine3", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)

AXIS_CONVERT_AY_TO_ZUP = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class ReferenceTarget:
    position_zup: np.ndarray
    rotation_zup: np.ndarray


@dataclass(frozen=True)
class SMPLXReferenceFrame:
    frame_id: int
    timestamp_ns: int
    joints_zup: np.ndarray
    raw_targets: dict[str, ReferenceTarget]
    scaled_targets: dict[str, ReferenceTarget]


class SMPLXGMRReference:
    """Convert GENMO FK output to original-GMR SMPL-X target semantics."""

    def __init__(self, *, user_yaw_deg: float = 0.0, global_scale: float = 1.0):
        if not math.isfinite(user_yaw_deg):
            raise ValueError("user_yaw_deg must be finite")
        if not math.isfinite(global_scale) or global_scale <= 0.0:
            raise ValueError("global_scale must be finite and > 0")
        angle = math.radians(float(user_yaw_deg))
        cosine, sine = math.cos(angle), math.sin(angle)
        self._world_rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ) @ AXIS_CONVERT_AY_TO_ZUP
        self.global_scale = float(global_scale)
        self.reset()

    def reset(self) -> None:
        self._origin: np.ndarray | None = None

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
    def _validate_rotations(rotations: np.ndarray) -> None:
        identity = np.eye(3, dtype=np.float64)
        gram = np.einsum("nji,njk->nik", rotations, rotations)
        determinants = np.linalg.det(rotations)
        valid = np.all(
            np.isclose(gram, identity, atol=1e-4, rtol=1e-4), axis=(1, 2)
        ) & np.isclose(determinants, 1.0, atol=1e-4, rtol=1e-4)
        if not np.all(valid):
            index = int(np.flatnonzero(~valid)[0])
            raise ValueError(f"rotations_ay[{index}] is not a proper SO(3) matrix")

    def adapt(
        self,
        joints_ay: Any,
        rotations_ay: Any,
        *,
        frame_id: int = 0,
        timestamp_ns: int | None = None,
    ) -> SMPLXReferenceFrame:
        joints = self._numpy(joints_ay, (22, 3), "joints_ay")
        rotations = self._numpy(rotations_ay, (22, 3, 3), "rotations_ay")
        self._validate_rotations(rotations)

        # Global rotations change world coordinates on the left only.  A
        # conjugation would incorrectly replace the SMPL-X local joint basis.
        joints_zup = np.einsum("ij,nj->ni", self._world_rotation, joints)
        rotations_zup = np.einsum("ij,njk->nik", self._world_rotation, rotations)
        self._validate_rotations(rotations_zup)

        if self._origin is None:
            ground_z = min(float(joints_zup[10, 2]), float(joints_zup[11, 2]))
            self._origin = np.asarray(
                (joints_zup[0, 0], joints_zup[0, 1], ground_z), dtype=np.float64
            )
        normalized_joints = (joints_zup - self._origin) * self.global_scale

        raw_targets: dict[str, ReferenceTarget] = {}
        normalized_targets: dict[str, ReferenceTarget] = {}
        for name in TARGET_NAMES:
            index = TARGET_JOINT_INDICES[name]
            rotation = rotations_zup[index].copy()
            raw_targets[name] = ReferenceTarget(joints_zup[index].copy(), rotation)
            normalized_targets[name] = ReferenceTarget(
                normalized_joints[index].copy(), rotation.copy()
            )

        stamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        return SMPLXReferenceFrame(
            frame_id=int(frame_id),
            timestamp_ns=stamp,
            joints_zup=normalized_joints,
            raw_targets=raw_targets,
            scaled_targets=normalized_targets,
        )
