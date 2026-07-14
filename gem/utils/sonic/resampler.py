# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Online fixed-rate interpolation for SONIC SMPL streaming."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _single_frame(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (1, *shape):
        array = array[0]
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} or {(1, *shape)}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _normalize_quaternion(quaternion: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8) or not np.isfinite(norm).all():
        raise ValueError(f"{name} contains a zero or invalid quaternion")
    return quaternion / norm


def _slerp_shortest(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized shortest-path quaternion SLERP for either component order."""
    q0 = _normalize_quaternion(np.asarray(q0, dtype=np.float64), "q0")
    q1 = _normalize_quaternion(np.asarray(q1, dtype=np.float64), "q1")

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe_denominator = np.where(np.abs(sin_theta) < 1e-8, 1.0, sin_theta)
    s0 = np.sin((1.0 - alpha) * theta) / safe_denominator
    s1 = np.sin(alpha * theta) / safe_denominator
    spherical = s0 * q0 + s1 * q1
    linear = (1.0 - alpha) * q0 + alpha * q1
    result = np.where(dot > 0.9995, linear, spherical)
    return _normalize_quaternion(result, "interpolated quaternion")


def _interpolate_pose_axis_angle(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    previous_quat = Rotation.from_rotvec(previous.reshape(-1, 3)).as_quat()
    current_quat = Rotation.from_rotvec(current.reshape(-1, 3)).as_quat()
    interpolated_quat = _slerp_shortest(previous_quat, current_quat, alpha)
    return Rotation.from_quat(interpolated_quat).as_rotvec().reshape(21, 3)


class SMPLRealtimeResampler:
    """Resample timestamped single-frame SMPL data onto a fixed-rate grid.

    The first sample is emitted immediately and anchors the output grid. Later
    calls may emit zero or multiple frames. Timestamp discontinuities and gaps
    longer than ``max_gap_seconds`` restart the grid instead of synthesizing a
    large burst of stale motion.
    """

    def __init__(
        self,
        target_fps: int = 50,
        max_gap_seconds: float = 0.5,
        max_output_frames: int = 1000,
    ) -> None:
        if isinstance(target_fps, bool) or not isinstance(target_fps, Integral):
            raise TypeError("target_fps must be an integer")
        if target_fps <= 0 or target_fps > 1_000_000_000:
            raise ValueError("target_fps must be in [1, 1000000000]")
        if (
            isinstance(max_gap_seconds, bool)
            or not isinstance(max_gap_seconds, Real)
            or not math.isfinite(max_gap_seconds)
            or max_gap_seconds <= 0
        ):
            raise ValueError("max_gap_seconds must be positive")
        if isinstance(max_output_frames, bool) or not isinstance(max_output_frames, Integral):
            raise TypeError("max_output_frames must be an integer")
        if max_output_frames <= 0:
            raise ValueError("max_output_frames must be positive")

        self.target_fps = int(target_fps)
        self.period_ns = int(round(1_000_000_000 / self.target_fps))
        self.max_gap_ns = int(round(max_gap_seconds * 1_000_000_000))
        self.max_output_frames = int(max_output_frames)
        self.reset()

    def reset(self) -> None:
        self._previous_timestamp_ns: int | None = None
        self._next_timestamp_ns: int | None = None
        self._previous_pose: np.ndarray | None = None
        self._previous_joints: np.ndarray | None = None
        self._previous_body_quat: np.ndarray | None = None

    def _empty(self, discontinuity: bool = False) -> dict[str, np.ndarray | bool]:
        return {
            "timestamp_ns": np.empty((0,), dtype=np.int64),
            "smpl_pose": np.empty((0, 21, 3), dtype=np.float32),
            "smpl_joints_local": np.empty((0, 24, 3), dtype=np.float32),
            "body_quat": np.empty((0, 4), dtype=np.float32),
            "discontinuity": discontinuity,
        }

    def _store_endpoint(
        self,
        timestamp_ns: int,
        smpl_pose: np.ndarray,
        smpl_joints_local: np.ndarray,
        body_quat: np.ndarray,
    ) -> None:
        self._previous_timestamp_ns = timestamp_ns
        self._previous_pose = smpl_pose.copy()
        self._previous_joints = smpl_joints_local.copy()
        self._previous_body_quat = body_quat.copy()

    def _restart(
        self,
        timestamp_ns: int,
        smpl_pose: np.ndarray,
        smpl_joints_local: np.ndarray,
        body_quat: np.ndarray,
        *,
        discontinuity: bool,
    ) -> dict[str, np.ndarray | bool]:
        self.reset()
        self._store_endpoint(timestamp_ns, smpl_pose, smpl_joints_local, body_quat)
        self._next_timestamp_ns = timestamp_ns + self.period_ns
        return {
            "timestamp_ns": np.array([timestamp_ns], dtype=np.int64),
            "smpl_pose": smpl_pose[None].copy(),
            "smpl_joints_local": smpl_joints_local[None].copy(),
            "body_quat": body_quat[None].copy(),
            "discontinuity": discontinuity,
        }

    def push(
        self,
        timestamp_ns: int,
        smpl_pose: Any,
        smpl_joints_local: Any,
        body_quat: Any,
    ) -> dict[str, np.ndarray | bool]:
        """Add one source frame and return all newly available target frames."""
        if isinstance(timestamp_ns, (bool, np.bool_)) or not isinstance(timestamp_ns, Integral):
            raise TypeError("timestamp_ns must be an integer nanosecond timestamp")
        timestamp_ns = int(timestamp_ns)

        pose = _single_frame(smpl_pose, (21, 3), "smpl_pose")
        joints = _single_frame(smpl_joints_local, (24, 3), "smpl_joints_local")
        quaternion = _single_frame(body_quat, (4,), "body_quat")
        quaternion = np.ascontiguousarray(
            _normalize_quaternion(quaternion, "body_quat"), dtype=np.float32
        )

        if self._previous_timestamp_ns is None:
            return self._restart(
                timestamp_ns,
                pose,
                joints,
                quaternion,
                discontinuity=False,
            )

        if timestamp_ns == self._previous_timestamp_ns:
            return self._empty()

        gap_ns = timestamp_ns - self._previous_timestamp_ns
        if gap_ns < 0 or gap_ns > self.max_gap_ns:
            return self._restart(
                timestamp_ns,
                pose,
                joints,
                quaternion,
                discontinuity=True,
            )

        assert self._next_timestamp_ns is not None
        assert self._previous_pose is not None
        assert self._previous_joints is not None
        assert self._previous_body_quat is not None
        if float(np.dot(self._previous_body_quat, quaternion)) < 0.0:
            quaternion = -quaternion

        target_timestamps = []
        target = self._next_timestamp_ns
        available_frames = (
            (timestamp_ns - target) // self.period_ns + 1 if target <= timestamp_ns else 0
        )
        if available_frames > self.max_output_frames:
            return self._restart(
                timestamp_ns,
                pose,
                joints,
                quaternion,
                discontinuity=True,
            )
        while target <= timestamp_ns:
            target_timestamps.append(target)
            target += self.period_ns

        if not target_timestamps:
            self._store_endpoint(timestamp_ns, pose, joints, quaternion)
            return self._empty()

        denominator = float(timestamp_ns - self._previous_timestamp_ns)
        output_pose = []
        output_joints = []
        output_body_quat = []
        for target_timestamp in target_timestamps:
            alpha = float(target_timestamp - self._previous_timestamp_ns) / denominator
            alpha = min(max(alpha, 0.0), 1.0)
            output_joints.append((1.0 - alpha) * self._previous_joints + alpha * joints)
            output_pose.append(_interpolate_pose_axis_angle(self._previous_pose, pose, alpha))
            output_body_quat.append(_slerp_shortest(self._previous_body_quat, quaternion, alpha))

        self._next_timestamp_ns = target
        self._store_endpoint(timestamp_ns, pose, joints, quaternion)
        return {
            "timestamp_ns": np.asarray(target_timestamps, dtype=np.int64),
            "smpl_pose": np.ascontiguousarray(output_pose, dtype=np.float32),
            "smpl_joints_local": np.ascontiguousarray(output_joints, dtype=np.float32),
            "body_quat": np.ascontiguousarray(output_body_quat, dtype=np.float32),
            "discontinuity": False,
        }
