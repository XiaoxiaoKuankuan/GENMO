# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Send the standard SMPL-X SMP1 target packet to GMR-CPP."""

from __future__ import annotations

import math
import os
import socket
import struct
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

SMPLX_TARGET_NAMES = (
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

SMPLX_MAGIC = b"SMP1"
SMPLX_VERSION = 1
HEADER = struct.Struct("<4sHHIQ")
PAYLOAD = struct.Struct("<" + "f" * (len(SMPLX_TARGET_NAMES) * 7))
PACKET_BYTES = HEADER.size + PAYLOAD.size
MAX_ABS_POSITION_M = 20.0

if PACKET_BYTES != 412:
    raise RuntimeError(f"unexpected SMP1 packet size: {PACKET_BYTES}")


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a finite 3x3 rotation matrix to a normalized wxyz quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(max(trace + 1.0, 1e-12)) * 2.0
        values = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        values = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        values = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        values = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid rotation matrix produced a degenerate quaternion")
    return (quaternion / norm).astype(np.float32)


class GMRUDPBridge:
    """Pack and send one fixed-size SMP1 datagram per GEM inference frame."""

    def __init__(
        self,
        host: str,
        port: int = 7005,
        debug: bool | None = None,
    ) -> None:
        if not host:
            raise ValueError("host must be non-empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be in [1, 65535]")
        self.host = host
        self.port = int(port)
        self.debug = (
            os.environ.get("GMR_UDP_DEBUG", "").strip().lower()
            in {"1", "true", "yes", "on"}
            if debug is None
            else bool(debug)
        )
        self.sequence = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._previous_quaternions: dict[str, np.ndarray] = {}
        self._rate_started = time.monotonic()
        self._rate_packets = 0
        print(f"[GMR UDP] SMP1 enabled: {self.host}:{self.port}, packet={PACKET_BYTES} bytes")

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        else:
            array = np.asarray(value, dtype=np.float64)
        if array.shape != shape:
            raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return array

    @staticmethod
    def _validate_rotation(rotation: np.ndarray, name: str) -> None:
        gram = rotation.T @ rotation
        determinant = float(np.linalg.det(rotation))
        if not np.allclose(gram, np.eye(3), atol=1e-4, rtol=1e-4) or not np.isclose(
            determinant, 1.0, atol=1e-4, rtol=1e-4
        ):
            raise ValueError(f"{name} is not a proper rotation (det={determinant:.6g})")

    @torch.no_grad()
    def send_smplx_targets(
        self,
        targets: Mapping[str, Any],
        source_stamp_ns: int | None = None,
    ) -> bytes:
        missing = set(SMPLX_TARGET_NAMES) - set(targets)
        extra = set(targets) - set(SMPLX_TARGET_NAMES)
        if missing or extra:
            raise ValueError(
                f"SMPL-X targets names mismatch; missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )

        values: list[float] = []
        positions: dict[str, np.ndarray] = {}
        for name in SMPLX_TARGET_NAMES:
            pose = targets[name]
            if isinstance(pose, Mapping):
                position_value = pose["position_zup"]
                rotation_value = pose["rotation_zup"]
            else:
                position_value = pose.position_zup
                rotation_value = pose.rotation_zup
            position = self._array(position_value, (3,), f"{name}.position_zup")
            rotation = self._array(rotation_value, (3, 3), f"{name}.rotation_zup")
            self._validate_rotation(rotation, f"{name}.rotation_zup")
            if np.max(np.abs(position)) > MAX_ABS_POSITION_M:
                raise ValueError(f"{name}.position_zup exceeds safety limit")
            quaternion = _matrix_to_quaternion_wxyz(rotation)
            previous = self._previous_quaternions.get(name)
            if previous is not None and float(np.dot(quaternion, previous)) < 0.0:
                quaternion = -quaternion
            self._previous_quaternions[name] = quaternion.copy()
            positions[name] = position
            values.extend((*position.tolist(), *quaternion.tolist()))

        stamp_ns = time.monotonic_ns() if source_stamp_ns is None else int(source_stamp_ns)
        if stamp_ns < 0 or stamp_ns > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("source_stamp_ns must fit uint64")
        packet = HEADER.pack(
            SMPLX_MAGIC,
            SMPLX_VERSION,
            len(SMPLX_TARGET_NAMES),
            self.sequence & 0xFFFFFFFF,
            stamp_ns,
        ) + PAYLOAD.pack(*values)
        self.sock.sendto(packet, (self.host, self.port))
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        self._rate_packets += 1

        now = time.monotonic()
        elapsed = now - self._rate_started
        if elapsed >= 5.0:
            print(
                f"[GMR UDP] send={self._rate_packets / elapsed:.1f}Hz "
                f"sequence={self.sequence} packet={len(packet)} bytes"
            )
            if self.debug:
                print(
                    "[GMR UDP debug] "
                    f"pelvis_z={positions['pelvis'][2]:.4f} "
                    f"left_foot_z={positions['left_foot'][2]:.4f} "
                    f"right_foot_z={positions['right_foot'][2]:.4f}"
                )
            self._rate_started = now
            self._rate_packets = 0
        return packet

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
