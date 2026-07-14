# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Send GEM FK joints to GMR-CPP using the fixed-size GEM1 UDP protocol."""

from __future__ import annotations

import math
import socket
import struct
import time

import numpy as np
import torch

BONE_NAMES = (
    "Pelvis",
    "Chest",
    "Left_UpperLeg",
    "Right_UpperLeg",
    "Left_LowerLeg",
    "Right_LowerLeg",
    "Left_Foot",
    "Right_Foot",
    "Left_UpperArm",
    "Right_UpperArm",
    "Left_Forearm",
    "Right_Forearm",
    "Left_Hand",
    "Right_Hand",
)

JOINT_INDICES = (0, 9, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21)

MAGIC = b"GEM1"
VERSION = 1
HEADER = struct.Struct("<4sHHIQ")
PAYLOAD = struct.Struct("<" + "f" * (len(BONE_NAMES) * 7))
PACKET_BYTES = HEADER.size + PAYLOAD.size
MAX_ABS_POSITION_M = 20.0

if PACKET_BYTES != 412:  # Guard protocol changes at import time.
    raise RuntimeError(f"unexpected GEM1 packet size: {PACKET_BYTES}")


def _rot_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a finite 3x3 rotation matrix to a normalized wxyz quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))

    if trace > 0.0:
        s = math.sqrt(max(trace + 1.0, 1e-12)) * 2.0
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s)

    quat = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid rotation matrix produced a degenerate quaternion")
    return (quat / norm).astype(np.float32)


class GMRUDPBridge:
    """Coordinate normalization and one-packet-per-inference GEM1 sender.

    FK is intentionally not performed here. ``send_fk`` consumes the 22 joints
    and world rotations already produced by ``EnDecoder.fk_v2``.
    """

    def __init__(
        self,
        host: str,
        port: int = 7001,
        yaw_deg: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        if not host:
            raise ValueError("host must be non-empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and > 0")
        if not math.isfinite(yaw_deg):
            raise ValueError("yaw_deg must be finite")

        self.host = host
        self.port = int(port)
        self.scale = float(scale)
        self.sequence = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # GEM rollout is AY/Y-up. GMR and MuJoCo are Z-up.
        self._axis_convert = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        )
        self._user_yaw = _rot_z(math.radians(float(yaw_deg)))
        self._yaw_inv: np.ndarray | None = None
        self._origin: np.ndarray | None = None

        self._rate_started = time.monotonic()
        self._rate_packets = 0

        print(
            f"[GMR UDP] enabled: {self.host}:{self.port}, "
            f"yaw={yaw_deg:.1f} deg, scale={self.scale:.3f}, "
            f"packet={PACKET_BYTES} bytes"
        )

    @staticmethod
    def _as_numpy(value: torch.Tensor, shape: tuple[int, ...], name: str) -> np.ndarray:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(value.shape)}")
        array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return array

    def reset_origin(self) -> None:
        """Re-capture initial yaw, horizontal origin, and foot ground height."""
        self._yaw_inv = None
        self._origin = None

    @torch.no_grad()
    def send_fk(
        self,
        joints_ay: torch.Tensor,
        rotations_ay: torch.Tensor,
        source_stamp_ns: int | None = None,
    ) -> None:
        joints = self._as_numpy(joints_ay, (22, 3), "joints_ay")
        rotations = self._as_numpy(rotations_ay, (22, 3, 3), "rotations_ay")

        if np.max(np.abs(joints)) > MAX_ABS_POSITION_M:
            raise ValueError(
                f"joints_ay exceeds {MAX_ABS_POSITION_M:.0f} m safety limit"
            )

        pelvis_zup_rotation = self._axis_convert @ rotations[0]
        if self._yaw_inv is None:
            yaw = math.atan2(
                float(pelvis_zup_rotation[1, 0]),
                float(pelvis_zup_rotation[0, 0]),
            )
            self._yaw_inv = _rot_z(-yaw)

        # User yaw must be applied after initial-yaw cancellation.
        total_rotation = self._user_yaw @ self._yaw_inv @ self._axis_convert
        positions_out = np.einsum("ij,nj->ni", total_rotation, joints)
        rotations_out = np.einsum("ij,njk->nik", total_rotation, rotations)

        if self._origin is None:
            ground_z = min(
                float(positions_out[JOINT_INDICES[6], 2]),
                float(positions_out[JOINT_INDICES[7], 2]),
            )
            self._origin = np.asarray(
                (positions_out[0, 0], positions_out[0, 1], ground_z),
                dtype=np.float64,
            )

        positions_out = (positions_out - self._origin) * self.scale
        if not np.isfinite(positions_out).all() or not np.isfinite(rotations_out).all():
            raise ValueError("coordinate conversion produced NaN or Inf")
        if np.max(np.abs(positions_out)) > MAX_ABS_POSITION_M:
            raise ValueError(
                f"normalized positions exceed {MAX_ABS_POSITION_M:.0f} m safety limit"
            )

        values: list[float] = []
        for joint_index in JOINT_INDICES:
            position = positions_out[joint_index]
            quat = _matrix_to_quaternion_wxyz(rotations_out[joint_index])
            values.extend((*position.tolist(), *quat.tolist()))

        stamp_ns = time.monotonic_ns() if source_stamp_ns is None else int(source_stamp_ns)
        if stamp_ns < 0 or stamp_ns > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("source_stamp_ns must fit uint64")

        packet = HEADER.pack(
            MAGIC,
            VERSION,
            len(BONE_NAMES),
            self.sequence & 0xFFFFFFFF,
            stamp_ns,
        ) + PAYLOAD.pack(*values)
        if len(packet) != PACKET_BYTES:
            raise RuntimeError(f"unexpected GEM1 packet size: {len(packet)}")

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
            self._rate_started = now
            self._rate_packets = 0

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
