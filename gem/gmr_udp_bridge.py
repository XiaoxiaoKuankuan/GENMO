# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Send legacy GEM1 or explicit SMPL-direct GEM2 targets to GMR-CPP."""

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

SMPL_TARGET_NAMES = (
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

JOINT_INDICES = (0, 9, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21)
GROUND_JOINT_INDICES = (10, 11)

MAGIC = b"GEM1"
VERSION = 1
GEM2_MAGIC = b"GEM2"
GEM2_VERSION = 2
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
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)

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
        debug: bool | None = None,
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
        self.debug = (
            os.environ.get("GMR_UDP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
            if debug is None
            else bool(debug)
        )
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
        self._previous_smpl_quaternions: dict[str, np.ndarray] = {}

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
        self._previous_smpl_quaternions.clear()

    @staticmethod
    def _segment_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
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
    def _validate_rotations(rotations: np.ndarray, name: str) -> None:
        """Reject matrices that are not proper rotations in SO(3)."""
        identity = np.eye(3, dtype=np.float64)
        gram = np.einsum("nji,njk->nik", rotations, rotations)
        orthogonal = np.all(np.isclose(gram, identity, atol=1e-4, rtol=1e-4), axis=(1, 2))
        determinants = np.linalg.det(rotations)
        proper = np.isclose(determinants, 1.0, atol=1e-4, rtol=1e-4)
        invalid = np.flatnonzero(~(orthogonal & proper))
        if invalid.size:
            index = int(invalid[0])
            error = float(np.max(np.abs(gram[index] - identity)))
            raise ValueError(
                f"{name}[{index}] is not a proper rotation "
                f"(orthogonality_error={error:.3g}, det={determinants[index]:.6g})"
            )

    def _pack_and_send(
        self,
        values: list[float],
        source_stamp_ns: int | None,
        debug_details: str | None = None,
        *,
        magic: bytes = MAGIC,
        version: int = VERSION,
        item_count: int = len(BONE_NAMES),
    ) -> bytes:
        stamp_ns = time.monotonic_ns() if source_stamp_ns is None else int(source_stamp_ns)
        if stamp_ns < 0 or stamp_ns > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("source_stamp_ns must fit uint64")

        packet = HEADER.pack(
            magic,
            version,
            item_count,
            self.sequence & 0xFFFFFFFF,
            stamp_ns,
        ) + PAYLOAD.pack(*values)
        if len(packet) != PACKET_BYTES:
            raise RuntimeError(f"unexpected {magic.decode()} packet size: {len(packet)}")

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
            if self.debug and debug_details:
                print(f"[GMR UDP debug] {debug_details}")
            self._rate_started = now
            self._rate_packets = 0
        return packet

    @torch.no_grad()
    def send_segments(
        self,
        segment_poses: Mapping[str, Any],
        source_stamp_ns: int | None = None,
    ) -> bytes:
        """Pack already-adapted Z-up anatomical segment poses without guessing frames."""
        missing = set(BONE_NAMES) - set(segment_poses)
        extra = set(segment_poses) - set(BONE_NAMES)
        if missing or extra:
            raise ValueError(
                f"segment_poses names mismatch; missing={sorted(missing)} extra={sorted(extra)}"
            )

        values: list[float] = []
        positions: list[np.ndarray] = []
        quaternions: list[np.ndarray] = []
        for name in BONE_NAMES:
            pose = segment_poses[name]
            if isinstance(pose, Mapping):
                position_value = pose["position_zup"]
                rotation_value = pose["rotation_zup"]
            else:
                position_value = pose.position_zup
                rotation_value = pose.rotation_zup
            position = self._segment_array(position_value, (3,), f"{name}.position_zup")
            rotation = self._segment_array(rotation_value, (3, 3), f"{name}.rotation_zup")
            self._validate_rotations(rotation[None], f"{name}.rotation_zup")
            if np.max(np.abs(position)) > MAX_ABS_POSITION_M:
                raise ValueError(f"{name}.position_zup exceeds safety limit")
            quaternion = _matrix_to_quaternion_wxyz(rotation)
            positions.append(position)
            quaternions.append(quaternion)
            values.extend((*position.tolist(), *quaternion.tolist()))

        debug_details = None
        if self.debug:
            debug_details = (
                f"segment_z pelvis:{positions[0][2]:.4f} "
                f"left_foot:{positions[6][2]:.4f} right_foot:{positions[7][2]:.4f}; "
                f"q pelvis:{quaternions[0].tolist()} "
                f"left_upper_arm:{quaternions[8].tolist()} "
                f"right_upper_arm:{quaternions[9].tolist()}"
            )
        return self._pack_and_send(values, source_stamp_ns, debug_details)

    @torch.no_grad()
    def send_smpl_targets(
        self,
        targets: Mapping[str, Any],
        source_stamp_ns: int | None = None,
    ) -> bytes:
        """Pack 14 explicit SMPL joint-center/anatomical targets as GEM2."""
        missing = set(SMPL_TARGET_NAMES) - set(targets)
        extra = set(targets) - set(SMPL_TARGET_NAMES)
        if missing or extra:
            raise ValueError(
                f"SMPL targets names mismatch; missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )

        values: list[float] = []
        quaternions: dict[str, np.ndarray] = {}
        positions: dict[str, np.ndarray] = {}
        for name in SMPL_TARGET_NAMES:
            pose = targets[name]
            if isinstance(pose, Mapping):
                position_value = pose["position_zup"]
                rotation_value = pose["rotation_zup"]
            else:
                position_value = pose.position_zup
                rotation_value = pose.rotation_zup
            position = self._segment_array(
                position_value, (3,), f"{name}.position_zup"
            )
            rotation = self._segment_array(
                rotation_value, (3, 3), f"{name}.rotation_zup"
            )
            self._validate_rotations(rotation[None], f"{name}.rotation_zup")
            if np.max(np.abs(position)) > MAX_ABS_POSITION_M:
                raise ValueError(f"{name}.position_zup exceeds safety limit")
            quaternion = _matrix_to_quaternion_wxyz(rotation)
            previous = self._previous_smpl_quaternions.get(name)
            if previous is not None and float(np.dot(quaternion, previous)) < 0.0:
                quaternion = -quaternion
            self._previous_smpl_quaternions[name] = quaternion.copy()
            positions[name] = position
            quaternions[name] = quaternion
            values.extend((*position.tolist(), *quaternion.tolist()))

        debug_details = None
        if self.debug:
            debug_details = (
                "GEM2 "
                f"pelvis_z={positions['SMPL_Pelvis'][2]:.4f} "
                f"left_ankle_z={positions['SMPL_LeftAnkle'][2]:.4f} "
                f"right_ankle_z={positions['SMPL_RightAnkle'][2]:.4f} "
                f"pelvis_q={quaternions['SMPL_Pelvis'].tolist()}"
            )
        return self._pack_and_send(
            values,
            source_stamp_ns,
            debug_details,
            magic=GEM2_MAGIC,
            version=GEM2_VERSION,
            item_count=len(SMPL_TARGET_NAMES),
        )

    @torch.no_grad()
    def send_fk(
        self,
        joints_ay: torch.Tensor,
        rotations_ay: torch.Tensor,
        source_stamp_ns: int | None = None,
    ) -> bytes:
        joints = self._as_numpy(joints_ay, (22, 3), "joints_ay")
        rotations = self._as_numpy(rotations_ay, (22, 3, 3), "rotations_ay")

        if np.max(np.abs(joints)) > MAX_ABS_POSITION_M:
            raise ValueError(f"joints_ay exceeds {MAX_ABS_POSITION_M:.0f} m safety limit")
        self._validate_rotations(rotations, "rotations_ay")

        # Local joint frames need a change of basis on both sides. In contrast,
        # world-space positions are vectors and only need the left transform.
        rotations_zup = np.einsum(
            "ij,njk,kl->nil",
            self._axis_convert,
            rotations,
            self._axis_convert.T,
        )
        if self._yaw_inv is None:
            yaw = math.atan2(
                float(rotations_zup[0, 1, 0]),
                float(rotations_zup[0, 0, 0]),
            )
            self._yaw_inv = _rot_z(-yaw)

        # User yaw must be applied after initial-yaw cancellation.
        world_yaw = self._user_yaw @ self._yaw_inv
        positions_out = np.einsum("ij,nj->ni", world_yaw @ self._axis_convert, joints)
        rotations_out = np.einsum("ij,njk->nik", world_yaw, rotations_zup)
        self._validate_rotations(rotations_out, "rotations_out")

        if self._origin is None:
            ground_z = min(
                float(positions_out[GROUND_JOINT_INDICES[0], 2]),
                float(positions_out[GROUND_JOINT_INDICES[1], 2]),
            )
            self._origin = np.asarray(
                (positions_out[0, 0], positions_out[0, 1], ground_z),
                dtype=np.float64,
            )

        positions_out = (positions_out - self._origin) * self.scale
        if not np.isfinite(positions_out).all() or not np.isfinite(rotations_out).all():
            raise ValueError("coordinate conversion produced NaN or Inf")
        if np.max(np.abs(positions_out)) > MAX_ABS_POSITION_M:
            raise ValueError(f"normalized positions exceed {MAX_ABS_POSITION_M:.0f} m safety limit")

        values: list[float] = []
        selected_quaternions: list[np.ndarray] = []
        for joint_index in JOINT_INDICES:
            position = positions_out[joint_index]
            quat = _matrix_to_quaternion_wxyz(rotations_out[joint_index])
            selected_quaternions.append(quat)
            values.extend((*position.tolist(), *quat.tolist()))

        debug_details = None
        if self.debug:
            selected_positions = positions_out[np.asarray(JOINT_INDICES)]
            z_values = selected_positions[[0, 6, 7], 2]
            foot_z = positions_out[np.asarray(GROUND_JOINT_INDICES), 2]
            debug_details = (
                f"joint_z pelvis:{z_values[0]:.4f} "
                f"left_ankle:{z_values[1]:.4f} right_ankle:{z_values[2]:.4f} "
                f"left_foot:{foot_z[0]:.4f} right_foot:{foot_z[1]:.4f}; "
                f"q pelvis:{selected_quaternions[0].tolist()} "
                f"left_upper_arm:{selected_quaternions[8].tolist()} "
                f"right_upper_arm:{selected_quaternions[9].tolist()}"
            )
        return self._pack_and_send(values, source_stamp_ns, debug_details)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
