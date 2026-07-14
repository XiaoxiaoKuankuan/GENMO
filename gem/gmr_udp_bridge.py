# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Send real-time GEM-SMPL poses to GMR-CPP through a fixed-size UDP packet."""

from __future__ import annotations

import math
import socket
import struct
import time
from typing import Dict

import numpy as np
import torch


_BONE_NAMES = (
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

_JOINT_INDICES = (0, 9, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21)

_HEADER = struct.Struct("<4sHHIQ")
_PAYLOAD = struct.Struct("<" + "f" * (len(_BONE_NAMES) * 7))
_MAGIC = b"GEM1"
_VERSION = 1


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues conversion for (..., 3) axis-angle tensors."""
    eps = torch.finfo(axis_angle.dtype).eps
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / theta.clamp_min(eps)

    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        (
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ),
        dim=-1,
    ).reshape(axis.shape[:-1] + (3, 3))

    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(axis.shape[:-1] + (3, 3))
    sin_theta = torch.sin(theta)[..., None]
    cos_theta = torch.cos(theta)[..., None]
    result = eye + sin_theta * k + (1.0 - cos_theta) * (k @ k)

    near_zero = (theta[..., 0] < 1e-8)[..., None, None]
    return torch.where(near_zero, eye, result)


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


class GMRUDPBridge:
    """Lazy SMPL-X FK plus UDP sender for the GMR-CPP GemReader."""

    def __init__(
        self,
        host: str,
        port: int = 7001,
        yaw_deg: float = 0.0,
        scale: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.scale = float(scale)
        self.device = torch.device(device)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sequence = 0
        self.body_model = None
        self.parents = None
        self.initial_xy = None
        self.initial_ground = None

        yaw = math.radians(float(yaw_deg))
        self.rz = np.array(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )

        self.axis_convert = np.array(
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=np.float32,
        )

        print(
            f"[GMR UDP] enabled: {self.host}:{self.port}, "
            f"yaw={yaw_deg:.1f} deg, scale={self.scale:.3f}"
        )

    def _ensure_model(self) -> None:
        if self.body_model is not None:
            return

        from gem.utils.smplx_utils import make_smplx

        self.body_model = make_smplx("supermotion").to(self.device).eval()
        parents = self.body_model.parents[:22]
        self.parents = [int(x) for x in parents.detach().cpu().tolist()]
        print(f"[GMR UDP] SMPL-X FK model loaded on {self.device}")

    def _prepare_params(
        self,
        body_params_global: Dict[str, torch.Tensor],
        body_params_incam: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        params: Dict[str, torch.Tensor] = {}

        for key, value in body_params_global.items():
            params[key] = value.to(self.device)

        for key in ("body_pose", "betas"):
            if key not in params and key in body_params_incam:
                params[key] = body_params_incam[key].to(self.device)

        if "global_orient" not in params or "transl" not in params:
            raise KeyError("GEM global result must contain global_orient and transl")
        if "body_pose" not in params:
            raise KeyError("GEM in-camera result must contain body_pose")

        return params

    @torch.no_grad()
    def send(
        self,
        body_params_global: Dict[str, torch.Tensor],
        body_params_incam: Dict[str, torch.Tensor],
    ) -> None:
        self._ensure_model()
        params = self._prepare_params(body_params_global, body_params_incam)

        output = self.body_model(**params)
        joints = output.joints[0, :22].detach().cpu().numpy().astype(np.float32)

        global_orient = params["global_orient"].reshape(-1, 3)[0]
        body_pose = params["body_pose"].reshape(-1, 3)[:21]
        local_axis_angle = torch.cat((global_orient[None], body_pose), dim=0)
        local_rot = _axis_angle_to_matrix(local_axis_angle)

        global_rot = []
        for joint_idx in range(22):
            parent = self.parents[joint_idx]
            if joint_idx == 0 or parent < 0:
                current = local_rot[joint_idx]
            else:
                current = global_rot[parent] @ local_rot[joint_idx]
            global_rot.append(current)
        global_rot_t = torch.stack(global_rot, dim=0)
        rotations = global_rot_t.detach().cpu().numpy().astype(np.float32)

        positions = (self.axis_convert @ joints.T).T
        rotations = np.einsum("ij,njk->nik", self.axis_convert, rotations)

        if self.initial_xy is None:
            self.initial_xy = positions[0, :2].copy()
            self.initial_ground = float(min(positions[7, 2], positions[8, 2]))

        origin = np.array(
            [self.initial_xy[0], self.initial_xy[1], self.initial_ground],
            dtype=np.float32,
        )
        positions = (positions - origin) * self.scale
        positions = (self.rz @ positions.T).T
        rotations = np.einsum("ij,njk->nik", self.rz, rotations)

        values = []
        for joint_idx in _JOINT_INDICES:
            position = positions[joint_idx]
            quaternion = _matrix_to_quaternion_wxyz(rotations[joint_idx])
            values.extend(
                (
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                )
            )

        values_np = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values_np).all():
            raise FloatingPointError("GEM produced non-finite GMR bridge values")

        header = _HEADER.pack(
            _MAGIC,
            _VERSION,
            len(_BONE_NAMES),
            self.sequence & 0xFFFFFFFF,
            time.time_ns(),
        )
        packet = header + _PAYLOAD.pack(*values)
        self.sock.sendto(packet, (self.host, self.port))
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
