# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Convert GEM SMPL-X joint centers into anatomical segment poses for GMR."""

from __future__ import annotations

import copy
import json
import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - adapter math itself only needs NumPy.
    torch = None

SEGMENT_NAMES = (
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

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SegmentPose:
    """One Z-up anatomical rigid-segment pose."""

    position_zup: np.ndarray
    rotation_zup: np.ndarray


@dataclass(frozen=True)
class AdapterFrame:
    """Raw and calibrated segment data produced for one GEM inference."""

    frame_id: int
    timestamp_ns: int
    joints_zup: np.ndarray
    raw_segments: dict[str, SegmentPose]
    scaled_segments: dict[str, SegmentPose]
    ground_z: float

    def debug_dict(self, udp_payload: bytes | None = None) -> dict[str, Any]:
        def encode(poses: dict[str, SegmentPose]) -> dict[str, Any]:
            return {
                name: {
                    "position": pose.position_zup.tolist(),
                    "rotation": pose.rotation_zup.tolist(),
                }
                for name, pose in poses.items()
            }

        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "ground_z": self.ground_z,
            "raw_joints": self.joints_zup.tolist(),
            "raw_segments": encode(self.raw_segments),
            "scaled_segments": encode(self.scaled_segments),
            "udp_payload": list(udp_payload) if udp_payload is not None else [],
        }


class BetaStabilizer:
    """Freeze or smooth GEM betas before FK so segment lengths stop breathing."""

    MODES = {"first", "mean", "ema", "per_frame"}

    def __init__(self, mode: str = "mean", warmup: int = 30, ema_alpha: float = 0.05):
        if mode not in self.MODES:
            raise ValueError(f"shape mode must be one of {sorted(self.MODES)}")
        if warmup < 1:
            raise ValueError("shape warmup must be >= 1")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.mode = mode
        self.warmup = int(warmup)
        self.ema_alpha = float(ema_alpha)
        self.count = 0
        self._sum: np.ndarray | None = None
        self._value: np.ndarray | None = None
        self._frozen: np.ndarray | None = None

    @property
    def frozen(self) -> bool:
        return self.mode in {"first", "mean"} and self._frozen is not None

    def reset(self) -> None:
        self.count = 0
        self._sum = None
        self._value = None
        self._frozen = None

    def update(self, betas: Any) -> Any:
        is_tensor = torch is not None and isinstance(betas, torch.Tensor)
        array = (
            betas.detach().to(device="cpu", dtype=torch.float64).numpy()
            if is_tensor
            else np.asarray(betas, dtype=np.float64)
        )
        if not np.isfinite(array).all():
            raise ValueError("betas contains NaN or Inf")

        if self.mode == "per_frame":
            result = array.copy()
        elif self.mode == "first":
            if self._frozen is None:
                self._frozen = array.copy()
            result = self._frozen
        elif self.mode == "ema":
            if self._value is None:
                self._value = array.copy()
            else:
                self._value += self.ema_alpha * (array - self._value)
            result = self._value
        else:
            if self._frozen is None:
                self._sum = array.copy() if self._sum is None else self._sum + array
                self.count += 1
                result = self._sum / self.count
                if self.count >= self.warmup:
                    self._frozen = result.copy()
            else:
                result = self._frozen

        if self.mode != "mean":
            self.count += 1
        if is_tensor:
            return torch.as_tensor(result, dtype=betas.dtype, device=betas.device)
        return np.asarray(result, dtype=np.asarray(betas).dtype)


class SegmentDebugPublisher:
    """Best-effort JSON/UDP side channel consumed by debug_gem_gmr_segments.py."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7002):
        self.destination = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, frame: AdapterFrame, udp_payload: bytes) -> None:
        message = json.dumps(frame.debug_dict(udp_payload), separators=(",", ":")).encode()
        if len(message) > 65507:
            raise ValueError(f"segment debug datagram is too large: {len(message)} bytes")
        self.socket.sendto(message, self.destination)

    def close(self) -> None:
        self.socket.close()


class GMRSegmentAdapter:
    """Stateful joint-to-segment conversion with heading and contact locking."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        forward_sign: int | None = None,
        heading_source: str | None = None,
        ground_mode: str | None = None,
        user_yaw_deg: float = 0.0,
        global_scale: float = 1.0,
    ):
        self.config = copy.deepcopy(config)
        if tuple(self.config["segment_order"]) != SEGMENT_NAMES:
            raise ValueError("segment_order must contain the canonical 14 GMR names")
        self.forward_sign = int(
            self.config.get("forward_sign", -1) if forward_sign is None else forward_sign
        )
        if self.forward_sign not in {-1, 1}:
            raise ValueError("forward_sign must be +1 or -1")
        self.heading_source = heading_source or self.config.get("heading_source", "joints")
        if self.heading_source not in {"joints", "pelvis"}:
            raise ValueError("heading_source must be joints or pelvis")
        self.ground_mode = ground_mode or self.config.get("ground_mode", "contact")
        if self.ground_mode not in {"initial", "per_frame", "contact"}:
            raise ValueError("ground_mode must be initial, per_frame, or contact")
        if not math.isfinite(global_scale) or global_scale <= 0.0:
            raise ValueError("global_scale must be finite and > 0")

        self.global_scale = float(global_scale)
        self.root_translation_scale = float(self.config["root_translation_scale"])
        self.edge_scales = {
            str(name): float(value) for name, value in self.config["edge_scales"].items()
        }
        self.segments = self.config["segments"]
        self.hierarchy = [tuple(edge) for edge in self.config["hierarchy"]]
        self.contact = self.config.get("contact", {})
        self._axis_convert = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        )
        self._user_yaw = self._rot_z(math.radians(float(user_yaw_deg)))
        self.reset()

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> GMRSegmentAdapter:
        return cls(json.loads(Path(path).read_text()), **kwargs)

    def reset(self) -> None:
        self._heading_rotation: np.ndarray | None = None
        self._origin_xy: np.ndarray | None = None
        self._ground_z: float | None = None
        self._previous_feet: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._previous_frames: dict[str, np.ndarray] = {}
        self._previous_forward: np.ndarray | None = None
        self._previous_lateral: np.ndarray | None = None

    @staticmethod
    def _rot_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))

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
        return np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=1e-5) and np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5, rtol=1e-5
        )

    def _common_axes(
        self, joints: np.ndarray, pelvis_rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        up = np.asarray((0.0, 0.0, 1.0))
        lateral = self._normalize((joints[2] - joints[1]) + (joints[17] - joints[16]))
        if lateral is None:
            lateral = self._previous_lateral
        if lateral is None:
            lateral = -pelvis_rotation[:, 1]
        lateral = self._normalize(lateral)

        if self.heading_source == "pelvis":
            forward_candidate = pelvis_rotation[:, 0].copy()
            forward_candidate[2] = 0.0
            forward = self._normalize(forward_candidate)
        else:
            forward = self._normalize(self.forward_sign * np.cross(lateral, up))
        if forward is None:
            forward = self._previous_forward
        if forward is None:
            raise ValueError("cannot derive body forward axis")

        lateral = self._normalize(np.cross(up, forward)) * -1.0
        left = -lateral
        self._previous_forward = forward
        self._previous_lateral = lateral
        return forward, left, up

    def _update_ground(self, joints: np.ndarray, timestamp_ns: int) -> float:
        feet = joints[np.asarray((10, 11))]
        foot_z = feet[:, 2]
        if self._ground_z is None:
            self._ground_z = float(np.min(foot_z))
        elif self.ground_mode == "per_frame":
            self._ground_z = float(np.min(foot_z))
        elif self.ground_mode == "contact":
            dt = 1.0 / 30.0
            if self._previous_timestamp_ns is not None:
                dt = max((timestamp_ns - self._previous_timestamp_ns) * 1e-9, 1e-3)
            velocity = (
                np.zeros(2)
                if self._previous_feet is None
                else (foot_z - self._previous_feet[:, 2]) / dt
            )
            height_threshold = float(self.contact.get("height_threshold_m", 0.04))
            velocity_threshold = float(self.contact.get("velocity_threshold_mps", 0.20))
            contacting = (np.abs(foot_z - self._ground_z) <= height_threshold) & (
                np.abs(velocity) <= velocity_threshold
            )
            if np.any(contacting):
                candidate = float(np.min(foot_z[contacting]))
                alpha = float(self.contact.get("ground_ema_alpha", 0.2))
                self._ground_z += alpha * (candidate - self._ground_z)

        self._previous_feet = feet.copy()
        self._previous_timestamp_ns = timestamp_ns
        return self._ground_z

    def _fallback_rotation(self, rotations: np.ndarray, joint_index: int) -> np.ndarray:
        fallback = rotations[joint_index]
        if not self._proper_rotation(fallback):
            raise ValueError(f"fallback joint rotation {joint_index} is not proper SO(3)")
        return fallback.copy()

    def _make_frame(
        self,
        name: str,
        primary: np.ndarray,
        reference: np.ndarray,
        layout: list[str],
        fallback: np.ndarray,
    ) -> np.ndarray:
        previous = self._previous_frames.get(name)
        primary_unit = self._normalize(primary)
        if primary_unit is None:
            return previous.copy() if previous is not None else fallback

        reference_projected = reference - primary_unit * np.dot(reference, primary_unit)
        reference_unit = self._normalize(reference_projected)
        if reference_unit is None:
            return previous.copy() if previous is not None else fallback

        if sorted(layout) != ["x", "y", "z"]:
            raise ValueError(f"{name} axis_layout must be a permutation of x,y,z")
        primary_axis = AXIS_INDEX[layout[0]]
        reference_axis = AXIS_INDEX[layout[1]]
        remaining_axis = AXIS_INDEX[layout[2]]

        if previous is not None and np.dot(reference_unit, previous[:, reference_axis]) < 0.0:
            reference_unit = -reference_unit

        rotation = np.zeros((3, 3), dtype=np.float64)
        rotation[:, primary_axis] = primary_unit
        rotation[:, reference_axis] = reference_unit
        permutation = (primary_axis, reference_axis, remaining_axis)
        cyclic = permutation in {(0, 1, 2), (1, 2, 0), (2, 0, 1)}
        third = np.cross(primary_unit, reference_unit)
        rotation[:, remaining_axis] = third if cyclic else -third
        if not self._proper_rotation(rotation):
            raise ValueError(f"derived frame for {name} is not proper SO(3)")
        return rotation

    def _derive_segments(
        self,
        joints: np.ndarray,
        rotations: np.ndarray,
        forward: np.ndarray,
        left: np.ndarray,
        up: np.ndarray,
    ) -> dict[str, SegmentPose]:
        result: dict[str, SegmentPose] = {}
        references = {
            "pelvis_forward": forward,
            "chest_forward": forward,
            "world_up": up,
            "pelvis_lateral": left,
        }
        pelvis_fallback = self._fallback_rotation(rotations, 0)

        for name in SEGMENT_NAMES:
            definition = self.segments[name]
            origin_cfg = definition["origin"]
            proximal_index = int(origin_cfg["proximal_joint"])
            distal_index = int(origin_cfg["distal_joint"])
            alpha = float(origin_cfg["alpha"])
            proximal = joints[proximal_index]
            distal = joints[distal_index]
            origin = proximal + alpha * (distal - proximal)

            inherited = definition.get("inherit_frame")
            if inherited:
                if inherited not in result:
                    raise ValueError(f"{name} inherits unavailable frame {inherited}")
                rotation = result[inherited].rotation_zup.copy()
            else:
                primary_name = definition["primary_axis"]
                if primary_name == "proximal_to_distal":
                    primary = distal - proximal
                elif primary_name == "pelvis_to_chest":
                    primary = joints[9] - joints[0]
                elif primary_name == "pelvis_to_neck":
                    primary = joints[12] - joints[0]
                else:
                    raise ValueError(f"unsupported primary_axis {primary_name!r}")
                reference = references[definition["reference"]]
                rotation = self._make_frame(
                    name,
                    primary,
                    reference,
                    definition["axis_layout"],
                    pelvis_fallback,
                )
            self._previous_frames[name] = rotation.copy()
            result[name] = SegmentPose(origin.copy(), rotation)
        return result

    def _scale_hierarchy(self, raw: dict[str, SegmentPose]) -> dict[str, SegmentPose]:
        root_name = "Pelvis"
        scaled: dict[str, SegmentPose] = {
            root_name: SegmentPose(
                raw[root_name].position_zup * self.root_translation_scale * self.global_scale,
                raw[root_name].rotation_zup.copy(),
            )
        }
        for parent, child in self.hierarchy:
            if parent not in scaled:
                raise ValueError(f"hierarchy parent {parent!r} must precede child {child!r}")
            edge_name = f"{parent}->{child}"
            edge_scale = self.edge_scales[edge_name] * self.global_scale
            child_position = scaled[parent].position_zup + edge_scale * (
                raw[child].position_zup - raw[parent].position_zup
            )
            scaled[child] = SegmentPose(child_position, raw[child].rotation_zup.copy())
        if set(scaled) != set(SEGMENT_NAMES):
            raise ValueError("hierarchy must reach all 14 segments")
        return {name: scaled[name] for name in SEGMENT_NAMES}

    def adapt(
        self,
        joints_ay: Any,
        joint_rots_ay: Any,
        *,
        frame_id: int = 0,
        timestamp_ns: int | None = None,
    ) -> AdapterFrame:
        joints_ay_np = self._numpy(joints_ay, (22, 3), "joints_ay")
        rotations_ay_np = self._numpy(joint_rots_ay, (22, 3, 3), "joint_rots_ay")
        stamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)

        joints = np.einsum("ij,nj->ni", self._axis_convert, joints_ay_np)
        rotations = np.einsum(
            "ij,njk,kl->nil",
            self._axis_convert,
            rotations_ay_np,
            self._axis_convert.T,
        )
        forward, _, _ = self._common_axes(joints, rotations[0])
        if self._heading_rotation is None:
            yaw = math.atan2(float(forward[1]), float(forward[0]))
            self._heading_rotation = self._user_yaw @ self._rot_z(-yaw)

        joints = np.einsum("ij,nj->ni", self._heading_rotation, joints)
        rotations = np.einsum("ij,njk->nik", self._heading_rotation, rotations)
        forward, left, up = self._common_axes(joints, rotations[0])

        ground_z = self._update_ground(joints, stamp)
        if self._origin_xy is None:
            self._origin_xy = joints[0, :2].copy()
        origin = np.asarray((self._origin_xy[0], self._origin_xy[1], ground_z))
        normalized_joints = joints - origin

        raw_segments = self._derive_segments(normalized_joints, rotations, forward, left, up)
        scaled_segments = self._scale_hierarchy(raw_segments)
        return AdapterFrame(
            frame_id=int(frame_id),
            timestamp_ns=stamp,
            joints_zup=normalized_joints,
            raw_segments=raw_segments,
            scaled_segments=scaled_segments,
            ground_z=float(ground_z),
        )
