# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Validated, time-based SMPL-X playback for persistent robot streaming.

This module deliberately contains no socket or GPU requirements.  It owns the
motion-file contract, READY-directory watcher, rotation interpolation, root
continuity, queue policy, safety transitions, and monotonic playback clock.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import torch

from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
)

ShapeMode = Literal["zero"]
MotionPolicy = Literal["queue", "latest", "interrupt"]

# SMPL-X body_pose excludes the pelvis/root, so global joints 16/17 map to
# body-pose entries 15/16. In the neutral rest skeleton the arms extend along
# +/-X; opposite Z rotations lower both upper arms along -Y (GENMO is Y-up).
LEFT_SHOULDER_BODY_INDEX = 15
RIGHT_SHOULDER_BODY_INDEX = 16
SYNTHETIC_IDLE_ARM_ANGLE_RAD = math.pi / 2.0


class PlayerState(str, Enum):
    """Explicit states exposed by the persistent motion player."""

    STARTING = "STARTING"
    IDLE = "IDLE"
    LOADING = "LOADING"
    BLENDING = "BLENDING"
    PLAYING = "PLAYING"
    RETURNING = "RETURNING"
    HOLDING = "HOLDING"
    ERROR = "ERROR"
    ESTOP = "ESTOP"


@dataclass(frozen=True)
class SMPLMotion:
    """One validated, CPU float32 SMPL-X motion sequence."""

    body_pose: torch.Tensor
    global_orient: torch.Tensor
    transl: torch.Tensor
    betas: torch.Tensor
    fps: float
    source_path: Path
    metadata: dict[str, Any]

    @property
    def num_frames(self) -> int:
        return int(self.body_pose.shape[0])

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps


@dataclass(frozen=True)
class SMPLFrame:
    """One finite pose sent through SMPL-X FK."""

    body_pose: torch.Tensor
    global_orient: torch.Tensor
    transl: torch.Tensor
    betas: torch.Tensor

    def clone(self) -> SMPLFrame:
        return SMPLFrame(
            self.body_pose.clone(),
            self.global_orient.clone(),
            self.transl.clone(),
            self.betas.clone(),
        )


def _as_cpu_float(value: Any, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"SMPL motion is missing {name}")
    try:
        tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} cannot be converted to a float32 tensor") from exc
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def _resolve_motion_file(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "smpl_params.pt"
    if not source.is_file():
        raise FileNotFoundError(f"SMPL motion file does not exist: {source}")
    return source


def load_smpl_motion(
    path: str | Path,
    *,
    shape_mode: ShapeMode = "zero",
    min_frames: int = 2,
) -> SMPLMotion:
    """Load and strictly validate global SMPL-X parameters from a ``.pt`` file."""
    if shape_mode != "zero":
        raise ValueError("The robot streamer only supports shape_mode=zero")
    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    source = _resolve_motion_file(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"SMPL motion payload must be a dictionary: {source}")
    group = payload.get("body_params_global")
    if not isinstance(group, dict):
        raise ValueError(f"SMPL motion is missing body_params_global: {source}")

    body_pose = _as_cpu_float(group.get("body_pose"), "body_pose")
    global_orient = _as_cpu_float(group.get("global_orient"), "global_orient")
    transl = _as_cpu_float(group.get("transl"), "transl")
    betas = _as_cpu_float(group.get("betas"), "betas")
    expected_last_dims = {
        "body_pose": (body_pose, 63),
        "global_orient": (global_orient, 3),
        "transl": (transl, 3),
        "betas": (betas, 10),
    }
    for name, (tensor, final_dim) in expected_last_dims.items():
        if tensor.ndim != 2 or tensor.shape[1] != final_dim:
            raise ValueError(f"{name} must have shape [L, {final_dim}], got {tuple(tensor.shape)}")
    length = int(body_pose.shape[0])
    if length < min_frames:
        raise ValueError(f"SMPL motion must contain at least {min_frames} frames, got {length}")
    for name, (tensor, _) in expected_last_dims.items():
        if tensor.shape[0] != length:
            raise ValueError(
                f"All SMPL fields must share first dimension {length}; {name} has {tensor.shape[0]}"
            )

    raw_fps = payload.get("fps")
    metadata = payload.get("metadata", {})
    if raw_fps is None and isinstance(metadata, dict):
        raw_fps = metadata.get("fps")
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SMPL motion fps must be a positive number: {raw_fps!r}") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"SMPL motion fps must be finite and > 0, got {fps}")

    # Robot FK has one body shape by policy, irrespective of source contents.
    betas = torch.zeros_like(betas)
    if torch.count_nonzero(betas).item() != 0:
        raise AssertionError("shape_mode=zero failed to produce zero betas")
    metadata_out = dict(metadata) if isinstance(metadata, dict) else {}
    for key in ("prompt", "seed", "source", "shape_mode", "num_frames"):
        if key in payload:
            metadata_out.setdefault(key, payload[key])
    metadata_out["shape_mode"] = "zero"
    return SMPLMotion(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        betas=betas,
        fps=fps,
        source_path=source.resolve(),
        metadata=metadata_out,
    )


def synthetic_idle_motion(fps: float = 30.0) -> SMPLMotion:
    """Return a simulation-only one-frame standing pose with both arms down."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and > 0")
    body_pose = torch.zeros(1, 63)
    body_pose_aa = body_pose.reshape(1, 21, 3)
    body_pose_aa[0, LEFT_SHOULDER_BODY_INDEX, 2] = -SYNTHETIC_IDLE_ARM_ANGLE_RAD
    body_pose_aa[0, RIGHT_SHOULDER_BODY_INDEX, 2] = SYNTHETIC_IDLE_ARM_ANGLE_RAD
    return SMPLMotion(
        body_pose=body_pose,
        global_orient=torch.zeros(1, 3),
        transl=torch.zeros(1, 3),
        betas=torch.zeros(1, 10),
        fps=float(fps),
        source_path=Path("<synthetic-idle>"),
        metadata={
            "source": "synthetic_idle_arms_down",
            "shape_mode": "zero",
            "arms": "down",
        },
    )


def frame_from_motion(motion: SMPLMotion, index: int) -> SMPLFrame:
    """Extract one independent frame from a validated motion."""
    if not 0 <= index < motion.num_frames:
        raise IndexError(f"frame index {index} is outside [0, {motion.num_frames})")
    return SMPLFrame(
        motion.body_pose[index].clone(),
        motion.global_orient[index].clone(),
        motion.transl[index].clone(),
        torch.zeros_like(motion.betas[index]),
    )


def smoothstep(alpha: float) -> float:
    """Cubic ease-in/ease-out used for root-position transitions."""
    value = min(max(float(alpha), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def interpolate_axis_angle(
    aa0: torch.Tensor,
    aa1: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Shortest-path quaternion SLERP between axis-angle rotations."""
    if aa0.shape != aa1.shape or aa0.shape[-1] != 3:
        raise ValueError(
            f"axis-angle inputs must have matching [..., 3] shapes, got {aa0.shape}/{aa1.shape}"
        )
    if not torch.isfinite(aa0).all() or not torch.isfinite(aa1).all():
        raise ValueError("axis-angle interpolation input contains NaN or Inf")
    t = min(max(float(alpha), 0.0), 1.0)
    q0 = matrix_to_quaternion(axis_angle_to_matrix(aa0))
    q1 = matrix_to_quaternion(axis_angle_to_matrix(aa1))
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    t_tensor = torch.as_tensor(t, dtype=q0.dtype, device=q0.device)
    safe = sin_theta.abs() > 1e-6
    safe_denominator = torch.where(safe, sin_theta, torch.ones_like(sin_theta))
    weight0 = torch.where(
        safe,
        torch.sin((1.0 - t_tensor) * theta) / safe_denominator,
        1.0 - t_tensor,
    )
    weight1 = torch.where(
        safe,
        torch.sin(t_tensor * theta) / safe_denominator,
        t_tensor,
    )
    quaternion = weight0 * q0 + weight1 * q1
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return quaternion_to_axis_angle(quaternion)


def interpolate_frames(frame0: SMPLFrame, frame1: SMPLFrame, alpha: float) -> SMPLFrame:
    """Interpolate rotations with SLERP and translation with cubic smoothstep."""
    eased = smoothstep(alpha)
    pose = interpolate_axis_angle(
        frame0.body_pose.reshape(21, 3), frame1.body_pose.reshape(21, 3), eased
    ).reshape(63)
    root = interpolate_axis_angle(frame0.global_orient, frame1.global_orient, eased)
    transl = frame0.transl + (frame1.transl - frame0.transl) * eased
    result = SMPLFrame(pose, root, transl, torch.zeros_like(frame0.betas))
    for name, value in vars(result).items():
        if not torch.isfinite(value).all():
            raise ValueError(f"interpolated {name} contains NaN or Inf")
    return result


def sample_motion_at(motion: SMPLMotion, elapsed_seconds: float) -> tuple[SMPLFrame, float, bool]:
    """Sample a motion from elapsed monotonic time rather than loop count."""
    frame_float = max(float(elapsed_seconds), 0.0) * motion.fps
    if frame_float >= motion.num_frames:
        return frame_from_motion(motion, motion.num_frames - 1), float(motion.num_frames - 1), True
    if frame_float >= motion.num_frames - 1:
        return frame_from_motion(motion, motion.num_frames - 1), frame_float, False
    lower = int(math.floor(frame_float))
    alpha = frame_float - lower
    return (
        interpolate_frames(
            frame_from_motion(motion, lower), frame_from_motion(motion, lower + 1), alpha
        ),
        frame_float,
        False,
    )


def _horizontal_heading(rotation: torch.Tensor) -> torch.Tensor:
    forward = rotation[..., :, 2].clone()
    forward[..., 1] = 0.0
    norm = forward.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(forward)
    fallback[..., 2] = 1.0
    return torch.where(norm > 1e-6, forward / norm.clamp(min=1e-8), fallback)


def align_motion_root_yaw(
    motion_global_orient: torch.Tensor,
    motion_transl: torch.Tensor,
    current_global_orient: torch.Tensor,
    current_transl: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align an AY/Y-up motion's first root yaw and position to the current frame.

    The yaw delta is derived from horizontal root heading vectors.  The same
    Y-axis rotation is applied to root orientations and relative translations,
    preserving the motion's internal turns and path.
    """
    if motion_global_orient.ndim != 2 or motion_global_orient.shape[-1] != 3:
        raise ValueError("motion_global_orient must have shape [L, 3]")
    if motion_transl.shape != motion_global_orient.shape:
        raise ValueError("motion_transl must have shape [L, 3]")
    if current_global_orient.shape != (3,):
        raise ValueError("current_global_orient must have shape [3]")
    if current_transl is None:
        current_transl = motion_transl[0]
    if current_transl.shape != (3,):
        raise ValueError("current_transl must have shape [3]")

    motion_rotations = axis_angle_to_matrix(motion_global_orient)
    current_rotation = axis_angle_to_matrix(current_global_orient)
    source_heading = _horizontal_heading(motion_rotations[0])
    current_heading = _horizontal_heading(current_rotation)
    sine = torch.cross(source_heading, current_heading, dim=-1)[1]
    cosine = torch.dot(source_heading, current_heading).clamp(-1.0, 1.0)
    yaw_delta = torch.atan2(sine, cosine)
    yaw_aa = torch.zeros(3, dtype=motion_transl.dtype, device=motion_transl.device)
    yaw_aa[1] = yaw_delta
    yaw_rotation = axis_angle_to_matrix(yaw_aa)

    aligned_orient = matrix_to_axis_angle(yaw_rotation @ motion_rotations)
    relative_transl = motion_transl - motion_transl[:1]
    rotated_relative = torch.einsum("ij,lj->li", yaw_rotation, relative_transl)
    aligned_transl = rotated_relative + current_transl
    if not torch.isfinite(aligned_orient).all() or not torch.isfinite(aligned_transl).all():
        raise ValueError("root alignment produced NaN or Inf")
    return aligned_orient, aligned_transl


def align_motion_to_frame(motion: SMPLMotion, current: SMPLFrame) -> SMPLMotion:
    """Return a root-continuous copy of ``motion`` aligned to ``current``."""
    orient, transl = align_motion_root_yaw(
        motion.global_orient,
        motion.transl,
        current.global_orient,
        current.transl,
    )
    metadata = dict(motion.metadata)
    metadata["root_aligned"] = True
    return replace(
        motion,
        body_pose=motion.body_pose.clone(),
        global_orient=orient,
        transl=transl,
        betas=torch.zeros_like(motion.betas),
        metadata=metadata,
    )


class MotionQueue:
    """Queue container implementing queue/latest/interrupt insertion policies."""

    def __init__(self) -> None:
        self._items: deque[SMPLMotion] = deque()

    def add(self, motion: SMPLMotion, policy: MotionPolicy = "queue") -> None:
        if policy not in {"queue", "latest", "interrupt"}:
            raise ValueError(f"Unknown new motion policy: {policy}")
        if policy in {"latest", "interrupt"}:
            self._items.clear()
        self._items.append(motion)

    def pop(self) -> SMPLMotion:
        return self._items.popleft()

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class MotionWatcher:
    """Polling READY-directory watcher with persistent consumed state."""

    def __init__(
        self,
        watch_dir: str | Path,
        *,
        replay_existing: bool = False,
        state_filename: str = ".watch_state.json",
    ) -> None:
        self.watch_dir = Path(watch_dir).expanduser()
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.watch_dir / state_filename
        self._consumed = set() if replay_existing else self._load_state()
        self._offered: set[str] = set()
        if not replay_existing:
            existing = {str(path.resolve()) for path in self._ready_dirs()}
            if existing - self._consumed:
                self._consumed.update(existing)
                self._save_state()

    def _load_state(self) -> set[str]:
        if not self.state_path.is_file():
            return set()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            values = payload.get("consumed", [])
            return {str(Path(value).resolve()) for value in values}
        except (OSError, ValueError, TypeError):
            return set()

    def _save_state(self) -> None:
        payload = {"consumed": sorted(self._consumed)}
        temp = self.state_path.with_name(f".{self.state_path.name}.tmp-{os.getpid()}")
        try:
            temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, self.state_path)
        except OSError as exc:
            temp.unlink(missing_ok=True)
            print(f"[Watch WARNING] Unable to save {self.state_path}: {exc}")

    def _ready_dirs(self) -> list[Path]:
        return [
            path
            for path in self.watch_dir.iterdir()
            if path.is_dir() and (path / "READY").is_file()
        ]

    @staticmethod
    def _completion_key(path: Path) -> tuple[float, str]:
        metadata_path = path / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                completed = metadata.get("completed_at")
                if isinstance(completed, str):
                    stamp = datetime.fromisoformat(completed.replace("Z", "+00:00")).timestamp()
                    return stamp, str(path)
            except (OSError, ValueError, TypeError):
                pass
        return (path / "READY").stat().st_mtime, str(path)

    def scan(self) -> list[Path]:
        candidates = []
        for path in self._ready_dirs():
            resolved = str(path.resolve())
            if resolved not in self._consumed and resolved not in self._offered:
                candidates.append(path)
                self._offered.add(resolved)
        return sorted(candidates, key=self._completion_key)

    def mark_consumed(self, path: str | Path) -> None:
        resolved = str(Path(path).resolve())
        self._consumed.add(resolved)
        self._offered.add(resolved)
        self._save_state()

    @property
    def consumed(self) -> frozenset[str]:
        return frozenset(self._consumed)


class MotionPlayer:
    """Safety-oriented state machine producing one continuous SMPL frame per tick."""

    def __init__(
        self,
        idle_motion: SMPLMotion,
        *,
        blend_seconds: float = 0.8,
        return_seconds: float = 1.0,
        estop_blend_seconds: float = 0.3,
        loop: bool = False,
        logger: Callable[[str], None] | None = print,
    ) -> None:
        for name, value in {
            "blend_seconds": blend_seconds,
            "return_seconds": return_seconds,
            "estop_blend_seconds": estop_blend_seconds,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        self.idle_motion = idle_motion
        self.idle_frame = frame_from_motion(idle_motion, 0)
        self.current_frame = self.idle_frame.clone()
        self.blend_seconds = float(blend_seconds)
        self.return_seconds = float(return_seconds)
        self.estop_blend_seconds = float(estop_blend_seconds)
        self.loop = bool(loop)
        self.logger = logger
        self.state = PlayerState.STARTING
        self.queue = MotionQueue()
        self.active_motion: SMPLMotion | None = None
        self._active_source: SMPLMotion | None = None
        self._phase_from = self.current_frame.clone()
        self._phase_to = self.current_frame.clone()
        self._phase_started = 0.0
        self._playing_started = 0.0
        self.frame_float = 0.0
        self.completed_count = 0
        self.motion_started_count = 0
        self.estop_latched = False
        self._loading_from_holding = False

    def _transition(self, state: PlayerState) -> None:
        previous = self.state
        self.state = state
        if previous != state and self.logger is not None:
            self.logger(f"[State] {previous.value} -> {state.value}")

    def start(self, now: float) -> None:
        self._phase_started = float(now)
        self._transition(PlayerState.HOLDING)

    def begin_loading(self) -> None:
        """Expose background loading without interrupting an active motion."""
        self._loading_from_holding = self.state in {PlayerState.HOLDING, PlayerState.IDLE}
        if self._loading_from_holding:
            self._transition(PlayerState.LOADING)

    def finish_loading(self) -> None:
        """Leave LOADING after a background load succeeds or fails."""
        if self.state == PlayerState.LOADING:
            self._transition(PlayerState.HOLDING)
        self._loading_from_holding = False

    def enqueue(self, motion: SMPLMotion, *, policy: MotionPolicy, now: float) -> None:
        if policy == "interrupt":
            self.queue.add(motion, "interrupt")
            if self.state not in {PlayerState.STARTING, PlayerState.ESTOP}:
                self._begin_motion(self.queue.pop(), now)
        else:
            self.queue.add(motion, policy)
            if self.state in {PlayerState.HOLDING, PlayerState.IDLE} and not self.estop_latched:
                self._begin_motion(self.queue.pop(), now)

    def _begin_motion(self, motion: SMPLMotion, now: float) -> None:
        aligned = align_motion_to_frame(motion, self.current_frame)
        self.active_motion = aligned
        self._active_source = motion
        self._phase_from = self.current_frame.clone()
        self._phase_to = frame_from_motion(aligned, 0)
        self._phase_started = float(now)
        self.frame_float = 0.0
        self.motion_started_count += 1
        self._transition(PlayerState.BLENDING)

    def _aligned_idle_target(self) -> SMPLFrame:
        aligned = align_motion_to_frame(self.idle_motion, self.current_frame)
        return frame_from_motion(aligned, 0)

    def _begin_return(self, now: float, *, error: bool = False) -> None:
        if self.loop and self._active_source is not None and not error:
            self.queue.add(self._active_source, "queue")
        self._phase_from = self.current_frame.clone()
        self._phase_to = self._aligned_idle_target()
        self._phase_started = float(now)
        self.active_motion = None
        self._active_source = None
        self._transition(PlayerState.ERROR if error else PlayerState.RETURNING)

    def enter_error(self, now: float) -> None:
        self.queue.clear()
        self._begin_return(now, error=True)

    def trigger_estop(self, now: float) -> None:
        if self.estop_latched:
            return
        self.estop_latched = True
        self.queue.clear()
        self.active_motion = None
        self._active_source = None
        self._phase_from = self.current_frame.clone()
        self._phase_to = self._aligned_idle_target()
        self._phase_started = float(now)
        self._transition(PlayerState.ESTOP)

    def reset_estop(self, now: float) -> None:
        """Explicitly release a latched software ESTOP after its file is removed."""
        if not self.estop_latched:
            return
        self.estop_latched = False
        self.current_frame = self._phase_to.clone()
        self.idle_frame = self.current_frame.clone()
        self._phase_started = float(now)
        self._transition(PlayerState.HOLDING)

    def tick(self, now: float, *, estop: bool = False) -> SMPLFrame:
        now = float(now)
        if self.state == PlayerState.STARTING:
            self.start(now)
        if estop:
            self.trigger_estop(now)

        if self.state == PlayerState.ESTOP:
            alpha = (now - self._phase_started) / self.estop_blend_seconds
            self.current_frame = interpolate_frames(self._phase_from, self._phase_to, alpha)
            if alpha >= 1.0:
                self.current_frame = self._phase_to.clone()
            return self.current_frame

        if self.state in {PlayerState.HOLDING, PlayerState.IDLE}:
            if len(self.queue) > 0 and not self.estop_latched:
                self._begin_motion(self.queue.pop(), now)
            else:
                return self.current_frame

        if self.state == PlayerState.LOADING:
            return self.current_frame

        if self.state == PlayerState.BLENDING:
            alpha = (now - self._phase_started) / self.blend_seconds
            self.current_frame = interpolate_frames(self._phase_from, self._phase_to, alpha)
            if alpha >= 1.0:
                self.current_frame = self._phase_to.clone()
                self._playing_started = now
                self._transition(PlayerState.PLAYING)
            return self.current_frame

        if self.state == PlayerState.PLAYING:
            if self.active_motion is None:
                self.enter_error(now)
                return self.current_frame
            frame, self.frame_float, finished = sample_motion_at(
                self.active_motion, now - self._playing_started
            )
            self.current_frame = frame
            if finished:
                self._begin_return(now)
            return self.current_frame

        if self.state in {PlayerState.RETURNING, PlayerState.ERROR}:
            alpha = (now - self._phase_started) / self.return_seconds
            self.current_frame = interpolate_frames(self._phase_from, self._phase_to, alpha)
            if alpha >= 1.0:
                self.current_frame = self._phase_to.clone()
                self.idle_frame = self.current_frame.clone()
                self.completed_count += 1
                self._transition(PlayerState.HOLDING)
                if len(self.queue) > 0:
                    self._begin_motion(self.queue.pop(), now)
            return self.current_frame

        return self.current_frame


class MonotonicDeadline:
    """Fixed-rate deadline scheduler that skips missed sends instead of bursting."""

    def __init__(self, fps: float, start_time: float) -> None:
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("publish fps must be finite and > 0")
        self.period = 1.0 / float(fps)
        self.next_deadline = float(start_time)

    def seconds_until(self, now: float) -> float:
        return max(self.next_deadline - float(now), 0.0)

    def advance(self, now: float) -> int:
        """Advance one deadline and return how many stale deadlines were skipped."""
        now = float(now)
        candidate = self.next_deadline + self.period
        skipped = 0
        if candidate <= now:
            skipped = int(math.floor((now - candidate) / self.period)) + 1
            candidate += skipped * self.period
        self.next_deadline = candidate
        return skipped
