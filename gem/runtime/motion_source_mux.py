# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Single fixed-rate GMR output multiplexer for live video and generated clips."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from gem.runtime.motion_streamer import (
    MonotonicDeadline,
    MotionPlayer,
    PlayerState,
    SMPLFrame,
    SMPLMotion,
    align_motion_root_yaw,
    frame_from_motion,
    interpolate_frames,
    load_smpl_motion,
    synthetic_idle_motion,
)
from gem.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle


class MuxState(str, Enum):
    """Externally visible source arbitration states."""

    IDLE = "IDLE"
    VIDEO_LIVE = "VIDEO_LIVE"
    CLIP_PENDING = "CLIP_PENDING"
    CLIP_PLAYING = "CLIP_PLAYING"
    ESTOP = "ESTOP"
    STOPPING = "STOPPING"


@dataclass(slots=True)
class MuxTick:
    """One selected output frame and its source diagnostics."""

    frame: SMPLFrame
    state: MuxState
    source: str
    timestamp_ns: int


def _motion_from_frame(
    frame: SMPLFrame,
    *,
    fps: float,
    source: str,
) -> SMPLMotion:
    return SMPLMotion(
        body_pose=frame.body_pose.detach().cpu().float().reshape(1, 63).clone(),
        global_orient=frame.global_orient.detach().cpu().float().reshape(1, 3).clone(),
        transl=frame.transl.detach().cpu().float().reshape(1, 3).clone(),
        betas=torch.zeros(1, 10, dtype=torch.float32),
        fps=float(fps),
        source_path=Path(f"<{source}>"),
        metadata={"source": source, "shape_mode": "zero"},
    )


def _validate_video_frame(frame: SMPLFrame) -> SMPLFrame:
    if not isinstance(frame, SMPLFrame):
        raise TypeError("video frame must be an SMPLFrame")
    shapes = {
        "body_pose": (frame.body_pose, (63,)),
        "global_orient": (frame.global_orient, (3,)),
        "transl": (frame.transl, (3,)),
        "betas": (frame.betas, (10,)),
    }
    values: dict[str, torch.Tensor] = {}
    for name, (value, expected) in shapes.items():
        tensor = torch.as_tensor(value).detach().cpu().float().clone()
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"video SMPL {name} must have shape {expected}, got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"video SMPL {name} contains NaN or Inf")
        values[name] = tensor
    values["betas"] = torch.zeros_like(values["betas"])
    return SMPLFrame(**values)


def _default_endecoder_loader(device: torch.device) -> torch.nn.Module:
    from scripts.demo.stream_smpl_params_to_gmr import load_endecoder

    return load_endecoder(device)


def _default_adapter_factory(yaw_deg: float, scale: float) -> Any:
    from gem.smplx_gmr_reference import SMPLXGMRReference

    return SMPLXGMRReference(user_yaw_deg=yaw_deg, global_scale=scale)


def _default_bridge_factory(host: str, port: int, debug: bool) -> Any:
    from gem.gmr_udp_bridge import GMRUDPBridge

    return GMRUDPBridge(host, port, debug=debug)


def _default_send_frame(
    frame: SMPLFrame,
    endecoder: torch.nn.Module,
    adapter: Any,
    bridge: Any,
    *,
    device: torch.device,
    timestamp_ns: int,
) -> Any:
    from scripts.demo.stream_smpl_params_to_gmr import send_frame_to_gmr

    return send_frame_to_gmr(
        frame,
        endecoder,
        adapter,
        bridge,
        device=device,
        timestamp_ns=timestamp_ns,
    )


class MotionSourceMux:
    """Arbitrate sources and remain the sole owner of SMP1/GMR output."""

    def __init__(
        self,
        *,
        gmr_host: str = "127.0.0.1",
        gmr_port: int = 7006,
        publish_fps: float = 30.0,
        shape_mode: str = "zero",
        mode: str = "sim",
        idle_motion: str | Path | SMPLMotion | None = None,
        blend_seconds: float = 0.8,
        return_seconds: float = 1.0,
        estop_blend_seconds: float = 0.3,
        video_stale_sec: float = 0.5,
        new_motion_policy: str = "queue",
        allow_interrupt_in_robot: bool = False,
        reset_origin_on_motion: bool = False,
        smplx_yaw_deg: float = 0.0,
        gmr_scale: float = 1.0,
        max_send_errors: int = 5,
        device: str | torch.device | None = None,
        endecoder: torch.nn.Module | None = None,
        on_video_resume_reset: Callable[[], None] | None = None,
        dry_run: bool = False,
        verbose: bool = False,
        bridge_factory: Callable[[str, int, bool], Any] = _default_bridge_factory,
        adapter_factory: Callable[[float, float], Any] = _default_adapter_factory,
        endecoder_loader: Callable[[torch.device], torch.nn.Module] = _default_endecoder_loader,
        frame_sender: Callable[..., Any] = _default_send_frame,
        clock: Callable[[], float] = time.monotonic,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if mode not in {"sim", "robot"}:
            raise ValueError("mode must be sim or robot")
        if shape_mode != "zero":
            raise ValueError("MotionSourceMux only supports shape_mode=zero")
        if not math.isfinite(publish_fps) or publish_fps <= 0:
            raise ValueError("publish_fps must be finite and > 0")
        if not math.isfinite(video_stale_sec) or video_stale_sec <= 0:
            raise ValueError("video_stale_sec must be finite and > 0")
        if new_motion_policy not in {"queue", "latest", "interrupt"}:
            raise ValueError("new_motion_policy must be queue, latest, or interrupt")
        if mode == "robot" and new_motion_policy == "interrupt" and not allow_interrupt_in_robot:
            raise RuntimeError(
                "Robot mode forbids interrupt unless allow_interrupt_in_robot is set"
            )
        if mode == "robot" and idle_motion is None:
            raise RuntimeError("Robot mode requires a verified idle SMPL-X motion file.")
        if max_send_errors <= 0:
            raise ValueError("max_send_errors must be > 0")
        if not 1 <= int(gmr_port) <= 65535:
            raise ValueError("gmr_port must be in [1, 65535]")

        self.gmr_host = str(gmr_host)
        self.gmr_port = int(gmr_port)
        self.publish_fps = float(publish_fps)
        self.shape_mode = shape_mode
        self.mode = mode
        self.video_stale_sec = float(video_stale_sec)
        self.new_motion_policy = new_motion_policy
        self.allow_interrupt_in_robot = bool(allow_interrupt_in_robot)
        self.reset_origin_on_motion = bool(reset_origin_on_motion)
        self.smplx_yaw_deg = float(smplx_yaw_deg)
        self.gmr_scale = float(gmr_scale)
        self.max_send_errors = int(max_send_errors)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.dry_run = bool(dry_run)
        self.verbose = bool(verbose)
        self._bridge_factory = bridge_factory
        self._adapter_factory = adapter_factory
        self._endecoder_loader = endecoder_loader
        self._frame_sender = frame_sender
        self._clock = clock
        self._clock_ns = clock_ns
        self._provided_endecoder = endecoder
        self.on_video_resume_reset = on_video_resume_reset

        if isinstance(idle_motion, SMPLMotion):
            idle = idle_motion
        elif idle_motion is not None:
            idle = load_smpl_motion(idle_motion, shape_mode="zero", min_frames=1)
        else:
            idle = synthetic_idle_motion(self.publish_fps)
            print(
                "WARNING: using synthetic idle pose; do not use this pose on a real "
                "robot without validation."
            )
        if torch.count_nonzero(idle.betas).item() != 0:
            idle = SMPLMotion(
                body_pose=idle.body_pose,
                global_orient=idle.global_orient,
                transl=idle.transl,
                betas=torch.zeros_like(idle.betas),
                fps=idle.fps,
                source_path=idle.source_path,
                metadata={**idle.metadata, "shape_mode": "zero"},
            )
        self.safe_idle_motion = idle
        self.player = MotionPlayer(
            idle,
            blend_seconds=blend_seconds,
            return_seconds=return_seconds,
            estop_blend_seconds=estop_blend_seconds,
            logger=self._log,
        )
        self.player.start(self._clock())

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.bridge: Any | None = None
        self.adapter: Any | None = None
        self.endecoder: torch.nn.Module | None = None
        self.state = MuxState.IDLE
        self.video_mode = False
        self._latest_video_frame: SMPLFrame | None = None
        self._latest_video_time = -math.inf
        self._video_alignment_pending = True
        self._video_alignment_rotation = torch.eye(3, dtype=torch.float32)
        self._video_source_origin = torch.zeros(3, dtype=torch.float32)
        self._video_target_origin = torch.zeros(3, dtype=torch.float32)
        self._video_blend_from: SMPLFrame | None = None
        self._video_blend_started = 0.0
        self.video_alignment_count = 0
        self._last_output = frame_from_motion(idle, 0)
        self._last_source = "idle"
        self._estop_latched = False
        self._started = False
        self._completed_observed = 0
        self._motion_started_observed = 0
        self._resume_reset_requested = False
        self._consecutive_send_errors = 0
        self.send_count = 0
        self.skipped_deadlines = 0
        self.last_send_error: str | None = None
        self.last_clip_path: str | None = None

    def _log(self, message: str) -> None:
        print(message)

    def start(self) -> None:
        """Create exactly one GMR path and start the fixed-rate send thread."""
        with self._lock:
            if self._started:
                return
            self.endecoder = (
                self._provided_endecoder
                if self._provided_endecoder is not None
                else self._endecoder_loader(self.device)
            )
            self.adapter = self._adapter_factory(self.smplx_yaw_deg, self.gmr_scale)
            if not self.dry_run:
                self.bridge = self._bridge_factory(self.gmr_host, self.gmr_port, self.verbose)
            self._stop_event.clear()
            self._started = True
            self._thread = threading.Thread(
                target=self._send_loop,
                name="genmo-gmr-mux",
                daemon=True,
            )
            self._thread.start()
        print(
            f"[Mux] Started single GMR sender at {self.publish_fps:g}Hz "
            f"({self.gmr_host}:{self.gmr_port}, dry_run={self.dry_run})"
        )

    def submit_video_frame(self, frame: SMPLFrame, *, timestamp: float | None = None) -> None:
        """Root-align and store only the newest finite zero-shape video frame."""
        validated = _validate_video_frame(frame)
        now = self._clock() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("video timestamp must be finite")
        with self._lock:
            if not self.video_mode:
                return
            if self._video_alignment_pending:
                aligned_orient, aligned_transl = align_motion_root_yaw(
                    validated.global_orient.reshape(1, 3),
                    validated.transl.reshape(1, 3),
                    self._last_output.global_orient,
                    self._last_output.transl,
                )
                source_rotation = axis_angle_to_matrix(validated.global_orient)
                aligned_rotation = axis_angle_to_matrix(aligned_orient[0])
                self._video_alignment_rotation = aligned_rotation @ source_rotation.transpose(
                    -1, -2
                )
                self._video_source_origin = validated.transl.clone()
                self._video_target_origin = self._last_output.transl.clone()
                self._video_alignment_pending = False
                self._video_blend_from = self._last_output.clone()
                self._video_blend_started = now
                self.video_alignment_count += 1
                aligned = SMPLFrame(
                    body_pose=validated.body_pose.clone(),
                    global_orient=aligned_orient[0],
                    transl=aligned_transl[0],
                    betas=torch.zeros_like(validated.betas),
                )
                print(
                    "[Mux] Video root aligned to current output: "
                    f"position={self._video_target_origin.tolist()}"
                )
            else:
                source_rotation = axis_angle_to_matrix(validated.global_orient)
                aligned = SMPLFrame(
                    body_pose=validated.body_pose.clone(),
                    global_orient=matrix_to_axis_angle(
                        self._video_alignment_rotation @ source_rotation
                    ),
                    transl=(
                        self._video_alignment_rotation
                        @ (validated.transl - self._video_source_origin)
                        + self._video_target_origin
                    ),
                    betas=torch.zeros_like(validated.betas),
                )
            self._latest_video_frame = _validate_video_frame(aligned)
            self._latest_video_time = now

    def start_video_mode(self) -> None:
        """Enable live video as the source used outside generated clips."""
        with self._lock:
            if self._estop_latched:
                raise RuntimeError("Cannot start video mode while ESTOP is latched")
            self.video_mode = True
            self._latest_video_frame = None
            self._latest_video_time = -math.inf
            self._video_alignment_pending = True
            self._video_blend_from = None
            if self.state == MuxState.IDLE:
                self.state = MuxState.VIDEO_LIVE

    def stop_video_mode(self) -> None:
        """Disable video and return to the verified/simulation idle source."""
        now = self._clock()
        with self._lock:
            self.video_mode = False
            self._latest_video_frame = None
            self._latest_video_time = -math.inf
            self._video_alignment_pending = True
            self._video_blend_from = None
            self.player.idle_motion = self.safe_idle_motion
            self.player.idle_frame = frame_from_motion(self.safe_idle_motion, 0)
            if self.player.state not in {
                PlayerState.BLENDING,
                PlayerState.PLAYING,
                PlayerState.RETURNING,
                PlayerState.ERROR,
            }:
                self.player.current_frame = self._last_output.clone()
                self.player.enter_error(now)
            self.state = MuxState.IDLE

    def set_idle(self) -> None:
        """Stop live/queued work and smoothly target the safe idle pose."""
        now = self._clock()
        with self._lock:
            self.video_mode = False
            self._latest_video_frame = None
            self._latest_video_time = -math.inf
            self._video_alignment_pending = True
            self._video_blend_from = None
            self.player.queue.clear()
            self.player.idle_motion = self.safe_idle_motion
            self.player.idle_frame = frame_from_motion(self.safe_idle_motion, 0)
            self.player.enter_error(now)
            self.state = MuxState.IDLE

    def _set_return_target(self) -> None:
        if (
            self.video_mode
            and self._latest_video_frame is not None
            and self._clock() - self._latest_video_time <= self.video_stale_sec
        ):
            target = _motion_from_frame(
                self._latest_video_frame,
                fps=self.publish_fps,
                source="video_return_snapshot",
            )
        else:
            target = self.safe_idle_motion
        self.player.idle_motion = target
        self.player.idle_frame = frame_from_motion(target, 0)

    def submit_generated_motion(
        self,
        path: str | Path,
        *,
        policy: str | None = None,
    ) -> SMPLMotion:
        """Load one complete READY artifact and enqueue it without touching UDP."""
        selected_policy = self.new_motion_policy if policy is None else policy
        if selected_policy not in {"queue", "latest", "interrupt"}:
            raise ValueError("motion policy must be queue, latest, or interrupt")
        if (
            self.mode == "robot"
            and selected_policy == "interrupt"
            and not self.allow_interrupt_in_robot
        ):
            raise RuntimeError("Robot mode forbids interrupt by default")
        source = Path(path).expanduser()
        directory = source if source.is_dir() else source.parent
        if source.is_dir() and not (source / "READY").is_file():
            raise RuntimeError(f"Generated motion directory has no READY marker: {source}")
        if source.is_file() and source.name == "smpl_params.pt":
            # Direct files remain supported, but published service artifacts must
            # not be consumed before their sibling READY marker exists.
            if directory.name.startswith(".tmp_"):
                raise RuntimeError(f"Refusing temporary, unpublished motion: {source}")
        motion = load_smpl_motion(source, shape_mode="zero")
        now = self._clock()
        with self._lock:
            if self._estop_latched:
                raise RuntimeError("Cannot enqueue a generated motion during ESTOP")
            if (
                self.player.state in {PlayerState.HOLDING, PlayerState.IDLE, PlayerState.STARTING}
                and self.video_mode
                and self._latest_video_frame is not None
            ):
                self.player.current_frame = self._latest_video_frame.clone()
            self._set_return_target()
            self.player.enqueue(motion, policy=selected_policy, now=now)
            self.last_clip_path = str(motion.source_path)
            self.state = MuxState.CLIP_PENDING
        print(
            f"[Mux] Generated motion queued: {motion.source_path} "
            f"({motion.num_frames} frames @ {motion.fps:g} FPS)"
        )
        return motion

    def estop(self) -> None:
        """Latch software ESTOP and begin a finite transition to safe idle."""
        now = self._clock()
        with self._lock:
            self._estop_latched = True
            self.player.idle_motion = self.safe_idle_motion
            self.player.idle_frame = frame_from_motion(self.safe_idle_motion, 0)
            self.player.trigger_estop(now)
            self.state = MuxState.ESTOP
        print("[Mux] ESTOP latched; returning to safe idle")

    def clear_estop(self) -> None:
        """Explicitly clear ESTOP; old queued work is never resumed."""
        now = self._clock()
        with self._lock:
            self._estop_latched = False
            self.player.reset_estop(now)
            self.player.queue.clear()
            self.state = MuxState.IDLE
        print("[Mux] ESTOP cleared; waiting for a new command")

    def _video_is_fresh(self, now: float) -> bool:
        return (
            self.video_mode
            and self._latest_video_frame is not None
            and now - self._latest_video_time <= self.video_stale_sec
        )

    def _safe_idle_at_current_position(self) -> SMPLFrame:
        """Place the safe idle pose under the current horizontal root position."""
        idle = frame_from_motion(self.safe_idle_motion, 0)
        transl = idle.transl.clone()
        transl[[0, 2]] = self._last_output.transl[[0, 2]]
        return SMPLFrame(
            body_pose=idle.body_pose.clone(),
            global_orient=idle.global_orient.clone(),
            transl=transl,
            betas=torch.zeros_like(idle.betas),
        )

    def _select_frame(self, now: float) -> tuple[SMPLFrame, str]:
        if self._estop_latched:
            frame = self.player.tick(now, estop=True)
            self.state = MuxState.ESTOP
            return frame, "estop_idle"

        player_active = (
            self.player.state
            in {
                PlayerState.BLENDING,
                PlayerState.PLAYING,
                PlayerState.RETURNING,
                PlayerState.ERROR,
                PlayerState.LOADING,
            }
            or len(self.player.queue) > 0
        )
        if player_active:
            frame = self.player.tick(now)
            self.state = (
                MuxState.CLIP_PENDING
                if self.player.state == PlayerState.BLENDING and self.player.frame_float == 0.0
                else MuxState.CLIP_PLAYING
            )
            return frame, "generated_clip"

        # Let MotionPlayer settle STARTING/HOLDING before selecting live input.
        if self.player.state == PlayerState.STARTING:
            self.player.tick(now)

        if self._video_is_fresh(now):
            assert self._latest_video_frame is not None
            frame = self._latest_video_frame.clone()
            if self._video_blend_from is not None:
                alpha = (now - self._video_blend_started) / self.player.blend_seconds
                frame = interpolate_frames(self._video_blend_from, frame, alpha)
                if alpha >= 1.0:
                    self._video_blend_from = None
            self.player.current_frame = frame.clone()
            self.state = MuxState.VIDEO_LIVE
            return frame, "video_live"

        # MotionPlayer has already completed a smooth return.  Continue
        # sending that exact frame instead of replacing it with the raw idle
        # frame and reintroducing a one-frame root-yaw/translation jump.
        frame = self.player.current_frame.clone()
        if self._last_source == "video_live":
            # A stale source must not leave the robot holding an arbitrary live
            # pose.  Keep its current floor position when applying safe idle;
            # returning to the idle file's world origin would create the same
            # discontinuity as starting an unaligned video rollout.
            frame = self._safe_idle_at_current_position()
            self.player.current_frame = frame.clone()
        self.state = MuxState.IDLE
        return frame, "idle"

    def tick_once(
        self,
        *,
        now: float | None = None,
        send: bool = False,
    ) -> MuxTick:
        """Select one frame, optionally send it, and expose a deterministic test hook."""
        now_value = self._clock() if now is None else float(now)
        if not math.isfinite(now_value):
            raise ValueError("tick time must be finite")
        with self._lock:
            frame, source = self._select_frame(now_value)
            frame = _validate_video_frame(frame)
            self._last_output = frame.clone()
            self._last_source = source
            completed_after = self.player.completed_count
            started_after = self.player.motion_started_count
            if (
                completed_after > self._completed_observed
                and self.video_mode
                and not self._estop_latched
                and self.player.active_motion is None
                and len(self.player.queue) == 0
            ):
                self._resume_reset_requested = True
                self._latest_video_frame = None
                self._latest_video_time = -math.inf
                self._video_alignment_pending = True
                self._video_blend_from = None
            self._completed_observed = completed_after
            if (
                started_after > self._motion_started_observed
                and self.reset_origin_on_motion
                and self.adapter is not None
            ):
                self.adapter.reset()
            self._motion_started_observed = started_after
            callback = self.on_video_resume_reset if self._resume_reset_requested else None
            self._resume_reset_requested = False
            timestamp_ns = self._clock_ns()
            tick = MuxTick(frame, self.state, source, timestamp_ns)

        # Callback must be a non-blocking reset request; never run GPU or file
        # work in the fixed-rate lock.
        if callback is not None:
            try:
                callback()
            except Exception as exc:
                print(f"[Mux WARNING] Video reset request failed: {exc}")
        if send:
            self._send_tick(tick)
        return tick

    def _send_tick(self, tick: MuxTick) -> None:
        if self.dry_run:
            self.send_count += 1
            return
        if self.bridge is None or self.adapter is None or self.endecoder is None:
            raise RuntimeError("MotionSourceMux GMR path is not initialized")
        try:
            self._frame_sender(
                tick.frame,
                self.endecoder,
                self.adapter,
                self.bridge,
                device=self.device,
                timestamp_ns=tick.timestamp_ns,
            )
            self._consecutive_send_errors = 0
            self.last_send_error = None
            self.send_count += 1
        except Exception as exc:
            self._consecutive_send_errors += 1
            self.last_send_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[Mux GMR ERROR] {self.last_send_error} "
                f"({self._consecutive_send_errors}/{self.max_send_errors})"
            )
            if self._consecutive_send_errors >= self.max_send_errors:
                with self._lock:
                    self.player.enter_error(self._clock())
                    self._consecutive_send_errors = 0

    def _send_loop(self) -> None:
        deadline = MonotonicDeadline(self.publish_fps, self._clock())
        stats_started = self._clock()
        stats_sends = 0
        while not self._stop_event.is_set():
            now = self._clock()
            delay = deadline.seconds_until(now)
            if delay > 0:
                self._stop_event.wait(delay)
                if self._stop_event.is_set():
                    break
                now = self._clock()
            skipped = deadline.advance(now)
            self.skipped_deadlines += skipped
            try:
                tick = self.tick_once(now=now, send=True)
            except Exception as exc:
                self.last_send_error = f"{type(exc).__name__}: {exc}"
                print(f"[Mux LOOP ERROR] {self.last_send_error}")
                self.estop()
                continue
            stats_sends += 1
            elapsed = now - stats_started
            if elapsed >= 5.0:
                print(
                    f"[Mux] send={stats_sends / elapsed:.1f}Hz "
                    f"state={tick.state.value} source={tick.source} "
                    f"queue={len(self.player.queue)}"
                )
                stats_started = now
                stats_sends = 0

    def status(self) -> dict[str, Any]:
        """Return source, player, safety, and single-sender diagnostics."""
        with self._lock:
            now = self._clock()
            return {
                "started": self._started,
                "state": self.state.value,
                "player_state": self.player.state.value,
                "video_mode": self.video_mode,
                "video_fresh": self._video_is_fresh(now),
                "video_age_sec": (
                    None
                    if self._latest_video_frame is None
                    else max(0.0, now - self._latest_video_time)
                ),
                "video_root_aligned": not self._video_alignment_pending,
                "video_alignment_count": self.video_alignment_count,
                "estop": self._estop_latched,
                "queue_size": len(self.player.queue),
                "active_motion": (
                    None
                    if self.player.active_motion is None
                    else str(self.player.active_motion.source_path)
                ),
                "last_clip_path": self.last_clip_path,
                "last_source": self._last_source,
                "publish_fps": self.publish_fps,
                "send_count": self.send_count,
                "skipped_deadlines": self.skipped_deadlines,
                "last_send_error": self.last_send_error,
                "gmr_sender_instances": int(self.bridge is not None),
                "shape_mode": "zero",
                "fk_betas_zero": True,
            }

    def close(self) -> None:
        """Stop the sole sender and release its bridge exactly once."""
        with self._lock:
            if not self._started:
                return
            self.state = MuxState.STOPPING
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            if self.bridge is not None:
                self.bridge.close()
            self.bridge = None
            self.adapter = None
            self.endecoder = None
            self._thread = None
            self._started = False
        print("[Mux] Single GMR sender stopped")
