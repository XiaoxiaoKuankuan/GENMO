#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Resident GMR/GMT safety bridge for TensorRT sliding music deployment."""

from __future__ import annotations

import argparse
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import redis
import torch
from scipy.spatial.transform import Rotation, Slerp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.gmr_udp_bridge import SMP1PacketEncoder  # noqa: E402
from gem.runtime.gmr_mujoco_viewer import GMRMujocoViewerClient  # noqa: E402
from gem.runtime.gmt_trajectory import (  # noqa: E402
    BUMI_QPOS_DIM,
    FLAG_AUDIO,
    FLAG_FIXED_IDLE,
    FLAG_TRANSITION,
    GmtPolicyContract,
    IncrementalGmtFrameTimeline,
    RedisTrajectoryPublisher,
    qpos_timeline_to_gmt_frames,
)
from gem.runtime.motion_streamer import MonotonicDeadline, synthetic_idle_motion  # noqa: E402
from gem.runtime.robot_stream import (  # noqa: E402
    GMRBatchClient,
    GMRMotionRetargetSession,
    IncrementalQposTimeline,
    RevisionTracker,
    RobotStreamChunk,
    has_complete_publish_context,
    heartbeat_expired,
    smpl_params_to_smp1_payload,
)
from gem.smplx_gmr_reference import SMPLXGMRReference  # noqa: E402
from scripts.demo.stream_smpl_params_to_gmr import load_endecoder  # noqa: E402

DEFAULT_GMR_ROOT = Path("/home/weili/GMR-CPP_e1jump_lowdpi")
DEFAULT_GMT_POLICY = Path(
    "/home/weili/docker_projects/bumi_GMT_deployment_listao/"
    "bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/"
    "policy/bumi/0724_lab_148500.onnx"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:7021")
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--gmr-binary", type=Path)
    parser.add_argument("--ik-config", type=Path)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--ground-clearance", type=float, default=0.05)
    parser.add_argument(
        "--gmr-vis",
        action="store_true",
        help="show the current 50 Hz GMR BUMI qpos in a MuJoCo window",
    )
    parser.add_argument("--gmr-viewer-binary", type=Path)
    parser.add_argument("--gmr-viewer-width", type=int, default=640)
    parser.add_argument("--gmr-viewer-height", type=int, default=480)
    parser.add_argument("--gmt-policy", type=Path, default=DEFAULT_GMT_POLICY)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-key", default="gmt_online_frame_bumi")
    parser.add_argument("--redis-ttl-ms", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audio-playback", choices=("off", "ffplay"), default="ffplay")
    parser.add_argument("--blend-seconds", type=float, default=0.8)
    parser.add_argument("--return-seconds", type=float, default=1.0)
    parser.add_argument("--ack-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--ack-stale-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-timeout-seconds", type=float, default=1.5)
    parser.add_argument("--critical-buffer-seconds", type=float, default=2.2)
    parser.add_argument("--estop-file", type=Path, default=Path("/tmp/genmo_estop"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "blend_seconds",
        "return_seconds",
        "ack_timeout_seconds",
        "ack_stale_seconds",
        "heartbeat_timeout_seconds",
        "critical_buffer_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and > 0")
    if not math.isfinite(args.ground_clearance) or args.ground_clearance < 0.0:
        raise ValueError("--ground-clearance must be finite and >= 0")
    if args.gmr_viewer_width < 160 or args.gmr_viewer_height < 120:
        raise ValueError("GMR viewer dimensions must be at least 160x120")
    if args.redis_ttl_ms <= 40:
        raise ValueError("--redis-ttl-ms must exceed two 50 Hz periods")


def _native_joint_names(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(str(value) for value in payload["joint_names_mujoco_qpos_order"])
    if len(names) != 21 or len(set(names)) != 21:
        raise ValueError("BUMI preset has an invalid native joint order")
    return names


def _intermediate_qpos(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, BUMI_QPOS_DIM), dtype=np.float32)
    first = np.asarray(start, dtype=np.float32).reshape(BUMI_QPOS_DIM)
    last = np.asarray(end, dtype=np.float32).reshape(BUMI_QPOS_DIM)
    alpha = np.arange(1, count + 1, dtype=np.float64) / float(count + 1)
    eased = alpha * alpha * (3.0 - 2.0 * alpha)
    output = first[None] + (last - first)[None] * eased[:, None]
    quats = np.stack((first[3:7], last[3:7]))[:, (1, 2, 3, 0)]
    if float(np.dot(quats[0], quats[1])) < 0.0:
        quats[1] *= -1.0
    output[:, 3:7] = Slerp(
        np.asarray((0.0, 1.0)), Rotation.from_quat(quats)
    )(eased).as_quat()[:, (3, 0, 1, 2)]
    return output.astype(np.float32)


def _align_action(action: np.ndarray, idle: np.ndarray) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).copy()
    target = np.asarray(idle, dtype=np.float32).reshape(BUMI_QPOS_DIM)
    source_rotation = Rotation.from_quat(result[0, (4, 5, 6, 3)])
    target_rotation = Rotation.from_quat(target[[4, 5, 6, 3]])
    delta = Rotation.from_euler(
        "z", target_rotation.as_euler("zyx")[0] - source_rotation.as_euler("zyx")[0]
    )
    relative = result[:, :3] - result[:1, :3]
    result[:, :3] = delta.apply(relative).astype(np.float32) + result[:1, :3]
    result[:, :2] += target[None, :2] - result[:1, :2]
    xyzw = (delta * Rotation.from_quat(result[:, (4, 5, 6, 3)])).as_quat()
    result[:, 3:7] = xyzw[:, (3, 0, 1, 2)]
    return result


@dataclass(frozen=True, slots=True)
class _PlanSnapshot:
    """One immutable plan revision consumed by the 50 Hz publisher."""

    qpos: np.ndarray
    frames: np.ndarray
    audio_start_frame: int
    audio_end_frame: int
    action_complete: bool
    terminal_idle_qpos: np.ndarray | None = None
    terminal_idle_frames: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2 or self.qpos.shape[1] != BUMI_QPOS_DIM:
            raise ValueError("plan qpos has an invalid shape")
        if self.frames.ndim != 2 or len(self.frames) != len(self.qpos):
            raise ValueError("plan frames have an invalid shape")
        if self.qpos.flags.writeable or self.frames.flags.writeable:
            raise ValueError("plan arrays must be immutable")


class _IncrementalPlanBuilder:
    """Build one action plan without revisiting its committed history."""

    def __init__(
        self,
        idle_qpos: np.ndarray,
        native_to_gmt: np.ndarray,
        *,
        blend_seconds: float,
        return_seconds: float,
    ) -> None:
        self.idle_qpos = np.asarray(idle_qpos, dtype=np.float32).copy()
        self.native_to_gmt = np.asarray(native_to_gmt, dtype=np.int64).copy()
        self.blend_frames = int(round(float(blend_seconds) * 50.0))
        self.return_frames = int(round(float(return_seconds) * 50.0))
        self.resampler = IncrementalQposTimeline()
        self.timeline = IncrementalGmtFrameTimeline(self.native_to_gmt, fps=50.0)
        self.source_origin: np.ndarray | None = None
        self.alignment_rotation: Rotation | None = None
        self.action_frames = 0
        self.audio_start_frame = 0
        self.closed = False

    def _align_suffix(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).copy()
        if self.source_origin is None:
            self.source_origin = result[0, :3].copy()
            source_rotation = Rotation.from_quat(result[0, (4, 5, 6, 3)])
            target_rotation = Rotation.from_quat(self.idle_qpos[[4, 5, 6, 3]])
            self.alignment_rotation = Rotation.from_euler(
                "z",
                target_rotation.as_euler("zyx")[0]
                - source_rotation.as_euler("zyx")[0],
            )
        assert self.alignment_rotation is not None
        assert self.source_origin is not None
        relative = result[:, :3] - self.source_origin[None]
        result[:, :3] = (
            self.alignment_rotation.apply(relative).astype(np.float32)
            + self.source_origin[None]
        )
        result[:, :2] += self.idle_qpos[None, :2] - self.source_origin[None, :2]
        xyzw = (
            self.alignment_rotation
            * Rotation.from_quat(result[:, (4, 5, 6, 3)])
        ).as_quat()
        result[:, 3:7] = xyzw[:, (3, 0, 1, 2)]
        return result

    def append(self, source_qpos: np.ndarray, *, is_last: bool) -> _PlanSnapshot:
        if self.closed:
            raise RuntimeError("cannot append to a completed plan")
        new_action = self.resampler.append(source_qpos)
        if len(new_action) <= 0:
            raise RuntimeError("source qpos did not produce a new 50 Hz sample")
        aligned = self._align_suffix(new_action)
        if self.action_frames == 0:
            prefix = np.repeat(self.idle_qpos[None], 10, axis=0)
            blend = _intermediate_qpos(
                self.idle_qpos, aligned[0], self.blend_frames
            )
            self.audio_start_frame = len(prefix) + len(blend)
            self.timeline.append(np.concatenate((prefix, blend, aligned), axis=0))
        else:
            self.timeline.append(aligned)
        self.action_frames += len(aligned)
        audio_end_frame = self.audio_start_frame + self.action_frames

        terminal_idle_qpos: np.ndarray | None = None
        terminal_idle_frames: np.ndarray | None = None
        if is_last:
            terminal_idle_qpos = aligned[-1].copy()
            terminal_idle_qpos[:2] = aligned[-1, :2]
            terminal_idle_qpos[2:] = self.idle_qpos[2:]
            returning = _intermediate_qpos(
                aligned[-1], terminal_idle_qpos, self.return_frames
            )
            self.timeline.append(
                np.concatenate(
                    (
                        returning,
                        np.repeat(terminal_idle_qpos[None], 101, axis=0),
                    ),
                    axis=0,
                )
            )
            terminal_idle_frames = qpos_timeline_to_gmt_frames(
                np.repeat(terminal_idle_qpos[None], 110, axis=0),
                fps=50.0,
                native_to_gmt=self.native_to_gmt,
            )
            terminal_idle_qpos.setflags(write=False)
            terminal_idle_frames.setflags(write=False)
            self.closed = True

        return _PlanSnapshot(
            qpos=self.timeline.qpos,
            frames=self.timeline.frames,
            audio_start_frame=self.audio_start_frame,
            audio_end_frame=audio_end_frame,
            action_complete=bool(is_last),
            terminal_idle_qpos=terminal_idle_qpos,
            terminal_idle_frames=terminal_idle_frames,
        )


class AudioController:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.process: subprocess.Popen[bytes] | None = None
        self.lock = threading.RLock()

    def start(self, path: Path, start_sec: float, duration_sec: float) -> bool:
        with self.lock:
            self.stop("replace")
            if self.mode == "off":
                return False
            ffplay = shutil.which("ffplay")
            if ffplay is None:
                print("[Audio WARNING] ffplay is unavailable")
                return False
            self.process = subprocess.Popen(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_sec:.9f}",
                    "-t",
                    f"{duration_sec:.9f}",
                    str(path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

    def stop(self, reason: str) -> None:
        del reason
        with self.lock:
            process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()

        def reap() -> None:
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        # Never make the 50 Hz safety path wait for an audio subprocess.
        threading.Thread(target=reap, daemon=True).start()


class BridgeRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.shutdown_requested = False
        self.chunk_queue: queue.Queue[RobotStreamChunk] = queue.Queue(maxsize=2)
        self.worker = threading.Thread(target=self._retarget_loop, daemon=True)
        self.control = threading.Thread(target=self._control_loop, daemon=True)
        self.tracker = RevisionTracker()
        self.state = "BOOT"
        self.last_error: str | None = None
        self.last_heartbeat = time.monotonic()
        self.request: dict[str, Any] | None = None
        self.plan_builder: _IncrementalPlanBuilder | None = None
        self.plan_snapshot: _PlanSnapshot | None = None
        self.publish_generation = 0
        self.stand_building = False
        self.cursor = 0
        self.audio_started = False
        self.publisher: RedisTrajectoryPublisher | None = None
        self.submitted_monotonic = 0.0
        self.last_ack_monotonic = 0.0
        self.ack_latency_ms: float | None = None
        self.last_ack_sequence = -1
        self.acked = False
        self.gmr_times_ms: list[float] = []
        self.gmr_warmup_ms: float | None = None
        self.retargeted_frames = 0
        self.plan_build_last_ms: float | None = None
        self.plan_build_max_ms = 0.0
        self.publish_ticks = 0
        self.publish_started = time.monotonic()
        self.publish_jitter_ms: deque[float] = deque(maxlen=30_000)
        self.publish_p99_jitter_ms: float | None = None
        self.publish_max_lock_wait_ms = 0.0

        root = args.gmr_root.expanduser().resolve(strict=True)
        self.gmr_root = root
        self.gmr_binary = (
            args.gmr_binary
            if args.gmr_binary is not None
            else root / "build/smplx_bumi3_batch_server"
        ).expanduser().resolve(strict=True)
        self.ik_config = (
            args.ik_config
            if args.ik_config is not None
            else root / "config/ik_configs/smplx_to_bumi3_auto.json"
        ).expanduser().resolve(strict=True)
        self.robot_xml = (
            args.robot_xml
            if args.robot_xml is not None
            else root / "assets/bumi3/mjcf/bumi3.xml"
        ).expanduser().resolve(strict=True)
        preset = (root / "config/robot_presets/bumi3.json").resolve(strict=True)

        self.redis = redis.Redis(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            socket_timeout=1.0,
        )
        if not self.redis.ping():
            raise RuntimeError("Redis did not respond to PING")
        self.contract = GmtPolicyContract.from_onnx(args.gmt_policy)
        self.native_to_gmt = self.contract.native_to_gmt_indices(
            _native_joint_names(preset)
        )
        self.audio = AudioController(args.audio_playback)
        self.gmr: GMRBatchClient | None = None
        self.gmr_motion = GMRMotionRetargetSession(warmup_iterations=1000)
        self.gmr_viewer: GMRMujocoViewerClient | None = None
        self.gmr_viewer_warning_reported = False
        self.idle_packet = self._make_idle_packet(args.device)
        self.idle_qpos = self._reset_gmr()
        if args.gmr_vis:
            viewer_binary = (
                args.gmr_viewer_binary
                if args.gmr_viewer_binary is not None
                else self.gmr_binary.parent / "bumi3_qpos_viewer"
            ).expanduser().resolve(strict=True)
            self.gmr_viewer = GMRMujocoViewerClient(
                [
                    str(viewer_binary),
                    "--xml",
                    str(self.robot_xml),
                    "--viewer-width",
                    str(args.gmr_viewer_width),
                    "--viewer-height",
                    str(args.gmr_viewer_height),
                    "--follow-body",
                    "base_link",
                ],
                cwd=self.gmr_root,
            )
            self.gmr_viewer.publish(self.idle_qpos)
            print(
                "[GMR viewer] MuJoCo visualization active at "
                f"{args.gmr_viewer_width}x{args.gmr_viewer_height}"
            )
        self.idle_frames = qpos_timeline_to_gmt_frames(
            np.repeat(self.idle_qpos[None], 110, axis=0),
            fps=50.0,
            native_to_gmt=self.native_to_gmt,
        )
        self.idle_publisher = RedisTrajectoryPublisher(
            self.redis, key=args.redis_key, ttl_ms=args.redis_ttl_ms
        )
        self.state = "STAND"

    def _gmr_command(self) -> list[str]:
        return [
            str(self.gmr_binary),
            "--ik-config",
            str(self.ik_config),
            "--xml",
            str(self.robot_xml),
            "--offset-to-ground",
            "--ground-clearance",
            f"{self.args.ground_clearance:.9g}",
        ]

    def _ensure_gmr(self) -> GMRBatchClient:
        if self.gmr is None or self.gmr.process.poll() is not None:
            if self.gmr is not None:
                self.gmr.close()
            self.gmr = GMRBatchClient(self._gmr_command(), cwd=self.gmr_root)
            # A new subprocess has no solver history, even if the request
            # revision itself did not change.
            self.gmr_motion.invalidate()
        return self.gmr

    def _reset_gmr(self) -> np.ndarray:
        try:
            qpos, elapsed_us = self._ensure_gmr().reset(self.idle_packet, iterations=1000)
            # Idle RESET is only the safe STAND state.  A subsequent dance must
            # still warm from its first real SMPL target, matching old UDP GMR.
            self.gmr_motion.invalidate()
            print(f"[GMR] reset to synthesized stand in {elapsed_us / 1000.0:.1f} ms")
            return qpos
        except Exception:
            self.gmr_motion.invalidate()
            if self.gmr is not None:
                self.gmr.close()
                self.gmr = None
            raise

    def _make_idle_packet(self, device_value: str) -> bytes:
        device = torch.device(device_value if torch.cuda.is_available() else "cpu")
        endecoder = load_endecoder(device)
        idle = synthetic_idle_motion(30.0)
        params = {
            "body_pose": idle.body_pose.to(device),
            "global_orient": idle.global_orient.to(device),
            "transl": idle.transl.to(device),
        }
        payload = smpl_params_to_smp1_payload(
            params,
            endecoder=endecoder,
            adapter=SMPLXGMRReference(user_yaw_deg=0.0, global_scale=1.0),
            encoder=SMP1PacketEncoder(debug=False),
            absolute_start_frame=0,
        )
        del endecoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return payload

    def start(self) -> None:
        self.worker.start()
        self.control.start()
        self._publish_loop()

    def _control_loop(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.args.bind)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        print(f"[Bridge] control endpoint {self.args.bind}")
        try:
            while not self.stop_event.is_set():
                if socket not in dict(poller.poll(100)):
                    continue
                try:
                    parts = socket.recv_multipart()
                    response = self.handle_message(parts)
                except Exception as exc:
                    response = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                socket.send_json(response)
        finally:
            socket.close(linger=0)
            context.term()

    def handle_message(self, parts: list[bytes]) -> dict[str, Any]:
        if len(parts) == 2:
            chunk = RobotStreamChunk.from_multipart(parts)
            with self.lock:
                if self.chunk_queue.full():
                    return {"ok": False, "backpressure": True, **self.status_locked()}
                if self.request is None:
                    raise ValueError("no active request accepts motion chunks")
                if (
                    chunk.checkpoint_sha256 != self.request["checkpoint_sha256"]
                    or chunk.engine_sha256 != self.request["engine_sha256"]
                ):
                    raise ValueError("chunk checkpoint/engine hash changed within request")
                self.tracker.accept(chunk)
                self.chunk_queue.put_nowait(chunk)
                return {"ok": True, "accepted": chunk.chunk_index, **self.status_locked()}
        if len(parts) != 1:
            raise ValueError("control request must contain one JSON frame")
        payload = json.loads(parts[0].decode("utf-8"))
        command = str(payload.get("command", "")).lower()
        if command == "status":
            with self.lock:
                return {"ok": True, **self.status_locked()}
        if command == "heartbeat":
            with self.lock:
                if int(payload.get("revision", -1)) == self.tracker.revision:
                    self.last_heartbeat = time.monotonic()
                return {"ok": True, **self.status_locked()}
        if command == "stand":
            self.request_stand("operator stand")
            with self.lock:
                return {"ok": True, **self.status_locked()}
        if command == "begin":
            return self.begin(payload)
        if command == "shutdown":
            self.shutdown_requested = True
            self.request_stand("shutdown")
            with self.lock:
                return {"ok": True, **self.status_locked()}
        raise ValueError(f"unsupported bridge command: {command}")

    def begin(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.args.estop_file.exists():
            raise RuntimeError(
                f"emergency stop is active: {self.args.estop_file}; remove it before begin"
            )
        if payload.get("contract_version") != "robot_stream_v1":
            raise ValueError("begin requires contract_version=robot_stream_v1")
        if not math.isclose(float(payload.get("source_fps", 0.0)), 30.0):
            raise ValueError("begin source_fps must be 30")
        audio_path = Path(str(payload["audio_path"])).expanduser().resolve(strict=True)
        total_frames = int(payload["total_frames"])
        revision = int(payload["revision"])
        request_id = str(payload["request_id"])
        prime_windows = int(payload.get("prime_windows", 2))
        audio_start_sec = float(payload.get("audio_start_sec", 0.0))
        audio_duration_sec = float(payload["audio_duration_sec"])
        if not request_id:
            raise ValueError("request_id must not be empty")
        if total_frames <= 0:
            raise ValueError("total_frames must be > 0")
        if not math.isfinite(audio_start_sec) or audio_start_sec < 0.0:
            raise ValueError("audio_start_sec must be finite and >= 0")
        if not math.isfinite(audio_duration_sec) or audio_duration_sec <= 0.0:
            raise ValueError("audio_duration_sec must be finite and > 0")
        if not math.isclose(audio_duration_sec * 30.0, total_frames, abs_tol=1e-5):
            raise ValueError("audio duration and total_frames do not describe one timeline")
        checkpoint_sha256 = str(payload["checkpoint_sha256"])
        engine_sha256 = str(payload["engine_sha256"])
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower())
            for value in (checkpoint_sha256, engine_sha256)
        ):
            raise ValueError("checkpoint/engine SHA256 must contain 64 hexadecimal characters")
        if prime_windows not in {1, 2}:
            raise ValueError("prime_windows must be 1 or 2")
        with self.lock:
            if self.state != "STAND":
                raise RuntimeError(f"bridge must be STAND before begin, got {self.state}")
            if revision <= self.tracker.revision:
                raise ValueError("begin revision must be newer than the bridge revision")
            idle_anchor_xy = self.idle_qpos[:2].copy()
        reset_qpos = self._reset_gmr()
        # RESET makes the stateful solver deterministic, but the synthesized
        # stand may already have followed a previous dance in world XY.  Keep
        # that safe world anchor so a new request cannot teleport the robot.
        reset_qpos[:2] = idle_anchor_xy
        idle_frames = qpos_timeline_to_gmt_frames(
            np.repeat(reset_qpos[None], 110, axis=0),
            fps=50.0,
            native_to_gmt=self.native_to_gmt,
        )
        plan_builder = _IncrementalPlanBuilder(
            reset_qpos,
            self.native_to_gmt,
            blend_seconds=self.args.blend_seconds,
            return_seconds=self.args.return_seconds,
        )
        with self.lock:
            self.idle_qpos = reset_qpos
            self.idle_frames = idle_frames
            self.tracker.begin(request_id, revision, total_frames)
            self.plan_builder = plan_builder
            self.plan_snapshot = None
            self.publish_generation += 1
            self.publisher = None
            self.cursor = 0
            self.acked = False
            self.last_ack_sequence = -1
            self.ack_latency_ms = None
            self.audio_started = False
            self.retargeted_frames = 0
            self.gmr_times_ms.clear()
            self.gmr_warmup_ms = None
            self.plan_build_last_ms = None
            self.plan_build_max_ms = 0.0
            self.last_heartbeat = time.monotonic()
            self.request = {
                "request_id": request_id,
                "revision": revision,
                "audio_path": audio_path,
                "audio_start_sec": audio_start_sec,
                "audio_duration_sec": audio_duration_sec,
                "total_frames": total_frames,
                "checkpoint_sha256": checkpoint_sha256,
                "engine_sha256": engine_sha256,
                "prime_windows": prime_windows,
            }
            self.state = "PREPARING"
            return {"ok": True, **self.status_locked()}

    def _retarget_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                qpos_values: list[np.ndarray] = []
                timings: list[float] = []
                for packet in chunk.packets():
                    with self.lock:
                        if chunk.revision != self.tracker.revision:
                            break
                    qpos, elapsed_us, warmup_elapsed_us = self.gmr_motion.retarget(
                        self._ensure_gmr(),
                        packet,
                        revision=chunk.revision,
                    )
                    with self.lock:
                        if chunk.revision != self.tracker.revision:
                            # The operator canceled while a blocking warm-up or
                            # frame solve was running.  Never reuse that state.
                            self.gmr_motion.invalidate()
                            break
                        if warmup_elapsed_us is not None:
                            self.gmr_warmup_ms = warmup_elapsed_us / 1000.0
                            warmup_message = (
                                "[GMR] warmed from first real motion frame "
                                f"for revision={chunk.revision} in "
                                f"{self.gmr_warmup_ms:.1f} ms"
                            )
                        else:
                            warmup_message = None
                    if warmup_message is not None:
                        print(warmup_message)
                    qpos_values.append(qpos)
                    timings.append(elapsed_us / 1000.0)
                if len(qpos_values) != chunk.frame_count:
                    continue
                with self.lock:
                    if chunk.revision != self.tracker.revision:
                        continue
                    builder = self.plan_builder
                    if builder is None:
                        continue
                plan_started = time.perf_counter()
                snapshot = builder.append(
                    np.stack(qpos_values), is_last=chunk.is_last
                )
                plan_elapsed_ms = (time.perf_counter() - plan_started) * 1000.0
                with self.lock:
                    if (
                        chunk.revision != self.tracker.revision
                        or builder is not self.plan_builder
                    ):
                        continue
                    self.plan_snapshot = snapshot
                    self.retargeted_frames += len(qpos_values)
                    self.gmr_times_ms.extend(timings)
                    self.plan_build_last_ms = plan_elapsed_ms
                    self.plan_build_max_ms = max(
                        self.plan_build_max_ms, plan_elapsed_ms
                    )
                    ready_message: str | None = None
                    prime_ready = bool(
                        chunk.is_last
                        or (
                            self.request is not None
                            and chunk.chunk_index + 1 >= self.request["prime_windows"]
                        )
                    )
                    if self.state == "PREPARING" and prime_ready:
                        self.state = "WAIT_ACK"
                        self.publisher = RedisTrajectoryPublisher(
                            self.redis,
                            key=self.args.redis_key,
                            ttl_ms=self.args.redis_ttl_ms,
                        )
                        self.submitted_monotonic = time.monotonic()
                        self.last_ack_monotonic = self.submitted_monotonic
                        self.last_ack_sequence = -1
                        ready_message = (
                            f"[Bridge] first qpos buffer ready: "
                            f"{len(snapshot.frames) / 50.0:.2f}s "
                            f"(incremental plan {plan_elapsed_ms:.1f} ms)"
                        )
                    elif self.state == "PREPARING":
                        self.state = "PRIMING"
                    elif self.state == "PRIMING" and prime_ready:
                        self.state = "WAIT_ACK"
                        self.publisher = RedisTrajectoryPublisher(
                            self.redis,
                            key=self.args.redis_key,
                            ttl_ms=self.args.redis_ttl_ms,
                        )
                        self.submitted_monotonic = time.monotonic()
                        self.last_ack_monotonic = self.submitted_monotonic
                        self.last_ack_sequence = -1
                        ready_message = (
                            f"[Bridge] primed qpos buffer ready: "
                            f"{len(snapshot.frames) / 50.0:.2f}s "
                            f"(incremental plan {plan_elapsed_ms:.1f} ms)"
                        )
                if ready_message is not None:
                    print(ready_message)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"[GMR ERROR] {self.last_error}")
                if self.gmr is not None:
                    self.gmr.close()
                    self.gmr = None
                self.request_stand("GMR failure")
            finally:
                self.chunk_queue.task_done()

    def request_stand(self, reason: str) -> None:
        with self.lock:
            if self.state == "STAND" or self.stand_building:
                return
            self.stand_building = True
            snapshot = self.plan_snapshot
            cursor = self.cursor
            current = self.idle_qpos.copy()
            if snapshot is not None:
                current = snapshot.qpos[min(cursor, len(snapshot.qpos) - 1)].copy()
            target = self.idle_qpos.copy()
            target[:2] = current[:2]
        try:
            transition = _intermediate_qpos(
                current, target, int(round(self.args.return_seconds * 50.0))
            )
            qpos = np.concatenate(
                (current[None], transition, np.repeat(target[None], 101, axis=0)),
                axis=0,
            )
            frames = qpos_timeline_to_gmt_frames(
                qpos, fps=50.0, native_to_gmt=self.native_to_gmt
            )
            terminal_frames = qpos_timeline_to_gmt_frames(
                np.repeat(target[None], 110, axis=0),
                fps=50.0,
                native_to_gmt=self.native_to_gmt,
            )
            for value in (qpos, frames, target, terminal_frames):
                value.setflags(write=False)
            stand_snapshot = _PlanSnapshot(
                qpos=qpos,
                frames=frames,
                audio_start_frame=0,
                audio_end_frame=0,
                action_complete=True,
                terminal_idle_qpos=target,
                terminal_idle_frames=terminal_frames,
            )
        except Exception:
            with self.lock:
                self.stand_building = False
            raise
        with self.lock:
            if self.state == "STAND":
                self.stand_building = False
                return
            self.tracker.revision += 1
            self.publish_generation += 1
            self.plan_builder = None
            self.plan_snapshot = stand_snapshot
            self.cursor = 0
            self.audio_started = False
            self.publisher = RedisTrajectoryPublisher(
                self.redis, key=self.args.redis_key, ttl_ms=self.args.redis_ttl_ms
            )
            self.acked = False
            self.submitted_monotonic = time.monotonic()
            self.last_ack_monotonic = self.submitted_monotonic
            self.last_ack_sequence = -1
            self.ack_latency_ms = None
            self.state = "STAND_WAIT_ACK"
            self.request = None
            self.stand_building = False
        self.audio.stop(reason)
        while True:
            try:
                self.chunk_queue.get_nowait()
                self.chunk_queue.task_done()
            except queue.Empty:
                break
        print(f"[Bridge] {reason}; returning to synthesized stand")

    def status_locked(self) -> dict[str, Any]:
        snapshot = self.plan_snapshot
        future = (
            0.0
            if snapshot is None
            else max(0, len(snapshot.frames) - 1 - self.cursor) / 50.0
        )
        return {
            "state": self.state,
            "request_id": None if self.request is None else self.request["request_id"],
            "revision": self.tracker.revision,
            "accepted_source_frames": self.tracker.next_frame,
            "retargeted_source_frames": self.retargeted_frames,
            "played_50hz_frames": self.cursor,
            "future_buffer_seconds": future,
            "queue_depth": self.chunk_queue.qsize(),
            "action_complete": False if snapshot is None else snapshot.action_complete,
            "gmt_acked": self.acked,
            "gmt_ack_latency_ms": self.ack_latency_ms,
            "gmr_mean_ms": (
                None if not self.gmr_times_ms else float(np.mean(self.gmr_times_ms[-300:]))
            ),
            "gmr_warmup_ms": self.gmr_warmup_ms,
            "incremental_plan_last_ms": self.plan_build_last_ms,
            "incremental_plan_max_ms": self.plan_build_max_ms,
            "gmr_viewer_enabled": self.gmr_viewer is not None,
            "gmr_viewer_alive": (
                False if self.gmr_viewer is None else self.gmr_viewer.alive
            ),
            "gmr_viewer_dropped_frames": (
                0 if self.gmr_viewer is None else self.gmr_viewer.dropped_frames
            ),
            "publish_hz": self.publish_ticks
            / max(1e-6, time.monotonic() - self.publish_started),
            "publish_p99_jitter_ms": self.publish_p99_jitter_ms,
            "publish_max_lock_wait_ms": self.publish_max_lock_wait_ms,
            "last_error": self.last_error,
        }

    def _publish_loop(self) -> None:
        # Initialization includes SMPL/GMR/policy loading and must not dilute
        # the measured 50 Hz publication rate.
        self.publish_ticks = 0
        self.publish_started = time.monotonic()
        self.publish_jitter_ms.clear()
        self.publish_p99_jitter_ms = None
        self.publish_max_lock_wait_ms = 0.0
        previous_tick = self.publish_started
        deadline = MonotonicDeadline(50.0, time.monotonic())
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                delay = deadline.seconds_until(now)
                if delay > 0.0:
                    time.sleep(delay)
                    now = time.monotonic()
                jitter_ms = (
                    None
                    if not self.publish_ticks
                    else abs(now - previous_tick - 0.02) * 1000.0
                )
                previous_tick = now
                skipped = deadline.advance(now)
                if jitter_ms is not None:
                    self.publish_jitter_ms.append(jitter_ms)
                self.publish_ticks += 1
                if self.publish_ticks % 250 == 0 and self.publish_jitter_ms:
                    # Only the publisher owns this deque, so the percentile is
                    # calculated without holding the shared state lock.
                    self.publish_p99_jitter_ms = float(
                        np.percentile(tuple(self.publish_jitter_ms), 99)
                    )

                lock_started = time.perf_counter()
                self.lock.acquire()
                lock_wait_ms = (time.perf_counter() - lock_started) * 1000.0
                self.publish_max_lock_wait_ms = max(
                    self.publish_max_lock_wait_ms, lock_wait_ms
                )
                try:
                    generation = self.publish_generation
                    state = self.state
                    snapshot = self.plan_snapshot
                    cursor = self.cursor
                    revision = self.tracker.revision
                    request = self.request
                    publisher = self.publisher
                    idle_publisher = self.idle_publisher
                    idle_frames = self.idle_frames
                    viewer_qpos = self.idle_qpos
                    heartbeat_failed = bool(
                        request is not None
                        and heartbeat_expired(
                            self.last_heartbeat,
                            now,
                            self.args.heartbeat_timeout_seconds,
                        )
                    )
                    active = state != "STAND"
                    if snapshot is not None:
                        viewer_qpos = snapshot.qpos[
                            min(cursor, len(snapshot.qpos) - 1)
                        ]
                finally:
                    self.lock.release()

                if self.args.estop_file.exists() and active:
                    self.request_stand("ESTOP")
                    continue
                if heartbeat_failed:
                    self.request_stand("heartbeat timeout")
                    continue

                ack = None
                published_plan = False
                if state in {"STAND", "PREPARING", "PRIMING"}:
                    idle_publisher.publish(
                        idle_frames,
                        0,
                        fps=50.0,
                        joint_order_hash=self.contract.joint_order_hash,
                        flags=FLAG_FIXED_IDLE,
                    )
                elif snapshot is not None and publisher is not None:
                    published_plan = True
                    flags = (
                        FLAG_AUDIO
                        if snapshot.audio_start_frame
                        <= cursor
                        < snapshot.audio_end_frame
                        else FLAG_TRANSITION
                    )
                    publisher.publish(
                        snapshot.frames,
                        cursor,
                        fps=50.0,
                        joint_order_hash=self.contract.joint_order_hash,
                        command_revision=revision,
                        plan_id=revision,
                        flags=flags,
                    )
                    ack = publisher.matching_ack()

                if self.gmr_viewer is not None:
                    displayed = self.gmr_viewer.publish(viewer_qpos)
                    if (
                        not displayed
                        and self.gmr_viewer.last_error is not None
                        and not self.gmr_viewer_warning_reported
                    ):
                        print(
                            "[GMR viewer WARNING] "
                            f"{self.gmr_viewer.last_error}; bridge remains active"
                        )
                        self.gmr_viewer_warning_reported = True

                stand_reason: str | None = None
                audio_start: tuple[Path, float, float] | None = None
                audio_stop_reason: str | None = None
                ack_started = False
                became_stand = False
                after_publish = time.monotonic()
                lock_started = time.perf_counter()
                self.lock.acquire()
                lock_wait_ms = (time.perf_counter() - lock_started) * 1000.0
                self.publish_max_lock_wait_ms = max(
                    self.publish_max_lock_wait_ms, lock_wait_ms
                )
                try:
                    # A control command may atomically replace the plan while
                    # Redis I/O is in flight.  Ignore results from that stale
                    # generation; the next 20 ms tick uses the new snapshot.
                    if (
                        published_plan
                        and generation == self.publish_generation
                        and publisher is self.publisher
                        and snapshot is not None
                        and self.plan_snapshot is not None
                    ):
                        live_snapshot = self.plan_snapshot
                        if ack is not None and ack.sequence > self.last_ack_sequence:
                            self.last_ack_sequence = ack.sequence
                            self.last_ack_monotonic = after_publish
                            if not self.acked:
                                self.acked = True
                                self.ack_latency_ms = (
                                    after_publish - self.submitted_monotonic
                                ) * 1000.0
                                self.state = (
                                    "RETURNING"
                                    if self.state == "STAND_WAIT_ACK"
                                    else "TRANSITION"
                                )
                                ack_started = True
                        if not self.acked:
                            if (
                                after_publish - self.submitted_monotonic
                                > self.args.ack_timeout_seconds
                            ):
                                self.last_error = "GMT ACK timeout"
                                if self.state != "STAND_WAIT_ACK":
                                    stand_reason = "GMT ACK timeout"
                        elif (
                            after_publish - self.last_ack_monotonic
                            > self.args.ack_stale_seconds
                        ):
                            self.last_error = "GMT ACK stale"
                            stand_reason = "GMT ACK stale"
                        else:
                            if (
                                self.request is not None
                                and self.cursor >= live_snapshot.audio_start_frame
                                and not self.audio_started
                            ):
                                audio_start = (
                                    self.request["audio_path"],
                                    self.request["audio_start_sec"],
                                    self.request["audio_duration_sec"],
                                )
                                self.audio_started = True
                                self.state = "PLAYING"
                            if (
                                self.request is not None
                                and self.audio_started
                                and self.cursor >= live_snapshot.audio_end_frame
                            ):
                                audio_stop_reason = "audio range complete"
                                self.state = "RETURNING"
                            future_seconds = (
                                len(live_snapshot.frames) - 1 - self.cursor
                            ) / 50.0
                            can_advance = has_complete_publish_context(
                                len(live_snapshot.frames), self.cursor
                            )
                            if not can_advance and not live_snapshot.action_complete:
                                stand_reason = "motion buffer underrun"
                            elif (
                                not live_snapshot.action_complete
                                and future_seconds <= self.args.critical_buffer_seconds
                            ):
                                stand_reason = "critical motion buffer"
                            elif can_advance:
                                self.cursor += 1 + skipped
                            if (
                                self.cursor >= len(live_snapshot.frames) - 101
                                and live_snapshot.action_complete
                            ):
                                audio_stop_reason = "action complete"
                                assert live_snapshot.terminal_idle_qpos is not None
                                assert live_snapshot.terminal_idle_frames is not None
                                self.idle_qpos = (
                                    live_snapshot.terminal_idle_qpos.copy()
                                )
                                self.idle_frames = live_snapshot.terminal_idle_frames
                                self.idle_publisher = RedisTrajectoryPublisher(
                                    self.redis,
                                    key=self.args.redis_key,
                                    ttl_ms=self.args.redis_ttl_ms,
                                )
                                self.plan_builder = None
                                self.plan_snapshot = None
                                self.publisher = None
                                self.request = None
                                self.state = "STAND"
                                self.acked = False
                                self.publish_generation += 1
                                became_stand = True
                    if self.shutdown_requested and self.state == "STAND":
                        self.stop_event.set()
                finally:
                    self.lock.release()

                if ack_started:
                    print("[Bridge] GMT ACK received; playback clock started")
                if audio_start is not None:
                    with self.lock:
                        audio_generation_is_current = bool(
                            generation == self.publish_generation
                            and self.audio_started
                            and self.request is not None
                        )
                    if audio_generation_is_current:
                        self.audio.start(*audio_start)
                        with self.lock:
                            audio_generation_is_current = bool(
                                generation == self.publish_generation
                                and self.audio_started
                                and self.request is not None
                            )
                        if not audio_generation_is_current:
                            self.audio.stop("stale audio generation")
                if audio_stop_reason is not None:
                    self.audio.stop(audio_stop_reason)
                if became_stand:
                    print("[Bridge] synthesized stand is active")
                if stand_reason is not None:
                    self.request_stand(stand_reason)
                if self.args.verbose and self.publish_ticks % 250 == 0:
                    with self.lock:
                        status = self.status_locked()
                    print(f"[Bridge status] {status}")
        finally:
            self.audio.stop("bridge exit")
            if self.gmr is not None:
                self.gmr.close()
            if self.gmr_viewer is not None:
                self.gmr_viewer.close()
            self.stop_event.set()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    runtime = BridgeRuntime(args)
    try:
        runtime.start()
    except KeyboardInterrupt:
        runtime.request_stand("KeyboardInterrupt")
    finally:
        runtime.stop_event.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
