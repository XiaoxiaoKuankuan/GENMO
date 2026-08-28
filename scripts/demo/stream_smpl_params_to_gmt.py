#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Retarget complete GENMO motions, then stream true rolling windows to BUMI GMT.

This is the strict temporal replacement for ``stream_smpl_params_to_gmr.py``:

1. a complete ``smpl_params.pt`` is passed through the real GMR-CPP BUMI IK;
2. the captured native BUMI ``qpos[T,28]`` is cached and resampled to 50 Hz;
3. root/joint velocities are derived once from the complete qpos timeline;
4. each Redis packet carries ``past 10 + current + future 99`` distinct frames;
5. playback and source audio start only after GMT acknowledges the new stream.

GMT's existing ``trajectory_v1`` loader consumes packet rows 0..20 as its real
``past 10 + current + future 10`` command window.  The legacy 35-float Redis
frame is never published by this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import redis
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.runtime.gmt_trajectory import (  # noqa: E402
    BUMI_QPOS_DIM,
    FLAG_AUDIO,
    FLAG_FIXED_IDLE,
    FLAG_TRANSITION,
    GmtPolicyContract,
    RedisTrajectoryPublisher,
    build_playback_timeline,
    qpos_timeline_to_gmt_frames,
    resample_qpos_timeline,
)
from gem.runtime.motion_streamer import (  # noqa: E402
    MonotonicDeadline,
    MotionWatcher,
    SMPLMotion,
    load_smpl_motion,
    synthetic_idle_motion,
)
from scripts.demo.stream_smpl_params_to_gmr import FFplayAudioController  # noqa: E402

DEFAULT_GMR_ROOT = Path("/home/weili/GMR-CPP_e1jump_lowdpi")
DEFAULT_GMT_POLICY = Path(
    "/home/weili/docker_projects/bumi_GMT_deployment_listao/"
    "bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/"
    "policy/bumi/0724_lab_148500.onnx"
)
CACHE_CONTRACT_VERSION = "genmo_full_gmr_bumi_qpos_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"--{name} must be finite and > 0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--watch_dir", type=Path)
    source.add_argument("--motion", type=Path)
    parser.add_argument("--gmt_policy", type=Path, default=DEFAULT_GMT_POLICY)
    parser.add_argument("--gmr_root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--ik_config", type=Path)
    parser.add_argument("--robot_xml", type=Path)
    parser.add_argument("--idle_motion", type=Path)
    parser.add_argument("--cache_root", type=Path, default=Path("outputs/gmr_bumi_cache"))
    parser.add_argument("--capture_port", type=int, default=17016)
    parser.add_argument("--ground_clearance", type=float, default=0.04)
    parser.add_argument("--redis_host", default="127.0.0.1")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--redis_db", type=int, default=0)
    parser.add_argument("--redis_key", default="gmt_online_frame_bumi")
    parser.add_argument("--redis_ttl_ms", type=int, default=250)
    parser.add_argument("--publish_fps", type=float, default=50.0)
    parser.add_argument("--poll_interval", type=float, default=0.2)
    parser.add_argument("--blend_seconds", type=float, default=0.8)
    parser.add_argument("--return_seconds", type=float, default=1.0)
    parser.add_argument("--ack_timeout_seconds", type=float, default=5.0)
    parser.add_argument("--ack_stale_seconds", type=float, default=1.0)
    parser.add_argument("--estop_file", type=Path, default=Path("/tmp/genmo_estop"))
    parser.add_argument(
        "--source_filter",
        choices=["any", "text_only", "music_only"],
        default="music_only",
    )
    parser.add_argument("--replay_existing", action="store_true")
    parser.add_argument("--audio_playback", choices=["off", "ffplay"], default="off")
    parser.add_argument("--audio_offset_sec", type=float, default=0.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for name in (
        "publish_fps",
        "poll_interval",
        "blend_seconds",
        "return_seconds",
        "ack_timeout_seconds",
        "ack_stale_seconds",
    ):
        _validate_positive(name, float(getattr(args, name)))
    if not math.isclose(args.publish_fps, 50.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("trajectory_v1 BUMI playback is fixed at --publish_fps 50")
    if not 1 <= args.capture_port <= 65535:
        raise ValueError("--capture_port must be in [1, 65535]")
    if not 1 <= args.redis_port <= 65535:
        raise ValueError("--redis_port must be in [1, 65535]")
    if args.redis_ttl_ms <= 40:
        raise ValueError("--redis_ttl_ms must be greater than two 50 Hz periods (40 ms)")
    if not args.redis_key or any(character.isspace() for character in args.redis_key):
        raise ValueError("--redis_key must be non-empty and contain no whitespace")
    if not math.isfinite(args.ground_clearance) or args.ground_clearance < 0.0:
        raise ValueError("--ground_clearance must be finite and >= 0")
    if not math.isfinite(args.audio_offset_sec):
        raise ValueError("--audio_offset_sec must be finite")
    return args


def _resolve_gmr_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.gmr_root.expanduser().resolve(strict=True)
    return {
        "root": root,
        "script": (root / "run_smplx_bumi3.sh").resolve(strict=True),
        "preset": (root / "config/robot_presets/bumi3.json").resolve(strict=True),
        "ik": (
            args.ik_config
            if args.ik_config is not None
            else root / "config/ik_configs/smplx_to_bumi3_auto.json"
        )
        .expanduser()
        .resolve(strict=True),
        "xml": (
            args.robot_xml if args.robot_xml is not None else root / "assets/bumi3/mjcf/bumi3.xml"
        )
        .expanduser()
        .resolve(strict=True),
    }


def _write_synthetic_idle(path: Path) -> Path:
    """Write a two-frame GENMO-format standing pose for real GMR retargeting."""
    if path.is_file():
        load_smpl_motion(path, shape_mode="zero", min_frames=2)
        return path
    idle = synthetic_idle_motion(30.0)
    payload = {
        "body_params_global": {
            "body_pose": idle.body_pose.repeat(2, 1),
            "global_orient": idle.global_orient.repeat(2, 1),
            "transl": idle.transl.repeat(2, 1),
            "betas": idle.betas.repeat(2, 1),
        },
        "fps": 30.0,
        "metadata": {
            "source": "synthetic_idle_arms_down",
            "shape_mode": "zero",
            "contract_version": "genmo_synthetic_idle_v1",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


@dataclass(frozen=True)
class RetargetedAction:
    motion: SMPLMotion
    qpos_50hz: np.ndarray
    source_dir: Path | None
    cache_dir: Path
    cache_key: str


class FullGmrRetargeter:
    """Run complete real-GMR captures with content-addressed qpos caching."""

    def __init__(self, args: argparse.Namespace, paths: dict[str, Path]) -> None:
        self.args = args
        self.paths = paths
        self.cache_root = args.cache_root.expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.capture_script = (PROJECT_ROOT / "scripts/retarget_smplx_to_bumi3_capture.py").resolve(
            strict=True
        )
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def _fingerprint(self, motion_path: Path, fps: float) -> tuple[str, dict[str, Any]]:
        inputs = {
            "contract_version": CACHE_CONTRACT_VERSION,
            "motion_sha256": _sha256_file(motion_path),
            "capture_script_sha256": _sha256_file(self.capture_script),
            "gmr_script_sha256": _sha256_file(self.paths["script"]),
            "ik_sha256": _sha256_file(self.paths["ik"]),
            "xml_sha256": _sha256_file(self.paths["xml"]),
            "preset_sha256": _sha256_file(self.paths["preset"]),
            "source_fps": float(fps),
            "ground_clearance": float(self.args.ground_clearance),
        }
        encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest(), inputs

    @staticmethod
    def _load_cached(
        cache_dir: Path,
        *,
        inputs: dict[str, Any],
        expected_frames: int,
    ) -> np.ndarray | None:
        manifest_path = cache_dir / "pipeline_cache.json"
        qpos_path = cache_dir / "qpos.npy"
        if not manifest_path.is_file() or not qpos_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            qpos = np.load(qpos_path, allow_pickle=False)
        except (OSError, ValueError, TypeError):
            return None
        if manifest.get("inputs") != inputs:
            return None
        if qpos.shape != (expected_frames, BUMI_QPOS_DIM) or not np.isfinite(qpos).all():
            return None
        norms = np.linalg.norm(qpos[:, 3:7], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            return None
        return np.asarray(qpos, dtype=np.float32)

    def retarget(self, motion_path: Path) -> tuple[np.ndarray, SMPLMotion, Path, str]:
        motion = load_smpl_motion(motion_path, shape_mode="zero", min_frames=2)
        source = motion.source_path
        cache_key, inputs = self._fingerprint(source, motion.fps)
        target = self.cache_root / cache_key
        cached = self._load_cached(
            target,
            inputs=inputs,
            expected_frames=motion.num_frames,
        )
        if cached is not None:
            print(f"[GMR cache] HIT {source.name} -> {target.name}")
            return cached, motion, target, cache_key
        if target.exists():
            raise RuntimeError(
                f"GMR cache exists but failed validation: {target}; inspect it before retrying"
            )

        temporary = Path(tempfile.mkdtemp(prefix=f".{cache_key[:12]}-", dir=self.cache_root))
        capture_key = f"genmo_gmr_capture_{os.getpid()}_{cache_key[:12]}"
        command = [
            sys.executable,
            "-u",
            str(self.capture_script),
            "--motion",
            str(source),
            "--output-dir",
            str(temporary),
            "--fps",
            f"{motion.fps:.9g}",
            "--port",
            str(self.args.capture_port),
            "--redis-key",
            capture_key,
            "--redis-host",
            self.args.redis_host,
            "--redis-port",
            str(self.args.redis_port),
            "--redis-db",
            str(self.args.redis_db),
            "--gmr-root",
            str(self.paths["root"]),
            "--ik-config",
            str(self.paths["ik"]),
            "--robot-xml",
            str(self.paths["xml"]),
            "--ground-clearance",
            f"{self.args.ground_clearance:.9g}",
        ]
        print(
            f"[GMR full] Retargeting all {motion.num_frames} SMPL frames "
            f"({motion.num_frames / motion.fps:.2f}s): {source}"
        )
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, start_new_session=True)
        with self._process_lock:
            self._process = process
        try:
            return_code = process.wait()
        finally:
            with self._process_lock:
                if self._process is process:
                    self._process = None
        if return_code != 0:
            raise RuntimeError(
                f"complete GMR retarget exited with code {return_code}; diagnostics: {temporary}"
            )
        qpos = np.load(temporary / "qpos.npy", allow_pickle=False)
        if qpos.shape != (motion.num_frames, BUMI_QPOS_DIM) or not np.isfinite(qpos).all():
            raise RuntimeError(f"invalid qpos produced in {temporary}: {qpos.shape}")
        cache_manifest = {
            "inputs": inputs,
            "source_motion": str(source),
            "frames": motion.num_frames,
            "fps": motion.fps,
            "created_unix_ns": time.time_ns(),
        }
        (temporary / "pipeline_cache.json").write_text(
            json.dumps(cache_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        print(f"[GMR cache] STORED {target}")
        return np.asarray(qpos, dtype=np.float32), motion, target, cache_key

    def prepare(self, motion_path: Path, source_dir: Path | None) -> RetargetedAction:
        qpos, motion, cache_dir, cache_key = self.retarget(motion_path)
        qpos_50hz = resample_qpos_timeline(qpos, motion.fps, self.args.publish_fps)
        return RetargetedAction(motion, qpos_50hz, source_dir, cache_dir, cache_key)

    def cancel(self) -> None:
        """Interrupt only the capture wrapper started by this instance."""
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=8.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)


def _load_native_joint_names(preset_path: Path) -> tuple[str, ...]:
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    names = tuple(str(value) for value in preset["joint_names_mujoco_qpos_order"])
    if len(names) != 21 or len(set(names)) != 21:
        raise ValueError(f"invalid native BUMI joint order in {preset_path}")
    return names


def _align_action_to_idle(action: np.ndarray, idle: np.ndarray) -> np.ndarray:
    """SE(2)-align a complete BUMI action root to the current idle root."""
    from scipy.spatial.transform import Rotation

    result = np.asarray(action, dtype=np.float32).copy()
    idle_qpos = np.asarray(idle, dtype=np.float32).reshape(BUMI_QPOS_DIM)
    source_rotation = Rotation.from_quat(result[0, (4, 5, 6, 3)])
    idle_rotation = Rotation.from_quat(idle_qpos[[4, 5, 6, 3]])
    source_yaw = source_rotation.as_euler("zyx")[0]
    idle_yaw = idle_rotation.as_euler("zyx")[0]
    delta = Rotation.from_euler("z", idle_yaw - source_yaw)
    relative = result[:, :3] - result[:1, :3]
    result[:, :3] = delta.apply(relative).astype(np.float32) + result[:1, :3]
    result[:, 0:2] += idle_qpos[None, 0:2] - result[:1, 0:2]
    orientations = Rotation.from_quat(result[:, (4, 5, 6, 3)])
    xyzw = (delta * orientations).as_quat().astype(np.float32)
    result[:, 3:7] = xyzw[:, (3, 0, 1, 2)]
    return result


@dataclass
class ActivePlan:
    action: RetargetedAction
    qpos: np.ndarray
    frames: np.ndarray
    audio_start_frame: int
    audio_end_frame: int
    publisher: RedisTrajectoryPublisher
    plan_id: int
    command_revision: int
    cursor: int = 0
    acked: bool = False
    submitted_monotonic: float = 0.0
    last_ack_monotonic: float = 0.0
    last_ack_sequence: int = -1
    audio_triggered: bool = False


def _plan_id(cache_key: str, revision: int) -> int:
    digest = hashlib.sha256(f"{cache_key}:{revision}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _make_plan(
    action: RetargetedAction,
    *,
    idle_qpos: np.ndarray,
    args: argparse.Namespace,
    client: redis.Redis,
    contract: GmtPolicyContract,
    native_to_gmt: np.ndarray,
    command_revision: int,
) -> ActivePlan:
    aligned = _align_action_to_idle(action.qpos_50hz, idle_qpos)
    playback = build_playback_timeline(
        aligned,
        idle_qpos,
        fps=args.publish_fps,
        blend_seconds=args.blend_seconds,
        return_seconds=args.return_seconds,
    )
    frames = qpos_timeline_to_gmt_frames(
        playback.qpos,
        fps=args.publish_fps,
        native_to_gmt=native_to_gmt,
    )
    plan_id = _plan_id(action.cache_key, command_revision)
    return ActivePlan(
        action=action,
        qpos=playback.qpos,
        frames=frames,
        audio_start_frame=playback.audio_start_frame,
        audio_end_frame=playback.audio_end_frame,
        publisher=RedisTrajectoryPublisher(
            client,
            key=args.redis_key,
            ttl_ms=args.redis_ttl_ms,
        ),
        plan_id=plan_id,
        command_revision=command_revision,
        submitted_monotonic=time.monotonic(),
    )


def _static_idle_frames(
    idle_qpos: np.ndarray,
    *,
    fps: float,
    native_to_gmt: np.ndarray,
) -> np.ndarray:
    repeated = np.repeat(
        np.asarray(idle_qpos, dtype=np.float32).reshape(1, BUMI_QPOS_DIM),
        2,
        axis=0,
    )
    return qpos_timeline_to_gmt_frames(
        repeated,
        fps=fps,
        native_to_gmt=native_to_gmt,
    )


def _motion_file_from_ready(path: Path) -> Path:
    motion = path / "smpl_params.pt"
    if not motion.is_file():
        raise FileNotFoundError(f"READY directory has no smpl_params.pt: {path}")
    return motion


def _banner(
    args: argparse.Namespace,
    contract: GmtPolicyContract,
    native_names: tuple[str, ...],
) -> None:
    print("=" * 72)
    print("GENMO -> full GMR -> BUMI trajectory_v1 -> GMT")
    print(f"  Input:       {args.watch_dir or args.motion}")
    print(f"  GMT policy:  {contract.path}")
    print(f"  Redis:       {args.redis_host}:{args.redis_port}/{args.redis_db}")
    print(f"  Motion key:  {args.redis_key}")
    print(f"  ACK key:     {args.redis_key}_ack")
    print(f"  Rate/window: {args.publish_fps:g} Hz, past=10 current=1 future=99")
    print(f"  GMR order:   {len(native_names)} native joints (name-validated)")
    print("  GMT command: packet rows 0..20 = true past10/current/future10")
    print("=" * 72)


def run(args: argparse.Namespace) -> int:
    paths = _resolve_gmr_paths(args)
    contract = GmtPolicyContract.from_onnx(args.gmt_policy)
    native_names = _load_native_joint_names(paths["preset"])
    native_to_gmt = contract.native_to_gmt_indices(native_names)
    client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        socket_timeout=1.0,
    )
    if not client.ping():
        raise RuntimeError("Redis did not respond to PING")
    _banner(args, contract, native_names)

    retargeter = FullGmrRetargeter(args, paths)
    idle_source = (
        args.idle_motion.expanduser().resolve(strict=True)
        if args.idle_motion is not None
        else _write_synthetic_idle(retargeter.cache_root / "synthetic_idle_v1.pt")
    )
    idle_raw, idle_motion, _, _ = retargeter.retarget(idle_source)
    idle_qpos = resample_qpos_timeline(idle_raw, idle_motion.fps, args.publish_fps)[0]
    idle_frames = _static_idle_frames(
        idle_qpos,
        fps=args.publish_fps,
        native_to_gmt=native_to_gmt,
    )
    idle_publisher = RedisTrajectoryPublisher(
        client,
        key=args.redis_key,
        ttl_ms=args.redis_ttl_ms,
    )
    audio = FFplayAudioController(
        args.audio_playback,
        offset_sec=args.audio_offset_sec,
    )
    watcher = (
        MotionWatcher(
            args.watch_dir,
            replay_existing=args.replay_existing,
            source_filter=args.source_filter,
            state_filename=".gmt_trajectory_watch_state.json",
        )
        if args.watch_dir is not None
        else None
    )
    pending: deque[tuple[Path, Path | None]] = deque()
    prepared: deque[RetargetedAction] = deque()
    if args.motion is not None:
        pending.append((args.motion.expanduser().resolve(strict=True), None))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="full-gmr")
    future: Future[RetargetedAction] | None = None
    future_source: tuple[Path, Path | None] | None = None
    active: ActivePlan | None = None
    revision = 0
    completed = 0
    deadline = MonotonicDeadline(args.publish_fps, time.monotonic())
    last_poll = -math.inf
    stats_started = time.monotonic()
    stats_packets = 0
    state = "IDLE"
    estop_latched = args.estop_file.exists()

    def discard_active(reason: str, *, consume: bool) -> None:
        nonlocal active, idle_qpos, idle_frames, idle_publisher, state
        if active is None:
            return
        audio.stop(reason)
        if consume and watcher is not None and active.action.source_dir is not None:
            watcher.mark_consumed(active.action.source_dir)
        active = None
        idle_frames = _static_idle_frames(
            idle_qpos,
            fps=args.publish_fps,
            native_to_gmt=native_to_gmt,
        )
        idle_publisher = RedisTrajectoryPublisher(
            client,
            key=args.redis_key,
            ttl_ms=args.redis_ttl_ms,
        )
        state = "IDLE"
        print(f"[Plan] discarded: {reason}; publishing synthesized standing trajectory")

    try:
        while True:
            now = time.monotonic()
            delay = deadline.seconds_until(now)
            if delay > 0.0:
                time.sleep(delay)
                now = time.monotonic()
            skipped = deadline.advance(now)

            if watcher is not None and now - last_poll >= args.poll_interval:
                for source_dir in watcher.scan():
                    try:
                        motion_path = _motion_file_from_ready(source_dir)
                    except Exception as exc:
                        print(f"[Watch ERROR] {source_dir}: {type(exc).__name__}: {exc}")
                        continue
                    pending.append((motion_path, source_dir))
                    print(f"[Watch] queued complete SMPL motion: {motion_path}")
                last_poll = now

            if future is None and pending:
                future_source = pending.popleft()
                motion_path, source_dir = future_source
                future = executor.submit(retargeter.prepare, motion_path, source_dir)
                print(f"[Prepare] full-sequence GMR started: {motion_path}")
            if future is not None and future.done():
                assert future_source is not None
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[Prepare ERROR] {future_source[0]}: {type(exc).__name__}: {exc}")
                    if args.once:
                        raise
                else:
                    prepared.append(result)
                    print(f"[Prepare] READY qpos={result.qpos_50hz.shape} cache={result.cache_dir}")
                future = None
                future_source = None

            estop_now = args.estop_file.exists()
            if estop_now:
                if not estop_latched:
                    print(f"[ESTOP] latched by {args.estop_file}")
                estop_latched = True
                discard_active("ESTOP", consume=True)
                while prepared:
                    dropped = prepared.popleft()
                    if watcher is not None and dropped.source_dir is not None:
                        watcher.mark_consumed(dropped.source_dir)
                state = "ESTOP"

            # Removing ESTOP does not resume old work.  A new prepared action is
            # the explicit reset event, matching the legacy streamer's safety rule.
            if not estop_now and active is None and prepared:
                if estop_latched:
                    print("[ESTOP] reset by a newly prepared action")
                    estop_latched = False
                revision += 1
                active = _make_plan(
                    prepared.popleft(),
                    idle_qpos=idle_qpos,
                    args=args,
                    client=client,
                    contract=contract,
                    native_to_gmt=native_to_gmt,
                    command_revision=revision,
                )
                state = "WAITING_ACK"
                print(
                    f"[Plan] stream={active.publisher.stream_id} plan={active.plan_id} "
                    f"frames={len(active.frames)}; waiting for GMT ACK"
                )

            if active is None:
                idle_publisher.publish(
                    idle_frames,
                    0,
                    fps=args.publish_fps,
                    joint_order_hash=contract.joint_order_hash,
                    flags=FLAG_FIXED_IDLE,
                )
            else:
                phase_flags = (
                    FLAG_AUDIO
                    if active.audio_start_frame <= active.cursor < active.audio_end_frame
                    else FLAG_TRANSITION
                )
                active.publisher.publish(
                    active.frames,
                    active.cursor,
                    fps=args.publish_fps,
                    joint_order_hash=contract.joint_order_hash,
                    command_revision=active.command_revision,
                    plan_id=active.plan_id,
                    flags=phase_flags,
                )
                ack = active.publisher.matching_ack()
                if ack is not None and ack.sequence > active.last_ack_sequence:
                    active.last_ack_sequence = ack.sequence
                    active.last_ack_monotonic = now
                    if not active.acked:
                        active.acked = True
                        state = "PLAYING"
                        if watcher is not None and active.action.source_dir is not None:
                            watcher.mark_consumed(active.action.source_dir)
                        print(
                            f"[ACK] GMT accepted stream={ack.stream_id} "
                            f"sequence={ack.sequence}; playback clock started"
                        )

                if not active.acked:
                    if now - active.submitted_monotonic > args.ack_timeout_seconds:
                        message = (
                            f"GMT did not ACK trajectory_v1 within "
                            f"{args.ack_timeout_seconds:g}s. Check GMT ONLINE mode, "
                            f"Redis key, and that rl_controllers was rebuilt."
                        )
                        discard_active(message, consume=False)
                        if args.once:
                            raise TimeoutError(message)
                else:
                    if now - active.last_ack_monotonic > args.ack_stale_seconds:
                        message = f"GMT ACK stopped for {now - active.last_ack_monotonic:.3f}s"
                        discard_active(message, consume=True)
                        if args.once:
                            raise TimeoutError(message)
                    elif active is not None:
                        if active.cursor >= active.audio_start_frame and not active.audio_triggered:
                            audio.start(active.action.motion)
                            active.audio_triggered = True
                        if active.cursor >= active.audio_end_frame and active.audio_triggered:
                            audio.stop("action ended")
                        if active.cursor >= len(active.frames) - 1:
                            finished = active
                            audio.stop("plan completed")
                            idle_qpos = finished.qpos[-1].copy()
                            if watcher is not None and finished.action.source_dir is not None:
                                watcher.mark_consumed(finished.action.source_dir)
                            active = None
                            completed += 1
                            idle_frames = _static_idle_frames(
                                idle_qpos,
                                fps=args.publish_fps,
                                native_to_gmt=native_to_gmt,
                            )
                            idle_publisher = RedisTrajectoryPublisher(
                                client,
                                key=args.redis_key,
                                ttl_ms=args.redis_ttl_ms,
                            )
                            state = "IDLE"
                            print(
                                "[Done] complete BUMI qpos played; synthesized standing "
                                "trajectory continues at the final floor position"
                            )
                        else:
                            active.cursor = min(
                                active.cursor + 1 + skipped,
                                len(active.frames) - 1,
                            )

            stats_packets += 1
            if now - stats_started >= 5.0:
                cursor = active.cursor if active is not None else 0
                total = len(active.frames) if active is not None else 0
                print(
                    f"[GMT] publish={stats_packets / (now - stats_started):.1f}Hz "
                    f"state={state} cursor={cursor}/{total} "
                    f"prepare={future is not None} queue={len(pending) + len(prepared)}"
                )
                stats_started = now
                stats_packets = 0
            if args.verbose and skipped:
                print(f"[Timing] skipped {skipped} expired 50 Hz tick(s)")

            if args.once and completed >= 1 and active is None and future is None and not prepared:
                # Publish one explicit idle packet before the direct-mode process exits.
                idle_publisher.publish(
                    idle_frames,
                    0,
                    fps=args.publish_fps,
                    joint_order_hash=contract.joint_order_hash,
                    flags=FLAG_FIXED_IDLE,
                )
                print("[Done] --once completed")
                return 0
    except KeyboardInterrupt:
        print("\n[Interrupted] stopping trajectory publisher and GMR capture")
        return 0
    finally:
        audio.close()
        retargeter.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
