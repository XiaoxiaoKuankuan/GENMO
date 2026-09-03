#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI 在线 qpos 到 GMT ``trajectory_v1`` 的常驻安全桥。

桥只接收 ``bumi_online_qpos_stream_v1``：30 Hz MuJoCo 原生 qpos28、wxyz 根四元数和
完整模型/资产/后处理指纹。每块先检查 CRC、revision、连续绝对帧、身份、有限值、根高、
关节限位以及跨块根/关节速度，再做状态保持的 30→50 Hz 线性/SLERP 插值，构造 GMT
110×55 滚动窗并通过 Redis 以 ``trajectory_v1`` 独立 50 Hz 发布。

桥在收到默认两个预生成块后进入 WAIT_ACK；只有匹配 stream/revision/plan 的 GMT ACK
到达后才推进时钟并启动音频。ACK 超时或陈旧、心跳超时、缓冲欠载、急停、安全门失败、
新 ``play`` 或 ``stand`` 都会提升 revision、清空旧请求，并用 smoothstep/SLERP 平滑返回
保持当前 XY 的站姿。本入口不加载人体模型或重定向组件，且 ``--gmt-policy`` 没有隐式
默认值，必须明确绑定当前 GMT 启动实际使用的策略文件。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import redis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.robots.bumi.feature_codec import (  # noqa: E402
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402
from gem.robots.bumi.postprocess import (  # noqa: E402
    BUMI_STREAMING_FOOT_LOCK_CONTRACT_VERSION,
)
from gem.runtime.bumi_gmt_plan import (  # noqa: E402
    BumiGmtPlanSnapshot,
    BumiIncrementalGmtPlanBuilder,
    interpolate_qpos,
)
from gem.runtime.bumi_music_deploy import (  # noqa: E402
    BUMI_SLIDING_QPOS_CONTRACT_VERSION,
)
from gem.runtime.bumi_online_stream import (  # noqa: E402
    BUMI_ONLINE_QPOS_STREAM_CONTRACT,
    BumiOnlineIdentity,
    BumiOnlineQposChunk,
    BumiOnlineRevisionTracker,
    MonotonicDeadline,
    bumi_joint_order_sha256,
    gmt_ack_failure,
    has_complete_publish_context,
    heartbeat_expired,
    motion_buffer_failure,
)
from gem.runtime.bumi_robot_stream import BumiQposSafetyGate  # noqa: E402
from gem.runtime.gmt_trajectory import (  # noqa: E402
    FLAG_AUDIO,
    FLAG_FIXED_IDLE,
    FLAG_TRANSITION,
    GmtPolicyContract,
    RedisTrajectoryPublisher,
    qpos_timeline_to_gmt_frames,
)
from gem.runtime.music_only_trt import sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:7022")
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--gmt-policy", type=Path, required=True)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-key", default="gmt_online_frame_bumi")
    parser.add_argument("--redis-ttl-ms", type=int, default=250)
    parser.add_argument("--audio-playback", choices=("off", "ffplay"), default="ffplay")
    parser.add_argument("--blend-seconds", type=float, default=0.8)
    parser.add_argument("--return-seconds", type=float, default=1.0)
    parser.add_argument("--ack-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--ack-stale-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-timeout-seconds", type=float, default=1.5)
    parser.add_argument("--critical-buffer-seconds", type=float, default=2.2)
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=0.4)
    parser.add_argument("--max-joint-velocity-radps", type=float, default=18.0)
    parser.add_argument("--max-root-linear-velocity-mps", type=float, default=4.0)
    parser.add_argument("--max-root-angular-velocity-radps", type=float, default=8.0)
    parser.add_argument("--min-root-height-m", type=float, default=0.25)
    parser.add_argument("--max-root-height-m", type=float, default=1.20)
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
    if args.redis_ttl_ms <= 40:
        raise ValueError("--redis-ttl-ms must exceed two 50 Hz periods")


class AudioController:
    """非阻塞管理 ffplay，绝不让音频进程阻塞 50 Hz 安全发布线程。"""

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
                print("[Audio WARNING] ffplay 不可用", flush=True)
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

        threading.Thread(target=reap, daemon=True).start()


class BumiOnlineBridge:
    """在线协议、安全门、增量 GMT 计划和发布时钟的状态机。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.shutdown_requested = False
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.tracker = BumiOnlineRevisionTracker()
        self.state = "BOOT"
        self.request: dict[str, Any] | None = None
        self.plan_builder: BumiIncrementalGmtPlanBuilder | None = None
        self.plan_snapshot: BumiGmtPlanSnapshot | None = None
        self.publisher: RedisTrajectoryPublisher | None = None
        self.publish_generation = 0
        self.cursor = 0
        self.accepted_chunks = 0
        self.last_heartbeat = time.monotonic()
        self.submitted_monotonic = 0.0
        self.last_ack_monotonic = 0.0
        self.last_ack_sequence = -1
        self.acked = False
        self.ack_latency_ms: float | None = None
        self.audio_started = False
        self.stand_building = False
        self.last_error: str | None = None
        # ``last_error`` 记录最近一次协议异常，迟到块可能更新它；安全返回的首要原因必须
        # 独立保存，直到下一次 BEGIN 才清空，避免根因被次生的 stale/no-active 错误覆盖。
        self.last_stand_reason: str | None = None
        self.plan_build_last_ms: float | None = None
        self.plan_build_max_ms = 0.0
        self.publish_ticks = 0
        self.publish_started = time.monotonic()
        self.publish_jitter_ms: deque[float] = deque(maxlen=30_000)
        self.publish_p99_jitter_ms: float | None = None

        self.kinematics_path = args.kinematics.expanduser().resolve(strict=True)
        self.gmt_policy_path = args.gmt_policy.expanduser().resolve(strict=True)
        self.kinematics = BumiKinematics(self.kinematics_path)
        self.kinematics_sha256 = sha256_file(self.kinematics_path)
        self.joint_order_sha256 = bumi_joint_order_sha256(self.kinematics.joint_order)
        self.contract = GmtPolicyContract.from_onnx(self.gmt_policy_path)
        self.gmt_policy_sha256 = sha256_file(self.gmt_policy_path)
        self.native_to_gmt = self.contract.native_to_gmt_indices(self.kinematics.joint_order)
        self.idle_qpos = self.kinematics.default_qpos.detach().cpu().numpy().copy()
        self.idle_qpos[7:] = self.contract.default_in_native_order(self.kinematics.joint_order)
        self.idle_frames = qpos_timeline_to_gmt_frames(
            np.repeat(self.idle_qpos[None], 110, axis=0),
            fps=50.0,
            native_to_gmt=self.native_to_gmt,
        )
        self.safety = BumiQposSafetyGate(
            self.kinematics,
            joint_limit_tolerance_rad=args.joint_limit_tolerance_rad,
            max_joint_velocity_radps=args.max_joint_velocity_radps,
            max_root_linear_velocity_mps=args.max_root_linear_velocity_mps,
            max_root_angular_velocity_radps=args.max_root_angular_velocity_radps,
            min_root_height_m=args.min_root_height_m,
            max_root_height_m=args.max_root_height_m,
        )
        self.redis = redis.Redis(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            socket_timeout=1.0,
        )
        if not self.redis.ping():
            raise RuntimeError("Redis did not respond to PING")
        self.idle_publisher = RedisTrajectoryPublisher(
            self.redis, key=args.redis_key, ttl_ms=args.redis_ttl_ms
        )
        self.audio = AudioController(args.audio_playback)
        self.state = "STAND"

    def start(self) -> None:
        self.control_thread.start()
        self._publish_loop()

    def _control_loop(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.args.bind)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        print(f"[BUMI Bridge] control endpoint {self.args.bind}", flush=True)
        try:
            while not self.stop_event.is_set():
                if socket not in dict(poller.poll(100)):
                    continue
                try:
                    response = self.handle_message(socket.recv_multipart())
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
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
            try:
                chunk = BumiOnlineQposChunk.from_multipart(parts)
            except Exception:
                self.request_stand("corrupt online qpos chunk")
                raise
            try:
                return self.accept_chunk(chunk)
            except Exception:
                with self.lock:
                    matches_active = bool(
                        self.request is not None
                        and chunk.revision == self.tracker.revision
                        and chunk.request_id == self.tracker.request_id
                    )
                if matches_active:
                    self.request_stand("online qpos chunk rejection")
                raise
        if len(parts) != 1:
            raise ValueError("control request must contain one JSON frame")
        payload = json.loads(parts[0].decode("utf-8"))
        command = str(payload.get("command", "")).lower()
        if command == "status":
            with self.lock:
                return {"ok": True, **self.status_locked()}
        if command == "heartbeat":
            with self.lock:
                if (
                    int(payload.get("revision", -1)) == self.tracker.revision
                    and str(payload.get("request_id", "")) == self.tracker.request_id
                ):
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
        if payload.get("contract_version") != BUMI_ONLINE_QPOS_STREAM_CONTRACT:
            raise ValueError(f"begin requires contract_version={BUMI_ONLINE_QPOS_STREAM_CONTRACT}")
        if not math.isclose(float(payload.get("source_fps", 0.0)), 30.0):
            raise ValueError("begin source_fps must be 30")
        identity = BumiOnlineIdentity.from_mapping(payload["identity"])
        if identity.representation_contract_version != BUMI_REPRESENTATION_CONTRACT_VERSION:
            raise ValueError("begin uses an unsupported BUMI representation contract")
        if identity.sliding_contract_version != BUMI_SLIDING_QPOS_CONTRACT_VERSION:
            raise ValueError("begin uses an unsupported BUMI overlap-add contract")
        if identity.foot_lock_contract_version not in {
            BUMI_STREAMING_FOOT_LOCK_CONTRACT_VERSION,
            "disabled",
        }:
            raise ValueError("begin uses an unsupported BUMI foot-lock contract")
        if identity.kinematics_sha256 != self.kinematics_sha256:
            raise ValueError("begin kinematics SHA does not match bridge --kinematics")
        if identity.joint_order_sha256 != self.joint_order_sha256:
            raise ValueError("begin native joint-order SHA does not match bridge kinematics")
        audio_path = Path(str(payload["audio_path"])).expanduser().resolve(strict=True)
        total_frames = int(payload["total_frames"])
        revision = int(payload["revision"])
        request_id = str(payload["request_id"])
        prime_chunks = int(payload.get("prime_chunks", 2))
        audio_start_sec = float(payload.get("audio_start_sec", 0.0))
        audio_duration_sec = float(payload["audio_duration_sec"])
        if prime_chunks not in {1, 2}:
            raise ValueError("prime_chunks must be 1 or 2")
        if total_frames <= 0 or not request_id:
            raise ValueError("request_id and total_frames are invalid")
        if not math.isfinite(audio_start_sec) or audio_start_sec < 0.0:
            raise ValueError("audio_start_sec must be finite and >= 0")
        if not math.isfinite(audio_duration_sec) or audio_duration_sec <= 0.0:
            raise ValueError("audio_duration_sec must be finite and > 0")
        if not math.isclose(audio_duration_sec * 30.0, total_frames, abs_tol=1.0e-5):
            raise ValueError("audio duration and total_frames do not describe one timeline")
        with self.lock:
            if self.state != "STAND":
                raise RuntimeError(f"bridge must be STAND before begin, got {self.state}")
            self.tracker.begin(request_id, revision, total_frames, identity)
            self.safety.reset()
            self.plan_builder = BumiIncrementalGmtPlanBuilder(
                self.idle_qpos,
                self.native_to_gmt,
                blend_seconds=self.args.blend_seconds,
                return_seconds=self.args.return_seconds,
            )
            self.plan_snapshot = None
            self.publisher = None
            self.publish_generation += 1
            self.cursor = 0
            self.accepted_chunks = 0
            self.last_heartbeat = time.monotonic()
            self.acked = False
            self.last_ack_sequence = -1
            self.ack_latency_ms = None
            self.audio_started = False
            self.plan_build_last_ms = None
            self.plan_build_max_ms = 0.0
            self.last_error = None
            self.last_stand_reason = None
            self.request = {
                "request_id": request_id,
                "revision": revision,
                "audio_path": audio_path,
                "audio_start_sec": audio_start_sec,
                "audio_duration_sec": audio_duration_sec,
                "total_frames": total_frames,
                "prime_chunks": prime_chunks,
                "identity": identity,
            }
            self.state = "PREPARING"
            return {"ok": True, **self.status_locked()}

    def accept_chunk(self, chunk: BumiOnlineQposChunk) -> dict[str, Any]:
        with self.lock:
            if self.request is None or self.plan_builder is None:
                raise ValueError("no active request accepts qpos chunks")
            self.tracker.accept(chunk)
            revision = self.tracker.revision
            builder = self.plan_builder
        try:
            safe_qpos = self.safety.validate(chunk.qpos())
            started = time.perf_counter()
            snapshot = builder.append(safe_qpos, is_last=chunk.is_last)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            self.request_stand(f"qpos safety/plan failure: {type(exc).__name__}: {exc}")
            raise
        with self.lock:
            if revision != self.tracker.revision or builder is not self.plan_builder:
                raise ValueError("qpos chunk became stale while its plan was being built")
            # 一个通过 CRC、revision、身份、安全门和增量计划检查的 qpos 块本身就是强
            # 存活证据。即使独立心跳线程恰逢调度延迟，也不应在连续生成期间误判超时。
            self.last_heartbeat = time.monotonic()
            self.plan_snapshot = snapshot
            self.accepted_chunks += 1
            self.plan_build_last_ms = elapsed_ms
            self.plan_build_max_ms = max(self.plan_build_max_ms, elapsed_ms)
            prime_ready = self.accepted_chunks >= self.request["prime_chunks"] or chunk.is_last
            if prime_ready and self.state in {"PREPARING", "PRIMING"}:
                self.state = "WAIT_ACK"
                self.publisher = RedisTrajectoryPublisher(
                    self.redis, key=self.args.redis_key, ttl_ms=self.args.redis_ttl_ms
                )
                self.submitted_monotonic = time.monotonic()
                self.last_ack_monotonic = self.submitted_monotonic
            elif self.state == "PREPARING":
                self.state = "PRIMING"
            return {"ok": True, "accepted": chunk.chunk_index, **self.status_locked()}

    def request_stand(self, reason: str) -> None:
        with self.lock:
            if self.stand_building:
                return
            if self.state == "STAND":
                self.tracker.invalidate()
                self.request = None
                return
            if self.state == "STAND_WAIT_ACK" and self.request is None:
                # 原始失败已经使活动 revision 失效；生成线程随后补发的 stand 只是清理
                # 动作，不应再次构造返回轨迹或重复提升 revision。
                return
            # 只让首次导致安全返回的原因成为本轮根因。返回途中收到的迟到块、重复 stand
            # 或 ACK 次生错误仍可写入 last_error，但不得覆盖这里。
            if self.request is not None or self.last_stand_reason is None:
                self.last_stand_reason = str(reason)
            self.stand_building = True
            snapshot = self.plan_snapshot
            current = self.idle_qpos.copy()
            if snapshot is not None and len(snapshot.qpos):
                current = snapshot.qpos[min(self.cursor, len(snapshot.qpos) - 1)].copy()
            target = self.idle_qpos.copy()
            target[:2] = current[:2]
        try:
            transition = interpolate_qpos(
                current, target, int(round(self.args.return_seconds * 50.0))
            )
            qpos = np.concatenate(
                (current[None], transition, np.repeat(target[None], 101, axis=0)), axis=0
            )
            frames = qpos_timeline_to_gmt_frames(qpos, fps=50.0, native_to_gmt=self.native_to_gmt)
            terminal_frames = qpos_timeline_to_gmt_frames(
                np.repeat(target[None], 110, axis=0),
                fps=50.0,
                native_to_gmt=self.native_to_gmt,
            )
            for value in (qpos, frames, target, terminal_frames):
                value.setflags(write=False)
            stand_snapshot = BumiGmtPlanSnapshot(
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
            self.tracker.invalidate()
            self.publish_generation += 1
            self.plan_builder = None
            self.plan_snapshot = stand_snapshot
            self.publisher = RedisTrajectoryPublisher(
                self.redis, key=self.args.redis_key, ttl_ms=self.args.redis_ttl_ms
            )
            self.cursor = 0
            self.acked = False
            self.last_ack_sequence = -1
            self.ack_latency_ms = None
            self.audio_started = False
            self.submitted_monotonic = time.monotonic()
            self.last_ack_monotonic = self.submitted_monotonic
            self.request = None
            self.state = "STAND_WAIT_ACK"
            self.stand_building = False
        self.audio.stop(reason)
        print(f"[BUMI Bridge] {reason}; 正在平滑返回站姿", flush=True)

    def status_locked(self) -> dict[str, Any]:
        snapshot = self.plan_snapshot
        future = 0.0 if snapshot is None else max(0, len(snapshot.frames) - 1 - self.cursor) / 50.0
        return {
            "state": self.state,
            "request_id": None if self.request is None else self.request["request_id"],
            "revision": self.tracker.revision,
            "accepted_source_frames": self.tracker.next_frame,
            "accepted_chunks": self.accepted_chunks,
            "backpressure_mode": "synchronous_one_chunk_in_flight",
            "played_50hz_frames": self.cursor,
            "future_buffer_seconds": future,
            "action_complete": False if snapshot is None else snapshot.action_complete,
            "gmt_acked": self.acked,
            "gmt_ack_latency_ms": self.ack_latency_ms,
            "incremental_plan_last_ms": self.plan_build_last_ms,
            "incremental_plan_max_ms": self.plan_build_max_ms,
            "joint_limit_tolerance_rad": self.args.joint_limit_tolerance_rad,
            "kinematics_sha256": self.kinematics_sha256,
            "joint_order_sha256": self.joint_order_sha256,
            "gmt_policy_sha256": self.gmt_policy_sha256,
            "active_identity": None if self.request is None else self.request["identity"].as_dict(),
            "publish_hz": self.publish_ticks / max(1.0e-6, time.monotonic() - self.publish_started),
            "publish_p99_jitter_ms": self.publish_p99_jitter_ms,
            "last_stand_reason": self.last_stand_reason,
            "last_error": self.last_error,
        }

    def _become_stand(self, snapshot: BumiGmtPlanSnapshot) -> None:
        assert snapshot.terminal_idle_qpos is not None
        assert snapshot.terminal_idle_frames is not None
        self.idle_qpos = snapshot.terminal_idle_qpos.copy()
        self.idle_frames = snapshot.terminal_idle_frames
        self.idle_publisher = RedisTrajectoryPublisher(
            self.redis, key=self.args.redis_key, ttl_ms=self.args.redis_ttl_ms
        )
        self.plan_builder = None
        self.plan_snapshot = None
        self.publisher = None
        self.request = None
        self.state = "STAND"
        self.acked = False
        self.publish_generation += 1

    def _force_stand_after_missing_ack(self, snapshot: BumiGmtPlanSnapshot) -> None:
        """返回站姿流无人 ACK 时退回固定 idle 发布，避免 shutdown 永久等待。"""

        self._become_stand(snapshot)
        self.last_error = "GMT did not ACK stand transition; fixed idle publication resumed"

    def _publish_loop(self) -> None:
        self.publish_ticks = 0
        self.publish_started = time.monotonic()
        previous_tick = self.publish_started
        deadline = MonotonicDeadline(50.0, self.publish_started)
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                delay = deadline.seconds_until(now)
                if delay > 0.0:
                    time.sleep(delay)
                    now = time.monotonic()
                jitter = (
                    None if not self.publish_ticks else abs(now - previous_tick - 0.02) * 1000.0
                )
                previous_tick = now
                skipped = deadline.advance(now)
                if jitter is not None:
                    self.publish_jitter_ms.append(jitter)
                self.publish_ticks += 1
                if self.publish_ticks % 250 == 0 and self.publish_jitter_ms:
                    self.publish_p99_jitter_ms = float(np.percentile(self.publish_jitter_ms, 99))

                with self.lock:
                    generation = self.publish_generation
                    state = self.state
                    snapshot = self.plan_snapshot
                    cursor = self.cursor
                    revision = self.tracker.revision
                    request = self.request
                    publisher = self.publisher
                    heartbeat_failed = bool(
                        request is not None
                        and heartbeat_expired(
                            self.last_heartbeat, now, self.args.heartbeat_timeout_seconds
                        )
                    )
                if self.args.estop_file.exists() and state != "STAND":
                    self.request_stand("ESTOP")
                    continue
                if heartbeat_failed:
                    self.request_stand("heartbeat timeout")
                    continue

                ack = None
                published_plan = False
                if state in {"STAND", "PREPARING", "PRIMING"}:
                    self.idle_publisher.publish(
                        self.idle_frames,
                        0,
                        fps=50.0,
                        joint_order_hash=self.contract.joint_order_hash,
                        flags=FLAG_FIXED_IDLE,
                    )
                elif snapshot is not None and publisher is not None:
                    published_plan = True
                    flags = (
                        FLAG_AUDIO
                        if snapshot.audio_start_frame <= cursor < snapshot.audio_end_frame
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

                stand_reason: str | None = None
                audio_start: tuple[Path, float, float] | None = None
                audio_stop = False
                became_stand = False
                after_publish = time.monotonic()
                with self.lock:
                    if (
                        # PREPARING/PRIMING 只发布固定站姿，此时动作 publisher 尚为
                        # None。不能用 ``publisher is self.publisher`` 单独判断，因为
                        # ``None is None`` 也成立，会在第一块计划生成后误跑 ACK 超时。
                        published_plan
                        and generation == self.publish_generation
                        and publisher is self.publisher
                        and snapshot is not None
                        and self.plan_snapshot is not None
                    ):
                        live = self.plan_snapshot
                        if ack is not None and ack.sequence > self.last_ack_sequence:
                            self.last_ack_sequence = ack.sequence
                            self.last_ack_monotonic = after_publish
                            if not self.acked:
                                self.acked = True
                                self.ack_latency_ms = (
                                    after_publish - self.submitted_monotonic
                                ) * 1000.0
                                self.state = (
                                    "RETURNING" if state == "STAND_WAIT_ACK" else "TRANSITION"
                                )
                        ack_failure = gmt_ack_failure(
                            acked=self.acked,
                            submitted_monotonic=self.submitted_monotonic,
                            last_ack_monotonic=self.last_ack_monotonic,
                            now=after_publish,
                            ack_timeout_seconds=self.args.ack_timeout_seconds,
                            ack_stale_seconds=self.args.ack_stale_seconds,
                        )
                        if ack_failure is not None:
                            self.last_error = ack_failure
                            if state == "STAND_WAIT_ACK" and ack_failure == "GMT ACK timeout":
                                self._force_stand_after_missing_ack(live)
                                became_stand = True
                            else:
                                stand_reason = ack_failure
                        else:
                            if (
                                self.request is not None
                                and self.cursor >= live.audio_start_frame
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
                                and self.cursor >= live.audio_end_frame
                            ):
                                audio_stop = True
                                self.state = "RETURNING"
                            can_advance = has_complete_publish_context(
                                len(live.frames), self.cursor
                            )
                            buffer_failure = motion_buffer_failure(
                                num_frames=len(live.frames),
                                cursor=self.cursor,
                                action_complete=live.action_complete,
                                critical_buffer_seconds=self.args.critical_buffer_seconds,
                            )
                            if buffer_failure is not None:
                                stand_reason = buffer_failure
                            elif can_advance:
                                self.cursor = min(self.cursor + 1 + skipped, len(live.frames) - 1)
                            if self.cursor >= len(live.frames) - 101 and live.action_complete:
                                audio_stop = True
                                self._become_stand(live)
                                became_stand = True
                    if self.shutdown_requested and self.state == "STAND":
                        self.stop_event.set()
                if audio_start is not None:
                    self.audio.start(*audio_start)
                if audio_stop:
                    self.audio.stop("audio/action complete")
                if stand_reason is not None:
                    self.request_stand(stand_reason)
                if became_stand:
                    print("[BUMI Bridge] 站姿已生效", flush=True)
                if self.args.verbose and self.publish_ticks % 250 == 0:
                    with self.lock:
                        print(f"[BUMI Bridge status] {self.status_locked()}", flush=True)
        finally:
            self.audio.stop("bridge exit")
            self.stop_event.set()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    runtime = BumiOnlineBridge(args)
    try:
        runtime.start()
    except KeyboardInterrupt:
        runtime.request_stand("KeyboardInterrupt")
    finally:
        runtime.stop_event.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
