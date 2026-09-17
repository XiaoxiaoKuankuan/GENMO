"""无需控制器的 BUMI 本地动作播放器，与常驻 GENMO 共用分块输入接口。

只接收内存中的 begin/chunk/status/stand/heartbeat 请求，不创建网络客户端，不读取
GMT 策略。使用训练运动学的默认姿态、现有姿态边界检查和 30→50 Hz 增量计划，按照
单调时钟更新预览快照与音乐。两个有效块预热、版本隔离、缓冲不足返回站姿及退出回收
均在本地完成；状态明确标记 local_monotonic，GMT ACK 不适用，不能冒充控制器验收。
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from pathlib import Path

import numpy as np

from gem.runtime.bumi_audio import AudioController
from gem.runtime.bumi_gmt_plan import BumiIncrementalGmtPlanBuilder, interpolate_qpos
from gem.runtime.bumi_online_stream import (
    BUMI_ONLINE_QPOS_STREAM_CONTRACT,
    BumiOnlineIdentity,
    BumiOnlineRevisionTracker,
    bumi_joint_order_sha256,
    motion_buffer_failure,
)
from gem.runtime.bumi_preview import PoseSnapshot
from gem.runtime.bumi_robot_stream import BumiQposSafetyGate


class LocalBumiPlayer:
    """动作接收端；播放线程只操作本地姿态及本对象创建的音频进程。"""

    def __init__(self, kinematics, kinematics_path, *, audio_mode="ffplay", autostart=True):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.kinematics_sha = hashlib.sha256(Path(kinematics_path).read_bytes()).hexdigest()
        self.joint_sha = bumi_joint_order_sha256(kinematics.joint_order)
        self.idle = kinematics.default_qpos.detach().cpu().numpy().copy()
        self.current_qpos = self.idle.copy()
        self.pose = PoseSnapshot(self.kinematics_sha)
        self.tracker = BumiOnlineRevisionTracker()
        self.audio = AudioController(audio_mode)
        self.safety = BumiQposSafetyGate(
            kinematics,
            joint_limit_tolerance_rad=0.576,
            max_joint_velocity_radps=25.92,
            max_root_linear_velocity_mps=5.76,
            max_root_angular_velocity_radps=14.4,
            min_root_height_m=0.25 / 1.2,
            max_root_height_m=1.44,
        )
        self.builder = None
        self.snapshot = None
        self.active = None
        self.return_qpos = None
        self.state = "STAND"
        self.cursor = 0
        self.started_at = 0.0
        self.audio_started = False
        self.last_error = None
        self.last_stand_reason = None
        self.thread = threading.Thread(target=self._loop, name="bumi-local-playback", daemon=True)
        if autostart:
            self.thread.start()

    def _status(self):
        future = (
            0.0
            if self.snapshot is None
            else max(0, len(self.snapshot.qpos) - 1 - self.cursor) / 50.0
        )
        return dict(
            ok=True,
            runtime_mode="preview",
            playback_mode="realtime",
            playback_clock="local_monotonic",
            state=self.state,
            revision=self.tracker.revision,
            request_id=None if self.active is None else self.active["request_id"],
            accepted_chunks=self.tracker.next_chunk,
            accepted_source_frames=self.tracker.next_frame,
            played_50hz_frames=self.cursor,
            future_buffer_seconds=future,
            action_complete=self.snapshot is not None and self.snapshot.action_complete,
            gmt_acked=None,
            last_error=self.last_error,
            last_stand_reason=self.last_stand_reason,
        )

    def request(self, payload):
        command = payload.get("command")
        if command == "preview_frame":
            return self.pose.read()
        with self.lock:
            if command in {"status", "heartbeat"}:
                return self._status()
            if command == "begin":
                return self._begin(payload)
            if command in {"stand", "shutdown"}:
                self._stand("operator " + command)
                return self._status()
            raise ValueError(f"独立播放器不支持 {command}")

    def _begin(self, payload):
        if self.state != "STAND":
            raise ValueError("独立播放器必须先返回 STAND")
        if (
            payload.get("contract_version") != BUMI_ONLINE_QPOS_STREAM_CONTRACT
            or payload.get("source_fps") != 30.0
        ):
            raise ValueError("独立播放器需要当前 30 Hz qpos 分块契约")
        if payload.get("playback_mode", "realtime") != "realtime":
            raise ValueError("独立预览使用 realtime 播放")
        identity = BumiOnlineIdentity.from_mapping(payload["identity"])
        if (
            identity.kinematics_sha256 != self.kinematics_sha
            or identity.joint_order_sha256 != self.joint_sha
        ):
            raise ValueError("独立播放器运动学/关节顺序不匹配")
        total = int(payload["total_frames"])
        duration = float(payload["audio_duration_sec"])
        start = float(payload["audio_start_sec"])
        if (
            not math.isfinite(duration)
            or duration <= 0
            or not math.isclose(duration * 30, total, abs_tol=1e-5)
        ):
            raise ValueError("音频时长与动作帧数不一致")
        if not math.isfinite(start) or start < 0 or payload["prime_chunks"] not in (1, 2):
            raise ValueError("音频起点或预生成块数无效")
        audio_path = Path(payload["audio_path"]).resolve(strict=True)
        self.tracker.begin(payload["request_id"], int(payload["revision"]), total, identity)
        self.safety.reset()
        # 计划构建器的 FK/插值是通用数值逻辑；本地只使用其 qpos，关节无需 GMT 重排。
        self.builder = BumiIncrementalGmtPlanBuilder(self.idle, np.arange(21))
        self.snapshot = None
        self.active = dict(payload, audio_path=audio_path)
        self.cursor = 0
        self.audio_started = False
        self.last_error = None
        self.last_stand_reason = None
        self.state = "PREPARING"
        return self._status()

    def chunk(self, chunk):
        with self.lock:
            if self.active is None or self.builder is None:
                raise ValueError("独立播放器没有活动请求")
            self.tracker.accept(chunk)
            builder = self.builder
            revision = self.tracker.revision
        try:
            safe = self.safety.validate(chunk.qpos())
            snapshot = builder.append(safe, is_last=chunk.is_last)
        except Exception as exc:
            with self.lock:
                if builder is self.builder and revision == self.tracker.revision:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._stand("qpos validation/plan failure")
            raise
        with self.lock:
            if builder is not self.builder or revision != self.tracker.revision:
                raise ValueError("独立播放器丢弃已取消任务的迟到块")
            self.snapshot = snapshot
            if self.state in {"PREPARING", "PRIMING"}:
                if self.tracker.next_chunk >= self.active["prime_chunks"] or chunk.is_last:
                    self.started_at = time.monotonic()
                    self.state = "TRANSITION"
                else:
                    self.state = "PRIMING"
            return dict(self._status(), accepted=chunk.chunk_index)

    def _stand(self, reason):
        if self.active is None and self.state == "RETURNING":
            return
        self.audio.stop(reason)
        self.tracker.invalidate()
        self.active = None
        self.builder = None
        self.snapshot = None
        self.last_stand_reason = reason
        self.audio_started = False
        if self.state == "STAND":
            return
        target = self.idle.copy()
        target[:2] = self.current_qpos[:2]
        self.return_qpos = np.concatenate(
            (self.current_qpos[None], interpolate_qpos(self.current_qpos, target, 50), target[None])
        )
        self.started_at = time.monotonic()
        self.cursor = 0
        self.state = "RETURNING"

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            request_id = None if self.active is None else self.active["request_id"]
            if self.return_qpos is not None:
                self.cursor = min(
                    max(0, int((now - self.started_at) * 50)), len(self.return_qpos) - 1
                )
                self.current_qpos = self.return_qpos[self.cursor].copy()
                if self.cursor == len(self.return_qpos) - 1:
                    self.idle = self.current_qpos.copy()
                    self.return_qpos = None
                    self.state = "STAND"
            elif self.snapshot is not None and self.state not in {"PREPARING", "PRIMING"}:
                snapshot = self.snapshot
                self.cursor = min(max(0, int((now - self.started_at) * 50)), len(snapshot.qpos) - 1)
                failure = motion_buffer_failure(
                    num_frames=len(snapshot.qpos),
                    cursor=self.cursor,
                    action_complete=snapshot.action_complete,
                    critical_buffer_seconds=2.2,
                )
                if failure:
                    self.last_error = failure
                    self._stand(failure)
                else:
                    self.current_qpos = snapshot.qpos[self.cursor].copy()
                    if (
                        not self.audio_started
                        and self.cursor >= snapshot.audio_start_frame
                        and self.active
                    ):
                        self.audio.start(
                            self.active["audio_path"],
                            self.active["audio_start_sec"],
                            self.active["audio_duration_sec"],
                        )
                        self.audio_started = True
                        self.state = "PLAYING"
                    if snapshot.action_complete and self.cursor >= snapshot.audio_end_frame:
                        self.audio.stop("action complete")
                        self.state = "RETURNING"
                    if snapshot.action_complete and self.cursor >= len(snapshot.qpos) - 101:
                        self.idle = snapshot.terminal_idle_qpos.copy()
                        self.current_qpos = self.idle.copy()
                        self.snapshot = None
                        self.builder = None
                        self.active = None
                        self.state = "STAND"
            self.pose.record(
                self.current_qpos,
                frame_index=self.cursor,
                revision=self.tracker.revision,
                request_id=request_id,
                state=self.state,
                now=now,
            )

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                self.tick(started)
                self.stop_event.wait(max(0.0, 0.02 - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.last_error = f"本地播放线程退出: {type(exc).__name__}: {exc}"
                self.state = "STAND"
                self.tracker.invalidate()
                self.active = None
            print(f"[Local Player ERROR] {self.last_error}", flush=True)
        finally:
            self.audio.stop("local player exit")

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.audio.stop("local player close")
