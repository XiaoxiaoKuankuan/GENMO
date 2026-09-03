# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI 常驻滚动生成专用的控制语法、身份合约和 qpos 二进制协议。

协议 ``bumi_online_qpos_stream_v1`` 与旧 ``robot_stream_v1``、SMP1/GMR 数据包以及
离线 ``bumi_qpos_stream_v1`` 完全隔离。每个负载是连续的 MuJoCo 原生
``float32[T,28]``、30 Hz、根四元数 ``wxyz``。JSON 头固定 request/revision、绝对帧号、
CRC32，并绑定 checkpoint、ONNX、实际推理 artifact/manifest、stats、kinematics、21
关节顺序、表示、120/30/90 overlap-add 和因果足锁版本；接收端在一个 revision 中要求
身份逐字段不变且帧号无重复、无缺口。

本文件还承载新控制台和新安全桥都需要的少量无状态逻辑：交互命令解析、心跳判定和
50 Hz 单调时钟。它不导入 SMPL、SMPL-X、SMP1、GMR 或旧 ``robot_stream.py``。
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import time
import zlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gem.runtime.gmt_trajectory import TRAJECTORY_PLAN_FRAMES

BUMI_ONLINE_QPOS_STREAM_CONTRACT = "bumi_online_qpos_stream_v1"
BUMI_QPOS_DIM = 28
BUMI_SOURCE_FPS = 30.0


def _validate_sha256(value: str, name: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    return result


def bumi_joint_order_sha256(joint_names: tuple[str, ...] | list[str]) -> str:
    """计算原生 qpos 中 21 个非根关节顺序的稳定指纹。"""

    names = tuple(str(name).strip() for name in joint_names)
    if len(names) != 21 or len(set(names)) != 21 or any(not name for name in names):
        raise ValueError("BUMI joint order must contain 21 unique non-empty names")
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BumiOnlineIdentity:
    """一次播放中不可变化的模型、资产和后处理身份。"""

    inference_backend: str
    checkpoint_sha256: str
    onnx_sha256: str
    inference_artifact_sha256: str
    inference_manifest_sha256: str
    stats_sha256: str
    kinematics_sha256: str
    joint_order_sha256: str
    representation_contract_version: str
    sliding_contract_version: str
    foot_lock_contract_version: str

    def __post_init__(self) -> None:
        if self.inference_backend not in {"onnx", "tensorrt"}:
            raise ValueError("inference_backend must be 'onnx' or 'tensorrt'")
        for name in (
            "checkpoint_sha256",
            "onnx_sha256",
            "inference_artifact_sha256",
            "inference_manifest_sha256",
            "stats_sha256",
            "kinematics_sha256",
            "joint_order_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in (
            "representation_contract_version",
            "sliding_contract_version",
            "foot_lock_contract_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BumiOnlineIdentity:
        if not isinstance(value, Mapping):
            raise ValueError("BUMI online identity must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError(
                f"BUMI online identity keys must be {sorted(expected)}, got {sorted(value)}"
            )
        return cls(**{name: str(value[name]) for name in expected})


@dataclass(frozen=True, slots=True)
class BumiOnlineQposChunk:
    """带完整身份、帧序和 CRC 的一个最终 qpos 后缀。"""

    request_id: str
    revision: int
    chunk_index: int
    absolute_start_frame: int
    total_frames: int
    is_last: bool
    identity: BumiOnlineIdentity
    payload: bytes

    def __post_init__(self) -> None:
        if not str(self.request_id).strip():
            raise ValueError("request_id must not be empty")
        if self.revision < 0 or self.chunk_index < 0 or self.absolute_start_frame < 0:
            raise ValueError("revision, chunk_index and absolute_start_frame must be >= 0")
        if self.total_frames <= 0:
            raise ValueError("total_frames must be > 0")
        if not isinstance(self.identity, BumiOnlineIdentity):
            raise TypeError("identity must be BumiOnlineIdentity")
        if len(self.payload) <= 0 or len(self.payload) % (BUMI_QPOS_DIM * 4) != 0:
            raise ValueError("qpos payload must contain complete float32[28] frames")
        end = self.absolute_start_frame + self.frame_count
        if end > self.total_frames:
            raise ValueError("qpos chunk extends beyond total_frames")
        if self.is_last != (end == self.total_frames):
            raise ValueError("is_last must exactly match the total_frames boundary")

    @property
    def frame_count(self) -> int:
        return len(self.payload) // (BUMI_QPOS_DIM * 4)

    @classmethod
    def from_qpos(
        cls,
        qpos: np.ndarray,
        *,
        request_id: str,
        revision: int,
        chunk_index: int,
        absolute_start_frame: int,
        total_frames: int,
        is_last: bool,
        identity: BumiOnlineIdentity,
    ) -> BumiOnlineQposChunk:
        values = np.asarray(qpos, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != BUMI_QPOS_DIM or len(values) <= 0:
            raise ValueError(f"qpos must have shape [T,{BUMI_QPOS_DIM}] with T > 0")
        if not np.isfinite(values).all():
            raise ValueError("qpos contains NaN or Inf")
        return cls(
            request_id=str(request_id),
            revision=int(revision),
            chunk_index=int(chunk_index),
            absolute_start_frame=int(absolute_start_frame),
            total_frames=int(total_frames),
            is_last=bool(is_last),
            identity=identity,
            payload=np.ascontiguousarray(values.astype("<f4", copy=False)).tobytes(),
        )

    def qpos(self) -> np.ndarray:
        result = np.frombuffer(self.payload, dtype="<f4").reshape(-1, BUMI_QPOS_DIM).copy()
        if not np.isfinite(result).all():
            raise ValueError("qpos payload contains NaN or Inf")
        return result

    def header(self) -> dict[str, Any]:
        return {
            "contract_version": BUMI_ONLINE_QPOS_STREAM_CONTRACT,
            "command": "chunk",
            "request_id": self.request_id,
            "revision": self.revision,
            "chunk_index": self.chunk_index,
            "absolute_start_frame": self.absolute_start_frame,
            "frame_count": self.frame_count,
            "total_frames": self.total_frames,
            "source_fps": BUMI_SOURCE_FPS,
            "qpos_dim": BUMI_QPOS_DIM,
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "is_last": self.is_last,
            "identity": self.identity.as_dict(),
            "payload_crc32": zlib.crc32(self.payload) & 0xFFFF_FFFF,
        }

    def multipart(self) -> list[bytes]:
        return [
            json.dumps(self.header(), sort_keys=True, separators=(",", ":")).encode("utf-8"),
            self.payload,
        ]

    @classmethod
    def from_multipart(cls, parts: list[bytes] | tuple[bytes, ...]) -> BumiOnlineQposChunk:
        if len(parts) != 2:
            raise ValueError("online qpos chunk requires one JSON header and one binary payload")
        header = json.loads(bytes(parts[0]).decode("utf-8"))
        payload = bytes(parts[1])
        fixed = {
            "contract_version": BUMI_ONLINE_QPOS_STREAM_CONTRACT,
            "command": "chunk",
            "source_fps": BUMI_SOURCE_FPS,
            "qpos_dim": BUMI_QPOS_DIM,
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
        }
        for name, expected in fixed.items():
            actual = header.get(name)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, abs_tol=1.0e-8):
                    raise ValueError(f"online qpos header {name} must be {expected}")
            elif actual != expected:
                raise ValueError(f"online qpos header {name} must be {expected!r}")
        if int(header.get("frame_count", -1)) * BUMI_QPOS_DIM * 4 != len(payload):
            raise ValueError("online qpos payload size does not match frame_count")
        if int(header.get("payload_crc32", -1)) != zlib.crc32(payload) & 0xFFFF_FFFF:
            raise ValueError("online qpos payload CRC32 mismatch")
        return cls(
            request_id=str(header["request_id"]),
            revision=int(header["revision"]),
            chunk_index=int(header["chunk_index"]),
            absolute_start_frame=int(header["absolute_start_frame"]),
            total_frames=int(header["total_frames"]),
            is_last=bool(header["is_last"]),
            identity=BumiOnlineIdentity.from_mapping(header["identity"]),
            payload=payload,
        )


class BumiOnlineRevisionTracker:
    """只接受当前 revision 中身份固定且绝对帧连续的在线分块。"""

    def __init__(self) -> None:
        self.revision = -1
        self.request_id = ""
        self.total_frames = 0
        self.next_chunk = 0
        self.next_frame = 0
        self.identity: BumiOnlineIdentity | None = None
        self.complete = False

    def begin(
        self,
        request_id: str,
        revision: int,
        total_frames: int,
        identity: BumiOnlineIdentity,
    ) -> None:
        if int(revision) <= self.revision:
            raise ValueError("new online request must use a newer revision")
        if not str(request_id).strip() or int(total_frames) <= 0:
            raise ValueError("online request_id and total_frames are invalid")
        if not isinstance(identity, BumiOnlineIdentity):
            raise TypeError("identity must be BumiOnlineIdentity")
        self.revision = int(revision)
        self.request_id = str(request_id)
        self.total_frames = int(total_frames)
        self.next_chunk = 0
        self.next_frame = 0
        self.identity = identity
        self.complete = False

    def invalidate(self) -> int:
        """提升 revision 并清空当前帧序，使所有在途旧块立即失效。"""

        self.revision += 1
        self.request_id = ""
        self.total_frames = 0
        self.next_chunk = 0
        self.next_frame = 0
        self.identity = None
        self.complete = False
        return self.revision

    def accept(self, chunk: BumiOnlineQposChunk) -> None:
        if self.complete:
            raise ValueError("online qpos request is already complete")
        if chunk.request_id != self.request_id or chunk.revision != self.revision:
            raise ValueError("stale or foreign online qpos chunk")
        if chunk.total_frames != self.total_frames or chunk.identity != self.identity:
            raise ValueError("online qpos total/identity changed within one request")
        if chunk.chunk_index != self.next_chunk or chunk.absolute_start_frame != self.next_frame:
            raise ValueError("online qpos chunk is duplicate or has a frame/index gap")
        self.next_chunk += 1
        self.next_frame += chunk.frame_count
        self.complete = bool(chunk.is_last)


@dataclass(frozen=True, slots=True)
class ConsoleCommand:
    name: str
    audio_path: Path | None = None
    duration_sec: float | None = None
    full: bool = False
    start_sec: float = 0.0
    seed: int = 42


def parse_console_line(line: str) -> ConsoleCommand | None:
    """解析 ``play/stand/status/quit/shutdown``，普通路径默认视为 play。"""

    stripped = line.strip()
    if not stripped:
        return None
    tokens = shlex.split(stripped)
    if not tokens:
        return None
    command = tokens[0].lower()
    if command in {"stand", "status", "quit", "exit", "shutdown", "help"}:
        if len(tokens) != 1:
            raise ValueError(f"{command} does not accept arguments")
        return ConsoleCommand("quit" if command == "exit" else command)
    if command == "play":
        tokens = tokens[1:]
    if not tokens:
        raise ValueError('use: play "/absolute/song.wav" [seconds|full] [--start S --seed N]')
    path = Path(tokens[0]).expanduser()
    option_start = 1
    if len(tokens) == 1 or tokens[1].startswith("--"):
        full, duration = True, None
    else:
        duration_token = tokens[1].lower()
        full = duration_token == "full"
        duration = None if full else float(duration_token)
        option_start = 2
    if duration is not None and (not math.isfinite(duration) or duration <= 0.0):
        raise ValueError("music duration must be finite and > 0")
    start, seed = 0.0, 42
    index = option_start
    while index < len(tokens):
        if index + 1 >= len(tokens):
            raise ValueError(f"missing value for {tokens[index]}")
        option, value = tokens[index : index + 2]
        if option == "--start":
            start = float(value)
        elif option == "--seed":
            seed = int(value)
        else:
            raise ValueError(f"unknown play option: {option}")
        index += 2
    if not math.isfinite(start) or start < 0.0:
        raise ValueError("--start must be finite and >= 0")
    return ConsoleCommand(
        "play", audio_path=path, duration_sec=duration, full=full, start_sec=start, seed=seed
    )


def heartbeat_expired(
    last_heartbeat: float, now: float | None = None, timeout: float = 1.5
) -> bool:
    if timeout <= 0.0:
        raise ValueError("heartbeat timeout must be > 0")
    current = time.monotonic() if now is None else float(now)
    return current - float(last_heartbeat) > timeout


def gmt_ack_failure(
    *,
    acked: bool,
    submitted_monotonic: float,
    last_ack_monotonic: float,
    now: float,
    ack_timeout_seconds: float,
    ack_stale_seconds: float,
) -> str | None:
    """返回 ACK 启动超时或运行陈旧原因；严格大于阈值才失败。"""

    if ack_timeout_seconds <= 0.0 or ack_stale_seconds <= 0.0:
        raise ValueError("ACK timeout/stale thresholds must be > 0")
    if not acked and float(now) - float(submitted_monotonic) > ack_timeout_seconds:
        return "GMT ACK timeout"
    if acked and float(now) - float(last_ack_monotonic) > ack_stale_seconds:
        return "GMT ACK stale"
    return None


def has_complete_publish_context(num_frames: int, cursor: int) -> bool:
    """要求 GMT future99 以及尾端中心差分额外所需的一帧。"""

    return (
        num_frames > 0 and 0 <= cursor < num_frames and cursor + TRAJECTORY_PLAN_FRAMES < num_frames
    )


def motion_buffer_failure(
    *,
    num_frames: int,
    cursor: int,
    action_complete: bool,
    critical_buffer_seconds: float,
    fps: float = 50.0,
) -> str | None:
    """判断未完成计划是否已经欠载或跌破临界未来缓冲。"""

    if critical_buffer_seconds <= 0.0 or fps <= 0.0:
        raise ValueError("critical buffer and fps must be > 0")
    if action_complete:
        return None
    if not has_complete_publish_context(num_frames, cursor):
        return "motion buffer underrun"
    future_seconds = (int(num_frames) - 1 - int(cursor)) / float(fps)
    if future_seconds <= float(critical_buffer_seconds):
        return "critical motion buffer"
    return None


class MonotonicDeadline:
    """固定频率调度；落后时跳过过期 deadline，禁止突发补发。"""

    def __init__(self, fps: float, start_time: float) -> None:
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("publish fps must be finite and > 0")
        self.period = 1.0 / float(fps)
        self.next_deadline = float(start_time)

    def seconds_until(self, now: float) -> float:
        return max(self.next_deadline - float(now), 0.0)

    def advance(self, now: float) -> int:
        candidate = self.next_deadline + self.period
        skipped = 0
        if candidate <= float(now):
            skipped = int(math.floor((float(now) - candidate) / self.period)) + 1
            candidate += skipped * self.period
        self.next_deadline = candidate
        return skipped


class WatermarkGate:
    """带迟滞的未来缓冲水位门：到高水位暂停，低于低水位才恢复。"""

    def __init__(self, low_seconds: float = 4.0, high_seconds: float = 12.0) -> None:
        if not 0.0 < float(low_seconds) < float(high_seconds):
            raise ValueError("watermarks require 0 < low_seconds < high_seconds")
        self.low_seconds = float(low_seconds)
        self.high_seconds = float(high_seconds)
        self.latched = False

    def should_pause(self, future_seconds: float) -> bool:
        future = float(future_seconds)
        if not math.isfinite(future) or future < 0.0:
            raise ValueError("future_seconds must be finite and >= 0")
        if future >= self.high_seconds:
            self.latched = True
        threshold = self.low_seconds if self.latched else self.high_seconds
        if future < threshold:
            self.latched = False
            return False
        return True


__all__ = [
    "BUMI_ONLINE_QPOS_STREAM_CONTRACT",
    "BUMI_QPOS_DIM",
    "BUMI_SOURCE_FPS",
    "BumiOnlineIdentity",
    "BumiOnlineQposChunk",
    "BumiOnlineRevisionTracker",
    "ConsoleCommand",
    "MonotonicDeadline",
    "WatermarkGate",
    "bumi_joint_order_sha256",
    "gmt_ack_failure",
    "has_complete_publish_context",
    "heartbeat_expired",
    "motion_buffer_failure",
    "parse_console_line",
]
