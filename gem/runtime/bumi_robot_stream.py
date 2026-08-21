# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI 原生 qpos28 的实时安全流协议与部署前物理边界检查。

BUMI GENMO 已经直接生成 MuJoCo 原生关节顺序的机器人轨迹，不再经过 SMPL/GMR，因此
不能发送旧的 SMP1 人体目标包。本协议把每个 30 Hz qpos 分块作为小端 float32 负载，
并在 JSON 头中固定 request/revision/绝对帧号、checkpoint 与引擎 SHA256、运动学和关节
顺序指纹以及 CRC32。接收端严格拒绝乱序、重复、缺帧、跨请求哈希变化和损坏负载。

``BumiQposSafetyGate`` 在数据进入 30→50 Hz/GMT 轨迹发布前检查有限值、四元数范数、
根高度、XML 导出的关节上下限、相邻根速度/角速度和关节速度。检查包含前一分块的末帧，
因此分块边界不会成为安全盲区；失败时整块拒绝，不做可能掩盖模型问题的静默裁剪。
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from gem.robots.bumi.kinematics import BumiKinematics

BUMI_QPOS_STREAM_CONTRACT = "bumi_qpos_stream_v1"
BUMI_QPOS_DIM = 28
BUMI_SOURCE_FPS = 30.0


def _validate_sha256(value: str, name: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    return result


def bumi_joint_order_sha256(joint_names: tuple[str, ...] | list[str]) -> str:
    """计算 qpos 负载所用 21 关节顺序的十六进制指纹。"""

    names = tuple(str(name).strip() for name in joint_names)
    if len(names) != 21 or len(set(names)) != 21 or any(not name for name in names):
        raise ValueError("BUMI joint order must contain 21 unique non-empty names")
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BumiQposChunk:
    """一个带完整身份、顺序和校验信息的连续 30 Hz qpos 分块。"""

    request_id: str
    revision: int
    chunk_index: int
    absolute_start_frame: int
    total_frames: int
    is_last: bool
    checkpoint_sha256: str
    engine_sha256: str
    kinematics_sha256: str
    joint_order_sha256: str
    payload: bytes

    def __post_init__(self) -> None:
        if not str(self.request_id).strip():
            raise ValueError("request_id must not be empty")
        if self.revision < 0 or self.chunk_index < 0 or self.absolute_start_frame < 0:
            raise ValueError("revision, chunk_index and absolute_start_frame must be >= 0")
        if self.total_frames <= 0:
            raise ValueError("total_frames must be > 0")
        if len(self.payload) <= 0 or len(self.payload) % (BUMI_QPOS_DIM * 4) != 0:
            raise ValueError("BUMI qpos payload must contain complete float32[28] frames")
        if self.absolute_start_frame + self.frame_count > self.total_frames:
            raise ValueError("BUMI qpos chunk extends beyond total_frames")
        if self.is_last and self.absolute_start_frame + self.frame_count != self.total_frames:
            raise ValueError("last BUMI qpos chunk must end exactly at total_frames")
        if not self.is_last and self.absolute_start_frame + self.frame_count == self.total_frames:
            raise ValueError("a BUMI qpos chunk ending at total_frames must be marked is_last")
        for name in (
            "checkpoint_sha256",
            "engine_sha256",
            "kinematics_sha256",
            "joint_order_sha256",
        ):
            _validate_sha256(getattr(self, name), name)

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
        checkpoint_sha256: str,
        engine_sha256: str,
        kinematics_sha256: str,
        joint_order_sha256: str,
    ) -> BumiQposChunk:
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
            checkpoint_sha256=_validate_sha256(checkpoint_sha256, "checkpoint_sha256"),
            engine_sha256=_validate_sha256(engine_sha256, "engine_sha256"),
            kinematics_sha256=_validate_sha256(kinematics_sha256, "kinematics_sha256"),
            joint_order_sha256=_validate_sha256(joint_order_sha256, "joint_order_sha256"),
            payload=np.ascontiguousarray(values.astype("<f4", copy=False)).tobytes(),
        )

    def qpos(self) -> np.ndarray:
        result = np.frombuffer(self.payload, dtype="<f4").reshape(-1, BUMI_QPOS_DIM).copy()
        if not np.isfinite(result).all():
            raise ValueError("BUMI qpos payload contains NaN or Inf")
        return result

    def header(self) -> dict[str, Any]:
        return {
            "contract_version": BUMI_QPOS_STREAM_CONTRACT,
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
            "checkpoint_sha256": self.checkpoint_sha256,
            "engine_sha256": self.engine_sha256,
            "kinematics_sha256": self.kinematics_sha256,
            "joint_order_sha256": self.joint_order_sha256,
            "payload_crc32": zlib.crc32(self.payload) & 0xFFFF_FFFF,
        }

    def multipart(self) -> list[bytes]:
        return [
            json.dumps(self.header(), sort_keys=True, separators=(",", ":")).encode("utf-8"),
            self.payload,
        ]

    @classmethod
    def from_multipart(cls, parts: list[bytes] | tuple[bytes, ...]) -> BumiQposChunk:
        if len(parts) != 2:
            raise ValueError("BUMI qpos chunk requires one JSON header and one binary payload")
        header = json.loads(bytes(parts[0]).decode("utf-8"))
        payload = bytes(parts[1])
        fixed = {
            "contract_version": BUMI_QPOS_STREAM_CONTRACT,
            "command": "chunk",
            "source_fps": BUMI_SOURCE_FPS,
            "qpos_dim": BUMI_QPOS_DIM,
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
        }
        for name, expected in fixed.items():
            actual = header.get(name)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, abs_tol=1e-8):
                    raise ValueError(f"BUMI qpos header {name} must be {expected}")
            elif actual != expected:
                raise ValueError(f"BUMI qpos header {name} must be {expected!r}")
        if int(header.get("frame_count", -1)) * BUMI_QPOS_DIM * 4 != len(payload):
            raise ValueError("BUMI qpos payload size does not match frame_count")
        if int(header.get("payload_crc32", -1)) != zlib.crc32(payload) & 0xFFFF_FFFF:
            raise ValueError("BUMI qpos payload CRC32 mismatch")
        return cls(
            request_id=str(header["request_id"]),
            revision=int(header["revision"]),
            chunk_index=int(header["chunk_index"]),
            absolute_start_frame=int(header["absolute_start_frame"]),
            total_frames=int(header["total_frames"]),
            is_last=bool(header["is_last"]),
            checkpoint_sha256=str(header["checkpoint_sha256"]),
            engine_sha256=str(header["engine_sha256"]),
            kinematics_sha256=str(header["kinematics_sha256"]),
            joint_order_sha256=str(header["joint_order_sha256"]),
            payload=payload,
        )


class BumiQposRevisionTracker:
    """只接受同一请求内连续、不重复且模型身份不变的分块。"""

    def __init__(self) -> None:
        self.revision = -1
        self.request_id = ""
        self.total_frames = 0
        self.next_chunk = 0
        self.next_frame = 0
        self.identity: tuple[str, str, str, str] | None = None
        self.complete = False

    def begin(self, request_id: str, revision: int, total_frames: int) -> None:
        if int(revision) <= self.revision:
            raise ValueError("new BUMI request must use a newer revision")
        if not str(request_id).strip() or int(total_frames) <= 0:
            raise ValueError("BUMI request_id and total_frames are invalid")
        self.revision = int(revision)
        self.request_id = str(request_id)
        self.total_frames = int(total_frames)
        self.next_chunk = 0
        self.next_frame = 0
        self.identity = None
        self.complete = False

    def accept(self, chunk: BumiQposChunk) -> None:
        if self.complete:
            raise ValueError("BUMI qpos request is already complete")
        if chunk.request_id != self.request_id or chunk.revision != self.revision:
            raise ValueError("stale or foreign BUMI qpos chunk")
        if chunk.total_frames != self.total_frames:
            raise ValueError("BUMI qpos total_frames changed within one request")
        if chunk.chunk_index != self.next_chunk:
            raise ValueError("BUMI qpos chunk index is duplicate or has a gap")
        if chunk.absolute_start_frame != self.next_frame:
            raise ValueError("BUMI qpos absolute frame is duplicate or has a gap")
        identity = (
            chunk.checkpoint_sha256,
            chunk.engine_sha256,
            chunk.kinematics_sha256,
            chunk.joint_order_sha256,
        )
        if self.identity is None:
            self.identity = identity
        elif identity != self.identity:
            raise ValueError("BUMI model/engine/kinematics identity changed within request")
        self.next_chunk += 1
        self.next_frame += chunk.frame_count
        self.complete = bool(chunk.is_last)


class BumiQposSafetyGate:
    """在实时发布边界执行跨分块 qpos 安全检查。"""

    def __init__(
        self,
        kinematics: BumiKinematics,
        *,
        joint_limit_tolerance_rad: float = 0.05,
        max_joint_velocity_radps: float = 18.0,
        max_root_linear_velocity_mps: float = 4.0,
        max_root_angular_velocity_radps: float = 8.0,
        min_root_height_m: float = 0.25,
        max_root_height_m: float = 1.20,
    ) -> None:
        if not isinstance(kinematics, BumiKinematics):
            raise TypeError("BumiQposSafetyGate requires BumiKinematics")
        self.kinematics = kinematics
        values = {
            "joint_limit_tolerance_rad": joint_limit_tolerance_rad,
            "max_joint_velocity_radps": max_joint_velocity_radps,
            "max_root_linear_velocity_mps": max_root_linear_velocity_mps,
            "max_root_angular_velocity_radps": max_root_angular_velocity_radps,
        }
        if any(not math.isfinite(float(value)) or value < 0.0 for value in values.values()):
            raise ValueError("BUMI safety tolerances/speed limits must be finite and >= 0")
        if not 0.0 <= min_root_height_m < max_root_height_m:
            raise ValueError("BUMI root height bounds are invalid")
        self.joint_limit_tolerance_rad = float(joint_limit_tolerance_rad)
        self.max_joint_velocity_radps = float(max_joint_velocity_radps)
        self.max_root_linear_velocity_mps = float(max_root_linear_velocity_mps)
        self.max_root_angular_velocity_radps = float(max_root_angular_velocity_radps)
        self.min_root_height_m = float(min_root_height_m)
        self.max_root_height_m = float(max_root_height_m)
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def validate(self, qpos: np.ndarray) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[1] != BUMI_QPOS_DIM or len(values) <= 0:
            raise ValueError(f"qpos must have shape [T,{BUMI_QPOS_DIM}] with T > 0")
        if not np.isfinite(values).all():
            raise ValueError("BUMI safety gate rejected NaN/Inf qpos")
        quat_norm = np.linalg.norm(values[:, 3:7], axis=1)
        if np.any(np.abs(quat_norm - 1.0) > 2.0e-3):
            raise ValueError("BUMI safety gate rejected non-unit root quaternion")
        values[:, 3:7] /= quat_norm[:, None]
        if self._previous is not None and float(np.dot(self._previous[3:7], values[0, 3:7])) < 0.0:
            values[:, 3:7] *= -1.0
        for index in range(1, len(values)):
            if float(np.dot(values[index - 1, 3:7], values[index, 3:7])) < 0.0:
                values[index, 3:7] *= -1.0
        if np.any(values[:, 2] < self.min_root_height_m) or np.any(
            values[:, 2] > self.max_root_height_m
        ):
            frame = int(np.argmax(np.maximum(self.min_root_height_m - values[:, 2], values[:, 2] - self.max_root_height_m)))
            raise ValueError(
                "BUMI safety gate rejected root height outside configured bounds: "
                f"frame={frame}, value={float(values[frame, 2]):.6f}, "
                f"bounds=[{self.min_root_height_m:.6f},{self.max_root_height_m:.6f}]"
            )
        lower = self.kinematics.joint_lower_limits.detach().cpu().numpy()
        upper = self.kinematics.joint_upper_limits.detach().cpu().numpy()
        joint_values = values[:, 7:]
        lower_excess = lower[None] - self.joint_limit_tolerance_rad - joint_values
        upper_excess = joint_values - (upper[None] + self.joint_limit_tolerance_rad)
        excess = np.maximum(lower_excess, upper_excess)
        if float(excess.max(initial=0.0)) > 0.0:
            frame, joint = np.unravel_index(int(np.argmax(excess)), excess.shape)
            side = "lower" if lower_excess[frame, joint] >= upper_excess[frame, joint] else "upper"
            bound = lower[joint] if side == "lower" else upper[joint]
            raise ValueError(
                "BUMI safety gate rejected XML joint-limit violation: "
                f"frame={frame}, joint={self.kinematics.joint_order[joint]}, "
                f"value={float(joint_values[frame, joint]):.6f}, {side}_bound={float(bound):.6f}, "
                f"tolerance={self.joint_limit_tolerance_rad:.6f}, "
                f"excess_after_tolerance={float(excess[frame, joint]):.6f}"
            )
        sequence = values if self._previous is None else np.concatenate(
            (self._previous[None], values), axis=0
        )
        if len(sequence) > 1:
            root_speed = np.linalg.norm(np.diff(sequence[:, :3], axis=0), axis=1) * BUMI_SOURCE_FPS
            joint_speed = np.abs(np.diff(sequence[:, 7:], axis=0)) * BUMI_SOURCE_FPS
            quat_dot = np.abs(np.sum(sequence[1:, 3:7] * sequence[:-1, 3:7], axis=1))
            angular_speed = 2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0)) * BUMI_SOURCE_FPS
            if float(root_speed.max(initial=0.0)) > self.max_root_linear_velocity_mps:
                frame = int(np.argmax(root_speed))
                raise ValueError(
                    "BUMI safety gate rejected excessive root linear velocity: "
                    f"transition={frame}->{frame + 1}, value={float(root_speed[frame]):.6f}, "
                    f"limit={self.max_root_linear_velocity_mps:.6f}"
                )
            if float(joint_speed.max(initial=0.0)) > self.max_joint_velocity_radps:
                frame, joint = np.unravel_index(int(np.argmax(joint_speed)), joint_speed.shape)
                raise ValueError(
                    "BUMI safety gate rejected excessive joint velocity: "
                    f"transition={frame}->{frame + 1}, joint={self.kinematics.joint_order[joint]}, "
                    f"value={float(joint_speed[frame, joint]):.6f}, "
                    f"limit={self.max_joint_velocity_radps:.6f}"
                )
            if float(angular_speed.max(initial=0.0)) > self.max_root_angular_velocity_radps:
                frame = int(np.argmax(angular_speed))
                raise ValueError(
                    "BUMI safety gate rejected excessive root angular velocity: "
                    f"transition={frame}->{frame + 1}, value={float(angular_speed[frame]):.6f}, "
                    f"limit={self.max_root_angular_velocity_radps:.6f}"
                )
        self._previous = values[-1].copy()
        return values


__all__ = [
    "BUMI_QPOS_DIM",
    "BUMI_QPOS_STREAM_CONTRACT",
    "BUMI_SOURCE_FPS",
    "BumiQposChunk",
    "BumiQposRevisionTracker",
    "BumiQposSafetyGate",
    "bumi_joint_order_sha256",
]
