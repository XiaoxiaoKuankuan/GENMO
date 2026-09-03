# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI qpos 分块到 GMT 50 Hz 滚动指令窗的增量计划构建器。

生成模型按 30 Hz、120/30 滑窗提交世界系 qpos28；GMT 策略按 50 Hz 消费 10 帧历史、
当前帧和 99 帧未来组成的 110×55 轨迹包。本模块保留跨分块插值边界，根平移/关节使用
线性插值，wxyz 四元数使用 SLERP，并只重算会受新样本影响的末端导数。首段相对安全
站立姿态做平面位置/航向对齐和缓入，末段回到保持当前 XY 的站立姿态并补足未来上下文。

每次 ``append`` 返回不可写快照，50 Hz 发布线程可以在不持有控制锁的情况下读取它；
这既避免 Redis I/O 阻塞生成线程，也保证 revision 切换后不会修改已经发布的历史。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gem.runtime.gmt_trajectory import (
    BUMI_QPOS_DIM,
    IncrementalGmtFrameTimeline,
    qpos_timeline_to_gmt_frames,
)
from gem.runtime.qpos_timeline import IncrementalQposTimeline


def interpolate_qpos(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    """用 smoothstep 与 SLERP 生成不含首末端点的 BUMI 过渡帧。"""

    if int(count) <= 0:
        return np.empty((0, BUMI_QPOS_DIM), dtype=np.float32)
    first = np.asarray(start, dtype=np.float32).reshape(BUMI_QPOS_DIM)
    last = np.asarray(end, dtype=np.float32).reshape(BUMI_QPOS_DIM)
    if not np.isfinite(first).all() or not np.isfinite(last).all():
        raise ValueError("qpos transition endpoints contain NaN or Inf")
    alpha = np.arange(1, int(count) + 1, dtype=np.float64) / float(int(count) + 1)
    eased = alpha * alpha * (3.0 - 2.0 * alpha)
    output = first[None] + (last - first)[None] * eased[:, None].astype(np.float32)
    quats = np.stack((first[3:7], last[3:7]))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    if float(np.dot(quats[0], quats[1])) < 0.0:
        quats[1] *= -1.0
    xyzw = Slerp(
        np.asarray((0.0, 1.0)), Rotation.from_quat(quats[:, (1, 2, 3, 0)])
    )(eased).as_quat()
    output[:, 3:7] = xyzw[:, (3, 0, 1, 2)]
    return output.astype(np.float32)


def align_qpos_action(action: np.ndarray, idle_qpos: np.ndarray) -> np.ndarray:
    """把完整动作首帧 XY/航向对齐站立姿态，同时保留后续相对位移。"""

    result = np.asarray(action, dtype=np.float32).copy()
    idle = np.asarray(idle_qpos, dtype=np.float32).reshape(BUMI_QPOS_DIM)
    if result.ndim != 2 or result.shape[1] != BUMI_QPOS_DIM or len(result) <= 0:
        raise ValueError(f"action must have shape [T,{BUMI_QPOS_DIM}] with T > 0")
    if not np.isfinite(result).all() or not np.isfinite(idle).all():
        raise ValueError("action/idle qpos contains NaN or Inf")
    source = Rotation.from_quat(result[0, (4, 5, 6, 3)])
    target = Rotation.from_quat(idle[[4, 5, 6, 3]])
    delta = Rotation.from_euler(
        "z", target.as_euler("zyx")[0] - source.as_euler("zyx")[0]
    )
    origin = result[0, :3].copy()
    relative = result[:, :3] - origin[None]
    result[:, :3] = delta.apply(relative).astype(np.float32) + origin[None]
    result[:, :2] += idle[None, :2] - origin[None, :2]
    xyzw = (delta * Rotation.from_quat(result[:, (4, 5, 6, 3)])).as_quat()
    result[:, 3:7] = xyzw[:, (3, 0, 1, 2)]
    return result


@dataclass(frozen=True, slots=True)
class BumiGmtPlanSnapshot:
    """供发布线程读取的不可变 GMT 计划修订。"""

    qpos: np.ndarray
    frames: np.ndarray
    audio_start_frame: int
    audio_end_frame: int
    action_complete: bool
    terminal_idle_qpos: np.ndarray | None = None
    terminal_idle_frames: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2 or self.qpos.shape[1] != BUMI_QPOS_DIM:
            raise ValueError("GMT plan qpos has an invalid shape")
        if self.frames.ndim != 2 or len(self.frames) != len(self.qpos):
            raise ValueError("GMT plan frames have an invalid shape")
        if self.qpos.flags.writeable or self.frames.flags.writeable:
            raise ValueError("GMT plan snapshots must be immutable")


class BumiIncrementalGmtPlanBuilder:
    """增量接收连续 30 Hz qpos，并构建与离线 30→50 结果一致的 GMT 计划。"""

    def __init__(
        self,
        idle_qpos: np.ndarray,
        native_to_gmt: np.ndarray,
        *,
        blend_seconds: float = 0.8,
        return_seconds: float = 1.0,
    ) -> None:
        self.idle_qpos = np.asarray(idle_qpos, dtype=np.float32).reshape(BUMI_QPOS_DIM).copy()
        self.native_to_gmt = np.asarray(native_to_gmt, dtype=np.int64).copy()
        if self.native_to_gmt.shape != (21,) or set(self.native_to_gmt.tolist()) != set(range(21)):
            raise ValueError("native_to_gmt must be a permutation of 0..20")
        if blend_seconds <= 0.0 or return_seconds <= 0.0:
            raise ValueError("blend_seconds and return_seconds must be > 0")
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
            source = Rotation.from_quat(result[0, (4, 5, 6, 3)])
            target = Rotation.from_quat(self.idle_qpos[[4, 5, 6, 3]])
            self.alignment_rotation = Rotation.from_euler(
                "z", target.as_euler("zyx")[0] - source.as_euler("zyx")[0]
            )
        assert self.alignment_rotation is not None and self.source_origin is not None
        relative = result[:, :3] - self.source_origin[None]
        result[:, :3] = (
            self.alignment_rotation.apply(relative).astype(np.float32)
            + self.source_origin[None]
        )
        result[:, :2] += self.idle_qpos[None, :2] - self.source_origin[None, :2]
        xyzw = (
            self.alignment_rotation * Rotation.from_quat(result[:, (4, 5, 6, 3)])
        ).as_quat()
        result[:, 3:7] = xyzw[:, (3, 0, 1, 2)]
        return result

    def append(self, source_qpos: np.ndarray, *, is_last: bool) -> BumiGmtPlanSnapshot:
        if self.closed:
            raise RuntimeError("cannot append to a completed BUMI GMT plan")
        new_action = self.resampler.append(source_qpos)
        if len(new_action) <= 0:
            raise RuntimeError("30 Hz qpos chunk did not produce a new 50 Hz sample")
        aligned = self._align_suffix(new_action)
        if self.action_frames == 0:
            prefix = np.repeat(self.idle_qpos[None], 10, axis=0)
            blend = interpolate_qpos(self.idle_qpos, aligned[0], self.blend_frames)
            self.audio_start_frame = len(prefix) + len(blend)
            self.timeline.append(np.concatenate((prefix, blend, aligned), axis=0))
        else:
            self.timeline.append(aligned)
        self.action_frames += len(aligned)
        audio_end_frame = self.audio_start_frame + self.action_frames

        terminal_qpos: np.ndarray | None = None
        terminal_frames: np.ndarray | None = None
        if is_last:
            terminal_qpos = aligned[-1].copy()
            terminal_qpos[2:] = self.idle_qpos[2:]
            returning = interpolate_qpos(aligned[-1], terminal_qpos, self.return_frames)
            self.timeline.append(
                np.concatenate(
                    (returning, np.repeat(terminal_qpos[None], 101, axis=0)), axis=0
                )
            )
            terminal_frames = qpos_timeline_to_gmt_frames(
                np.repeat(terminal_qpos[None], 110, axis=0),
                fps=50.0,
                native_to_gmt=self.native_to_gmt,
            )
            terminal_qpos.setflags(write=False)
            terminal_frames.setflags(write=False)
            self.closed = True
        return BumiGmtPlanSnapshot(
            qpos=self.timeline.qpos,
            frames=self.timeline.frames,
            audio_start_frame=self.audio_start_frame,
            audio_end_frame=audio_end_frame,
            action_complete=bool(is_last),
            terminal_idle_qpos=terminal_qpos,
            terminal_idle_frames=terminal_frames,
        )


__all__ = [
    "BumiGmtPlanSnapshot",
    "BumiIncrementalGmtPlanBuilder",
    "align_qpos_action",
    "interpolate_qpos",
]
