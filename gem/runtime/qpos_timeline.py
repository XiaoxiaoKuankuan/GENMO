# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""与人体表示和重定向完全解耦的 BUMI qpos 增量重采样器。

本模块只处理 MuJoCo 原生 ``float32[T,28]`` qpos：输入固定为 30 Hz，输出固定落在
50 Hz 全局时间栅格。根平移和 21 个关节采用线性插值，根 ``wxyz`` 四元数先跨块做
符号连续化，再沿最短弧 SLERP。已经返回的 50 Hz 帧永不重算；上一块最后一个 30 Hz
样本仅作为下一块的插值边界，因此分块调用与一次性调用得到相同的完整时间线。

文件独立于旧 ``robot_stream.py``，使原生 BUMI 在线链路不会间接加载 SMPL、SMP1 或
GMR，同时旧类和旧入口继续保持原有实现及行为。
"""

from __future__ import annotations

import math

import numpy as np

BUMI_QPOS_DIM = 28
SOURCE_FPS = 30.0
TARGET_FPS = 50.0


class IncrementalQposTimeline:
    """把连续 30 Hz qpos 追加到固定的全局 50 Hz 采样栅格。"""

    def __init__(self) -> None:
        self._source_frames = 0
        self._last_source: np.ndarray | None = None
        self._target_frames = 0
        self._target_chunks: list[np.ndarray] = []

    @property
    def source_frames(self) -> int:
        return self._source_frames

    @property
    def target_frames(self) -> int:
        return self._target_frames

    def append(self, values: np.ndarray) -> np.ndarray:
        qpos = np.asarray(values, dtype=np.float32).copy()
        if qpos.ndim != 2 or qpos.shape[1] != BUMI_QPOS_DIM or len(qpos) <= 0:
            raise ValueError(f"qpos chunk must have shape [T,{BUMI_QPOS_DIM}]")
        if not np.isfinite(qpos).all():
            raise ValueError("qpos chunk contains NaN or Inf")

        for index in range(len(qpos)):
            quaternion = qpos[index, 3:7]
            norm = float(np.linalg.norm(quaternion))
            if not math.isfinite(norm) or norm < 1.0e-8:
                raise ValueError("quaternion norm is too small or non-finite")
            qpos[index, 3:7] = quaternion / np.float32(norm)
            previous = self._last_source if index == 0 else qpos[index - 1]
            if previous is not None and float(np.dot(previous[3:7], qpos[index, 3:7])) < 0.0:
                qpos[index, 3:7] *= -1.0

        old_source_frames = self._source_frames
        new_source_frames = old_source_frames + len(qpos)
        duration = (new_source_frames - 1) / SOURCE_FPS
        new_target_frames = int(math.floor(duration * TARGET_FPS)) + 1
        target_indices = np.arange(self._target_frames, new_target_frames, dtype=np.int64)

        if self._last_source is None:
            local_source = qpos
            local_start = 0
        else:
            local_source = np.concatenate((self._last_source[None], qpos), axis=0)
            local_start = old_source_frames - 1

        if len(target_indices) == 0:
            result = np.empty((0, BUMI_QPOS_DIM), dtype=np.float32)
        elif len(local_source) == 1:
            result = local_source.copy()
        else:
            from scipy.spatial.transform import Rotation, Slerp

            source_times = (
                local_start + np.arange(len(local_source), dtype=np.float64)
            ) / SOURCE_FPS
            target_times = target_indices.astype(np.float64) / TARGET_FPS
            target_times = np.clip(target_times, source_times[0], source_times[-1])
            result = np.empty((len(target_indices), BUMI_QPOS_DIM), dtype=np.float32)
            for dimension in (*range(3), *range(7, BUMI_QPOS_DIM)):
                result[:, dimension] = np.interp(
                    target_times, source_times, local_source[:, dimension]
                ).astype(np.float32)
            source_xyzw = local_source[:, (4, 5, 6, 3)]
            target_xyzw = (
                Slerp(source_times, Rotation.from_quat(source_xyzw))(target_times)
                .as_quat()
                .astype(np.float32)
            )
            result[:, 3:7] = target_xyzw[:, (3, 0, 1, 2)]

        result.setflags(write=False)
        if len(result):
            self._target_chunks.append(result)
        self._source_frames = new_source_frames
        self._last_source = qpos[-1].copy()
        self._target_frames = new_target_frames
        return result

    def target(self) -> np.ndarray:
        if not self._target_chunks:
            return np.empty((0, BUMI_QPOS_DIM), dtype=np.float32)
        return np.concatenate(self._target_chunks, axis=0)

    def reset(self) -> None:
        self._source_frames = 0
        self._last_source = None
        self._target_frames = 0
        self._target_chunks.clear()


__all__ = ["IncrementalQposTimeline"]
