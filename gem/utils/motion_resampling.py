"""动作时间轴重采样公共函数。

轴角旋转使用逐关节 SLERP，平移和标量使用线性插值，保留原有帧数舍入及末帧边界。
本模块不读取音乐数据集、不依赖其目录结构，供 KIT-ML 等源动作准备流程使用。
"""

from __future__ import annotations

import numpy as np


def _resample_axis_angle(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Slerp ``[T,J,3]`` axis-angle rotations on the target frame clock."""
    from scipy.spatial.transform import Rotation, Slerp

    frames, joints, _ = values.shape
    target_frames = max(1, int(round(frames * target_fps / source_fps)))
    if frames == 1:
        return np.repeat(values, target_frames, axis=0).astype(np.float32)
    source_times = np.arange(frames, dtype=np.float64) / source_fps
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times = np.minimum(target_times, source_times[-1])
    result = np.empty((target_frames, joints, 3), dtype=np.float32)
    for joint in range(joints):
        rotations = Rotation.from_rotvec(values[:, joint].astype(np.float64))
        result[:, joint] = (
            Slerp(source_times, rotations)(target_times).as_rotvec().astype(np.float32)
        )
    return result


def _resample_linear(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    frames = values.shape[0]
    target_frames = max(1, int(round(frames * target_fps / source_fps)))
    if frames == 1:
        return np.repeat(values, target_frames, axis=0).astype(np.float32)
    source_times = np.arange(frames, dtype=np.float64) / source_fps
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times = np.minimum(target_times, source_times[-1])
    flattened = values.reshape(frames, -1)
    result = np.stack(
        [
            np.interp(target_times, source_times, flattened[:, index])
            for index in range(flattened.shape[1])
        ],
        axis=-1,
    )
    return result.reshape((target_frames, *values.shape[1:])).astype(np.float32)
