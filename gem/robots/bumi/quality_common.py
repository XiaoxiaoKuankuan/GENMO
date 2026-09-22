"""BUMI 质量判定共享的中性值对象和帧区间算法。

这里仅定义与输入格式、机器人资产和报告版本无关的三态质量枚举，以及把坏帧
布尔序列转换为区间、扩张后求安全区间的纯 NumPy 算法。当前 UMR 和
robot_retargeter 可以依赖本模块，而本模块不会反向导入任何 producer，避免仅为了
``PASS / REVIEW / REJECT`` 枚举就加载具体资产契约。

区间统一采用左闭右开 ``[start, end)`` 语义；halo 和最短长度的比较规则完全保持
既有实现，因此旧报告与当前报告的安全片段边界不会因模块迁移发生变化。
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class QualityStatus(str, Enum):
    """质量决策优先级：REJECT > REVIEW > PASS。"""

    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


def mask_to_intervals(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """把布尔帧 mask 转为左闭右开 ``[start,end)`` 区间。"""

    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if not values.size:
        return ()
    changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def safe_intervals_from_bad_mask(
    mask: np.ndarray,
    *,
    halo_frames: int,
    minimum_frames: int,
) -> tuple[tuple[int, int], ...]:
    """扩张坏帧区间后，返回长度达到门槛的安全片段。"""

    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    expanded = np.zeros_like(values)
    for start, end in mask_to_intervals(values):
        expanded[max(0, start - halo_frames) : min(len(values), end + halo_frames)] = True
    return tuple(
        interval
        for interval in mask_to_intervals(~expanded)
        if interval[1] - interval[0] >= minimum_frames
    )
