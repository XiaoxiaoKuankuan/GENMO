"""BUMI 动作生产链共享的、与具体生产者无关的基础工具。

本模块刻意不导入任何具体数据 producer 的契约，只承载多条生产链都需要的确定性
基础能力：流式计算文件 SHA256、兼容读取由
NumPy 2 写出而在 NumPy 1 环境中加载的可信本地 pickle，以及汇总并门禁 Root
局部 Z 轴相对世界 Z 轴的倾角分布。把这些能力放在中性边界后，当前 30 Hz
producer 无需为了一个 hash、reader 或统计函数反向依赖已退役的 legacy 格式。

``NumpyCompatibleUnpickler`` 仍然只适用于调用方已经信任的本地文件；它不把
pickle 变成安全格式，也不会放宽除 ``numpy._core`` 到 ``numpy.core`` 之外的类
解析。Root 倾角门禁保持原阈值和严格的大于比较，确保迁移前后发布结论一致。
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np

ROOT_TILT_MAX_MEDIAN_DEG = 45.0
ROOT_TILT_MAX_P95_DEG = 75.0
ROOT_TILT_MAX_OVER_45DEG_FRACTION = 0.50


class NumpyCompatibleUnpickler(pickle.Unpickler):
    """兼容 NumPy 2 ``numpy._core`` 路径的受信任本地 pickle reader。"""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == "numpy._core" or module.startswith("numpy._core."):
                legacy_module = "numpy.core" + module[len("numpy._core") :]
                return super().find_class(legacy_module, name)
            raise


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """流式返回文件内容 SHA256，避免把大型动作或资产一次性读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_tilt_statistics(tilt_degrees: np.ndarray) -> dict[str, float | int]:
    """汇总 Root 局部 Z 轴相对世界 Z 轴倾角，供所有 producer 统一发布。"""

    values = np.asarray(tilt_degrees, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("root tilt values must be a non-empty finite sequence")
    return {
        "num_frames": int(values.size),
        "median_deg": float(np.median(values)),
        "p95_deg": float(np.percentile(values, 95)),
        "max_deg": float(np.max(values)),
        "over_45deg_fraction": float(np.mean(values > 45.0)),
    }


def enforce_root_tilt_gate(
    tilt_degrees: np.ndarray,
    *,
    context: str,
    max_median_deg: float = ROOT_TILT_MAX_MEDIAN_DEG,
    max_p95_deg: float = ROOT_TILT_MAX_P95_DEG,
    max_over_45deg_fraction: float = ROOT_TILT_MAX_OVER_45DEG_FRACTION,
) -> dict[str, float | int]:
    """拒绝 Root 倾角统计异常的动作，防止错误坐标数据进入发布或训练。"""

    stats = root_tilt_statistics(tilt_degrees)
    if (
        stats["median_deg"] > max_median_deg
        or stats["p95_deg"] > max_p95_deg
        or stats["over_45deg_fraction"] > max_over_45deg_fraction
    ):
        raise ValueError(
            f"{context}: root orientation gate failed: {stats}; thresholds="
            f"median<={max_median_deg}, p95<={max_p95_deg}, "
            f"over45_fraction<={max_over_45deg_fraction}"
        )
    return stats
