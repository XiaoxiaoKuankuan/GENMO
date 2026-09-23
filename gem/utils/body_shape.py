"""实时 SMPL-X 推理共用的身体形状稳定策略。

视频模型可以逐帧输出 10 维 SMPL-X ``betas``，但渲染、诊断和下游会话通常需要一个
时间上稳定的身体形状。本模块提供零形状、首帧冻结、预热均值冻结、指数移动平均和逐帧
透传五种策略，并同时支持 NumPy 与 Torch 张量；Torch 输入保持原设备和 dtype。

该模块不包含机器人资产、关节映射、传输协议或部署逻辑，可由摄像头和 SONIC 推理路径
安全复用。
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - NumPy-only环境不要求安装Torch。
    torch = None


class BetaStabilizer:
    """在 SMPL-X FK 或渲染前冻结、平滑或透传身体形状。"""

    MODES = {"zero", "first", "mean", "ema", "per_frame"}

    def __init__(self, mode: str = "mean", warmup: int = 30, ema_alpha: float = 0.05):
        if mode not in self.MODES:
            raise ValueError(f"shape mode must be one of {sorted(self.MODES)}")
        if warmup < 1:
            raise ValueError("shape warmup must be >= 1")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.mode = mode
        self.warmup = int(warmup)
        self.ema_alpha = float(ema_alpha)
        self.reset()

    @property
    def frozen(self) -> bool:
        if self.mode == "zero":
            return True
        return self.mode in {"first", "mean"} and self._frozen is not None

    def reset(self) -> None:
        self.count = 0
        self._sum: np.ndarray | None = None
        self._value: np.ndarray | None = None
        self._frozen: np.ndarray | None = None

    def update(self, betas: Any) -> Any:
        is_tensor = torch is not None and isinstance(betas, torch.Tensor)
        array = (
            betas.detach().to(device="cpu", dtype=torch.float64).numpy()
            if is_tensor
            else np.asarray(betas, dtype=np.float64)
        )
        if not np.isfinite(array).all():
            raise ValueError("betas contains NaN or Inf")

        if self.mode == "zero":
            result = np.zeros_like(array)
        elif self.mode == "per_frame":
            result = array.copy()
        elif self.mode == "first":
            if self._frozen is None:
                self._frozen = array.copy()
            result = self._frozen
        elif self.mode == "ema":
            if self._value is None:
                self._value = array.copy()
            else:
                self._value += self.ema_alpha * (array - self._value)
            result = self._value
        elif self._frozen is None:
            self._sum = array.copy() if self._sum is None else self._sum + array
            self.count += 1
            result = self._sum / self.count
            if self.count >= self.warmup:
                self._frozen = result.copy()
        else:
            result = self._frozen

        if self.mode not in {"zero", "mean"}:
            self.count += 1
        if is_tensor:
            return torch.as_tensor(result, dtype=betas.dtype, device=betas.device)
        return np.asarray(result, dtype=np.asarray(betas).dtype)


__all__ = ["BetaStabilizer"]
