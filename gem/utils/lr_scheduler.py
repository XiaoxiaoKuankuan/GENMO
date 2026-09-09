# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""GENMO 按 optimizer step 使用的线性预热加余弦学习率调度器。

MotionMillion 从零训练需要在前 5,000 个 optimizer step 将学习率稳定地线性提升
到 AdamW 基础学习率，随后在 300,000 step 内余弦下降到绝对下限 2e-6。本模块
把该规则封装成标准 ``LambdaLR``，可由 Hydra 直接实例化，并支持 Lightning 完整
checkpoint 恢复 ``last_epoch`` 状态；它不改变已有实验所使用的 MultiStepLR。
"""

from __future__ import annotations

import math

import torch


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    """先线性预热、再余弦退火到绝对 ``min_lr``。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps 必须为正数")
        if total_steps <= warmup_steps:
            raise ValueError("total_steps 必须大于 warmup_steps")
        if min_lr < 0:
            raise ValueError("min_lr 不能为负数")
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

        lambdas = []
        for group in optimizer.param_groups:
            base_lr = float(group["lr"])
            if base_lr <= 0 or self.min_lr > base_lr:
                raise ValueError("每个参数组都必须满足 0 <= min_lr <= base_lr")
            min_factor = self.min_lr / base_lr

            def schedule(step: int, floor: float = min_factor) -> float:
                if step < self.warmup_steps:
                    return max(step + 1, 1) / self.warmup_steps
                progress = min(
                    max((step - self.warmup_steps) / (self.total_steps - self.warmup_steps), 0.0),
                    1.0,
                )
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return floor + (1.0 - floor) * cosine

            lambdas.append(schedule)
        super().__init__(optimizer, lr_lambda=lambdas, last_epoch=last_epoch)


__all__ = ["LinearWarmupCosineAnnealingLR"]
