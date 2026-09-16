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


class ContinuationWarmupCosineLR(torch.optim.lr_scheduler.LRScheduler):
    """完整恢复 AdamW 后，以绝对步数开启独立的续训学习率阶段。

    首次读取旧调度器状态时，只接收已完成步数和恢复后的实际学习率，不让旧
    total_steps 覆盖新阶段。阶段内再次恢复则严格核对阶段身份并恢复进度。
    本类不加载、清空或修改 optimizer 的一、二阶动量。
    """

    def __init__(
        self, optimizer, *, start_step, stage_steps, warmup_steps, peak_lr, min_lr, total_steps=None
    ):
        if total_steps is not None:
            raise ValueError("续训使用 stage_steps，旧 total_steps 必须显式清空")
        if not 0 < warmup_steps < stage_steps or start_step < 0:
            raise ValueError("续训需要 start_step >= 0 且 0 < warmup_steps < stage_steps")
        if not 0 < min_lr <= peak_lr:
            raise ValueError("续训需要 0 < min_lr <= peak_lr")
        self.stage_signature = {
            "start_step": int(start_step),
            "stage_steps": int(stage_steps),
            "warmup_steps": int(warmup_steps),
            "peak_lr": float(peak_lr),
            "min_lr": float(min_lr),
        }
        self.start_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.restored = False
        super().__init__(optimizer)

    def get_lr(self):
        if not self.restored:
            return self.start_lrs
        cfg = self.stage_signature
        offset = max(0, self.last_epoch - cfg["start_step"])
        if offset <= cfg["warmup_steps"]:
            fraction = offset / cfg["warmup_steps"]
            return [lr + (cfg["peak_lr"] - lr) * fraction for lr in self.start_lrs]
        progress = min(
            1.0, (offset - cfg["warmup_steps"]) / (cfg["stage_steps"] - cfg["warmup_steps"])
        )
        lr = cfg["min_lr"] + (cfg["peak_lr"] - cfg["min_lr"]) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
        return [lr] * len(self.start_lrs)

    def load_state_dict(self, state_dict):
        if "stage_signature" in state_dict:
            if state_dict["stage_signature"] != self.stage_signature:
                raise ValueError("checkpoint 续训阶段与当前配置不一致，拒绝隐式重置")
            super().load_state_dict(state_dict)
        else:
            if int(state_dict["last_epoch"]) != self.stage_signature["start_step"]:
                raise ValueError("旧 checkpoint 步数必须等于续训 start_step")
            # Lightning 先恢复 optimizer，再恢复 scheduler；读取的是真实保存 LR。
            self.start_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            if any(not 0 < lr <= self.stage_signature["peak_lr"] for lr in self.start_lrs):
                raise ValueError("恢复学习率必须为正且不高于续训峰值")
            self.last_epoch = int(state_dict["last_epoch"])
            self._step_count = int(state_dict.get("_step_count", self.last_epoch + 1))
            self.restored = True
        self._last_lr = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr


__all__ = ["LinearWarmupCosineAnnealingLR", "ContinuationWarmupCosineLR"]
