# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""训练速度统计回调。

该 callback 记录 DataLoader 等待时间和单个 batch 的训练耗时，并写入
Lightning log/progress bar，方便判断瓶颈在数据加载还是模型计算。
"""

from collections import deque
from time import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only


class TrainSpeedTimer(pl.Callback):
    def __init__(self, N_avg=5):
        """
        This callback times the training speed (averge over recent 5 iterations)
            1. Data waiting time: this should be small, otherwise the data loading should be improved
            2. Single batch time: this is the time for one batch of training (excluding data waiting)
        """
        super().__init__()
        self.last_batch_end = None
        self.this_batch_start = None

        # time queues for averaging
        self.data_waiting_time_queue = deque(maxlen=N_avg)
        self.single_batch_time_queue = deque(maxlen=N_avg)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Count the time of data waiting"""
        if self.last_batch_end is not None:
            # This should be small, otherwise the data loading should be improved
            data_waiting = time() - self.last_batch_end

            # Average the time
            self.data_waiting_time_queue.append(data_waiting)
            average_time = sum(self.data_waiting_time_queue) / len(self.data_waiting_time_queue)

            # Log to prog-bar
            pl_module.log(
                "train_timer/data_waiting",
                average_time,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )

        self.this_batch_start = time()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Effective training time elapsed (excluding data waiting)
        single_batch = time() - self.this_batch_start

        # Average the time
        self.single_batch_time_queue.append(single_batch)
        average_time = sum(self.single_batch_time_queue) / len(self.single_batch_time_queue)

        # Log iter time
        pl_module.log(
            "train_timer/single_batch",
            average_time,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        # 按当前 rank 的实际 micro-batch 和 world size 记录全局数据吞吐。
        # 梯度累积不会改变每秒处理的样本数，因此这里不乘 accumulate 倍数。
        local_batch_size = None
        if isinstance(batch, dict):
            local_batch_size = batch.get("B")
            if isinstance(local_batch_size, torch.Tensor):
                local_batch_size = int(local_batch_size.item())
        if local_batch_size is None:
            try:
                local_batch_size = len(batch)
            except TypeError:
                local_batch_size = None
        if local_batch_size is not None:
            samples_per_second = (
                int(local_batch_size) * int(trainer.world_size) / max(average_time, 1.0e-9)
            )
            pl_module.log(
                "train_timer/global_samples_per_second",
                samples_per_second,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        max_steps = int(getattr(trainer, "max_steps", -1))
        if max_steps > 0:
            accumulate = int(getattr(trainer, "accumulate_grad_batches", 1))
            remaining_steps = max(max_steps - int(trainer.global_step), 0)
            pl_module.log(
                "train_timer/eta_hours",
                remaining_steps * average_time * accumulate / 3600.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        device = pl_module.device
        if device.type == "cuda":
            gib = 1024**3
            pl_module.log(
                "train_timer/gpu_memory_allocated_gib",
                torch.cuda.memory_allocated(device) / gib,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            pl_module.log(
                "train_timer/gpu_peak_memory_allocated_gib",
                torch.cuda.max_memory_allocated(device) / gib,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        # Set timer for counting data waiting
        self.last_batch_end = time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Reset the timer
        self.last_batch_end = None
        self.this_batch_start = None
        # Clear the queue
        self.data_waiting_time_queue.clear()
        self.single_batch_time_queue.clear()
