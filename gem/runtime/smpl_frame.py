# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""视频推理会话使用的单帧 SMPL-X 参数值对象。

该模块只定义四个有限张量字段的不可变容器，供摄像头后端与常驻视频会话传递同一帧的
身体姿态、根姿态、平移和体型。它不读取动作目录，不维护播放队列、时钟或状态机，也不
包含机器人重定向、网络传输或控制逻辑，因此可独立服务于现行视频推理路径。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SMPLFrame:
    """一帧 SMPL-X 身体参数。"""

    body_pose: torch.Tensor
    global_orient: torch.Tensor
    transl: torch.Tensor
    betas: torch.Tensor

    def clone(self) -> SMPLFrame:
        """返回四个张量均独立复制的新帧。"""

        return SMPLFrame(
            self.body_pose.clone(),
            self.global_orient.clone(),
            self.transl.clone(),
            self.betas.clone(),
        )
