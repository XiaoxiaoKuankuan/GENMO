# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI 音乐分支的运行时包入口。

生成、播放、模型包检查和 GMT 通信由各 BUMI 子模块显式提供；包初始化仅导出
独立的单调时钟调度器，不自动装载已退役的人体、文本、视频、多模态或 GMR 引擎。
现有 qpos、DDIM、重采样、接触与通信算法继续使用原实现。
"""

from .playback_timing import MonotonicDeadline

__all__ = ["MonotonicDeadline"]
