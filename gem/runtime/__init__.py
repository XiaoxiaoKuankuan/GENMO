# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""闭环 BUMI 运行时包：通过明确的子模块提供生成、轨迹和部署能力。

不再在包初始化时导入已退役的 SMPL、文本、视频和多模态常驻服务；当前 BUMI
消费者继续显式导入 bumi_*、gmt_* 等子模块，不改变 qpos30、采样、速度派生或协议。
"""
