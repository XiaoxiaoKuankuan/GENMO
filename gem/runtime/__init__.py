# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI music-only 独立部署的运行时包入口。

部署入口显式导入所需的 BUMI 子模块，本文件不自动装载文本、人体、视频或多模态引擎，
避免这些功能把训练框架和模型构造器重新带入部署目录。完整仓库的包级公共接口保持原状；
此精简仅属于 deploy/bumi-music-only-gmt 分支。轨迹生成、后处理与通信仍复用原实现。
"""
