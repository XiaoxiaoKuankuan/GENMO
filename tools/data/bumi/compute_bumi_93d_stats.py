#!/usr/bin/env python3
"""兼容旧文件名并转发到当前 BUMI qpos30 统计量工具。

BUMI 神经表示已从历史 93D 迁移为真正决定 qpos28 的 qpos30，因此这个旧文件名不能再
生成可供当前模型加载的 93D stats。为避免服务器脚本因路径变化直接失效，本入口完整转发
给 ``compute_bumi_30d_stats.py``：命令行参数、五库指纹、按数据集关节容差、运动学绑定和
输出契约都由唯一的 qpos30 实现负责。输出仍会明确标记
``genmo.bumi_qpos30_stats.v3``，绝不会把新结果伪装成旧表示。
"""

from __future__ import annotations

from compute_bumi_30d_stats import main

if __name__ == "__main__":
    main()
