"""验证当前 NPZ 质量筛选继续复用的布尔区间与安全片段工具。

从旧 legacy 质量测试中迁出半开区间和坏帧两侧余量的回归，保留现行筛选报告
依赖的区间语义，不加载已退役的质量 YAML 或生产数据。
"""

import numpy as np

from gem.robots.bumi.quality_filter import mask_to_intervals, safe_intervals_from_bad_mask


def test_interval_helpers_use_half_open_ranges_and_halo() -> None:
    mask = np.zeros(400, dtype=np.bool_)
    mask[150:170] = True
    assert mask_to_intervals(mask) == ((150, 170),)
    assert safe_intervals_from_bad_mask(mask, halo_frames=15, minimum_frames=120) == (
        (0, 135),
        (185, 400),
    )
