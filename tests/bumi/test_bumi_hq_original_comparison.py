"""验证高质量原动作与模型生成对比工具的时间网格和文件物化语义。

这里不加载真实 MuJoCo 或长视频，只覆盖最容易造成对比错位的 50→30 Hz 帧数反解，以及
自包含网页目录优先硬链接、源身份变化时原子刷新的规则。完整 40 项媒体和轨迹由正式运行
后的逐文件验证负责。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.build_bumi_hq_original_comparison import (
    materialize_file,
    target_30hz_frames,
)


@pytest.mark.parametrize(
    ("source_frames", "target_frames"),
    ((4879, 2928), (3685, 2212), (7540, 4525), (599, 360), (415, 250)),
)
def test_target_30hz_frames_exactly_inverts_offline_grid(
    source_frames: int, target_frames: int
) -> None:
    assert target_30hz_frames(source_frames) == target_frames


def test_materialize_file_prefers_hardlink_reuses_and_refreshes_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    target = tmp_path / "nested" / "target.mp4"
    source.write_bytes(b"formal-video")
    assert materialize_file(source, target) == "hardlink"
    assert os.stat(source).st_ino == os.stat(target).st_ino
    assert materialize_file(source, target) == "reused"

    target.unlink()
    target.write_bytes(b"different")
    assert materialize_file(source, target) == "refreshed_hardlink"
    assert target.read_bytes() == b"formal-video"
    assert os.stat(source).st_ino == os.stat(target).st_ino
