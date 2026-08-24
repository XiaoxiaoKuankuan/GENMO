"""验证高质量原动作与模型生成对比工具的时间网格和文件物化语义。

这里不加载真实 MuJoCo 或长视频，只覆盖最容易造成对比错位的 50→30 Hz 帧数反解，以及
自包含网页目录优先硬链接、源身份变化时原子刷新的规则，以及公开站点数据不会泄露本地
绝对路径的约束。完整可变数量媒体和轨迹由正式运行后的逐文件验证负责。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.build_bumi_hq_original_comparison import (
    build_index,
    build_site_data,
    materialize_file,
    summarize,
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


def test_site_data_only_exposes_relative_public_media() -> None:
    limits = {
        "strict_xml_limit_exceeded": False,
        "tolerance_limit_exceeded": False,
        "strict_xml_max_excess_rad": 0.0,
    }
    metrics = {
        "foot_penetration_max_m": 0.01,
        "root_tilt_max_rad": 0.02,
        "joint_velocity_p95_radps": 0.03,
    }
    result = {
        "status": "passed",
        "dataset": "aistpp",
        "dataset_label": "AIST++",
        "audio_key": "mBR0",
        "representative_motion": "example.npz",
        "comparison_duration_sec": 8.0,
        "generated_duration_sec": 8.0,
        "source_audio_duration_sec": 120.0,
        "source_clip_shorter_than_audio": False,
        "comparison_video_relative": "videos/comparison/aistpp/mBR0.mp4",
        "original_metrics": metrics,
        "generated_overlap_metrics": metrics,
        "original_joint_limits": limits,
        "generated_overlap_joint_limits": limits,
        "source_motion": "/private/source/example.npz",
    }
    site_data = build_site_data(
        [result],
        {"completed": 1},
        items=[{"dataset": "aistpp", "audio_key": "mBR0", "high_quality_motion_count": 2}],
        model_label="s350000",
    )
    assert site_data["items"][0]["comparison_video"] == ("/videos/comparison/aistpp/mBR0.mp4")
    assert "/private/" not in repr(site_data)


def test_empty_partial_summary_still_builds_recovery_index(tmp_path: Path) -> None:
    summary = summarize([])
    build_index(tmp_path, [], summary, model_label="s350000")
    document = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "完成 0/0 项" in document
    assert "n/a" in document
