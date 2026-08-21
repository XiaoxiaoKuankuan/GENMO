"""验证 BUMI 四库高质量完整音乐评测的选择与关节限位契约。

测试使用最小临时 CSV/空 WAV 和伪运动学对象，不加载真实 ONNX 或 MuJoCo。它覆盖四数据集
动作名到音频键的严格映射、score=1 去重选择，以及 0/1 基关节编号、XML 原始超限和部署
容差后超限之间的区别，防止批量页面给出错误的高质量来源或关节统计。
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.validate_bumi_hq_music_full import (
    DATASETS,
    analyze_joint_limits,
    audio_key_for_motion,
    select_hq_audio,
)


def test_fourset_audio_key_mapping() -> None:
    assert audio_key_for_motion("finedance", "001.npz") == "001"
    assert audio_key_for_motion("compas3d", "Pair1_song1_take1_follower.npz") == "Pair1_song1_take1"
    assert audio_key_for_motion("aioz_gdance", "abc_02_0_300_dancer_03.npz") == "abc_02_0_300"
    assert audio_key_for_motion("aistpp", "gBR_sBM_cAll_d04_mBR0_ch01.npz") == "mBR0"


def _write_ratings(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("motion_name", "score"))
        writer.writerows(rows)


def test_score_one_selection_deduplicates_shared_audio(tmp_path: Path) -> None:
    ratings = tmp_path / "ratings"
    audio = tmp_path / "audio"
    rows = {
        "finedance": [("001.npz", 1), ("002.npz", 2)],
        "compas3d": [
            ("Pair1_song1_take1_follower.npz", 1),
            ("Pair1_song1_take1_leader.npz", 1),
        ],
        "aioz_gdance": [
            ("clip_02_0_300_dancer_00.npz", 1),
            ("clip_02_0_300_dancer_01.npz", 1),
        ],
        "aistpp": [
            ("gBR_sBM_cAll_d04_mBR0_ch01.npz", 1),
            ("gBR_sBM_cAll_d05_mBR0_ch02.npz", 1),
        ],
    }
    for dataset, _, filename in DATASETS:
        _write_ratings(ratings / filename, rows[dataset])
        key = audio_key_for_motion(dataset, rows[dataset][0][0])
        target = (
            audio / "aistpp" / "wav" / f"{key}.wav"
            if dataset == "aistpp"
            else audio / dataset / f"{key}.wav"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    selected, summary = select_hq_audio(ratings, audio, per_dataset=10)
    assert len(selected) == 4
    assert summary["compas3d"]["score_1_action_count"] == 2
    assert summary["compas3d"]["selected_audio_count"] == 1
    compas = next(row for row in selected if row["dataset"] == "compas3d")
    assert compas["high_quality_motion_count"] == 2


def test_joint_limit_report_distinguishes_raw_and_tolerated_excess() -> None:
    kinematics = SimpleNamespace(
        joint_lower_limits=torch.full((21,), -1.0),
        joint_upper_limits=torch.full((21,), 1.0),
        joint_order=tuple(f"joint_{index}" for index in range(21)),
        source_mjcf_sha256="a" * 64,
    )
    qpos = torch.zeros(4, 28)
    qpos[:, 3] = 1.0
    qpos[1, 7 + 3] = 1.20
    qpos[2, 7 + 7] = -1.02
    report = analyze_joint_limits(qpos, kinematics, tolerance_rad=0.05)

    assert report["strict_xml_limit_exceeded"] is True
    assert report["strict_xml_violating_joint_count"] == 2
    assert report["tolerance_limit_exceeded"] is True
    assert report["tolerance_violating_joint_count"] == 1
    largest = report["joints"][0]
    assert largest["joint_index_0based"] == 3
    assert largest["joint_number_1based"] == 4
    assert largest["joint_name"] == "joint_3"
    assert abs(largest["max_excess_rad"] - 0.2) < 1.0e-6
    assert abs(largest["max_excess_after_tolerance_rad"] - 0.15) < 1.0e-6
