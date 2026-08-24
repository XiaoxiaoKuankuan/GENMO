"""验证 BUMI 四库高质量完整音乐评测的选择与关节限位契约。

测试使用最小临时 CSV/空 WAV 和伪运动学对象，不加载真实 ONNX 或 MuJoCo。它覆盖四数据集
动作名到音频键的严格映射、score=1 去重选择，以及 0/1 基关节编号、XML 原始超限和部署
容差后超限之间的区别，并校验渲染失败后只复用身份完全一致的动作产物，防止批量页面给出
错误的高质量来源或关节统计，也避免恢复流程错误复用其他模型的推理结果。
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
    reusable_artifact,
    select_hq_audio,
    truncate_music_features,
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

    selected_by_dataset, summary_by_dataset = select_hq_audio(
        ratings,
        audio,
        per_dataset={
            "finedance": 3,
            "compas3d": 2,
            "aioz_gdance": 1,
            "aistpp": 1,
        },
    )
    assert len(selected_by_dataset) == 4
    assert summary_by_dataset["finedance"]["requested_audio_count"] == 3
    assert summary_by_dataset["finedance"]["selected_audio_count"] == 1


def test_music_feature_truncation_preserves_original_duration() -> None:
    features = torch.arange(300 * 35, dtype=torch.float32).reshape(300, 35)
    clipped, metadata, original_duration = truncate_music_features(
        features,
        {"selected_duration_sec": 10.0, "feature_frames": 300},
        max_duration_sec=8.0,
    )
    assert clipped.shape == (240, 35)
    assert torch.equal(clipped, features[:240])
    assert original_duration == 10.0
    assert metadata["original_selected_duration_sec"] == 10.0
    assert metadata["selected_duration_sec"] == 8.0
    assert metadata["feature_frames"] == 240


def test_reusable_artifact_requires_matching_identity(tmp_path: Path) -> None:
    artifact_path = tmp_path / "motion.pt"
    identity = {
        "checkpoint_sha256": "a" * 64,
        "onnx_sha256": "b" * 64,
        "audio": "/music/example.wav",
        "seed": 42,
        "cfg_scale": 2.5,
        "ddim_steps": 20,
        "max_duration_sec": 8.0,
        "sliding_qpos_contract_version": "contract-v1",
    }
    torch.save(
        {
            **identity,
            "qpos": torch.zeros(240, 28),
            "qpos_canonical": torch.zeros(240, 28),
        },
        artifact_path,
    )
    arguments = {
        "item": {"audio": identity["audio"]},
        **{
            ("sliding_contract_version" if key == "sliding_qpos_contract_version" else key): value
            for key, value in identity.items()
            if key != "audio"
        },
    }
    assert reusable_artifact(artifact_path, **arguments) is not None
    arguments["onnx_sha256"] = "c" * 64
    assert reusable_artifact(artifact_path, **arguments) is None


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
