"""验证自建 BUMI CSV/WAV 到正式训练数据根的关键发布契约。

测试只使用系统临时目录中的极小合成 CSV、PCM WAV 和测试 kinematics，不读取或改写
真实自建数据。它覆盖 APT啦啦操显式排除、普通/帧率别名配对、xyzw→wxyz、50→30 Hz、
动作与裁后音频/EDGE35 等长、FK 地面归一化、正式 reader/SHA 校验，以及特征提取失败
时不发布目标目录并清理 staging 的全有或全无语义。
"""

from __future__ import annotations

import csv
import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gem.datasets.music_dance.music_dance_bumi import BumiMusicDatasetReader, sha256_file
from gem.robots.bumi.kinematics import BumiKinematics
from tools.data.bumi.build_bumi_music_dataset_from_csv import (
    convert_dataset,
    load_csv_qpos,
    resample_qpos_to_30hz,
    target_frame_count,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KINEMATICS_PATH = REPO_ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
QUALITY_CONFIG_PATH = (
    REPO_ROOT / "configs/bumi/quality_filter_csv_mine_robot_retargeter_fe934_v2.yaml"
)


def _write_wav(path: Path, *, seconds: float, sample_rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(seconds * sample_rate))
    payload = np.zeros((frames, 2), dtype="<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def _write_csv(path: Path, header: list[str], *, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = np.zeros(28, dtype=np.float64)
    row[2] = 0.7
    row[6] = 1.0  # 源 CSV 根四元数为 xyzw identity。
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([row.tolist()] * frames)


def _source_tree(tmp_path: Path) -> dict[str, Path]:
    raw = yaml.safe_load(QUALITY_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["selection"]["expected_candidates"] = 3
    raw["selection"]["expected_accepted"] = 2
    quality = tmp_path / "quality.yaml"
    quality.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    source = tmp_path / "source"
    header = raw["source"]["csv_header"]
    _write_csv(source / "dance_2_csv/bumi_短歌_30fps.csv", header, frames=12)
    _write_wav(source / "dance_2_音频/短歌.wav", seconds=0.5)
    _write_csv(source / "dance_3_csv/bumi_五十_50fps.csv", header, frames=20)
    _write_wav(source / "dance_3_音频/五十.wav", seconds=0.5)
    _write_csv(source / "dance_3_csv/bumi_APT啦啦操_30fps.csv", header, frames=12)
    _write_wav(source / "dance_3_音频/APT啦啦操_30fps.wav", seconds=0.5)
    retarget = tmp_path / "retarget.json"
    retarget.write_text('{"contract": "synthetic"}\n', encoding="utf-8")
    return {"source": source, "quality": quality, "retarget": retarget}


def _fake_edge(audio_path: Path, *, target_fps: int):
    assert target_fps == 30
    with wave.open(str(audio_path), "rb") as handle:
        duration = handle.getnframes() / handle.getframerate()
    frames = int(round(duration * target_fps)) + 1
    value = torch.arange(frames * 35, dtype=torch.float32).reshape(frames, 35)
    return value, {"feature_frames": frames}


def test_frame_count_and_resampling_use_strict_common_tail() -> None:
    assert target_frame_count(12, 30, 24000, 48000) == 12
    assert target_frame_count(20, 50, 24000, 48000) == 12
    assert target_frame_count(3756, 50, 3591552, 48000) == 2244
    qpos = np.zeros((20, 28), dtype=np.float64)
    qpos[:, 0] = np.arange(20) / 50.0
    qpos[:, 3] = 1.0
    result = resample_qpos_to_30hz(qpos, 50, 12)
    torch.testing.assert_close(result[:, 0], torch.arange(12) / 30.0)
    torch.testing.assert_close(result[:, 3], torch.ones(12))


def test_csv_joint_columns_are_reordered_by_name(tmp_path: Path) -> None:
    """旧 CSV 列顺序必须按名字转成 fe934 顺序，不能原样拼到 qpos。"""

    inputs = _source_tree(tmp_path)
    raw = yaml.safe_load(inputs["quality"].read_text(encoding="utf-8"))
    from tools.data.bumi.build_bumi_music_dataset_from_csv import (
        discover_source_pairs,
        load_quality_config,
    )

    config = load_quality_config(inputs["quality"])
    pair = discover_source_pairs(inputs["source"], config)[0]
    values = np.loadtxt(pair.csv_path, delimiter=",", skiprows=1)
    source_index = raw["source"]["source_joint_names"].index("waist_yaw_joint")
    values[:, 7 + source_index] = 0.1
    with pair.csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(raw["source"]["csv_header"])
        writer.writerows(values.tolist())
    target_order = BumiKinematics(KINEMATICS_PATH).joint_order
    qpos = load_csv_qpos(pair, config, target_order)
    assert target_order[0] == "waist_yaw_joint"
    torch.testing.assert_close(
        torch.from_numpy(qpos[:, 7]), torch.full((12,), 0.1, dtype=torch.float64)
    )


def test_converter_excludes_apt_and_publishes_strict_dataset(
    tmp_path: Path,
) -> None:
    inputs = _source_tree(tmp_path)
    output = tmp_path / "formal"
    report = convert_dataset(
        source_root=inputs["source"],
        output_root=output,
        kinematics_path=KINEMATICS_PATH,
        quality_config_path=inputs["quality"],
        retarget_config_path=inputs["retarget"],
        feature_extractor=_fake_edge,
    )
    assert report["candidate_sequences"] == 3
    assert report["accepted_sequences"] == 2
    assert report["excluded_songs"] == ["APT啦啦操"]
    assert report["output_frames"] == 24
    rows = [
        json.loads(line)
        for line in (output / "manifests/train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in rows] == ["dance_2__短歌", "dance_3__五十"]
    assert {row["num_frames"] for row in rows} == {12}
    quality_rows = [
        json.loads(line)
        for line in (output / "reports/quality_report.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    apt = next(row for row in quality_rows if row["song"] == "APT啦啦操")
    assert apt["status"] == "EXCLUDED"
    assert apt["audio_alias_used"] is True
    kinematics = BumiKinematics(KINEMATICS_PATH)
    reader = BumiMusicDatasetReader(
        output,
        "mine_bumi",
        "train",
        kinematics,
        joint_limit_tolerance=0.0001,
        validate_payloads_on_init=True,
        validate_source_hashes_on_init=True,
    )
    for row in reader.rows:
        sequence = reader.load_aligned_sequence(row)
        assert sequence["qpos"].shape == (12, 28)
        assert sequence["music"].shape == (12, 35)
        torch.testing.assert_close(sequence["qpos"][:, 3], torch.ones(12))
        torch.testing.assert_close(sequence["qpos"][:, 4:7], torch.zeros(12, 3))
        assert sequence["foot_contact"].shape == (12, 2)
        assert sequence["foot_contact_available"].all()
        body = kinematics.forward_kinematics(sequence["qpos"])["body_pos_w"]
        assert float(body[..., 2].amin()) == pytest.approx(0.0, abs=2.0e-5)
        audio_path = output / row["audio_path"]
        with wave.open(str(audio_path), "rb") as handle:
            assert handle.getnframes() == 12 * 1600
            assert handle.getnframes() / handle.getframerate() == pytest.approx(12 / 30)
        assert sha256_file(audio_path) == row["source_audio_sha256"]
    assert math.isfinite(report["root_z_adjustment_m"]["mean"])


def test_feature_failure_removes_staging_and_does_not_publish(
    tmp_path: Path,
) -> None:
    inputs = _source_tree(tmp_path)
    output = tmp_path / "failed"

    def fail_feature(_audio_path: Path, *, target_fps: int):
        raise RuntimeError(f"synthetic EDGE failure at {target_fps} Hz")

    with pytest.raises(RuntimeError, match="synthetic EDGE failure"):
        convert_dataset(
            source_root=inputs["source"],
            output_root=output,
            kinematics_path=KINEMATICS_PATH,
            quality_config_path=inputs["quality"],
            retarget_config_path=inputs["retarget"],
            feature_extractor=fail_feature,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))


def test_verified_reference_reuses_only_audio_and_music_features(tmp_path: Path) -> None:
    """参考库只复用同源WAV/EDGE35，动作、当前顺序和接触仍重新发布。"""

    inputs = _source_tree(tmp_path)
    reference = tmp_path / "reference"
    convert_dataset(
        source_root=inputs["source"],
        output_root=reference,
        kinematics_path=KINEMATICS_PATH,
        quality_config_path=inputs["quality"],
        retarget_config_path=inputs["retarget"],
        feature_extractor=_fake_edge,
    )

    def must_not_extract(_audio_path: Path, *, target_fps: int):
        raise AssertionError(f"参考复用模式不应重新提取EDGE35: {target_fps}")

    output = tmp_path / "reused"
    report = convert_dataset(
        source_root=inputs["source"],
        output_root=output,
        kinematics_path=KINEMATICS_PATH,
        quality_config_path=inputs["quality"],
        retarget_config_path=inputs["retarget"],
        reference_root=reference,
        feature_extractor=must_not_extract,
    )
    assert report["music_audio_reused_from_reference"] is True
    assert report["reference_root"] == str(reference.resolve())
    for row in map(
        json.loads,
        (output / "manifests/train.jsonl").read_text(encoding="utf-8").splitlines(),
    ):
        payload = torch.load(output / row["motion_path"], map_location="cpu", weights_only=False)
        assert payload["joint_order_conversion"]["method"] == "exact_name_reorder"
        assert payload["foot_contact_contract_version"].startswith("genmo.bumi_foot_contact.")
