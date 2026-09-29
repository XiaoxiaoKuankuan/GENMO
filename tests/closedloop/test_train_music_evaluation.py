"""训练音乐专项的选样、输入缺失策略和阈值配置回归测试。

使用临时清单验证固定种子选样不受清单行顺序影响、同组最长条目和跨组音频去重、
train/val 隔离及无视频时可省略音频。配置测试确认新 yaw 分离模式不误用整体姿态
阈值，且训练集专项不会被报告为原四库 val 的48集验收；不加载模型或启动仿真。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from gem.closedloop.evaluation_music import check_music_files, select_train_music
from tools.eval.run_closedloop_baseline import validate_config, validation_scope

ROOT = Path(__file__).resolve().parents[2]


def _row(name, group, frames=900, audio=None):
    return {"sample_id": name, "num_frames": frames, "split": "train", "fps": 30,
            "resplit_provenance": {"group_id": group},
            "music_feature_path": f"music/{name}.pt", "audio_path": f"audio/{name}.wav",
            "source_audio_sha256": hashlib.sha256((audio or name).encode()).hexdigest()}


def _manifest(tmp_path, rows):
    path = tmp_path / "Mine/manifests/train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_selection_is_seeded_order_independent_and_uses_longest_group_member(tmp_path):
    rows = [_row("short", "g1", 300), _row("long", "g1", 900), _row("second", "g2")]
    path = _manifest(tmp_path, rows)
    chosen = select_train_music(tmp_path, ["Mine"], 2, seed=42)
    path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    repeated = select_train_music(tmp_path, ["Mine"], 2, seed=42)
    ids = [sample["row"]["sample_id"] for sample in chosen]
    assert ids == [sample["row"]["sample_id"] for sample in repeated]
    assert set(ids) == {"long", "second"}
    assert all(sample["row"]["split"] == "train" for sample in chosen)


def test_duplicate_audio_cannot_fill_requested_song_count(tmp_path):
    _manifest(tmp_path, [_row("one", "g1", audio="same"), _row("two", "g2", audio="same")])
    with pytest.raises(ValueError, match="requested 2 independent train songs, found 1"):
        select_train_music(tmp_path, ["Mine"], 2)


def test_train_selector_rejects_val_rows_even_if_filename_says_train(tmp_path):
    row = _row("wrong_split", "g1")
    row["split"] = "val"
    _manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="Invalid train/fps"):
        select_train_music(tmp_path, ["Mine"], 1)


def test_optional_audio_is_not_claimed_verified_and_features_remain_required(tmp_path):
    row = _row("song", "g1")
    feature = tmp_path / "Mine" / row["music_feature_path"]
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"feature-placeholder")
    row["source_music_feature_sha256"] = hashlib.sha256(feature.read_bytes()).hexdigest()
    sample = {"dataset": "Mine", "row": row}
    report = check_music_files(tmp_path, [sample], require_audio=False)
    assert not report["missing"] and not report["invalid_sha256"]
    assert report["verified"] == [str(feature)]
    assert report["optional_audio_missing"] == [str(tmp_path / "Mine" / row["audio_path"])]
    assert check_music_files(tmp_path, [sample])["missing"]
    feature.unlink()
    assert str(feature) in check_music_files(tmp_path, [sample], require_audio=False)["missing"]


def _config():
    config = yaml.safe_load((ROOT / "configs/closedloop/stage8_frozen_isaac.yaml").read_text())
    config["termination"] = {"orientation_mode": "separated_yaw", "root_height_error_m": .2,
        "non_yaw_orientation_error_rad": .6, "yaw_error_rad": 1.5,
        "end_effector_relative_height_error_m": .15}
    config["evaluation"].update(split="train", require_audio=False, selection_seed=42)
    return config


def test_new_threshold_profile_is_explicit_and_does_not_mutate_old_defaults():
    config = _config()
    validate_config(config)
    mixed = copy.deepcopy(config)
    mixed["termination"]["global_orientation_error_rad"] = 1.2
    with pytest.raises(ValueError, match="cannot also specify"):
        validate_config(mixed)
    original = yaml.safe_load((ROOT / "configs/closedloop/stage8_frozen_isaac.yaml").read_text())
    validate_config(original)
    assert original["termination"]["global_orientation_error_rad"] == 1.2


def test_training_evaluation_is_not_the_original_val_matrix():
    config = _config()
    selected = [{"dataset": name, "group_id": str(group)}
                for name in config["evaluation"]["datasets"] for group in range(2)]
    scope = validation_scope(config, selected, 48, {"full_calibration": True})
    assert scope["requested_matrix_completed"]
    assert not scope["full_matrix"]
    assert scope["dataset_split"] == "train"
    assert scope["termination"]["yaw_error_rad"] == 1.5
