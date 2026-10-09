"""配对示范活动度的 CPU 契约与数值测试。

使用可手算的 30 Hz 关节运动、严格 train 清单和本任务临时 motion payload，验证
50 Hz 同窗 RMS、首步非伪造零速度、左右括点线性插值、尾部显式保持和关节顺序。
负例覆盖来源 SHA、音乐 SHA、清单行、单位/帧率元数据、路径逃逸、缺失动作和时间越界；
数据只作为独立奖励监督，既有 TrainMusicSampler 不因新增目标而读取动作或扩大条件。
所有文件写入 pytest tmp_path，由统一测试命令使用系统临时目录并在退出时清理；
测试不依赖生产数据、服务器、Isaac 或 GPU，不把构造数据结果当作动力学证明。
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER
from gem.closedloop.dppo.target_activity import PairedActivityTarget, load_paired_activity


def target(q, **kwargs):
    return PairedActivityTarget(q, joint_names=GMT_EXPECTED_JOINT_ORDER,
                               source={"test": "analytical_motion"}, **kwargs)


def write_pair(tmp_path, *, frames=60):
    root = tmp_path / "data"
    directory = root / "AIST++"
    (directory / "manifests").mkdir(parents=True)
    (directory / "meta").mkdir()
    (directory / "motions").mkdir()
    music = directory / "music.pt"
    torch.save(torch.zeros(frames, 35), music)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    names = list(reversed(GMT_EXPECTED_JOINT_ORDER))
    info = {"contract_version": "genmo.bumi_music.v1", "fps": 30, "robot_name": "bumi",
            "qpos_order": "mujoco_native", "quaternion_convention": "wxyz", "qpos_dim": 28,
            "joint_dim": 21, "quality_filter_applied": True, "dataset_name": "aistpp_bumi",
            "joint_names": names, "source_mjcf_sha256": "a" * 64,
            "quality_config_sha256": "b" * 64, "retarget_config_sha256": "c" * 64,
            "ground_semantics": "umr_foot_sole_ground_zero_v1", "root_z_adjusted": False}
    (directory / "meta/dataset_info.json").write_text(json.dumps(info))
    row = {"sample_id": "paired", "split": "train", "fps": 30, "num_frames": frames,
           "dataset": "aistpp_bumi", "motion_path": "motions/paired.pt", "quality_accepted": True,
           "music_feature_path": "music.pt", "source_music_feature_sha256": sha(music),
           "source_motion_sha256": "d" * 64, "source_audio_sha256": "e" * 64,
           "resplit_provenance": {"group_id": "group"}}
    qpos = torch.zeros(frames, 28, dtype=torch.float64)
    qpos[:, 3] = 1
    qpos[:, 7:] = torch.arange(frames, dtype=torch.float64)[:, None] / 30 * torch.arange(1, 22)
    payload = {**info, "qpos": qpos, "source_motion_sha256": row["source_motion_sha256"],
               "source_sample_id": row["sample_id"], "quality_accepted": True}
    path = directory / "motions/paired.pt"
    torch.save(payload, path)
    manifest = directory / "manifests/train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    sample = {"dataset": "AIST++", "group_id": "group", "row": row, "manifest_sha256": sha(manifest)}
    return root, sample, path, payload


def test_constant_velocity_first_interval_and_full_same_time_window():
    expected = np.arange(1, 22, dtype=np.float64)
    obj = target(np.arange(60)[:, None] / 30 * expected)
    first = obj(612)
    assert first["activity_rad_s"] == pytest.approx(np.sqrt(np.mean(expected ** 2)))
    assert first["window_count"] == 1 and not first["window_complete"]
    assert first["window_begin_tick"] == 600 and first["window_end_tick"] == 612
    full = obj(960)
    assert full["window_count"] == 25 and full["window_complete"]
    assert full["window_begin_tick"] == 660 and full["window_duration_s"] == .5
    np.testing.assert_allclose(full["per_joint_rms_rad_s"], expected)


def test_piecewise_source_uses_exact_interval_differences_not_frame_velocity_hold():
    # t=.02 -> q=.6；t=.04 -> q=1.6；两个区间速度为30与50。
    q = np.repeat(np.array([0., 1., 4., 9.])[:, None], 21, axis=1)
    obj = target(q)
    assert obj(612)["activity_rad_s"] == pytest.approx(30)
    assert obj(624)["activity_rad_s"] == pytest.approx(np.sqrt((30 ** 2 + 50 ** 2) / 2))


def test_window_configuration_and_nonzero_music_start():
    obj = target(np.arange(90)[:, None].repeat(21, axis=1) / 30,
                 music_start_tick=1800, window_s=.1)
    result = obj(1920)
    assert result["window_count"] == 5 and result["window_begin_tick"] == 1860
    assert result["window_complete"] and result["activity_rad_s"] == pytest.approx(1.)


def test_final_source_tail_is_explicit_hold_not_unknown_extrapolation():
    obj = target(np.repeat(np.array([0., 1., 2.])[:, None], 21, axis=1), window_s=.02)
    tail = obj(660)
    assert tail["activity_rad_s"] == 0 and tail["terminal_hold_interval_count"] == 1
    assert "hold_last_source_pose" in tail["source"]["tail_policy"]
    with pytest.raises(ValueError, match="exceeds paired music duration"):
        obj(672)


@pytest.mark.parametrize("tick", [600, 599, 601, 613, 612., True])
def test_invalid_control_time_never_rounded_or_padded(tick):
    with pytest.raises(ValueError):
        target(np.zeros((60, 21)))(tick)


@pytest.mark.parametrize("kwargs", [{"dt": .01}, {"dt": float("nan")}, {"window_s": .015},
                                     {"window_s": 0}, {"window_s": 1e-12}, {"music_start_tick": 1.5}])
def test_invalid_time_contract(kwargs):
    with pytest.raises(ValueError):
        target(np.zeros((60, 21)), **kwargs)


def test_target_owns_immutable_motion_and_copied_provenance():
    values = np.arange(60)[:, None].repeat(21, axis=1) / 30
    source = {"declared": "paired"}
    obj = PairedActivityTarget(values, joint_names=GMT_EXPECTED_JOINT_ORDER, source=source)
    values[:] = 0
    source["declared"] = "modified"
    one = obj(612)
    one["source"]["declared"] = "mutated_return"
    assert obj(612)["activity_rad_s"] == pytest.approx(1)
    assert obj(612)["source"]["declared"] == "paired"


def test_cached_windows_equal_scalar_reference_at_boundaries_and_random_order():
    rng = np.random.default_rng(738)
    obj = target(rng.normal(size=(1901, 21)))
    controls = len(obj._positions)*20//12
    for index in list(range(1, controls+1))+rng.integers(1, controls+1, size=128).tolist():
        count = min(index, obj.window_steps)
        endpoints = np.arange(index-count, index+1)*12
        reference = np.diff(obj._interpolate(endpoints), axis=0)/obj.dt
        actual = obj(600+index*12)
        assert actual['activity_rad_s'] == float(np.sqrt(np.mean(reference**2)))
        assert actual['per_joint_rms_rad_s'] == np.sqrt(np.mean(reference**2, axis=0)).tolist()
        assert len(obj._velocity) <= 512


def test_manifest_index_keeps_duplicate_rows_and_invalidates_changed_content(tmp_path):
    from gem.closedloop.dppo.target_activity import _MANIFEST_INDEX
    _MANIFEST_INDEX.clear()
    root, sample, _, _ = write_pair(tmp_path)
    load_paired_activity(root, sample)
    manifest = root/'AIST++/manifests/train.jsonl'
    manifest.write_text(manifest.read_text()*2)
    sample['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='unique train manifest row'):
        load_paired_activity(root, sample)
    assert len(_MANIFEST_INDEX) == 1


def test_strict_pair_load_hash_provenance_and_joint_order(tmp_path):
    root, sample, path, _ = write_pair(tmp_path)
    before = copy.deepcopy(sample)
    result = load_paired_activity(root, sample)(612)
    assert sample == before
    assert result["valid"] and result["source"]["supervision_scope"] == "reward_only"
    assert result["source"]["motion_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["source"]["motion_file_sha256"] != result["source"]["source_motion_sha256"]
    assert result["source"]["source_motion_sha256"] == sample["row"]["source_motion_sha256"]
    assert result["source"]["joint_order"] == list(GMT_EXPECTED_JOINT_ORDER)
    np.testing.assert_allclose(result["per_joint_rms_rad_s"], np.arange(21, 0, -1))


@pytest.mark.parametrize("key,value,pattern", [
    ("source_motion_sha256", "f" * 64, "source_motion_sha256 mismatch"),
    ("source_sample_id", "different", "sample identity"),
    ("fps", 60, "metadata mismatch: fps"),
    ("qpos_order", "unknown", "metadata mismatch: qpos_order"),
    ("joint_names", ["wrong"] * 21, "joint order"),
    ("quality_accepted", False, "quality mismatch"),
    ("retarget_config_sha256", "f" * 64, "source metadata SHA mismatch"),
])
def test_bad_motion_identity_fails_closed(tmp_path, key, value, pattern):
    root, sample, path, payload = write_pair(tmp_path)
    payload[key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=pattern):
        load_paired_activity(root, sample)


def test_missing_motion_never_uses_generated_reference(tmp_path):
    root, sample, path, _ = write_pair(tmp_path)
    path.unlink()
    sample["generated_reference"] = np.zeros((60, 28))
    with pytest.raises(FileNotFoundError, match="missing motion_path"):
        load_paired_activity(root, sample)


def test_music_and_manifest_hash_cannot_drift(tmp_path):
    root, sample, _, _ = write_pair(tmp_path)
    altered = copy.deepcopy(sample)
    altered["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="manifest SHA mismatch"):
        load_paired_activity(root, altered)
    (root / "AIST++/music.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="music feature SHA mismatch"):
        load_paired_activity(root, sample)


def test_manifest_row_and_group_must_match_selection(tmp_path):
    root, sample, _, _ = write_pair(tmp_path)
    changed = copy.deepcopy(sample)
    changed["row"]["split"] = "val"
    with pytest.raises(ValueError, match="unique train manifest row"):
        load_paired_activity(root, changed)
    changed = copy.deepcopy(sample)
    changed["group_id"] = "other_song"
    with pytest.raises(ValueError, match="group identity mismatch"):
        load_paired_activity(root, changed)


def test_motion_path_escape_is_not_loaded(tmp_path):
    root, sample, _, _ = write_pair(tmp_path)
    sample["row"]["motion_path"] = "../../unrelated.pt"
    manifest = root / "AIST++/manifests/train.jsonl"
    manifest.write_text(json.dumps(sample["row"]) + "\n")
    sample["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="escapes dataset root"):
        load_paired_activity(root, sample)


def test_explicit_source_directory_symlink_is_supported_without_file_escape(tmp_path):
    root, sample, _, _ = write_pair(tmp_path)
    linked_root = tmp_path / "selected"
    linked_root.mkdir()
    (linked_root / "AIST++").symlink_to(root / "AIST++", target_is_directory=True)
    result = load_paired_activity(linked_root, sample)(612)
    assert result["source"]["resolved_source_root"] == str(root / "AIST++")
    escaped = copy.deepcopy(sample)
    escaped["dataset"] = "../data/AIST++"
    with pytest.raises(ValueError, match="four training sources"):
        load_paired_activity(linked_root, escaped)


@pytest.mark.parametrize("kind", ["nonfinite", "length", "quaternion", "missing"])
def test_invalid_motion_arrays_are_not_reward_targets(tmp_path, kind):
    root, sample, path, payload = write_pair(tmp_path)
    if kind == "nonfinite":
        payload["qpos"][0, 7] = float("nan")
    elif kind == "length":
        payload["qpos"] = payload["qpos"][:-1]
    elif kind == "quaternion":
        payload["qpos"][:, 3] = 0
    else:
        del payload["qpos"]
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_paired_activity(root, sample)
