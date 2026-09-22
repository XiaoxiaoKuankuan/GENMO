"""BUMI closed-loop Stage 1 数据构造的因果性、mask 和 split 回归测试。

本文件只在 pytest 的临时目录中创建极小的版本化 qpos28/EDGE35 配对数据，并使用仓库内
fe934 运动学 JSON。测试覆盖 30→50 Hz 最新可见样本保持、30 Hz 后向速度、GMT 关节具名
置换、历史起点 padding、qpos30 右侧 halo、逐坐标 known/target mask、P=0/可变P、contact
监督、动态 H、collate 和 train-only proprio stats 入口。所有产物均位于 ``tmp_path``，不会
读取或覆盖服务器正式数据、qpos30 stats、checkpoint、训练日志，也不会启动训练或 Isaac/GMT。

这些测试证明的是示范数据 producer 的静态/运动学因果契约；它们不能把 demo proxy 解释为
真实 frozen-GMT rollout，更不构成动力学、部署或实机安全验证。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

from gem.closedloop.contracts import (
    GMT_EXPECTED_JOINT_ORDER,
    GMT_NOMINAL_DEFAULT_JOINT_POS_RAD,
    PROPRIO_SLICES,
    validate_stage1_training_batch,
)
from gem.closedloop.stage1_dataset import (
    CAUSAL_PROPRIO_CONSTRUCTION_VERSION,
    BumiClosedLoopStage1Dataset,
    CausalDemoProprio48Builder,
    collate_stage1_training_samples,
)
from gem.datasets.music_dance.music_dance_bumi import sha256_file
from gem.robots.bumi.contacts import BUMI_CONTACT_CONTRACT_VERSION
from gem.robots.bumi.kinematics import BumiKinematics
from gem.utils.rotation_conversions import axis_angle_to_quaternion
from tools.data.bumi.compute_closedloop_proprio48_stats import (
    PROPRIO_STATS_CONTRACT_VERSION,
    parse_dataset,
)
from tools.data.bumi.compute_closedloop_proprio48_stats import (
    main as compute_proprio_stats_main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KINEMATICS_PATH = REPO_ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
DATASET_CONFIG = REPO_ROOT / "configs/closedloop/stage1_dataset_server1_fourset_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_qpos(length: int, kinematics: BumiKinematics) -> torch.Tensor:
    qpos = kinematics.default_qpos.view(1, 28).repeat(length, 1)
    frame = torch.arange(length, dtype=torch.float32)
    qpos[:, 0] = frame * 0.01
    qpos[:, 1] = frame * -0.002
    yaw = frame * 0.01
    qpos[:, 3:7] = axis_angle_to_quaternion(
        torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=-1)
    )
    return qpos


def _write_dataset(
    root: Path,
    *,
    length: int,
    split: str = "train",
    include_contact: bool = True,
    qpos: torch.Tensor | None = None,
) -> Path:
    kinematics = BumiKinematics(KINEMATICS_PATH)
    root.mkdir(parents=True)
    for name in ("meta", "manifests", "motions", "musicfeat_v2", "audio"):
        (root / name).mkdir()
    info = {
        "contract_version": "genmo.bumi_music.v1",
        "robot_name": "bumi",
        "qpos_dim": 28,
        "joint_dim": 21,
        "joint_names": list(kinematics.joint_order),
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "fps": 30,
        "quality_filter_applied": True,
        "mjcf_sha256": "a" * 64,
        "kinematics_sha256": kinematics.kinematics_sha256,
        "retarget_config_sha256": "b" * 64,
        "quality_config_sha256": "c" * 64,
        "source_mjcf_sha256": "a" * 64,
        "ground_semantics": "umr_foot_sole_ground_zero_v1",
        "root_z_adjusted": False,
        "reader_joint_limit_tolerance_rad": 0.0001,
        "root_orientation_gate": {
            "scope": "per_dataset_all_frames",
            "all_sequences_recomputed_and_dataset_passed": True,
        },
    }
    (root / "meta/dataset_info.json").write_text(json.dumps(info), encoding="utf-8")

    qpos = _make_qpos(length, kinematics) if qpos is None else qpos.clone().float()
    motion = {
        "qpos": qpos,
        "fps": 30,
        "robot_name": "bumi",
        "joint_names": list(kinematics.joint_order),
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "quality_accepted": True,
        "source_motion_sha256": "d" * 64,
        "source_mjcf_sha256": "a" * 64,
        "quality_config_sha256": "c" * 64,
        "retarget_config_sha256": "b" * 64,
        "ground_semantics": "umr_foot_sole_ground_zero_v1",
        "root_z_adjusted": False,
    }
    if include_contact:
        motion["foot_contact"] = torch.stack(
            (
                (torch.arange(length) % 2 == 0).float(),
                (torch.arange(length) % 3 == 0).float(),
            ),
            dim=-1,
        )
        motion["foot_contact_contract_version"] = BUMI_CONTACT_CONTRACT_VERSION
    motion_path = root / "motions/sample.pt"
    torch.save(motion, motion_path)

    music = torch.zeros(length, 35)
    music[:, 0] = torch.arange(length, dtype=torch.float32)
    music[:, 34] = (torch.arange(length) % 4 == 0).float()
    music_path = root / "musicfeat_v2/sample.pt"
    torch.save(music, music_path)
    audio_path = root / "audio/sample.wav"
    audio_path.write_bytes(b"closedloop-test-wave")
    row = {
        "sample_id": f"sample-{split}",
        "sequence_id": "sequence-0",
        "dataset": "test_bumi",
        "motion_path": "motions/sample.pt",
        "music_feature_path": "musicfeat_v2/sample.pt",
        "fps": 30,
        "num_frames": length,
        "split": split,
        "quality_accepted": True,
        "music_group_id": "music-0",
        "audio_key": "audio-0",
        "audio_path": "audio/sample.wav",
        "source_motion_sha256": "d" * 64,
        "source_music_feature_sha256": _sha256(music_path),
        "source_audio_sha256": _sha256(audio_path),
    }
    (root / f"manifests/{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


def _dataset(root: Path, **kwargs) -> BumiClosedLoopStage1Dataset:
    return BumiClosedLoopStage1Dataset(
        root=root,
        dataset_name="test_bumi",
        kinematics_path=KINEMATICS_PATH,
        split=kwargs.pop("split", "train"),
        random_decision=False,
        joint_limit_tolerance=0.0001,
        **kwargs,
    )


def _qpos_with_nominal_and_ramp(
    frames: int, builder: CausalDemoProprio48Builder, *, yaw_step: float, joint_speed: float
) -> torch.Tensor:
    qpos = builder.kinematics.default_qpos.view(1, 28).repeat(frames, 1)
    index = torch.arange(frames, dtype=torch.float32)
    yaw = index * yaw_step
    qpos[:, 3:7] = axis_angle_to_quaternion(
        torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=-1)
    )
    nominal = torch.tensor(GMT_NOMINAL_DEFAULT_JOINT_POS_RAD)
    joint_gmt = nominal[None, :] + index[:, None] * (joint_speed / 30.0)
    qpos[:, 7 + builder.gmt_from_source] = joint_gmt
    return qpos


def test_causal_150hz_grid_backward_velocity_and_joint_mapping() -> None:
    builder = CausalDemoProprio48Builder(BumiKinematics(KINEMATICS_PATH))
    assert tuple(builder.gmt_from_source.tolist()) == (
        9,
        15,
        0,
        10,
        16,
        1,
        5,
        11,
        17,
        2,
        6,
        12,
        18,
        3,
        7,
        13,
        19,
        4,
        8,
        14,
        20,
    )
    qpos = _qpos_with_nominal_and_ramp(6, builder, yaw_step=0.02, joint_speed=0.3)
    history, valid, times = builder.build_history(qpos[:4], decision_frame=3, history_steps=6)
    torch.testing.assert_close(
        times, torch.tensor([0.0, 0.02, 0.04, 0.06, 0.08, 0.10], dtype=torch.float64)
    )
    assert valid.tolist() == [False, False, True, True, True, True]
    assert bool(torch.isfinite(history).all())

    pos_start, pos_stop = PROPRIO_SLICES["joint_pos_rel"]
    vel_start, vel_stop = PROPRIO_SLICES["joint_vel_rel"]
    expected_source = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.float32)
    expected_position = expected_source[:, None] * (0.3 / 30.0)
    torch.testing.assert_close(
        history[2:, pos_start:pos_stop], expected_position[2:].expand(-1, 21), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        history[2:, vel_start:vel_stop], torch.full((4, 21), 0.3), atol=2e-5, rtol=0
    )
    ang_start, ang_stop = PROPRIO_SLICES["base_ang_vel"]
    torch.testing.assert_close(
        history[2:, ang_start:ang_stop],
        torch.tensor([[0.0, 0.0, 0.6]]).expand(4, -1),
        atol=2e-5,
        rtol=0,
    )
    assert history.shape[-1] == 48


def test_gravity_sign_quaternion_sign_and_causal_prefix_boundary() -> None:
    builder = CausalDemoProprio48Builder(BumiKinematics(KINEMATICS_PATH))
    qpos = _qpos_with_nominal_and_ramp(5, builder, yaw_step=0.0, joint_speed=0.0)
    roll = torch.tensor([[math.pi / 2, 0.0, 0.0]])
    qpos[1, 3:7] = axis_angle_to_quaternion(roll)[0]
    observation, valid = builder.source_observations(qpos[:2])
    gravity_start, gravity_stop = PROPRIO_SLICES["projected_gravity"]
    torch.testing.assert_close(
        observation[0, gravity_start:gravity_stop], torch.tensor([0.0, 0.0, -1.0])
    )
    torch.testing.assert_close(
        observation[1, gravity_start:gravity_stop],
        torch.tensor([0.0, -1.0, 0.0]),
        atol=1e-5,
        rtol=0,
    )
    assert valid.tolist() == [False, True]

    sign_flipped = qpos[:2].clone()
    sign_flipped[1, 3:7] *= -1.0
    flipped, _ = builder.source_observations(sign_flipped)
    torch.testing.assert_close(flipped, observation, atol=1e-5, rtol=0)

    with pytest.raises(ValueError, match="must contain frames 0..decision_frame exactly"):
        builder.build_history(qpos, decision_frame=2, history_steps=3)
    first = builder.build_history(qpos[:3], decision_frame=2, history_steps=3)
    changed_future = qpos.clone()
    changed_future[3:, 0] = 999.0
    second = builder.build_history(changed_future[:3], decision_frame=2, history_steps=3)
    for lhs, rhs in zip(first, second, strict=True):
        torch.testing.assert_close(lhs, rhs)


def test_history_computation_uses_only_the_required_bounded_past(monkeypatch) -> None:
    builder = CausalDemoProprio48Builder(BumiKinematics(KINEMATICS_PATH))
    qpos = _qpos_with_nominal_and_ramp(301, builder, yaw_step=0.001, joint_speed=0.0)
    observed_source_lengths: list[int] = []
    original = builder.source_observations

    def record_length(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed_source_lengths.append(len(value))
        return original(value)

    monkeypatch.setattr(builder, "source_observations", record_length)
    history, valid, _times = builder.build_history(qpos, decision_frame=300, history_steps=50)
    assert observed_source_lengths == [32]
    assert history.shape == (50, 48)
    assert bool(valid.all())


def test_stage1_sample_shapes_prefix_masks_and_collate(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "paired", length=140)
    dataset = _dataset(
        root,
        history_steps=7,
        prefix_min_frames=4,
        prefix_max_frames=4,
        duration_aware_sampling=False,
    )
    sample = dataset.get_window(0, start_frame=10)
    assert sample["music_features"].shape == (120, 35)
    assert sample["proprio_history"].shape == (7, 48)
    assert sample["known_qpos30"].shape == (120, 30)
    assert sample["target_qpos30"].shape == (120, 30)
    assert sample["target_qpos30_valid"].shape == (120, 30)
    assert sample["target_contact"].shape == (120, 2)
    assert sample["target_contact_valid"].shape == (120, 2)
    assert int(sample["prefix_frames"]) == 4
    assert bool(sample["known_qpos30_mask"][:3, :2].all())
    assert not bool(sample["known_qpos30_mask"][3:, :2].any())
    assert bool(sample["known_qpos30_mask"][:4, 2:].all())
    assert not bool(sample["known_qpos30_mask"][4:, 2:].any())
    assert torch.count_nonzero(sample["known_qpos30"][~sample["known_qpos30_mask"]]) == 0
    assert sample["meta"]["prefix_source"] == "teacher_forced_demo_reference_v1"
    assert sample["meta"]["proprio_construction_version"] == CAUSAL_PROPRIO_CONSTRUCTION_VERSION

    batch = collate_stage1_training_samples([sample, sample])
    assert batch["proprio_history"].shape == (2, 7, 48)
    assert batch["target_qpos30_valid"].shape == (2, 120, 30)
    validate_stage1_training_batch(batch, history_steps=7)


def test_right_halo_and_sequence_tail_have_coordinate_valid_masks(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "halo", length=121)
    dataset = _dataset(
        root,
        prefix_min_frames=10,
        prefix_max_frames=10,
        duration_aware_sampling=False,
    )
    full = dataset.get_window(0, start_frame=0)
    assert bool(full["target_qpos30_valid"][-1, :2].all())
    assert bool(full["future_valid"].all())

    tail = dataset.get_window(0, start_frame=120)
    assert tail["future_valid"].sum().item() == 1
    assert bool(tail["target_qpos30_valid"][0, 2:].all())
    assert not bool(tail["target_qpos30_valid"][0, :2].any())
    assert not bool(tail["target_qpos30_valid"][1:].any())
    assert int(tail["prefix_frames"]) == 0
    assert not bool(tail["known_qpos30_mask"].any())
    assert bool((tail["target_qpos30_valid"] & ~tail["known_qpos30_mask"]).any())
    assert bool(torch.isfinite(tail["target_qpos30"]).all())
    assert tail["meta"]["requested_prefix_frames"] == 10
    assert tail["meta"]["effective_prefix_frames"] == 0


def test_empty_and_variable_prefix_are_supported_without_contact_condition(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "prefix", length=40)
    empty = _dataset(
        root,
        prefix_min_frames=0,
        prefix_max_frames=0,
        duration_aware_sampling=False,
    ).get_window(0, start_frame=4)
    assert int(empty["prefix_frames"]) == 0
    assert not bool(empty["known_qpos30_mask"].any())
    assert "known_contact" not in empty

    variable_dataset = _dataset(
        root,
        prefix_min_frames=0,
        prefix_max_frames=8,
        prefix_random_seed=123,
        duration_aware_sampling=False,
    )
    first = variable_dataset.get_window(0, start_frame=4)
    second = variable_dataset.get_window(0, start_frame=4)
    assert 0 <= int(first["prefix_frames"]) <= 8
    assert int(first["prefix_frames"]) == int(second["prefix_frames"])
    torch.testing.assert_close(first["known_qpos30"], second["known_qpos30"])


def test_missing_contact_uses_existing_fk_label_contract_only_as_target(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "derived-contact", length=6, include_contact=False)
    sample = _dataset(root, duration_aware_sampling=False).get_window(0, start_frame=2)
    assert sample["meta"]["contact_source"] == "derived_from_full_sequence_qpos_fk"
    assert bool(sample["target_contact_valid"][:4].all())
    assert not bool(sample["target_contact_valid"][4:].any())
    assert "known_contact" not in sample


def test_manifest_split_is_preserved(tmp_path: Path) -> None:
    val_root = _write_dataset(tmp_path / "val", length=8, split="val")
    val_dataset = _dataset(val_root, split="val", duration_aware_sampling=False)
    sample = val_dataset.get_window(0, start_frame=0)
    assert sample["meta"]["split"] == "val"
    with pytest.raises(FileNotFoundError, match="train.jsonl"):
        _dataset(val_root, split="train", duration_aware_sampling=False)


def test_train_only_stats_entrypoint_and_fourset_config(tmp_path: Path, monkeypatch) -> None:
    root = _write_dataset(tmp_path / "stats-source", length=8)
    output = tmp_path / "stats" / "proprio48.json"
    arguments = [
        "compute_closedloop_proprio48_stats.py",
        "--kinematics",
        str(KINEMATICS_PATH),
        "--dataset",
        f"test_bumi={root}",
        "--output",
        str(output),
        "--joint-limit-tolerance",
        "0.0001",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    compute_proprio_stats_main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["contract_version"] == PROPRIO_STATS_CONTRACT_VERSION
    assert report["split"] == "train"
    assert report["feature_dim"] == 48
    assert report["is_placeholder"] is False
    assert report["num_valid_samples_50hz"] > 0
    assert report["joint_names"] == list(GMT_EXPECTED_JOINT_ORDER)
    assert report["dataset_fingerprints"]["test_bumi"]["sequences"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        compute_proprio_stats_main()

    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    assert config["server1_example"]["sources"] == [
        "AIST++",
        "AIOZ-GDANCE",
        "FineDance",
        "Mine",
    ]
    assert config["server1_example"]["excludes"] == ["CoMPAS3D"]
    assert config["qpos30_stats"]["mode"] == "reuse_existing_read_only"
    assert config["proprio48_stats"]["auto_compute"] is False
    assert set(config["datasets"]["train"]) == {"aistpp", "aioz_gdance", "finedance", "mine"}
    assert set(config["datasets"]["val"]) == {"aistpp", "aioz_gdance", "finedance"}
    assert set(config["datasets"]["test"]) == {"aistpp", "aioz_gdance", "finedance"}
    assert sha256_file(output) == _sha256(output)


def test_stats_dataset_argument_requires_an_explicit_absolute_root() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="absolute root"):
        parse_dataset("test_bumi=relative/data")
    name, root = parse_dataset("test_bumi=/tmp/closedloop-stats-source")
    assert name == "test_bumi"
    assert root == Path("/tmp/closedloop-stats-source")
