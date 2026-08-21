from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch


def write_test_kinematics(path: Path) -> Path:
    joint_names = [f"test_joint_{index:02d}" for index in range(21)]
    body_names = [f"test_body_{index:02d}" for index in range(21)]
    default_qpos = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, *([0.0] * 21)]
    spec = {
        "contract_version": "genmo.bumi_kinematics.v1",
        "robot_name": "bumi",
        "source_mjcf": "test_only.xml",
        "source_mjcf_sha256": "1" * 64,
        "proxy_config_sha256": "2" * 64,
        "root_body": "test_root",
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "qpos_dim": 28,
        "joint_dim": 21,
        "qpos_layout": "root_xyz_3 + root_quaternion_wxyz_4 + joint_dof_21",
        "body_order": ["test_root", *body_names],
        "feature_body_names": body_names,
        "body_name_to_index": {
            name: index for index, name in enumerate(["test_root", *body_names])
        },
        "joint_order": joint_names,
        "joint_name_to_qpos_index": {name: index for index, name in enumerate(joint_names)},
        "joint_name_to_qpos_address": {name: index + 7 for index, name in enumerate(joint_names)},
        "joint_qpos_addresses": list(range(7, 28)),
        "parent_body_indices": [0] * 21,
        "child_body_indices": list(range(1, 22)),
        "joint_axes": [[0.0, 0.0, 1.0] for _ in range(21)],
        "joint_origin_xyz": [[0.05 * (index + 1), 0.0, -0.5] for index in range(21)],
        "joint_origin_quat_wxyz": [[1.0, 0.0, 0.0, 0.0] for _ in range(21)],
        "joint_anchor_xyz": [[0.0, 0.0, 0.0] for _ in range(21)],
        "joint_lower_limits": [-1.0] * 21,
        "joint_upper_limits": [1.0] * 21,
        "default_qpos": default_qpos,
        "sole_proxies": [
            {
                "name": "test_left_sole",
                "feature_body_name": body_names[0],
                "feature_body_index": 1,
                "local_position": [0.0, 0.0, -0.5],
                "local_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "radius": 0.0,
                "foot_id": 0,
            },
            {
                "name": "test_right_sole",
                "feature_body_name": body_names[1],
                "feature_body_index": 2,
                "local_position": [0.0, 0.0, -0.5],
                "local_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "radius": 0.0,
                "foot_id": 1,
            },
        ],
        "evaluation_proxies": [],
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def test_kinematics_path(tmp_path: Path) -> Path:
    return write_test_kinematics(tmp_path / "test_kinematics.json")


def write_dataset(
    root: Path,
    kinematics_path: Path,
    *,
    length: int = 8,
    contract_version: str = "genmo.bumi_music.v1",
    info_joint_names: list[str] | None = None,
    motion_joint_names: list[str] | None = None,
    quaternion: torch.Tensor | None = None,
    quality_accepted: bool = True,
    root_z_adjusted: bool = False,
    ground_semantics: str = "legacy_body_origin_min_zero",
    reader_joint_limit_tolerance_rad: float | None = None,
) -> Path:
    spec = json.loads(kinematics_path.read_text(encoding="utf-8"))
    joint_names = spec["joint_order"]
    (root / "meta").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "motions").mkdir()
    (root / "musicfeat_v2").mkdir()
    info = {
        "contract_version": contract_version,
        "robot_name": "bumi",
        "qpos_dim": 28,
        "joint_dim": 21,
        "joint_names": joint_names if info_joint_names is None else info_joint_names,
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "fps": 30,
        "quality_filter_applied": True,
        "mjcf_sha256": "a" * 64,
        "kinematics_sha256": sha256(kinematics_path),
        "retarget_config_sha256": "b" * 64,
        "quality_config_sha256": "c" * 64,
        "source_mjcf_sha256": "a" * 64,
        "ground_semantics": ground_semantics,
        "root_z_adjusted": root_z_adjusted,
    }
    if reader_joint_limit_tolerance_rad is not None:
        info["reader_joint_limit_tolerance_rad"] = reader_joint_limit_tolerance_rad
    (root / "meta" / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    qpos = torch.zeros(length, 28)
    qpos[:, 0] = torch.arange(length)
    qpos[:, 2] = 1.0
    qpos[:, 3] = 1.0
    if quaternion is not None:
        qpos[:, 3:7] = quaternion
    motion = {
        "qpos": qpos,
        "fps": 30,
        "robot_name": "bumi",
        "joint_names": joint_names if motion_joint_names is None else motion_joint_names,
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "quality_accepted": quality_accepted,
        "source_motion_sha256": "d" * 64,
        "source_mjcf_sha256": "a" * 64,
        "quality_config_sha256": "c" * 64,
        "retarget_config_sha256": "b" * 64,
        "ground_semantics": ground_semantics,
        "root_z_adjusted": root_z_adjusted,
    }
    torch.save(motion, root / "motions" / "sample.pt")
    music = torch.zeros(length, 35)
    music[:, 0] = torch.arange(length)
    music[:, 34] = (torch.arange(length) % 4 == 0).float()
    torch.save(music, root / "musicfeat_v2" / "sample.pt")
    (root / "audio").mkdir()
    (root / "audio" / "sample.wav").write_bytes(b"test-wave-payload")
    row = {
        "sample_id": "sample",
        "sequence_id": "sequence",
        "dataset": "test_bumi",
        "motion_path": "motions/sample.pt",
        "music_feature_path": "musicfeat_v2/sample.pt",
        "fps": 30,
        "num_frames": length,
        "split": "train",
        "quality_accepted": quality_accepted,
        "music_group_id": "test_song",
        "audio_key": "sample",
        "audio_path": "audio/sample.wav",
        "source_motion_sha256": "d" * 64,
        "source_music_feature_sha256": sha256(root / "musicfeat_v2" / "sample.pt"),
        "source_audio_sha256": sha256(root / "audio" / "sample.wav"),
    }
    (root / "manifests" / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def dataset_factory(tmp_path: Path, test_kinematics_path: Path):
    counter = 0

    def factory(**kwargs):
        nonlocal counter
        counter += 1
        root = tmp_path / f"dataset_{counter}"
        return write_dataset(root, test_kinematics_path, **kwargs)

    return factory
