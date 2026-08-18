from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

from gem.datamodule.music_robot_trainX_testY import DataModule
from gem.datasets.music_dance.music_dance_bumi import (
    BumiMusicDanceDataset,
    sha256_file,
)


def make_dataset(root, kinematics_path, **kwargs):
    return BumiMusicDanceDataset(
        root=root,
        dataset_name="test_bumi",
        kinematics_path=kinematics_path,
        split="train",
        motion_frames=120,
        strict_alignment=True,
        strict_contract=True,
        require_quality_filter=True,
        validate_payloads_on_init=True,
        **kwargs,
    )


def test_manifest_sync_crop_and_short_padding(dataset_factory, test_kinematics_path) -> None:
    root = dataset_factory(length=8)
    item = make_dataset(root, test_kinematics_path)[0]
    assert item["qpos"].shape == (120, 28)
    assert item["music_embed"].shape == (120, 35)
    assert item["length"] == 8
    assert item["mask"]["valid"].sum().item() == 8
    assert item["mask"]["has_music_mask"].sum().item() == 8
    torch.testing.assert_close(item["qpos"][:8, 0], item["music_embed"][:8, 0])
    assert torch.equal(item["qpos"][8:], item["qpos"][7:8].expand(112, -1))
    assert torch.count_nonzero(item["music_embed"][8:]) == 0


def test_long_random_crop_stays_aligned(dataset_factory, test_kinematics_path) -> None:
    root = dataset_factory(length=150)
    item = make_dataset(root, test_kinematics_path)[0]
    torch.testing.assert_close(item["qpos"][:, 0], item["music_embed"][:, 0])


def test_bad_quaternion_is_rejected(dataset_factory, test_kinematics_path) -> None:
    root = dataset_factory(length=8, quaternion=torch.zeros(8, 4))
    with pytest.raises(ValueError, match="quaternion norm"):
        make_dataset(root, test_kinematics_path)


def test_wrong_motion_joint_order_is_rejected(dataset_factory, test_kinematics_path) -> None:
    wrong = [f"wrong_{index}" for index in range(21)]
    root = dataset_factory(length=8, motion_joint_names=wrong)
    with pytest.raises(ValueError, match="joint_names"):
        make_dataset(root, test_kinematics_path)


def test_wrong_contract_version_is_rejected(dataset_factory, test_kinematics_path) -> None:
    root = dataset_factory(length=8, contract_version="wrong")
    with pytest.raises(ValueError, match="contract_version"):
        make_dataset(root, test_kinematics_path)


def test_failed_quality_is_rejected(dataset_factory, test_kinematics_path) -> None:
    root = dataset_factory(length=8, quality_accepted=False)
    with pytest.raises(ValueError, match="quality_accepted"):
        make_dataset(root, test_kinematics_path)


def _formal_datamodule(root, kinematics_path, stats_path, expected_sequences=1):
    dataset = {
        "_target_": (
            "gem.datasets.music_dance.music_dance_bumi.BumiMusicDanceDataset"
        ),
        "root": str(root),
        "dataset_name": "test_bumi",
        "kinematics_path": str(kinematics_path),
        "split": "train",
        "motion_frames": 120,
        "duration_aware_sampling": False,
        "validate_payloads_on_init": False,
        "random_crop": False,
    }
    return DataModule(
        dataset_opts=OmegaConf.create({"train": {"test": dataset}}),
        loader_opts=OmegaConf.create({"train": {"batch_size": 1, "num_workers": 0}}),
        sampling_strategy="deduplicated_hierarchical",
        samples_per_epoch=8,
        stats_path=stats_path,
        require_stats_fingerprint_match=True,
        expected_train_sequences=expected_sequences,
    )


def test_formal_datamodule_requires_exact_stats_fingerprint(
    dataset_factory, test_kinematics_path, tmp_path
) -> None:
    root = dataset_factory(length=8)
    stats_path = tmp_path / "stats.json"
    stats = {
        "dataset_fingerprints": {
            "test_bumi": {
                "dataset_info_sha256": sha256_file(root / "meta" / "dataset_info.json"),
                "train_manifest_sha256": sha256_file(root / "manifests" / "train.jsonl"),
                "sequences": 1,
            }
        }
    }
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    module = _formal_datamodule(root, test_kinematics_path, stats_path)
    assert len(module.train_datasets) == 1

    stats["dataset_fingerprints"]["test_bumi"]["train_manifest_sha256"] = "0" * 64
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match="stats fingerprint mismatch"):
        _formal_datamodule(root, test_kinematics_path, stats_path)


def test_formal_datamodule_requires_expected_train_count(
    dataset_factory, test_kinematics_path, tmp_path
) -> None:
    root = dataset_factory(length=8)
    stats_path = tmp_path / "unused.json"
    with pytest.raises(ValueError, match="train sequence count mismatch"):
        _formal_datamodule(root, test_kinematics_path, stats_path, expected_sequences=2)
