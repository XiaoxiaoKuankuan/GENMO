from __future__ import annotations

import pytest
import torch

from gem.datasets.music_dance.music_dance_bumi import BumiMusicDanceDataset


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
