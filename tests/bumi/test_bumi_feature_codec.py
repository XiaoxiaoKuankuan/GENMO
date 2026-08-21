from __future__ import annotations

import math
import os

import pytest
import torch

from gem.robots.bumi.feature_codec import (
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BumiMotionFeatureCodec,
)
from gem.robots.bumi.kinematics import BumiKinematics
from gem.utils.rotation_conversions import (
    axis_angle_to_quaternion,
    quaternion_to_matrix,
)


def test_feature_slices_are_exactly_93d() -> None:
    assert dict(BUMI_FEATURE_SLICES) == {
        "root_pos_local": (0, 3),
        "root_rot_local": (3, 9),
        "joint_dof": (9, 30),
        "body_link_pos_local": (30, 93),
    }
    assert BUMI_FEATURE_DIM == 93


def test_root_rot6d_round_trip(test_kinematics_path) -> None:
    codec = BumiMotionFeatureCodec(BumiKinematics(test_kinematics_path))
    axis_angle = torch.tensor([[0.2, -0.1, 1.1], [-0.4, 0.3, -2.2]])
    quaternion = axis_angle_to_quaternion(axis_angle)
    recovered = codec.rotation_features_to_quat(codec.rotation_quat_to_features(quaternion))
    torch.testing.assert_close(
        quaternion_to_matrix(recovered), quaternion_to_matrix(quaternion), atol=1e-5, rtol=1e-5
    )


def test_canonical_anchor_and_qpos_round_trip(test_kinematics_path) -> None:
    codec = BumiMotionFeatureCodec(BumiKinematics(test_kinematics_path))
    qpos = torch.zeros(3, 28)
    qpos[:, 2] = torch.tensor([1.1, 1.2, 1.3])
    qpos[:, 3:7] = axis_angle_to_quaternion(
        torch.tensor([[0.0, 0.0, math.pi / 2]]).expand(3, -1)
    )
    qpos[:, 0] = 5.0
    qpos[:, 1] = torch.tensor([7.0, 8.0, 9.0])
    encoded = codec.encode(qpos)
    torch.testing.assert_close(encoded.physical_features[0, :3], torch.tensor([0.0, 0.0, 0.1]))
    torch.testing.assert_close(encoded.physical_features[1, :2], torch.tensor([1.0, 0.0]), atol=1e-5, rtol=1e-5)
    decoded = codec.decode_to_canonical_qpos(encoded.physical_features)
    torch.testing.assert_close(decoded, encoded.canonical_qpos, atol=1e-5, rtol=1e-5)
    world = codec.apply_world_anchor(decoded, torch.tensor([5.0, 7.0, math.pi / 2]))
    torch.testing.assert_close(world[..., :3], qpos[..., :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        quaternion_to_matrix(world[..., 3:7]), quaternion_to_matrix(qpos[..., 3:7]), atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(torch.linalg.vector_norm(world[..., 3:7], dim=-1), torch.ones(3))


@pytest.mark.skipif(
    not os.environ.get("BUMI_KINEMATICS_PATH"),
    reason="requires a real BUMI_KINEMATICS_PATH asset",
)
def test_real_bumi_codec_shape_only() -> None:
    kinematics = BumiKinematics(os.environ["BUMI_KINEMATICS_PATH"])
    codec = BumiMotionFeatureCodec(kinematics)
    qpos = kinematics.default_qpos.view(1, 28).repeat(2, 1)
    assert codec.encode(qpos).physical_features.shape == (2, 93)
