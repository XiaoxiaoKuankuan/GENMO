from __future__ import annotations

import math
import os

import pytest
import torch

from gem.robots.bumi.feature_codec import (
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
    BumiMotionFeatureCodec,
)
from gem.robots.bumi.kinematics import BumiKinematics
from gem.utils.rotation_conversions import (
    axis_angle_to_quaternion,
    quaternion_to_matrix,
)


def test_feature_slices_are_exactly_qpos30() -> None:
    assert dict(BUMI_FEATURE_SLICES) == {
        "root_delta_xy_heading": (0, 2),
        "root_height_offset": (2, 3),
        "root_rot_local": (3, 9),
        "joint_dof": (9, 30),
    }
    assert BUMI_FEATURE_DIM == 30
    assert BUMI_REPRESENTATION_CONTRACT_VERSION == "genmo.bumi_motion_features.qpos30.v3"
    assert BUMI_ANCHOR_MODE == "first_frame_xy_yaw_heading_delta_absolute_height"


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
    qpos[:, 3:7] = axis_angle_to_quaternion(torch.tensor([[0.0, 0.0, math.pi / 2]]).expand(3, -1))
    qpos[:, 0] = 5.0
    qpos[:, 1] = torch.tensor([7.0, 8.0, 9.0])
    encoded = codec.encode(qpos)
    torch.testing.assert_close(
        encoded.physical_features[0, :3],
        torch.tensor([1.0, 0.0, 0.1]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        encoded.physical_features[:, :2],
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        atol=1e-5,
        rtol=1e-5,
    )
    decoded = codec.decode_to_canonical_qpos(encoded.physical_features)
    torch.testing.assert_close(decoded, encoded.canonical_qpos, atol=1e-5, rtol=1e-5)
    world = codec.apply_world_anchor(decoded, torch.tensor([5.0, 7.0, math.pi / 2]))
    torch.testing.assert_close(world[..., :3], qpos[..., :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        quaternion_to_matrix(world[..., 3:7]),
        quaternion_to_matrix(qpos[..., 3:7]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(torch.linalg.vector_norm(world[..., 3:7], dim=-1), torch.ones(3))


def test_horizontal_rollout_does_not_integrate_root_height(test_kinematics_path) -> None:
    codec = BumiMotionFeatureCodec(BumiKinematics(test_kinematics_path))
    quaternion = axis_angle_to_quaternion(torch.tensor([[0.0, 0.0, math.pi / 2]]).expand(4, -1))
    delta = torch.tensor([[0.2, 0.0]]).expand(4, -1)
    height = torch.tensor([[0.1], [-0.2], [0.3], [0.0]])
    position = codec.rollout_root_position(delta, height, quaternion)
    torch.testing.assert_close(position[:, 0], torch.zeros(4), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        position[:, 1], torch.tensor([0.0, 0.2, 0.4, 0.6]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(position[:, 2:3], height)


def test_curved_heading_delta_round_trip_and_world_invariance(test_kinematics_path) -> None:
    codec = BumiMotionFeatureCodec(BumiKinematics(test_kinematics_path))
    frames = 12
    yaw = torch.linspace(-0.8, 1.1, frames)
    qpos = codec.kinematics.default_qpos.view(1, 28).repeat(frames, 1)
    qpos[:, 0] = 3.0 + torch.linspace(0.0, 1.4, frames)
    qpos[:, 1] = -2.0 + torch.sin(torch.linspace(0.0, math.pi, frames))
    qpos[:, 2] += 0.08 * torch.cos(torch.linspace(0.0, 2.0 * math.pi, frames))
    qpos[:, 3:7] = axis_angle_to_quaternion(
        torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=-1)
    )
    encoded = codec.encode(qpos)
    canonical = codec.decode_to_canonical_qpos(encoded.physical_features)
    recovered = codec.apply_world_anchor(canonical, torch.tensor([qpos[0, 0], qpos[0, 1], yaw[0]]))
    torch.testing.assert_close(recovered[:, :3], qpos[:, :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        quaternion_to_matrix(recovered[:, 3:7]),
        quaternion_to_matrix(qpos[:, 3:7]),
        atol=1e-5,
        rtol=1e-5,
    )

    moved = codec.apply_world_anchor(canonical, torch.tensor([-7.0, 4.0, 2.2]))
    moved_features = codec.encode(moved).physical_features
    torch.testing.assert_close(moved_features, encoded.physical_features, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not os.environ.get("BUMI_KINEMATICS_PATH"),
    reason="requires a real BUMI_KINEMATICS_PATH asset",
)
def test_real_bumi_codec_shape_only() -> None:
    kinematics = BumiKinematics(os.environ["BUMI_KINEMATICS_PATH"])
    codec = BumiMotionFeatureCodec(kinematics)
    qpos = kinematics.default_qpos.view(1, 28).repeat(2, 1)
    encoded = codec.encode(qpos)
    assert encoded.physical_features.shape == (2, 30)
    assert encoded.body_link_pos_root.shape == (2, 21, 3)


def test_link_positions_are_fk_only_and_not_network_features(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    codec = BumiMotionFeatureCodec(kinematics)
    qpos = kinematics.default_qpos.view(1, 28).repeat(4, 1)
    qpos[:, 7] = torch.linspace(0.0, 0.3, 4)
    encoded = codec.encode(qpos)
    split = codec.split_features(encoded.physical_features)
    assert split.body_link_pos_root is None
    decoded = codec.decode_to_canonical_qpos(encoded.physical_features)
    fk = kinematics.forward_kinematics(decoded)
    link_from_fk = codec.body_positions_in_root_frame(
        decoded[:, :3], decoded[:, 3:7], fk["body_pos_w"][:, 1:]
    )
    torch.testing.assert_close(link_from_fk, encoded.body_link_pos_root)
