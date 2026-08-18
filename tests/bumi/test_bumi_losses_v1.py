from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.losses import (
    BUMI_LOSS_NAMES,
    BumiRobotLosses,
    so3_geodesic_angle,
    temporal_difference_mask,
)
from gem.utils.rotation_conversions import axis_angle_to_matrix


def _weights() -> dict[str, float]:
    values = {name: 0.0 for name in BUMI_LOSS_NAMES}
    values.update(
        {
            "repr_root_pos": 1.0,
            "repr_root_rot": 1.0,
            "repr_joint": 1.0,
            "repr_body_pos": 1.0,
            "root_pos": 0.1,
            "root_rot": 0.1,
            "joint_dof": 0.1,
            "fk_body_pos": 0.5,
            "fk_consistency": 0.1,
            "joint_velocity": 0.01,
            "joint_acceleration": 0.002,
            "joint_limit": 0.01,
            "root_height": 0.05,
        }
    )
    return values


def test_difference_masks_require_two_three_four_real_frames() -> None:
    valid = torch.tensor([[True, True, True, False, False]])
    assert temporal_difference_mask(valid, 1).tolist() == [[True, True, False, False]]
    assert temporal_difference_mask(valid, 2).tolist() == [[True, False, False]]
    assert temporal_difference_mask(valid, 3).tolist() == [[False, False]]


def test_so3_wraparound_near_plus_minus_pi_is_continuous() -> None:
    pred = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, math.pi - 1.0e-4]]))
    target = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, -math.pi + 1.0e-4]]))
    assert float(so3_geodesic_angle(pred, target)) < 3.0e-4


def test_physical_v1_fp32_logs_and_warmup(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    codec = BumiMotionFeatureCodec(kinematics)
    endecoder = SimpleNamespace(kinematics=kinematics, codec=codec)
    loss = BumiRobotLosses(
        endecoder,
        _weights(),
        contract_version="physical_v1",
        auxiliary_warmup_steps=10000,
        ground_semantics="legacy_body_origin_min_zero",
    )
    qpos = kinematics.default_qpos.view(1, 1, 28).repeat(1, 6, 1)
    qpos[:, :, 0] = torch.arange(6) * 0.01
    encoded = codec.encode(qpos)
    target = encoded.physical_features
    offset = torch.zeros_like(target)
    offset[:, :, 9] = torch.linspace(0.0, 0.1, 6)
    pred = (target.clone() + offset).requires_grad_(True)
    parts = codec.split_features(pred)
    pred_qpos = codec.decode_to_canonical_qpos(pred)
    fk = kinematics.forward_kinematics(pred_qpos)
    inputs = {
        "target_x": target,
        "target_physical_features": target,
        "target_body_link_pos_local": encoded.body_link_pos_local,
        "mask": {"valid": torch.tensor([[True, True, True, True, False, False]])},
    }
    decode = {
        "root_pos_local": parts.root_pos_local,
        "root_rot_local_6d": pred[..., 3:9],
        "joint_dof": parts.joint_dof,
        "body_link_pos_local_raw": parts.body_link_pos_local,
    }
    at_zero = loss(
        inputs, {"pred_x": pred.half()}, decode, pred_qpos, fk["body_pos_w"][..., 1:, :],
        fk["body_quat_w"], global_step=0
    )
    assert at_zero["loss"].dtype == torch.float32
    assert float(at_zero["weighted_joint_dof_loss"]) == 0.0
    assert float(at_zero["weighted_repr_joint_loss"]) > 0.0
    at_full = loss(
        inputs, {"pred_x": pred.half()}, decode, pred_qpos, fk["body_pos_w"][..., 1:, :],
        fk["body_quat_w"], global_step=10000
    )
    assert float(at_full["weighted_joint_dof_loss"]) > 0.0
    for name in BUMI_LOSS_NAMES:
        for prefix in ("raw", "normalized", "weighted"):
            assert torch.isfinite(at_full[f"{prefix}_{name}_loss"])
    at_full["loss"].backward()
    assert pred.grad is not None and bool(torch.isfinite(pred.grad).all())


def test_ground_losses_are_hard_disabled_for_legacy_ground(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    endecoder = SimpleNamespace(
        kinematics=kinematics, codec=BumiMotionFeatureCodec(kinematics)
    )
    weights = _weights()
    weights["penetration"] = 0.001
    with pytest.raises(ValueError, match="cannot enable"):
        BumiRobotLosses(
            endecoder,
            weights,
            contract_version="physical_v1",
            ground_semantics="legacy_body_origin_min_zero",
        )
