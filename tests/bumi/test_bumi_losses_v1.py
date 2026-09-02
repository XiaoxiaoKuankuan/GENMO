"""BUMI qpos30/contact/FK 物理损失的纯 CPU 合约测试。

测试确认网络损失只接收 30 维 qpos 表示，link 监督来自预测 qpos 的可微 FK；同时覆盖
完整 SO(3) 根旋转、专用 roll/pitch tilt、可靠 GT 接触门控 foot-slide、接触 head BCE 和
辅助项 warmup。测试运动学由 ``conftest`` 临时生成，不写入正式训练目录。
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.losses import (
    BUMI_EXCESS_LOSS_NAMES,
    BUMI_LOSS_CONTRACT_V3,
    BUMI_LOSS_CONTRACT_VERSION,
    BUMI_LOSS_NAMES,
    BumiRobotLosses,
    derivative_excess_loss_values,
    root_tilt_loss_values,
    so3_geodesic_angle,
    temporal_difference_mask,
)
from gem.utils.rotation_conversions import axis_angle_to_matrix


def _weights() -> dict[str, float]:
    values = {name: 0.0 for name in BUMI_LOSS_NAMES}
    values.update(
        {
            "repr_root_pos": 1.0,
            "repr_root_rot": 2.0,
            "repr_joint": 1.0,
            "root_pos": 0.2,
            "root_rot": 1.0,
            "root_tilt": 1.0,
            "joint_dof": 0.2,
            "fk_body_pos": 1.0,
            "joint_velocity": 0.05,
            "joint_acceleration": 0.005,
            "joint_jerk": 0.001,
            "joint_limit": 0.1,
            "contact_bce": 1.0,
            "foot_slide": 0.05,
            "penetration": 0.05,
            "root_height": 0.1,
        }
    )
    return values


def _encoded_inputs(
    encoded: object,
    valid: torch.Tensor,
    contact: torch.Tensor,
) -> dict[str, object]:
    return {
        "target_x": encoded.physical_features,
        "target_physical_features": encoded.physical_features,
        "target_qpos_canonical": encoded.canonical_qpos,
        "target_body_link_pos_root": encoded.body_link_pos_root,
        "target_foot_contact": contact,
        "target_foot_contact_mask": valid[..., None].expand_as(contact),
        "target_contact_ground_height": torch.zeros(valid.shape[:-1]),
        "mask": {"valid": valid},
    }


def test_difference_masks_require_two_three_four_real_frames() -> None:
    valid = torch.tensor([[True, True, True, False, False]])
    assert temporal_difference_mask(valid, 1).tolist() == [[True, True, False, False]]
    assert temporal_difference_mask(valid, 2).tolist() == [[True, False, False]]
    assert temporal_difference_mask(valid, 3).tolist() == [[False, False]]


def test_derivative_excess_only_penalizes_prediction_above_target() -> None:
    prediction = torch.tensor([[[3.0, -1.0], [9.0, -8.0]]], requires_grad=True)
    target = torch.tensor([[[4.0, -1.0], [5.0, -10.0]]])
    mask = torch.tensor([[True, True]])
    raw, normalized = derivative_excess_loss_values(prediction, target, mask, scale=2.0)
    assert float(raw) == pytest.approx(0.875)
    assert float(normalized) == pytest.approx(0.375)
    normalized.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0].abs().sum().item() == pytest.approx(0.0)
    assert prediction.grad[0, 1, 0].item() > 0.0


def test_so3_wraparound_near_plus_minus_pi_is_continuous() -> None:
    pred = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, math.pi - 1.0e-4]]))
    target = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, -math.pi + 1.0e-4]]))
    assert float(so3_geodesic_angle(pred, target)) < 3.0e-4


def test_root_tilt_penalizes_roll_pitch_but_not_yaw() -> None:
    target = axis_angle_to_matrix(torch.zeros(1, 1, 3))
    yaw = axis_angle_to_matrix(torch.tensor([[[0.0, 0.0, 1.2]]]))
    roll = axis_angle_to_matrix(torch.tensor([[[0.9, 0.0, 0.0]]]))
    rolled_target = axis_angle_to_matrix(torch.tensor([[[0.4, 0.0, 0.0]]]))
    yawed_same_roll = yaw @ rolled_target
    valid = torch.ones(1, 1, dtype=torch.bool)
    yaw_raw, _ = root_tilt_loss_values(yaw, target, valid)
    roll_raw, _ = root_tilt_loss_values(roll, target, valid)
    yawed_roll_raw, _ = root_tilt_loss_values(yawed_same_roll, rolled_target, valid)
    assert float(yaw_raw) == pytest.approx(0.0, abs=1.0e-7)
    assert float(yawed_roll_raw) == pytest.approx(0.0, abs=1.0e-7)
    assert float(roll_raw) > 0.1


def test_qpos30_contact_losses_fp32_fk_and_warmup(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    codec = BumiMotionFeatureCodec(kinematics)
    endecoder = SimpleNamespace(kinematics=kinematics, codec=codec)
    loss = BumiRobotLosses(
        endecoder,
        _weights(),
        contract_version=BUMI_LOSS_CONTRACT_VERSION,
        auxiliary_warmup_steps=10000,
        ground_semantics="mixed_floor_zero_fk_contact_v2",
    )
    qpos = kinematics.default_qpos.view(1, 1, 28).repeat(1, 6, 1)
    encoded = codec.encode(qpos)
    target = encoded.physical_features
    pred = target.clone()
    pred[:, :, 0] = 0.01
    pred[:, :, 9] = torch.linspace(0.0, 0.1, 6)
    pred.requires_grad_(True)
    parts = codec.split_features(pred)
    pred_qpos = codec.decode_to_canonical_qpos(pred)
    fk = kinematics.forward_kinematics(pred_qpos)
    valid = torch.tensor([[True, True, True, True, False, False]])
    contact = torch.ones(1, 6, 2)
    inputs = _encoded_inputs(encoded, valid, contact)
    decode = {
        "root_delta_xy_heading": parts.root_delta_xy_heading,
        "root_height_offset": parts.root_height_offset,
        "root_rot_local_6d": pred[..., 3:9],
        "joint_dof": parts.joint_dof,
    }
    model_output = {
        "pred_x": pred.half(),
        "static_conf_logits": torch.zeros_like(contact, requires_grad=True),
    }
    at_zero = loss(inputs, model_output, decode, pred_qpos, fk, global_step=0)
    assert at_zero["loss"].dtype == torch.float32
    assert float(at_zero["weighted_joint_dof_loss"]) == 0.0
    assert float(at_zero["weighted_repr_joint_loss"]) > 0.0
    assert float(at_zero["weighted_contact_bce_loss"]) > 0.0
    at_full = loss(inputs, model_output, decode, pred_qpos, fk, global_step=10000)
    assert float(at_full["weighted_joint_dof_loss"]) > 0.0
    assert float(at_full["weighted_foot_slide_loss"]) > 0.0
    for name in BUMI_LOSS_NAMES:
        for prefix in ("raw", "normalized", "weighted"):
            assert torch.isfinite(at_full[f"{prefix}_{name}_loss"])
    at_full["loss"].backward()
    assert pred.grad is not None and bool(torch.isfinite(pred.grad).all())


def test_contact_and_slide_weights_are_mandatory(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    endecoder = SimpleNamespace(kinematics=kinematics, codec=BumiMotionFeatureCodec(kinematics))
    weights = _weights()
    weights["foot_slide"] = 0.0
    with pytest.raises(ValueError, match="positive contact_bce and foot_slide"):
        BumiRobotLosses(
            endecoder,
            weights,
            contract_version=BUMI_LOSS_CONTRACT_VERSION,
            ground_semantics="mixed_floor_zero_fk_contact_v2",
        )


def test_v3_excess_losses_are_versioned_and_emitted(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    codec = BumiMotionFeatureCodec(kinematics)
    endecoder = SimpleNamespace(kinematics=kinematics, codec=codec)
    weights = _weights()
    weights.update(
        {
            "joint_acceleration_excess": 0.05,
            "joint_jerk_excess": 0.003,
        }
    )
    loss = BumiRobotLosses(
        endecoder,
        weights,
        contract_version=BUMI_LOSS_CONTRACT_V3,
        ground_semantics="mixed_floor_zero_fk_contact_v2",
    )
    qpos = kinematics.default_qpos.view(1, 1, 28).repeat(1, 6, 1)
    encoded = codec.encode(qpos)
    pred = encoded.physical_features.clone()
    pred[0, :, 9] = torch.tensor([0.0, 0.1, -0.1, 0.1, -0.1, 0.0])
    pred.requires_grad_(True)
    parts = codec.split_features(pred)
    pred_qpos = codec.decode_to_canonical_qpos(pred)
    fk = kinematics.forward_kinematics(pred_qpos)
    valid = torch.ones(1, 6, dtype=torch.bool)
    contact = torch.ones(1, 6, 2)
    output = loss(
        _encoded_inputs(encoded, valid, contact),
        {
            "pred_x": pred,
            "static_conf_logits": torch.zeros_like(contact, requires_grad=True),
        },
        {
            "root_delta_xy_heading": parts.root_delta_xy_heading,
            "root_height_offset": parts.root_height_offset,
            "root_rot_local_6d": pred[..., 3:9],
            "joint_dof": parts.joint_dof,
        },
        pred_qpos,
        fk,
    )
    for name in BUMI_EXCESS_LOSS_NAMES:
        assert float(output[f"weighted_{name}_loss"]) > 0.0
    output["loss"].backward()
    assert pred.grad is not None and bool(torch.isfinite(pred.grad).all())


def test_v2_rejects_v3_only_excess_weights(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    endecoder = SimpleNamespace(kinematics=kinematics, codec=BumiMotionFeatureCodec(kinematics))
    weights = _weights()
    weights["joint_acceleration_excess"] = 0.05
    with pytest.raises(ValueError, match="Unknown BUMI loss weights"):
        BumiRobotLosses(
            endecoder,
            weights,
            contract_version=BUMI_LOSS_CONTRACT_VERSION,
            ground_semantics="mixed_floor_zero_fk_contact_v2",
        )


def test_old_loss_contract_is_rejected(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    endecoder = SimpleNamespace(kinematics=kinematics, codec=BumiMotionFeatureCodec(kinematics))
    with pytest.raises(ValueError, match="qpos30"):
        BumiRobotLosses(
            endecoder,
            _weights(),
            contract_version="physical_v1",
            ground_semantics="legacy_body_origin_min_zero",
        )
