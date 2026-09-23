"""Stage 1 损失的逐坐标 mask、真实 FK 和跨前缀差分回归测试。

测试使用仓库 fe934 BUMI 运动学资产与明确标记为 placeholder 的临时非恒等 stats，保证
physical/normalized 域真的不同；不生成正式统计量，也不运行第 5 步训练。所有数据保留
120 帧布局，真实有效部分可以很短。重点覆盖已知条件和未知目标的隔离、无 halo 的末帧
XY 排除、contact 独立 mask、padding 数值不变性、跨 prefix 边界速度，以及完整 v5
损失的有限反向梯度。测试只证明静态/运动学训练实现，不构成闭环或动力学验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from gem.closedloop.contracts import validate_stage1_training_batch
from gem.closedloop.losses import Stage1BumiLosses, coordinate_difference_mask
from gem.robots.bumi.endecoder import STATS_CONTRACT_VERSION, BumiEndecoder
from gem.robots.bumi.feature_codec import (
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_SLICES,
    BUMI_QUATERNION_CONVENTION,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.losses import BUMI_LOSS_NAMES_BY_CONTRACT, BumiRobotLosses

ROOT = Path(__file__).resolve().parents[2]
KINEMATICS = ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"


@pytest.fixture(scope="module")
def endecoder(tmp_path_factory) -> BumiEndecoder:
    kin = BumiKinematics(KINEMATICS)
    path = tmp_path_factory.mktemp("stage1_loss_stats") / "test_only_placeholder_stats.json"
    path.write_text(
        json.dumps(
            {
                "contract_version": STATS_CONTRACT_VERSION,
                "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
                "robot_name": "bumi",
                "feature_dim": 30,
                "anchor_mode": BUMI_ANCHOR_MODE,
                "quaternion_convention": BUMI_QUATERNION_CONVENTION,
                "training_clip_std_min": 0.01,
                "feature_slices": dict(BUMI_FEATURE_SLICES),
                "joint_names": list(kin.joint_order),
                "kinematics_sha256": kin.kinematics_sha256,
                "root_height_reference_m": float(kin.default_qpos[2]),
                "mean": torch.linspace(-0.7, 0.9, 30).tolist(),
                "std": torch.linspace(0.4, 1.7, 30).tolist(),
                "is_placeholder": True,
            }
        ),
        encoding="utf-8",
    )
    return BumiEndecoder(KINEMATICS, path, allow_placeholder_stats=True)


def _weights() -> dict[str, float]:
    config = yaml.safe_load(
        (ROOT / "configs/pipeline/music_only_bumi_qpos30_contact_v5.yaml").read_text()
    )
    return config["args"]["weights"]


def _batch(
    endecoder: BumiEndecoder, *, length: int = 7, prefix: int = 3
) -> dict[str, torch.Tensor]:
    qpos = endecoder.kinematics.default_qpos.view(1, 1, 28).repeat(1, 120, 1)
    qpos[0, :, 0] = torch.arange(120) * 0.01
    qpos[0, :, 7] += torch.arange(120) * 0.0001
    target = endecoder.codec.encode(qpos).physical_features
    future_valid = torch.arange(120)[None, :] < length
    target_valid = future_valid[..., None].expand(-1, -1, 30).clone()
    target_valid[:, length - 1, :2] = False
    known_mask = torch.zeros_like(target_valid)
    known_mask[:, :prefix, 2:] = True
    known_mask[:, : max(prefix - 1, 0), :2] = True
    batch = {
        "music_features": torch.zeros(1, 120, 35),
        "music_valid": future_valid.clone(),
        "proprio_history": torch.zeros(1, 4, 48),
        "proprio_history_valid": torch.zeros(1, 4, dtype=torch.bool),
        "proprio_history_times": torch.arange(-3, 1, dtype=torch.float64)[None] / 50,
        "known_qpos30": torch.where(known_mask, target, 0.0),
        "known_qpos30_mask": known_mask,
        "future_valid": future_valid,
        "future_times": torch.arange(120, dtype=torch.float64)[None] / 30,
        "decision_time": torch.zeros(1, dtype=torch.float64),
        "target_qpos30": torch.where(target_valid, target, 0.0),
        "target_qpos30_valid": target_valid,
        "target_contact": torch.zeros(1, 120, 2),
        "target_contact_valid": future_valid[..., None].expand(-1, -1, 2).clone(),
    }
    validate_stage1_training_batch(batch, history_steps=4)
    return batch


def _prediction(endecoder, batch) -> torch.Tensor:
    neutral = batch["target_qpos30"].new_zeros(30)
    neutral[3] = neutral[7] = 1.0
    neutral[9:] = endecoder.kinematics.default_qpos[7:]
    return endecoder.normalize(
        torch.where(batch["target_qpos30_valid"], batch["target_qpos30"], neutral)
    )


def test_reconstruction_means_only_valid_unknown_elements(endecoder):
    batch = _batch(endecoder)
    pred = _prediction(endecoder, batch)
    unknown = batch["target_qpos30_valid"] & ~batch["known_qpos30_mask"]
    pred = (pred + unknown.float() * 2.0).requires_grad_()
    criterion = Stage1BumiLosses(endecoder, _weights())
    _, values = criterion(batch, pred, torch.zeros(1, 120, 2))
    for name in ("repr_root_pos", "repr_root_rot", "repr_joint"):
        assert values[f"raw_{name}_loss"].item() == pytest.approx(4.0)
    assert values["raw_reconstruction_loss"].item() == pytest.approx(4.0)
    assert values["unknown_qpos30_elements"].item() == unknown.sum().item()


def test_full_v5_backward_blocks_known_and_terminal_invalid_coordinates(endecoder):
    batch = _batch(endecoder)
    pred = _prediction(endecoder, batch).requires_grad_()
    logits = torch.zeros(1, 120, 2, requires_grad=True)
    with torch.no_grad():
        pred[0, 2, :2] += 0.02  # prefix 最后状态帧：XY 已是未知，但本帧 joints 仍已知。
        pred[0, 3, 9] += 0.05
    loss, values = Stage1BumiLosses(endecoder, _weights())(batch, pred, logits)
    loss.backward()
    assert set(BUMI_LOSS_NAMES_BY_CONTRACT["physical_qpos30_contact_v5"]) <= {
        key.removeprefix("weighted_").removesuffix("_loss")
        for key in values
        if key.startswith("weighted_")
    }
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all() and torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(pred.grad[batch["known_qpos30_mask"]]) == 0
    assert torch.count_nonzero(pred.grad[~batch["target_qpos30_valid"]]) == 0
    assert pred.grad[0, 2, :2].abs().sum() > 0
    assert pred.grad[0, 3, 9].abs() > 0
    assert torch.count_nonzero(logits.grad[~batch["target_contact_valid"]]) == 0


def test_padding_invalid_targets_and_unknown_condition_placeholders_are_inert(endecoder):
    batch = _batch(endecoder)
    pred = _prediction(endecoder, batch)
    logits = torch.zeros(1, 120, 2)
    criterion = Stage1BumiLosses(endecoder, _weights())
    _, before = criterion(batch, pred, logits)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["target_qpos30"][~changed["target_qpos30_valid"]] = 1234.0
    changed["target_contact"][~changed["target_contact_valid"]] = -234.0
    changed["known_qpos30"][~changed["known_qpos30_mask"]] = -5432.0
    pred[~changed["target_qpos30_valid"] | changed["known_qpos30_mask"]] = 876.0
    logits[~changed["target_contact_valid"]] = 432.0
    _, after = criterion(changed, pred, logits)
    for key in before:
        torch.testing.assert_close(after[key], before[key], msg=key)


def test_joint_velocity_keeps_prefix_boundary_support_and_nonzero_target_speed(endecoder):
    batch = _batch(endecoder, length=4, prefix=3)
    pred = _prediction(endecoder, batch)
    physical = endecoder.denormalize(pred)
    physical[0, 3, 9] += 0.1
    pred = endecoder.normalize(physical).requires_grad_()
    _, values = Stage1BumiLosses(endecoder, _weights())(batch, pred, torch.zeros(1, 120, 2))
    # 3 个真实速度区间 × 21 joints，仅跨边界该坐标误差为 0.1 rad * 30 = 3 rad/s。
    assert values["raw_joint_velocity_loss"].item() == pytest.approx(
        (3.0 - 0.5) / (3 * 21), rel=1e-4
    )
    values["raw_joint_velocity_loss"].backward()
    assert pred.grad[0, 3, 9] > 0
    assert pred.grad[0, 2, 9] == 0


def test_contact_uses_per_foot_label_validity_and_independent_head(endecoder):
    batch = _batch(endecoder)
    batch["target_contact_valid"].zero_()
    batch["target_contact_valid"][0, 1, 0] = True
    batch["target_contact"].fill_(98.0)
    batch["target_contact"][0, 1, 0] = 1.0
    logits = torch.full((1, 120, 2), -8.0, requires_grad=True)
    _, values = Stage1BumiLosses(endecoder, _weights())(
        batch, _prediction(endecoder, batch), logits
    )
    expected = torch.nn.functional.softplus(torch.tensor(8.0))
    torch.testing.assert_close(values["raw_contact_bce_loss"], expected)
    values["raw_contact_bce_loss"].backward()
    assert logits.grad[0, 1, 0] < 0
    assert torch.count_nonzero(logits.grad) == 1


def test_p0_and_single_valid_state_do_not_create_nan(endecoder):
    batch = _batch(endecoder, length=1, prefix=0)
    pred = _prediction(endecoder, batch).requires_grad_()
    batch["target_contact_valid"].zero_()
    loss, values = Stage1BumiLosses(endecoder, _weights())(batch, pred, torch.zeros(1, 120, 2))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert values["raw_joint_velocity_loss"] == 0
    assert values["raw_contact_bce_loss"] == 0


def test_explicit_halo_allows_terminal_delta_reconstruction_without_fake_state(endecoder):
    batch = _batch(endecoder, length=120, prefix=0)
    batch["target_qpos30_valid"][0, -1, :2] = True
    batch["target_qpos30"][0, -1, :2] = torch.tensor([0.03, -0.01])
    pred = _prediction(endecoder, batch).requires_grad_()
    with torch.no_grad():
        pred[0, -1, :2] += 1.0
    _, values = Stage1BumiLosses(endecoder, _weights())(batch, pred, torch.zeros(1, 120, 2))
    values["raw_repr_root_pos_loss"].backward()
    assert pred.grad[0, -1, :2].min() > 0
    assert values["raw_root_pos_loss"].abs() < 1.0e-8


def test_physics_matches_existing_v5_on_p0_full_window(endecoder):
    batch = _batch(endecoder, length=120, prefix=0)
    pred = _prediction(endecoder, batch)
    pred[..., 9:] += 0.001
    logits = torch.zeros(1, 120, 2)
    new = Stage1BumiLosses(endecoder, _weights())
    _, values = new(batch, pred, logits)
    target_norm = _prediction(endecoder, batch)
    target_decode = endecoder.decode(target_norm)
    target_qpos = endecoder.compose_qpos(target_decode)
    target_body = endecoder.authoritative_body_link_positions_root(target_qpos)
    decode = endecoder.decode(pred)
    qpos = endecoder.compose_qpos(decode)
    old_inputs = {
        "target_x": target_norm,
        "target_physical_features": endecoder.denormalize(target_norm),
        "target_qpos_canonical": target_qpos,
        "target_body_link_pos_root": target_body,
        "target_foot_contact": batch["target_contact"],
        "target_foot_contact_mask": batch["target_contact_valid"],
        "target_contact_ground_height": torch.zeros(1),
        "mask": {"valid": batch["future_valid"]},
    }
    old = BumiRobotLosses(endecoder, _weights(), ground_semantics="mixed_floor_zero_fk_contact_v2")
    reference = old(
        old_inputs,
        {"pred_x_start": pred, "static_conf_logits": logits},
        decode,
        qpos,
        endecoder.kinematics.forward_kinematics(qpos),
    )
    for name in new.loss_names:
        if name.startswith("repr_"):
            continue  # 新主重建有逐坐标分母；旧实现只使用整帧分母，不能作为该部分参考。
        torch.testing.assert_close(
            values[f"raw_{name}_loss"],
            reference[f"raw_{name}_loss"],
            atol=1e-5,
            rtol=1e-4,
            msg=name,
        )


def test_strict_weight_and_floor_zero_contract(endecoder):
    bad = _weights() | {"invented_loss": 1.0}
    with pytest.raises(ValueError, match="Unknown BUMI loss weights"):
        Stage1BumiLosses(endecoder, bad)
    with pytest.raises(ValueError, match="floor-zero"):
        Stage1BumiLosses(endecoder, _weights(), ground_semantics="legacy_body_origin_min_zero")


def test_coordinate_difference_mask_never_uses_missing_support():
    valid = torch.tensor([[[True, True], [True, False], [True, True], [False, True]]])
    assert coordinate_difference_mask(valid, 1).tolist() == [
        [[True, False], [True, False], [False, True]]
    ]
    assert coordinate_difference_mask(valid, 2).tolist() == [[[True, False], [False, False]]]
