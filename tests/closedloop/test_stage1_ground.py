"""Stage1 混合数据地面监督的来源、缓存与原 BUMI 损失一致性测试。

本文件在 pytest 临时目录创建完整的 qpos28/EDGE35 序列，分别声明原有 floor-zero 和
legacy body-origin 坐标语义。验证 legacy 地面来自完整序列的现有 FK 足底分位估计，
不会从所选 crop 或网络预测估计；原始 payload、root Z、contact 标签不因适配改变。
同时验证缓存随源文件身份失效、meta 仅用于 loss、缺失/错误来源被拒绝，以及混合 batch
的地面相关物理项与旧 BumiRobotLosses 相同。测试不会访问服务器或产生正式训练产物；
使用的小型模型/临时统计量只验证软件契约，不能替代真实模型与真实数据短训验收。
"""

from __future__ import annotations

import copy
import json

import pytest
import torch

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS, validate_stage1_training_batch
from gem.closedloop.losses import Stage1BumiLosses
from gem.closedloop.stage1_dataset import (
    STAGE1_GROUND_SUPERVISION_VERSION,
    collate_stage1_training_samples,
)
from gem.robots.bumi.contacts import derive_bumi_foot_contact
from gem.robots.bumi.losses import BumiRobotLosses
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory
from tests.closedloop.test_stage1_dataset import _dataset, _write_dataset
from tests.closedloop.test_stage1_losses import _prediction, _weights
from tests.closedloop.test_stage1_losses import endecoder as endecoder


def _legacy_dataset(tmp_path, *, length=160):
    root = _write_dataset(tmp_path / "legacy", length=length)
    info_path = root / "meta/dataset_info.json"
    info = json.loads(info_path.read_text())
    info["ground_semantics"] = "legacy_body_origin_min_zero"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    motion_path = root / "motions/sample.pt"
    payload = torch.load(motion_path, weights_only=True)
    payload["ground_semantics"] = "legacy_body_origin_min_zero"
    payload["qpos"][:40, 2] -= 0.12
    payload["qpos"][40:, 2] += 0.20
    torch.save(payload, motion_path)
    return _dataset(root, duration_aware_sampling=False, history_steps=4), motion_path


def test_legacy_ground_uses_full_sequence_preserves_payload_and_caches(tmp_path, monkeypatch):
    dataset, path = _legacy_dataset(tmp_path)
    before = path.read_bytes()
    sequence = dataset.reader.load_aligned_sequence(dataset.rows[0])
    reference = derive_bumi_foot_contact(
        sequence["qpos"], dataset.kinematics, estimate_ground_mask=torch.tensor(True)
    )
    cropped = derive_bumi_foot_contact(
        sequence["qpos"][80:], dataset.kinematics, estimate_ground_mask=torch.tensor(True)
    )
    assert abs(float(reference.ground_height - cropped.ground_height)) > 0.25
    first = dataset.get_window(0, start_frame=80)
    ground = first["meta"]["ground_supervision"]
    assert ground["contract_version"] == STAGE1_GROUND_SUPERVISION_VERSION
    assert ground["ground_height_world_m"] == pytest.approx(float(reference.ground_height))
    assert ground["source_sequence_frames"] == 160
    assert ground["source_motion_path"] == str(path.resolve())
    assert ground["kinematics_sha256"] == dataset.kinematics.kinematics_sha256
    torch.testing.assert_close(first["target_contact"][:80], sequence["foot_contact"][80:])
    torch.testing.assert_close(
        first["target_qpos30"][:80, 2],
        sequence["qpos"][80:, 2] - dataset.codec.default_root_height,
    )

    def no_recompute(*_args, **_kwargs):
        raise AssertionError("same source must reuse cached ground, not rerun full-sequence FK")

    monkeypatch.setattr(dataset.kinematics, "forward_kinematics", no_recompute)
    cached = dataset._ground_supervision(sequence)
    assert cached == ground and cached is not ground
    assert len(dataset._ground_supervision_cache) == 1
    assert path.read_bytes() == before
    batch = collate_stage1_training_samples([first])
    validate_stage1_training_batch(batch, history_steps=4)
    assert "ground_supervision" not in STAGE1_CONDITION_KEYS


def test_ground_cache_invalidates_when_source_payload_changes(tmp_path):
    dataset, path = _legacy_dataset(tmp_path)
    sequence = dataset.reader.load_aligned_sequence(dataset.rows[0])
    first = dataset._ground_supervision(sequence)
    payload = torch.load(path, weights_only=True)
    payload["qpos"][:, 2] += 0.3
    torch.save(payload, path)
    changed = dataset.reader.load_aligned_sequence(dataset.rows[0])
    second = dataset._ground_supervision(changed)
    assert second["ground_height_world_m"] - first["ground_height_world_m"] == pytest.approx(
        0.3, abs=2e-6
    )


def test_explicit_floor_zero_does_not_estimate_ground(tmp_path, monkeypatch):
    dataset = _dataset(_write_dataset(tmp_path / "floor", length=8))
    sequence = dataset.reader.load_aligned_sequence(dataset.rows[0])

    def no_estimate(*_args, **_kwargs):
        raise AssertionError("explicit floor-zero must not estimate a data-dependent ground")

    monkeypatch.setattr(dataset.kinematics, "forward_kinematics", no_estimate)
    ground = dataset._ground_supervision(sequence)
    assert ground["ground_height_world_m"] == 0.0
    assert ground["method"] == "explicit_floor_zero"


def test_mixed_ground_physics_matches_existing_bumi_v5(tmp_path, endecoder):
    legacy, _ = _legacy_dataset(tmp_path)
    floor = _dataset(_write_dataset(tmp_path / "floor", length=160), history_steps=4)
    batch = collate_stage1_training_samples(
        [legacy.get_window(0, start_frame=30), floor.get_window(0, start_frame=30)]
    )
    pred = _prediction(endecoder, batch)
    pred[..., 2] += 0.03
    pred = pred.requires_grad_()
    logits = torch.zeros(2, 120, 2, requires_grad=True)
    criterion = Stage1BumiLosses(endecoder, _weights())
    total, result = criterion(batch, pred, logits)
    target_norm = _prediction(endecoder, batch)
    target_qpos = endecoder.compose_qpos(endecoder.decode(target_norm))
    decode = endecoder.decode(pred)
    qpos = endecoder.compose_qpos(decode)
    old_inputs = {
        "target_x": target_norm,
        "target_physical_features": endecoder.denormalize(target_norm),
        "target_qpos_canonical": target_qpos,
        "target_body_link_pos_root": endecoder.authoritative_body_link_positions_root(target_qpos),
        "target_foot_contact": batch["target_contact"],
        "target_foot_contact_mask": batch["target_contact_valid"],
        "target_contact_ground_height": criterion._ground_height_world(batch, pred).flatten(),
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
    for name in (
        "foot_contact_height",
        "foot_contact_height_topk",
        "penetration",
        "penetration_topk",
        "penetration_max",
        "contact_bce",
    ):
        torch.testing.assert_close(result[f"raw_{name}_loss"], reference[f"raw_{name}_loss"])
    total.backward()
    assert torch.isfinite(pred.grad).all() and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("fault", ["missing", "crop", "asset", "method", "nonfinite"])
def test_legacy_ground_rejects_missing_or_inconsistent_provenance(tmp_path, endecoder, fault):
    dataset, _ = _legacy_dataset(tmp_path)
    batch = collate_stage1_training_samples([dataset.get_window(0, start_frame=40)])
    ground = batch["meta"][0]["ground_supervision"]
    if fault == "missing":
        del batch["meta"][0]["ground_supervision"]
    elif fault == "crop":
        ground["source_sequence_frames"] = 120
    elif fault == "asset":
        ground["kinematics_sha256"] = "0" * 64
    elif fault == "method":
        ground["method"] = "estimated_from_current_prediction"
    else:
        ground["ground_height_world_m"] = float("nan")
    with pytest.raises(ValueError):
        Stage1BumiLosses(endecoder, _weights())._ground_height_world(batch, torch.zeros(1, 120, 30))


def test_ground_metadata_cannot_enter_actor_conditions(actor_factory):
    actor, batch = actor_factory(history_steps=4, starts=(45,))
    before = actor.adapt_conditions(batch)
    changed = copy.deepcopy(batch)
    changed["meta"][0]["ground_supervision"] = {"ground_height_world_m": float("nan")}
    after = actor.adapt_conditions(changed)
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(after[key], value)
