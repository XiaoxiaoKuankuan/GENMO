"""验证 BUMI 统一根高、历史统计量兼容以及按实际待机姿态贴地的边界。

所有 JSON 写入 pytest 的临时目录，不接触正式数据、checkpoint 或部署模型。通过还原
旧基准下的编解码作为独立对照，检查旧模型归一化输入、任意高度预测的世界 qpos 和
梯度保持一致；同时检查新 stats 的显式根高、非法/混合契约拒绝和资产字节不变。
站姿测试使用带俯仰关节的合成运动学，分别验证零位与屈膝的 FK 贴地，不能作为实机证明。
"""

from __future__ import annotations

import json

import pytest
import torch

from gem.robots.bumi.endecoder import BumiEndecoder, STATS_CONTRACT_VERSION
from gem.robots.bumi.feature_codec import (
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
    BumiMotionFeatureCodec,
)
from gem.robots.bumi.kinematics import BUMI_DEFAULT_ROOT_HEIGHT_M, BumiKinematics


def _stats(path, kin, *, version="genmo.bumi_qpos30_stats.v3", reference=None):
    value = dict(
        contract_version=version,
        representation_contract_version=BUMI_REPRESENTATION_CONTRACT_VERSION,
        robot_name="bumi",
        feature_dim=BUMI_FEATURE_DIM,
        anchor_mode=BUMI_ANCHOR_MODE,
        quaternion_convention="wxyz",
        feature_slices=dict(BUMI_FEATURE_SLICES),
        joint_names=list(kin.joint_order),
        kinematics_sha256=kin.kinematics_sha256,
        mean=torch.linspace(-0.2, 0.3, 30).tolist(),
        std=torch.linspace(0.1, 1.0, 30).tolist(),
        training_clip_std_min=0.01,
    )
    if version == STATS_CONTRACT_VERSION:
        value["root_height_reference_m"] = reference
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


@pytest.mark.parametrize("source_height", [0.55, 0.65, 1.0])
def test_legacy_stats_preserve_network_input_world_output_and_gradient(
    test_kinematics_path, tmp_path, source_height
):
    spec = json.loads(test_kinematics_path.read_text())
    spec["default_qpos"][2] = source_height
    test_kinematics_path.write_text(json.dumps(spec))
    asset_bytes = test_kinematics_path.read_bytes()
    kin = BumiKinematics(test_kinematics_path)
    assert float(kin.default_qpos[2]) == pytest.approx(0.48120910, abs=1e-8)
    assert float(kin.source_default_qpos[2]) == pytest.approx(source_height)
    stats_path = tmp_path / "legacy_stats.json"
    stats = _stats(stats_path, kin)
    stats_bytes = stats_path.read_bytes()
    endecoder = BumiEndecoder(test_kinematics_path, stats_path, enable_contact_targets=False)

    # 旧实现的原点来自资产 qpos0，独立恢复它作为对照。
    old_kin = BumiKinematics(test_kinematics_path)
    old_kin.default_qpos.copy_(old_kin.source_default_qpos)
    old_codec = BumiMotionFeatureCodec(old_kin)
    mean, std = torch.tensor(stats["mean"]), torch.tensor(stats["std"])
    qpos = old_kin.default_qpos[None].repeat(4, 1)
    qpos[:, 0] = torch.tensor([2.0, 2.1, 2.3, 2.4])
    qpos[:, 2] += torch.tensor([-0.2, 0.0, 0.35, 0.1])
    qpos[:, 7:] = torch.linspace(-0.3, 0.3, 21)
    old_normalized = (old_codec.encode(qpos).physical_features - mean) / std
    new_normalized = endecoder.normalize(endecoder.codec.encode(qpos).physical_features)
    torch.testing.assert_close(new_normalized, old_normalized, atol=1e-6, rtol=1e-5)

    # 非 GT 的高度预测同样保持世界输出，不能仅凭 GT 往返通过来判断兼容。
    prediction = old_normalized.clone()
    prediction[:, 2] += torch.tensor([0.1, -0.3, 0.2, 0.05])
    prediction.requires_grad_(True)
    anchor = torch.tensor([2.0, 0.0, 0.0])
    old_world = old_codec.apply_world_anchor(
        old_codec.decode_to_canonical_qpos(prediction * std + mean), anchor
    )
    new_world = endecoder.codec.apply_world_anchor(
        endecoder.codec.decode_to_canonical_qpos(endecoder.denormalize(prediction)), anchor
    )
    torch.testing.assert_close(new_world, old_world, atol=1e-6, rtol=1e-5)
    old_grad = torch.autograd.grad(old_world[:, 2].sum(), prediction, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new_world[:, 2].sum(), prediction)[0]
    torch.testing.assert_close(new_grad, old_grad)
    assert test_kinematics_path.read_bytes() == asset_bytes
    assert stats_path.read_bytes() == stats_bytes
    assert "source_default_qpos" not in kin.state_dict()


def test_new_stats_reference_is_explicit_and_not_applied_twice(test_kinematics_path, tmp_path):
    kin = BumiKinematics(test_kinematics_path)
    path = tmp_path / "current_stats.json"
    stats = _stats(path, kin, version=STATS_CONTRACT_VERSION, reference=BUMI_DEFAULT_ROOT_HEIGHT_M)
    endecoder = BumiEndecoder(test_kinematics_path, path, enable_contact_targets=False)
    torch.testing.assert_close(endecoder.mean, torch.tensor(stats["mean"]), atol=0, rtol=0)
    torch.testing.assert_close(endecoder.std, torch.tensor(stats["std"]), atol=0, rtol=0)
    assert endecoder.stats_root_height_reference_m == BUMI_DEFAULT_ROOT_HEIGHT_M


def test_music_stats_writer_and_loader_share_the_height_reference(
    test_kinematics_path, dataset_factory, tmp_path, monkeypatch
):
    from tools.data.bumi.compute_bumi_30d_stats import main

    root = dataset_factory(length=120)
    path = tmp_path / "computed_stats.json"
    monkeypatch.setattr(
        "sys.argv",
        ["compute_bumi_30d_stats.py", "--kinematics", str(test_kinematics_path),
         "--dataset", f"test_bumi={root}", "--output", str(path)],
    )
    main()
    stats = json.loads(path.read_text())
    assert stats["contract_version"] == STATS_CONTRACT_VERSION
    assert stats["root_height_reference_m"] == pytest.approx(0.48120910, abs=1e-8)
    endecoder = BumiEndecoder(test_kinematics_path, path, enable_contact_targets=False)
    qpos = endecoder.kinematics.source_default_qpos[None].repeat(120, 1)
    normalized = endecoder.normalize(endecoder.codec.encode(qpos).physical_features)
    torch.testing.assert_close(normalized[:, 2], torch.zeros(120), atol=1e-5, rtol=0)


@pytest.mark.parametrize("reference", [None, True, 0.0, -0.1, float("nan"), float("inf"), "0.48120910"])
def test_new_stats_require_a_valid_reference(test_kinematics_path, tmp_path, reference):
    path = tmp_path / "invalid_stats.json"
    _stats(path, BumiKinematics(test_kinematics_path), version=STATS_CONTRACT_VERSION, reference=reference)
    with pytest.raises(ValueError, match="root_height_reference_m"):
        BumiEndecoder(test_kinematics_path, path, enable_contact_targets=False)


def test_explicit_reference_cannot_be_mislabeled_as_legacy(test_kinematics_path, tmp_path):
    path = tmp_path / "mixed_stats.json"
    stats = _stats(path, BumiKinematics(test_kinematics_path))
    stats["root_height_reference_m"] = BUMI_DEFAULT_ROOT_HEIGHT_M
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="requires"):
        BumiEndecoder(test_kinematics_path, path, enable_contact_targets=False)


@pytest.mark.parametrize("bent", [False, True])
def test_idle_pose_uses_actual_fk_without_changing_the_reference(test_kinematics_path, bent):
    spec = json.loads(test_kinematics_path.read_text())
    spec["joint_axes"] = [[0.0, 1.0, 0.0] for _ in range(21)]
    test_kinematics_path.write_text(json.dumps(spec))
    kin = BumiKinematics(test_kinematics_path)
    default = kin.default_qpos.clone()
    source = kin.source_default_qpos.clone()
    joints = torch.full((21,), 0.25 if bent else 0.0)
    pose = kin.make_standing_qpos(joints if bent else None)
    fk = kin.forward_kinematics(pose)
    sole = kin.get_sole_proxy_points(fk["body_pos_w"], fk["body_quat_w"])
    assert float(sole["bottom_height"].amin()) == pytest.approx(0.002, abs=1e-6)
    torch.testing.assert_close(pose[7:], joints)
    torch.testing.assert_close(kin.default_qpos, default, atol=0, rtol=0)
    torch.testing.assert_close(kin.source_default_qpos, source, atol=0, rtol=0)
    if bent:
        assert float(pose[2]) != pytest.approx(float(kin.make_standing_qpos()[2]), abs=1e-4)
