"""BUMI qpos30/contact→长音乐滑窗→安全流→50 Hz GMT 计划的纯 CPU 合约测试。

测试刻意不创建持久输出，也不依赖 Redis、TensorRT 或真实机器人。它覆盖 30D DDIM 与
接触 head、两窗 120/30 世界对齐与几何感知 overlap-add、FK 足底锁定、qpos 二进制包
CRC/revision/绝对帧号、跨分块速度与 XML 关节限位安全门，以及增量 30→50 Hz 计划和
离线重采样的一致性。
所有临时运动学/统计文件由 pytest ``tmp_path`` 管理，测试结束会自动删除。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.robots.bumi.contacts import derive_bumi_foot_contact
from gem.robots.bumi.endecoder import BumiEndecoder
from gem.robots.bumi.feature_codec import (
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.robots.bumi.kinematics import BumiKinematics, sha256_file
from gem.robots.bumi.metrics import compute_bumi_kinematic_metrics
from gem.robots.bumi.postprocess import lock_bumi_foot_contacts
from gem.runtime.bumi_gmt_plan import BumiIncrementalGmtPlanBuilder
from gem.runtime.bumi_music_deploy import BumiSlidingQposGenerator
from gem.runtime.bumi_robot_stream import (
    BumiQposChunk,
    BumiQposRevisionTracker,
    BumiQposSafetyGate,
    bumi_joint_order_sha256,
)
from gem.runtime.gmt_trajectory import qpos_timeline_to_gmt_frames, resample_qpos_timeline
from scripts.demo.demo_music_bumi import resolve_world_anchor


def _stats(path: Path, kinematics: BumiKinematics) -> Path:
    payload = {
        "contract_version": "genmo.bumi_qpos30_stats.v3",
        "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "robot_name": "bumi",
        "feature_dim": BUMI_FEATURE_DIM,
        "anchor_mode": BUMI_ANCHOR_MODE,
        "quaternion_convention": "wxyz",
        "feature_slices": {name: list(value) for name, value in BUMI_FEATURE_SLICES.items()},
        "joint_names": list(kinematics.joint_order),
        "kinematics_sha256": kinematics.kinematics_sha256,
        "mean": [0.0] * BUMI_FEATURE_DIM,
        "std": [1.0] * BUMI_FEATURE_DIM,
        "training_clip_std_min": 0.01,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_old_root_position_stats_are_rejected(test_kinematics_path: Path, tmp_path: Path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    stats = _stats(tmp_path / "old_stats.json", kinematics)
    payload = json.loads(stats.read_text(encoding="utf-8"))
    payload["contract_version"] = "genmo.bumi_stats.v1"
    payload.pop("representation_contract_version")
    stats.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract_version"):
        BumiEndecoder(test_kinematics_path, stats, enable_contact_targets=False)


class _FixedMotionStep:
    def __init__(self, normalized: torch.Tensor) -> None:
        self.normalized = normalized

    def __call__(self, noisy, timestep, music, length, guidance):
        del timestep, music, length, guidance
        motion = self.normalized.to(noisy).unsqueeze(0).expand_as(noisy)
        contact = noisy.new_full((noisy.shape[0], noisy.shape[1], 2), -20.0)
        return motion, contact


class _MusicSelectedMotionStep:
    """按窗口首个音乐标量选择固定预测，构造可复现的跨窗姿态差异。"""

    def __init__(self, first: torch.Tensor, second: torch.Tensor) -> None:
        self.first = first
        self.second = second

    def __call__(self, noisy, timestep, music, length, guidance):
        del timestep, length, guidance
        selected = self.second if float(music[0, 0, 0]) > 0.5 else self.first
        motion = selected.to(noisy).unsqueeze(0).expand_as(noisy)
        contact = noisy.new_full((noisy.shape[0], noisy.shape[1], 2), -20.0)
        return motion, contact


class _ReusedContactBufferStep:
    """模拟 TensorRT 每次调用覆写同一个输出 buffer。"""

    def __init__(self, normalized: torch.Tensor) -> None:
        self.normalized = normalized
        self.contact: torch.Tensor | None = None

    def __call__(self, noisy, timestep, music, length, guidance):
        del timestep, length, guidance
        if self.contact is None:
            self.contact = noisy.new_empty((noisy.shape[0], noisy.shape[1], 2))
        value = 10.0 if float(music[0, 0, 0]) > 0.5 else -10.0
        self.contact.fill_(value)
        return self.normalized.to(noisy).unsqueeze(0).expand_as(noisy), self.contact


def _qpos(frames: int, *, fps: float = 30.0, root_z: float = 1.0) -> np.ndarray:
    value = np.zeros((frames, 28), dtype=np.float32)
    value[:, 0] = np.arange(frames, dtype=np.float32) / np.float32(fps)
    value[:, 2] = root_z
    value[:, 3] = 1.0
    return value


def test_bumi_sliding_blends_overlap_and_preserves_world_root(
    test_kinematics_path: Path, tmp_path: Path
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    stats = _stats(tmp_path / "stats.json", kinematics)
    endecoder = BumiEndecoder(test_kinematics_path, stats, enable_contact_targets=False)
    canonical_window = torch.from_numpy(_qpos(120))
    normalized = endecoder.normalize(endecoder.codec.encode(canonical_window).physical_features)
    generated = BumiSlidingQposGenerator(
        _FixedMotionStep(normalized),
        endecoder,
        device="cpu",
        steps=2,
        overlap_atol=1.0e-4,
    ).generate(torch.zeros(210, 35), seed=7)
    assert generated.qpos.shape == (210, 28)
    assert [len(chunk.qpos) for chunk in generated.chunks] == [120, 90]
    assert [chunk.absolute_start_frame for chunk in generated.chunks] == [0, 120]
    np.testing.assert_allclose(generated.qpos[:, 0].numpy(), np.arange(210) / 30.0, atol=1.0e-5)
    np.testing.assert_allclose(generated.qpos[:, 2].numpy(), 1.0, atol=1.0e-6)


def test_bumi_sliding_blends_independent_root_rotation_and_joint_predictions(
    test_kinematics_path: Path, tmp_path: Path
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    stats = _stats(tmp_path / "stats.json", kinematics)
    endecoder = BumiEndecoder(test_kinematics_path, stats, enable_contact_targets=False)
    first_qpos = torch.from_numpy(_qpos(120))
    second_qpos = torch.from_numpy(_qpos(120))
    source_yaw = torch.linspace(1.0, 1.6, 120)
    distance = torch.arange(120, dtype=torch.float32) / 30.0
    second_qpos[:, 0] = distance * torch.cos(source_yaw[0])
    second_qpos[:, 1] = distance * torch.sin(source_yaw[0])
    second_qpos[:, 3] = torch.cos(source_yaw * 0.5)
    second_qpos[:, 6] = torch.sin(source_yaw * 0.5)
    second_qpos[:, 7] = 1.0
    first_normalized = endecoder.normalize(endecoder.codec.encode(first_qpos).physical_features)
    second_normalized = endecoder.normalize(endecoder.codec.encode(second_qpos).physical_features)
    music = torch.zeros(210, 35)
    music[90:, 0] = 1.0
    generated = BumiSlidingQposGenerator(
        _MusicSelectedMotionStep(first_normalized, second_normalized),
        endecoder,
        device="cpu",
        steps=2,
        overlap_atol=1.0e-4,
    ).generate(music, seed=11)

    expected_x = torch.arange(210, dtype=torch.float32) / 30.0
    torch.testing.assert_close(generated.qpos[:, 0], expected_x, atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(
        torch.cat(tuple(chunk.qpos for chunk in generated.chunks)), generated.qpos
    )
    expected_alpha = torch.arange(1, 31, dtype=torch.float32) / 31.0
    torch.testing.assert_close(generated.qpos[90:120, 7], expected_alpha, atol=1.0e-5, rtol=0.0)
    assert float(generated.qpos[120, 7]) == pytest.approx(1.0, abs=1.0e-6)
    joint_steps = torch.diff(generated.qpos[89:122, 7]).abs()
    assert float(joint_steps.max()) <= 1.0 / 31.0 + 1.0e-5

    quaternion = generated.qpos[:, 3:7]
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    angular_steps = 2.0 * torch.acos(
        (quaternion[:-1] * quaternion[1:]).sum(dim=-1).abs().clamp(max=1.0)
    )
    assert float(angular_steps[89:121].max()) < 0.02


def test_sliding_clones_reused_tensorrt_contact_output(
    test_kinematics_path: Path, tmp_path: Path
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    stats = _stats(tmp_path / "stats.json", kinematics)
    endecoder = BumiEndecoder(test_kinematics_path, stats, enable_contact_targets=False)
    canonical = torch.from_numpy(_qpos(120))
    normalized = endecoder.normalize(endecoder.codec.encode(canonical).physical_features)
    music = torch.zeros(210, 35)
    music[90:, 0] = 1.0
    generated = BumiSlidingQposGenerator(
        _ReusedContactBufferStep(normalized),
        endecoder,
        device="cpu",
        steps=2,
        apply_foot_lock=False,
    ).generate(music, seed=17)
    torch.testing.assert_close(generated.foot_contact_logits[:90], torch.full((90, 2), -10.0))
    torch.testing.assert_close(generated.foot_contact_logits[120:], torch.full((90, 2), 10.0))


def test_fk_foot_lock_only_changes_root_xy_and_reduces_slide(
    test_kinematics_path: Path,
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    qpos = torch.from_numpy(_qpos(12))
    qpos[:, 0] = torch.arange(12, dtype=torch.float32) * 0.01
    original = qpos.clone()
    result = lock_bumi_foot_contacts(
        qpos,
        torch.full((12, 2), 10.0),
        kinematics,
        max_correction_per_frame=0.08,
    )
    torch.testing.assert_close(result.qpos[:, 2:], original[:, 2:])
    assert float(result.mean_contact_slide_after_mps) < 1.0e-5
    assert float(result.mean_contact_slide_before_mps) > 0.2


def test_fk_contact_labels_respect_ground_semantics_and_speed(
    test_kinematics_path: Path,
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    qpos = torch.from_numpy(_qpos(6))
    qpos[:, 0] = 0.0
    valid = torch.ones(6, dtype=torch.bool)
    exact = derive_bumi_foot_contact(qpos, kinematics, valid_mask=valid)
    assert bool(exact.contact.all())

    shifted = qpos.clone()
    shifted[:, 2] += 0.4
    exact_shifted = derive_bumi_foot_contact(shifted, kinematics, valid_mask=valid)
    assert not bool(exact_shifted.contact.any())
    estimated = derive_bumi_foot_contact(
        shifted,
        kinematics,
        valid_mask=valid,
        ground_height=0.0,
        estimate_ground_mask=torch.tensor(True),
    )
    assert bool(estimated.contact.all())
    assert float(estimated.ground_height) == pytest.approx(0.4, abs=1.0e-5)

    moving = qpos.clone()
    moving[:, 0] = torch.arange(6, dtype=torch.float32) * 0.02
    fast = derive_bumi_foot_contact(moving, kinematics, valid_mask=valid)
    assert not bool(fast.contact.any())


def test_fk_contact_mixed_ground_estimation_is_batched_and_ignores_invalid_frames(
    test_kinematics_path: Path,
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    qpos = torch.from_numpy(_qpos(6)).unsqueeze(0).repeat(3, 1, 1)
    qpos[:, :, 0] = 0.0
    qpos[1, :, 2] += 0.4
    qpos[2, :, 2] += 0.8
    valid = torch.ones(3, 6, dtype=torch.bool)
    valid[2] = False
    targets = derive_bumi_foot_contact(
        qpos,
        kinematics,
        valid_mask=valid,
        ground_height=0.0,
        estimate_ground_mask=torch.tensor([False, True, True]),
    )
    torch.testing.assert_close(targets.ground_height, torch.tensor([0.0, 0.4, 0.0]))
    assert bool(targets.contact[0].all())
    assert bool(targets.contact[1].all())
    assert not bool(targets.contact[2].any())


def test_kinematic_metrics_accept_per_sample_ground_height(test_kinematics_path: Path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    qpos = torch.from_numpy(_qpos(6)).unsqueeze(0).repeat(2, 1, 1)
    qpos[:, :, 0] = 0.0
    qpos[1, :, 2] += 0.4
    metrics = compute_bumi_kinematic_metrics(
        qpos,
        kinematics,
        ground_height=torch.tensor([0.1, 0.4]),
    )
    assert float(metrics["foot_penetration_mean_m"]) == pytest.approx(0.05, abs=1.0e-6)
    assert float(metrics["foot_penetration_max_m"]) == pytest.approx(0.1, abs=1.0e-6)


def _chunk(qpos: np.ndarray, *, index: int, start: int, total: int, last: bool) -> BumiQposChunk:
    return BumiQposChunk.from_qpos(
        qpos,
        request_id="request-a",
        revision=3,
        chunk_index=index,
        absolute_start_frame=start,
        total_frames=total,
        is_last=last,
        checkpoint_sha256="c" * 64,
        engine_sha256="e" * 64,
        kinematics_sha256="a" * 64,
        joint_order_sha256="b" * 64,
    )


def test_bumi_qpos_wire_crc_revision_and_identity() -> None:
    first = _chunk(_qpos(2), index=0, start=0, total=4, last=False)
    decoded = BumiQposChunk.from_multipart(first.multipart())
    np.testing.assert_array_equal(decoded.qpos(), first.qpos())
    corrupted = first.multipart()
    corrupted[1] = corrupted[1][:-1] + bytes((corrupted[1][-1] ^ 1,))
    with pytest.raises(ValueError, match="CRC32"):
        BumiQposChunk.from_multipart(corrupted)
    tracker = BumiQposRevisionTracker()
    tracker.begin("request-a", 3, 4)
    tracker.accept(decoded)
    with pytest.raises(ValueError, match="index"):
        tracker.accept(decoded)
    tracker.accept(_chunk(_qpos(2), index=1, start=2, total=4, last=True))
    assert tracker.complete
    with pytest.raises(ValueError, match="already complete"):
        tracker.accept(_chunk(_qpos(2), index=1, start=2, total=4, last=True))
    with pytest.raises(ValueError, match="newer revision"):
        tracker.begin("request-b", 3, 1)


def test_bumi_qpos_terminal_chunk_requires_last_marker() -> None:
    with pytest.raises(ValueError, match="must be marked is_last"):
        _chunk(_qpos(2), index=0, start=0, total=2, last=False)


def test_bumi_safety_gate_checks_chunk_boundary_and_xml_limits(
    test_kinematics_path: Path,
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    gate = BumiQposSafetyGate(
        kinematics,
        max_root_linear_velocity_mps=2.0,
        max_joint_velocity_radps=5.0,
        min_root_height_m=0.5,
        max_root_height_m=1.2,
    )
    first = _qpos(2)
    first[:, 0] *= 0.1
    gate.validate(first)
    discontinuous = _qpos(1)
    discontinuous[0, 0] = 10.0
    with pytest.raises(ValueError, match="root linear velocity"):
        gate.validate(discontinuous)
    gate.reset()
    limited = _qpos(1)
    limited[0, 7] = 1.2
    with pytest.raises(ValueError, match="joint-limit"):
        gate.validate(limited)


def test_bumi_incremental_plan_matches_offline_30_to_50() -> None:
    source = _qpos(210)
    idle = _qpos(1)[0]
    builder = BumiIncrementalGmtPlanBuilder(
        idle, np.arange(21), blend_seconds=0.02, return_seconds=0.02
    )
    first = builder.append(source[:120], is_last=False)
    final = builder.append(source[120:], is_last=True)
    assert not first.qpos.flags.writeable and not final.frames.flags.writeable
    resampled = resample_qpos_timeline(source, 30.0, 50.0)
    action_start = final.audio_start_frame
    np.testing.assert_allclose(
        final.qpos[action_start : action_start + len(resampled)], resampled, atol=1.0e-6
    )
    expected_frames = qpos_timeline_to_gmt_frames(final.qpos, fps=50.0, native_to_gmt=np.arange(21))
    np.testing.assert_allclose(final.frames, expected_frames, atol=1.0e-5)


def test_joint_order_hash_is_stable(test_kinematics_path: Path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    first = bumi_joint_order_sha256(list(kinematics.joint_order))
    second = bumi_joint_order_sha256(list(kinematics.joint_order))
    assert first == second and len(first) == 64
    assert sha256_file(test_kinematics_path) == kinematics.kinematics_sha256


def test_bumi_render_defaults_to_world_anchor() -> None:
    args = SimpleNamespace(
        world_root_x=None,
        world_root_y=None,
        world_root_z=None,
        world_root_yaw=None,
        render_mjcf=Path("bumi3.xml"),
    )
    assert resolve_world_anchor(args, default_root_height=0.65) == {
        "root_xy": [0.0, 0.0],
        "yaw": 0.0,
        "anchor_z": 0.65,
    }
    args.render_mjcf = None
    assert resolve_world_anchor(args, default_root_height=0.65) is None
