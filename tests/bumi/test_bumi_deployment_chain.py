"""BUMI ONNX→长音乐滑窗→安全流→50 Hz GMT 计划的纯 CPU 合约测试。

测试刻意不创建持久输出，也不依赖 Redis、TensorRT 或真实机器人。它覆盖 93D DDIM 的
可配置维度、两窗 120/30 世界对齐与几何感知 overlap-add、qpos 二进制包
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

from gem.robots.bumi.endecoder import BumiEndecoder
from gem.robots.bumi.feature_codec import BUMI_ANCHOR_MODE, BUMI_FEATURE_SLICES
from gem.robots.bumi.kinematics import BumiKinematics, sha256_file
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
        "contract_version": "genmo.bumi_stats.v1",
        "robot_name": "bumi",
        "feature_dim": 93,
        "anchor_mode": BUMI_ANCHOR_MODE,
        "quaternion_convention": "wxyz",
        "feature_slices": {name: list(value) for name, value in BUMI_FEATURE_SLICES.items()},
        "joint_names": list(kinematics.joint_order),
        "kinematics_sha256": kinematics.kinematics_sha256,
        "mean": [0.0] * 93,
        "std": [1.0] * 93,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _FixedMotionStep:
    def __init__(self, normalized: torch.Tensor) -> None:
        self.normalized = normalized

    def __call__(self, noisy, timestep, music, length, guidance):
        del timestep, music, length, guidance
        return self.normalized.to(noisy).unsqueeze(0).expand_as(noisy)


class _MusicSelectedMotionStep:
    """按窗口首个音乐标量选择固定预测，构造可复现的跨窗姿态差异。"""

    def __init__(self, first: torch.Tensor, second: torch.Tensor) -> None:
        self.first = first
        self.second = second

    def __call__(self, noisy, timestep, music, length, guidance):
        del timestep, length, guidance
        selected = self.second if float(music[0, 0, 0]) > 0.5 else self.first
        return selected.to(noisy).unsqueeze(0).expand_as(noisy)


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
