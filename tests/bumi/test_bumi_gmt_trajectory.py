"""验证当前 BUMI qpos 到 GMT 的时间轴、增量计划和数据包契约。

保留旧混合测试中的导数、关节映射、CRC、ACK、边界补帧以及完整/分块一致性；
计划构建和对齐改为调用现行 BumiIncrementalGmtPlanBuilder，消除旧 SMPL 桥依赖。
测试用合成 qpos 和内存 Redis 替身，不启动 GMT 或连接真实 Redis。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from gem.runtime.bumi_gmt_plan import (
    BumiIncrementalGmtPlanBuilder,
    align_qpos_action,
    interpolate_qpos,
)
from gem.runtime.gmt_trajectory import (
    BUMI_QPOS_DIM,
    TRAJECTORY_CURRENT_INDEX,
    TRAJECTORY_FRAME_COUNT,
    TRAJECTORY_FRAME_DIM,
    TRAJECTORY_PACKET_BYTES,
    GmtTrajectoryAck,
    GmtTrajectoryPacket,
    RedisTrajectoryPublisher,
    build_playback_timeline,
    joint_order_sha256,
    packet_for_cursor,
    qpos_timeline_to_gmt_frames,
    resample_qpos_timeline,
    rolling_window_indices,
)


def make_qpos(num_frames: int = 200, fps: float = 50.0) -> np.ndarray:
    qpos = np.zeros((num_frames, BUMI_QPOS_DIM), dtype=np.float32)
    qpos[:, 3] = 1.0
    seconds = np.arange(num_frames, dtype=np.float32) / np.float32(fps)
    qpos[:, 0] = seconds
    for joint in range(21):
        qpos[:, 7 + joint] = seconds * np.float32(joint + 1)
    return qpos


def make_frames(num_frames: int = 200) -> np.ndarray:
    return qpos_timeline_to_gmt_frames(make_qpos(num_frames), fps=50.0, native_to_gmt=np.arange(21))


def order_hash() -> bytes:
    return joint_order_sha256([f"joint_{index}" for index in range(21)])


def test_complete_qpos_derivatives_and_joint_permutation() -> None:
    permutation = np.arange(20, -1, -1)
    frames = qpos_timeline_to_gmt_frames(make_qpos(), fps=50.0, native_to_gmt=permutation)
    assert frames.shape == (200, TRAJECTORY_FRAME_DIM)
    np.testing.assert_allclose(frames[50, 7:10], [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(frames[:, 10:13], 0.0, atol=1e-6)
    np.testing.assert_allclose(frames[50, 13:34], make_qpos()[50, 7:][permutation])
    np.testing.assert_allclose(frames[50, 34:55], np.arange(21, 0, -1), atol=2e-4)


def test_rolling_packet_contains_true_past_current_and_future() -> None:
    frames = make_frames()
    packet = packet_for_cursor(
        frames,
        60,
        fps=50.0,
        joint_order_hash=order_hash(),
        stream_id=7,
        sequence=9,
    )
    indices = rolling_window_indices(len(frames), 60)
    np.testing.assert_array_equal(indices, np.arange(50, 160))
    np.testing.assert_allclose(packet.frames, frames[indices])
    np.testing.assert_allclose(packet.frames[TRAJECTORY_CURRENT_INDEX], frames[60])
    # GMT's half-window=10 reads rows 0..20.  Linear root x proves these are
    # 21 temporal samples, not one legacy frame copied 21 times.
    assert np.unique(packet.frames[:21, 0]).size == 21
    np.testing.assert_allclose(packet.frames[:21, 0], frames[50:71, 0])


def test_rolling_boundaries_repeat_only_missing_context() -> None:
    frames = make_frames(20)
    first = packet_for_cursor(
        frames,
        0,
        fps=50.0,
        joint_order_hash=order_hash(),
        stream_id=1,
        sequence=0,
    )
    assert np.all(first.frames[:11, 0] == frames[0, 0])
    assert np.unique(first.frames[11:21, 0]).size == 10
    last = packet_for_cursor(
        frames,
        19,
        fps=50.0,
        joint_order_hash=order_hash(),
        stream_id=1,
        sequence=1,
    )
    np.testing.assert_allclose(last.frames[10], frames[-1])
    np.testing.assert_allclose(
        last.frames[11:], np.repeat(frames[-1:], len(last.frames) - 11, axis=0)
    )


def test_packet_roundtrip_size_crc_and_finite_validation() -> None:
    packet = packet_for_cursor(
        make_frames(),
        40,
        fps=50.0,
        joint_order_hash=order_hash(),
        stream_id=123,
        sequence=456,
        command_revision=2,
        plan_id=3,
    )
    encoded = packet.encode()
    assert len(encoded) == TRAJECTORY_PACKET_BYTES
    decoded = GmtTrajectoryPacket.decode(encoded)
    assert decoded.stream_id == 123
    assert decoded.sequence == 456
    assert decoded.command_revision == 2
    assert decoded.plan_id == 3
    np.testing.assert_array_equal(decoded.frames, packet.frames)

    corrupted = bytearray(encoded)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="CRC32"):
        GmtTrajectoryPacket.decode(corrupted)

    invalid = packet.frames.copy()
    invalid[0, 3:7] = 0.0
    with pytest.raises(ValueError, match="quaternion"):
        GmtTrajectoryPacket.decode(
            GmtTrajectoryPacket(
                packet.stream_id,
                packet.sequence,
                packet.published_unix_ns,
                packet.command_revision,
                packet.plan_id,
                packet.fps,
                packet.flags,
                packet.joint_order_hash,
                invalid,
            ).encode()
        )


def test_resample_and_playback_have_real_context() -> None:
    source = make_qpos(31, fps=30.0)
    resampled = resample_qpos_timeline(source, 30.0, 50.0)
    assert resampled.shape == (51, BUMI_QPOS_DIM)
    np.testing.assert_allclose(resampled[-1], source[-1], atol=1e-5)
    idle = make_qpos(1)[0]
    playback = build_playback_timeline(
        resampled, idle, fps=50.0, blend_seconds=0.8, return_seconds=1.0
    )
    assert playback.audio_start_frame == 10 + 40
    assert playback.audio_end_frame == 10 + 40 + len(resampled)
    np.testing.assert_allclose(playback.qpos[:10], np.repeat(idle[None], 10, axis=0))
    np.testing.assert_allclose(playback.qpos[-100:], np.repeat(playback.qpos[-1:], 100, axis=0))
    assert not np.array_equal(playback.qpos[50], playback.qpos[51])


def test_incremental_plan_snapshots_match_complete_offline_plan() -> None:
    source = make_qpos(390, fps=30.0)
    idle = make_qpos(1)[0]
    idle[:2] = [2.0, -1.0]
    builder = BumiIncrementalGmtPlanBuilder(
        idle,
        np.arange(21),
        blend_seconds=0.8,
        return_seconds=1.0,
    )
    snapshots = []
    for start, end in ((0, 120), (120, 210), (210, 300), (300, 390)):
        snapshots.append(builder.append(source[start:end], is_last=end == len(source)))

    final = snapshots[-1]
    action = align_qpos_action(resample_qpos_timeline(source, 30.0, 50.0), idle)
    prefix = np.repeat(idle[None], 10, axis=0)
    blend = interpolate_qpos(idle, action[0], 40)
    target_idle = idle.copy()
    target_idle[:2] = action[-1, :2]
    returning = interpolate_qpos(action[-1], target_idle, 50)
    expected_qpos = np.concatenate(
        (
            prefix,
            blend,
            action,
            returning,
            np.repeat(target_idle[None], 101, axis=0),
        ),
        axis=0,
    )
    expected_frames = qpos_timeline_to_gmt_frames(
        expected_qpos, fps=50.0, native_to_gmt=np.arange(21)
    )
    np.testing.assert_allclose(final.qpos, expected_qpos, atol=1e-6)
    np.testing.assert_allclose(final.frames, expected_frames, atol=1e-5)
    assert final.audio_start_frame == 50
    assert final.audio_end_frame == 50 + len(action)
    assert final.action_complete
    assert all(not snapshot.qpos.flags.writeable for snapshot in snapshots)
    assert all(not snapshot.frames.flags.writeable for snapshot in snapshots)


def test_planar_alignment_moves_complete_action_without_collapsing_it() -> None:
    action = make_qpos(20)
    idle = make_qpos(1)[0]
    idle[:2] = [4.0, -3.0]
    aligned = align_qpos_action(action, idle)
    np.testing.assert_allclose(aligned[0, :2], idle[:2], atol=1e-6)
    np.testing.assert_allclose(np.diff(aligned[:, 0]), np.diff(action[:, 0]), atol=1e-6)
    assert np.unique(aligned[:, 0]).size == len(aligned)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}

    def set(self, key: str, value: bytes, *, px: int) -> None:
        self.values[key] = value
        self.ttls[key] = px

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)


def test_publisher_requires_matching_stream_revision_and_plan_ack() -> None:
    client = FakeRedis()
    publisher = RedisTrajectoryPublisher(client, stream_id=99)
    packet = publisher.publish(
        make_frames(),
        20,
        fps=50.0,
        joint_order_hash=order_hash(),
        command_revision=4,
        plan_id=5,
    )
    wrong_plan = GmtTrajectoryAck(99, packet.sequence, 4, 6, time.time_ns())
    client.values[publisher.ack_key] = wrong_plan.encode()
    assert publisher.matching_ack() is None
    correct = GmtTrajectoryAck(99, packet.sequence, 4, 5, time.time_ns())
    client.values[publisher.ack_key] = correct.encode()
    assert publisher.matching_ack() == correct


def test_packet_constant_contract_dimensions() -> None:
    packet = packet_for_cursor(
        make_frames(),
        10,
        fps=50.0,
        joint_order_hash=order_hash(),
        stream_id=1,
        sequence=1,
    )
    assert packet.frames.shape == (TRAJECTORY_FRAME_COUNT, TRAJECTORY_FRAME_DIM)
