"""Wire, sequencing and buffering tests for robot_stream_v1."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from gem.gmr_udp_bridge import PACKET_BYTES
from gem.runtime.gmt_trajectory import (
    BUMI_QPOS_DIM,
    body_angular_velocity,
    resample_qpos_timeline,
)
from gem.runtime.robot_stream import (
    GMR_PROTOCOL_VERSION,
    GMR_RESPONSE_HEADER,
    GMR_RESPONSE_MAGIC,
    GMRMotionRetargetSession,
    IncrementalQposTimeline,
    RevisionTracker,
    RobotStreamChunk,
    decode_gmr_response,
    encode_gmr_request,
    has_complete_publish_context,
    heartbeat_expired,
    parse_console_line,
)


def make_chunk(
    *,
    revision: int = 3,
    index: int = 0,
    start: int = 0,
    frames: int = 2,
    total: int = 4,
    last: bool = False,
) -> RobotStreamChunk:
    return RobotStreamChunk(
        request_id="request-a",
        revision=revision,
        chunk_index=index,
        absolute_start_frame=start,
        frame_count=frames,
        total_frames=total,
        is_last=last,
        checkpoint_sha256="c" * 64,
        engine_sha256="e" * 64,
        payload=bytes([index + 1]) * (frames * PACKET_BYTES),
    )


def test_robot_stream_chunk_roundtrip_and_crc() -> None:
    original = make_chunk()
    decoded = RobotStreamChunk.from_multipart(original.multipart())
    assert decoded == original
    corrupted = original.multipart()
    corrupted[1] = corrupted[1][:-1] + bytes([corrupted[1][-1] ^ 1])
    with pytest.raises(ValueError, match="CRC32"):
        RobotStreamChunk.from_multipart(corrupted)


def test_revision_tracker_rejects_stale_duplicate_and_gap() -> None:
    tracker = RevisionTracker()
    tracker.begin("request-a", 3, 4)
    tracker.accept(make_chunk())
    with pytest.raises(ValueError, match="chunk index"):
        tracker.accept(make_chunk())
    tracker.accept(make_chunk(index=1, start=2, frames=2, total=4, last=True))
    with pytest.raises(ValueError, match="newer revision"):
        tracker.begin("request-b", 3, 1)


def test_gmr_binary_response_sequence_crc_and_qpos() -> None:
    request = encode_gmr_request(1, 9, b"SMP1" + bytes(PACKET_BYTES - 4))
    assert request[:4] == b"GMRQ"
    qpos = np.arange(BUMI_QPOS_DIM, dtype="<f4")
    payload = qpos.tobytes()
    header = GMR_RESPONSE_HEADER.pack(
        GMR_RESPONSE_MAGIC,
        GMR_PROTOCOL_VERSION,
        0,
        9,
        BUMI_QPOS_DIM,
        len(payload),
        zlib.crc32(payload) & 0xFFFF_FFFF,
        123,
    )
    sequence, decoded, elapsed = decode_gmr_response(header, payload)
    assert sequence == 9 and elapsed == 123
    np.testing.assert_array_equal(decoded, qpos)
    bad = bytearray(payload)
    bad[-1] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        decode_gmr_response(header, bytes(bad))


class _FakeGMRClient:
    def __init__(self, *, fail_reset: bool = False) -> None:
        self.calls: list[tuple[str, bytes, int | None]] = []
        self.fail_reset = fail_reset

    def reset(self, packet: bytes, *, iterations: int = 1000):
        self.calls.append(("reset", packet, iterations))
        if self.fail_reset:
            raise RuntimeError("reset failed")
        return np.full(BUMI_QPOS_DIM, -1.0, dtype=np.float32), 25_000

    def frame(self, packet: bytes):
        self.calls.append(("frame", packet, None))
        value = float(sum(call[0] == "frame" for call in self.calls))
        return np.full(BUMI_QPOS_DIM, value, dtype=np.float32), 800


def test_gmr_motion_session_warms_once_from_each_revision_first_frame() -> None:
    client = _FakeGMRClient()
    session = GMRMotionRetargetSession(warmup_iterations=1000)
    first = b"SMP1" + bytes(PACKET_BYTES - 4)
    second = b"SMP1" + bytes([1]) * (PACKET_BYTES - 4)

    qpos0, frame_us0, warm_us0 = session.retarget(client, first, revision=7)
    qpos1, frame_us1, warm_us1 = session.retarget(client, second, revision=7)
    qpos2, frame_us2, warm_us2 = session.retarget(client, second, revision=8)

    assert client.calls == [
        ("reset", first, 1000),
        ("frame", first, None),
        ("frame", second, None),
        ("reset", second, 1000),
        ("frame", second, None),
    ]
    assert warm_us0 == 25_000 and warm_us1 is None and warm_us2 == 25_000
    assert frame_us0 == frame_us1 == frame_us2 == 800
    np.testing.assert_array_equal(qpos0, np.full(BUMI_QPOS_DIM, 1.0, dtype=np.float32))
    np.testing.assert_array_equal(qpos1, np.full(BUMI_QPOS_DIM, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(qpos2, np.full(BUMI_QPOS_DIM, 3.0, dtype=np.float32))


def test_gmr_motion_session_invalidate_and_failed_reset_require_rewarm() -> None:
    packet = b"SMP1" + bytes(PACKET_BYTES - 4)
    client = _FakeGMRClient()
    session = GMRMotionRetargetSession()
    session.retarget(client, packet, revision=4)
    session.invalidate()
    session.retarget(client, packet, revision=4)
    assert [call[0] for call in client.calls] == ["reset", "frame", "reset", "frame"]

    failed = _FakeGMRClient(fail_reset=True)
    session.invalidate()
    with pytest.raises(RuntimeError, match="reset failed"):
        session.retarget(failed, packet, revision=4)
    assert session.warmed_revision is None


def test_gmr_motion_session_rejects_invalid_revision_and_packet() -> None:
    session = GMRMotionRetargetSession()
    client = _FakeGMRClient()
    with pytest.raises(ValueError, match="revision"):
        session.retarget(client, bytes(PACKET_BYTES), revision=-1)
    with pytest.raises(ValueError, match="exactly one SMP1"):
        session.retarget(client, b"short", revision=0)


def make_qpos(frames: int) -> np.ndarray:
    result = np.zeros((frames, BUMI_QPOS_DIM), dtype=np.float32)
    result[:, 3] = 1.0
    seconds = np.arange(frames, dtype=np.float32) / 30.0
    result[:, 0] = seconds
    result[:, 7:] = seconds[:, None]
    return result


def test_incremental_resampling_matches_complete_timeline() -> None:
    source = make_qpos(210)
    timeline = IncrementalQposTimeline()
    first = timeline.append(source[:120])
    second = timeline.append(source[120:])
    expected = resample_qpos_timeline(source, 30.0, 50.0)
    np.testing.assert_allclose(timeline.target(), expected, atol=1e-6)
    np.testing.assert_allclose(np.concatenate((first, second)), expected, atol=1e-6)


def test_quaternion_sign_slerp_and_pi_wrap_are_continuous() -> None:
    source = make_qpos(4)
    source[1::2, 3:7] *= -1.0
    resampled = resample_qpos_timeline(source, 30.0, 50.0)
    np.testing.assert_allclose(np.linalg.norm(resampled[:, 3:7], axis=1), 1.0, atol=1e-6)
    assert np.all(np.sum(resampled[1:, 3:7] * resampled[:-1, 3:7], axis=1) >= 0.0)

    angles = np.deg2rad(np.asarray([179.0, -179.0], dtype=np.float64))
    wrapped = np.zeros((2, 4), dtype=np.float32)
    wrapped[:, 0] = np.cos(angles / 2.0)
    wrapped[:, 3] = np.sin(angles / 2.0)
    angular_velocity = body_angular_velocity(wrapped, 50.0)
    assert abs(float(angular_velocity[0, 2])) < 2.0


def test_console_grammar_supports_path_duration_full_and_commands() -> None:
    play = parse_console_line('play "/tmp/a song.wav" 20 --start 3 --seed 9')
    assert play is not None
    assert play.audio_path.as_posix() == "/tmp/a song.wav"
    assert play.duration_sec == 20.0 and play.start_sec == 3.0 and play.seed == 9
    alias = parse_console_line('"/tmp/a song.wav" full')
    assert alias is not None and alias.full and alias.duration_sec is None
    path_only = parse_console_line('"/tmp/a song.wav"')
    assert path_only is not None and path_only.full and path_only.duration_sec is None
    play_only = parse_console_line('play "/tmp/a song.wav" --seed 7')
    assert play_only is not None and play_only.full and play_only.seed == 7
    assert parse_console_line("stand").name == "stand"
    assert parse_console_line("exit").name == "quit"
    with pytest.raises(ValueError, match="duration"):
        parse_console_line("/tmp/a.wav -1")


def test_heartbeat_timeout_boundary() -> None:
    assert not heartbeat_expired(10.0, 11.49, 1.5)
    assert heartbeat_expired(10.0, 11.51, 1.5)


def test_motion_cursor_requires_true_future_and_derivative_lookahead() -> None:
    # A 110-row packet at cursor 10 reads through row 109.  Row 110 is retained
    # only so row 109 has a real centered-difference future sample.
    assert has_complete_publish_context(111, 10)
    assert not has_complete_publish_context(110, 10)
    assert not has_complete_publish_context(111, 11)
