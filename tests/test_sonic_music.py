"""音乐部署的全局时间网格、连续旋转、音频 DAC 与真实会话错误处理测试。

适配测试使用本机 SONIC FK 和实际 G1 XML 限位，不运行生成模型或控制策略。
重点检查分窗提交与整段转换完全一致、尾窗帧数准确以及跨 ±π 的最短弧插值。
音频回调采用可控的 PortAudio 时间戳验证排队延迟不会算成已经播放的媒体时间。
"""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zmq

from gem.runtime.sonic_music import AudioClock, PoseTimeline, SessionClient

SONIC = Path("/home/weili/GR00T-WholeBodyControl")


@pytest.mark.parametrize("frames", [1, 30, 120, 121, 210, 211])
def test_chunking_matches_whole_timeline(frames):
    if not SONIC.exists():
        pytest.skip("需要本地 SONIC FK 资产")
    torch.set_num_threads(2)
    duration = frames / 30
    x = torch.linspace(0, 1, frames)
    pose = torch.zeros(frames, 63)
    pose[:, 0] = x * 0.1
    root = torch.zeros(frames, 3)
    root[:, 1] = x * 0.4
    whole = PoseTimeline(SONIC, duration)
    expected = whole.push(dict(body_pose=pose, global_orient=root), is_last=True)
    chunked = PoseTimeline(SONIC, duration)
    chunks = []
    for start in range(0, frames, 30):
        end = min(start + 30, frames)
        payload = chunked.push(
            dict(body_pose=pose[start:end], global_orient=root[start:end]), is_last=end == frames
        )
        if payload:
            chunks.append(payload)
    for key in expected:
        np.testing.assert_allclose(
            np.concatenate([p[key] for p in chunks]), expected[key], atol=1e-6
        )
    assert len(expected["frame_index"]) == 100 + int(np.ceil(duration * 50 - 1e-9))
    assert len(chunked.finish()["frame_index"]) == 60


def test_rotation_crosses_pi_on_short_arc():
    root = np.array([[0, np.deg2rad(179), 0], [0, np.deg2rad(-179), 0]])
    _, orient = PoseTimeline._interpolate(np.zeros((2, 63)), root, np.array([1 / 60]), 0)
    assert abs(abs(orient[0, 1]) - np.pi) < 1e-7


def test_dac_clock_excludes_queued_audio_and_updates_anchor():
    audio = AudioClock(np.ones((48000, 2)), 48000, "off")
    audio.output = "device"
    audio.arm(1_100_000_000)
    out = np.zeros((4800, 2), np.float32)
    timing = SimpleNamespace(currentTime=1, outputBufferDacTime=1.1)
    with patch("gem.runtime.sonic_music.time.monotonic_ns", return_value=1_000_000_000):
        audio._callback(out, 4800, timing, False)
    assert audio.next_sample == 4800
    assert audio.position(1_050_000_000) == 0
    assert audio.position(1_150_000_000) == pytest.approx(0.05)
    timing.outputBufferDacTime = 1.2005
    with patch("gem.runtime.sonic_music.time.monotonic_ns", return_value=1_000_000_000):
        audio._callback(out, 4800, timing, False)
    assert audio.position(1_250_500_000) == pytest.approx(0.15)


def test_audio_underflow_is_explicit_fault():
    audio = AudioClock(np.ones((48000, 2)), 48000, "off")
    audio.arm(1)
    audio._callback(
        np.zeros((10, 2), np.float32),
        10,
        SimpleNamespace(currentTime=0, outputBufferDacTime=0),
        True,
    )
    assert "underflow" in audio.error


def test_calibrated_clock_does_not_include_python_callback_delay():
    audio = AudioClock(np.ones((48000, 2)), 48000, "off")
    audio.output, audio.pa_to_monotonic_ns = "device", 0
    audio.arm(1_100_000_000)
    with patch("gem.runtime.sonic_music.time.monotonic_ns", return_value=1_150_000_000):
        audio._callback(
            np.zeros((4800, 2), np.float32),
            4800,
            SimpleNamespace(currentTime=1, outputBufferDacTime=1.1),
            False,
        )
    assert audio.position(1_150_000_000) == pytest.approx(0.05)


def test_request_retry_keeps_same_sequence_and_payload():
    context = zmq.Context()
    ready, requests = threading.Event(), []
    endpoint = []

    def server():
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        endpoint.append(f"tcp://127.0.0.1:{port}")
        ready.set()
        for i in range(2):
            requests.append(socket.recv_multipart())
            if i == 0:
                time.sleep(0.035)
            socket.send_json(dict(ok=True))
        socket.close()

    worker = threading.Thread(target=server, daemon=True)
    worker.start()
    assert ready.wait(2)
    client = SessionClient(endpoint[0], "retry", timeout_ms=25)
    try:
        assert client.call("append", b"binary") == dict(ok=True)
        worker.join(2)
        assert requests[0] == requests[1]
        assert client.seq == 1
    finally:
        client.close()
        context.term()


def test_resident_stance_and_heading_survive_successive_tracks():
    if not SONIC.exists():
        pytest.skip("需要本地 SONIC FK 资产")
    torch.set_num_threads(2)
    from scipy.spatial.transform import Rotation

    from gem.runtime.motion_streamer import synthetic_idle_motion

    idle = synthetic_idle_motion(arm_open_degrees=15).body_pose.numpy()[0]
    timeline = PoseTimeline(SONIC, 1, arm_open_degrees=15, initial_root=np.array([0, 2.8, 0]))
    pose = torch.from_numpy(np.repeat(idle[None], 30, axis=0))
    root = torch.zeros(30, 3)
    root[:, 1] = torch.linspace(3.1, 3.5, 30)
    first = timeline.push(dict(body_pose=pose, global_orient=root), is_last=True)
    np.testing.assert_allclose(first["smpl_pose"][0].reshape(63), idle, atol=1e-6)
    end = timeline.finish()
    np.testing.assert_allclose(end["smpl_pose"][-1].reshape(63), idle, atol=1e-6)
    expected = Rotation.from_rotvec([0, 3.2, 0]).as_matrix()
    np.testing.assert_allclose(
        Rotation.from_rotvec(timeline.standing_root).as_matrix(), expected, atol=1e-6
    )
    next_track = PoseTimeline(SONIC, 1, arm_open_degrees=15, initial_root=timeline.standing_root)
    following = next_track.push(dict(body_pose=pose, global_orient=root), is_last=True)
    # 四元数可能相差整体符号，比较旋转矩阵而非直接逐元素比较。
    for q in (end["body_quat"][-1], following["body_quat"][0]):
        assert np.isfinite(q).all()
    np.testing.assert_allclose(
        Rotation.from_quat(end["body_quat"][-1][[1, 2, 3, 0]]).as_matrix(),
        Rotation.from_quat(following["body_quat"][0][[1, 2, 3, 0]]).as_matrix(),
        atol=1e-6,
    )
    stopped = next_track.graceful_tail(120)
    np.testing.assert_allclose(stopped["smpl_pose"][-1].reshape(63), idle, atol=1e-6)
    assert len(stopped["frame_index"]) == 60
