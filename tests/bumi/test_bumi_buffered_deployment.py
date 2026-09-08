"""整段生成与仿真步消费的回归测试。

单元测试使用伪模型检查生成完成前不发送动作。设置 GENMO_GMT_DEPLOYMENT_ROOT 时，
会在 pytest 临时目录编译实际 C++ 接收器并启动独立本机 Redis，逐帧比对 21×52 输入，
覆盖快/慢网络心跳、重复包、暂停、末帧、错误模式、CRC 和掉线保护。不访问生产键，
不启动 Gazebo 或实机；子进程在 fixture 结束时关闭，产物由任务清理临时目录。
便捷脚本检查仅解析 Shell 参数，核对 v5 s200000 的统一路径和后置覆盖，不加载真实模型。
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import redis
import torch
from scipy.spatial.transform import Rotation

from gem.robots.bumi.kinematics import BumiKinematics
from gem.runtime.bumi_online_stream import BumiOnlineIdentity, BumiOnlineQposChunk
from gem.runtime.gmt_buffered_trajectory import (
    BUFFERED_ACK_FORMAT,
    BUFFERED_ACK_MAGIC,
    BUFFERED_ACK_SIZE,
    RedisBufferedTrajectoryPublisher,
    encode_buffered_clip,
)
from gem.runtime.gmt_trajectory import (
    GmtTrajectoryPacket,
    RedisTrajectoryPublisher,
    packet_for_cursor,
)
from scripts.demo import demo_bumi_gmt_bridge as bridge_module
from scripts.demo import demo_music_bumi_console as console_module
from tests.bumi.test_bumi_online_deployment import _bridge_args, _FakePolicy, _FakeRedis, _identity

JOINT_HASH = hashlib.sha256("\n".join(f"joint_{i}" for i in range(21)).encode()).digest()


def _buffered_wrapper_args(name):
    """只解析固定 exec 行，拒绝执行模型加载、网络连接或控制操作。"""
    script = Path(__file__).resolve().parents[2] / "scripts/demo" / name
    tokens = shlex.split(script.read_text().split("\nexec ", 1)[1].replace("\\\n", " "))
    assert tokens[:2] == [".venv/bin/python", "-u"]
    assert tokens[-1] == "$@", "用户参数必须保留在默认参数之后"
    return tokens[2], tokens[3:-1]


def test_buffered_wrapper_console_v5_defaults():
    entry, argv = _buffered_wrapper_args("run_bumi_buffered_console.sh")
    assert entry == "scripts/demo/demo_music_bumi_buffered_console.py"
    args = console_module.build_parser().parse_args(argv)
    capture = Path(
        "inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_s200000_20260907"
    )
    assert args.checkpoint == capture / "s200000.ckpt"
    assert args.kinematics == capture / "assets/bumi_kinematics_robot_retargeter_fe934_v1.json"
    assert args.stats == capture / "assets/bumi_qpos30_stats_train_5set_pass_v2_mine_fe934_v2.json"
    assert args.onnx == Path(
        "outputs/onnx/bumi_music/rr_pass_v2_5set_v5_s200000_20260907/"
        "bumi_music_denoiser_v5_s200000_t120_qpos30_contact.onnx"
    )
    assert (args.backend, args.onnx_provider, args.ddim_steps) == ("onnx", "cuda", 20)
    assert args.guidance_scale == 2.5 and not args.no_foot_lock


def test_buffered_wrapper_bridge_matches_console():
    _, console_argv = _buffered_wrapper_args("run_bumi_buffered_console.sh")
    entry, bridge_argv = _buffered_wrapper_args("run_bumi_buffered_bridge.sh")
    assert entry == "scripts/demo/demo_bumi_gmt_buffered_bridge.py"
    console = console_module.build_parser().parse_args(console_argv)
    bridge = bridge_module.build_parser().parse_args(bridge_argv)
    assert bridge.kinematics == console.kinematics
    assert bridge.gmt_policy == Path(
        "/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs/"
        "src/legged_rl/rl_controller/rl_controllers/policy/bumi/model_135000_stage2.onnx"
    )


def test_buffered_wrapper_user_overrides_preserved():
    _, argv = _buffered_wrapper_args("run_bumi_buffered_console.sh")
    args = console_module.build_parser().parse_args(argv + ["--onnx-provider", "cpu"])
    assert args.onnx_provider == "cpu"
    _, argv = _buffered_wrapper_args("run_bumi_buffered_bridge.sh")
    args = bridge_module.build_parser().parse_args(argv + ["--kinematics", "override.json"])
    assert args.kinematics == Path("override.json")


def frames(count=57):
    values = np.zeros((count, 55), dtype=np.float32)
    values[:, 2] = 0.5
    yaw = np.linspace(0, 1, count)
    values[:, 3] = np.cos(yaw / 2)
    values[:, 6] = np.sin(yaw / 2)
    values[:, 7:] = np.arange(count)[:, None] / 100 + np.arange(48)[None, :] / 1000
    return values


def test_complete_clip_encoding_and_ack_isolation():
    client = _FakeRedis()
    pub = RedisBufferedTrajectoryPublisher(client, key="test", stream_id=123)
    values = frames()
    pub.publish(values, 0, fps=50, joint_order_hash=JOINT_HASH, command_revision=7, plan_id=7)
    assert client.get(pub.clip_key)[:8] == b"OMGBF001"
    assert client.get(pub.key)[:8] == b"OMGBS001"
    encoded = client.get(pub.clip_key)
    pub.publish(values, 20, fps=50, joint_order_hash=JOINT_HASH, command_revision=7, plan_id=7)
    assert client.get(pub.clip_key) is encoded
    assert pub.current_frame == 0
    client.set(
        pub.ack_key,
        struct.pack(
            BUFFERED_ACK_FORMAT,
            BUFFERED_ACK_MAGIC,
            1,
            BUFFERED_ACK_SIZE,
            123,
            1,
            7,
            7,
            100,
            11,
            len(values),
        ),
    )
    assert pub.matching_ack().sequence == 1
    assert pub.current_frame == 11
    with pytest.raises(ValueError, match="immutable"):
        pub.publish(
            values.copy(), 0, fps=50, joint_order_hash=JOINT_HASH, command_revision=7, plan_id=7
        )
    with pytest.raises(ValueError):
        GmtTrajectoryPacket.decode(client.get(pub.key))
    with pytest.raises(ValueError, match="quaternions"):
        encode_buffered_clip(
            np.zeros((3, 55)), joint_order_hash=JOINT_HASH, stream_id=1, revision=1, plan_id=1
        )


def test_console_sends_only_after_full_generation(tmp_path, monkeypatch):
    events = []

    class Generator:
        windows_generated = 0
        emitted_frames = 0
        pending_frames = 0

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, features, seed):
            for i, count in enumerate((90, 120)):
                self.windows_generated += 1
                start = self.emitted_frames
                self.emitted_frames += count
                events.append(f"generate_{i}")
                yield SimpleNamespace(
                    qpos=torch.zeros(count, 28), absolute_start_frame=start, is_last=i == 1
                )

    class Client:
        def request(self, payload):
            events.append(payload["command"])
            return {"ok": True}

        def chunk(self, chunk):
            events.append("send")
            assert events[:3] == ["begin", "generate_0", "generate_1"]
            assert chunk.is_last and chunk.chunk_index == 0 and len(chunk.qpos()) == 210
            return {"ok": True}

    monkeypatch.setattr(console_module, "BumiStreamingQposGenerator", Generator)
    runtime = console_module.ResidentBumiConsole.__new__(console_module.ResidentBumiConsole)
    runtime.args = SimpleNamespace(
        playback_mode="buffered",
        ddim_steps=20,
        guidance_scale=2.5,
        no_foot_lock=False,
        low_water_seconds=4,
        high_water_seconds=12,
    )
    runtime.device = torch.device("cpu")
    runtime.runner = runtime.endecoder = None
    runtime.identity = _identity()
    runtime.timing_lock = threading.Lock()
    runtime.request_state_lock = threading.RLock()
    runtime.current_revision = 7
    runtime.bridge = Client()
    runtime.last_error = None
    runtime._features = lambda *args: (torch.zeros(210, 35), {"selected_duration_sec": 7}, True)
    audio = tmp_path / "song.wav"
    audio.touch()
    command = SimpleNamespace(audio_path=audio, start_sec=0, full=True, duration_sec=None, seed=42)
    runtime._generate(command, "req", 7, threading.Event())
    assert runtime.last_error is None
    assert events == ["begin", "generate_0", "generate_1", "send"]
    assert runtime.last_timing["submitted_frames"] == 210


def test_bridge_cursor_waits_for_consumer_and_survives_pause(
    test_kinematics_path, tmp_path, monkeypatch
):
    kin = BumiKinematics(test_kinematics_path)
    client = _FakeRedis()
    monkeypatch.setattr(bridge_module.redis, "Redis", lambda **kwargs: client)
    monkeypatch.setattr(
        bridge_module.GmtPolicyContract,
        "from_onnx",
        classmethod(lambda cls, path: _FakePolicy(kin.joint_order)),
    )
    policy = tmp_path / "policy.onnx"
    policy.touch()
    audio = tmp_path / "audio.wav"
    audio.touch()
    args = _bridge_args(test_kinematics_path, policy, tmp_path / "estop")
    args.playback_mode = "buffered"
    bridge = bridge_module.BumiOnlineBridge(args)
    identity = BumiOnlineIdentity(
        **{
            **_identity().as_dict(),
            "kinematics_sha256": bridge.kinematics_sha256,
            "joint_order_sha256": bridge.joint_order_sha256,
        }
    )
    bridge.begin(
        {
            "contract_version": "bumi_online_qpos_stream_v1",
            "playback_mode": "buffered",
            "request_id": "req",
            "revision": 1,
            "audio_path": str(audio),
            "audio_duration_sec": 7,
            "total_frames": 210,
            "source_fps": 30,
            "identity": identity.as_dict(),
        }
    )
    qpos = np.repeat(kin.default_qpos.numpy()[None], 210, axis=0)
    partial = BumiOnlineQposChunk.from_qpos(
        qpos[:90],
        request_id="req",
        revision=1,
        chunk_index=0,
        absolute_start_frame=0,
        total_frames=210,
        is_last=False,
        identity=identity,
    )
    with pytest.raises(ValueError, match="one complete"):
        bridge.accept_chunk(partial)
    complete = BumiOnlineQposChunk.from_qpos(
        qpos,
        request_id="req",
        revision=1,
        chunk_index=0,
        absolute_start_frame=0,
        total_frames=210,
        is_last=True,
        identity=identity,
    )
    bridge.accept_chunk(complete)
    pub = bridge.publisher
    count = len(bridge.plan_snapshot.frames)
    thread = threading.Thread(target=bridge._publish_loop, daemon=True)
    thread.start()

    def wait_until(predicate):
        for _ in range(100):
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("bridge state did not advance")

    def ack(frame):
        client.set(
            pub.ack_key,
            struct.pack(
                BUFFERED_ACK_FORMAT,
                BUFFERED_ACK_MAGIC,
                1,
                BUFFERED_ACK_SIZE,
                pub.stream_id,
                pub.last_packet.sequence,
                1,
                1,
                time.time_ns(),
                frame,
                count,
            ),
        )

    try:
        wait_until(lambda: pub.last_packet is not None)
        ack(0)
        wait_until(lambda: bridge.acked)
        time.sleep(0.25)  # 超过旧 ACK stale 阈值，缓存模式允许仿真暂停。
        assert bridge.cursor == 0 and bridge.state == "TRANSITION"
        ack(20)
        wait_until(lambda: bridge.cursor == 20)
        assert bridge.state == "PLAYING"
        ack(count - 101)
        wait_until(lambda: bridge.state == "STAND")
    finally:
        bridge.stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.fixture(scope="module")
def cpp_driver(tmp_path_factory):
    deployment = os.environ.get("GENMO_GMT_DEPLOYMENT_ROOT")
    if not deployment:
        pytest.skip("设置 GENMO_GMT_DEPLOYMENT_ROOT 后编译实际部署接收器")
    out = tmp_path_factory.mktemp("buffered_cpp")
    binary = out / "receiver"
    include = Path(deployment) / "src/legged_rl/rl_controller/rl_controllers/include"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-DRL_CONTROLLERS_HAS_HIREDIS",
            "-I/usr/include/eigen3",
            f"-I{include}",
            str(Path(__file__).with_name("buffered_receiver_driver.cpp")),
            "-lhiredis",
            "-lz",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    return binary


@pytest.fixture
def isolated_redis(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [
            "redis-server",
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = redis.Redis(host="127.0.0.1", port=port)
    try:
        for _ in range(100):
            try:
                if client.ping():
                    break
            except redis.ConnectionError:
                time.sleep(0.01)
        else:
            raise RuntimeError("test Redis did not start")
        yield client, port
    finally:
        client.close()
        proc.terminate()
        proc.wait(timeout=5)


class Receiver:
    def __init__(self, binary, port, key, allow=True, names=None):
        self.proc = subprocess.Popen(
            [str(binary), str(port), key, str(int(allow))]
            + ([] if names is None else [str(names)]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        while self.proc.stdout.readline().strip() != "READY":
            if self.proc.poll() is not None:
                raise RuntimeError("receiver exited before READY")

    def step(self, command="tick"):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if line.startswith("RESULT "):
                parts = line.split()
                return bool(int(parts[1])), int(parts[2]), np.array(parts[3:], dtype=np.float32)
            if not line:
                raise RuntimeError("receiver exited")

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)
        self.proc.stdout.close()


def test_actual_cpp_policy_steps_and_52d_parity(cpp_driver, isolated_redis):
    client, port = isolated_redis
    values = frames()
    pub = RedisBufferedTrajectoryPublisher(client, key="buffered_test")
    receiver = Receiver(cpp_driver, port, pub.key)
    try:
        for index in range(len(values) + 3):
            # 心跳次数变化，消费端始终每 tick 一帧；同一心跳也可被多个 tick 消费。
            for _ in range(0 if index % 3 == 1 else 1 + index % 4):
                pub.publish(values, 0, fps=50, joint_order_hash=JOINT_HASH)
            fresh, frame, window = receiver.step()
            assert fresh and frame == min(index, len(values) - 1)
            rows = packet_for_cursor(
                values, frame, fps=50, joint_order_hash=JOINT_HASH, stream_id=1, sequence=1
            ).frames[:21]
            gravity = Rotation.from_quat(rows[:, [4, 5, 6, 3]]).inv().apply([0, 0, -1])
            expected = np.concatenate((rows[:, 2:3], gravity, rows[:, 7:]), axis=1)
            np.testing.assert_allclose(window.reshape(21, 52), expected, atol=2e-6, rtol=1e-6)
            assert pub.matching_ack() is not None and pub.current_frame == frame
            assert receiver.step("peek")[1] == frame
        client.delete(pub.key)
        time.sleep(0.23)
        assert not receiver.step("peek")[0]
    finally:
        receiver.close()


@pytest.mark.skipif(
    os.environ.get("GENMO_BUMI_REAL_GENERATION") != "1",
    reason="显式开启后才使用真实 CUDA s190000 和火力全开音乐",
)
def test_real_cuda_full_song_to_cpp_cache(cpp_driver, isolated_redis, tmp_path, monkeypatch):
    _, port = isolated_redis
    root = Path(__file__).resolve().parents[2]
    checkpoint = (
        root
        / "inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_s190000_20260904"
    )
    kin = checkpoint / "assets/bumi_kinematics_robot_retargeter_fe934_v1.json"
    stats = checkpoint / "assets/bumi_qpos30_stats_train_5set_pass_v2_mine_fe934_v2.json"
    onnx = (
        root
        / "outputs/onnx/bumi_music/rr_pass_v2_5set_s190000_20260904/bumi_music_denoiser_s190000_t120_qpos30_contact.onnx"
    )
    deployment = Path(os.environ["GENMO_GMT_DEPLOYMENT_ROOT"])
    policy = (
        deployment
        / "src/legged_rl/rl_controller/rl_controllers/policy/bumi/model_135000_stage2.onnx"
    )
    args = bridge_module.build_parser().parse_args(
        [
            "--kinematics",
            str(kin),
            "--gmt-policy",
            str(policy),
            "--redis-port",
            str(port),
            "--redis-key",
            "actual_song_test",
            "--audio-playback",
            "off",
            "--playback-mode",
            "buffered",
            "--estop-file",
            str(tmp_path / "estop"),
        ]
    )
    bridge = bridge_module.BumiOnlineBridge(args)

    class Client:
        def __init__(self, *args):
            pass

        def request(self, payload):
            return bridge.handle_message([json.dumps(payload).encode()])

        def chunk(self, chunk):
            assert chunk.is_last and chunk.absolute_start_frame == 0
            return bridge.accept_chunk(chunk)

        def close(self):
            pass

    monkeypatch.setattr(console_module, "BridgeClient", Client)
    console_args = console_module.build_parser().parse_args(
        [
            "--backend",
            "onnx",
            "--checkpoint",
            str(checkpoint / "s190000.ckpt"),
            "--onnx",
            str(onnx),
            "--kinematics",
            str(kin),
            "--stats",
            str(stats),
            "--onnx-provider",
            "cuda",
            "--ddim-steps",
            "20",
            "--playback-mode",
            "buffered",
        ]
    )
    console = console_module.ResidentBumiConsole(console_args)
    receiver = None
    try:
        console.initialize()
        audio = (
            root
            / "inputs/evals/bumi_s190000_mine10_20260904/mine_active/audio/dance_3__火力全开.wav"
        )
        command = SimpleNamespace(
            audio_path=audio, start_sec=0, full=True, duration_sec=None, seed=42
        )
        console._replace_request_state(None, 1)
        console._generate(command, "actual_song", 1, threading.Event())
        assert console.last_error is None
        assert bridge.plan_snapshot.action_complete and bridge.accepted_chunks == 1
        snapshot = bridge.plan_snapshot
        names = tmp_path / "joint_names.txt"
        names.write_text("\n".join(bridge.contract.joint_names))
        receiver = Receiver(cpp_driver, port, args.redis_key, names=names)
        maximum_error = 0.0
        for index in range(len(snapshot.frames)):
            bridge.publisher.publish(
                snapshot.frames,
                0,
                fps=50,
                joint_order_hash=bridge.contract.joint_order_hash,
                command_revision=1,
                plan_id=1,
            )
            fresh, actual_index, window = receiver.step()
            assert fresh and actual_index == index
            rows = snapshot.frames[
                np.clip(np.arange(index - 10, index + 11), 0, len(snapshot.frames) - 1)
            ]
            gravity = Rotation.from_quat(rows[:, [4, 5, 6, 3]]).inv().apply([0, 0, -1])
            expected = np.concatenate((rows[:, 2:3], gravity, rows[:, 7:]), axis=1)
            maximum_error = max(
                maximum_error, float(np.max(np.abs(window.reshape(21, 52) - expected)))
            )
            np.testing.assert_allclose(window.reshape(21, 52), expected, atol=2e-6, rtol=1e-6)
        print(
            f"REAL_CUDA_FULL_SONG source={bridge.tracker.next_frame} buffered={len(snapshot.frames)} "
            f"windows={console.last_timing['generated_windows']} max_52d_error={maximum_error:.9g}"
        )
    finally:
        if receiver is not None:
            receiver.close()
        console.close()
        console.heartbeat_thread.join(timeout=2)
        bridge.redis.close()


def test_cpp_rejects_buffered_without_opt_in_and_bad_crc(cpp_driver, isolated_redis):
    client, port = isolated_redis
    pub = RedisBufferedTrajectoryPublisher(client, key="bad_clip")
    pub.publish(frames(), 0, fps=50, joint_order_hash=JOINT_HASH)
    receiver = Receiver(cpp_driver, port, pub.key, allow=False)
    try:
        assert not receiver.step()[0]
        assert pub.matching_ack() is None
    finally:
        receiver.close()
    damaged = bytearray(client.get(pub.clip_key))
    damaged[-1] ^= 1
    client.set(pub.clip_key, damaged)
    receiver = Receiver(cpp_driver, port, pub.key)
    try:
        assert not receiver.step()[0]
        assert pub.matching_ack() is None
        idle = RedisTrajectoryPublisher(client, key=pub.key)
        idle.publish(frames(110), 0, fps=50, joint_order_hash=JOINT_HASH, flags=1)
        assert receiver.step()[0]
    finally:
        receiver.close()
