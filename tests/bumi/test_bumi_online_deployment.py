"""BUMI 常驻滚动生成和独立在线协议的纯 CPU 回归测试。

测试使用临时运动学/统计文件和确定性伪去噪器，不启动 Redis、GMT、TensorRT、仿真或
实机。覆盖单窗、两窗、普通长序列及非 90 倍数尾窗，验证 120/30/90 在线提交无重复、
缺帧和丢尾；在线 raw qpos、contact、因果足锁结果与既有整首离线基准一致；同时覆盖
CRC、身份固定、revision/帧序、心跳、发布上下文和新入口的依赖隔离。
"""

from __future__ import annotations

import ast
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.robots.bumi.endecoder import BumiEndecoder
from gem.robots.bumi.feature_codec import (
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.postprocess import (
    BUMI_STREAMING_FOOT_LOCK_CONTRACT_VERSION,
    BumiStreamingFootLocker,
    lock_bumi_foot_contacts,
)
from gem.runtime.bumi_music_deploy import (
    BUMI_SLIDING_QPOS_CONTRACT_VERSION,
    BumiSlidingQposGenerator,
    BumiStreamingQposGenerator,
)
from gem.runtime.bumi_online_stream import (
    BumiOnlineIdentity,
    BumiOnlineQposChunk,
    BumiOnlineRevisionTracker,
    WatermarkGate,
    gmt_ack_failure,
    has_complete_publish_context,
    heartbeat_expired,
    motion_buffer_failure,
)
from gem.runtime.gmt_trajectory import resample_qpos_timeline
from gem.runtime.qpos_timeline import IncrementalQposTimeline
from scripts.demo import demo_bumi_gmt_bridge as bridge_module
from scripts.demo import demo_music_bumi_console as console_module


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


def _qpos(frames: int) -> torch.Tensor:
    value = torch.zeros(frames, 28)
    value[:, 0] = torch.arange(frames, dtype=torch.float32) / 30.0
    value[:, 2] = 1.0
    yaw = torch.linspace(0.0, 0.2, frames)
    value[:, 3] = torch.cos(yaw * 0.5)
    value[:, 6] = torch.sin(yaw * 0.5)
    value[:, 7] = torch.linspace(0.0, 0.2, frames)
    return value


class _FixedContactStep:
    def __init__(self, normalized: torch.Tensor) -> None:
        self.normalized = normalized

    def __call__(self, noisy, timestep, music, length, guidance):
        del timestep, music, length, guidance
        motion = self.normalized.to(noisy).unsqueeze(0).expand_as(noisy)
        contact = noisy.new_full((noisy.shape[0], noisy.shape[1], 2), 10.0)
        return motion, contact


@pytest.mark.parametrize("total_frames", [60, 120, 121, 210, 257, 300])
def test_online_120_30_90_matches_offline_and_flushes_tail(
    test_kinematics_path: Path,
    tmp_path: Path,
    total_frames: int,
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    endecoder = BumiEndecoder(
        test_kinematics_path,
        _stats(tmp_path / f"stats_{total_frames}.json", kinematics),
        enable_contact_targets=False,
    )
    normalized = endecoder.normalize(endecoder.codec.encode(_qpos(120)).physical_features)
    music = torch.zeros(total_frames, 35)
    offline = BumiSlidingQposGenerator(
        _FixedContactStep(normalized),
        endecoder,
        device="cpu",
        steps=2,
        apply_foot_lock=True,
    )
    assert offline.generator.noise_device == torch.device("cpu")
    offline_result = offline.generate(music, seed=71)
    online = BumiStreamingQposGenerator(
        _FixedContactStep(normalized),
        endecoder,
        device="cpu",
        steps=2,
        apply_foot_lock=True,
    )
    assert online.generator.noise_device == torch.device("cpu")
    online_chunks = tuple(online.generate(music, seed=71))
    assert online_chunks
    assert [chunk.absolute_start_frame for chunk in online_chunks] == list(
        np.cumsum([0, *[len(chunk.qpos) for chunk in online_chunks[:-1]]])
    )
    assert all(len(chunk.qpos) == 90 for chunk in online_chunks[:-1])
    assert online_chunks[-1].is_last
    assert sum(len(chunk.qpos) for chunk in online_chunks) == total_frames
    torch.testing.assert_close(
        torch.cat([chunk.qpos_raw for chunk in online_chunks]),
        offline_result.qpos_raw,
        atol=2.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        torch.cat([chunk.foot_contact_logits for chunk in online_chunks]),
        offline_result.foot_contact_logits,
    )
    torch.testing.assert_close(
        torch.cat([chunk.qpos for chunk in online_chunks]),
        offline_result.qpos,
        atol=2.0e-5,
        rtol=1.0e-5,
    )


def test_causal_foot_lock_is_identical_across_arbitrary_chunk_boundaries(
    test_kinematics_path: Path,
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    qpos = _qpos(137)
    logits = torch.full((137, 2), 10.0)
    logits[37:44, 0] = -10.0
    expected = lock_bumi_foot_contacts(qpos, logits, kinematics)
    locker = BumiStreamingFootLocker(kinematics)
    boundaries = (0, 17, 63, 92, 137)
    actual = [
        locker.process(qpos[start:end], logits[start:end])
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    torch.testing.assert_close(torch.cat([item.qpos for item in actual]), expected.qpos)
    torch.testing.assert_close(
        torch.cat([item.correction_xy for item in actual]), expected.correction_xy
    )
    torch.testing.assert_close(
        torch.cat([item.active_contact for item in actual]), expected.active_contact
    )


def _identity(suffix: str = "a") -> BumiOnlineIdentity:
    return BumiOnlineIdentity(
        inference_backend="tensorrt",
        checkpoint_sha256="1" * 64,
        onnx_sha256="2" * 64,
        inference_artifact_sha256="3" * 64,
        inference_manifest_sha256="4" * 64,
        stats_sha256="5" * 64,
        kinematics_sha256="6" * 64,
        joint_order_sha256=suffix * 64,
        representation_contract_version=BUMI_REPRESENTATION_CONTRACT_VERSION,
        sliding_contract_version=BUMI_SLIDING_QPOS_CONTRACT_VERSION,
        foot_lock_contract_version=BUMI_STREAMING_FOOT_LOCK_CONTRACT_VERSION,
    )


def _chunk(index: int, start: int, frames: int, total: int, *, identity=None):
    qpos = _qpos(frames).numpy()
    return BumiOnlineQposChunk.from_qpos(
        qpos,
        request_id="online-request",
        revision=9,
        chunk_index=index,
        absolute_start_frame=start,
        total_frames=total,
        is_last=start + frames == total,
        identity=_identity() if identity is None else identity,
    )


def test_online_protocol_crc_revision_frame_order_and_full_identity() -> None:
    first = _chunk(0, 0, 90, 121)
    decoded = BumiOnlineQposChunk.from_multipart(first.multipart())
    np.testing.assert_array_equal(decoded.qpos(), first.qpos())
    corrupted = first.multipart()
    corrupted[1] = corrupted[1][:-1] + bytes((corrupted[1][-1] ^ 1,))
    with pytest.raises(ValueError, match="CRC32"):
        BumiOnlineQposChunk.from_multipart(corrupted)
    tracker = BumiOnlineRevisionTracker()
    tracker.begin("online-request", 9, 121, _identity())
    tracker.accept(decoded)
    with pytest.raises(ValueError, match="identity"):
        tracker.accept(_chunk(1, 90, 31, 121, identity=_identity("b")))
    tracker.accept(_chunk(1, 90, 31, 121))
    assert tracker.complete and tracker.next_frame == 121
    with pytest.raises(ValueError, match="newer revision"):
        tracker.begin("next", 9, 1, _identity())


def test_incremental_30_to_50_matches_single_pass_for_90_frame_chunks() -> None:
    source = _qpos(257).numpy()
    timeline = IncrementalQposTimeline()
    pieces = [
        timeline.append(source[:90]),
        timeline.append(source[90:180]),
        timeline.append(source[180:]),
    ]
    actual = np.concatenate(pieces)
    expected = resample_qpos_timeline(source, 30.0, 50.0)
    np.testing.assert_allclose(actual, expected, atol=1.0e-6)


def test_online_timeout_and_future_context_boundaries() -> None:
    assert not heartbeat_expired(10.0, now=11.49, timeout=1.5)
    assert heartbeat_expired(10.0, now=11.51, timeout=1.5)
    assert has_complete_publish_context(111, 10)
    assert not has_complete_publish_context(110, 10)
    assert (
        motion_buffer_failure(
            num_frames=110,
            cursor=10,
            action_complete=False,
            critical_buffer_seconds=2.0,
        )
        == "motion buffer underrun"
    )
    assert (
        motion_buffer_failure(
            num_frames=116,
            cursor=10,
            action_complete=False,
            critical_buffer_seconds=2.2,
        )
        == "critical motion buffer"
    )
    assert (
        gmt_ack_failure(
            acked=False,
            submitted_monotonic=10.0,
            last_ack_monotonic=10.0,
            now=15.01,
            ack_timeout_seconds=5.0,
            ack_stale_seconds=1.0,
        )
        == "GMT ACK timeout"
    )
    assert (
        gmt_ack_failure(
            acked=True,
            submitted_monotonic=10.0,
            last_ack_monotonic=12.0,
            now=13.01,
            ack_timeout_seconds=5.0,
            ack_stale_seconds=1.0,
        )
        == "GMT ACK stale"
    )


def test_online_high_low_watermark_hysteresis() -> None:
    gate = WatermarkGate(low_seconds=4.0, high_seconds=12.0)
    assert not gate.should_pause(11.9)
    assert gate.should_pause(12.0)
    assert gate.should_pause(8.0)
    assert gate.should_pause(4.0)
    assert not gate.should_pause(3.99)
    assert not gate.should_pause(8.0)


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def ping(self) -> bool:
        return True

    def set(self, key: str, value: bytes, **kwargs) -> bool:
        del kwargs
        self.values[key] = value
        return True

    def get(self, key: str):
        return self.values.get(key)


class _FakePolicy:
    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.joint_names = joint_names
        self.default_joint_pos = np.zeros(21, dtype=np.float32)
        self.joint_order_hash = b"p" * 32

    def native_to_gmt_indices(self, native_joint_names) -> np.ndarray:
        assert tuple(native_joint_names) == self.joint_names
        return np.arange(21, dtype=np.int64)

    def default_in_native_order(self, native_joint_names) -> np.ndarray:
        assert tuple(native_joint_names) == self.joint_names
        return np.zeros(21, dtype=np.float32)


def _bridge_args(kinematics: Path, policy: Path, estop: Path) -> SimpleNamespace:
    return SimpleNamespace(
        bind="tcp://127.0.0.1:0",
        kinematics=kinematics,
        gmt_policy=policy,
        redis_host="127.0.0.1",
        redis_port=6379,
        redis_db=0,
        redis_key="test_online_bumi",
        redis_ttl_ms=250,
        audio_playback="off",
        blend_seconds=0.02,
        return_seconds=0.02,
        ack_timeout_seconds=0.2,
        ack_stale_seconds=0.1,
        heartbeat_timeout_seconds=1.5,
        critical_buffer_seconds=0.05,
        joint_limit_tolerance_rad=0.05,
        max_joint_velocity_radps=18.0,
        max_root_linear_velocity_mps=4.0,
        max_root_angular_velocity_radps=8.0,
        min_root_height_m=0.25,
        max_root_height_m=1.20,
        estop_file=estop,
        verbose=False,
    )


def test_new_bumi_bridge_defaults_use_operator_selected_motion_thresholds() -> None:
    """锁定用户逐次指定的新BUMI在线桥运动安全阈值。"""

    args = bridge_module.build_parser().parse_args(
        ["--kinematics", "kinematics.json", "--gmt-policy", "policy.onnx"]
    )
    assert args.joint_limit_tolerance_rad == pytest.approx(0.576)
    assert args.max_joint_velocity_radps == pytest.approx(25.92)
    assert args.max_root_linear_velocity_mps == pytest.approx(5.76)
    assert args.max_root_angular_velocity_radps == pytest.approx(14.4)
    assert args.min_root_height_m == pytest.approx(0.25 / 1.2)
    assert args.max_root_height_m == pytest.approx(1.44)


def test_bridge_primes_two_chunks_and_stand_invalidates_late_revision(
    test_kinematics_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"test-policy")
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"test-audio")
    fake_policy = _FakePolicy(kinematics.joint_order)
    fake_redis = _FakeRedis()
    monkeypatch.setattr(bridge_module.redis, "Redis", lambda **kwargs: fake_redis)
    monkeypatch.setattr(
        bridge_module.GmtPolicyContract,
        "from_onnx",
        classmethod(lambda cls, path: fake_policy),
    )
    bridge = bridge_module.BumiOnlineBridge(
        _bridge_args(test_kinematics_path, policy_path, tmp_path / "estop")
    )
    identity = _identity()
    identity = BumiOnlineIdentity(
        **{
            **identity.as_dict(),
            "kinematics_sha256": bridge.kinematics_sha256,
            "joint_order_sha256": bridge.joint_order_sha256,
        }
    )
    begin_payload = {
        "contract_version": "bumi_online_qpos_stream_v1",
        "request_id": "online-request",
        "revision": 9,
        "audio_path": str(audio_path),
        "audio_start_sec": 0.0,
        "audio_duration_sec": 7.0,
        "total_frames": 210,
        "source_fps": 30.0,
        "prime_chunks": 2,
        "identity": identity.as_dict(),
    }
    bridge.args.estop_file.write_text("active", encoding="utf-8")
    with pytest.raises(RuntimeError, match="emergency stop"):
        bridge.begin(begin_payload)
    bridge.args.estop_file.unlink()
    begin = bridge.begin(begin_payload)
    assert begin["state"] == "PREPARING"
    assert begin["joint_limit_tolerance_rad"] == pytest.approx(0.05)
    assert begin["max_joint_velocity_radps"] == pytest.approx(18.0)
    assert begin["max_root_linear_velocity_mps"] == pytest.approx(4.0)
    assert begin["max_root_angular_velocity_radps"] == pytest.approx(8.0)
    assert begin["min_root_height_m"] == pytest.approx(0.25)
    assert begin["max_root_height_m"] == pytest.approx(1.20)
    qpos = _qpos(210).numpy()
    first = BumiOnlineQposChunk.from_qpos(
        qpos[:90],
        request_id="online-request",
        revision=9,
        chunk_index=0,
        absolute_start_frame=0,
        total_frames=210,
        is_last=False,
        identity=identity,
    )
    second = BumiOnlineQposChunk.from_qpos(
        qpos[90:],
        request_id="online-request",
        revision=9,
        chunk_index=1,
        absolute_start_frame=90,
        total_frames=210,
        is_last=True,
        identity=identity,
    )
    bridge.last_heartbeat = 0.0
    accepted_at = time.monotonic()
    assert bridge.accept_chunk(first)["state"] == "PRIMING"
    assert bridge.last_heartbeat >= accepted_at
    # 第一块已经形成 plan_snapshot，但两块预生成尚未完成，动作 publisher 仍为
    # None。发布线程必须只发固定站姿，不能因为 ``None is None`` 而误跑 ACK 超时。
    publish_thread = threading.Thread(target=bridge._publish_loop, daemon=True)
    publish_thread.start()
    try:
        time.sleep(bridge.args.ack_timeout_seconds + 0.1)
        with bridge.lock:
            assert bridge.state == "PRIMING"
            assert bridge.request is not None
            assert bridge.publisher is None
            assert bridge.last_stand_reason is None
    finally:
        bridge.stop_event.set()
        publish_thread.join(timeout=1.0)
        assert not publish_thread.is_alive()
    ready = bridge.accept_chunk(second)
    assert ready["state"] == "WAIT_ACK"
    assert ready["action_complete"] and ready["accepted_source_frames"] == 210
    bridge.request_stand("test cancel")
    assert bridge.state == "STAND_WAIT_ACK" and bridge.tracker.revision == 10
    assert bridge.status_locked()["last_stand_reason"] == "test cancel"
    with pytest.raises(ValueError, match="no active request"):
        bridge.accept_chunk(second)
    bridge.last_error = "ValueError: no active request accepts qpos chunks"
    bridge.request_stand("operator stand")
    status = bridge.status_locked()
    assert status["revision"] == 10
    assert status["last_stand_reason"] == "test cancel"
    assert status["last_error"] == "ValueError: no active request accepts qpos chunks"


def test_console_stale_heartbeat_response_cannot_clear_new_request() -> None:
    console = object.__new__(console_module.ResidentBumiConsole)
    console.request_state_lock = threading.RLock()
    console.current_request_id = "old-request"
    console.current_revision = 19
    console.last_error = None
    console.args = SimpleNamespace(heartbeat_seconds=0.001)

    class _StopAfterOneHeartbeat:
        calls = 0

        def wait(self, timeout: float) -> bool:
            assert timeout == 0.001
            self.calls += 1
            return self.calls > 1

    class _DelayedOldHeartbeatBridge:
        def request(self, payload: dict[str, object]) -> dict[str, object]:
            assert payload == {
                "command": "heartbeat",
                "request_id": "old-request",
                "revision": 19,
            }
            # 模拟旧心跳在请求期间阻塞，而交互线程已经建立 revision=20 的新任务。
            console._replace_request_state("new-request", 20)
            return {"ok": True, "state": "STAND"}

    console.stop = _StopAfterOneHeartbeat()
    console.bridge = _DelayedOldHeartbeatBridge()
    console._heartbeat_loop()
    assert console._request_state() == ("new-request", 20)
    assert console._clear_request_if_matches("new-request", 20)
    assert console._request_state() == (None, 20)


def test_new_online_imports_are_isolated_from_legacy_human_chain() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "gem/runtime/bumi_online_stream.py",
        root / "gem/runtime/qpos_timeline.py",
        root / "gem/runtime/bumi_gmt_plan.py",
        root / "scripts/demo/demo_music_bumi_console.py",
        root / "scripts/demo/demo_bumi_gmt_bridge.py",
    )
    forbidden = {
        "gem.runtime.robot_stream",
        "gem.gmr_udp_bridge",
        "gem.smplx_gmr_reference",
        "scripts.demo.stream_smpl_params_to_gmr",
    }
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not (imports & forbidden), f"{path} imports {imports & forbidden}"
