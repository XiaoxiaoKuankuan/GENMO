"""BUMI 独立预览、原生姿态快照和配置兼容的隔离回归测试。

所有临时音频占位、运动学与配置写入 pytest 临时目录。通过禁止网络客户端验证本地
播放器独立运行；用可控单调时钟覆盖预生成、音频起点、自然结束、取消、迟到块、缓冲
不足和无效姿态。Bridge 测试复用已有假 Redis/策略，不启动任何真实控制器或硬件。
真实 GPU、图形窗口及模型资源另由部署验收执行，单元测试不冒充实物跟踪验证。
"""

import hashlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gem.robots.bumi.kinematics import BumiKinematics
from gem.runtime.bumi_deployment_config import deployment_command, load_deployment_config
from gem.runtime.bumi_local_player import LocalBumiPlayer
from gem.runtime.bumi_online_stream import BUMI_ONLINE_QPOS_STREAM_CONTRACT, BumiOnlineQposChunk
from gem.runtime.bumi_preview import (
    LocalPreviewReader,
    MujocoPreview,
    PoseSnapshot,
    validate_robot_assets,
)
from scripts.demo import demo_bumi_gmt_bridge as bridge_module
from tests.bumi.test_bumi_online_deployment import _bridge_args, _FakePolicy, _FakeRedis, _identity

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("bent", [False, True])
def test_standing_pose_grounds_feet_without_changing_model_asset(test_kinematics_path, bent):
    spec = json.loads(test_kinematics_path.read_text())
    # 使用俯仰关节，让屈膝姿态确实改变脚底高度，避免仅验证固定 z 偏移。
    spec["joint_axes"] = [[0, 1, 0] for _ in range(21)]
    test_kinematics_path.write_text(json.dumps(spec))
    original = test_kinematics_path.read_bytes()
    kinematics = BumiKinematics(test_kinematics_path)
    default = kinematics.default_qpos.numpy().copy()
    joints = np.full(21, 0.25 if bent else 0.0)
    pose = kinematics.make_standing_qpos(joints if bent else None)
    fk = kinematics.forward_kinematics(pose)
    sole = kinematics.get_sole_proxy_points(fk["body_pos_w"], fk["body_quat_w"])
    assert sole["bottom_height"].amin().item() == pytest.approx(0.002, abs=1e-6)
    np.testing.assert_array_equal(kinematics.default_qpos.numpy(), default)
    np.testing.assert_array_equal(pose.numpy()[[0, 1, 3, 4, 5, 6]], default[[0, 1, 3, 4, 5, 6]])
    np.testing.assert_allclose(pose.numpy()[7:], joints)
    assert test_kinematics_path.read_bytes() == original


def test_viewer_waits_for_its_render_thread_before_glfw_cleanup(monkeypatch):
    import glfw
    import mujoco.viewer

    from scripts.demo.bumi_mujoco_viewer import managed_viewer

    stop = threading.Event()
    destroyed = threading.Event()

    def launch(*a, **kw):
        def render():
            stop.wait()
            time.sleep(0.03)
            destroyed.set()

        threading.Thread(target=render, daemon=True).start()
        return SimpleNamespace(close=stop.set)

    def terminate():
        assert destroyed.is_set(), "GLFW 不能早于渲染线程结束"

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(glfw, "terminate", terminate)
    with managed_viewer(None, None):
        assert not destroyed.is_set()
    assert destroyed.is_set()


def test_render_resource_hash_missing_and_wrong_robot(test_kinematics_path, tmp_path):
    root = tmp_path / "assets"
    (root / "mjcf").mkdir(parents=True)
    (root / "meshes").mkdir()
    xml = root / "mjcf/robot.xml"
    mesh = root / "meshes/robot.stl"
    xml.write_text(
        '<mujoco><compiler meshdir="../meshes"/><asset><mesh file="robot.stl"/></asset></mujoco>'
    )
    mesh.write_bytes(b"fixture mesh, not loaded by MuJoCo")
    spec = json.loads(test_kinematics_path.read_text())
    spec["source_mjcf_sha256"] = hashlib.sha256(xml.read_bytes()).hexdigest()
    test_kinematics_path.write_text(json.dumps(spec))
    payload = dict(
        contract_version="genmo.bumi_viewer.v1",
        mjcf="mjcf/robot.xml",
        kinematics_sha256=hashlib.sha256(test_kinematics_path.read_bytes()).hexdigest(),
        files={
            p.relative_to(root).as_posix(): dict(
                size_bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
            )
            for p in (xml, mesh)
        },
    )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload))
    assert validate_robot_assets(manifest, test_kinematics_path)[0] == xml
    mesh.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="指纹"):
        validate_robot_assets(manifest, test_kinematics_path)
    mesh.unlink()
    with pytest.raises(FileNotFoundError):
        validate_robot_assets(manifest, test_kinematics_path)
    spec["default_qpos"][2] = 0.9
    test_kinematics_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="kinematics"):
        validate_robot_assets(manifest, test_kinematics_path)


def test_slow_viewer_drops_display_frames_without_blocking(tmp_path, monkeypatch):
    import gem.runtime.bumi_preview as module

    viewer = object.__new__(MujocoPreview)
    viewer.stop = threading.Event()
    viewer.kinematics_sha256 = "a" * 64
    viewer.sent_frames = viewer.dropped_frames = 0
    viewer.last_error = None

    class Source:
        calls = 0
        closed = False

        def request(self, payload):
            self.calls += 1
            if self.calls == 4:
                viewer.stop.set()
            return dict(
                ok=True,
                available=True,
                age_seconds=0,
                sequence=self.calls,
                kinematics_sha256="a" * 64,
                qpos=[0] * 28,
            )

        def close(self):
            self.closed = True

    source = Source()
    viewer.source_factory = lambda: source

    def full_pipe(*args):
        raise BlockingIOError()

    monkeypatch.setattr(module.os, "write", full_pipe)
    with (tmp_path / "pipe").open("wb") as stream:
        viewer.process = SimpleNamespace(stdin=stream, poll=lambda: None)
        started = time.monotonic()
        viewer._loop()
    assert time.monotonic() - started < 0.5
    assert viewer.dropped_frames == 4 and viewer.sent_frames == 0 and source.closed


def test_missing_display_reports_failure_before_spawning(monkeypatch):
    import gem.runtime.bumi_preview as module

    monkeypatch.setattr(module, "validate_robot_assets", lambda *args: None)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("不应启动窗口"))
    with pytest.raises(RuntimeError, match="DISPLAY"):
        MujocoPreview("unused", Path(__file__), lambda: None)


@pytest.fixture
def local(test_kinematics_path, tmp_path, monkeypatch):
    import xmlrpc.client

    import redis
    import zmq

    def forbidden(*a, **k):
        raise AssertionError("独立模式不得创建网络客户端")

    monkeypatch.setattr(redis, "Redis", forbidden)
    monkeypatch.setattr(zmq, "Context", forbidden)
    monkeypatch.setattr(xmlrpc.client, "ServerProxy", forbidden)
    player = LocalBumiPlayer(
        BumiKinematics(test_kinematics_path),
        test_kinematics_path,
        audio_mode="off",
        autostart=False,
    )
    identity = replace(
        _identity(), kinematics_sha256=player.kinematics_sha, joint_order_sha256=player.joint_sha
    )
    audio = tmp_path / "占位.wav"
    audio.write_bytes(b"test only")
    payload = dict(
        command="begin",
        contract_version=BUMI_ONLINE_QPOS_STREAM_CONTRACT,
        source_fps=30.0,
        total_frames=300,
        audio_duration_sec=10.0,
        audio_start_sec=0.0,
        request_id="one",
        revision=0,
        prime_chunks=2,
        identity=identity.as_dict(),
        audio_path=str(audio),
    )
    yield player, identity, payload
    player.close()


def make_chunk(player, identity, payload, start, size, index, *, bad=False):
    qpos = np.repeat(player.idle[None], size, axis=0)
    qpos[:, 7] = np.sin(np.arange(start, start + size) / 50.0) * 0.1
    if bad:
        qpos[:, 2] = 100
    return BumiOnlineQposChunk.from_qpos(
        qpos,
        request_id=payload["request_id"],
        revision=payload["revision"],
        chunk_index=index,
        absolute_start_frame=start,
        total_frames=payload["total_frames"],
        is_last=start + size == payload["total_frames"],
        identity=identity,
    )


def test_local_primes_and_audio_matches_playback_then_returns(local, monkeypatch):
    player, identity, begin = local
    starts = []
    monkeypatch.setattr(player.audio, "start", lambda *args: starts.append(args))
    player.request(begin)
    player.chunk(make_chunk(player, identity, begin, 0, 90, 0))
    assert player.state == "PRIMING"
    player.tick()
    assert not starts
    player.chunk(make_chunk(player, identity, begin, 90, 210, 1))
    assert player.state == "TRANSITION"
    epoch = player.started_at
    player.tick(epoch + 0.5)
    assert not starts
    player.tick(epoch + 1.1)
    assert len(starts) == 1 and player.state == "PLAYING"
    pose = player.pose.read(now=epoch + 1.1)
    np.testing.assert_allclose(pose["qpos"], player.snapshot.qpos[pose["frame_index"]])
    player.tick(epoch + 12.1)
    assert player.state == "STAND"
    assert player.request({"command": "status"})["gmt_acked"] is None
    assert player.active is None and player.snapshot is None


def test_short_final_chunk_bypasses_two_chunk_prime(local):
    player, identity, begin = local
    begin.update(total_frames=30, audio_duration_sec=1.0)
    player.request(begin)
    player.chunk(make_chunk(player, identity, begin, 0, 30, 0))
    assert player.state == "TRANSITION"


def test_local_cancel_rejects_late_revision_and_can_play_again(local):
    player, identity, begin = local
    player.request(begin)
    old = make_chunk(player, identity, begin, 0, 90, 0)
    player.chunk(old)
    player.request({"command": "stand"})
    player.tick(player.started_at + 1.2)
    begin.update(request_id="two", revision=player.tracker.revision + 1)
    player.request(begin)
    with pytest.raises(ValueError, match="stale"):
        player.chunk(old)
    assert player.active["request_id"] == "two"
    assert player.tracker.next_frame == 0


def test_local_underflow_and_bad_pose_return_stand(local):
    player, identity, begin = local
    player.request(begin)
    player.chunk(make_chunk(player, identity, begin, 0, 90, 0))
    player.chunk(make_chunk(player, identity, begin, 90, 90, 1))
    player.tick(player.started_at + 5.0)
    assert player.state == "RETURNING" and player.last_error
    player.tick(player.started_at + 1.2)
    begin.update(revision=player.tracker.revision + 1)
    player.request(begin)
    with pytest.raises(ValueError):
        player.chunk(make_chunk(player, identity, begin, 0, 90, 0, bad=True))
    assert player.state == "RETURNING" and player.active is None


def test_reader_close_never_stops_local_player(local):
    player, _, _ = local
    reader = LocalPreviewReader(player)
    reader.close()
    assert not player.stop_event.is_set()
    player.tick()
    assert reader.request({"command": "preview_frame"})["available"]


def test_generation_failure_stops_local_audio_and_returns_stand(local, monkeypatch):
    from gem.runtime.bumi_online_stream import ConsoleCommand
    from scripts.demo.demo_music_bumi_console import ResidentBumiConsole

    player, identity, begin = local
    player.request(begin)
    player.chunk(make_chunk(player, identity, begin, 0, 90, 0))
    console = object.__new__(ResidentBumiConsole)
    console.bridge = player
    console.timing_lock = threading.Lock()
    console.request_state_lock = threading.RLock()
    console.current_request_id = "one"
    console.current_revision = 0

    def bad_features(*args):
        raise RuntimeError("injected feature failure")

    monkeypatch.setattr(console, "_features", bad_features)
    console._generate(
        ConsoleCommand("play", audio_path=Path(begin["audio_path"]), duration_sec=10.0),
        "one",
        0,
        threading.Event(),
    )
    assert "injected feature failure" in console.last_error
    assert player.state == "RETURNING" and player.audio.process is None
    assert console.current_request_id is None
    player.tick(player.started_at + 1.2)
    assert player.state == "STAND"


def test_snapshot_is_copy_and_age_does_not_advance_cursor():
    pose = PoseSnapshot("a" * 64)
    assert not pose.read()["available"]
    q = np.zeros(28)
    q[3] = 1
    pose.record(q, frame_index=17, revision=2, request_id="abc", state="PLAYING", now=10.0)
    q[0] = 99
    first = pose.read(now=10.1)
    first["qpos"][0] = 5
    assert pose.read(now=10.5)["qpos"][0] == 0
    assert pose.read(now=10.5)["age_seconds"] == 0.5
    assert pose.read(now=11.0)["frame_index"] == 17
    with pytest.raises(ValueError):
        pose.record(np.zeros(28), frame_index=0, revision=0, request_id=None, state="STAND")


def test_config_legacy_and_explicit_preview(tmp_path):
    path = tmp_path / "deployment.ini"
    text = (ROOT / "deployment.ini").read_text()
    path.write_text(text.replace("mode = gmt", "mode = preview"))
    config = load_deployment_config(path)
    _, argv = deployment_command(config, "genmo")
    assert "--preview" in argv and argv[argv.index("--runtime-mode") + 1] == "preview"
    with pytest.raises(ValueError, match="独立"):
        deployment_command(config, "bridge")
    # 旧四段配置无新增字段，默认完全保留旧的无预览 GMT 路径。
    import configparser

    parser = configparser.ConfigParser()
    parser.read_string(text)
    parser.remove_section("runtime")
    parser.remove_section("preview")
    with path.open("w") as stream:
        parser.write(stream)
    old = load_deployment_config(path)
    assert old.runtime_mode == "gmt" and not old.preview_enabled
    assert "--preview" not in deployment_command(old, "genmo")[1]


@pytest.mark.parametrize("replacement", ["enabled = false", "mode = wrong"])
def test_invalid_preview_config_rejected(tmp_path, replacement):
    text = (ROOT / "deployment.ini").read_text().replace("mode = gmt", "mode = preview")
    text = text.replace(
        "enabled = true" if replacement.startswith("enabled") else "mode = preview", replacement
    )
    path = tmp_path / "deployment.ini"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_deployment_config(path)


def test_bridge_snapshot_is_published_frame_before_cursor_advance(
    test_kinematics_path, tmp_path, monkeypatch
):
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"mock")
    fake_policy = _FakePolicy(BumiKinematics(test_kinematics_path).joint_order)
    monkeypatch.setattr(bridge_module.redis, "Redis", lambda **kw: _FakeRedis())
    monkeypatch.setattr(
        bridge_module.GmtPolicyContract, "from_onnx", classmethod(lambda cls, p: fake_policy)
    )
    bridge = bridge_module.BumiOnlineBridge(
        _bridge_args(test_kinematics_path, policy, tmp_path / "estop")
    )
    published = []

    class Publisher:
        def publish(self, frames, cursor, **kwargs):
            published.append((frames[cursor].copy(), cursor))
            if len(published) >= 3:
                bridge.stop_event.set()

        def matching_ack(self):
            return None

    bridge.idle_publisher = Publisher()
    bridge._publish_loop()
    frame = bridge.handle_message([json.dumps({"command": "preview_frame"}).encode()])
    np.testing.assert_allclose(frame["qpos"], bridge.idle_qpos)
    assert frame["sequence"] == 3 and frame["frame_index"] == 0
    bridge.stop_event.clear()
    published.clear()
    from gem.runtime.bumi_gmt_plan import BumiIncrementalGmtPlanBuilder

    builder = BumiIncrementalGmtPlanBuilder(bridge.idle_qpos, np.arange(21))
    action = np.repeat(bridge.idle_qpos[None], 300, axis=0)
    action[:, 7] = np.linspace(0, 0.2, 300)
    snapshot = builder.append(action, is_last=True)
    bridge.plan_snapshot = snapshot
    bridge.publisher = Publisher()
    bridge.state = "PLAYING"
    bridge.cursor = 75
    bridge.submitted_monotonic = time.monotonic()
    bridge.last_ack_monotonic = time.monotonic()
    bridge._publish_loop()
    frame = bridge.preview_pose.read()
    assert frame["frame_index"] == published[-1][1]
    assert bridge.cursor > frame["frame_index"]
    np.testing.assert_allclose(frame["qpos"], snapshot.qpos[published[-1][1]])
