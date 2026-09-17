#!/usr/bin/env python3
"""BUMI MuJoCo 预览的真实 GPU/桌面及隔离 Redis 验收入口。

显式提供音乐、部署清单和报告路径，依次验证两段十秒的常驻生成、音频、换歌与窗口。
独立模式期间禁止创建 ZMQ/Redis/ROS 客户端。可选提供现有 GMT policy，仅在临时 Redis
端口和专用 key 上检查 Bridge 已发布姿态与解码包一致、ACK 中断与恢复，不启动控制器。
临时 Redis、音频、窗口和测试目录使用 finally 回收；正式报告由调用者指定路径保存。
本入口不证明仿真/实物跟踪，也不改变已有 GMT 服务的配置或通信 key。
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gem.runtime.bumi_online_stream import BumiOnlineQposChunk, parse_console_line  # noqa: E402
from scripts.demo.demo_music_bumi_console import (  # noqa: E402
    BridgeClient,
    ResidentBumiConsole,
    build_parser,
)


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def isolated_bridge(kinematics, policy, identity, qpos):
    import redis

    from gem.runtime.bumi_preview import MujocoPreview
    from gem.runtime.gmt_trajectory import GmtTrajectoryAck, GmtTrajectoryPacket
    from scripts.demo.demo_bumi_gmt_bridge import BumiOnlineBridge
    from scripts.demo.demo_bumi_gmt_bridge import build_parser as bridge_parser

    with tempfile.TemporaryDirectory(prefix="bumi-preview-redis-") as tmp:
        port, control_port = unused_port(), unused_port()
        while control_port == port:
            control_port = unused_port()
        server = subprocess.Popen(
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
                tmp,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        client = redis.Redis(host="127.0.0.1", port=port, socket_timeout=0.2)
        bridge = viewer = request = None
        publish_thread = receiver = None
        stopped = threading.Event()
        ack_enabled = threading.Event()
        ack_enabled.set()
        packets = {}
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    if client.ping():
                        break
                except redis.RedisError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("临时 Redis 启动失败")
                time.sleep(0.05)
            args = bridge_parser().parse_args(
                [
                    "--kinematics",
                    str(kinematics),
                    "--gmt-policy",
                    str(policy),
                    "--redis-port",
                    str(port),
                    "--redis-key",
                    "genmo_preview_acceptance",
                    "--bind",
                    f"tcp://127.0.0.1:{control_port}",
                    "--audio-playback",
                    "off",
                    "--estop-file",
                    str(Path(tmp) / "estop"),
                ]
            )
            bridge = BumiOnlineBridge(args)

            def consume():
                while not stopped.wait(0.003):
                    raw = client.get(args.redis_key)
                    if raw is None:
                        continue
                    pkt = GmtTrajectoryPacket.decode(raw)
                    packets[(pkt.stream_id, pkt.sequence)] = pkt
                    if ack_enabled.is_set():
                        client.set(
                            args.redis_key + "_ack",
                            GmtTrajectoryAck(
                                pkt.stream_id,
                                pkt.sequence,
                                pkt.command_revision,
                                pkt.plan_id,
                                time.time_ns(),
                            ).encode(),
                            px=1000,
                        )

            receiver = threading.Thread(target=consume, daemon=True)
            receiver.start()
            publish_thread = threading.Thread(target=bridge.start, daemon=True)
            publish_thread.start()
            endpoint = f"tcp://127.0.0.1:{control_port}"
            request = BridgeClient(endpoint, 1000)
            request.request({"command": "status"})
            viewer = MujocoPreview(
                ROOT / "assets/bumi_viewer/manifest.json",
                kinematics,
                lambda: BridgeClient(endpoint, 100),
            )
            audio = Path(tmp) / "unused.wav"
            audio.write_bytes(b"unused; audio off")
            begin = dict(
                command="begin",
                contract_version="bumi_online_qpos_stream_v1",
                request_id="isolated",
                revision=9,
                source_fps=30.0,
                total_frames=len(qpos),
                audio_duration_sec=len(qpos) / 30.0,
                audio_start_sec=0.0,
                audio_path=str(audio),
                prime_chunks=2,
                identity=identity.as_dict(),
            )
            assert request.request(begin)["ok"]
            for i, (start, end) in enumerate(((0, 90), (90, len(qpos)))):
                chunk = BumiOnlineQposChunk.from_qpos(
                    qpos[start:end],
                    request_id="isolated",
                    revision=9,
                    chunk_index=i,
                    absolute_start_frame=start,
                    total_frames=len(qpos),
                    is_last=end == len(qpos),
                    identity=identity,
                )
                assert request.chunk(chunk)["ok"]
            comparisons = 0
            max_error = 0.0
            states = set()
            saw_ack = False
            ack_cut = False
            restored = False
            started = time.monotonic()
            while time.monotonic() - started < 7.0:
                now = time.monotonic()
                status = request.request(
                    {"command": "heartbeat", "request_id": "isolated", "revision": 9}
                )
                saw_ack |= status["gmt_acked"]
                states.add(status["state"])
                if saw_ack and now - started > 2.0 and not ack_cut:
                    ack_enabled.clear()
                    ack_cut = True
                if status["last_stand_reason"] == "GMT ACK stale" and not restored:
                    ack_enabled.set()
                    restored = True
                frame = request.request({"command": "preview_frame"})
                pkt = packets.get((frame.get("stream_id"), frame.get("packet_sequence")))
                if frame.get("available") and pkt is not None:
                    reference = pkt.frames[10]
                    native = np.asarray(frame["qpos"])
                    err = max(
                        np.max(np.abs(native[:3] - reference[:3])),
                        min(
                            np.max(np.abs(native[3:7] - reference[3:7])),
                            np.max(np.abs(native[3:7] + reference[3:7])),
                        ),
                        np.max(np.abs(native[7:][bridge.native_to_gmt] - reference[13:34])),
                    )
                    max_error = max(max_error, float(err))
                    comparisons += 1
                time.sleep(0.01)
            assert (
                saw_ack and restored and "STAND" in states and comparisons > 30 and max_error < 1e-6
            )
            assert viewer.status()["alive"]
            return dict(
                pass_result=True,
                redis_port=port,
                control_port=control_port,
                comparisons=comparisons,
                max_qpos_error=max_error,
                states=sorted(states),
                ack_loss_returned_stand=restored,
                viewer=viewer.status(),
                publish_hz=bridge.status_locked()["publish_hz"],
            )
        finally:
            if viewer is not None:
                viewer.close()
            if request is not None:
                request.close()
            if bridge is not None:
                bridge.stop_event.set()
                bridge.audio.stop("test cleanup")
            if publish_thread is not None:
                publish_thread.join(timeout=2.0)
            if bridge is not None:
                bridge.control_thread.join(timeout=2.0)
                bridge.redis.close()
            stopped.set()
            if receiver is not None:
                receiver.join(timeout=1.0)
            client.close()
            server.terminate()
            server.wait(timeout=3.0)
            if viewer is not None:
                assert viewer.process.returncode == 0, viewer.status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs=2, required=True)
    parser.add_argument("--gmt-policy", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    console_args = build_parser().parse_args(
        [
            "--deployment-manifest",
            str(args.deployment_manifest),
            "--runtime-mode",
            "preview",
            "--preview",
        ]
    )
    results = []
    captured = []
    console = None

    def forbidden(*a, **k):
        raise AssertionError("独立预览创建了网络客户端")

    try:
        with ExitStack() as stack:
            for target in (
                "redis.Redis",
                "zmq.Context",
                "xmlrpc.client.ServerProxy",
                "scripts.demo.demo_music_bumi_console.BridgeClient",
            ):
                stack.enter_context(patch(target, side_effect=forbidden))
            console = ResidentBumiConsole(console_args)
            console.initialize()
            original_chunk = console.bridge.chunk

            def capture(chunk):
                captured.append(chunk.qpos())
                return original_chunk(chunk)

            console.bridge.chunk = capture
            for audio in args.audio:
                console.start_play(parse_console_line(f'play "{audio}" 10'))
                start = time.monotonic()
                audio_seen = False
                states = set()
                poses = 0
                while time.monotonic() - start < 90.0:
                    status = console.status()
                    states.add(status["bridge"]["state"])
                    process = console.bridge.audio.process
                    audio_seen |= process is not None and process.poll() is None
                    pose = console.bridge.pose.read()
                    if pose["available"]:
                        assert np.isfinite(pose["qpos"]).all()
                        poses += 1
                    if console.last_error:
                        raise RuntimeError(console.last_error)
                    if not status["generation_active"] and status["bridge"]["state"] == "STAND":
                        break
                    time.sleep(0.04)
                assert not status["generation_active"] and status["bridge"]["state"] == "STAND", (
                    status
                )
                assert status["last_timing"]["submitted_frames"] == 300 and audio_seen
                assert status["preview"]["alive"] and "PLAYING" in states
                results.append(
                    dict(
                        audio=str(audio),
                        states=sorted(states),
                        audio_process_observed=audio_seen,
                        finite_pose_samples=poses,
                        timing=status["last_timing"],
                        viewer=status["preview"],
                    )
                )
                print("[验收] 完成十秒音乐:", audio, flush=True)
            # 新播放中关闭窗口，检查本地时钟与生成仍可继续，再验证 stand 返回。
            console.start_play(parse_console_line(f'play "{args.audio[0]}" 10'))
            deadline = time.monotonic() + 30
            while console.bridge.state != "PLAYING":
                if console.last_error or time.monotonic() > deadline:
                    raise RuntimeError(console.last_error or "cancel test timeout")
                time.sleep(0.05)
            console.preview.close()
            assert console.preview.process.returncode == 0, console.preview.status()
            before = console.bridge.cursor
            time.sleep(0.15)
            assert console.bridge.cursor > before and not console.bridge.stop_event.is_set()
            console.stand()
            console._wait_for_stand()
            window_close_kept_playing = True
            identity = console.identity
            kinematics = console.kinematics_path
            complete = np.concatenate(captured[:3])[:300]
        console.close()
        console = None
        communication = (
            isolated_bridge(kinematics, args.gmt_policy, identity, complete)
            if args.gmt_policy
            else None
        )
        report = dict(
            pass_result=True,
            standalone_network_clients_forbidden=True,
            songs=results,
            window_close_kept_playing=window_close_kept_playing,
            isolated_bridge=communication,
            scope="real GENMO/MuJoCo GUI/audio and isolated test receiver; no simulation/hardware",
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        if console is not None:
            console.close()


if __name__ == "__main__":
    main()
