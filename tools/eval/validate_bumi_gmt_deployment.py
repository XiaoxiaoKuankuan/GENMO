#!/usr/bin/env python3
"""使用真实 GMT C++ 接收代码验收独立 BUMI 部署目录，严格隔离生产控制链路。

脚本在系统临时目录编译指定 GMT 工作区的协议单测及接收探针，启动随机非 6379 端口
的临时 Redis 和随机端口的 GENMO Bridge，再从指定部署根目录导入常驻 Console，以
部署清单加载真实模型生成一段音乐动作。探针只解包/构造窗口/回 ACK，不启动 ROS、
Gazebo 或实机。结束时停止本脚本创建的进程并清理全部临时编译文件、缓存和日志。

报告记录实际导入来源、模型身份、采样耗时、Bridge 状态、1092 维窗口和过期检测。
该报告证明推理与通信，不代表动力学跟踪或真实机器人控制验收。--output 是用户要求
保留的验收结论；所有其余运行产物使用 TemporaryDirectory 自动清理。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from xmlrpc.server import SimpleXMLRPCServer

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_process(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--gmt-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    deployment_root = args.deployment_root.resolve(strict=True)
    manifest_path = args.deployment_manifest.resolve(strict=True)
    gmt_root = args.gmt_root.resolve(strict=True)
    audio = args.audio.resolve(strict=True)
    if args.seconds <= 0 or args.timeout <= 0:
        parser.error("seconds and timeout must be positive")
    # 子进程及本进程都从真正的部署目录导入，不能借原仓库补齐被裁掉的模块。
    sys.path.insert(0, str(deployment_root))
    import redis
    import torch

    from gem.runtime.bumi_deployment_bundle import load_bumi_deployment_manifest
    from gem.runtime.bumi_online_stream import parse_console_line
    from gem.runtime.gmt_trajectory import GmtPolicyContract
    from scripts.demo import demo_music_bumi_console as console_module

    torch.set_num_threads(4)
    if not Path(console_module.__file__).resolve().is_relative_to(deployment_root):
        raise RuntimeError("Console import did not originate in deployment root")
    bundle = load_bumi_deployment_manifest(manifest_path)
    controller = gmt_root / "src/legged_rl/rl_controller/rl_controllers"
    # 只为隔离验收构造只读参数夹具：取当前obs的BUMI分支，绝不启动ROS/Gazebo。
    # 正常Bridge读取真实ROS master；此夹具不能被表述为真实ROS控制器已启动。
    launch_xml = ET.parse(controller / "launch/load_ac_controller.launch").getroot()
    values = [
        param.attrib["value"]
        for group in launch_xml.findall("group")
        if group.get("if") == "$(eval robot_type=='bumi')"
        for param in group.findall("param")
        if param.get("name") == "gmtPolicyFile"
    ]
    if len(values) != 1:
        raise ValueError("isolated obs fixture requires one BUMI gmtPolicyFile in its launch")
    policy_path = Path(values[0].replace("$(find rl_controllers)", str(controller)))
    if "$(" in str(policy_path):
        raise ValueError("isolated obs fixture cannot resolve this launch substitution")
    policy = GmtPolicyContract.from_onnx(policy_path)
    report = {
        "pass": False,
        "deployment_root": str(deployment_root),
        "console_source": console_module.__file__,
        "gmt_root": str(gmt_root),
        "audio": str(audio),
        "seconds": args.seconds,
        "source_checkpoint_sha256": bundle.source_checkpoint_sha256,
        "policy_discovery": "isolated read-only XML-RPC fixture from obs launch; no real ROS started",
        "gmt_policy_path": str(policy_path),
    }
    processes = []
    console = None
    log_streams = []
    error = None
    ros_server = None
    ros_thread = None
    with tempfile.TemporaryDirectory(prefix="genmo-gmt-deployment-") as temporary:
        tmp = Path(temporary)
        os.environ["NUMBA_CACHE_DIR"] = str(tmp / "numba")
        os.environ["MPLCONFIGDIR"] = str(tmp / "matplotlib")
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            TMPDIR=temporary,
            PYTHONPATH=str(deployment_root),
        )
        port, zmq_port = free_port(), free_port()
        while zmq_port == port:
            zmq_port = free_port()
        if port == 6379:
            raise RuntimeError("refusing production Redis port")
        key = "genmo_deployment_validation"
        endpoint = f"tcp://127.0.0.1:{zmq_port}"
        report.update(redis_port=port, redis_key=key, bridge_endpoint=endpoint)
        cppflags = shlex.split(
            subprocess.check_output(
                ["pkg-config", "--cflags", "--libs", "eigen3", "hiredis"], text=True
            )
        )

        def run(command, **kwargs):
            result = subprocess.run(command, capture_output=True, text=True, env=env, **kwargs)
            if result.returncode:
                raise RuntimeError(
                    f"command failed ({result.returncode}): {command}\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
                )
            return result.stdout

        def launch(command, name):
            log = (tmp / f"{name}.log").open("w")
            log_streams.append(log)
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=tmp)
            processes.append(proc)
            return proc

        try:
            ros_server = SimpleXMLRPCServer(("127.0.0.1", 0), logRequests=False)
            parameters = {"/gmtPolicyFile": str(policy_path), "/robot_type": "bumi"}
            ros_server.register_function(
                lambda caller, name: [1, "ok", parameters[name]]
                if name in parameters else [-1, "missing", ""],
                "getParam",
            )
            ros_thread = threading.Thread(target=ros_server.serve_forever, daemon=True)
            ros_thread.start()
            ros_uri = f"http://127.0.0.1:{ros_server.server_address[1]}"
            report["ros_parameter_fixture_uri"] = ros_uri
            common = [
                "g++",
                "-std=c++17",
                "-O2",
                "-pthread",
                "-DRL_CONTROLLERS_HAS_HIREDIS",
                f"-I{controller / 'include'}",
                f"-I{controller / 'third_party/cnpy'}",
            ]
            run(
                common
                + [
                    str(controller / "test/test_gmt_trajectory_protocol.cpp"),
                    str(controller / "third_party/cnpy/cnpy.cpp"),
                    "-o",
                    str(tmp / "protocol_tests"),
                    "-lgtest",
                    "-lz",
                ]
                + cppflags
            )
            run(
                common
                + [
                    str(ROOT / "tests/fixtures/bumi_gmt_receiver_probe.cpp"),
                    "-o",
                    str(tmp / "receiver"),
                    "-lz",
                ]
                + cppflags
            )
            redis_proc = launch(
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
                    temporary,
                ],
                "redis",
            )
            client = redis.Redis(host="127.0.0.1", port=port, socket_timeout=1)
            for _ in range(100):
                try:
                    if client.ping():
                        break
                except redis.RedisError:
                    pass
                if redis_proc.poll() is not None:
                    raise RuntimeError("isolated Redis exited")
                time.sleep(0.05)
            else:
                raise TimeoutError("isolated Redis did not become ready")
            env["GENMO_GMT_TEST_REDIS_PORT"] = str(port)
            report["cpp_unit_tests"] = run(
                [
                    str(tmp / "protocol_tests"),
                    "--gtest_filter=GmtTrajectoryProtocol.*:MotionLoaderRedisContract.SequenceAckAndLegacyCompatibility",
                ]
            )
            names = tmp / "joint_names.txt"
            names.write_text("\n".join(policy.joint_names) + "\n")
            receiver = launch([str(tmp / "receiver"), str(port), key, str(names)], "receiver")
            bridge = launch(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    str(deployment_root / "scripts/demo/demo_bumi_gmt_bridge.py"),
                    "--bind",
                    endpoint,
                    "--redis-port",
                    str(port),
                    "--redis-key",
                    key,
                    "--kinematics",
                    str(bundle.paths["kinematics"]),
                    "--ros-master-uri",
                    ros_uri,
                    "--audio-playback",
                    "off",
                    "--estop-file",
                    str(tmp / "estop"),
                    "--verbose",
                ],
                "bridge",
            )
            probe = console_module.BridgeClient(endpoint, 500)
            try:
                for _ in range(100):
                    try:
                        status = probe.request({"command": "status"})
                        if status.get("ok"):
                            break
                    except Exception:
                        pass
                    if bridge.poll() is not None or receiver.poll() is not None:
                        raise RuntimeError("Bridge or C++ receiver exited during startup")
                    time.sleep(0.1)
                else:
                    raise TimeoutError("Bridge did not become ready")
            finally:
                probe.close()
            console_args = console_module.build_parser().parse_args(
                [
                    "--deployment-manifest",
                    str(manifest_path),
                    "--bridge",
                    endpoint,
                    "--backend",
                    "tensorrt",
                ]
            )
            console = console_module.ResidentBumiConsole(console_args)
            console.initialize()
            report["identity"] = console.identity.as_dict()
            command = parse_console_line(f"play {shlex.quote(str(audio))} {args.seconds} --seed 42")
            console.start_play(command)
            deadline = time.monotonic() + args.timeout
            saw_playing = False
            maximum_frames = 0
            while time.monotonic() < deadline:
                status = console.status()
                report["last_status"] = status
                state = status["bridge"]["state"]
                maximum_frames = max(maximum_frames, status["bridge"]["accepted_source_frames"])
                saw_playing |= state == "PLAYING"
                if status["last_error"] or status["bridge"]["last_error"]:
                    raise RuntimeError(f"generation/bridge failed: {status}")
                if bridge.poll() is not None or receiver.poll() is not None:
                    raise RuntimeError("Bridge or C++ receiver exited")
                if saw_playing and not status["generation_active"] and state == "STAND":
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("generation/playback did not complete")
            expected_frames = round(args.seconds * 30)
            if (
                not saw_playing
                or maximum_frames != expected_frames
                or status["last_timing"].get("submitted_frames") != expected_frames
            ):
                raise RuntimeError("playback did not submit/accept the complete requested sequence")
            p95 = status["last_timing"].get("continuation_p95_seconds")
            report.update(
                saw_playing=saw_playing,
                maximum_accepted_frames=maximum_frames,
                realtime_pass=p95 is not None and p95 < 3.0,
            )
            console.close()
            console = None
            stop_process(bridge)
            time.sleep(0.4)
            stop_process(receiver)
            receiver_text = (tmp / "receiver.log").read_text()
            report["receiver"] = [
                json.loads(line) for line in receiver_text.splitlines() if line.startswith("{")
            ]
            if receiver.returncode != 0 or not report["receiver"][-1].get("stale_detected"):
                raise RuntimeError("C++ receiver did not validate stream and stale timeout")
            report["pass"] = bool(report["realtime_pass"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            report["error"] = error
            report["diagnostics"] = {
                p.name: p.read_text(errors="replace")[-6000:] for p in tmp.glob("*.log")
            }
        finally:
            if console is not None:
                console.close()
            for process in reversed(processes):
                stop_process(process)
            for stream in log_streams:
                stream.close()
            if ros_server is not None:
                ros_server.shutdown()
                ros_server.server_close()
            if ros_thread is not None:
                ros_thread.join(timeout=2)
            report["temporary_directory"] = temporary
    report["temporary_removed"] = not Path(report["temporary_directory"]).exists()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"pass": report["pass"], "report": str(output), "error": error}, ensure_ascii=False
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
