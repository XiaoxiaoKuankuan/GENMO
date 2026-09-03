#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI GENMO 常驻滚动推理控制台。

本入口只服务“BUMI 模型直接输出机器人 qpos”的在线链路：TensorRT 正式后端或 ONNX
诊断后端在进程内常驻；收到 ``play`` 后提取 30 Hz EDGE35，按 120 帧窗口、30 帧重叠、
90 帧步长逐窗独立 DDIM，在线 overlap-add、根位移单次积分和因果足锁，然后立即把已经
最终确定的 qpos28 后缀发送给独立 GMT 安全桥。第一窗提交 90 帧并暂存 30 帧，下一窗
融合后继续提交，末窗刷新全部尾帧。

控制台支持 ``play/stand/status/quit/shutdown``，默认先生成两个有效块；桥收到预生成块并
获得 GMT ACK 后才启动音乐和 50 Hz 播放。后续生成由 12 秒高水位、4 秒低水位节流，且
持续窗口生成、拼接和后处理 P95 必须低于一个 90 帧步长对应的 3 秒。本文件不导入 GMR、
SMPL、SMPL-X、SMP1 或旧 ``robot_stream.py``，不会改变既有部署入口。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import select
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.robots.bumi.endecoder import BumiEndecoder  # noqa: E402
from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION  # noqa: E402
from gem.robots.bumi.postprocess import (  # noqa: E402
    BUMI_STREAMING_FOOT_LOCK_CONTRACT_VERSION,
)
from gem.runtime.bumi_music_deploy import (  # noqa: E402
    BUMI_SLIDING_QPOS_CONTRACT_VERSION,
    BumiOrtStepRunner,
    BumiStreamingQposGenerator,
    BumiTensorRTStepRunner,
)
from gem.runtime.bumi_music_onnx import BUMI_ONNX_CONTRACT_VERSION  # noqa: E402
from gem.runtime.bumi_online_stream import (  # noqa: E402
    BUMI_ONLINE_QPOS_STREAM_CONTRACT,
    BumiOnlineIdentity,
    BumiOnlineQposChunk,
    ConsoleCommand,
    WatermarkGate,
    bumi_joint_order_sha256,
    parse_console_line,
)
from gem.runtime.music_only_trt import (  # noqa: E402
    exact_motion_frame_count,
    plan_sliding_windows,
    sha256_file,
)
from gem.utils.music_features import (  # noqa: E402
    align_features_to_length,
    extract_edge_baseline35,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("tensorrt", "onnx"), default="tensorrt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-metadata", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--bridge", default="tcp://127.0.0.1:7022")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--feature-cache-size", type=int, default=32)
    parser.add_argument("--request-timeout-ms", type=int, default=5000)
    parser.add_argument("--high-water-seconds", type=float, default=12.0)
    parser.add_argument("--low-water-seconds", type=float, default=4.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=0.5)
    parser.add_argument("--no-foot-lock", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    return parser


HELP = {
    "play": 'play "/absolute/song.wav" [seconds|full] [--start S --seed N]',
    "path_alias": '"/absolute/song.wav"  （默认整首）',
    "commands": ["stand", "status", "quit", "shutdown", "help"],
}


class BridgeClient:
    """带超时和线程锁的 ZMQ REQ 客户端。"""

    def __init__(self, endpoint: str, timeout_ms: int) -> None:
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.connect(endpoint)
        self.lock = threading.Lock()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.socket.send_json(payload)
            return self.socket.recv_json()

    def chunk(self, chunk: BumiOnlineQposChunk) -> dict[str, Any]:
        with self.lock:
            self.socket.send_multipart(chunk.multipart())
            return self.socket.recv_json()

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ResidentBumiConsole:
    """管理常驻推理后端、滚动生成线程、心跳和交互 revision。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        self.onnx_path = args.onnx.expanduser().resolve(strict=True)
        self.kinematics_path = args.kinematics.expanduser().resolve(strict=True)
        self.stats_path = args.stats.expanduser().resolve(strict=True)
        self.metadata_path = (
            args.onnx_metadata.expanduser().resolve(strict=True)
            if args.onnx_metadata is not None
            else self.onnx_path.with_suffix(self.onnx_path.suffix + ".json").resolve(strict=True)
        )
        self.metadata = self._validate_onnx_identity()
        self.endecoder = (
            BumiEndecoder(
                kinematics_path=self.kinematics_path,
                stats_path=self.stats_path,
                enable_contact_targets=False,
            )
            .to(self.device)
            .eval()
        )
        checkpoint_sha = sha256_file(self.checkpoint)
        onnx_sha = sha256_file(self.onnx_path)
        if args.backend == "onnx":
            self.runner = BumiOrtStepRunner(
                self.onnx_path, device=self.device, provider=args.onnx_provider
            )
            inference_path = self.onnx_path
            manifest_path = self.metadata_path
        else:
            if args.engine is None:
                raise ValueError("--backend=tensorrt requires --engine")
            inference_path = args.engine.expanduser().resolve(strict=True)
            self.runner = BumiTensorRTStepRunner(
                inference_path,
                device=self.device,
                use_cuda_graph=not args.no_cuda_graph,
            )
            if self.runner.manifest is None:
                raise RuntimeError("TensorRT backend requires engine.json")
            if self.runner.manifest.get("checkpoint_sha256") != checkpoint_sha:
                raise ValueError("TensorRT manifest checkpoint SHA does not match --checkpoint")
            if self.runner.manifest.get("onnx_sha256") != onnx_sha:
                raise ValueError("TensorRT manifest ONNX SHA does not match --onnx")
            manifest_path = inference_path.parent / "engine.json"
            manifest_path.resolve(strict=True)
        self.identity = BumiOnlineIdentity(
            inference_backend=args.backend,
            checkpoint_sha256=checkpoint_sha,
            onnx_sha256=onnx_sha,
            inference_artifact_sha256=sha256_file(inference_path),
            inference_manifest_sha256=sha256_file(manifest_path),
            stats_sha256=sha256_file(self.stats_path),
            kinematics_sha256=sha256_file(self.kinematics_path),
            joint_order_sha256=bumi_joint_order_sha256(self.endecoder.kinematics.joint_order),
            representation_contract_version=BUMI_REPRESENTATION_CONTRACT_VERSION,
            sliding_contract_version=BUMI_SLIDING_QPOS_CONTRACT_VERSION,
            foot_lock_contract_version=(
                BUMI_STREAMING_FOOT_LOCK_CONTRACT_VERSION if not args.no_foot_lock else "disabled"
            ),
        )
        self.bridge = BridgeClient(args.bridge, args.request_timeout_ms)
        self.feature_cache: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, dict[str, Any]]] = (
            OrderedDict()
        )
        self.feature_cache_lock = threading.Lock()
        self.cancel = threading.Event()
        self.stop = threading.Event()
        self.generation_thread: threading.Thread | None = None
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        # 生成线程、心跳线程和交互主线程都会读写当前请求。旧心跳响应可能晚于新 BEGIN
        # 返回，必须把“读取身份”和“按身份清空”变成同一套受锁保护的条件操作，不能让旧
        # revision 的 STAND 响应误删新任务。
        self.request_state_lock = threading.RLock()
        self.current_revision = -1
        self.current_request_id: str | None = None
        self.last_error: str | None = None
        self.last_timing: dict[str, Any] = {}
        self.timing_lock = threading.Lock()

    def _request_state(self) -> tuple[str | None, int]:
        """返回供跨线程请求使用的一致 ``request_id/revision`` 快照。"""

        with self.request_state_lock:
            return self.current_request_id, self.current_revision

    def _replace_request_state(self, request_id: str | None, revision: int) -> None:
        """原子替换控制台当前请求身份。"""

        with self.request_state_lock:
            self.current_request_id = request_id
            self.current_revision = int(revision)

    def _clear_request_if_matches(self, request_id: str, revision: int) -> bool:
        """只清理由同一次请求产生的状态，拒绝陈旧心跳响应跨 revision 清理。"""

        with self.request_state_lock:
            if self.current_request_id != request_id or self.current_revision != int(revision):
                return False
            self.current_request_id = None
            return True

    def _clear_request(self) -> None:
        """由本地显式取消无条件清空当前活动请求。"""

        with self.request_state_lock:
            self.current_request_id = None

    def _validate_onnx_identity(self) -> dict[str, Any]:
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract_version") != BUMI_ONNX_CONTRACT_VERSION:
            raise ValueError("ONNX metadata is not the BUMI guided denoiser contract")
        if metadata.get("representation_contract_version") != BUMI_REPRESENTATION_CONTRACT_VERSION:
            raise ValueError("ONNX metadata is not the current BUMI qpos30 representation")
        if int(metadata.get("sequence_length", -1)) != 120:
            raise ValueError("BUMI online runtime requires fixed [1,120,30]")
        expected = {
            "checkpoint": (metadata.get("checkpoint") or {}).get("sha256"),
            "kinematics": (metadata.get("kinematics") or {}).get("sha256"),
            "stats": (metadata.get("stats") or {}).get("sha256"),
        }
        actual = {
            "checkpoint": sha256_file(self.checkpoint),
            "kinematics": sha256_file(self.kinematics_path),
            "stats": sha256_file(self.stats_path),
        }
        for name in expected:
            if expected[name] != actual[name]:
                raise ValueError(
                    f"BUMI ONNX {name} identity mismatch: metadata={expected[name]}, actual={actual[name]}"
                )
        return metadata

    def initialize(self) -> None:
        status = self.bridge.request({"command": "status"})
        if not status.get("ok"):
            raise RuntimeError(f"bridge status failed: {status}")
        self._replace_request_state(None, int(status["revision"]))
        self.runner(
            torch.zeros(1, 120, 30, device=self.device),
            torch.tensor([999], device=self.device),
            torch.zeros(1, 120, 35, device=self.device),
            torch.tensor([120], device=self.device),
            torch.tensor([self.args.guidance_scale], device=self.device),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.heartbeat_thread.start()
        print(f"[BUMI Console] {self.args.backend} 后端已常驻，安全桥可达", flush=True)

    def _feature_key(self, path: Path, start: float, duration: float | None) -> tuple[Any, ...]:
        stat = path.stat()
        return (
            str(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            round(start, 6),
            None if duration is None else round(duration, 6),
            "edge_baseline35_v1",
        )

    def _features(
        self, path: Path, start: float, duration: float | None
    ) -> tuple[torch.Tensor, dict[str, Any], bool]:
        key = self._feature_key(path, start, duration)
        with self.feature_cache_lock:
            cached = self.feature_cache.get(key)
            if cached is not None:
                self.feature_cache.move_to_end(key)
                return cached[0], copy.deepcopy(cached[1]), True
        features, metadata = extract_edge_baseline35(
            path, start_sec=start, duration_sec=duration, target_fps=30
        )
        features = features.cpu().float().contiguous()
        with self.feature_cache_lock:
            self.feature_cache[key] = (features, copy.deepcopy(metadata))
            while len(self.feature_cache) > self.args.feature_cache_size:
                self.feature_cache.popitem(last=False)
        return features, metadata, False

    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(self.args.heartbeat_seconds):
            request_id, revision = self._request_state()
            if request_id is None:
                continue
            try:
                response = self.bridge.request(
                    {
                        "command": "heartbeat",
                        "request_id": request_id,
                        "revision": revision,
                    }
                )
                if response.get("state") == "STAND":
                    self._clear_request_if_matches(request_id, revision)
            except Exception as exc:
                self.last_error = f"heartbeat: {type(exc).__name__}: {exc}"

    def _wait_for_stand(self, timeout: float = 8.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.bridge.request({"command": "status"})
            if status.get("state") == "STAND":
                return status
            time.sleep(0.05)
        raise TimeoutError("bridge did not reach STAND before timeout")

    def start_play(self, command: ConsoleCommand) -> dict[str, Any]:
        if self.generation_thread is not None and self.generation_thread.is_alive():
            self.cancel.set()
            self.bridge.request({"command": "stand"})
            self.generation_thread.join(timeout=10.0)
            if self.generation_thread.is_alive():
                raise TimeoutError("previous BUMI window did not cancel within 10 seconds")
        status = self.bridge.request({"command": "status"})
        if status.get("state") != "STAND":
            self.bridge.request({"command": "stand"})
            status = self._wait_for_stand()
        self.cancel = threading.Event()
        revision = int(status["revision"]) + 1
        request_id = uuid.uuid4().hex
        self._replace_request_state(None, revision)
        self.last_error = None
        self.generation_thread = threading.Thread(
            target=self._generate,
            args=(command, request_id, revision, self.cancel),
            daemon=True,
        )
        self.generation_thread.start()
        return {"ok": True, "accepted": True, "request_id": request_id, "revision": revision}

    def _generate(
        self,
        command: ConsoleCommand,
        request_id: str,
        revision: int,
        cancel: threading.Event,
    ) -> None:
        assert command.audio_path is not None
        started = time.perf_counter()
        with self.timing_lock:
            self.last_timing = {
                "request_id": request_id,
                "phase": "feature_extract",
                "generated_windows": 0,
                "submitted_frames": 0,
                "pending_overlap_frames": 0,
                "window_seconds": [],
                "continuation_window_seconds": [],
            }
        try:
            audio_path = command.audio_path.resolve(strict=True)
            feature_started = time.perf_counter()
            features, metadata, cache_hit = self._features(
                audio_path,
                command.start_sec,
                None if command.full else command.duration_sec,
            )
            selected_duration = float(metadata["selected_duration_sec"])
            feature_seconds = time.perf_counter() - feature_started
            frame_count = exact_motion_frame_count(
                len(features), selected_duration if command.full else command.duration_sec
            )
            features = align_features_to_length(features, frame_count, policy="trim_or_pad_last")
            duration = frame_count / 30.0
            windows = plan_sliding_windows(frame_count)
            prime_chunks = min(2, len(windows))
            begin = self.bridge.request(
                {
                    "contract_version": BUMI_ONLINE_QPOS_STREAM_CONTRACT,
                    "command": "begin",
                    "request_id": request_id,
                    "revision": revision,
                    "audio_path": str(audio_path),
                    "audio_start_sec": command.start_sec,
                    "audio_duration_sec": duration,
                    "total_frames": frame_count,
                    "source_fps": 30.0,
                    "seed": command.seed,
                    "prime_chunks": prime_chunks,
                    "identity": self.identity.as_dict(),
                }
            )
            if not begin.get("ok"):
                raise RuntimeError(f"bridge rejected begin: {begin}")
            with self.request_state_lock:
                if self.current_revision != revision:
                    raise RuntimeError("newer console revision superseded this BEGIN")
                self.current_request_id = request_id
            if cancel.is_set():
                self.bridge.request({"command": "stand"})
                return
            generator = BumiStreamingQposGenerator(
                self.runner,
                self.endecoder,
                device=self.device,
                steps=self.args.ddim_steps,
                guidance_scale=self.args.guidance_scale,
                apply_foot_lock=not self.args.no_foot_lock,
            )
            iterator = iter(generator.generate(features, seed=command.seed))
            window_times: list[float] = []
            watermarks = WatermarkGate(self.args.low_water_seconds, self.args.high_water_seconds)
            chunk_index = 0
            while True:
                if cancel.is_set():
                    return
                if chunk_index >= prime_chunks:
                    while True:
                        bridge_status = self.bridge.request({"command": "status"})
                        if bridge_status.get("state") in {"STAND", "STAND_WAIT_ACK"}:
                            raise RuntimeError(
                                f"bridge left active request: {bridge_status.get('state')}"
                            )
                        future = float(bridge_status.get("future_buffer_seconds", 0.0))
                        if not watermarks.should_pause(future):
                            break
                        if cancel.wait(0.1):
                            return
                window_started = time.perf_counter()
                try:
                    generated = next(iterator)
                except StopIteration:
                    break
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                if cancel.is_set():
                    return
                elapsed = time.perf_counter() - window_started
                window_times.append(elapsed)
                continuation = window_times[1:]
                p95 = None if not continuation else float(np.percentile(continuation, 95))
                if p95 is not None and p95 >= 3.0:
                    raise RuntimeError(
                        f"real-time performance gate failed: continuation P95={p95:.3f}s >= 3s"
                    )
                chunk = BumiOnlineQposChunk.from_qpos(
                    generated.qpos.numpy(),
                    request_id=request_id,
                    revision=revision,
                    chunk_index=chunk_index,
                    absolute_start_frame=generated.absolute_start_frame,
                    total_frames=frame_count,
                    is_last=generated.is_last,
                    identity=self.identity,
                )
                response = self.bridge.chunk(chunk)
                if not response.get("ok"):
                    raise RuntimeError(f"bridge rejected qpos chunk: {response}")
                chunk_index += 1
                with self.timing_lock:
                    self.last_timing.update(
                        {
                            "phase": "queued" if generated.is_last else "streaming",
                            "feature_seconds": feature_seconds,
                            "feature_cache_hit": cache_hit,
                            "feature_metadata": metadata,
                            "generated_windows": generator.windows_generated,
                            "submitted_frames": generator.emitted_frames,
                            "pending_overlap_frames": generator.pending_frames,
                            "future_buffer_seconds": response.get("future_buffer_seconds"),
                            "window_seconds": list(window_times),
                            "continuation_window_seconds": list(continuation),
                            "continuation_p95_seconds": p95,
                            "prime_chunks": prime_chunks,
                        }
                    )
                print(
                    f"[BUMI Window {chunk_index}/{len(windows)}] "
                    f"frames={generated.absolute_start_frame}:"
                    f"{generated.absolute_start_frame + len(generated.qpos)} "
                    f"pending={generator.pending_frames} elapsed={elapsed:.3f}s",
                    flush=True,
                )
            with self.timing_lock:
                self.last_timing["total_generation_seconds"] = time.perf_counter() - started
            print(f"[BUMI Generate] 已连续提交 {frame_count} 帧", flush=True)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[BUMI Generate ERROR] {self.last_error}", flush=True)
            try:
                self.bridge.request({"command": "stand"})
            except Exception:
                pass
            self._clear_request_if_matches(request_id, revision)

    def stand(self) -> dict[str, Any]:
        self.cancel.set()
        self._clear_request()
        return self.bridge.request({"command": "stand"})

    def status(self) -> dict[str, Any]:
        bridge = self.bridge.request({"command": "status"})
        request_id, revision = self._request_state()
        with self.timing_lock:
            timing = copy.deepcopy(self.last_timing)
        return {
            "ok": True,
            "identity": self.identity.as_dict(),
            "cuda_graph": getattr(self.runner, "cuda_graph", None) is not None,
            "generation_active": self.generation_thread is not None
            and self.generation_thread.is_alive(),
            "request_id": request_id,
            "revision": revision,
            "last_error": self.last_error,
            "last_timing": timing,
            "bridge": bridge,
        }

    def serve(self) -> None:
        print(json.dumps(HELP, indent=2, ensure_ascii=False))
        while not self.stop.is_set():
            print("bumi> ", end="", flush=True)
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            while not readable and not self.stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            if self.stop.is_set():
                break
            line = sys.stdin.readline()
            if line == "":
                break
            try:
                command = parse_console_line(line)
                if command is None:
                    continue
                if command.name == "play":
                    response = self.start_play(command)
                elif command.name == "stand":
                    response = self.stand()
                elif command.name == "status":
                    response = self.status()
                elif command.name == "help":
                    response = {"ok": True, "help": HELP}
                elif command.name == "shutdown":
                    self.stand()
                    self._wait_for_stand()
                    response = self.bridge.request({"command": "shutdown"})
                    self.stop.set()
                elif command.name == "quit":
                    self.stand()
                    self._wait_for_stand()
                    response = {"ok": True, "message": "控制台退出，桥保持 STAND"}
                    self.stop.set()
                else:
                    raise ValueError(f"unsupported console command {command.name}")
            except Exception as exc:
                response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            print(json.dumps(response, indent=2, ensure_ascii=False, default=str))

    def close(self) -> None:
        self.cancel.set()
        self.stop.set()
        if self.generation_thread is not None:
            self.generation_thread.join(timeout=5.0)
        self.bridge.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.low_water_seconds < args.high_water_seconds:
        parser.error("require 0 < --low-water-seconds < --high-water-seconds")
    if args.feature_cache_size <= 0 or args.request_timeout_ms <= 0:
        parser.error("cache size and timeout must be positive")
    if args.ddim_steps <= 0 or not math.isfinite(args.guidance_scale):
        parser.error("DDIM steps must be positive and guidance scale finite")
    console = ResidentBumiConsole(args)
    try:
        console.initialize()
        console.serve()
    finally:
        console.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
