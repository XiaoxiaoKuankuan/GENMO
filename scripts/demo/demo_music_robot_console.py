#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Resident TensorRT music console for the independent BUMI safety bridge."""

from __future__ import annotations

import argparse
import copy
import json
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

from gem.gmr_udp_bridge import SMP1PacketEncoder  # noqa: E402
from gem.runtime.music_only_trt import (  # noqa: E402
    OVERLAP_FRAMES,
    SlidingDDIMGenerator,
    StreamingSmplDecoder,
    TensorRTStepRunner,
    derive_window_seed,
    exact_motion_frame_count,
    padded_music_window,
    plan_sliding_windows,
    sha256_file,
)
from gem.runtime.robot_stream import (  # noqa: E402
    ConsoleCommand,
    RobotStreamChunk,
    parse_console_line,
    smpl_params_to_smp1_payload,
)
from gem.smplx_gmr_reference import SMPLXGMRReference  # noqa: E402
from gem.utils.music_features import (  # noqa: E402
    align_features_to_length,
    extract_edge_baseline35,
)
from scripts.demo.stream_smpl_params_to_gmr import load_endecoder  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    "outputs/gem_smpl_music_only_4set_physics_v1/version_0/checkpoints/s050000.ckpt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bridge", default="tcp://127.0.0.1:7021")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--feature-cache-size", type=int, default=32)
    parser.add_argument("--request-timeout-ms", type=int, default=5000)
    parser.add_argument("--high-water-seconds", type=float, default=12.0)
    parser.add_argument("--low-water-seconds", type=float, default=4.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=0.5)
    parser.add_argument("--no-cuda-graph", action="store_true")
    return parser


HELP = {
    "play": 'play "/absolute/song.wav" [seconds|full] [--start S --seed N]',
    "path_alias": '"/absolute/song.wav"  (defaults to full song)',
    "commands": ["stand", "status", "quit", "shutdown", "help"],
}


class BridgeClient:
    """Thread-safe REQ client; all replies remain bounded by socket timeouts."""

    def __init__(self, endpoint: str, timeout_ms: int) -> None:
        import zmq

        self.zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(endpoint)
        self.lock = threading.Lock()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.socket.send_json(payload)
            return self.socket.recv_json()

    def chunk(self, chunk: RobotStreamChunk) -> dict[str, Any]:
        with self.lock:
            self.socket.send_multipart(chunk.multipart())
            return self.socket.recv_json()

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ResidentRobotConsole:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        self.engine_path = args.engine.expanduser().resolve(strict=True)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.engine_sha256 = sha256_file(self.engine_path)
        self.bridge = BridgeClient(args.bridge, args.request_timeout_ms)
        self.runner = TensorRTStepRunner(
            self.engine_path,
            device=self.device,
            use_cuda_graph=not args.no_cuda_graph,
        )
        if (
            self.runner.manifest is None
            or self.runner.manifest.get("checkpoint_sha256") != self.checkpoint_sha256
        ):
            raise RuntimeError(
                "TensorRT engine manifest checkpoint SHA256 does not match --checkpoint"
            )
        self.generator = SlidingDDIMGenerator(
            self.runner,
            device=self.device,
            steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
        )
        self.endecoder = load_endecoder(self.device)
        self.feature_cache: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, dict[str, Any]]] = (
            OrderedDict()
        )
        self.feature_cache_lock = threading.Lock()
        self.cancel = threading.Event()
        self.stop = threading.Event()
        self.generation_thread: threading.Thread | None = None
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.current_revision = -1
        self.current_request_id: str | None = None
        self.last_error: str | None = None
        self.last_timing: dict[str, Any] = {}
        self.timing_lock = threading.Lock()

    def _prime_windows(self, bridge_status: dict[str, Any]) -> tuple[int, float | None]:
        with self.timing_lock:
            timings = list(self.last_timing.get("continuation_window_seconds", []))
        gmr_mean_ms = bridge_status.get("gmr_mean_ms")
        if not timings or gmr_mean_ms is None:
            return 2, None
        combined = np.asarray(timings, dtype=np.float64) + float(gmr_mean_ms) * 90.0 / 1000.0
        p95 = float(np.percentile(combined, 95))
        if p95 >= 3.0:
            raise RuntimeError(
                f"real-time performance gate failed: continuation P95={p95:.3f}s >= 3s"
            )
        return (1 if p95 < 1.5 else 2), p95

    def initialize(self) -> None:
        status = self.bridge.request({"command": "status"})
        if not status.get("ok"):
            raise RuntimeError(f"bridge status failed: {status}")
        self.current_revision = int(status["revision"])
        # Warm one fixed TensorRT enqueue; the generator performs real diffusion
        # only after a play command supplies music.
        zeros_motion = torch.zeros(1, 120, 151, device=self.device)
        zeros_music = torch.zeros(1, 120, 35, device=self.device)
        self.runner(
            zeros_motion,
            torch.tensor([999], device=self.device),
            zeros_music,
            torch.tensor([120], device=self.device),
            torch.tensor([self.args.guidance_scale], device=self.device),
        )
        torch.cuda.synchronize(self.device)
        self.heartbeat_thread.start()
        print("[Console] TensorRT engine resident and bridge reachable")

    def _feature_key(
        self, path: Path, start_sec: float, duration_sec: float | None
    ) -> tuple[Any, ...]:
        stat = path.stat()
        return (
            str(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            round(start_sec, 6),
            None if duration_sec is None else round(duration_sec, 6),
            "edge_baseline35_v1",
        )

    def _features(
        self, path: Path, start_sec: float, duration_sec: float | None
    ) -> tuple[torch.Tensor, dict[str, Any], bool]:
        key = self._feature_key(path, start_sec, duration_sec)
        with self.feature_cache_lock:
            value = self.feature_cache.get(key)
            if value is not None:
                self.feature_cache.move_to_end(key)
                return value[0], copy.deepcopy(value[1]), True
        features, metadata = extract_edge_baseline35(
            path, start_sec=start_sec, duration_sec=duration_sec, target_fps=30
        )
        features = features.cpu().float().contiguous()
        with self.feature_cache_lock:
            self.feature_cache[key] = (features, copy.deepcopy(metadata))
            while len(self.feature_cache) > self.args.feature_cache_size:
                self.feature_cache.popitem(last=False)
        return features, metadata, False

    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(self.args.heartbeat_seconds):
            request_id = self.current_request_id
            if request_id is None:
                continue
            try:
                response = self.bridge.request(
                    {
                        "command": "heartbeat",
                        "request_id": request_id,
                        "revision": self.current_revision,
                    }
                )
                if response.get("state") == "STAND":
                    self.current_request_id = None
            except Exception as exc:
                self.last_error = f"heartbeat: {type(exc).__name__}: {exc}"

    def _wait_for_stand(self, timeout: float = 8.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.bridge.request({"command": "status"})
            if last.get("state") == "STAND":
                return last
            time.sleep(0.1)
        raise TimeoutError(f"bridge did not reach STAND: {last}")

    def stand(self) -> dict[str, Any]:
        self.cancel.set()
        response = self.bridge.request({"command": "stand"})
        self.current_request_id = None
        return response

    def start_play(self, command: ConsoleCommand) -> dict[str, Any]:
        assert command.audio_path is not None
        self.cancel.set()
        if self.generation_thread is not None and self.generation_thread.is_alive():
            self.bridge.request({"command": "stand"})
            self.generation_thread.join(timeout=10.0)
            if self.generation_thread.is_alive():
                raise TimeoutError("previous TensorRT window did not cancel within 10 seconds")
        status = self.bridge.request({"command": "status"})
        if status.get("state") != "STAND":
            self.bridge.request({"command": "stand"})
            status = self._wait_for_stand()
        self.cancel = threading.Event()
        revision = int(status["revision"]) + 1
        request_id = uuid.uuid4().hex
        self.current_revision = revision
        # Heartbeats start only after BEGIN is accepted.  During EDGE35 feature
        # extraction the bridge still owns STAND and has no active revision.
        self.current_request_id = None
        self.last_error = None
        self.generation_thread = threading.Thread(
            target=self._generate,
            args=(command, request_id, revision, self.cancel),
            daemon=True,
        )
        self.generation_thread.start()
        return {
            "ok": True,
            "accepted": True,
            "request_id": request_id,
            "revision": revision,
        }

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
                "generated_source_frames": 0,
                "window_seconds": [],
                "trt_seconds": [],
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
            feature_seconds = time.perf_counter() - feature_started
            with self.timing_lock:
                self.last_timing.update(
                    {
                        "phase": "prepare_bridge",
                        "feature_seconds": feature_seconds,
                        "feature_cache_hit": cache_hit,
                        "feature_metadata": metadata,
                    }
                )
            if cancel.is_set():
                return
            selected_audio_duration = float(metadata["selected_duration_sec"])
            if (
                not command.full
                and command.duration_sec is not None
                and selected_audio_duration + 1e-3 < command.duration_sec
            ):
                raise ValueError(
                    f"selected audio has only {selected_audio_duration:.6f}s after "
                    f"start={command.start_sec:.6f}s, fewer than requested "
                    f"{command.duration_sec:.6f}s"
                )
            frame_count = exact_motion_frame_count(
                len(features),
                selected_audio_duration if command.full else command.duration_sec,
            )
            features = align_features_to_length(
                features, frame_count, policy="trim_or_pad_last"
            )
            duration = frame_count / 30.0
            pre_begin_status = self.bridge.request({"command": "status"})
            prime_windows, previous_p95 = self._prime_windows(pre_begin_status)
            begin = self.bridge.request(
                {
                    "contract_version": "robot_stream_v1",
                    "command": "begin",
                    "request_id": request_id,
                    "revision": revision,
                    "audio_path": str(audio_path),
                    "audio_start_sec": command.start_sec,
                    "audio_duration_sec": duration,
                    "total_frames": frame_count,
                    "source_fps": 30.0,
                    "seed": command.seed,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "engine_sha256": self.engine_sha256,
                    "prime_windows": prime_windows,
                }
            )
            if not begin.get("ok"):
                raise RuntimeError(f"bridge rejected begin: {begin}")
            if cancel.is_set():
                self.bridge.request({"command": "stand"})
                return
            self.current_request_id = request_id

            decoder = StreamingSmplDecoder(self.endecoder, self.device)
            adapter = SMPLXGMRReference(user_yaw_deg=0.0, global_scale=1.0)
            encoder = SMP1PacketEncoder(debug=False)
            previous: torch.Tensor | None = None
            window_times: list[float] = []
            trt_times: list[float] = []
            windows = plan_sliding_windows(frame_count)
            high_water_latched = False
            for window in windows:
                if cancel.is_set():
                    return
                while True:
                    bridge_status = self.bridge.request({"command": "status"})
                    if bridge_status.get("state") in {"STAND", "STAND_WAIT_ACK"}:
                        raise RuntimeError(
                            f"bridge left active request: {bridge_status.get('state')}"
                        )
                    future_seconds = float(
                        bridge_status.get("future_buffer_seconds", 0.0)
                    )
                    if future_seconds >= self.args.high_water_seconds:
                        high_water_latched = True
                    resume_threshold = (
                        self.args.low_water_seconds
                        if high_water_latched
                        else self.args.high_water_seconds
                    )
                    if future_seconds < resume_threshold:
                        high_water_latched = False
                        break
                    if cancel.wait(0.1):
                        return

                window_started = time.perf_counter()
                music_window = padded_music_window(features, window)
                known = None
                if previous is not None:
                    known = previous[-OVERLAP_FRAMES:].clone()
                trt_started = time.perf_counter()
                generated = self.generator.generate_window(
                    music_window,
                    valid_length=window.valid_length,
                    seed=derive_window_seed(command.seed, window.index),
                    known_x0=known,
                )
                torch.cuda.synchronize(self.device)
                trt_times.append(time.perf_counter() - trt_started)
                if cancel.is_set():
                    return
                new_start = window.known_length
                params = decoder.decode_new(
                    generated, start=new_start, end=window.valid_length
                )
                absolute_start = window.start + new_start
                payload = smpl_params_to_smp1_payload(
                    params,
                    endecoder=self.endecoder,
                    adapter=adapter,
                    encoder=encoder,
                    absolute_start_frame=absolute_start,
                )
                chunk = RobotStreamChunk(
                    request_id=request_id,
                    revision=revision,
                    chunk_index=window.index,
                    absolute_start_frame=absolute_start,
                    frame_count=window.new_length,
                    total_frames=frame_count,
                    is_last=window.index == len(windows) - 1,
                    checkpoint_sha256=self.checkpoint_sha256,
                    engine_sha256=self.engine_sha256,
                    payload=payload,
                )
                while not cancel.is_set():
                    response = self.bridge.chunk(chunk)
                    if response.get("ok"):
                        break
                    if not response.get("backpressure"):
                        raise RuntimeError(f"bridge rejected chunk: {response}")
                    cancel.wait(0.05)
                previous = generated.detach()
                window_times.append(time.perf_counter() - window_started)
                with self.timing_lock:
                    self.last_timing.update(
                        {
                            "phase": "streaming",
                            "generated_source_frames": absolute_start
                            + window.new_length,
                            "window_seconds": list(window_times),
                            "trt_seconds": list(trt_times),
                            "continuation_window_seconds": list(window_times[1:]),
                            "prime_windows": prime_windows,
                            "previous_continuation_p95_seconds": previous_p95,
                        }
                    )
                print(
                    f"[Window {window.index + 1}/{len(windows)}] "
                    f"frames={absolute_start}:{absolute_start + window.new_length} "
                    f"elapsed={window_times[-1]:.3f}s",
                    flush=True,
                )
            with self.timing_lock:
                self.last_timing.update(
                    {
                        "phase": "queued",
                        "total_generation_seconds": time.perf_counter() - started,
                    }
                )
            print(f"[Generate] all {frame_count} source frames queued")
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[Generate ERROR] {self.last_error}")
            try:
                self.bridge.request({"command": "stand"})
            except Exception:
                pass
            self.current_request_id = None

    def status(self) -> dict[str, Any]:
        bridge = self.bridge.request({"command": "status"})
        with self.timing_lock:
            timing = copy.deepcopy(self.last_timing)
        return {
            "ok": True,
            "checkpoint_sha256": self.checkpoint_sha256,
            "engine_sha256": self.engine_sha256,
            "cuda_graph": self.runner.cuda_graph is not None,
            "generation_active": (
                self.generation_thread is not None and self.generation_thread.is_alive()
            ),
            "request_id": self.current_request_id,
            "revision": self.current_revision,
            "last_error": self.last_error,
            "last_timing": timing,
            "bridge": bridge,
        }

    def serve(self) -> None:
        print(json.dumps(HELP, indent=2, ensure_ascii=False))
        while not self.stop.is_set():
            print("robot> ", end="", flush=True)
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
                    response = {"ok": True, "message": "console stopped; bridge remains STAND"}
                    self.stop.set()
                else:
                    raise ValueError(f"unsupported console command {command.name}")
            except Exception as exc:
                response = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            print(json.dumps(response, indent=2, ensure_ascii=False, default=str))

    def close(self) -> None:
        self.cancel.set()
        self.stop.set()
        if self.generation_thread is not None:
            self.generation_thread.join(timeout=5.0)
        self.bridge.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.feature_cache_size < 1:
        raise ValueError("--feature-cache-size must be >= 1")
    if not 0.0 < args.low_water_seconds < args.high_water_seconds:
        raise ValueError("water levels must satisfy 0 < low < high")
    if not 0.0 < args.heartbeat_seconds < 1.5:
        raise ValueError("--heartbeat-seconds must be in (0, 1.5)")
    console = ResidentRobotConsole(args)
    try:
        console.initialize()
        console.serve()
    except KeyboardInterrupt:
        try:
            console.stand()
            console._wait_for_stand()
        except Exception:
            pass
    finally:
        console.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
