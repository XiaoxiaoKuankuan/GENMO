#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Run the unified resident video/text/music motion-control service."""

from __future__ import annotations

import argparse
import json
import math
import select
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.runtime.motion_source_mux import MotionSourceMux  # noqa: E402
from gem.runtime.resident_multimodal_motion import (  # noqa: E402
    ResidentMultimodalMotionEngine,
    UnsupportedModeError,
)
from gem.runtime.resident_video_session import (  # noqa: E402
    ResidentVideoModelStack,
    ResidentVideoSession,
)

VIDEO_FUSION_ERROR = "True video multimodal fusion is not supported by the real-time ONNX path."
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac"})


def resolve_server_file(
    value: str,
    *,
    allowed_roots: tuple[Path, ...],
    suffixes: frozenset[str],
    label: str,
) -> Path:
    """Resolve one regular server-side file and reject traversal/symlink escape."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty server path")
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise PermissionError(f"{label} may not contain '..': {value}")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    if resolved.suffix.lower() not in suffixes:
        allowed = ", ".join(sorted(suffixes))
        raise ValueError(f"{label} must use one of these extensions: {allowed}")
    if allowed_roots and not any(
        resolved == root or resolved.is_relative_to(root) for root in allowed_roots
    ):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise PermissionError(
            f"{label} resolves outside the configured allowed roots: {resolved}; roots={roots}"
        )
    return resolved


def resolve_allowed_roots(values: list[Path]) -> tuple[Path, ...]:
    """Resolve and validate repeated allow-list roots at startup."""
    result: list[Path] = []
    for value in values:
        root = value.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"Allowed root is not a directory: {root}")
        result.append(root)
    return tuple(result)


def error_response(exc: Exception, request_id: Any = None) -> dict[str, Any]:
    """Return one stable structured protocol error."""
    return {
        "ok": False,
        "request_id": None if request_id is None else str(request_id),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


class MultimodalService:
    """Coordinate one generator, one video session, and one GMR mux."""

    def __init__(
        self,
        *,
        engine: Any,
        video_stack: Any,
        video_init: str = "eager",
        allowed_video_roots: tuple[Path, ...] = (),
        mux: Any | None = None,
        video_session: Any | None = None,
        mux_factory: Callable[..., Any] = MotionSourceMux,
        mux_options: dict[str, Any] | None = None,
        video_session_factory: Callable[..., Any] = ResidentVideoSession,
    ) -> None:
        if video_init not in {"eager", "lazy"}:
            raise ValueError("video_init must be eager or lazy")
        self.engine = engine
        self.video_stack = video_stack
        self.video_init = video_init
        self.allowed_video_roots = tuple(allowed_video_roots)
        self.allowed_audio_roots = tuple(getattr(engine, "allowed_audio_roots", ()))
        self.mux = mux
        self.video_session = video_session
        self._mux_factory = mux_factory
        self._mux_options = dict(mux_options or {})
        self._video_session_factory = video_session_factory
        self.generation_pause = threading.Event()
        self.initialized = False
        self.shutdown_requested = False
        self.request_count = 0

    def initialize(self) -> None:
        """Load all configured models once and start the sole GMR sender."""
        if self.initialized:
            return
        self.engine.initialize()
        if self.video_init == "eager":
            self.video_stack.initialize()
            if hasattr(self.engine, "record_external_memory_stage"):
                self.engine.record_external_memory_stage("after video model stack load")

        if self.mux is None:
            shared_endecoder = (
                self.video_stack.endecoder
                if getattr(self.video_stack, "initialized", False)
                else None
            )
            self.mux = self._mux_factory(
                endecoder=shared_endecoder,
                **self._mux_options,
            )
        self.mux.start()
        if (
            self.video_init == "lazy"
            and getattr(self.video_stack, "shared_endecoder", None) is None
            and getattr(self.mux, "endecoder", None) is not None
        ):
            self.video_stack.shared_endecoder = self.mux.endecoder

        if self.video_session is None:
            self.video_session = self._video_session_factory(
                self.video_stack,
                frame_sink=self.mux.submit_video_frame,
                generation_pause=self.generation_pause,
            )
        self.mux.on_video_resume_reset = self.video_session.request_reset
        self.initialized = True
        print("[Multimodal] Unified service initialized; one GMR sender is active")

    @staticmethod
    def _require_known_fields(payload: dict[str, Any], allowed: set[str]) -> None:
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unsupported request fields: {unknown}")

    def _handle_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("mode")
        if mode in {"video_text", "video_music", "video_text_music"}:
            raise UnsupportedModeError(VIDEO_FUSION_ERROR)
        if mode not in {"text", "music", "text_music"}:
            raise UnsupportedModeError(f"Unsupported generation mode: {mode}")
        allowed = {"op", "mode", "request_id", "seed"}
        if mode in {"text", "text_music"}:
            allowed.add("prompt")
        if mode in {"music", "text_music"}:
            allowed.update({"audio_path", "start_sec"})
        self._require_known_fields(payload, allowed)

        request: dict[str, Any] = {
            "mode": mode,
            "request_id": payload.get("request_id"),
            "seed": payload.get("seed", 42),
        }
        if mode in {"text", "text_music"}:
            request["prompt"] = payload.get("prompt")
        if mode in {"music", "text_music"}:
            audio_path = resolve_server_file(
                payload.get("audio_path"),
                allowed_roots=self.allowed_audio_roots,
                suffixes=SUPPORTED_AUDIO_SUFFIXES,
                label="audio_path",
            )
            request["audio_path"] = str(audio_path)
            request["start_sec"] = payload.get("start_sec", 0.0)

        active_video = bool(self.video_session.status().get("active", False))
        if active_video:
            self.video_session.pause_inference()
        try:
            response = self.engine.generate(request)
            if response.get("ok"):
                self.mux.submit_generated_motion(response["output_dir"])
                response = dict(response)
                response["submitted_to_mux"] = True
                response["mux_state"] = self.mux.status()["state"]
            return response
        finally:
            if active_video:
                self.video_session.resume_inference()

    def _handle_video_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_known_fields(payload, {"op", "camera_id", "video_path"})
        camera_id = payload.get("camera_id")
        video_value = payload.get("video_path")
        if (camera_id is None) == (video_value is None):
            raise ValueError("video_start requires exactly one of camera_id or video_path")
        if camera_id is not None:
            if not isinstance(camera_id, int) or isinstance(camera_id, bool):
                raise TypeError("camera_id must be an integer")
            if camera_id < 0:
                raise ValueError("camera_id must be >= 0")
            self.video_session.start_source(camera_id=camera_id)
        else:
            video_path = resolve_server_file(
                video_value,
                allowed_roots=self.allowed_video_roots,
                suffixes=SUPPORTED_VIDEO_SUFFIXES,
                label="video_path",
            )
            self.video_session.start_source(video_path=video_path)
        self.mux.start_video_mode()
        return {
            "ok": True,
            "op": "video_start",
            "video": self.video_session.status(),
            "mux": self.mux.status(),
        }

    def handle(self, payload: Any) -> tuple[dict[str, Any], bool]:
        """Handle one protocol object; all recoverable failures stay local."""
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        self.request_count += 1
        try:
            if not isinstance(payload, dict):
                raise TypeError("request must be a JSON object")
            op = payload.get("op")
            if not isinstance(op, str) or not op:
                raise ValueError("request requires a non-empty string 'op'")
            if op == "generate":
                return self._handle_generate(payload), False
            if op == "video_start":
                return self._handle_video_start(payload), False
            if op == "video_stop":
                self._require_known_fields(payload, {"op"})
                self.video_session.stop_source()
                self.mux.stop_video_mode()
                return {"ok": True, "op": op, "mux": self.mux.status()}, False
            if op == "idle":
                self._require_known_fields(payload, {"op"})
                self.video_session.stop_source()
                self.mux.set_idle()
                return {"ok": True, "op": op, "mux": self.mux.status()}, False
            if op == "estop":
                self._require_known_fields(payload, {"op"})
                if self.video_session.status().get("active", False):
                    self.video_session.pause_inference()
                self.mux.estop()
                return {"ok": True, "op": op, "mux": self.mux.status()}, False
            if op == "clear_estop":
                self._require_known_fields(payload, {"op"})
                self.mux.clear_estop()
                return {"ok": True, "op": op, "mux": self.mux.status()}, False
            if op == "status":
                self._require_known_fields(payload, {"op"})
                return {
                    "ok": True,
                    "status": {
                        "service_initialized": self.initialized,
                        "request_count": self.request_count,
                        "engine": self.engine.status(),
                        "video": self.video_session.status(),
                        "mux": self.mux.status(),
                    },
                }, False
            if op == "clear_cache":
                self._require_known_fields(payload, {"op", "target"})
                target = payload.get("target", "all")
                removed = self.engine.clear_cache(target)
                return {"ok": True, "removed": removed}, False
            if op == "shutdown":
                self._require_known_fields(payload, {"op"})
                self.shutdown_requested = True
                return {"ok": True, "message": "service stopping"}, True
            raise ValueError(f"unknown operation: {op}")
        except Exception as exc:
            return error_response(exc, request_id), False

    def close(self) -> None:
        """Stop sources/GMR before releasing the resident GPU models."""
        try:
            if self.video_session is not None:
                self.video_session.close()
        finally:
            try:
                if self.mux is not None:
                    self.mux.close()
            finally:
                self.engine.close()
        self.initialized = False


def build_parser() -> argparse.ArgumentParser:
    """Build the immutable unified-service configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("inputs/pretrained/gem_smpl.ckpt"),
    )
    parser.add_argument("--t5_model", default="t5-3b")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text_encoder_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--video_init", choices=("eager", "lazy"), default="eager")
    parser.add_argument("--clip_frames", type=int, default=120)
    parser.add_argument("--clip_fps", type=int, default=30)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/multimodal_motion"),
    )
    parser.add_argument("--transport", choices=("zmq", "stdin"), default="zmq")
    parser.add_argument("--bind", default="tcp://127.0.0.1:7020")
    parser.add_argument("--gmr_host", default="127.0.0.1")
    parser.add_argument("--gmr_port", type=int, default=7006)
    parser.add_argument("--publish_fps", type=float, default=30.0)
    parser.add_argument("--shape_mode", choices=("zero",), default="zero")
    parser.add_argument("--mode", choices=("sim", "robot"), default="sim")
    parser.add_argument("--idle_motion", type=Path)
    parser.add_argument("--blend_seconds", type=float, default=0.8)
    parser.add_argument("--return_seconds", type=float, default=1.0)
    parser.add_argument("--estop_blend_seconds", type=float, default=0.3)
    parser.add_argument("--video_stale_sec", type=float, default=0.5)
    parser.add_argument(
        "--new_motion_policy",
        choices=("queue", "latest", "interrupt"),
        default="queue",
    )
    parser.add_argument("--allow_interrupt_in_robot", action="store_true")
    parser.add_argument("--reset_origin_on_motion", action="store_true")
    parser.add_argument("--smplx_yaw_deg", type=float, default=0.0)
    parser.add_argument("--gmr_scale", type=float, default=1.0)
    parser.add_argument("--max_send_errors", type=int, default=5)
    parser.add_argument("--no_imgfeat", action="store_true")
    parser.add_argument("--context_frames", type=int, default=120)
    parser.add_argument("--yolo_period", type=int, default=5)
    parser.add_argument("--vitpose_period", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--focal", type=float)
    parser.add_argument("--bbox_scale", type=float, default=0.75)
    parser.add_argument("--no_postproc", action="store_true")
    parser.add_argument("--text_cache_size", type=int, default=128)
    parser.add_argument("--music_cache_size", type=int, default=32)
    parser.add_argument("--allowed_audio_root", action="append", type=Path, default=[])
    parser.add_argument("--allowed_video_root", action="append", type=Path, default=[])
    parser.add_argument("--min_free_gib", type=float, default=4.0)
    parser.add_argument("--strict_memory", action="store_true")
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe or inconsistent service configuration before loading models."""
    if args.clip_frames <= 0:
        raise ValueError("--clip_frames must be > 0")
    if args.clip_fps != 30:
        raise ValueError("--clip_fps must be 30")
    if args.ddim_steps <= 0:
        raise ValueError("--ddim_steps must be > 0")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        raise ValueError("--guidance_scale must be finite and >= 0")
    if args.width <= 0 or args.height <= 0 or args.context_frames <= 0:
        raise ValueError("image dimensions and --context_frames must be > 0")
    if args.mode == "robot" and args.idle_motion is None:
        raise RuntimeError("Robot mode requires a verified idle SMPL-X motion file.")
    if (
        args.mode == "robot"
        and args.new_motion_policy == "interrupt"
        and not args.allow_interrupt_in_robot
    ):
        raise RuntimeError("Robot mode forbids interrupt by default")
    if args.min_free_gib < 0:
        raise ValueError("--min_free_gib must be >= 0")


def create_service(args: argparse.Namespace) -> MultimodalService:
    """Create unloaded components; ``initialize`` controls the one-time order."""
    audio_roots = resolve_allowed_roots(args.allowed_audio_root)
    video_roots = resolve_allowed_roots(args.allowed_video_root)
    engine = ResidentMultimodalMotionEngine(
        ckpt_path=args.ckpt_path,
        t5_model=args.t5_model,
        local_files_only=args.local_files_only,
        device=args.device,
        text_dtype=args.text_encoder_dtype,
        clip_frames=args.clip_frames,
        clip_fps=args.clip_fps,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        focal=args.focal,
        bbox_scale=args.bbox_scale,
        output_root=args.output_root,
        postproc=not args.no_postproc,
        text_cache_size=args.text_cache_size,
        music_cache_size=args.music_cache_size,
        allowed_audio_roots=audio_roots,
        min_free_gib=args.min_free_gib,
        strict_memory=args.strict_memory,
        warmup_enabled=not args.no_warmup,
    )
    video_stack = ResidentVideoModelStack(
        no_imgfeat=args.no_imgfeat,
        context_frames=args.context_frames,
        device=args.device,
        warmup_enabled=not args.no_warmup,
        warmup_width=args.width,
        warmup_height=args.height,
    )
    mux_options = {
        "gmr_host": args.gmr_host,
        "gmr_port": args.gmr_port,
        "publish_fps": args.publish_fps,
        "shape_mode": args.shape_mode,
        "mode": args.mode,
        "idle_motion": args.idle_motion,
        "blend_seconds": args.blend_seconds,
        "return_seconds": args.return_seconds,
        "estop_blend_seconds": args.estop_blend_seconds,
        "video_stale_sec": args.video_stale_sec,
        "new_motion_policy": args.new_motion_policy,
        "allow_interrupt_in_robot": args.allow_interrupt_in_robot,
        "reset_origin_on_motion": args.reset_origin_on_motion,
        "smplx_yaw_deg": args.smplx_yaw_deg,
        "gmr_scale": args.gmr_scale,
        "max_send_errors": args.max_send_errors,
        "device": args.device,
        "verbose": args.verbose,
    }
    return MultimodalService(
        engine=engine,
        video_stack=video_stack,
        video_init=args.video_init,
        allowed_video_roots=video_roots,
        mux_options=mux_options,
    )


def serve_zmq(
    service: MultimodalService,
    bind: str,
    stop_event: threading.Event,
) -> None:
    """Serve sequential REP requests while source/GMR threads remain independent."""
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(bind)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print(f"[Multimodal] ZMQ REP listening on {bind}")
    try:
        while not stop_event.is_set():
            if socket not in dict(poller.poll(200)):
                continue
            try:
                payload = socket.recv_json()
                response, should_stop = service.handle(payload)
            except Exception as exc:
                response, should_stop = error_response(exc), False
            socket.send_json(response)
            if should_stop:
                stop_event.set()
    finally:
        poller.unregister(socket)
        socket.close(linger=0)
        context.term()


def serve_stdin(
    service: MultimodalService,
    stop_event: threading.Event,
) -> None:
    """Serve one JSON object per line for local diagnosis."""
    print("Enter one JSON request per line; Ctrl-D or shutdown exits.")
    while not stop_event.is_set():
        readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not readable:
            continue
        line = sys.stdin.readline()
        if line == "":
            return
        try:
            payload = json.loads(line)
            response, should_stop = service.handle(payload)
        except Exception as exc:
            response, should_stop = error_response(exc), False
        print(json.dumps(response, indent=2, ensure_ascii=False), flush=True)
        if should_stop:
            stop_event.set()


def main(argv: list[str] | None = None) -> int:
    """Initialize all residents, serve requests, and close in safety order."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    stop_event = threading.Event()

    def stop_handler(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    service = create_service(args)
    try:
        service.initialize()
        if args.transport == "zmq":
            serve_zmq(service, args.bind, stop_event)
        else:
            serve_stdin(service, stop_event)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
