#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Run a resident GEM-SMPL music-to-motion service."""

from __future__ import annotations

import argparse
import json
import math
import select
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.runtime.resident_music_motion import ResidentMusicMotionEngine  # noqa: E402


@dataclass(frozen=True, slots=True)
class RequestDefaults:
    """Defaults applied to fields omitted by one music request."""

    start_sec: float = 0.0
    duration_sec: float | None = 10.0
    seed: int = 42


HELP = {
    "plain_path": "Enter one server-local WAV, MP3, or FLAC path.",
    "json": {
        "request_id": "music-001",
        "audio_path": "/path/on/server/song.mp3",
        "start_sec": 15,
        "duration_sec": 10,
        "seed": 7,
    },
    "commands": ["/status", "/help", "/clear-cache", "/quit"],
}


def build_parser() -> argparse.ArgumentParser:
    """Build the resident music service command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("inputs/pretrained/gem_smpl.ckpt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", type=Path, default=Path("outputs/music_motion"))
    parser.add_argument("--transport", choices=("stdin", "zmq"), default="stdin")
    parser.add_argument("--bind", default="tcp://127.0.0.1:7011")
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--shape_mode", choices=("zero",), default="zero")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--focal", type=float)
    parser.add_argument("--no_postproc", action="store_true")
    parser.add_argument("--feature_cache_size", type=int, default=32)
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    warmup.add_argument("--no_warmup", dest="warmup", action="store_false")
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=600)
    parser.add_argument("--min_free_gib", type=float, default=2.0)
    parser.add_argument("--strict_memory", action="store_true")
    parser.add_argument("--latest_file", type=Path)
    parser.add_argument(
        "--allowed_audio_root",
        action="append",
        default=[],
        type=Path,
        help="Repeat to allow only resolved server-local audio paths under these roots.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid fixed-service and default-request settings."""
    if not math.isfinite(args.start_sec) or args.start_sec < 0:
        raise ValueError("--start_sec must be finite and >= 0")
    if not math.isfinite(args.duration_sec) or args.duration_sec <= 0:
        raise ValueError("--duration_sec must be finite and > 0")
    if args.ddim_steps <= 0:
        raise ValueError("--ddim_steps must be > 0")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        raise ValueError("--guidance_scale must be finite and >= 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.focal is not None and (not math.isfinite(args.focal) or args.focal <= 0):
        raise ValueError("--focal must be finite and > 0")
    if args.feature_cache_size < 0:
        raise ValueError("--feature_cache_size must be >= 0")
    if args.warmup_frames <= 0 or args.max_frames <= 0:
        raise ValueError("--warmup_frames and --max_frames must be > 0")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        raise ValueError("--min_free_gib must be finite and >= 0")


def _strip_matching_quotes(value: str) -> str:
    """Remove one matching pair of surrounding quotes from an stdin path."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def apply_request_defaults(
    payload: dict[str, Any],
    defaults: RequestDefaults,
) -> dict[str, Any]:
    """Apply audio-range defaults without exposing fixed model/DDIM settings."""
    if not isinstance(payload, dict):
        raise TypeError("request must be a JSON object")
    result = dict(payload)
    result.setdefault("start_sec", defaults.start_sec)
    result.setdefault("duration_sec", defaults.duration_sec)
    result.setdefault("seed", defaults.seed)
    return result


def parse_stdin_line(
    line: str,
    defaults: RequestDefaults,
) -> tuple[str, Any] | None:
    """Parse a command, JSON request, or one complete server-local audio path."""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.lower() in {"/status", "/help", "/clear-cache", "/quit"}:
        return "command", stripped.lower()
    if stripped.startswith("/") and Path(stripped).suffix.lower() not in {
        ".wav",
        ".mp3",
        ".flac",
    }:
        command = stripped.lower()
        raise ValueError(f"unknown command: {command}")
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON request: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON request must be an object")
    else:
        audio_path = _strip_matching_quotes(stripped)
        if not audio_path:
            raise ValueError("audio path must not be empty")
        payload = {"audio_path": audio_path}
    return "request", apply_request_defaults(payload, defaults)


def error_response(
    exc: Exception,
    request_id: Any = None,
    *,
    total_seconds: float = 0.0,
) -> dict[str, Any]:
    """Return the common structured protocol error."""
    return {
        "ok": False,
        "request_id": str(request_id) if request_id is not None else None,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "total_seconds": float(total_seconds),
    }


def handle_command(
    engine: ResidentMusicMotionEngine,
    command: str,
) -> tuple[dict[str, Any], bool]:
    """Execute one management command and return the stop decision."""
    normalized = command.strip().lower().lstrip("/")
    if normalized == "status":
        return {"ok": True, "status": engine.status()}, False
    if normalized == "help":
        return {"ok": True, "help": HELP}, False
    if normalized == "clear-cache":
        removed = engine.clear_cache()
        return {"ok": True, "removed_features": removed}, False
    if normalized == "quit":
        return {"ok": True, "message": "service stopping"}, True
    raise ValueError(f"unknown command: {command}")


def handle_json_message(
    engine: ResidentMusicMotionEngine,
    payload: Any,
    defaults: RequestDefaults,
) -> tuple[dict[str, Any], bool]:
    """Handle one ZMQ object while keeping request errors recoverable."""
    started = time.perf_counter()
    request_id = payload.get("request_id") if isinstance(payload, dict) else None
    try:
        if not isinstance(payload, dict):
            raise TypeError("request must be a JSON object")
        command = payload.get("command")
        if command is not None:
            if not isinstance(command, str):
                raise TypeError("command must be a string")
            return handle_command(engine, command)
        return engine.generate(apply_request_defaults(payload, defaults)), False
    except Exception as exc:
        return error_response(
            exc,
            request_id,
            total_seconds=time.perf_counter() - started,
        ), False


def serve_stdin(
    engine: ResidentMusicMotionEngine,
    defaults: RequestDefaults,
    stop_event: threading.Event,
    *,
    input_fn=input,
    output_fn=print,
) -> None:
    """Serve sequential stdin requests until EOF, signal, or ``/quit``."""
    output_fn(json.dumps(HELP, indent=2, ensure_ascii=False))
    while not stop_event.is_set():
        try:
            if input_fn is input:
                output_fn("music-motion> ", end="", flush=True)
                while not stop_event.is_set():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if readable:
                        break
                if stop_event.is_set():
                    break
                line = sys.stdin.readline()
                if line == "":
                    break
            else:
                line = input_fn("music-motion> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            stop_event.set()
            break
        try:
            parsed = parse_stdin_line(line, defaults)
            if parsed is None:
                continue
            kind, value = parsed
            if kind == "command":
                response, should_stop = handle_command(engine, value)
            else:
                response, should_stop = engine.generate(value), False
        except Exception as exc:
            response, should_stop = error_response(exc), False
        output_fn(json.dumps(response, indent=2, ensure_ascii=False))
        if should_stop:
            stop_event.set()


def serve_zmq(
    engine: ResidentMusicMotionEngine,
    defaults: RequestDefaults,
    bind: str,
    stop_event: threading.Event,
) -> None:
    """Serve sequential JSON REP requests without concurrent GEM inference."""
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(bind)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print(f"[ResidentMusic] ZMQ REP listening on {bind}")
    try:
        while not stop_event.is_set():
            if socket not in dict(poller.poll(200)):
                continue
            try:
                payload = socket.recv_json()
                response, should_stop = handle_json_message(engine, payload, defaults)
            except Exception as exc:
                response, should_stop = error_response(exc), False
            socket.send_json(response)
            if should_stop:
                stop_event.set()
    finally:
        poller.unregister(socket)
        socket.close(linger=0)
        context.term()


def create_engine(args: argparse.Namespace) -> ResidentMusicMotionEngine:
    """Create the engine from immutable startup arguments."""
    return ResidentMusicMotionEngine(
        ckpt_path=args.ckpt_path,
        device=args.device,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        focal=args.focal,
        output_root=args.output_root,
        postproc=not args.no_postproc,
        shape_mode=args.shape_mode,
        feature_cache_size=args.feature_cache_size,
        min_free_gib=args.min_free_gib,
        strict_memory=args.strict_memory,
        warmup_frames=args.warmup_frames,
        warmup_enabled=args.warmup,
        latest_file=args.latest_file,
        max_frames=args.max_frames,
        allowed_audio_roots=args.allowed_audio_root,
    )


def main(argv: list[str] | None = None) -> int:
    """Initialize once, serve until shutdown, and release GEM once."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    stop_event = threading.Event()

    def stop_handler(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    engine = create_engine(args)
    defaults = RequestDefaults(args.start_sec, args.duration_sec, args.seed)
    try:
        engine.initialize()
        if args.transport == "stdin":
            serve_stdin(engine, defaults, stop_event)
        else:
            serve_zmq(engine, defaults, args.bind, stop_event)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
