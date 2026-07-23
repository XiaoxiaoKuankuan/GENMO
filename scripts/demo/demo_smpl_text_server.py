# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Run a resident same-GPU T5-3B + GEM-SMPL text-motion service."""

from __future__ import annotations

import argparse
import json
import select
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.runtime.resident_text_motion import ResidentTextMotionEngine  # noqa: E402


@dataclass(frozen=True, slots=True)
class RequestDefaults:
    """Defaults applied only to fields omitted by one request."""

    num_frames: int = 120
    fps: float = 30.0
    seed: int = 42


HELP = {
    "plain_text": "Enter a non-empty prompt using default frames/fps/seed.",
    "json": {
        "request_id": "optional-id",
        "prompt": "A person walks forward.",
        "num_frames": 120,
        "fps": 30,
        "seed": 42,
    },
    "commands": ["/status", "/help", "/clear-cache", "/quit"],
}


def build_parser() -> argparse.ArgumentParser:
    """Build the resident service CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=Path, default=Path("inputs/pretrained/gem_smpl.ckpt"))
    parser.add_argument("--t5_model", default="t5-3b")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--text_encoder_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--output_root", type=Path, default=Path("outputs/text_motion"))
    parser.add_argument("--transport", choices=("stdin", "zmq"), default="stdin")
    parser.add_argument("--bind", default="tcp://127.0.0.1:7010")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--shape_mode", choices=("zero",), default="zero")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--bbox_scale", type=float, default=0.75)
    parser.add_argument("--no_postproc", action="store_true")
    parser.add_argument("--embedding_cache_size", type=int, default=128)
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    warmup.add_argument("--no_warmup", dest="warmup", action="store_false")
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--warmup_prompt", default="A person stands still.")
    parser.add_argument("--min_free_gib", type=float, default=2.0)
    parser.add_argument("--strict_memory", action="store_true")
    parser.add_argument(
        "--latest_file",
        type=Path,
        default=Path("outputs/text_motion/latest_ready.json"),
    )
    parser.add_argument("--max_frames", type=int, default=900)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid fixed service or default-request settings before model loading."""
    if args.num_frames <= 0 or args.num_frames > args.max_frames:
        raise ValueError("--num_frames must be within --max_frames")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.ddim_steps <= 0:
        raise ValueError("--ddim_steps must be > 0")
    if args.guidance_scale < 0:
        raise ValueError("--guidance_scale must be >= 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if not 0.0 < args.bbox_scale <= 1.5:
        raise ValueError("--bbox_scale must satisfy 0 < value <= 1.5")
    if args.embedding_cache_size < 0:
        raise ValueError("--embedding_cache_size must be >= 0")
    if args.warmup_frames <= 0 or args.max_frames <= 0:
        raise ValueError("--warmup_frames and --max_frames must be > 0")
    if args.min_free_gib < 0:
        raise ValueError("--min_free_gib must be >= 0")


def parse_stdin_line(line: str, defaults: RequestDefaults) -> tuple[str, Any] | None:
    """Parse one stdin line as a management command, JSON request, or plain prompt."""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("/"):
        command = stripped.lower()
        if command not in {"/status", "/help", "/clear-cache", "/quit"}:
            raise ValueError(f"unknown command: {stripped}")
        return "command", command
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON request: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON request must be an object")
    else:
        payload = {"prompt": stripped}
    return "request", apply_request_defaults(payload, defaults)


def apply_request_defaults(
    payload: dict[str, Any], defaults: RequestDefaults
) -> dict[str, Any]:
    """Add per-service request defaults without allowing fixed DDIM/model overrides."""
    if not isinstance(payload, dict):
        raise TypeError("request must be a JSON object")
    result = dict(payload)
    result.setdefault("num_frames", defaults.num_frames)
    result.setdefault("fps", defaults.fps)
    result.setdefault("seed", defaults.seed)
    return result


def error_response(exc: Exception, request_id: Any = None) -> dict[str, Any]:
    """Return the common structured protocol error."""
    return {
        "ok": False,
        "request_id": str(request_id) if request_id is not None else None,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def handle_command(
    engine: ResidentTextMotionEngine, command: str
) -> tuple[dict[str, Any], bool]:
    """Execute one management command and indicate whether the service should stop."""
    if command in {"/status", "status"}:
        return {"ok": True, "status": engine.status()}, False
    if command in {"/help", "help"}:
        return {"ok": True, "help": HELP}, False
    if command in {"/clear-cache", "clear-cache"}:
        removed = engine.clear_cache()
        return {"ok": True, "removed_embeddings": removed}, False
    if command in {"/quit", "quit"}:
        return {"ok": True, "message": "service stopping"}, True
    raise ValueError(f"unknown command: {command}")


def handle_json_message(
    engine: ResidentTextMotionEngine,
    payload: Any,
    defaults: RequestDefaults,
) -> tuple[dict[str, Any], bool]:
    """Handle one ZMQ JSON object without allowing an error to stop the service."""
    request_id = payload.get("request_id") if isinstance(payload, dict) else None
    try:
        if not isinstance(payload, dict):
            raise TypeError("request must be a JSON object")
        command = payload.get("command")
        if command is not None:
            if not isinstance(command, str):
                raise TypeError("command must be a string")
            return handle_command(engine, command.strip().lower().lstrip("/"))
        request = apply_request_defaults(payload, defaults)
        return engine.generate(request), False
    except Exception as exc:
        return error_response(exc, request_id), False


def serve_stdin(
    engine: ResidentTextMotionEngine,
    defaults: RequestDefaults,
    stop_event: threading.Event,
    *,
    input_fn=input,
    output_fn=print,
) -> None:
    """Serve prompts interactively while keeping request failures recoverable."""
    output_fn(json.dumps(HELP, indent=2, ensure_ascii=False))
    while not stop_event.is_set():
        try:
            if input_fn is input:
                output_fn("text-motion> ", end="", flush=True)
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
                line = input_fn("text-motion> ")
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
    engine: ResidentTextMotionEngine,
    defaults: RequestDefaults,
    bind: str,
    stop_event: threading.Event,
) -> None:
    """Serve sequential JSON REP requests without concurrent GPU inference."""
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(bind)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print(f"[Resident] ZMQ REP listening on {bind}")
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


def create_engine(args: argparse.Namespace) -> ResidentTextMotionEngine:
    """Create the engine from fixed startup arguments."""
    return ResidentTextMotionEngine(
        ckpt_path=args.ckpt_path,
        t5_model=args.t5_model,
        device=args.device,
        text_dtype=args.text_encoder_dtype,
        local_files_only=args.local_files_only,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        bbox_scale=args.bbox_scale,
        output_root=args.output_root,
        postproc=not args.no_postproc,
        shape_mode=args.shape_mode,
        embedding_cache_size=args.embedding_cache_size,
        min_free_gib=args.min_free_gib,
        warmup_frames=args.warmup_frames,
        warmup_prompt=args.warmup_prompt,
        warmup_enabled=args.warmup,
        strict_memory=args.strict_memory,
        latest_file=args.latest_file,
        max_frames=args.max_frames,
    )


def main(argv: list[str] | None = None) -> int:
    """Initialize once, serve until shutdown, then release both GPU models."""
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
    defaults = RequestDefaults(args.num_frames, args.fps, args.seed)
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
