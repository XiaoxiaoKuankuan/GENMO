# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Send one JSON text-motion request to the resident ZMQ service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4


def build_parser() -> argparse.ArgumentParser:
    """Build the client CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:7010")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt_file", type=Path)
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request_id")
    parser.add_argument("--timeout_seconds", type=float, default=30.0)
    return parser


def resolve_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    """Resolve one non-empty inline or UTF-8 file prompt."""
    if prompt_file is not None:
        prompt = prompt_file.read_text(encoding="utf-8")
    value = (prompt or "").strip()
    if not value:
        raise ValueError("prompt must not be empty")
    return value


def build_request(args: argparse.Namespace) -> dict:
    """Build the UTF-8 JSON request payload."""
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout_seconds must be > 0")
    return {
        "request_id": args.request_id or uuid4().hex,
        "prompt": resolve_prompt(args.prompt, args.prompt_file),
        "num_frames": args.num_frames,
        "fps": args.fps,
        "seed": args.seed,
    }


def send_request(
    endpoint: str, payload: dict, timeout_seconds: float
) -> dict:
    """Send one REQ and wait up to the configured timeout for its JSON reply."""
    import zmq

    timeout_ms = max(1, round(timeout_seconds * 1000))
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.connect(endpoint)
    try:
        socket.send_json(payload)
        return socket.recv_json()
    except zmq.Again as exc:
        raise TimeoutError(
            f"text-motion service did not reply within {timeout_seconds:g}s at {endpoint}"
        ) from exc
    finally:
        socket.close(linger=0)
        context.term()


def main(argv: list[str] | None = None) -> int:
    """Send and print one formatted response."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = build_request(args)
        response = send_request(args.endpoint, request, args.timeout_seconds)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
