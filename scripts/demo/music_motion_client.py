#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Send one server-local audio-path request to the resident music service."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4


def build_parser() -> argparse.ArgumentParser:
    """Build the resident music client CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:7011")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--start_sec", type=float, default=0.0)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--duration_sec", type=float, default=10.0)
    duration.add_argument(
        "--full",
        action="store_true",
        help="Generate from start_sec to the end, still subject to server max_frames.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request_id")
    parser.add_argument("--timeout_seconds", type=float, default=60.0)
    parser.add_argument(
        "--metadata_json",
        help="Inline JSON object or path to a UTF-8 JSON object file.",
    )
    return parser


def _parse_metadata(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--metadata_json is invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("--metadata_json must contain a JSON object")
    return metadata


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    """Build a path-only JSON request; audio bytes are never uploaded."""
    if not math.isfinite(args.start_sec) or args.start_sec < 0:
        raise ValueError("--start_sec must be finite and >= 0")
    if not args.full and (not math.isfinite(args.duration_sec) or args.duration_sec <= 0):
        raise ValueError("--duration_sec must be finite and > 0")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("--timeout_seconds must be finite and > 0")
    payload: dict[str, Any] = {
        "request_id": args.request_id or uuid4().hex,
        "audio_path": str(args.audio.expanduser()),
        "start_sec": float(args.start_sec),
        "duration_sec": None if args.full else float(args.duration_sec),
        "seed": int(args.seed),
    }
    metadata = _parse_metadata(args.metadata_json)
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def send_request(
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Send one ZMQ REQ and wait for a JSON reply within the timeout."""
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
            f"music-motion service did not reply within {timeout_seconds:g}s at {endpoint}"
        ) from exc
    finally:
        socket.close(linger=0)
        context.term()


def main(argv: list[str] | None = None) -> int:
    """Send one request and print the formatted response."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_request(args)
        response = send_request(args.endpoint, payload, args.timeout_seconds)
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
