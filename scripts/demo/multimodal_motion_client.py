#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Send one strictly validated request to the unified multimodal service."""

from __future__ import annotations

import argparse
import json
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build mutually exclusive generation and control actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:7020")
    parser.add_argument("--timeout_ms", type=int, default=300000)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--mode", choices=("text", "music", "text_music"))
    action.add_argument("--video_start", action="store_true")
    action.add_argument("--video_stop", action="store_true")
    action.add_argument("--idle", action="store_true")
    action.add_argument("--estop", action="store_true")
    action.add_argument("--clear_estop", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--clear_cache", action="store_true")
    action.add_argument("--shutdown", action="store_true")
    parser.add_argument("--request_id")
    parser.add_argument("--prompt")
    parser.add_argument("--audio")
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera_id", type=int)
    source.add_argument("--video_path")
    parser.add_argument("--cache_target", choices=("all", "text", "music"), default="all")
    return parser


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    """Convert validated CLI arguments to the exact JSON protocol."""
    if args.timeout_ms <= 0:
        raise ValueError("--timeout_ms must be > 0")
    if args.mode is not None:
        if args.mode in {"text", "text_music"}:
            if not isinstance(args.prompt, str) or not args.prompt.strip():
                raise ValueError(f"--mode {args.mode} requires --prompt")
        elif args.prompt is not None:
            raise ValueError("--mode music does not accept --prompt")
        if args.mode in {"music", "text_music"}:
            if not isinstance(args.audio, str) or not args.audio.strip():
                raise ValueError(f"--mode {args.mode} requires --audio")
            if args.start_sec < 0:
                raise ValueError("--start_sec must be >= 0")
        elif args.audio is not None:
            raise ValueError("--mode text does not accept --audio")
        if args.camera_id is not None or args.video_path is not None:
            raise ValueError("generation modes do not accept video source arguments")
        request: dict[str, Any] = {
            "op": "generate",
            "mode": args.mode,
            "seed": args.seed,
        }
        if args.request_id is not None:
            request["request_id"] = args.request_id
        if args.mode in {"text", "text_music"}:
            request["prompt"] = args.prompt.strip()
        if args.mode in {"music", "text_music"}:
            request["audio_path"] = args.audio
            request["start_sec"] = args.start_sec
        return request

    if args.prompt is not None or args.audio is not None:
        raise ValueError("--prompt/--audio are valid only with --mode")
    if args.video_start:
        if (args.camera_id is None) == (args.video_path is None):
            raise ValueError("--video_start requires exactly one of --camera_id or --video_path")
        request = {"op": "video_start"}
        if args.camera_id is not None:
            if args.camera_id < 0:
                raise ValueError("--camera_id must be >= 0")
            request["camera_id"] = args.camera_id
        else:
            request["video_path"] = args.video_path
        return request
    if args.camera_id is not None or args.video_path is not None:
        raise ValueError("video source arguments require --video_start")
    if args.video_stop:
        return {"op": "video_stop"}
    if args.idle:
        return {"op": "idle"}
    if args.estop:
        return {"op": "estop"}
    if args.clear_estop:
        return {"op": "clear_estop"}
    if args.status:
        return {"op": "status"}
    if args.clear_cache:
        return {"op": "clear_cache", "target": args.cache_target}
    if args.shutdown:
        return {"op": "shutdown"}
    raise AssertionError("argparse accepted no action")


def main(argv: list[str] | None = None) -> int:
    """Send one REQ/REP exchange and print the complete response."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = build_request(args)
    except ValueError as exc:
        parser.error(str(exc))

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(args.endpoint)
    try:
        socket.send_json(request)
        response = socket.recv_json()
    except zmq.Again:
        parser.error(f"request timed out after {args.timeout_ms} ms: {args.endpoint}")
    finally:
        socket.close(linger=0)
        context.term()
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0 if response.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
