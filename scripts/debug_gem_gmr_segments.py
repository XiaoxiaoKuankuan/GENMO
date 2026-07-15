#!/usr/bin/env python3
"""Record the GEM segment-adapter debug side channel to NPZ and JSON."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from gem.gmr_segment_adapter import SEGMENT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7002)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/gmr_segments_debug.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0")

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind((args.bind, args.port))
    receiver.settimeout(0.25)
    frames = []
    deadline = time.monotonic() + args.duration
    print(
        f"Recording GEM segment diagnostics on {args.bind}:{args.port} for {args.duration:.1f}s ..."
    )
    try:
        while time.monotonic() < deadline:
            try:
                payload, _ = receiver.recvfrom(65535)
            except TimeoutError:
                continue
            try:
                frame = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if set(frame.get("scaled_segments", {})) != set(SEGMENT_NAMES):
                continue
            if len(frame.get("raw_joints", [])) != 22:
                continue
            frames.append(frame)
    finally:
        receiver.close()

    if not frames:
        raise RuntimeError("no debug frames received; run demo_webcam.py with --gmr_debug_skeleton")

    def segment_array(key: str, field: str) -> np.ndarray:
        return np.asarray(
            [[frame[key][name][field] for name in SEGMENT_NAMES] for frame in frames],
            dtype=np.float64,
        )

    packet_lengths = {len(frame.get("udp_payload", [])) for frame in frames}
    if packet_lengths != {412}:
        raise RuntimeError(f"unexpected UDP payload lengths: {sorted(packet_lengths)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frame_id=np.asarray([frame["frame_id"] for frame in frames], dtype=np.int64),
        timestamp_ns=np.asarray([frame["timestamp_ns"] for frame in frames], dtype=np.uint64),
        ground_z=np.asarray([frame["ground_z"] for frame in frames]),
        raw_joints=np.asarray([frame["raw_joints"] for frame in frames]),
        raw_segment_origins=segment_array("raw_segments", "position"),
        raw_segment_rotations=segment_array("raw_segments", "rotation"),
        scaled_segment_origins=segment_array("scaled_segments", "position"),
        scaled_segment_rotations=segment_array("scaled_segments", "rotation"),
        udp_payload=np.asarray([frame["udp_payload"] for frame in frames], dtype=np.uint8),
        segment_names=np.asarray(SEGMENT_NAMES),
    )
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {"segment_order": SEGMENT_NAMES, "frame_count": len(frames), "frames": frames},
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output} and {json_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
