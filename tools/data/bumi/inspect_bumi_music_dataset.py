#!/usr/bin/env python3
"""Inspect BUMI dataset metadata/manifests without changing any artifact."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.music_dance.music_dance_bumi import (  # noqa: E402
    BumiMusicDatasetReader,
)
from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument(
        "--load-payloads",
        action="store_true",
        help="Also perform the full motion/music payload scan (off by default).",
    )
    args = parser.parse_args()
    kinematics = BumiKinematics(args.kinematics)
    reader = BumiMusicDatasetReader(
        args.root,
        args.dataset_name,
        args.split,
        kinematics,
        validate_payloads_on_init=args.load_payloads,
    )
    frames = [int(row["num_frames"]) for row in reader.rows]
    report = {
        "root": str(reader.root),
        "contract_version": reader.dataset_info["contract_version"],
        "dataset_name": args.dataset_name,
        "split": args.split,
        "sequences": len(reader.rows),
        "frames": sum(frames),
        "hours": sum(frames) / 30.0 / 3600.0,
        "min_frames": min(frames),
        "max_frames": max(frames),
        "manifest_datasets": dict(Counter(str(row["dataset"]) for row in reader.rows)),
        "kinematics_sha256": kinematics.kinematics_sha256,
        "payload_scan_executed": bool(args.load_payloads),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
