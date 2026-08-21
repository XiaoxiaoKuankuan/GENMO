#!/usr/bin/env python3
"""Run the strict genmo.bumi_music.v1 contract scan for one or more splits."""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--joint-limit-tolerance",
        type=float,
        default=1.0e-3,
        help="reader 允许的关节限位越界弧度；必须与目标数据版本绑定的策略一致",
    )
    args = parser.parse_args()
    kinematics = BumiKinematics(args.kinematics)
    split_reports = []
    for split in args.splits:
        reader = BumiMusicDatasetReader(
            args.root,
            args.dataset_name,
            split,
            kinematics,
            strict_alignment=True,
            strict_contract=True,
            require_quality_filter=True,
            joint_limit_tolerance=args.joint_limit_tolerance,
            validate_payloads_on_init=True,
            validate_source_hashes_on_init=True,
        )
        frames = sum(int(row["num_frames"]) for row in reader.rows)
        split_reports.append(
            {
                "split": split,
                "sequences": len(reader.rows),
                "frames": frames,
                "hours": frames / 30.0 / 3600.0,
            }
        )
    report = {
        "status": "passed",
        "contract_version": "genmo.bumi_music.v1",
        "dataset_name": args.dataset_name,
        "root": str(args.root.expanduser().resolve()),
        "kinematics_sha256": kinematics.kinematics_sha256,
        "splits": split_reports,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
