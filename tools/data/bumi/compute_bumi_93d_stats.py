#!/usr/bin/env python3
"""Compute train-split BUMI 93D stats over deterministic crop-aligned windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.music_dance.music_dance_bumi import (  # noqa: E402
    BumiMusicDatasetReader,
)
from gem.robots.bumi.feature_codec import (  # noqa: E402
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_SLICES,
    BumiMotionFeatureCodec,
)
from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must use DATASET_NAME=/absolute/root")
    name, root = value.split("=", 1)
    if not name or not root:
        raise argparse.ArgumentTypeError("--dataset requires non-empty name and root")
    return name, Path(root).expanduser().resolve()


def deterministic_window_starts(
    sequence_length: int, window: int = 120, stride: int = 120
) -> list[int]:
    if sequence_length <= 0 or window <= 0 or stride <= 0:
        raise ValueError("sequence_length, window, and stride must be positive")
    if sequence_length <= window:
        return [0]
    last_start = sequence_length - window
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


class StreamingWelford:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.m2 = torch.zeros(dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().cpu().to(dtype=torch.float64).reshape(-1, self.mean.numel())
        if values.shape[0] == 0:
            return
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Non-finite BUMI feature encountered during stats computation")
        batch_count = int(values.shape[0])
        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + delta.square() * (self.count * batch_count / total)
        self.count = total

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0:
            raise ValueError("No BUMI feature frames were accumulated")
        variance = self.m2 / self.count
        return self.mean.float(), variance.clamp_min(0.0).sqrt().float().clamp_min(1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--dataset", action="append", required=True, type=parse_dataset)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--motion-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument(
        "--joint-limit-tolerance",
        type=float,
        default=1.0e-3,
        help="reader 允许的关节限位越界弧度；必须与目标数据版本绑定的策略一致",
    )
    args = parser.parse_args()
    if args.motion_frames != 120:
        raise ValueError("Formal BUMI music stats require --motion-frames=120")
    kinematics = BumiKinematics(args.kinematics)
    codec = BumiMotionFeatureCodec(kinematics)
    accumulator = StreamingWelford(93)
    fingerprints: dict[str, dict[str, str | int]] = {}
    window_count = 0
    for dataset_name, root in args.dataset:
        reader = BumiMusicDatasetReader(
            root,
            dataset_name,
            "train",
            kinematics,
            strict_alignment=True,
            strict_contract=True,
            require_quality_filter=True,
            joint_limit_tolerance=args.joint_limit_tolerance,
            validate_payloads_on_init=False,
        )
        dataset_windows = 0
        for row in reader.rows:
            sequence = reader.load_aligned_sequence(row)
            qpos = sequence["qpos"]
            starts = deterministic_window_starts(
                len(qpos), window=args.motion_frames, stride=args.stride
            )
            for start in starts:
                crop = qpos[start : start + args.motion_frames]
                # Short sequences contribute only real frames. Training padding
                # is invalid-masked and must not alter mean/std.
                physical = codec.encode(crop).physical_features
                accumulator.update(physical)
                window_count += 1
                dataset_windows += 1
        fingerprints[dataset_name] = {
            "dataset_info_sha256": sha256_file(reader.dataset_info_path),
            "train_manifest_sha256": sha256_file(reader.manifest_path),
            "sequences": len(reader.rows),
            "windows": dataset_windows,
        }
    mean, std = accumulator.finalize()
    report = {
        "contract_version": "genmo.bumi_stats.v1",
        "robot_name": "bumi",
        "feature_dim": 93,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_slices": {key: list(value) for key, value in BUMI_FEATURE_SLICES.items()},
        "joint_names": list(kinematics.joint_order),
        "anchor_mode": BUMI_ANCHOR_MODE,
        "quaternion_convention": "wxyz",
        "kinematics_sha256": kinematics.kinematics_sha256,
        "dataset_fingerprints": fingerprints,
        "motion_frames": 120,
        "stride": int(args.stride),
        "last_legal_window_included": True,
        "num_windows": window_count,
        "num_feature_frames": accumulator.count,
        "std_minimum": 1.0e-6,
        "is_placeholder": False,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "windows": window_count, "frames": accumulator.count}, indent=2
        )
    )


if __name__ == "__main__":
    main()
