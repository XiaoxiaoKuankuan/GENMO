#!/usr/bin/env python3
"""计算 BUMI qpos30 表示的正式 train-split 统计量。

工具按 120 帧训练窗口和确定性 stride 遍历一个或多个 BUMI train manifest，只累计有效
qpos 帧编码后的 30 维特征；短序列 padding 不参与统计，长序列最后一个合法窗口必定纳入。
输出 JSON 同时绑定表示版本、feature slices、运动学 SHA256、各数据集 info/manifest 指纹
和统计范围。Endecoder 会逐项验证这些字段，所以旧 93D stats、错误运动学或不同数据版本
都不能被静默复用。

JSON 保存真实标准差（最小仅防除零到 1e-6），训练归一化再按配置把 std 下限裁到 0.01。
这保留 main 防止极小方差爆炸的思想，但不会把实际约 0.006 m/帧的 BUMI 根位移缩小百倍。
"""

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

from gem.datasets.music_dance.music_dance_bumi import BumiMusicDatasetReader  # noqa: E402
from gem.robots.bumi.endecoder import STATS_CONTRACT_VERSION  # noqa: E402
from gem.robots.bumi.feature_codec import (  # noqa: E402
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
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


def parse_named_tolerance(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--dataset-joint-limit-tolerance must use DATASET_NAME=RADIANS"
        )
    name, raw = value.split("=", 1)
    try:
        tolerance = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid joint tolerance {raw!r}") from exc
    if not name or not torch.isfinite(torch.tensor(tolerance)) or tolerance < 0.0:
        raise argparse.ArgumentTypeError("dataset tolerance requires a name and finite value >= 0")
    return name, tolerance


def deterministic_window_starts(
    sequence_length: int,
    window: int = 120,
    stride: int = 120,
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
        help="reader 允许的关节限位越界弧度；必须与目标数据版本策略一致",
    )
    parser.add_argument(
        "--dataset-joint-limit-tolerance",
        action="append",
        default=[],
        type=parse_named_tolerance,
        help="按库覆盖 reader 容差，例如 mine_bumi=0.25；其余沿用全局值",
    )
    args = parser.parse_args()
    if args.motion_frames != 120:
        raise ValueError("Formal BUMI music stats require --motion-frames=120")
    kinematics = BumiKinematics(args.kinematics)
    codec = BumiMotionFeatureCodec(kinematics)
    dataset_names = [name for name, _root in args.dataset]
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError(f"duplicate --dataset names: {dataset_names}")
    tolerance_overrides = dict(args.dataset_joint_limit_tolerance)
    if len(tolerance_overrides) != len(args.dataset_joint_limit_tolerance):
        raise ValueError("duplicate --dataset-joint-limit-tolerance names")
    unknown_tolerances = set(tolerance_overrides) - set(dataset_names)
    if unknown_tolerances:
        raise ValueError(
            f"joint tolerance overrides reference unknown datasets: {sorted(unknown_tolerances)}"
        )
    accumulator = StreamingWelford(BUMI_FEATURE_DIM)
    fingerprints: dict[str, dict[str, str | int]] = {}
    window_count = 0
    for dataset_name, root in args.dataset:
        joint_limit_tolerance = tolerance_overrides.get(
            dataset_name, float(args.joint_limit_tolerance)
        )
        reader = BumiMusicDatasetReader(
            root,
            dataset_name,
            "train",
            kinematics,
            strict_alignment=True,
            strict_contract=True,
            require_quality_filter=True,
            joint_limit_tolerance=joint_limit_tolerance,
            validate_payloads_on_init=False,
        )
        dataset_windows = 0
        for row in reader.rows:
            qpos = reader.load_aligned_sequence(row)["qpos"]
            starts = deterministic_window_starts(
                len(qpos), window=args.motion_frames, stride=args.stride
            )
            for start in starts:
                crop = qpos[start : start + args.motion_frames]
                accumulator.update(codec.encode(crop).physical_features)
                window_count += 1
                dataset_windows += 1
        fingerprints[dataset_name] = {
            "dataset_info_sha256": sha256_file(reader.dataset_info_path),
            "train_manifest_sha256": sha256_file(reader.manifest_path),
            "sequences": len(reader.rows),
            "windows": dataset_windows,
            "reader_joint_limit_tolerance_rad": joint_limit_tolerance,
        }
    mean, std = accumulator.finalize()
    report = {
        "contract_version": STATS_CONTRACT_VERSION,
        "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "robot_name": "bumi",
        "feature_dim": BUMI_FEATURE_DIM,
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
        "stored_std_minimum": 1.0e-6,
        "training_clip_std_min": 0.01,
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
