#!/usr/bin/env python3
"""显式计算 closed-loop Stage 1 proprio48 的 train-split 有效数据统计量。

该工具只读取用户逐项传入的正式 BUMI 音乐数据根，并强制使用每个根的 ``train`` manifest。
它复用 ``BumiMusicDatasetReader`` 的数据/运动学/质量契约，再调用 Stage 1 的因果历史构造器，
仅累计 50 Hz 时间线上 ``valid=true`` 的原始物理 proprio48；序列开头缺少后向速度的样本、
padding、val 和 test 均不会进入统计。输出记录字段顺序、GMT 名义 default、关节顺序、数据
manifest 指纹和构造版本，供后续 GENMO condition normalizer 使用，绝不裁切或复用 GMT
policy 69 维 normalizer。

本文件只是手动入口：导入模块或构造 Dataset 不会自动执行统计，也不会生成默认均值/方差。
为防止覆盖正式资产，``--output`` 已存在时工具直接失败；调用方应选择一个新的明确路径。
本工具不修改现有 qpos30 stats，不启动训练、GMT/Isaac Lab 或任何全量数据转换。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.closedloop.contracts import (  # noqa: E402
    GMT_EXPECTED_JOINT_ORDER,
    GMT_NOMINAL_DEFAULT_JOINT_POS_RAD,
    PROPRIO_CONTRACT_VERSION,
    PROPRIO_DIM,
    PROPRIO_FIELD_SPECS,
    PROPRIO_FPS,
    PROPRIO_SLICES,
)
from gem.closedloop.stage1_dataset import (  # noqa: E402
    CAUSAL_PROPRIO_CONSTRUCTION_VERSION,
    CausalDemoProprio48Builder,
)
from gem.datasets.music_dance.music_dance_bumi import (  # noqa: E402
    BumiMusicDatasetReader,
    sha256_file,
)
from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402

PROPRIO_STATS_CONTRACT_VERSION = "genmo.bumi_proprio48_stats.v1"


def parse_dataset(value: str) -> tuple[str, Path]:
    """解析 ``DATASET_NAME=/absolute/root``，不推断或扫描数据位置。"""

    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must use DATASET_NAME=/absolute/root")
    name, raw_root = value.split("=", 1)
    expanded_root = Path(raw_root).expanduser()
    if not name or not raw_root or not expanded_root.is_absolute():
        raise argparse.ArgumentTypeError("--dataset requires a name and an absolute root")
    root = expanded_root.resolve()
    return name, root


class StreamingWelford:
    """以 float64 合并批次矩，避免把全部 50 Hz 帧保存在内存。"""

    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.m2 = torch.zeros(dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        batch = values.detach().cpu().to(torch.float64).reshape(-1, self.mean.numel())
        if batch.shape[0] == 0:
            return
        if not bool(torch.isfinite(batch).all()):
            raise ValueError("non-finite proprio48 value encountered")
        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
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
            raise ValueError("no valid train-split proprio48 samples were accumulated")
        variance = (self.m2 / self.count).clamp_min(0.0)
        return self.mean.float(), variance.sqrt().float().clamp_min(1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--dataset", action="append", required=True, type=parse_dataset)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--joint-limit-tolerance",
        type=float,
        default=1.0e-4,
        help="reader 的正式数据关节限位容差，单位 rad",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing proprio48 stats: {output}")
    names = [name for name, _root in args.dataset]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate --dataset names: {names}")
    if not torch.isfinite(torch.tensor(args.joint_limit_tolerance)) or (
        args.joint_limit_tolerance < 0.0
    ):
        raise ValueError("joint-limit-tolerance must be finite and non-negative")

    kinematics = BumiKinematics(args.kinematics.expanduser().resolve())
    builder = CausalDemoProprio48Builder(kinematics)
    accumulator = StreamingWelford(PROPRIO_DIM)
    fingerprints: dict[str, dict[str, str | int | float]] = {}
    total_source_frames = 0
    total_50hz_samples = 0
    for dataset_name, root in args.dataset:
        reader = BumiMusicDatasetReader(
            root,
            dataset_name,
            "train",
            kinematics,
            strict_alignment=True,
            strict_contract=True,
            require_quality_filter=True,
            joint_limit_tolerance=float(args.joint_limit_tolerance),
            validate_payloads_on_init=False,
        )
        dataset_source_frames = 0
        dataset_valid_50hz = 0
        for row in reader.rows:
            qpos = reader.load_aligned_sequence(row)["qpos"]
            values, valid, _times = builder.full_causal_50hz_timeline(qpos)
            accumulator.update(values[valid])
            dataset_source_frames += len(qpos)
            dataset_valid_50hz += int(valid.sum())
        total_source_frames += dataset_source_frames
        total_50hz_samples += dataset_valid_50hz
        fingerprints[dataset_name] = {
            "dataset_info_sha256": sha256_file(reader.dataset_info_path),
            "train_manifest_sha256": sha256_file(reader.manifest_path),
            "sequences": len(reader.rows),
            "source_frames_30hz": dataset_source_frames,
            "valid_samples_50hz": dataset_valid_50hz,
            "reader_joint_limit_tolerance_rad": float(args.joint_limit_tolerance),
        }

    mean, std = accumulator.finalize()
    if accumulator.count != total_50hz_samples:
        raise RuntimeError("internal proprio48 stats count mismatch")
    report = {
        "contract_version": PROPRIO_STATS_CONTRACT_VERSION,
        "proprio_contract_version": PROPRIO_CONTRACT_VERSION,
        "construction_version": CAUSAL_PROPRIO_CONSTRUCTION_VERSION,
        "split": "train",
        "feature_dim": PROPRIO_DIM,
        "fps": PROPRIO_FPS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "stored_std_minimum": 1.0e-6,
        "is_placeholder": False,
        "feature_slices": {name: list(value) for name, value in PROPRIO_SLICES.items()},
        "field_semantics": [
            {
                "name": field.name,
                "slice": [field.start, field.stop],
                "unit": field.unit,
                "coordinate_frame": field.coordinate_frame,
                "raw_semantics": field.raw_semantics,
            }
            for field in PROPRIO_FIELD_SPECS
        ],
        "joint_names": list(GMT_EXPECTED_JOINT_ORDER),
        "nominal_default_joint_pos_rad": list(GMT_NOMINAL_DEFAULT_JOINT_POS_RAD),
        "default_pose_scope": "demo_proxy_uses_gmt_nominal_without_startup_randomization",
        "resampling": "latest_available_30hz_source_observation_hold_on_50hz_grid",
        "velocity": "30hz_backward_difference_then_hold_no_future_sample",
        "normalizer_scope": "future_genmo_condition_only_not_gmt_policy69",
        "kinematics_sha256": kinematics.kinematics_sha256,
        "dataset_fingerprints": fingerprints,
        "num_source_frames_30hz": total_source_frames,
        "num_valid_samples_50hz": total_50hz_samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "split": "train",
                "valid_samples_50hz": total_50hz_samples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
