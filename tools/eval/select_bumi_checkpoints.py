#!/usr/bin/env python3
"""用固定评测套件批量评估、排序并选择 BUMI GENMO checkpoint。

“多 checkpoint 选优”要求每个候选模型使用完全相同的音乐片段、目标轨迹（若有）、
CFG、DDIM 步数和逐样本 seed。脚本逐一调用正式 BUMI demo，保留每个样本的 JSON 指标，
并汇总关节限位、穿地/滑脚、根倾角、速度/加速度/jerk、节拍对齐以及可选目标误差。
不同量纲先在候选集合内做同向 min-max 归一化，再按固定权重求综合分；先过硬门槛、
再按分数和 global step 排序，避免把“最新”或“训练 loss 最低”误当成部署最优。

套件是版本化 JSON，样本可引用 WAV 或预计算 EDGE35。默认仅保留真正用于决策的报告，
中间 motion ``.pt`` 在读取指标后自动删除；传入 ``--keep-motion-artifacts`` 才保留轨迹。
最终 ``selection.json`` 记录所有命令、原始/归一化指标、拒绝原因和 best checkpoint SHA，
``best_checkpoint.txt`` 仅写最优模型绝对路径，便于后续 ONNX 导出脚本直接读取。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

SUITE_CONTRACT = "genmo.bumi_checkpoint_suite.v1"
SELECTION_CONTRACT = "genmo.bumi_checkpoint_selection.qpos30_contact.v2"

LOWER_WEIGHTS = {
    "joint_limit_violation_rate": 4.0,
    "foot_penetration_max_m": 2.0,
    "foot_penetration_mean_m": 1.0,
    "foot_sliding_mean_mps": 2.0,
    "root_tilt_max_rad": 2.0,
    "joint_velocity_p95_radps": 0.5,
    "joint_acceleration_p95_radps2": 1.0,
    "joint_jerk_p95_radps3": 1.5,
    "root_linear_velocity_p95_mps": 0.5,
    "root_angular_velocity_p95_radps": 0.5,
    "beat_alignment_mean_distance_s": 2.0,
    "joint_angle_mae_rad": 1.0,
    "root_trajectory_error_m": 1.0,
    "fk_body_position_error_m": 1.0,
}
HIGHER_WEIGHTS = {"beat_alignment_score": 2.0}


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _fixed_seed(base_seed: int, sample_id: str) -> int:
    payload = f"bumi-checkpoint-suite-v1:{int(base_seed)}:{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract_version") != SUITE_CONTRACT:
        raise ValueError(f"suite contract_version must be {SUITE_CONTRACT}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("checkpoint suite must contain a non-empty samples list")
    ids: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("every checkpoint suite sample must be an object")
        sample_id = str(sample.get("id", "")).strip()
        if (
            not sample_id
            or sample_id in ids
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample_id) is None
            or ".." in sample_id
        ):
            raise ValueError("checkpoint suite sample ids must be unique and non-empty")
        ids.add(sample_id)
        sources = [name for name in ("wav", "edge35") if sample.get(name)]
        if len(sources) != 1:
            raise ValueError(f"suite sample {sample_id} requires exactly one wav/edge35")
        source = Path(os.path.expandvars(str(sample[sources[0]]))).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"suite sample source does not exist: {source}")
        sample[sources[0]] = str(source)
        frames = int(sample.get("num_frames", 120))
        if frames != 120:
            raise ValueError("checkpoint selection currently fixes every sample to 120 frames")
        if sample.get("target_motion"):
            target = Path(os.path.expandvars(str(sample["target_motion"]))).expanduser()
            if not target.is_absolute():
                target = path.parent / target
            sample["target_motion"] = str(target.resolve(strict=True))
    return payload


def _checkpoint_step(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else -1


def _demo_command(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    sample: dict[str, Any],
    motion: Path,
    report: Path,
) -> list[str]:
    sample_id = str(sample["id"])
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/demo/demo_music_bumi.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(motion),
        "--report",
        str(report),
        "--exp",
        args.exp,
        "--kinematics",
        str(args.kinematics.expanduser().resolve(strict=True)),
        "--stats",
        str(args.stats.expanduser().resolve(strict=True)),
        "--num-frames",
        "120",
        "--cfg-scale",
        str(args.cfg_scale),
        "--ddim-steps",
        str(args.ddim_steps),
        "--seed",
        str(_fixed_seed(args.seed, sample_id)),
        "--device",
        args.device,
    ]
    if sample.get("wav"):
        command.extend(("--wav", str(Path(str(sample["wav"])).expanduser().resolve())))
        command.extend(("--start-sec", str(float(sample.get("start_sec", 0.0)))))
        command.extend(("--duration-sec", "4.0"))
    else:
        command.extend(("--edge35", str(Path(str(sample["edge35"])).expanduser().resolve())))
    if sample.get("target_motion"):
        command.extend(
            (
                "--target-motion",
                str(Path(str(sample["target_motion"])).expanduser().resolve()),
                "--target-start-frame",
                str(int(sample.get("target_start_frame", 0))),
            )
        )
    return command


def aggregate_metrics(reports: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("BUMI demo report has no metrics object")
        for name, raw in metrics.items():
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"BUMI demo metric {name} is non-finite")
            values[str(name)].append(value)
    return {name: float(sum(items) / len(items)) for name, items in values.items()}


def validate_demo_report(
    report: dict[str, Any],
    *,
    checkpoint_sha256: str,
    expected_seed: int,
    cfg_scale: float,
    ddim_steps: int,
) -> None:
    """拒绝把旧参数或其他 checkpoint 的缓存报告混入本次固定评测。"""

    if report.get("contract_version") != "genmo.bumi_demo_report.qpos30_contact.v2":
        raise ValueError("cached report is not a BUMI demo report")
    if (report.get("checkpoint") or {}).get("sha256") != checkpoint_sha256:
        raise ValueError("cached report checkpoint SHA does not match this candidate")
    if int(report.get("seed", -1)) != int(expected_seed):
        raise ValueError("cached report seed does not match the fixed suite seed")
    if not math.isclose(float(report.get("cfg_scale", math.nan)), float(cfg_scale)):
        raise ValueError("cached report CFG scale does not match this evaluation")
    if int(report.get("ddim_steps", -1)) != int(ddim_steps):
        raise ValueError("cached report DDIM steps do not match this evaluation")
    if report.get("qpos_shape") != [120, 28] or report.get("normalized_motion_shape") != [
        120,
        30,
    ]:
        raise ValueError("cached report does not describe one fixed 120-frame BUMI sample")


def rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_joint_violation_rate: float,
    max_foot_penetration_m: float,
    max_root_tilt_rad: float,
) -> list[dict[str, Any]]:
    """先做硬门槛，再对所有候选共有指标归一化并稳定排序。"""

    if not candidates:
        raise ValueError("no checkpoint candidates to rank")
    common = set(LOWER_WEIGHTS) | set(HIGHER_WEIGHTS)
    for candidate in candidates:
        common &= set(candidate["metrics"])
        reasons = []
        metrics = candidate["metrics"]
        if metrics.get("joint_limit_violation_rate", math.inf) > max_joint_violation_rate:
            reasons.append("joint_limit_violation_rate")
        if metrics.get("foot_penetration_max_m", math.inf) > max_foot_penetration_m:
            reasons.append("foot_penetration_max_m")
        if metrics.get("root_tilt_max_rad", math.inf) > max_root_tilt_rad:
            reasons.append("root_tilt_max_rad")
        candidate["eligible"] = not reasons
        candidate["hard_gate_reasons"] = reasons
        candidate["normalized_metrics"] = {}
    if not common:
        raise ValueError("checkpoint candidates have no common ranking metrics")
    total_weight = sum(LOWER_WEIGHTS.get(name, HIGHER_WEIGHTS.get(name, 0.0)) for name in common)
    for name in sorted(common):
        raw = [float(candidate["metrics"][name]) for candidate in candidates]
        low, high = min(raw), max(raw)
        for candidate, value in zip(candidates, raw):
            if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
                normalized = 0.0
            elif name in HIGHER_WEIGHTS:
                normalized = (high - value) / (high - low)
            else:
                normalized = (value - low) / (high - low)
            candidate["normalized_metrics"][name] = float(normalized)
    for candidate in candidates:
        candidate["score"] = float(
            sum(
                candidate["normalized_metrics"][name]
                * LOWER_WEIGHTS.get(name, HIGHER_WEIGHTS.get(name, 0.0))
                for name in common
            )
            / total_weight
        )
        candidate["ranking_metrics"] = sorted(common)
    return sorted(
        candidates,
        key=lambda item: (
            not bool(item["eligible"]),
            float(item["score"]),
            -int(item["global_step"]),
            str(item["checkpoint"]),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp", default="gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_50k")
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--max-joint-violation-rate", type=float, default=0.05)
    parser.add_argument("--max-foot-penetration-m", type=float, default=0.08)
    parser.add_argument("--max-root-tilt-rad", type=float, default=1.3)
    parser.add_argument("--keep-motion-artifacts", action="store_true")
    parser.add_argument("--reuse-reports", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 2 <= args.ddim_steps <= 1000:
        raise ValueError("--ddim-steps must be in 2..1000")
    checkpoints = [path.expanduser().resolve(strict=True) for path in args.checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("--checkpoints contains duplicates")
    suite_path = args.suite.expanduser().resolve(strict=True)
    suite = load_suite(suite_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_sha = sha256_file(checkpoint)
        candidate_id = f"{checkpoint.stem}_{checkpoint_sha[:12]}"
        candidate_dir = output_dir / "runs" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        commands = []
        for sample in suite["samples"]:
            sample_id = str(sample["id"])
            motion = candidate_dir / f"{sample_id}.pt"
            report = candidate_dir / f"{sample_id}.json"
            command = _demo_command(
                args=args,
                checkpoint=checkpoint,
                sample=sample,
                motion=motion,
                report=report,
            )
            commands.append(command)
            try:
                if not (args.reuse_reports and report.is_file()):
                    subprocess.run(command, cwd=REPO_ROOT, check=True)
                report_payload = json.loads(report.read_text(encoding="utf-8"))
                validate_demo_report(
                    report_payload,
                    checkpoint_sha256=checkpoint_sha,
                    expected_seed=_fixed_seed(args.seed, sample_id),
                    cfg_scale=args.cfg_scale,
                    ddim_steps=args.ddim_steps,
                )
                reports.append(report_payload)
            finally:
                if not args.keep_motion_artifacts:
                    motion.unlink(missing_ok=True)
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "global_step": _checkpoint_step(checkpoint),
                "reports": [
                    str(candidate_dir / f"{sample['id']}.json") for sample in suite["samples"]
                ],
                "commands": commands,
                "metrics": aggregate_metrics(reports),
            }
        )
    ranked = rank_candidates(
        candidates,
        max_joint_violation_rate=args.max_joint_violation_rate,
        max_foot_penetration_m=args.max_foot_penetration_m,
        max_root_tilt_rad=args.max_root_tilt_rad,
    )
    for index, candidate in enumerate(ranked, start=1):
        candidate["rank"] = index
    eligible = [candidate for candidate in ranked if candidate["eligible"]]
    result = {
        "contract_version": SELECTION_CONTRACT,
        "suite": {"path": str(suite_path), "sha256": sha256_file(suite_path)},
        "fixed_settings": {
            "seed": args.seed,
            "cfg_scale": args.cfg_scale,
            "ddim_steps": args.ddim_steps,
            "frames_per_sample": 120,
        },
        "hard_gates": {
            "max_joint_violation_rate": args.max_joint_violation_rate,
            "max_foot_penetration_m": args.max_foot_penetration_m,
            "max_root_tilt_rad": args.max_root_tilt_rad,
        },
        "best": None if not eligible else eligible[0]["checkpoint"],
        "best_sha256": None if not eligible else eligible[0]["checkpoint_sha256"],
        "candidates": ranked,
    }
    selection = output_dir / "selection.json"
    selection.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if eligible:
        (output_dir / "best_checkpoint.txt").write_text(
            eligible[0]["checkpoint"] + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not eligible:
        raise RuntimeError(f"no checkpoint passed hard gates; inspect {selection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
