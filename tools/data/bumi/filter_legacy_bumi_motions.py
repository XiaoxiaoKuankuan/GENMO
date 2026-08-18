#!/usr/bin/env python3
"""对 GMR 生产 BUMI3 的 legacy pickle 执行可审计、默认只读的质量预筛选。

脚本扫描 ``data/motions/<dataset>/*.pkl``，先验证生成数据所用 source MJCF 的
SHA256，再调用 ``gem.robots.bumi`` 的严格 legacy reader 和质量规则。默认行为只在
独立输出目录生成 JSONL/CSV/汇总报告，不删除、不重写、不移动源文件；只有同时提供
``--apply`` 与全新的 ``--materialize-root`` 才会把 PASS（可选 REVIEW）以 hardlink
或 copy 物化到新目录。物化过程使用同文件系统 staging 目录和原子 rename，失败时
不会留下伪装成完整数据集的目标目录。

每行报告包含源文件 SHA、配置 SHA、source MJCF SHA、三态结论、稳定 reason code、
动力学统计、贴地帧区间及可选安全区间。报告可以在不重新计算动作的情况下复核、
统计和恢复来源，也是后续 ``genmo.bumi_music.v1`` 转换设置 quality 标记的唯一依据。
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.legacy_motion import (  # noqa: E402
    LEGACY_BUMI_MOTION_CONTRACT_VERSION,
    load_legacy_bumi_motion,
    sha256_file,
)
from gem.robots.bumi.quality_filter import (  # noqa: E402
    BUMI_QUALITY_REPORT_VERSION,
    BumiQualityConfig,
    QualityStatus,
    evaluate_legacy_bumi_motion,
    load_bumi_quality_config,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "bumi" / "quality_filter_v1.yaml"
REPORT_FILENAMES = (
    "quality_report.jsonl",
    "quality_report.csv",
    "quality_summary.json",
    "review_candidates.jsonl",
    "quality_config.snapshot.yaml",
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _relative_source(path: Path, input_root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"motion path escapes input root: {path} -> {resolved}") from exc


def discover_motion_paths(input_root: str | Path) -> list[Path]:
    """确定性发现源目录内的 pickle，拒绝逃逸 root 的 symlink。"""

    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths: list[Path] = []
    for candidate in root.rglob("*.pkl"):
        if candidate.is_file():
            _relative_source(candidate, root)
            paths.append(candidate.resolve())
    paths.sort(key=lambda value: value.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"no .pkl motions found under {root}")
    return paths


def verify_source_mjcf(path: str | Path, expected_sha256: str) -> tuple[Path, str]:
    """硬校验生成 pickle 的 source MJCF，防止错误资产下的限位和 FK 解释。"""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(
            f"source MJCF SHA256 mismatch: expected={expected_sha256}, actual={actual}, "
            f"path={source}"
        )
    return source, actual


def _identity(path: Path, input_root: Path) -> dict[str, str]:
    relative = _relative_source(path, input_root)
    if len(relative.parts) < 2:
        raise ValueError(f"motion must live below a dataset directory: {relative.as_posix()}")
    dataset = relative.parts[0]
    sample_id = relative.with_suffix("").as_posix()
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "source_relative_path": relative.as_posix(),
    }


def _evaluate_one(
    path_text: str,
    input_root_text: str,
    config: BumiQualityConfig,
    config_sha256: str,
    source_mjcf_sha256: str,
) -> dict[str, Any]:
    path = Path(path_text)
    input_root = Path(input_root_text)
    identity: dict[str, Any]
    try:
        identity = _identity(path, input_root)
    except Exception as exc:
        identity = {
            "dataset": "__invalid__",
            "sample_id": path.stem,
            "source_relative_path": str(path),
        }
        identity_error: Exception | None = exc
    else:
        identity_error = None
    base = {
        "report_contract_version": BUMI_QUALITY_REPORT_VERSION,
        "source_motion_contract_version": LEGACY_BUMI_MOTION_CONTRACT_VERSION,
        **identity,
        "quality_config_sha256": config_sha256,
        "source_mjcf_sha256": source_mjcf_sha256,
    }
    try:
        source_sha = sha256_file(path)
    except Exception as exc:
        return {
            **base,
            "source_sha256": None,
            "status": QualityStatus.REJECT.value,
            "quality_accepted": False,
            "reason_codes": ["SOURCE_READ_ERROR"],
            "metrics": {},
            "floor_intervals": [],
            "valid_intervals": [],
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
        }
    if identity_error is not None:
        return {
            **base,
            "source_sha256": source_sha,
            "status": QualityStatus.REJECT.value,
            "quality_accepted": False,
            "reason_codes": ["SOURCE_PATH_CONTRACT"],
            "metrics": {},
            "floor_intervals": [],
            "valid_intervals": [],
            "error_type": type(identity_error).__name__,
            "error_message": str(identity_error)[:2000],
        }
    try:
        motion = load_legacy_bumi_motion(path, expected_fps=config.fps)
        decision = evaluate_legacy_bumi_motion(motion, config)
    except Exception as exc:
        return {
            **base,
            "source_sha256": source_sha,
            "status": QualityStatus.REJECT.value,
            "quality_accepted": False,
            "reason_codes": ["MOTION_CONTRACT_ERROR"],
            "metrics": {},
            "floor_intervals": [],
            "valid_intervals": [],
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
        }
    return {
        **base,
        "source_sha256": source_sha,
        **decision.to_dict(),
        "error_type": None,
        "error_message": None,
    }


def scan_motions(
    paths: list[Path],
    *,
    input_root: Path,
    config: BumiQualityConfig,
    config_sha256: str,
    source_mjcf_sha256: str,
    workers: int,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """按输入顺序返回结果；并行 worker 数不影响报告排序与结论。"""

    arguments = [
        (str(path), str(input_root), config, config_sha256, source_mjcf_sha256) for path in paths
    ]
    if workers <= 1:
        iterator = (_evaluate_one(*values) for values in arguments)
        return list(
            tqdm(iterator, total=len(arguments), desc="BUMI quality", disable=not show_progress)
        )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        iterator = executor.map(_evaluate_one_star, arguments, chunksize=8)
        return list(
            tqdm(iterator, total=len(arguments), desc="BUMI quality", disable=not show_progress)
        )


def _evaluate_one_star(arguments: tuple[Any, ...]) -> dict[str, Any]:
    return _evaluate_one(*arguments)


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _finite_numbers(values: Iterable[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and value == value and abs(float(value)) != float("inf"):
            output.append(float(value))
    return output


def _percentiles(values: Iterable[Any]) -> dict[str, float | int] | None:
    import numpy as np

    finite = _finite_numbers(values)
    if not finite:
        return None
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    input_root: Path,
    config_path: Path,
    config_sha256: str,
    source_mjcf: Path,
    source_mjcf_sha256: str,
) -> dict[str, Any]:
    """汇总状态、reason、数据集分布以及关键序列级指标分位数。"""

    status_counts = Counter(str(row["status"]) for row in rows)
    reason_counts = Counter(str(reason) for row in rows for reason in row.get("reason_codes", ()))
    by_dataset: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    for dataset, values in sorted(grouped.items()):
        frames = sum(int(_nested(row, "metrics", "num_frames") or 0) for row in values)
        by_dataset[dataset] = {
            "sequences": len(values),
            "frames": frames,
            "hours": frames / 30.0 / 3600.0,
            "status_counts": dict(Counter(str(row["status"]) for row in values)),
            "reason_counts": dict(
                Counter(str(reason) for row in values for reason in row.get("reason_codes", ()))
            ),
        }
    distribution_paths = {
        "root_height_p05": ("metrics", "root_height_p05"),
        "floor_frame_ratio": ("metrics", "floor_style", "frame_ratio"),
        "floor_max_consecutive_frames": (
            "metrics",
            "floor_style",
            "max_consecutive_frames",
        ),
        "joint_velocity_l2_p95": (
            "metrics",
            "dynamics",
            "joint_velocity_l2",
            "p95",
        ),
        "joint_acceleration_l2_p95": (
            "metrics",
            "dynamics",
            "joint_acceleration_l2",
            "p95",
        ),
        "joint_jerk_l2_p95": (
            "metrics",
            "dynamics",
            "joint_jerk_l2",
            "p95",
        ),
        "root_linear_velocity_p95": (
            "metrics",
            "dynamics",
            "root_linear_velocity",
            "p95",
        ),
        "root_angular_velocity_p95": (
            "metrics",
            "dynamics",
            "root_angular_velocity",
            "p95",
        ),
    }
    distributions = {
        name: value
        for name, path in distribution_paths.items()
        if (value := _percentiles(_nested(row, *path) for row in rows)) is not None
    }
    total_frames = sum(int(_nested(row, "metrics", "num_frames") or 0) for row in rows)
    return {
        "report_contract_version": BUMI_QUALITY_REPORT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_root": str(input_root),
        "source_files_modified": False,
        "source_motion_contract_version": LEGACY_BUMI_MOTION_CONTRACT_VERSION,
        "source_mjcf": str(source_mjcf),
        "source_mjcf_sha256": source_mjcf_sha256,
        "source_mjcf_verified": True,
        "quality_config": str(config_path),
        "quality_config_sha256": config_sha256,
        "sequences": len(rows),
        "frames": total_frames,
        "hours": total_frames / 30.0 / 3600.0,
        "quality_accepted_sequences": status_counts.get(QualityStatus.PASS.value, 0),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "by_dataset": by_dataset,
        "metric_distributions": distributions,
    }


def _flat_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "sample_id": row.get("sample_id"),
        "source_relative_path": row.get("source_relative_path"),
        "source_sha256": row.get("source_sha256"),
        "status": row.get("status"),
        "quality_accepted": row.get("quality_accepted"),
        "reason_codes": "|".join(map(str, row.get("reason_codes", ()))),
        "num_frames": _nested(row, "metrics", "num_frames"),
        "root_height_p05": _nested(row, "metrics", "root_height_p05"),
        "floor_frame_ratio": _nested(row, "metrics", "floor_style", "frame_ratio"),
        "floor_max_run": _nested(row, "metrics", "floor_style", "max_consecutive_frames"),
        "joint_velocity_p95": _nested(row, "metrics", "dynamics", "joint_velocity_l2", "p95"),
        "joint_acceleration_p95": _nested(
            row, "metrics", "dynamics", "joint_acceleration_l2", "p95"
        ),
        "joint_jerk_p95": _nested(row, "metrics", "dynamics", "joint_jerk_l2", "p95"),
        "root_linear_velocity_p95": _nested(
            row, "metrics", "dynamics", "root_linear_velocity", "p95"
        ),
        "root_angular_velocity_p95": _nested(
            row, "metrics", "dynamics", "root_angular_velocity", "p95"
        ),
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
    }


def write_reports(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
    config_path: Path,
    *,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [output_dir / name for name in REPORT_FILENAMES if (output_dir / name).exists()]
    if occupied and not overwrite:
        raise FileExistsError(
            f"report artifacts already exist: {occupied}; pass --overwrite to atomically replace them"
        )
    _write_jsonl(output_dir / "quality_report.jsonl", rows)
    review_rows = [row for row in rows if row["status"] != QualityStatus.PASS.value]
    review_rows.sort(
        key=lambda row: (
            0 if row["status"] == QualityStatus.REJECT.value else 1,
            -int(_nested(row, "metrics", "floor_style", "max_consecutive_frames") or 0),
            str(row["sample_id"]),
        )
    )
    _write_jsonl(output_dir / "review_candidates.jsonl", review_rows)
    _write_json(output_dir / "quality_summary.json", dict(summary))
    _atomic_write_text(
        output_dir / "quality_config.snapshot.yaml",
        config_path.read_text(encoding="utf-8"),
    )
    flat = [_flat_csv_row(row) for row in rows]
    csv_path = output_dir / "quality_report.csv"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_dir,
        prefix=f".{csv_path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    try:
        os.replace(temporary, csv_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_or_link(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, target)
    return "copy"


def materialize_selection(
    rows: list[dict[str, Any]],
    *,
    input_root: Path,
    output_root: Path,
    include_review: bool,
    mode: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """把选中源文件物化到全新目录；永不删除或改写源文件。"""

    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"materialize root already exists: {output_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("materialize root must be outside the source input root")
    accepted_status = {QualityStatus.PASS.value}
    if include_review:
        accepted_status.add(QualityStatus.REVIEW.value)
    selected = [row for row in rows if row["status"] in accepted_status]
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    method_counts: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    try:
        for row in selected:
            relative = Path(str(row["source_relative_path"]))
            source = (input_root / relative).resolve()
            source.relative_to(input_root)
            if sha256_file(source) != row["source_sha256"]:
                raise ValueError(f"source changed after scan: {source}")
            method = _copy_or_link(source, staging / "motions" / relative, mode)
            method_counts[method] += 1
            manifest.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "motion_path": (Path("motions") / relative).as_posix(),
                    "source_relative_path": relative.as_posix(),
                    "source_sha256": row["source_sha256"],
                    "quality_status": row["status"],
                    "quality_accepted": row["status"] == QualityStatus.PASS.value,
                    "quality_config_sha256": row["quality_config_sha256"],
                    "source_mjcf_sha256": row["source_mjcf_sha256"],
                }
            )
        _write_jsonl(staging / "manifests" / "selected.jsonl", manifest)
        selection = {
            "report_contract_version": BUMI_QUALITY_REPORT_VERSION,
            "source_root": str(input_root),
            "source_files_modified": False,
            "include_review": include_review,
            "selected_sequences": len(selected),
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "materialization_counts": dict(method_counts),
            "source_summary": dict(summary),
        }
        _write_json(staging / "meta" / "quality_selection.json", selection)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_root": str(output_root),
        "selected_sequences": len(selected),
        "materialization_counts": dict(method_counts),
        "include_review": include_review,
    }


def _ensure_output_outside_source(output: Path, source: Path, name: str) -> Path:
    value = output.expanduser().resolve()
    if value == source or source in value.parents:
        raise ValueError(f"{name} must be outside input root so the source stays read-only")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=REPO_ROOT / "data" / "motions")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-mjcf",
        type=Path,
        required=True,
        help="生成这些 legacy pickle 的原始 MJCF；内容 SHA 必须匹配配置。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--limit", type=int, help="仅用于代码验证；按排序后的前 N 条扫描。")
    parser.add_argument("--overwrite", action="store_true", help="原子替换同名报告文件。")
    parser.add_argument("--apply", action="store_true", help="显式允许物化选中动作。")
    parser.add_argument("--materialize-root", type=Path)
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="物化 REVIEW；默认正式集合只包含 PASS。",
    )
    parser.add_argument("--materialize-mode", choices=("auto", "hardlink", "copy"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.apply != (args.materialize_root is not None):
        raise ValueError("materialization requires both --apply and --materialize-root")

    input_root = args.input_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = _ensure_output_outside_source(args.output_dir, input_root, "--output-dir")
    config = load_bumi_quality_config(config_path)
    source_mjcf, source_sha = verify_source_mjcf(args.source_mjcf, config.source_mjcf_sha256)
    config_sha = sha256_file(config_path)
    paths = discover_motion_paths(input_root)
    if args.limit is not None:
        paths = paths[: args.limit]
    rows = scan_motions(
        paths,
        input_root=input_root,
        config=config,
        config_sha256=config_sha,
        source_mjcf_sha256=source_sha,
        workers=args.workers,
    )
    summary = build_summary(
        rows,
        input_root=input_root,
        config_path=config_path,
        config_sha256=config_sha,
        source_mjcf=source_mjcf,
        source_mjcf_sha256=source_sha,
    )
    write_reports(output_dir, rows, summary, config_path, overwrite=args.overwrite)
    result: dict[str, Any] = {
        "report_dir": str(output_dir),
        "sequences": len(rows),
        "status_counts": summary["status_counts"],
        "source_files_modified": False,
    }
    if args.apply:
        materialize_root = _ensure_output_outside_source(
            args.materialize_root, input_root, "--materialize-root"
        )
        result["materialized"] = materialize_selection(
            rows,
            input_root=input_root,
            output_root=materialize_root,
            include_review=args.include_review,
            mode=args.materialize_mode,
            summary=summary,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
