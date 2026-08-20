#!/usr/bin/env python3
"""按人工评分表筛选并备份四数据集的人体动作。

本工具面向 ``music_only_4set_v1`` 审阅包：输入四张只包含 ``motion_name`` 与
``score`` 的 CSV，严格选取指定分数（当前人工约定中 ``1`` 表示高质量），再从审阅包
中找出同名的 30 Hz neutral SMPL-X body-only NPZ。输出使用独立的原子化目录，动作在
同一文件系统上优先建立硬链接，既不修改源审阅包，也不重复占用大量空间；跨文件系统
时才退化为复制。

筛选不是简单按文件名复制。程序会检查四张表是否完整覆盖源审阅包、评分值是否一致、
源 NPZ 的字段/形状/帧率/内部 ID 是否符合人体轨迹契约，并用 master index 中的 SHA256
逐文件复核内容。AIST++ 的 ``*_armfix.npz`` 是机器人重定向阶段的别名，不是另一条人体
动作；它会显式映射回同名基础人体 NPZ，并在 ``index/aliases.jsonl`` 留下折叠记录，避免
后训练时重复采样同一人体轨迹。最终目录同时保存原评分表、可机读索引、动作校验和与
统计报告，便于以后重建带音乐的微调数据集。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.music_dance.curation.common import (  # noqa: E402
    DATASET_ORDER,
    atomic_write_text,
    link_file,
    read_csv,
    read_jsonl,
    sha256_file,
    validate_review_npz,
    write_csv,
    write_json,
    write_jsonl,
)

RATING_FILES = {
    "aistpp": "bumi3_aistpp_ratings.csv",
    "aioz_gdance": "bumi3_aioz_gdance.csv",
    "finedance": "bumi3_finedance_ratings.csv",
    "compas3d": "bumi3_compas3d_ratings.csv",
}
RATING_COLUMNS = ("motion_name", "score")
SELECTED_COLUMNS = (
    "dataset",
    "sample_id",
    "review_id",
    "score",
    "rating_motion_names",
    "split",
    "num_frames",
    "duration_sec",
    "motion_path",
    "sha256",
    "materialization",
)
REQUIRED_NPZ_METADATA = ("dataset", "sample_id", "coordinate_system")
EXPECTED_COORDINATE_SYSTEM = "right_handed_y_up_metric"


def _canonical_rating_name(dataset: str, motion_name: str) -> str:
    """把评分文件名映射到审阅包中唯一的人体动作文件名。"""
    if not motion_name or Path(motion_name).name != motion_name or not motion_name.endswith(".npz"):
        raise ValueError(f"{dataset}: invalid motion_name {motion_name!r}")
    if dataset == "aistpp" and motion_name.endswith("_armfix.npz"):
        return f"{motion_name.removesuffix('_armfix.npz')}.npz"
    return motion_name


def _load_rating_rows(rating_root: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    fingerprints: dict[str, str] = {}
    for dataset in DATASET_ORDER:
        path = rating_root / RATING_FILES[dataset]
        columns, rows = read_csv(path)
        if tuple(columns) != RATING_COLUMNS:
            raise ValueError(
                f"{path}: expected exact columns {list(RATING_COLUMNS)}, got {columns}"
            )
        seen: set[str] = set()
        normalized: list[dict[str, str]] = []
        for line_number, row in enumerate(rows, 2):
            motion_name = str(row["motion_name"]).strip()
            score = str(row["score"]).strip()
            if motion_name in seen:
                raise ValueError(f"{path}:{line_number}: duplicate motion_name {motion_name!r}")
            if score not in {"1", "2", "3"}:
                raise ValueError(f"{path}:{line_number}: score must be 1, 2 or 3, got {score!r}")
            seen.add(motion_name)
            normalized.append(
                {
                    "motion_name": motion_name,
                    "canonical_motion_name": _canonical_rating_name(dataset, motion_name),
                    "score": score,
                }
            )
        if not normalized:
            raise ValueError(f"{path}: rating table is empty")
        rows_by_dataset[dataset] = normalized
        fingerprints[RATING_FILES[dataset]] = sha256_file(path)
    return rows_by_dataset, fingerprints


def _source_rows(
    export_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    master_path = export_root / "index" / "master.jsonl"
    master = read_jsonl(master_path)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in master:
        dataset = str(row.get("dataset", ""))
        if dataset not in DATASET_ORDER:
            raise ValueError(f"{master_path}: unsupported dataset {dataset!r}")
        relative = Path(str(row.get("review_motion_path", "")))
        expected_relative = Path("motions") / dataset / f"{row.get('sample_id')}.npz"
        if relative != expected_relative:
            raise ValueError(
                f"{row.get('review_id')}: review_motion_path must be {expected_relative}, got {relative}"
            )
        key = (dataset, relative.name)
        if key in index:
            raise ValueError(f"{master_path}: duplicate source motion {key}")
        index[key] = row
    if set(row["dataset"] for row in master) != set(DATASET_ORDER):
        raise ValueError(f"{master_path}: source package must contain all four datasets")
    return master, index


def build_selection_plan(
    export_root: str | Path,
    rating_root: str | Path,
    selected_score: str = "1",
) -> dict[str, Any]:
    """校验评分表与源索引，并生成唯一人体动作选择计划。"""
    export_root = Path(export_root).expanduser().resolve()
    rating_root = Path(rating_root).expanduser().resolve()
    if selected_score not in {"1", "2", "3"}:
        raise ValueError(f"selected_score must be 1, 2 or 3, got {selected_score!r}")
    rating_rows, rating_fingerprints = _load_rating_rows(rating_root)
    master, source_index = _source_rows(export_root)

    source_names_by_dataset = {
        dataset: {name for (row_dataset, name) in source_index if row_dataset == dataset}
        for dataset in DATASET_ORDER
    }
    selected: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    rating_counts: Counter[str] = Counter()
    for dataset in DATASET_ORDER:
        rows = rating_rows[dataset]
        canonical_scores: dict[str, set[str]] = {}
        canonical_raw_names: dict[str, list[str]] = {}
        for row in rows:
            canonical = row["canonical_motion_name"]
            canonical_scores.setdefault(canonical, set()).add(row["score"])
            canonical_raw_names.setdefault(canonical, []).append(row["motion_name"])
            if row["score"] == selected_score:
                rating_counts[dataset] += 1
        conflicts = {name: scores for name, scores in canonical_scores.items() if len(scores) != 1}
        if conflicts:
            raise ValueError(f"{dataset}: aliases have conflicting scores: {conflicts}")
        canonical_names = set(canonical_scores)
        source_names = source_names_by_dataset[dataset]
        if canonical_names != source_names:
            raise ValueError(
                f"{dataset}: rating/source coverage mismatch; "
                f"unknown_ratings={sorted(canonical_names - source_names)[:10]}, "
                f"unrated_sources={sorted(source_names - canonical_names)[:10]}"
            )
        for canonical in sorted(canonical_names):
            score = next(iter(canonical_scores[canonical]))
            if score != selected_score:
                continue
            source = source_index[(dataset, canonical)]
            rating_names = sorted(canonical_raw_names[canonical])
            row = {
                **source,
                "rating_score": score,
                "rating_motion_names": rating_names,
                "selected_motion_path": str(Path("motions") / dataset / canonical),
            }
            selected.append(row)
            if len(rating_names) > 1 or rating_names[0] != canonical:
                aliases.append(
                    {
                        "dataset": dataset,
                        "canonical_motion_name": canonical,
                        "rating_motion_names": rating_names,
                        "score": score,
                        "reason": "robot_retarget_alias_collapsed_to_unique_human_motion",
                    }
                )

    selected.sort(key=lambda row: (DATASET_ORDER.index(row["dataset"]), row["sample_id"]))
    total_frames = sum(int(row["num_frames"]) for row in selected)
    return {
        "export_root": export_root,
        "rating_root": rating_root,
        "master": master,
        "selected": selected,
        "aliases": aliases,
        "selected_score": selected_score,
        "rating_fingerprints": rating_fingerprints,
        "rating_row_counts": {dataset: len(rating_rows[dataset]) for dataset in DATASET_ORDER},
        "selected_rating_row_counts": dict(rating_counts),
        "source_counts": dict(Counter(row["dataset"] for row in master)),
        "selected_counts": dict(Counter(row["dataset"] for row in selected)),
        "selected_split_counts": dict(Counter(row["split"] for row in selected)),
        "total_frames": total_frames,
        "total_hours": total_frames / 30.0 / 3600.0,
    }


def _validate_selected_npz(path: Path, row: dict[str, Any]) -> int:
    frames = validate_review_npz(path, row)
    with np.load(path, allow_pickle=False) as payload:
        missing = set(REQUIRED_NPZ_METADATA) - set(payload.files)
        if missing:
            raise ValueError(f"{path}: missing identity metadata {sorted(missing)}")
        dataset = str(np.asarray(payload["dataset"]).item())
        sample_id = str(np.asarray(payload["sample_id"]).item())
        coordinate = str(np.asarray(payload["coordinate_system"]).item())
        if dataset != row["dataset"] or sample_id != row["sample_id"]:
            raise ValueError(f"{path}: embedded dataset/sample_id differs from master index")
        if coordinate != EXPECTED_COORDINATE_SYSTEM:
            raise ValueError(f"{path}: unexpected coordinate_system {coordinate!r}")
    digest = sha256_file(path)
    if digest != row["review_sha256"]:
        raise ValueError(f"{path}: SHA256 differs from immutable source master index")
    return frames


def _base_report(plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    return {
        "schema": "manual_high_quality_human_motion_v1",
        "output_root": str(output_root),
        "source_export_root": str(plan["export_root"]),
        "source_master_sha256": sha256_file(plan["export_root"] / "index" / "master.jsonl"),
        "rating_root": str(plan["rating_root"]),
        "rating_file_sha256": plan["rating_fingerprints"],
        "selected_score": plan["selected_score"],
        "rating_row_counts": plan["rating_row_counts"],
        "selected_rating_row_counts": plan["selected_rating_row_counts"],
        "source_counts": plan["source_counts"],
        "selected_unique_motion_counts": plan["selected_counts"],
        "selected_split_counts": plan["selected_split_counts"],
        "selected_rating_row_count": sum(plan["selected_rating_row_counts"].values()),
        "selected_unique_motion_count": len(plan["selected"]),
        "collapsed_alias_count": len(plan["aliases"]),
        "total_frames": plan["total_frames"],
        "total_hours_at_30fps": plan["total_hours"],
        "motion_contract": {
            "fps": 30.0,
            "pose": "float32[T,66] axis-angle",
            "transl": "float32[T,3] metric Y-up",
            "betas": "float32[T,10]",
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        },
        "source_modified": False,
    }


def _selected_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "review_id": row["review_id"],
            "score": row["rating_score"],
            "rating_motion_names": ";".join(row["rating_motion_names"]),
            "split": row["split"],
            "num_frames": row["num_frames"],
            "duration_sec": f"{float(row['duration_sec']):.6f}",
            "motion_path": row["selected_motion_path"],
            "sha256": row["review_sha256"],
            "materialization": row.get("materialization", ""),
        }
        for row in rows
    ]


def _materialize(plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    materialization: Counter[str] = Counter()
    try:
        written_rows: list[dict[str, Any]] = []
        checksum_lines: list[str] = []
        verified_frames = 0
        for row in plan["selected"]:
            source = plan["export_root"] / row["review_motion_path"]
            verified_frames += _validate_selected_npz(source, row)
            target = staging / row["selected_motion_path"]
            method = link_file(source, target)
            materialization[method] += 1
            written = {**row, "materialization": method}
            written_rows.append(written)
            checksum_lines.append(f"{row['review_sha256']}  {row['selected_motion_path']}\n")

        for filename in RATING_FILES.values():
            source = plan["rating_root"] / filename
            target = staging / "ratings" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        write_jsonl(staging / "index" / "selected.jsonl", written_rows)
        write_csv(
            staging / "index" / "selected.csv", _selected_csv_rows(written_rows), SELECTED_COLUMNS
        )
        write_jsonl(staging / "index" / "aliases.jsonl", plan["aliases"])
        atomic_write_text(staging / "index" / "motion_sha256.txt", "".join(checksum_lines))
        report = {
            **_base_report(plan, output_root),
            "materialization_counts": dict(materialization),
            "verified_motion_count": len(written_rows),
            "verified_frames": verified_frames,
            "final_pass": verified_frames == plan["total_frames"],
        }
        if not report["final_pass"]:
            raise RuntimeError("verified frame total differs from the selection plan")
        write_json(staging / "reports" / "selection_report.json", report)
        atomic_write_text(
            staging / "README.md",
            "# 四数据集人工高质量人体动作备用包\n\n"
            "本目录严格按 ratings 中 score=1 的名称，从 music_only_4set_v1 审阅包筛出。"
            "motions 是 30 Hz neutral SMPL-X body-only NPZ，源审阅包未被修改；同盘文件优先"
            "使用硬链接。index/selected.jsonl 保留原 split、音乐键和源 manifest 信息，可在后续"
            "重建音乐配对与微调数据集。AIST++ 的机器人 `_armfix` 别名在人体层只保留一条基础"
            "动作，折叠关系见 index/aliases.jsonl。\n",
        )
        os.replace(staging, output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def select_human_motions(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_selection_plan(args.export_root, args.rating_root, str(args.score))
    output_root = Path(args.output_root).expanduser().resolve()
    if args.dry_run:
        report = {**_base_report(plan, output_root), "dry_run": True, "final_pass": True}
    else:
        report = _materialize(plan, output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True, help="完整人体动作审阅包根目录")
    parser.add_argument("--rating-root", required=True, help="四张 motion_name/score CSV 所在目录")
    parser.add_argument("--output-root", required=True, help="独立备用高质量数据版本输出目录")
    parser.add_argument("--score", default="1", choices=("1", "2", "3"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main() -> None:
    select_human_motions(build_parser().parse_args())


if __name__ == "__main__":
    main()
