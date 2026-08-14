#!/usr/bin/env python3
"""Apply strict human decisions into a separate, non-destructive curated dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.music_dance.curation.common import (  # noqa: E402
    DATASET_DIRS,
    DATASET_ORDER,
    SPLITS,
    atomic_torch_save,
    atomic_write_text,
    link_file,
    read_jsonl,
    resolve_relative,
    sha256_file,
    write_json,
    write_jsonl,
)
from tools.data.music_dance.curation.validate_review_results import (  # noqa: E402
    validate_decisions,
)


def build_curation_plan(
    master: list[dict[str, Any]], decisions: dict[str, dict[str, str]]
) -> dict[str, Any]:
    accepted = [row for row in master if decisions[row["review_id"]]["decision"] == "keep"]
    rejected = [row for row in master if decisions[row["review_id"]]["decision"] == "reject"]
    pending = [
        row
        for row in master
        if decisions[row["review_id"]]["decision"] not in {"keep", "reject"}
    ]
    before_refs = Counter(str(row["music_key"]) for row in master)
    after_refs = Counter(str(row["music_key"]) for row in accepted)
    orphan_keys = sorted(key for key in before_refs if after_refs[key] == 0)
    retained_shared_keys = sorted(
        key for key, count in before_refs.items() if count > 1 and after_refs[key] > 0
    )
    reasons: Counter[str] = Counter()
    for row in rejected:
        reasons.update(
            token for token in decisions[row["review_id"]]["issue_codes"].split(";") if token
        )
    return {
        "accepted": accepted,
        "rejected": rejected,
        "pending": pending,
        "before_refs": before_refs,
        "after_refs": after_refs,
        "orphan_keys": orphan_keys,
        "retained_shared_keys": retained_shared_keys,
        "reason_counts": reasons,
    }


def _summary(plan: dict[str, Any], output_root: Path, quarantine: bool) -> dict[str, Any]:
    accepted = plan["accepted"]
    rejected = plan["rejected"]
    master = accepted + rejected + plan["pending"]
    return {
        "output_root": str(output_root),
        "source_sample_count": len(master),
        "accepted_sample_count": len(accepted),
        "rejected_sample_count": len(rejected),
        "pending_sample_count": len(plan["pending"]),
        "accepted_counts_by_dataset": dict(Counter(row["dataset"] for row in accepted)),
        "rejected_counts_by_dataset": dict(Counter(row["dataset"] for row in rejected)),
        "accepted_counts_by_split": dict(Counter(row["split"] for row in accepted)),
        "rejected_counts_by_split": dict(Counter(row["split"] for row in rejected)),
        "source_hours": sum(int(row["num_frames"]) for row in master) / 30.0 / 3600.0,
        "accepted_hours": sum(int(row["num_frames"]) for row in accepted) / 30.0 / 3600.0,
        "rejected_hours": sum(int(row["num_frames"]) for row in rejected) / 30.0 / 3600.0,
        "unique_music_before": len(plan["before_refs"]),
        "unique_music_after": len(plan["after_refs"]),
        "zero_reference_music_count": len(plan["orphan_keys"]),
        "retained_shared_music_count": len(plan["retained_shared_keys"]),
        "rejection_reason_counts": dict(plan["reason_counts"]),
        "quarantine_materialization_requested": quarantine,
    }


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], str(row.get("group_id", row["sample_id"])))].append(row)
    output: list[dict[str, Any]] = []
    for (split, group_id), values in sorted(groups.items()):
        output.append(
            {
                "group_id": group_id,
                "split": split,
                "sample_ids": [row["sample_id"] for row in values],
                "num_retained_people": len(values),
                "music_feature_paths": sorted({row["music_feature_path"] for row in values}),
            }
        )
    return output


def _hardlink_or_copy(source: Path, target: Path) -> str:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _materialize(
    output_root: Path,
    export_root: Path,
    plan: dict[str, Any],
    decisions: dict[str, dict[str, str]],
    *,
    quarantine_zero_ref_music: bool,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"curated output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    accepted = plan["accepted"]
    rejected = plan["rejected"]
    rows_by_dataset = {
        dataset: [row for row in accepted if row["dataset"] == dataset]
        for dataset in DATASET_ORDER
    }
    restore_rows: list[dict[str, Any]] = []
    try:
        for dataset in DATASET_ORDER:
            dataset_rows = rows_by_dataset[dataset]
            target_root = staging / DATASET_DIRS[dataset]
            source_roots = {Path(row["source_root"]).resolve() for row in dataset_rows}
            # A fully rejected dataset still needs its source identity.  Fall back to any master row.
            if not source_roots:
                all_dataset_rows = [
                    row for row in accepted + rejected if row["dataset"] == dataset
                ]
                source_roots = {Path(row["source_root"]).resolve() for row in all_dataset_rows}
            if len(source_roots) != 1:
                raise ValueError(f"{dataset}: expected one source root, got {source_roots}")
            source_root = next(iter(source_roots))
            target_root.mkdir(parents=True, exist_ok=True)
            if dataset == "aistpp":
                link_file(source_root / "annot_aist_30fps.pt", target_root / "annot_aist_30fps.pt")
                for split in SPLITS:
                    atomic_torch_save(
                        [row["sample_id"] for row in dataset_rows if row["split"] == split],
                        target_root / f"{split}.pt",
                    )
            else:
                for motion_relative in sorted({row["motion_path"] for row in dataset_rows}):
                    link_file(
                        resolve_relative(source_root, motion_relative, "motion_path"),
                        target_root / motion_relative,
                    )
                for split in SPLITS:
                    source_rows = [
                        row["source_manifest_row"]
                        for row in dataset_rows
                        if row["split"] == split
                    ]
                    write_jsonl(target_root / "manifests" / f"{split}.jsonl", source_rows)
                if dataset in {"aioz_gdance", "compas3d"}:
                    write_jsonl(target_root / "manifests" / "groups.jsonl", _group_rows(dataset_rows))

            for music_relative in sorted({row["music_feature_path"] for row in dataset_rows}):
                link_file(
                    resolve_relative(source_root, music_relative, "music_feature_path"),
                    target_root / music_relative,
                )

        if quarantine_zero_ref_music:
            row_for_music: dict[str, dict[str, Any]] = {}
            for row in accepted + rejected:
                row_for_music.setdefault(str(row["music_key"]), row)
            for music_key in plan["orphan_keys"]:
                row = row_for_music[music_key]
                source_root = Path(row["source_root"]).resolve()
                source = resolve_relative(
                    source_root, row["music_feature_path"], "music_feature_path"
                )
                relative = (
                    Path("quarantine")
                    / DATASET_DIRS[row["dataset"]]
                    / row["music_feature_path"]
                )
                method = _hardlink_or_copy(source, staging / relative)
                restore_rows.append(
                    {
                        "music_key": music_key,
                        "dataset": row["dataset"],
                        "source_path": str(source),
                        "quarantine_path": relative.as_posix(),
                        "sha256": sha256_file(source),
                        "materialization": method,
                        "source_was_modified": False,
                    }
                )

        accepted_rows = []
        for row in accepted:
            accepted_rows.append(
                {
                    **row,
                    "decision": decisions[row["review_id"]]["decision"],
                    "issue_codes": decisions[row["review_id"]]["issue_codes"],
                }
            )
        rejected_rows = []
        for row in rejected:
            rejected_rows.append(
                {
                    **row,
                    "decision": decisions[row["review_id"]]["decision"],
                    "issue_codes": decisions[row["review_id"]]["issue_codes"],
                    "reviewer": decisions[row["review_id"]].get("reviewer", ""),
                    "notes": decisions[row["review_id"]].get("notes", ""),
                }
            )
        write_jsonl(staging / "reports" / "accepted_master.jsonl", accepted_rows)
        write_jsonl(staging / "reports" / "rejected_samples.jsonl", rejected_rows)
        write_jsonl(staging / "reports" / "zero_reference_music.jsonl", restore_rows)
        write_jsonl(staging / "reports" / "restore_manifest.jsonl", restore_rows)
        report = {
            **_summary(plan, output_root, quarantine_zero_ref_music),
            "export_root": str(export_root),
            "source_roots_modified": False,
            "raw_audio_modified": False,
            "curated_dataset_dirs": DATASET_DIRS,
            "final_pass": True,
        }
        write_json(staging / "reports" / "curation_report.json", report)
        atomic_write_text(
            staging / "README.md",
            "# Music-only 四数据集人工筛选结果\n\n"
            "本目录由人工 decisions.csv 生成。源数据集未被修改；motions 优先通过硬链接复用，"
            "跨文件系统时自动复制，"
            "active musicfeat_v2 只包含仍被至少一个 keep 动作引用的 EDGE35。quarantine 中的"
            "文件是硬链接或恢复副本，删除它们不会删除源数据。\n",
        )
        os.replace(staging, output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def apply_results(args: argparse.Namespace) -> dict[str, Any]:
    export_root = Path(args.export_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    _, decisions = validate_decisions(
        export_root, args.decisions, strict=True, write_report=False
    )
    master = read_jsonl(export_root / "index" / "master.jsonl")
    plan = build_curation_plan(master, decisions)
    if plan["pending"]:
        raise RuntimeError("strict application cannot continue with pending decisions")
    report = {
        **_summary(plan, output_root, args.quarantine_zero_ref_music),
        "dry_run": bool(args.dry_run),
        "final_pass": True,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report
    materialized = _materialize(
        output_root,
        export_root,
        plan,
        decisions,
        quarantine_zero_ref_music=args.quarantine_zero_ref_music,
    )
    print(json.dumps(materialized, indent=2, ensure_ascii=False))
    return materialized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output-root", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--quarantine-zero-ref-music", action="store_true")
    return parser


def main() -> None:
    apply_results(build_parser().parse_args())


if __name__ == "__main__":
    main()
