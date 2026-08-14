#!/usr/bin/env python3
"""Validate a motion-only human review export and its immutable identity index."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import validate_aist_metric_translation  # noqa: E402
from tools.data.music_dance.curation.common import (  # noqa: E402
    DATASET_ORDER,
    DECISION_COLUMNS,
    read_csv,
    read_jsonl,
    sha256_file,
    validate_review_npz,
    write_json,
)

FORBIDDEN_MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mp4",
    ".mov",
    ".pt",
    ".pth",
    ".ckpt",
}
EXPECTED_FULL_COUNTS = {
    "by_dataset": {"aistpp": 1020, "aioz_gdance": 6011, "finedance": 183, "compas3d": 72},
    "by_split": {"train": 6095, "val": 614, "test": 577},
    "total": 7286,
}


def _read_checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: malformed SHA256SUMS line") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{path}:{line_number}: malformed SHA256 digest")
        if relative in values:
            raise ValueError(f"{path}:{line_number}: duplicate checksum path {relative}")
        values[relative] = digest
    return values


def validate_package(
    export_root: str | Path,
    *,
    verify_checksums: bool = True,
    expect_full_four_set: bool = False,
    require_blank_decisions: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    root = Path(export_root).expanduser().resolve()
    errors: list[str] = []
    master_path = root / "index" / "master.jsonl"
    fingerprint_path = root / "index" / "source_fingerprints.json"
    checksum_path = root / "index" / "SHA256SUMS"
    decisions_path = root / "review" / "decisions.csv"
    for path in (master_path, fingerprint_path, checksum_path, decisions_path, root / "README.md"):
        if not path.is_file():
            errors.append(f"missing required package file: {path}")
    if errors:
        raise ValueError("; ".join(errors))

    master = read_jsonl(master_path)
    fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    review_ids = [str(row.get("review_id")) for row in master]
    if len(review_ids) != len(set(review_ids)):
        errors.append("master index contains duplicate review_id")
    counts_by_dataset = Counter(str(row.get("dataset")) for row in master)
    counts_by_split = Counter(str(row.get("split")) for row in master)
    unknown_datasets = set(counts_by_dataset) - set(DATASET_ORDER)
    if unknown_datasets:
        errors.append(f"master index contains unknown datasets: {sorted(unknown_datasets)}")
    if int(fingerprints.get("sample_count", -1)) != len(master):
        errors.append("source_fingerprints sample_count differs from master index")
    if dict(fingerprints.get("counts_by_dataset", {})) != dict(counts_by_dataset):
        errors.append("source_fingerprints dataset counts differ from master index")
    if dict(fingerprints.get("counts_by_split", {})) != dict(counts_by_split):
        errors.append("source_fingerprints split counts differ from master index")

    checksums = _read_checksums(checksum_path)
    for relative, expected_digest in checksums.items():
        checksum_target = root / relative
        try:
            checksum_target.resolve().relative_to(root)
        except ValueError:
            errors.append(f"SHA256SUMS path escapes export root: {relative}")
            continue
        if not checksum_target.is_file():
            errors.append(f"SHA256SUMS references missing file: {relative}")
        elif verify_checksums and sha256_file(checksum_target) != expected_digest:
            errors.append(f"{relative}: SHA256 mismatch")
    indexed_npz: set[str] = set()
    total_frames = 0
    for row in master:
        relative = str(row.get("review_motion_path", ""))
        path = root / relative
        try:
            path.resolve().relative_to(root)
        except ValueError:
            errors.append(f"{row.get('review_id')}: review_motion_path escapes export root")
            continue
        if relative in indexed_npz:
            errors.append(f"duplicate review_motion_path: {relative}")
            continue
        indexed_npz.add(relative)
        if not path.is_file():
            errors.append(f"{row.get('review_id')}: missing review NPZ {path}")
            continue
        try:
            total_frames += validate_review_npz(path, row)
        except Exception as exc:  # report all damaged package members together
            errors.append(str(exc))
        if row.get("dataset") == "aistpp":
            if row.get("source_coordinate_system") != "right_handed_y_up_metric":
                errors.append(
                    f"{row.get('review_id')}: AIST++ source must be right_handed_y_up_metric"
                )
            if row.get("review_coordinate_system") != "right_handed_y_up_metric":
                errors.append(
                    f"{row.get('review_id')}: AIST++ review motion must be right_handed_y_up_metric"
                )
            if row.get("coordinate_transform") != "identity":
                errors.append(
                    f"{row.get('review_id')}: AIST++ review export must use identity transform"
                )
            try:
                with np.load(path, allow_pickle=False) as payload:
                    coordinate = str(np.asarray(payload["coordinate_system"]).item())
                    if coordinate != "right_handed_y_up_metric":
                        errors.append(
                            f"{row.get('review_id')}: NPZ coordinate_system={coordinate!r}"
                        )
                    validate_aist_metric_translation(
                        payload["transl"], sequence_id=str(row.get("sample_id"))
                    )
            except Exception as exc:
                errors.append(str(exc))
        if relative not in checksums:
            errors.append(f"{relative}: absent from SHA256SUMS")
        if row.get("review_sha256") != checksums.get(relative):
            errors.append(f"{relative}: master review_sha256 differs from SHA256SUMS")

    disk_npz = {path.relative_to(root).as_posix() for path in (root / "motions").rglob("*.npz")}
    if disk_npz != indexed_npz:
        errors.append(
            f"disk/master NPZ mismatch: orphan={sorted(disk_npz - indexed_npz)[:10]}, "
            f"missing={sorted(indexed_npz - disk_npz)[:10]}"
        )
    expected_checksum_paths = indexed_npz | {
        "index/master.jsonl",
        "index/source_fingerprints.json",
        "README.md",
    }
    if set(checksums) != expected_checksum_paths:
        errors.append(
            f"SHA256SUMS/index mismatch: extra={sorted(set(checksums) - expected_checksum_paths)[:10]}, "
            f"missing={sorted(expected_checksum_paths - set(checksums))[:10]}"
        )
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_MEDIA_SUFFIXES
    ]
    if forbidden:
        errors.append(
            f"review package contains forbidden music/video/model files: {forbidden[:10]}"
        )

    columns, decision_rows = read_csv(decisions_path)
    missing_columns = set(DECISION_COLUMNS) - set(columns)
    if missing_columns:
        errors.append(f"decision template missing columns: {sorted(missing_columns)}")
    template_ids = [row.get("review_id", "") for row in decision_rows]
    if template_ids != review_ids:
        errors.append("decision template order/identity differs from master index")
    if require_blank_decisions and any(row.get("decision", "").strip() for row in decision_rows):
        errors.append("fresh review package decision template is not blank")

    if expect_full_four_set:
        if len(master) != EXPECTED_FULL_COUNTS["total"]:
            errors.append(
                f"full export expected {EXPECTED_FULL_COUNTS['total']} samples, got {len(master)}"
            )
        if dict(counts_by_dataset) != EXPECTED_FULL_COUNTS["by_dataset"]:
            errors.append(f"full export dataset counts differ: {dict(counts_by_dataset)}")
        if dict(counts_by_split) != EXPECTED_FULL_COUNTS["by_split"]:
            errors.append(f"full export split counts differ: {dict(counts_by_split)}")

    report = {
        "export_root": str(root),
        "export_id": fingerprints.get("export_id"),
        "sample_count": len(master),
        "counts_by_dataset": dict(counts_by_dataset),
        "counts_by_split": dict(counts_by_split),
        "total_frames": total_frames,
        "total_hours": total_frames / 30.0 / 3600.0,
        "indexed_npz_count": len(indexed_npz),
        "forbidden_media_files": forbidden,
        "checksum_verification_enabled": verify_checksums,
        "blank_decisions_required": require_blank_decisions,
        "error_count": len(errors),
        "errors": errors,
        "final_pass": not errors,
    }
    if write_report:
        write_json(root / "reports" / "package_validation_report.json", report)
    if errors:
        raise ValueError("review package validation failed:\n- " + "\n- ".join(errors[:30]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--expect-full-four-set", action="store_true")
    parser.add_argument(
        "--allow-filled-decisions",
        action="store_true",
        help="validate package integrity without requiring a fresh blank decision template",
    )
    args = parser.parse_args()
    report = validate_package(
        args.export_root,
        verify_checksums=not args.skip_checksums,
        expect_full_four_set=args.expect_full_four_set,
        require_blank_decisions=not args.allow_filled_decisions,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
