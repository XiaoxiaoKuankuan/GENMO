#!/usr/bin/env python3
"""Validate reviewer decisions against an immutable motion export index."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.music_dance.curation.common import (  # noqa: E402
    DECISION_COLUMNS,
    ISSUE_CODES,
    VALID_DECISIONS,
    read_csv,
    read_jsonl,
    resolve_decisions_path,
    write_json,
)


def _parse_issue_codes(value: str) -> list[str]:
    return [token.strip() for token in re.split(r"[;,]", value) if token.strip()]


def validate_decisions(
    export_root: str | Path,
    decisions: str | Path,
    *,
    strict: bool,
    write_report: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    root = Path(export_root).expanduser().resolve()
    master = read_jsonl(root / "index" / "master.jsonl")
    master_by_id = {str(row["review_id"]): row for row in master}
    if len(master_by_id) != len(master):
        raise ValueError("master index contains duplicate review_id")
    decision_path = resolve_decisions_path(root, decisions)
    columns, rows = read_csv(decision_path)
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = set(DECISION_COLUMNS) - set(columns)
    if missing_columns:
        raise ValueError(f"decision CSV missing required columns: {sorted(missing_columns)}")

    decisions_by_id: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, 2):
        review_id = row.get("review_id", "").strip()
        if not review_id:
            errors.append(f"CSV line {line_number}: review_id is blank")
            continue
        if review_id in decisions_by_id:
            errors.append(f"CSV line {line_number}: duplicate review_id {review_id}")
            continue
        if review_id not in master_by_id:
            errors.append(f"CSV line {line_number}: unknown review_id {review_id}")
            continue
        expected = master_by_id[review_id]
        if row.get("export_id", "").strip() != str(expected["export_id"]):
            errors.append(f"{review_id}: export_id differs from immutable master")
        if row.get("dataset", "").strip() != str(expected["dataset"]):
            errors.append(f"{review_id}: dataset differs from immutable master")
        if row.get("sample_id", "").strip() != str(expected["sample_id"]):
            errors.append(f"{review_id}: sample_id differs from immutable master")
        try:
            duration = float(row.get("duration_sec", ""))
        except ValueError:
            errors.append(f"{review_id}: duration_sec is not numeric")
        else:
            if abs(duration - float(expected["duration_sec"])) > 1e-4:
                errors.append(f"{review_id}: duration_sec differs from immutable master")

        decision = row.get("decision", "").strip().lower()
        codes = _parse_issue_codes(row.get("issue_codes", ""))
        unknown_codes = sorted(set(codes) - ISSUE_CODES)
        if unknown_codes:
            errors.append(f"{review_id}: unknown issue codes {unknown_codes}")
        if decision and decision not in VALID_DECISIONS:
            errors.append(f"{review_id}: invalid decision {decision!r}")
        if decision == "reject" and not codes:
            errors.append(f"{review_id}: reject decision requires at least one issue code")
        if strict and decision in {"", "unsure"}:
            errors.append(f"{review_id}: strict mode does not allow blank/unsure decision")
        elif not strict and decision in {"", "unsure"}:
            warnings.append(f"{review_id}: pending second review")
        normalized = dict(row)
        normalized["decision"] = decision
        normalized["issue_codes"] = ";".join(codes)
        decisions_by_id[review_id] = normalized

    unknown_csv_ids = set(decisions_by_id) - set(master_by_id)
    missing_ids = set(master_by_id) - set(decisions_by_id)
    if unknown_csv_ids:
        errors.append(f"decision CSV has unknown IDs: {sorted(unknown_csv_ids)[:20]}")
    if strict and missing_ids:
        errors.append(f"decision CSV is missing {len(missing_ids)} review IDs")
    elif missing_ids:
        warnings.append(f"decision CSV is missing {len(missing_ids)} review IDs")

    decision_counts = Counter(
        row["decision"] or "blank" for row in decisions_by_id.values()
    )
    reason_counts: Counter[str] = Counter()
    for row in decisions_by_id.values():
        reason_counts.update(_parse_issue_codes(row["issue_codes"]))
    report = {
        "export_root": str(root),
        "decision_csv": str(decision_path),
        "strict": strict,
        "master_sample_count": len(master),
        "decision_row_count": len(rows),
        "recognized_decision_count": len(decisions_by_id),
        "missing_review_id_count": len(missing_ids),
        "decision_counts": dict(decision_counts),
        "issue_code_counts": dict(reason_counts),
        "warning_count": len(warnings),
        "warnings": warnings[:100],
        "error_count": len(errors),
        "errors": errors[:100],
        "final_pass": not errors,
    }
    if write_report:
        write_json(root / "reports" / "review_results_validation.json", report)
    if errors:
        raise ValueError("review result validation failed:\n- " + "\n- ".join(errors[:30]))
    return report, decisions_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report, _ = validate_decisions(
        args.export_root, args.decisions, strict=args.strict, write_report=True
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

