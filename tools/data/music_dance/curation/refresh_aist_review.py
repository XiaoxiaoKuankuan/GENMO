#!/usr/bin/env python3
"""Safely replace only AIST++ motions in an existing four-set review package.

This command exists for correcting a published AIST++ review export without
rewriting the other three datasets or a reviewer-edited decisions.csv.  It
stages every replacement first, verifies the unchanged package members, keeps
a recoverable backup outside the export root, and validates the complete
package after the atomic directory swap.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from tools.data.music_dance.curation.common import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_save_npz,
    atomic_write_text,
    git_commit,
    make_review_id,
    read_csv,
    read_jsonl,
    resolve_relative,
    sha256_file,
    sha256_motion,
    validate_canonical_motion,
    validate_review_npz,
    write_json,
    write_jsonl,
)
from tools.data.music_dance.curation.export_motion_review import (  # noqa: E402
    DEFAULT_ROOTS,
    _load_record_motion,
    _readme,
    _smplx_model,
    collect_aist_records,
    validate_aist_identity_forward_equivalence,
)
from tools.data.music_dance.curation.validate_review_package import (  # noqa: E402
    _read_checksums,
    validate_package,
)


def _default_backup_root(export_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return export_root.parent / "backups" / f"{export_root.name}_aistpp_before_fix_{stamp}"


def _verify_decision_identity(export_root: Path, master: list[dict[str, Any]]) -> None:
    columns, rows = read_csv(export_root / "review" / "decisions.csv")
    required = {"export_id", "review_id", "dataset", "sample_id", "duration_sec"}
    if not required <= set(columns):
        raise ValueError(f"decisions.csv is missing columns: {sorted(required - set(columns))}")
    expected_ids = [str(row["review_id"]) for row in master]
    actual_ids = [str(row.get("review_id", "")) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("decisions.csv order/identity differs from master.jsonl")


def _stage_corrected_aist(
    *,
    stage_root: Path,
    export_root: Path,
    aist_root: Path,
    existing_master: list[dict[str, Any]],
    existing_fingerprints: dict[str, Any],
    aist_forward_checks: int,
    seed: int,
    verify_existing_checksums: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    splits = tuple(existing_fingerprints.get("splits", ("train", "val", "test")))
    records, annotation, aist_source = collect_aist_records(aist_root, splits)
    old_aist = {
        str(row["review_id"]): row for row in existing_master if row.get("dataset") == "aistpp"
    }
    for record in records:
        record["review_id"] = make_review_id("aistpp", record["sample_id"])
    new_by_id = {str(record["review_id"]): record for record in records}
    if set(new_by_id) != set(old_aist):
        raise ValueError(
            "corrected AIST++ identities differ from the existing review package: "
            f"added={sorted(set(new_by_id) - set(old_aist))[:10]}, "
            f"missing={sorted(set(old_aist) - set(new_by_id))[:10]}"
        )

    music_lengths: dict[Path, int] = {}
    for record in records:
        previous = old_aist[record["review_id"]]
        immutable_pairs = {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "music_feature_path": record["music_feature_path"],
            "source_num_frames": int(record["source_num_frames"]),
        }
        for field, expected in immutable_pairs.items():
            actual = previous.get(field)
            if field == "source_num_frames":
                actual = int(actual)
            if actual != expected:
                raise ValueError(
                    f"{record['review_id']}: corrected AIST++ {field}={expected!r} "
                    f"differs from existing package {actual!r}"
                )
        music_path = resolve_relative(aist_root, record["music_feature_path"], "music_feature_path")
        if music_path not in music_lengths:
            music = load_music_feature_tensor(music_path)
            validate_musicfeat_v2(music, source=music_path)
            music_lengths[music_path] = int(music.shape[0])
        record["music_num_frames"] = music_lengths[music_path]
        mismatch = abs(int(record["source_num_frames"]) - record["music_num_frames"])
        if mismatch > 2:
            raise ValueError(f"{record['review_id']}: motion/music mismatch is {mismatch} frames")

    rng = random.Random(seed)
    checked_ids = {
        row["review_id"] for row in rng.sample(records, min(aist_forward_checks, len(records)))
    }
    model = _smplx_model() if checked_ids else None
    forward_checks: list[dict[str, Any]] = []
    staged_rows: dict[str, dict[str, Any]] = {}
    staged_aist = stage_root / "motions" / "aistpp"
    staged_aist.mkdir(parents=True, exist_ok=True)
    roots = {"aistpp": aist_root}
    total = len(records)
    for index, record in enumerate(records, 1):
        source_motion = _load_record_motion(record, roots, annotation)
        review_motion = {key: value.clone() for key, value in source_motion.items()}
        frames = validate_canonical_motion(review_motion, record["review_id"])
        if record["review_id"] in checked_ids:
            assert model is not None
            maximum = validate_aist_identity_forward_equivalence(
                source_motion, review_motion, model
            )
            forward_checks.append({"review_id": record["review_id"], "max_abs_error": maximum})
        relative = Path("motions") / "aistpp" / f"{record['sample_id']}.npz"
        target = stage_root / relative
        atomic_save_npz(
            target,
            pose=review_motion["pose"].numpy(),
            transl=review_motion["transl"].numpy(),
            betas=review_motion["betas"].numpy(),
            fps=np.asarray(30.0, dtype=np.float32),
            num_frames=np.asarray(frames, dtype=np.int64),
            review_id=np.asarray(record["review_id"]),
            dataset=np.asarray("aistpp"),
            sample_id=np.asarray(record["sample_id"]),
            coordinate_system=np.asarray("right_handed_y_up_metric"),
        )
        validate_review_npz(target)
        updated = dict(old_aist[record["review_id"]])
        updated.update(record)
        updated.update(
            schema_version=SCHEMA_VERSION,
            source_root=str(aist_root),
            source_coordinate_system="right_handed_y_up_metric",
            review_coordinate_system="right_handed_y_up_metric",
            coordinate_transform="identity",
            num_frames=frames,
            fps=30.0,
            duration_sec=frames / 30.0,
            review_motion_path=relative.as_posix(),
            source_sha256=sha256_motion(
                source_motion["pose"], source_motion["transl"], source_motion["betas"]
            ),
            review_sha256=sha256_file(target),
            music_key=f"aistpp::{record['music_feature_path']}",
        )
        staged_rows[record["review_id"]] = updated
        if index % 100 == 0 or index == total:
            print(f"[refresh-aist] staged {index}/{total}")

    updated_master = [
        staged_rows[str(row["review_id"])] if row.get("dataset") == "aistpp" else dict(row)
        for row in existing_master
    ]
    fingerprints = copy.deepcopy(existing_fingerprints)
    fingerprints["git_commit"] = git_commit(REPO_ROOT)
    fingerprints["review_coordinate"] = "y_up"
    fingerprints["aist_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    fingerprints["aist_coordinate_correction"] = {
        "source_coordinate_system": "right_handed_y_up_metric",
        "review_coordinate_system": "right_handed_y_up_metric",
        "coordinate_transform": "identity",
        "translation_unit": "metre",
    }
    fingerprints.setdefault("sources", {})["aistpp"] = aist_source
    fingerprints["counts_by_dataset"] = dict(Counter(str(row["dataset"]) for row in updated_master))
    fingerprints["counts_by_split"] = dict(Counter(str(row["split"]) for row in updated_master))
    fingerprints["sample_count"] = len(updated_master)
    fingerprints["total_frames"] = sum(int(row["num_frames"]) for row in updated_master)

    write_jsonl(stage_root / "index" / "master.jsonl", updated_master)
    write_json(stage_root / "index" / "source_fingerprints.json", fingerprints)
    atomic_write_text(stage_root / "README.md", _readme(str(fingerprints["export_id"])))

    old_checksums = _read_checksums(export_root / "index" / "SHA256SUMS")
    checksum_rows: list[tuple[str, str]] = []
    unchanged_verified = 0
    for row in updated_master:
        relative = str(row["review_motion_path"])
        if row["dataset"] == "aistpp":
            digest = str(row["review_sha256"])
        else:
            path = export_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            expected = old_checksums.get(relative)
            if expected is None:
                raise ValueError(f"unchanged package member is absent from SHA256SUMS: {relative}")
            digest = sha256_file(path) if verify_existing_checksums else expected
            if verify_existing_checksums and digest != expected:
                raise ValueError(f"refusing refresh because unchanged file is damaged: {relative}")
            if str(row.get("review_sha256")) != expected:
                raise ValueError(f"unchanged master/checksum mismatch: {relative}")
            unchanged_verified += 1
        checksum_rows.append((digest, relative))
    for relative in ("index/master.jsonl", "index/source_fingerprints.json", "README.md"):
        checksum_rows.append((sha256_file(stage_root / relative), relative))
    checksum_rows.sort(key=lambda item: item[1])
    atomic_write_text(
        stage_root / "index" / "SHA256SUMS",
        "".join(f"{digest}  {relative}\n" for digest, relative in checksum_rows),
    )
    details = {
        "aist_sample_count": len(records),
        "unchanged_motion_count": len(existing_master) - len(records),
        "unchanged_motion_checksum_count": unchanged_verified,
        "aist_forward_identity_checks": forward_checks,
        "aist_forward_max_abs_error": max(
            (row["max_abs_error"] for row in forward_checks), default=0.0
        ),
    }
    return updated_master, fingerprints, details


def refresh_aist_review(args: argparse.Namespace) -> dict[str, Any]:
    export_root = Path(args.export_root).expanduser().resolve()
    aist_root = Path(args.aist_root).expanduser().resolve()
    if not args.overwrite:
        raise ValueError("AIST++ replacement requires explicit --overwrite")
    required = (
        export_root / "motions" / "aistpp",
        export_root / "index" / "master.jsonl",
        export_root / "index" / "source_fingerprints.json",
        export_root / "index" / "SHA256SUMS",
        export_root / "review" / "decisions.csv",
        export_root / "README.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"existing review package is incomplete: {missing}")
    existing_master = read_jsonl(export_root / "index" / "master.jsonl")
    existing_fingerprints = json.loads(
        (export_root / "index" / "source_fingerprints.json").read_text(encoding="utf-8")
    )
    _verify_decision_identity(export_root, existing_master)
    decisions_before = sha256_file(export_root / "review" / "decisions.csv")

    stage_path = Path(
        tempfile.mkdtemp(prefix=f".{export_root.name}.aist-refresh.", dir=export_root.parent)
    )
    backup_root = (
        Path(args.backup_root).expanduser().resolve()
        if args.backup_root
        else _default_backup_root(export_root)
    )
    if backup_root.exists():
        shutil.rmtree(stage_path, ignore_errors=True)
        raise FileExistsError(f"backup target already exists: {backup_root}")
    metadata_relatives = (
        Path("index/master.jsonl"),
        Path("index/source_fingerprints.json"),
        Path("index/SHA256SUMS"),
        Path("README.md"),
    )
    installed_new_aist = False
    moved_old_aist = False
    try:
        updated_master, fingerprints, details = _stage_corrected_aist(
            stage_root=stage_path,
            export_root=export_root,
            aist_root=aist_root,
            existing_master=existing_master,
            existing_fingerprints=existing_fingerprints,
            aist_forward_checks=args.aist_forward_checks,
            seed=args.seed,
            verify_existing_checksums=not args.skip_existing_checksums,
        )
        backup_root.mkdir(parents=True)
        backup_aist = backup_root / "motions" / "aistpp"
        backup_aist.parent.mkdir(parents=True)
        for relative in metadata_relatives:
            source = export_root / relative
            target = backup_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(export_root / "review" / "decisions.csv", backup_root / "decisions.csv")

        current_aist = export_root / "motions" / "aistpp"
        os.replace(current_aist, backup_aist)
        moved_old_aist = True
        os.replace(stage_path / "motions" / "aistpp", current_aist)
        installed_new_aist = True
        for relative in metadata_relatives:
            os.replace(stage_path / relative, export_root / relative)

        if sha256_file(export_root / "review" / "decisions.csv") != decisions_before:
            raise RuntimeError("decisions.csv changed during AIST++ refresh")
        package = validate_package(
            export_root,
            verify_checksums=True,
            expect_full_four_set=len(updated_master) == 7286,
            require_blank_decisions=False,
            write_report=True,
        )
        report = {
            "export_root": str(export_root),
            "aist_root": str(aist_root),
            "backup_root": str(backup_root),
            "decisions_sha256_before": decisions_before,
            "decisions_sha256_after": sha256_file(export_root / "review" / "decisions.csv"),
            "source_artifact_sha256": next(
                item["sha256"]
                for item in fingerprints["sources"]["aistpp"]["artifacts"]
                if item["path"] == "annot_aist_30fps.pt"
            ),
            **details,
            "package_validation": package,
            "final_pass": True,
        }
        write_json(export_root / "reports" / "aist_refresh_report.json", report)
        old_export_report = export_root / "reports" / "export_report.json"
        if old_export_report.is_file():
            value = json.loads(old_export_report.read_text(encoding="utf-8"))
            value.update(
                git_commit=fingerprints.get("git_commit"),
                aist_refreshed_at_utc=fingerprints["aist_refreshed_at_utc"],
                aist_coordinate_correction=fingerprints["aist_coordinate_correction"],
                aist_forward_identity_checks=details["aist_forward_identity_checks"],
                aist_forward_max_abs_error=details["aist_forward_max_abs_error"],
            )
            value.pop("aist_forward_equivalence_checks", None)
            write_json(old_export_report, value)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report
    except Exception:
        # Roll back the only destructive part if installation began.  The old
        # files remain recoverable even if rollback itself encounters trouble.
        if moved_old_aist:
            current_aist = export_root / "motions" / "aistpp"
            failed_new = backup_root / "failed_new_aistpp"
            if installed_new_aist and current_aist.exists():
                os.replace(current_aist, failed_new)
            backup_aist = backup_root / "motions" / "aistpp"
            if backup_aist.exists():
                os.replace(backup_aist, current_aist)
            for relative in metadata_relatives:
                backup = backup_root / relative
                if backup.is_file():
                    shutil.copy2(backup, export_root / relative)
        raise
    finally:
        shutil.rmtree(stage_path, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--aist-root", default=DEFAULT_ROOTS["aistpp"])
    parser.add_argument("--backup-root")
    parser.add_argument("--aist-forward-checks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing-checksums", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.aist_forward_checks < 0:
        raise ValueError("--aist-forward-checks must be non-negative")
    refresh_aist_review(args)


if __name__ == "__main__":
    main()
