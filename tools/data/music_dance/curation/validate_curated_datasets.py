#!/usr/bin/env python3
"""核验当前 BUMI 音乐生产上游的四库筛选结果。

检查保留样本、原始划分、动作与 EDGE35 对齐、音乐去重及文件边界。旧 SMPL
训练 loader/151D encoder smoke 已退役；本工具只验源数据，不实例化模型或训练配置。
"""

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

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_aist_artifact,
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from tools.data.music_dance.curation.common import (  # noqa: E402
    DATASET_DIRS,
    DATASET_ORDER,
    SPLITS,
    canonical_motion,
    read_jsonl,
    resolve_relative,
    safe_torch_load,
    validate_canonical_motion,
    write_json,
)


def _validate_aist(root: Path, rows: list[dict[str, Any]], errors: list[str]) -> None:
    try:
        annotation = load_aist_artifact(root / "annot_aist_30fps.pt")
    except Exception as exc:
        errors.append(f"AIST++ annotation load failed: {exc}")
        return
    expected_by_split = {
        split: [row["sample_id"] for row in rows if row["split"] == split]
        for split in SPLITS
    }
    for split in SPLITS:
        try:
            actual = list(load_aist_artifact(root / f"{split}.pt"))
        except Exception as exc:
            errors.append(f"AIST++ {split}.pt load failed: {exc}")
            continue
        if actual != expected_by_split[split]:
            errors.append(f"AIST++ {split}.pt identities/order differ from accepted master")
    music_cache: dict[str, int] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in annotation:
            errors.append(f"AIST++ accepted sample absent from annotation: {sample_id}")
            continue
        motion = annotation[sample_id]
        try:
            pose = np.asarray(motion["smpl_pose_global"])
            transl = np.asarray(motion["smpl_trans_global"])
            if pose.ndim != 2 or pose.shape[1] < 66 or transl.shape != (len(pose), 3):
                raise ValueError(f"pose/transl shapes are {pose.shape}/{transl.shape}")
            if not np.isfinite(pose[:, :66]).all() or not np.isfinite(transl).all():
                raise ValueError("motion contains NaN or Inf")
            music_path = resolve_relative(root, row["music_feature_path"], "music_feature_path")
            if row["music_feature_path"] not in music_cache:
                features = load_music_feature_tensor(music_path)
                validate_musicfeat_v2(features, music_path)
                music_cache[row["music_feature_path"]] = len(features)
            mismatch = abs(len(pose) - music_cache[row["music_feature_path"]])
            if mismatch > 2:
                raise ValueError(f"motion/music mismatch is {mismatch} frames")
        except Exception as exc:
            errors.append(f"AIST++ {sample_id}: {exc}")


def _validate_manifest_dataset(
    dataset: str, root: Path, rows: list[dict[str, Any]], errors: list[str]
) -> None:
    expected_by_split = {
        split: [row["sample_id"] for row in rows if row["split"] == split]
        for split in SPLITS
    }
    actual_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = root / "manifests" / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"{dataset}: missing curated manifest {path}")
            continue
        values = read_jsonl(path)
        if [row.get("sample_id") for row in values] != expected_by_split[split]:
            errors.append(f"{dataset}: {split} manifest identities/order differ from accepted master")
        actual_rows.extend(values)
    music_cache: dict[str, int] = {}
    for row in actual_rows:
        sample_id = str(row.get("sample_id"))
        try:
            motion_path = resolve_relative(root, row["motion_path"], "motion_path")
            motion = canonical_motion(safe_torch_load(motion_path), motion_path)
            frames = validate_canonical_motion(motion, motion_path)
            if frames != int(row["num_frames"]):
                raise ValueError(f"motion T={frames} differs from manifest {row['num_frames']}")
            music_path = resolve_relative(root, row["music_feature_path"], "music_feature_path")
            if row["music_feature_path"] not in music_cache:
                features = load_music_feature_tensor(music_path)
                validate_musicfeat_v2(features, music_path)
                music_cache[row["music_feature_path"]] = len(features)
            if music_cache[row["music_feature_path"]] != frames:
                raise ValueError(
                    f"motion/music lengths differ: {frames}/{music_cache[row['music_feature_path']]}"
                )
        except Exception as exc:
            errors.append(f"{dataset} {sample_id}: {exc}")


def validate_curated(root: str | Path, *, strict: bool = False) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    accepted_path = root / "reports" / "accepted_master.jsonl"
    rejected_path = root / "reports" / "rejected_samples.jsonl"
    report_path = root / "reports" / "curation_report.json"
    if not accepted_path.is_file() or not rejected_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("curated reports/accepted_master, rejected_samples or report is missing")
    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    curation_report = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    accepted_ids = {row["review_id"] for row in accepted}
    rejected_ids = {row["review_id"] for row in rejected}
    if len(accepted_ids) != len(accepted):
        errors.append("accepted master contains duplicate review_id")
    if accepted_ids & rejected_ids:
        errors.append("accepted and rejected review_id sets overlap")
    split_membership: dict[str, str] = {}
    for row in accepted:
        if row["review_id"] in split_membership:
            errors.append(f"split leakage: {row['review_id']}")
        split_membership[row["review_id"]] = row["split"]

    rows_by_dataset = {
        dataset: [row for row in accepted if row["dataset"] == dataset]
        for dataset in DATASET_ORDER
    }
    _validate_aist(root / DATASET_DIRS["aistpp"], rows_by_dataset["aistpp"], errors)
    for dataset in DATASET_ORDER[1:]:
        _validate_manifest_dataset(
            dataset, root / DATASET_DIRS[dataset], rows_by_dataset[dataset], errors
        )

    expected_music = {
        (DATASET_DIRS[row["dataset"]], row["music_feature_path"])
        for row in accepted
    }
    actual_music: set[tuple[str, str]] = set()
    for dataset in DATASET_ORDER:
        dataset_root = root / DATASET_DIRS[dataset]
        feature_root = dataset_root / "musicfeat_v2"
        if feature_root.exists():
            actual_music.update(
                (DATASET_DIRS[dataset], path.relative_to(dataset_root).as_posix())
                for path in feature_root.rglob("*.pt")
            )
    if actual_music != expected_music:
        errors.append(
            f"active music/master mismatch: orphan={len(actual_music-expected_music)}, "
            f"missing={len(expected_music-actual_music)}"
        )
    forbidden_raw_audio = [
        path.relative_to(root).as_posix()
        for suffix in ("*.wav", "*.mp3", "*.m4a", "*.aac", "*.flac")
        for path in root.rglob(suffix)
    ]
    if forbidden_raw_audio:
        errors.append(f"curated root unexpectedly contains raw audio: {forbidden_raw_audio[:10]}")

    if int(curation_report.get("accepted_sample_count", -1)) != len(accepted):
        errors.append("curation report accepted count differs from accepted master")
    if strict and curation_report.get("pending_sample_count") != 0:
        errors.append("strict curated validation requires zero pending samples")
    report = {
        "curated_root": str(root),
        "accepted_sample_count": len(accepted),
        "rejected_sample_count": len(rejected),
        "accepted_counts_by_dataset": dict(Counter(row["dataset"] for row in accepted)),
        "accepted_counts_by_split": dict(Counter(row["split"] for row in accepted)),
        "accepted_hours": sum(int(row["num_frames"]) for row in accepted) / 30.0 / 3600.0,
        "active_music_feature_count": len(actual_music),
        "raw_audio_file_count": len(forbidden_raw_audio),
        "validation_scope": "source_motion_music_only",
        "error_count": len(errors),
        "errors": errors[:100],
        "final_pass": not errors,
    }
    write_json(root / "reports" / "validation_report.json", report)
    if errors:
        raise ValueError("curated validation failed:\n- " + "\n- ".join(errors[:30]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = validate_curated(args.root, strict=args.strict)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
