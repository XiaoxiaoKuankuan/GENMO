#!/usr/bin/env python3
"""Export full-length motion-only NPZ files for external human review."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_aist_artifact,
    load_music_feature_tensor,
    validate_aist_metric_translation,
    validate_musicfeat_v2,
)
from gem.utils.smplx_utils import make_smplx  # noqa: E402
from tools.data.music_dance.curation.common import (  # noqa: E402
    DATASET_ORDER,
    DECISION_COLUMNS,
    DEFAULT_EXPORT_ID,
    SCHEMA_VERSION,
    SPLITS,
    atomic_save_npz,
    canonical_motion,
    git_commit,
    make_review_id,
    read_jsonl,
    resolve_relative,
    safe_torch_load,
    sha256_file,
    sha256_motion,
    validate_canonical_motion,
    validate_review_npz,
    write_csv,
    write_json,
    write_jsonl,
)

DEFAULT_ROOTS = {
    "aistpp": "inputs/AIST++",
    "aioz_gdance": "/data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE",
    "finedance": "/data0/user/liwei/datasets/music_dance_genmo/FineDance",
    "compas3d": "/data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D",
}


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _aist_music_id(sample_id: str) -> str | None:
    for token in sample_id.split("_"):
        if (
            len(token) == 4
            and token.startswith("m")
            and token[1:3].isalpha()
            and token[3].isdigit()
        ):
            return token
    return None


def collect_aist_records(
    root: Path, splits: tuple[str, ...]
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    annotation_path = root / "annot_aist_30fps.pt"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    annotation = load_aist_artifact(annotation_path)
    if not isinstance(annotation, dict):
        raise ValueError(f"{annotation_path}: annotation must be a dict")
    records: list[dict[str, Any]] = []
    artifacts = [_artifact(annotation_path, root)]
    for split in splits:
        split_path = root / f"{split}.pt"
        ids = load_aist_artifact(split_path)
        artifacts.append(_artifact(split_path, root))
        if not isinstance(ids, (list, tuple, set)):
            raise ValueError(f"{split_path}: split must be a sequence")
        for value in ids:
            sample_id = str(value)
            if sample_id not in annotation:
                raise ValueError(f"{sample_id}: absent from {annotation_path}")
            raw = annotation[sample_id]
            try:
                pose = torch.as_tensor(raw["smpl_pose_global"]).float()
                transl = torch.as_tensor(raw["smpl_trans_global"]).float()
            except KeyError as exc:
                raise ValueError(f"{sample_id}: missing AIST++ field {exc.args[0]}") from exc
            if pose.ndim != 2 or pose.shape[1] < 66 or transl.shape != (len(pose), 3):
                raise ValueError(
                    f"AIST++:{sample_id}: expected pose [T,>=66] and transl [T,3], "
                    f"got {tuple(pose.shape)} and {tuple(transl.shape)}"
                )
            if not torch.isfinite(pose[:, :66]).all() or not torch.isfinite(transl).all():
                raise ValueError(f"AIST++:{sample_id}: motion contains NaN or Inf")
            validate_aist_metric_translation(transl, sequence_id=sample_id)
            music_relative = f"musicfeat_v2/{sample_id}_musicfeat_fps30.pt"
            records.append(
                {
                    "dataset": "aistpp",
                    "sample_id": sample_id,
                    "split": split,
                    "group_id": sample_id,
                    "music_id": _aist_music_id(sample_id),
                    "motion_path": "annot_aist_30fps.pt",
                    "motion_key": sample_id,
                    "music_feature_path": music_relative,
                    # Official AIST++ SMPL pose/trans after dividing smpl_trans by
                    # smpl_scaling is already right-handed Y-up.  Do not apply the
                    # CoMPAS3D Z-up conversion to this artifact.
                    "source_coordinate_system": "right_handed_y_up_metric",
                    "source_manifest_row": None,
                    "source_num_frames": len(pose),
                }
            )
    source = {
        "dataset": "aistpp",
        "root": str(root),
        "kind": "aist_monolithic_annotation",
        "artifacts": artifacts,
    }
    return records, annotation, source


def collect_manifest_records(
    dataset: str, root: Path, splits: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for split in splits:
        manifest_path = root / "manifests" / f"{split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        artifacts.append(_artifact(manifest_path, root))
        for row in read_jsonl(manifest_path):
            if row.get("split") != split:
                raise ValueError(f"{manifest_path}: row split differs from filename")
            required = {"sample_id", "motion_path", "music_feature_path", "fps", "num_frames"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"{manifest_path}: missing fields {sorted(missing)}")
            if not np.isclose(float(row["fps"]), 30.0):
                raise ValueError(f"{row['sample_id']}: expected 30 FPS, got {row['fps']}")
            motion_path = resolve_relative(root, row["motion_path"], "motion_path")
            if not motion_path.is_file():
                raise FileNotFoundError(motion_path)
            group_id = row.get("group_id", row.get("sequence_id", row["sample_id"]))
            records.append(
                {
                    "dataset": dataset,
                    "sample_id": str(row["sample_id"]),
                    "split": split,
                    "group_id": str(group_id),
                    "music_id": row.get("song_id", row.get("song_name", group_id)),
                    "person_id": row.get("person_id"),
                    "role": row.get("role"),
                    "motion_path": str(row["motion_path"]),
                    "motion_key": None,
                    "music_feature_path": str(row["music_feature_path"]),
                    "source_coordinate_system": "right_handed_y_up_metric",
                    "source_manifest_row": row,
                    "source_num_frames": int(row["num_frames"]),
                }
            )
    source = {
        "dataset": dataset,
        "root": str(root),
        "kind": "jsonl_manifest",
        "artifacts": artifacts,
    }
    return records, source


def _smplx_model():
    return make_smplx("supermotion")


def _load_record_motion(
    record: dict[str, Any], roots: dict[str, Path], aist_annotation: dict[str, Any]
) -> dict[str, torch.Tensor]:
    if record["dataset"] == "aistpp":
        raw = aist_annotation[record["motion_key"]]
        pose = torch.as_tensor(raw["smpl_pose_global"]).float()
        return canonical_motion(
            {
                "pose": pose[:, :66],
                "transl": torch.as_tensor(raw["smpl_trans_global"]).float(),
                # This is the exact shape contract used by the current AIST loader.
                "betas": torch.zeros(len(pose), 10, dtype=torch.float32),
            },
            f"AIST++:{record['sample_id']}",
        )
    motion_path = resolve_relative(roots[record["dataset"]], record["motion_path"], "motion_path")
    motion = canonical_motion(safe_torch_load(motion_path), motion_path)
    if len(motion["pose"]) != int(record["source_num_frames"]):
        raise ValueError(
            f"{record['review_id']}: motion T={len(motion['pose'])} differs from "
            f"manifest {record['source_num_frames']}"
        )
    return motion


def validate_aist_identity_forward_equivalence(
    source: dict[str, torch.Tensor],
    review: dict[str, torch.Tensor],
    model: Any,
    *,
    max_frames: int = 3,
    tolerance: float = 2e-4,
) -> float:
    frames = len(source["pose"])
    indices = torch.linspace(0, frames - 1, min(max_frames, frames)).round().long().unique()
    source_inputs = {
        key: value[indices]
        for key, value in source.items()
        if key in {"global_orient", "body_pose", "transl", "betas"}
    }
    review_inputs = {
        key: value[indices]
        for key, value in review.items()
        if key in {"global_orient", "body_pose", "transl", "betas"}
    }
    with torch.no_grad():
        source_vertices = model(**source_inputs).vertices.cpu()
        review_vertices = model(**review_inputs).vertices.cpu()
    maximum = float((review_vertices - source_vertices).abs().max().item())
    if maximum > tolerance:
        raise ValueError(
            f"AIST++ Y-up identity SMPL-X forward equivalence failed: "
            f"max_abs_error={maximum:.6g}, tolerance={tolerance:.6g}"
        )
    return maximum


def _readme(export_id: str) -> str:
    return f"""# 四数据集动作人工筛选包 `{export_id}`

本目录只包含完整长度的 neutral SMPL-X body 参数，不包含音乐、EDGE35、WAV、视频或
SMPL-X 模型权重。每条 NPZ 为 30 FPS：

* `pose`: `[T,66]` axis-angle，其中 `:3` 是 global orient，`3:66` 是 21 个 body joints；
* `transl`: `[T,3]`，单位米；
* `betas`: `[T,10]`；
* `review_id`: 必须原样写回 `review/decisions.csv`。

审阅副本统一为右手系 Y-up。AIST++ 在 `smpl_trans / smpl_scaling` 后本来就是米制
Y-up，因此审阅副本必须保持 identity，不能再次做 Z-up 到 Y-up 的旋转。源训练数据没有
被修改。请只编辑 CSV 中的 `decision`、`issue_codes`、`reviewer`、`notes` 四列。
`decision` 只允许 `keep`、`reject`、`unsure`；reject 必须填写问题代码。
"""


def export_package(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    if (output_root / "index" / "master.jsonl").exists() and not args.overwrite:
        raise FileExistsError(
            f"completed export already exists: {output_root}; pass --overwrite to rebuild files"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    roots = {
        "aistpp": Path(args.aist_root).expanduser().resolve(),
        "aioz_gdance": Path(args.aioz_root).expanduser().resolve(),
        "finedance": Path(args.finedance_root).expanduser().resolve(),
        "compas3d": Path(args.compas3d_root).expanduser().resolve(),
    }
    splits = tuple(args.splits)
    records, aist_annotation, aist_source = collect_aist_records(roots["aistpp"], splits)
    sources = {"aistpp": aist_source}
    for dataset in DATASET_ORDER[1:]:
        dataset_records, source = collect_manifest_records(dataset, roots[dataset], splits)
        records.extend(dataset_records)
        sources[dataset] = source

    if args.limit_per_dataset is not None:
        limited: list[dict[str, Any]] = []
        for dataset in DATASET_ORDER:
            limited.extend(
                [r for r in records if r["dataset"] == dataset][: args.limit_per_dataset]
            )
        records = limited

    seen_review: set[str] = set()
    seen_samples: set[tuple[str, str]] = set()
    music_cache: dict[Path, int] = {}
    for record in records:
        key = (record["dataset"], record["sample_id"])
        if key in seen_samples:
            raise ValueError(f"duplicate source sample: {key}")
        seen_samples.add(key)
        review_id = make_review_id(*key)
        if review_id in seen_review:
            raise ValueError(f"duplicate review_id: {review_id}")
        seen_review.add(review_id)
        record["review_id"] = review_id
        music_path = resolve_relative(
            roots[record["dataset"]], record["music_feature_path"], "music_feature_path"
        )
        if music_path not in music_cache:
            features = load_music_feature_tensor(music_path)
            validate_musicfeat_v2(features, source=music_path)
            music_cache[music_path] = int(features.shape[0])
        record["music_num_frames"] = music_cache[music_path]
        difference = abs(int(record["source_num_frames"]) - music_cache[music_path])
        if record["dataset"] == "aistpp":
            if difference > 2:
                raise ValueError(
                    f"{review_id}: AIST++ motion/music mismatch is {difference} frames"
                )
        elif difference != 0:
            raise ValueError(f"{review_id}: canonical motion/music mismatch is {difference} frames")

    smplx_model = None
    if args.aist_forward_checks and any(r["dataset"] == "aistpp" for r in records):
        smplx_model = _smplx_model()
    rng = random.Random(args.seed)
    aist_checks = [r for r in records if r["dataset"] == "aistpp"]
    checked_ids = {
        row["review_id"]
        for row in rng.sample(aist_checks, min(args.aist_forward_checks, len(aist_checks)))
    }
    forward_errors: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []
    checksum_rows: list[tuple[str, str]] = []
    total = len(records)
    for index, record in enumerate(records, 1):
        source_motion = _load_record_motion(record, roots, aist_annotation)
        # All four canonical conversion artifacts are right-handed Y-up.  In
        # particular AIST++ is already Y-up after its per-sequence metric scale
        # correction, so every review export is an identity copy.
        review_motion = {key: value.clone() for key, value in source_motion.items()}
        transform_name = "identity"
        frames = validate_canonical_motion(review_motion, record["review_id"])
        if record["review_id"] in checked_ids:
            assert smplx_model is not None
            maximum = validate_aist_identity_forward_equivalence(
                source_motion, review_motion, smplx_model
            )
            forward_errors.append({"review_id": record["review_id"], "max_abs_error": maximum})

        npz_relative = Path("motions") / record["dataset"] / f"{record['sample_id']}.npz"
        npz_path = output_root / npz_relative
        atomic_save_npz(
            npz_path,
            pose=review_motion["pose"].numpy(),
            transl=review_motion["transl"].numpy(),
            betas=review_motion["betas"].numpy(),
            fps=np.asarray(30.0, dtype=np.float32),
            num_frames=np.asarray(frames, dtype=np.int64),
            review_id=np.asarray(record["review_id"]),
            dataset=np.asarray(record["dataset"]),
            sample_id=np.asarray(record["sample_id"]),
            coordinate_system=np.asarray(
                "right_handed_y_up_metric"
                if args.review_coordinate == "y_up"
                else record["source_coordinate_system"]
            ),
        )
        validate_review_npz(npz_path)
        npz_sha = sha256_file(npz_path)
        source_root = roots[record["dataset"]]
        if record["dataset"] == "aistpp":
            source_sha = sha256_motion(
                source_motion["pose"], source_motion["transl"], source_motion["betas"]
            )
        else:
            source_sha = sha256_file(
                resolve_relative(source_root, record["motion_path"], "motion_path")
            )
        row = {
            "schema_version": SCHEMA_VERSION,
            "export_id": args.export_id,
            **record,
            "source_root": str(source_root),
            "review_coordinate_system": (
                "right_handed_y_up_metric"
                if args.review_coordinate == "y_up"
                else record["source_coordinate_system"]
            ),
            "coordinate_transform": transform_name,
            "num_frames": frames,
            "fps": 30.0,
            "duration_sec": frames / 30.0,
            "review_motion_path": npz_relative.as_posix(),
            "source_sha256": source_sha,
            "review_sha256": npz_sha,
            "music_key": f"{record['dataset']}::{record['music_feature_path']}",
        }
        master_rows.append(row)
        checksum_rows.append((npz_sha, npz_relative.as_posix()))
        if index % 100 == 0 or index == total:
            print(f"[export] {index}/{total}")

    counts_by_dataset = Counter(row["dataset"] for row in master_rows)
    counts_by_split = Counter(row["split"] for row in master_rows)
    music_by_dataset = {
        dataset: len({row["music_key"] for row in master_rows if row["dataset"] == dataset})
        for dataset in DATASET_ORDER
    }
    source_fingerprints = {
        "schema_version": SCHEMA_VERSION,
        "export_id": args.export_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(REPO_ROOT),
        "review_coordinate": args.review_coordinate,
        "splits": list(splits),
        "sources": sources,
        "counts_by_dataset": dict(counts_by_dataset),
        "counts_by_split": dict(counts_by_split),
        "unique_music_features_by_dataset": music_by_dataset,
        "sample_count": len(master_rows),
        "total_frames": sum(int(row["num_frames"]) for row in master_rows),
    }
    write_jsonl(output_root / "index" / "master.jsonl", master_rows)
    write_json(output_root / "index" / "source_fingerprints.json", source_fingerprints)
    decisions = [
        {
            "export_id": row["export_id"],
            "review_id": row["review_id"],
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "duration_sec": f"{row['duration_sec']:.6f}",
            "decision": "",
            "issue_codes": "",
            "reviewer": "",
            "notes": "",
        }
        for row in master_rows
    ]
    write_csv(output_root / "review" / "decisions.csv", decisions, DECISION_COLUMNS)
    from tools.data.music_dance.curation.common import atomic_write_text

    atomic_write_text(output_root / "README.md", _readme(args.export_id))
    immutable_files = [
        output_root / "index" / "master.jsonl",
        output_root / "index" / "source_fingerprints.json",
        output_root / "README.md",
    ]
    for path in immutable_files:
        checksum_rows.append((sha256_file(path), path.relative_to(output_root).as_posix()))
    checksum_rows.sort(key=lambda item: item[1])
    atomic_write_text(
        output_root / "index" / "SHA256SUMS",
        "".join(f"{digest}  {relative}\n" for digest, relative in checksum_rows),
    )
    report = {
        **source_fingerprints,
        "total_hours": source_fingerprints["total_frames"] / 30.0 / 3600.0,
        "aist_forward_identity_checks": forward_errors,
        "aist_forward_max_abs_error": max(
            (row["max_abs_error"] for row in forward_errors), default=0.0
        ),
        "package_contains_music_files": False,
        "final_pass": True,
    }
    write_json(output_root / "reports" / "export_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--export-id", default=DEFAULT_EXPORT_ID)
    parser.add_argument("--aist-root", default=DEFAULT_ROOTS["aistpp"])
    parser.add_argument("--aioz-root", default=DEFAULT_ROOTS["aioz_gdance"])
    parser.add_argument("--finedance-root", default=DEFAULT_ROOTS["finedance"])
    parser.add_argument("--compas3d-root", default=DEFAULT_ROOTS["compas3d"])
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--review-coordinate", choices=("y_up", "source"), default="y_up")
    parser.add_argument("--aist-forward-checks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.aist_forward_checks < 0:
        raise ValueError("--aist-forward-checks must be non-negative")
    if args.limit_per_dataset is not None and args.limit_per_dataset <= 0:
        raise ValueError("--limit-per-dataset must be positive")
    export_package(args)


if __name__ == "__main__":
    main()
