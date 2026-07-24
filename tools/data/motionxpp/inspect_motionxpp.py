#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""审计 Motion-X++ ZIP/目录结构、配对、schema、坐标和来源重叠。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionxpp.common import (  # noqa: E402
    IGNORED_RANGES,
    KNOWN_PROVENANCE,
    OFFICIAL_FPS,
    atomic_write_json,
    atomic_write_text,
    build_asset_index,
    discover_subsets,
    paired_asset_indices,
    parse_keypoint_asset,
    parse_motion_asset,
    parse_text_asset,
)

DEFAULT_ROOT = Path("inputs/Motion-Xplusplus")
DEFAULT_OUTPUT = Path("outputs/motionxpp_inspect")
OFFICIAL_DATASET_CARD = "https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus"
OFFICIAL_CODE = "https://github.com/IDEA-Research/Motion-X"


def _schema_for_subset(
    root: Path, subset: str, sample_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    motion, text, keypoints = paired_asset_indices(root, subset)
    paired = sorted(set(motion.assets) & set(text.assets))
    motion_samples: list[dict[str, Any]] = []
    text_samples: list[dict[str, Any]] = []
    keypoint_samples: list[dict[str, Any]] = []
    translation_parts: list[torch.Tensor] = []
    for stem in paired[:sample_count]:
        parsed = parse_motion_asset(motion.assets[stem])
        translation_parts.append(parsed["transl"])
        motion_samples.append(
            {
                "stem": stem,
                "source_path": motion.assets[stem].source_path,
                "raw_shape": parsed["raw_shape"],
                "body_shape": [
                    int(parsed["global_orient"].shape[0]),
                    66,
                ],
                "dtype": str(parsed["body_pose"].dtype),
                "source_format": parsed["source_format"],
                "embedded_fps": parsed["fps"],
                "betas_shape": list(parsed["betas"].shape),
            }
        )
        captions, metadata = parse_text_asset(text.assets[stem])
        text_samples.append(
            {
                "stem": stem,
                "source_path": text.assets[stem].source_path,
                "format": text.assets[stem].suffix,
                "caption_count": len(captions),
                "caption_example": captions[0]["caption"][:240],
                "metadata_keys": sorted(metadata),
            }
        )
        if keypoints is not None and stem in keypoints.assets:
            audited = parse_keypoint_asset(keypoints.assets[stem])
            keypoint_samples.append(
                {
                    "stem": stem,
                    "source_path": keypoints.assets[stem].source_path,
                    **{key: value for key, value in audited.items() if key != "kp2d"},
                }
            )

    translation_stats: dict[str, Any] | None = None
    if translation_parts:
        translations = torch.cat(translation_parts)
        translation_stats = {
            "mean_xyz": translations.mean(0).tolist(),
            "std_xyz": translations.std(0).tolist(),
            "range_xyz": (translations.amax(0) - translations.amin(0)).tolist(),
            "median_xyz": translations.median(0).values.tolist(),
        }
    kp_image_sizes = [item["image_size"] for item in keypoint_samples]
    kp_intrinsics = [item["has_camera_intrinsics"] for item in keypoint_samples]
    schema = {
        "subset": subset,
        "motion_samples": motion_samples,
        "text_samples": text_samples,
        "keypoint_samples": keypoint_samples,
        "translation_statistics": translation_stats,
        "fps": {
            "embedded_in_sample": any(item["embedded_fps"] for item in motion_samples),
            "official_value": OFFICIAL_FPS,
            "source": ("Motion-X official README: all motions have been unified in 30 fps"),
            "url": OFFICIAL_CODE,
        },
        "image_size_available": bool(kp_image_sizes)
        and all(value is not None for value in kp_image_sizes),
        "camera_intrinsics_available": bool(kp_intrinsics) and all(kp_intrinsics),
        "ignored_smplx_dimensions": IGNORED_RANGES,
    }
    keypoint_stems = set(keypoints.assets) if keypoints is not None else set()
    pairing = {
        "subset": subset,
        "motion_count": len(motion.assets),
        "text_count": len(text.assets),
        "keypoint_count": len(keypoint_stems),
        "motion_text_paired_count": len(set(motion.assets) & set(text.assets)),
        "all_three_paired_count": len(set(motion.assets) & set(text.assets) & keypoint_stems),
        "motion_without_text": sorted(set(motion.assets) - set(text.assets)),
        "text_without_motion": sorted(set(text.assets) - set(motion.assets)),
        "motion_without_keypoints": sorted(set(motion.assets) - keypoint_stems),
        "keypoints_without_motion": sorted(keypoint_stems - set(motion.assets)),
        "stem_collisions": {
            "motion": motion.collisions,
            "text": text.collisions,
            "keypoints": keypoints.collisions if keypoints is not None else {},
        },
        "pairable": bool(paired),
    }
    return schema, pairing


def inspect_motionxpp(
    root: str | Path,
    output_dir: str | Path,
    *,
    sample_count: int = 3,
) -> dict[str, Any]:
    """执行完整名字审计和小样本 schema 审计并写五个报告。"""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Motion-X++ root does not exist: {root}")
    discovered = discover_subsets(root)
    all_subsets = sorted(set().union(*map(set, discovered.values())))
    if not all_subsets:
        raise FileNotFoundError(f"No ZIP archives or extracted subsets were found under {root}")

    inventory_subsets: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    pairings: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    recommended: list[str] = []
    for subset in all_subsets:
        modality_rows: dict[str, Any] = {}
        for modality in ("motion", "text", "keypoints"):
            if subset not in discovered[modality]:
                modality_rows[modality] = {"present": False}
                continue
            index = build_asset_index(root, modality, subset)
            suffix_counts = Counter(ref.suffix for ref in index.assets.values())
            modality_rows[modality] = {
                "present": True,
                "containers": index.source_containers,
                "file_count": len(index.assets),
                "extensions": dict(sorted(suffix_counts.items())),
                "example_paths": [ref.source_path for ref in list(index.assets.values())[:3]],
                "collision_count": len(index.collisions),
            }
        inventory_subsets.append({"subset": subset, "modalities": modality_rows})
        if subset in discovered["motion"] and subset in discovered["text"]:
            schema, pairing = _schema_for_subset(root, subset, sample_count)
            schemas.append(schema)
            pairings.append(pairing)
        else:
            pairing = {
                "subset": subset,
                "pairable": False,
                "reason": "motion or semantic text modality is missing",
            }
            pairings.append(pairing)

        provenance = KNOWN_PROVENANCE.get(
            subset.lower(),
            {
                "source": "unknown",
                "overlaps": [],
                "reason": ("本地/官方已审计资料未给出该 subset 的明确来源；未凭模糊名字删除。"),
            },
        )
        pairable = bool(pairing.get("pairable"))
        has_overlap = bool(provenance["overlaps"])
        row = {
            "subset": subset,
            **provenance,
            "pairable_motion_and_semantic_text": pairable,
            "recommended": pairable and not has_overlap,
            "decision_basis": "explicit subset provenance plus exact stem pairing",
        }
        overlaps.append(row)
        if row["recommended"]:
            recommended.append(subset)

    inventory = {
        "root": str(root),
        "input_forms_supported": ["zip archives", "extracted directories"],
        "modalities": discovered,
        "subsets": inventory_subsets,
        "official_dataset_card": OFFICIAL_DATASET_CARD,
    }
    schema_report = {
        "root": str(root),
        "sample_count_per_subset": sample_count,
        "subsets": schemas,
        "coordinate_conclusion": {
            "target": "AY / Y-up",
            "source_up_axis": "y",
            "confidence": "high, but build CLI still requires explicit confirmation",
            "evidence": [
                (
                    "Motion-X official processing converts raw Z-up mocap to a standardized "
                    "Y-up representation with R_x(-90 degrees)."
                ),
                (
                    "Real smplx322 translation samples have their body-height baseline on "
                    "the second coordinate, consistent with +Y up."
                ),
            ],
            "build_argument": "--source-up-axis y",
        },
        "keypoint_training_conclusion": {
            "condition_on_keypoints": False,
            "reason": (
                "真实 keypoint JSON 有 COCO-WholeBody 像素坐标和置信度，但 images/顶层 "
                "没有 width/height，且没有相机内参；不能可靠构造校准的 2D 条件。"
            ),
        },
    }
    pairing_report = {
        "root": str(root),
        "subsets": pairings,
        "total_motion": sum(item.get("motion_count", 0) for item in pairings),
        "total_text": sum(item.get("text_count", 0) for item in pairings),
        "total_motion_text_pairs": sum(
            item.get("motion_text_paired_count", 0) for item in pairings
        ),
    }
    overlap_report = {
        "compared_against": ["AMASS", "HumanML3D/HumanML", "AIST++"],
        "method": (
            "使用 Motion-X 官方 subset provenance；不凭模糊目录名删除。构建阶段另做 "
            "归一化 motion 内容 hash 去重。"
        ),
        "subsets": overlaps,
        "recommended_subsets": recommended,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "inventory.json", inventory)
    atomic_write_json(output_dir / "pairing_report.json", pairing_report)
    atomic_write_json(output_dir / "schema_report.json", schema_report)
    atomic_write_json(output_dir / "overlap_report.json", overlap_report)
    atomic_write_text(
        output_dir / "recommended_subsets.txt",
        "".join(f"{subset}\n" for subset in recommended),
    )
    return {
        "inventory": inventory,
        "pairing": pairing_report,
        "schema": schema_report,
        "overlap": overlap_report,
        "recommended": recommended,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行审计并打印简短摘要。"""
    args = build_parser().parse_args(argv)
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    result = inspect_motionxpp(args.root, args.output_dir, sample_count=args.sample_count)
    print("=" * 72)
    print("Motion-X++ 数据审计完成")
    print(f"  root:             {Path(args.root).expanduser().resolve()}")
    print(f"  motion files:     {result['pairing']['total_motion']}")
    print(f"  text files:       {result['pairing']['total_text']}")
    print(f"  exact pairs:      {result['pairing']['total_motion_text_pairs']}")
    print(f"  recommended:      {', '.join(result['recommended'])}")
    print("  source FPS:       30 (官方统一)")
    print("  source up-axis:   Y（实际数据和官方预处理代码；构建时显式确认）")
    print("  keypoint cond:    disabled（缺 image size / calibrated K）")
    print(f"  reports:          {args.output_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
