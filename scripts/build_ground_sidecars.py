#!/usr/bin/env python3
"""Build or verify physics-v1 per-sequence ground sidecars.

Examples:
    python scripts/build_ground_sidecars.py --kind manifest \
      --root /data0/user/liwei/datasets/music_dance_genmo/FineDance

    python scripts/build_ground_sidecars.py --kind aist \
      --root inputs/AIST++ --annot-file annot_aist_30fps.pt --split-file train.pt

    python scripts/build_ground_sidecars.py --kind manifest --root DATASET \
      --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import load_aist_artifact  # noqa: E402
from gem.datasets.music_dance.music_dance_smpl import (  # noqa: E402
    MusicDanceSmplDataset,
    _read_jsonl,
    _resolve_relative,
    _safe_torch_load,
)
from gem.utils.ground_sidecar import (  # noqa: E402
    SOLE_V437_INDICES,
    estimate_ground_height,
    load_ground_sidecar,
    make_ground_record,
    sha256_file,
)
from gem.utils.smplx_utils import make_smplx  # noqa: E402


def _manifest_sources(root: Path, split: str) -> Iterator[dict[str, Any]]:
    rows = _read_jsonl(root / "manifests" / f"{split}.jsonl")
    for row in rows:
        motion_path = _resolve_relative(root, row["motion_path"], "motion_path")
        motion = MusicDanceSmplDataset._canonical_motion(
            _safe_torch_load(motion_path), motion_path
        )
        yield {
            "sample_id": str(row["sample_id"]),
            "source_path": motion_path,
            "source_sha256": sha256_file(motion_path),
            "num_frames": int(row["num_frames"]),
            "motion": motion,
        }


def _aist_sources(
    root: Path, annot_file: str, split_file: str
) -> Iterator[dict[str, Any]]:
    annot_path = root / annot_file
    motion_files = load_aist_artifact(annot_path)
    split_ids = load_aist_artifact(root / split_file)
    artifact_sha256 = sha256_file(annot_path)
    for sequence_id in split_ids:
        if sequence_id not in motion_files:
            raise ValueError(f"{sequence_id}: split ID is missing from {annot_path}")
        payload = motion_files[sequence_id]
        pose = torch.as_tensor(payload["smpl_pose_global"]).float()
        translation = torch.as_tensor(payload["smpl_trans_global"]).float()
        num_frames = int(payload["bbox_xyxy"].shape[0])
        # The official AIST++ artifact stores all 24 SMPL joint rotations as
        # axis-angle vectors [F, 72].  GEM's SMPL-22 contract consumes the root
        # plus the first 21 body joints (the first 66 values), matching
        # AISTPlusPlusSmplDataset._load_data.  Do not require an already-trimmed
        # [F, 66] pose here: that rejects every valid official artifact.
        if pose.shape != (num_frames, 72) or translation.shape != (num_frames, 3):
            raise ValueError(
                f"{sequence_id}: invalid AIST++ canonical motion shapes: "
                f"smpl_pose_global={tuple(pose.shape)}, "
                f"smpl_trans_global={tuple(translation.shape)}, "
                f"bbox_frames={num_frames}; expected [F,72], [F,3], F"
            )
        yield {
            "sample_id": str(sequence_id),
            "source_path": annot_path,
            "source_sha256": artifact_sha256,
            "num_frames": num_frames,
            "motion": {
                "global_orient": pose[:, :3],
                "body_pose": pose[:, 3:66],
                "transl": translation,
                "betas": torch.zeros(num_frames, 10),
            },
        }


@torch.no_grad()
def _sole_positions(
    model: torch.nn.Module,
    motion: dict[str, torch.Tensor],
    *,
    device: torch.device,
    chunk_frames: int,
) -> torch.Tensor:
    chunks = []
    indices = torch.as_tensor(SOLE_V437_INDICES, dtype=torch.long, device=device)
    num_frames = int(motion["body_pose"].shape[0])
    for start in range(0, num_frames, chunk_frames):
        end = min(start + chunk_frames, num_frames)
        params = {
            key: value[start:end].to(device=device, dtype=torch.float32).unsqueeze(0)
            for key, value in motion.items()
        }
        vertices, _ = model(**params)
        chunks.append(
            vertices[0].index_select(-2, indices).detach().cpu().float()
        )
    return torch.cat(chunks, dim=0)


def _source_iterator(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if args.kind == "manifest":
        return _manifest_sources(args.root, args.split)
    return _aist_sources(args.root, args.annot_file, args.split_file)


def build(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {args.output}")
    device = torch.device(args.device)
    model = make_smplx("supermotion_v437coco17").to(device).eval()
    records = []
    valid_count = 0
    for source in _source_iterator(args):
        if source["num_frames"] != int(source["motion"]["body_pose"].shape[0]):
            raise ValueError(f"{source['sample_id']}: source num_frames mismatch")
        positions = _sole_positions(
            model,
            source["motion"],
            device=device,
            chunk_frames=args.chunk_frames,
        )
        estimate = estimate_ground_height(positions, fps=30.0)
        record = make_ground_record(
            sample_id=source["sample_id"],
            source_motion_sha256=source["source_sha256"],
            num_frames=source["num_frames"],
            fps=30.0,
            estimate=estimate,
        )
        records.append(record)
        valid_count += int(record["ground_valid"])
        print(
            f"[{len(records):05d}] {source['sample_id']} "
            f"ground_valid={record['ground_valid']} ground_y={record['ground_y']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(args.output)
    print(
        f"Wrote {len(records)} records to {args.output}; "
        f"ground_valid={valid_count}/{len(records)}"
    )


def verify(args: argparse.Namespace) -> None:
    records = load_ground_sidecar(args.output)
    expected = 0
    seen: set[str] = set()
    for source in _source_iterator(args):
        expected += 1
        sample_id = source["sample_id"]
        seen.add(sample_id)
        if sample_id not in records:
            raise ValueError(f"sidecar is missing sample_id={sample_id}")
        record = records[sample_id]
        if record["source_motion_sha256"] != source["source_sha256"]:
            raise ValueError(f"{sample_id}: source motion SHA256 is stale")
        if int(record["num_frames"]) != int(source["num_frames"]):
            raise ValueError(f"{sample_id}: sidecar frame count is stale")
    extra = set(records) - seen
    if extra:
        raise ValueError(f"sidecar contains {len(extra)} unexpected sample IDs")
    print(f"Verified {expected} records and source SHA256 values in {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("manifest", "aist"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="train", help="manifest split")
    parser.add_argument("--annot-file", default="annot_aist_30fps.pt")
    parser.add_argument("--split-file", default="train.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.root / "physics" / "ground_v1.jsonl"
    )
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    verify(arguments) if arguments.verify_only else build(arguments)
