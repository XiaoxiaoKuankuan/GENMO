#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Extract EDGE baseline35 features for one song or aligned AIST++ WAVs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from gem.utils.music_features import align_features_to_length, extract_edge_baseline35


def _write_json(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_tensor(path: Path, tensor: torch.Tensor, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu().float(), path)


def _motion_target_frames(path: Path) -> int:
    """Read an official 60 FPS AIST++ motion and return len(smpl_poses[::2])."""
    with path.open("rb") as file:
        motion = pickle.load(file)
    if not isinstance(motion, dict) or "smpl_poses" not in motion:
        raise ValueError(f"AIST++ motion is missing 'smpl_poses': {path}")
    poses = motion["smpl_poses"]
    if getattr(poses, "ndim", 0) != 2 or poses.shape[-1] < 66:
        raise ValueError(f"Unexpected smpl_poses shape in {path}: {getattr(poses, 'shape', None)}")
    return int(len(poses[::2]))


def extract_single(audio: Path, output: Path, overwrite: bool) -> int:
    """Extract and save a single arbitrary audio file."""
    features, metadata = extract_edge_baseline35(audio)
    _save_tensor(output, features, overwrite)
    _write_json(output.with_suffix(output.suffix + ".json"), metadata, overwrite)
    print(f"Saved {tuple(features.shape)} float32 features to {output}")
    return 0


def extract_batch(
    annotations_root: Path,
    aligned_wav_dir: Path,
    output_dir: Path,
    allow_missing: bool,
    overwrite: bool,
) -> int:
    """Extract same-stem, per-sequence WAV features aligned to official motions."""
    motions_dir = annotations_root / "motions"
    if not motions_dir.is_dir():
        raise FileNotFoundError(f"AIST++ motions directory does not exist: {motions_dir}")
    if not aligned_wav_dir.is_dir():
        raise FileNotFoundError(
            f"Aligned per-sequence WAV directory does not exist: {aligned_wav_dir}"
        )
    motion_paths = sorted(motions_dir.glob("*.pkl"))
    if not motion_paths:
        raise FileNotFoundError(f"No AIST++ motion .pkl files found in: {motions_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    missing_path = output_dir / "missing_wavs.json"
    if not overwrite:
        existing = [path for path in (manifest_path, missing_path) if path.exists()]
        existing.extend(
            output_dir / f"{motion_path.stem}_musicfeat_fps30.pt"
            for motion_path in motion_paths
            if (output_dir / f"{motion_path.stem}_musicfeat_fps30.pt").exists()
        )
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

    missing: list[dict[str, str]] = []
    manifest: list[dict[str, Any]] = []
    for index, motion_path in enumerate(motion_paths, start=1):
        sequence = motion_path.stem
        wav_path = aligned_wav_dir / f"{sequence}.wav"
        if not wav_path.is_file():
            missing.append(
                {
                    "sequence": sequence,
                    "expected_audio": str(wav_path.resolve()),
                    "source_motion": str(motion_path.resolve()),
                }
            )
            continue

        target_frames = _motion_target_frames(motion_path)
        raw_features, metadata = extract_edge_baseline35(
            wav_path,
            aist_sequence_name=sequence,
        )
        original_frames = int(raw_features.shape[0])
        features = align_features_to_length(raw_features, target_frames, "trim_or_pad_last")
        output_path = output_dir / f"{sequence}_musicfeat_fps30.pt"
        _save_tensor(output_path, features, overwrite)
        manifest.append(
            {
                "sequence": sequence,
                "source_audio": str(wav_path.resolve()),
                "source_motion": str(motion_path.resolve()),
                "original_audio_frames": original_frames,
                "target_motion_frames": target_frames,
                "final_feature_frames": int(features.shape[0]),
                "trim_or_pad_count": target_frames - original_frames,
                "source_audio_size_bytes": wav_path.stat().st_size,
                "estimated_or_prior_bpm": metadata["estimated_or_prior_bpm"],
                "output": str(output_path.resolve()),
            }
        )
        print(f"[{index}/{len(motion_paths)}] {sequence}: {original_frames} -> {target_frames}")

    _write_json(
        manifest_path,
        {
            "annotations_root": str(annotations_root.resolve()),
            "aligned_wav_dir": str(aligned_wav_dir.resolve()),
            "feature_type": "edge_baseline35",
            "feature_dim": 35,
            "target_fps": 30,
            "sequences": manifest,
        },
        overwrite,
    )
    _write_json(missing_path, missing, overwrite)
    print(f"Extracted {len(manifest)} sequences; missing aligned WAVs: {len(missing)}")
    return 0 if allow_missing or not missing else 2


def build_parser() -> argparse.ArgumentParser:
    """Build the feature extraction CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audio", type=Path, help="Single arbitrary audio file")
    mode.add_argument(
        "--annotations-root",
        type=Path,
        help="AIST++ official annotations root for batch mode",
    )
    parser.add_argument("--output", type=Path, help="Single-file output .pt path")
    parser.add_argument("--aligned-wav-dir", type=Path, help="Same-stem per-sequence WAVs")
    parser.add_argument("--output-dir", type=Path, help="Batch output musicfeat_v2 directory")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing files in the selected output location",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run single-file or AIST++ batch feature extraction."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.audio is not None:
        if args.output is None:
            parser.error("single-file mode requires --output")
        if args.aligned_wav_dir is not None or args.output_dir is not None:
            parser.error("--aligned-wav-dir/--output-dir are batch-mode arguments")
        return extract_single(args.audio, args.output, args.overwrite)
    if args.aligned_wav_dir is None or args.output_dir is None:
        parser.error("batch mode requires --aligned-wav-dir and --output-dir")
    if args.output is not None:
        parser.error("--output is only valid in single-file mode")
    return extract_batch(
        args.annotations_root,
        args.aligned_wav_dir,
        args.output_dir,
        args.allow_missing,
        args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
