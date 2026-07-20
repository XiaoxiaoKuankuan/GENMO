# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Extract one verified SMPL-X frame as a neutral-shape idle reference."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.runtime.motion_streamer import load_smpl_motion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a validated one-frame SMPL-X idle pose from a motion."
    )
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape_mode", choices=["zero"], default="zero")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def extract_idle_pose(
    motion_path: Path,
    frame_index: int,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate a source motion and atomically save one all-zero-beta frame."""
    motion = load_smpl_motion(motion_path, shape_mode="zero", min_frames=1)
    if not 0 <= frame_index < motion.num_frames:
        raise IndexError(
            f"--frame {frame_index} is outside [0, {motion.num_frames}) for {motion.source_path}"
        )
    output = output_path.expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing idle motion: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    slc = slice(frame_index, frame_index + 1)
    global_params = {
        "body_pose": motion.body_pose[slc].clone(),
        "global_orient": motion.global_orient[slc].clone(),
        "transl": motion.transl[slc].clone(),
        "betas": torch.zeros(1, 10, dtype=torch.float32),
    }
    payload = {
        "body_params_global": global_params,
        "body_params_incam": {key: value.clone() for key, value in global_params.items()},
        "fps": motion.fps,
        "num_frames": 1,
        "source": "verified_idle_extract",
        "shape_mode": "zero",
        "source_motion": str(motion.source_path),
        "source_frame": frame_index,
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = extract_idle_pose(
        args.motion,
        args.frame,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"[Idle] Saved verified zero-shape frame to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
