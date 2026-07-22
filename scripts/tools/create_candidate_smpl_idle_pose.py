# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Create a manual SMPL-X standing-idle candidate for simulation validation."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch


# SMPL-X body_pose excludes the pelvis. The shoulder entries are global joints
# 16/17 minus one. In the zero pose both arms point horizontally; opposite
# rotations about local Z lower them. Leaving a small angle from the torso gives
# an arms-slightly-open stance without inventing robot joint angles.
LEFT_SHOULDER_BODY_INDEX = 15
RIGHT_SHOULDER_BODY_INDEX = 16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inputs/motions/smplx_idle_stand_slight_open_candidate.pt"),
    )
    parser.add_argument(
        "--arm_open_deg",
        type=float,
        default=15.0,
        help="Angle of each straight arm away from the side of the torso.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def create_candidate(
    output: Path,
    *,
    arm_open_deg: float,
    fps: float,
    overwrite: bool,
) -> Path:
    if not math.isfinite(arm_open_deg) or not 0.0 <= arm_open_deg <= 45.0:
        raise ValueError("--arm_open_deg must be finite and in [0, 45]")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("--fps must be finite and > 0")

    output = output.expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing idle candidate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    body_pose = torch.zeros(1, 63, dtype=torch.float32)
    body_pose_aa = body_pose.reshape(1, 21, 3)
    lower_from_horizontal = math.radians(90.0 - arm_open_deg)
    body_pose_aa[0, LEFT_SHOULDER_BODY_INDEX, 2] = -lower_from_horizontal
    body_pose_aa[0, RIGHT_SHOULDER_BODY_INDEX, 2] = lower_from_horizontal

    global_params = {
        "body_pose": body_pose,
        "global_orient": torch.zeros(1, 3, dtype=torch.float32),
        "transl": torch.zeros(1, 3, dtype=torch.float32),
        "betas": torch.zeros(1, 10, dtype=torch.float32),
    }
    payload = {
        "body_params_global": global_params,
        "body_params_incam": {key: value.clone() for key, value in global_params.items()},
        "fps": float(fps),
        "num_frames": 1,
        "source": "manual_idle_candidate",
        "shape_mode": "zero",
        "arm_open_deg": float(arm_open_deg),
        "verified": False,
        "notes": "Experience-based human SMPL-X idle; validate in MuJoCo/Gazebo before robot use.",
    }

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output.resolve()


def main() -> int:
    args = build_parser().parse_args()
    output = create_candidate(
        args.output,
        arm_open_deg=args.arm_open_deg,
        fps=args.fps,
        overwrite=args.overwrite,
    )
    print(f"[Idle candidate] {output}")
    print(f"[Idle candidate] arms={args.arm_open_deg:g} deg from torso, verified=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
