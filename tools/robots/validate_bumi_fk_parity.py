#!/usr/bin/env python3
"""Offline numerical parity check between GENMO Torch FK and MuJoCo FK."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    mjcf_path: Path, kinematics_path: Path, samples: int, seed: int
) -> dict[str, float | int | bool]:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path.expanduser().resolve()))
    spec = json.loads(kinematics_path.expanduser().resolve().read_text(encoding="utf-8"))
    expected_mjcf_sha = spec.get("source_mjcf_sha256")
    actual_mjcf_sha = sha256_file(mjcf_path.expanduser().resolve())
    if expected_mjcf_sha != actual_mjcf_sha:
        raise ValueError(
            f"Kinematics source_mjcf_sha256={expected_mjcf_sha!r} does not match "
            f"validation MJCF SHA-256={actual_mjcf_sha!r}"
        )
    kinematics = BumiKinematics(kinematics_path).double()
    if model.nq != 28:
        raise ValueError(f"MuJoCo BUMI nq must be 28, got {model.nq}")
    rng = np.random.default_rng(seed)
    qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float64)[None], samples, axis=0)
    qpos[:, :3] += rng.uniform(-1.0, 1.0, size=(samples, 3))
    quaternion = rng.normal(size=(samples, 4))
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    qpos[:, 3:7] = quaternion
    lower = kinematics.joint_lower_limits.cpu().numpy()
    upper = kinematics.joint_upper_limits.cpu().numpy()
    qpos[:, 7:] = rng.uniform(lower, upper, size=(samples, 21))
    with torch.no_grad():
        torch_fk = kinematics.forward_kinematics_full(torch.from_numpy(qpos))
    body_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in kinematics.body_order
    ]
    if any(value < 0 for value in body_ids):
        raise ValueError("At least one exported feature body is absent from the validation MJCF")
    mujoco_position = np.zeros_like(torch_fk["body_pos_w"].cpu().numpy())
    mujoco_rotation = np.zeros_like(torch_fk["body_rot_w"].cpu().numpy())
    data = mujoco.MjData(model)
    for index in range(samples):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        mujoco_position[index] = data.xpos[body_ids]
        mujoco_rotation[index] = data.xmat[body_ids].reshape(-1, 3, 3)
    position_error = np.abs(torch_fk["body_pos_w"].cpu().numpy() - mujoco_position)
    rotation_error = np.abs(torch_fk["body_rot_w"].cpu().numpy() - mujoco_rotation)
    report: dict[str, float | int | bool] = {
        "samples": int(samples),
        "seed": int(seed),
        "max_position_error_m": float(position_error.max(initial=0.0)),
        "mean_position_error_m": float(position_error.mean()),
        "max_rotation_matrix_error": float(rotation_error.max(initial=0.0)),
        "mean_rotation_matrix_error": float(rotation_error.mean()),
        "position_threshold_m": 1.0e-5,
        "rotation_matrix_threshold": 1.0e-4,
    }
    report["valid"] = bool(
        report["max_position_error_m"] < report["position_threshold_m"]
        and report["max_rotation_matrix_error"] < report["rotation_matrix_threshold"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    report = validate(args.mjcf, args.kinematics, args.samples, args.seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
