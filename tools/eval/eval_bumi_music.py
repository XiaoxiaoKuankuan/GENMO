#!/usr/bin/env python3
"""Evaluate BUMI motion kinematics without making dynamics/controller claims."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402
from gem.robots.bumi.metrics import (  # noqa: E402
    compute_bumi_kinematic_metrics,
    metrics_to_json,
)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_artifact(payload: Any, path: Path, qpos_key: str) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise ValueError(f"BUMI artifact must be a dictionary: {path}")
    expected = {
        "robot_name": "bumi",
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"{path}: {key} must be {expected_value!r}")
    qpos = torch.as_tensor(payload.get(qpos_key)).float()
    if qpos.ndim != 2 or qpos.shape[1] != 28 or not bool(torch.isfinite(qpos).all()):
        raise ValueError(f"{path}: {qpos_key} must be finite [T,28], got {qpos.shape}")
    return qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--qpos-key", default="qpos")
    parser.add_argument("--target-qpos-key", default="qpos")
    parser.add_argument("--music-features", type=Path)
    parser.add_argument("--ground-height", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prediction_path = args.prediction.expanduser().resolve()
    prediction = torch_load(prediction_path)
    pred_qpos = validate_artifact(prediction, prediction_path, args.qpos_key)
    kinematics = BumiKinematics(args.kinematics)
    if tuple(map(str, prediction.get("joint_names", ()))) != kinematics.joint_order:
        raise ValueError("Prediction joint_names do not match kinematics joint_order")
    target_qpos = None
    target_contact = None
    if args.target is not None:
        target_path = args.target.expanduser().resolve()
        target = torch_load(target_path)
        target_qpos = validate_artifact(target, target_path, args.target_qpos_key)
        if target_qpos.shape != pred_qpos.shape:
            raise ValueError(f"Prediction/target qpos shapes differ: {pred_qpos.shape}/{target_qpos.shape}")
        if isinstance(target, dict) and "foot_contact" in target:
            target_contact = torch.as_tensor(target["foot_contact"]).float()
    music_beats = None
    if args.music_features is not None:
        music = torch_load(args.music_features.expanduser().resolve())
        if not isinstance(music, torch.Tensor) or music.ndim != 2 or music.shape[1] != 35:
            raise ValueError("--music-features must contain raw EDGE35 Tensor[T,35]")
        if music.shape[0] != pred_qpos.shape[0]:
            raise ValueError("music and prediction frame counts must match")
        music_beats = music[:, 34]
    pred_contact_logits = prediction.get("pred_foot_contact_logits")
    if pred_contact_logits is not None:
        pred_contact_logits = torch.as_tensor(pred_contact_logits).float()
    if args.ground_height is not None:
        ground_height = float(args.ground_height)
    elif args.qpos_key == "qpos_canonical" or not bool(
        prediction.get("world_anchor_applied", False)
    ):
        ground_height = -float(kinematics.default_qpos[2])
    else:
        ground_height = 0.0
    metrics = compute_bumi_kinematic_metrics(
        pred_qpos,
        kinematics,
        target_qpos=target_qpos,
        target_contact=target_contact,
        pred_contact_logits=pred_contact_logits,
        music_beats=music_beats,
        ground_height=ground_height,
    )
    report = {
        "contract_version": "genmo.bumi_kinematic_eval.v1",
        "prediction": str(prediction_path),
        "target": None if args.target is None else str(args.target.expanduser().resolve()),
        "qpos_key": args.qpos_key,
        "ground_height": ground_height,
        "metrics": metrics_to_json(metrics),
        "scope": "kinematic motion quality only",
        "dynamics_tracking_validated": False,
        "gmt_tracking_validated": False,
        "torque_feasibility_validated": False,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
