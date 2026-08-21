#!/usr/bin/env python3
"""Offline MuJoCo rendering of a BUMI qpos28 artifact (kinematics only)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np
import torch


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_qpos(path: Path, key: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"BUMI motion artifact must be a dictionary: {path}")
    expected = {
        "robot_name": "bumi",
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{path}: {field} must be {value!r}, got {payload.get(field)!r}")
    qpos = torch.as_tensor(payload.get(key)).detach().cpu().double()
    if qpos.ndim != 2 or qpos.shape[1] != 28 or qpos.shape[0] <= 0:
        raise ValueError(f"{path}: {key} must have shape [T,28], got {qpos.shape}")
    if not bool(torch.isfinite(qpos).all()):
        raise ValueError(f"{path}: {key} contains NaN or Inf")
    norm = torch.linalg.vector_norm(qpos[:, 3:7], dim=-1)
    if float((norm - 1.0).abs().max()) > 1.0e-3:
        raise ValueError(f"{path}: root quaternion is not normalized wxyz")
    return qpos.numpy(), int(payload.get("fps", 30)), payload


def mujoco_joint_order(model: mujoco.MjModel) -> list[str]:
    ids = []
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        if address >= 7:
            ids.append(joint_id)
    ids.sort(key=lambda value: int(model.jnt_qposadr[value]))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, value) for value in ids]
    if any(name is None for name in names):
        raise ValueError("Every BUMI actuated MJCF joint must have a name")
    return [str(name) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qpos-key", default="qpos")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera")
    args = parser.parse_args()
    qpos, fps, payload = load_qpos(args.motion.expanduser().resolve(), args.qpos_key)
    if fps != 30:
        raise ValueError(f"BUMI render input must be 30 FPS, got {fps}")
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.expanduser().resolve()))
    if model.nq != 28:
        raise ValueError(f"BUMI MJCF nq must be 28, got {model.nq}")
    artifact_joint_names = tuple(map(str, payload.get("joint_names", ())))
    mjcf_joint_names = tuple(mujoco_joint_order(model))
    if artifact_joint_names != mjcf_joint_names:
        raise ValueError(
            "Motion joint_names do not exactly match MuJoCo-native MJCF qpos order; "
            "rendering refuses to guess a reorder"
        )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames = []
    camera: str | int = -1 if args.camera is None else args.camera
    try:
        for frame_qpos in qpos:
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.stack(frames), fps=fps, codec="libx264")
    print(f"Rendered {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
