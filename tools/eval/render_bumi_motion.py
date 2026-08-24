#!/usr/bin/env python3
"""离线渲染 BUMI qpos28 轨迹（仅运动学，不执行动力学仿真）。

工具严格校验轨迹的机器人标识、四元数约定、MuJoCo 原生关节顺序和 30 Hz 帧率，
再逐帧执行 ``mj_forward`` 并写入 H.264 视频。视频帧采用流式编码，避免完整音乐对应的
数千帧 RGB 图像同时驻留内存；因此既适合短片验证，也适合数分钟完整音频的批量评测。
``--max-frames`` 可在加载并完成轨迹契约校验后只渲染前缀帧，供固定时长网页验收避免先渲染
整段长动作再由 ffmpeg 截断；该选项不循环、不补帧，也不会改变源动作产物。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    qpos, fps, payload = load_qpos(args.motion.expanduser().resolve(), args.qpos_key)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames 必须为正整数")
        qpos = qpos[: args.max_frames]
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
    camera: str | int = -1 if args.camera is None else args.camera
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("BUMI 视频渲染需要 ffmpeg")
    encoder = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{args.width}x{args.height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    rendered_frames = 0
    try:
        assert encoder.stdin is not None
        for frame_qpos in qpos:
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
            if frame.shape != (args.height, args.width, 3):
                raise RuntimeError(f"MuJoCo 返回了异常画面形状：{frame.shape}")
            encoder.stdin.write(frame.tobytes())
            rendered_frames += 1
        encoder.stdin.close()
        assert encoder.stderr is not None
        error_text = encoder.stderr.read().decode("utf-8", errors="replace").strip()
        returncode = encoder.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败（{returncode}）：{error_text}")
    except BaseException:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()
    print(f"Rendered {rendered_frames} frames to {output}")


if __name__ == "__main__":
    main()
