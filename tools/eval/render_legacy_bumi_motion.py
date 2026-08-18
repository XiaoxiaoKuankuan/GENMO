#!/usr/bin/env python3
"""用生成数据时的生产 source MJCF 渲染一条 GMR BUMI3 legacy pickle。

该工具专用于质量筛选人工复核：它先校验 source MJCF SHA，使用严格 legacy reader
把根四元数从 xyzw 转成 MuJoCo wxyz，核对真实 MJCF qpos 关节顺序，再逐帧执行
``mj_forward``。视频默认覆盖完整动作长度，并叠加质量状态、reason code、帧号和
当前是否命中 FLOOR；相机跟随 Root XY，避免长距离舞蹈走出画面。视频旁同时写出
同名 ``.quality.json``，保证视觉判断可追溯到配置与数值报告。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import av
import cv2
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.legacy_motion import (  # noqa: E402
    LEGACY_BUMI_JOINT_ORDER,
    load_legacy_bumi_motion,
)
from gem.robots.bumi.quality_filter import (  # noqa: E402
    evaluate_legacy_bumi_motion,
    load_bumi_quality_config,
)
from tools.data.bumi.filter_legacy_bumi_motions import (  # noqa: E402
    DEFAULT_CONFIG,
    verify_source_mjcf,
)


def mujoco_joint_order(model: mujoco.MjModel) -> tuple[str, ...]:
    """读取 free root 之后按 qpos address 排序的 21 个关节名。"""

    ids = [joint_id for joint_id in range(model.njnt) if int(model.jnt_qposadr[joint_id]) >= 7]
    ids.sort(key=lambda joint_id: int(model.jnt_qposadr[joint_id]))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in ids]
    if any(name is None for name in names):
        raise ValueError("every source MJCF actuated joint must have a name")
    return tuple(map(str, names))


def _overlay(
    frame: np.ndarray,
    *,
    sample_id: str,
    status: str,
    reasons: tuple[str, ...],
    frame_index: int,
    total_frames: int,
    floor: bool,
) -> np.ndarray:
    output = np.ascontiguousarray(frame)
    lines = [
        f"{sample_id}  {status}",
        f"frame {frame_index + 1}/{total_frames}" + ("  FLOOR" if floor else ""),
        " | ".join(reasons) if reasons else "PASS",
    ]
    # renderer 返回 RGB；OpenCV 在这里仅写像素，不执行 BGR/RGB 转换。
    color = (230, 30, 30) if floor else (30, 220, 30)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (20, 35 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--source-mjcf", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("render width and height must be positive even integers")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    config = load_bumi_quality_config(args.config)
    mjcf_path, source_sha = verify_source_mjcf(args.source_mjcf, config.source_mjcf_sha256)
    motion = load_legacy_bumi_motion(args.motion, expected_fps=config.fps)
    decision = evaluate_legacy_bumi_motion(motion, config)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if model.nq != 28:
        raise ValueError(f"source BUMI MJCF nq must be 28, got {model.nq}")
    joint_order = mujoco_joint_order(model)
    if joint_order != LEGACY_BUMI_JOINT_ORDER:
        raise ValueError(
            "source MJCF joint order differs from legacy pickle contract: "
            f"expected={LEGACY_BUMI_JOINT_ORDER}, got={joint_order}"
        )
    qpos = motion.qpos_wxyz(joint_order)
    frame_count = min(len(qpos), args.max_frames or len(qpos))
    floor_mask = np.zeros(len(qpos), dtype=np.bool_)
    for start, end in decision.floor_intervals:
        floor_mask[start:end] = True

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    camera = mujoco.MjvCamera()
    camera.distance = float(args.camera_distance)
    camera.azimuth = float(args.camera_azimuth)
    camera.elevation = float(args.camera_elevation)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.stem}.",
        suffix=output.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    container: av.container.OutputContainer | None = None
    try:
        container = av.open(str(temporary), mode="w")
        stream = container.add_stream(args.codec, rate=config.fps)
        stream.width = args.width
        stream.height = args.height
        stream.pix_fmt = "yuv420p"
        for frame_index in range(frame_count):
            data.qpos[:] = qpos[frame_index]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = (qpos[frame_index, 0], qpos[frame_index, 1], 0.45)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            annotated = _overlay(
                frame,
                sample_id=motion.path.stem,
                status=decision.status.value,
                reasons=decision.reason_codes,
                frame_index=frame_index,
                total_frames=frame_count,
                floor=bool(floor_mask[frame_index]),
            )
            video_frame = av.VideoFrame.from_ndarray(annotated, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        os.replace(temporary, output)
    finally:
        if container is not None:
            container.close()
        renderer.close()
        temporary.unlink(missing_ok=True)
    sidecar = {
        "motion": str(motion.path),
        "source_mjcf": str(mjcf_path),
        "source_mjcf_sha256": source_sha,
        "rendered_frames": frame_count,
        "fps": config.fps,
        **decision.to_dict(),
    }
    sidecar_path = output.with_suffix(".quality.json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {frame_count} frames to {output}")
    print(f"Quality sidecar: {sidecar_path}")


if __name__ == "__main__":
    main()
