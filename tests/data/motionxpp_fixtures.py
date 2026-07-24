"""Motion-X++ 测试用极小 ZIP/目录构造器。"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np


def smplx322(frames: int, *, offset: float = 0.0, nan: bool = False) -> np.ndarray:
    value = np.zeros((frames, 322), dtype=np.float32)
    value[:, 1] = np.linspace(0.0, 0.2, frames, dtype=np.float32)
    value[:, 309] = np.linspace(0.0, 1.0 + offset, frames, dtype=np.float32)
    value[:, 310] = 1.0
    value[:, 311] = offset
    if nan:
        value[3, 10] = np.nan
    return value


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _npz_bytes(**values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    return buffer.getvalue()


def keypoints(frames: int) -> dict:
    images = [{"id": index, "file_name": f"clip/{index + 1:06d}.png"} for index in range(frames)]
    annotations = []
    for index in range(frames):
        body = [[100.0 + joint, 200.0 + index, float(joint % 2)] for joint in range(17)]
        annotations.append(
            {
                "id": index,
                "image_id": index,
                "file_name": images[index]["file_name"],
                "body_kpts": body,
                "foot_kpts": [[0.0, 0.0, 0.0]] * 6,
                "lefthand_kpts": [[0.0, 0.0, 0.0]] * 21,
                "righthand_kpts": [[0.0, 0.0, 0.0]] * 21,
                "face_kpts": [[0.0, 0.0, 0.0]] * 68,
            }
        )
    return {"images": images, "annotations": annotations}


def make_motionxpp_root(
    root: Path,
    *,
    subset: str = "toy",
    zipped: bool = True,
    records: list[dict] | None = None,
) -> Path:
    """创建与真实长 ZIP 前缀兼容的小数据根。"""
    records = records or [
        {"stem": "walk_clip1", "frames": 40, "caption": "A person walks."},
        {"stem": "turn_clip1", "frames": 60, "caption": "A person turns."},
    ]
    bases = {
        "motion": root / "motion/motion_generation/smplx322",
        "text": root / "text/semantic_label",
        "keypoints": root / "motion/keypoints",
    }
    for base in bases.values():
        base.mkdir(parents=True, exist_ok=True)
    prefix = "long/internal/Motion-X++/v7"
    if zipped:
        archives = {
            name: zipfile.ZipFile(base / f"{subset}.zip", "w") for name, base in bases.items()
        }
        try:
            for record in records:
                stem = record["stem"]
                motion = smplx322(
                    record["frames"],
                    offset=float(record.get("offset", 0.0)),
                    nan=bool(record.get("nan", False)),
                )
                if record.get("dict_format"):
                    payload = _npz_bytes(
                        global_orient=motion[:, :3],
                        body_pose=motion[:, 3:66],
                        transl=motion[:, 309:312],
                        betas=motion[:, 312:322],
                        fps=np.asarray(record.get("fps", 30.0), dtype=np.float32),
                    )
                    motion_suffix = ".npz"
                else:
                    payload = _npy_bytes(motion)
                    motion_suffix = ".npy"
                archives["motion"].writestr(
                    f"{prefix}/motion/motion_generation/smplx322/{subset}/{stem}{motion_suffix}",
                    payload,
                )
                caption = record.get("caption", "A person moves.")
                if isinstance(caption, list):
                    text_payload = "\n".join(caption)
                else:
                    text_payload = str(caption)
                archives["text"].writestr(
                    f"{prefix}/text/semantic_label/{subset}/{stem}.txt",
                    text_payload,
                )
                archives["keypoints"].writestr(
                    f"{prefix}/motion/keypoints/{subset}/{stem}.json",
                    json.dumps(keypoints(record["frames"])),
                )
        finally:
            for archive in archives.values():
                archive.close()
    else:
        for record in records:
            stem = record["stem"]
            motion = smplx322(
                record["frames"],
                offset=float(record.get("offset", 0.0)),
                nan=bool(record.get("nan", False)),
            )
            motion_dir = bases["motion"] / subset
            text_dir = bases["text"] / subset
            kp_dir = bases["keypoints"] / subset
            motion_dir.mkdir(exist_ok=True)
            text_dir.mkdir(exist_ok=True)
            kp_dir.mkdir(exist_ok=True)
            np.save(motion_dir / f"{stem}.npy", motion)
            caption = record.get("caption", "A person moves.")
            text_dir.joinpath(f"{stem}.txt").write_text(
                "\n".join(caption) if isinstance(caption, list) else str(caption),
                encoding="utf-8",
            )
            kp_dir.joinpath(f"{stem}.json").write_text(
                json.dumps(keypoints(record["frames"])), encoding="utf-8"
            )
    return root
