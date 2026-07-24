#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Motion-X++ ZIP/目录读取、解析和原子写入的共享实现。"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gem.utils.rotation_conversions import (
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)

MOTION_RELATIVE = Path("motion/motion_generation/smplx322")
TEXT_RELATIVE = Path("text/semantic_label")
KEYPOINT_RELATIVE = Path("motion/keypoints")
MODALITY_RELATIVES = {
    "motion": MOTION_RELATIVE,
    "text": TEXT_RELATIVE,
    "keypoints": KEYPOINT_RELATIVE,
}
MOTION_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".json"}
TEXT_SUFFIXES = {".txt", ".json", ".npy", ".npz"}
KEYPOINT_SUFFIXES = {".json", ".npy", ".npz"}
SUFFIXES_BY_MODALITY = {
    "motion": MOTION_SUFFIXES,
    "text": TEXT_SUFFIXES,
    "keypoints": KEYPOINT_SUFFIXES,
}
OFFICIAL_FPS = 30.0
MIN_FRAMES = 25

# Motion-X++/Motion-X 官方格式。
ROOT_ORIENT_SLICE = slice(0, 3)
BODY_POSE_SLICE = slice(3, 66)
TRANSL_SLICE = slice(309, 312)
BETAS_SLICE = slice(312, 322)
IGNORED_RANGES = {
    "hand_pose": [66, 156],
    "jaw_pose": [156, 159],
    "face_expression": [159, 209],
    "face_shape": [209, 309],
}

# 这些是明确的数据来源，而非对文件名做模糊推断。
KNOWN_PROVENANCE: dict[str, dict[str, Any]] = {
    "amass": {
        "source": "AMASS",
        "overlaps": ["AMASS", "HumanML3D"],
        "reason": "Motion-X 官方将 AMASS 列为独立来源；当前 GENMO 已训练 AMASS/HumanML3D。",
    },
    "humanml": {
        "source": "HumanML3D",
        "overlaps": ["HumanML3D", "AMASS"],
        "reason": "Motion-X 官方 mocap 子集中的 humanml 来自 HumanML3D/AMASS。",
    },
    "humanml3d": {
        "source": "HumanML3D",
        "overlaps": ["HumanML3D", "AMASS"],
        "reason": "明确的 HumanML3D 来源。",
    },
    "aist": {
        "source": "AIST++",
        "overlaps": ["AIST++"],
        "reason": "Motion-X 官方将 AIST++ 列为独立来源；当前 GENMO 已训练 AIST++。",
    },
    "aistpp": {
        "source": "AIST++",
        "overlaps": ["AIST++"],
        "reason": "明确的 AIST++ 来源。",
    },
    "idea400": {
        "source": "IDEA400",
        "overlaps": [],
        "reason": "Motion-X 自采 IDEA400；不属于当前 AMASS/HumanML3D/AIST++。",
    },
    "haa500": {
        "source": "HAA500",
        "overlaps": [],
        "reason": "HAA500 网络视频来源；不属于当前 AMASS/HumanML3D/AIST++。",
    },
    "humman": {
        "source": "HuMMan",
        "overlaps": [],
        "reason": "HuMMan 是独立数据源；当前目标训练集未包含 HuMMan。",
    },
    "animation": {
        "source": "online animation videos",
        "overlaps": [],
        "reason": "Motion-X++ 在线动画视频来源，无明确 AMASS/HumanML3D/AIST++ 血缘。",
    },
    "kungfu": {
        "source": "online kung-fu videos",
        "overlaps": [],
        "reason": "Motion-X++ 功夫视频来源，无明确 AMASS/HumanML3D/AIST++ 血缘。",
    },
    "music": {
        "source": "online instrument-performance videos",
        "overlaps": [],
        "reason": (
            "归档内容是乐器演奏视频（如 Play_Flute/Play_Guitar），不是 AIST++ 的 "
            "gXX_sXX_cXX_dXX_mXX_chXX 舞蹈序列。"
        ),
    },
    "perform": {
        "source": "online performance videos",
        "overlaps": [],
        "reason": "Motion-X++ 表演/生活视频来源，无明确 AMASS/HumanML3D/AIST++ 血缘。",
    },
}


class MotionXppError(RuntimeError):
    """Motion-X++ 数据契约无法满足时抛出。"""


class FilteredMotionError(MotionXppError):
    """样本命中预定义过滤规则（目前为长度不足）时抛出。"""


@dataclass(frozen=True)
class AssetRef:
    """ZIP member 或已解压普通文件的可序列化引用。"""

    modality: str
    subset: str
    container: Path
    member: str
    suffix: str
    is_zip: bool

    @property
    def stem(self) -> str:
        """返回不依赖 ZIP 内部长前缀的相对 basename stem。"""
        return Path(self.member).stem

    @property
    def source_path(self) -> str:
        """返回可审计的容器加 member 路径。"""
        if self.is_zip:
            return f"{self.container}!/{self.member}"
        return str(self.container)

    def read_bytes(self) -> bytes:
        """按需读取内容，不在对象中持有 ZipFile 句柄。"""
        if self.is_zip:
            with zipfile.ZipFile(self.container) as archive:
                return archive.read(self.member)
        return self.container.read_bytes()


@dataclass
class AssetIndex:
    """一个 subset/modality 的 stem 索引及冲突信息。"""

    subset: str
    modality: str
    assets: dict[str, AssetRef]
    collisions: dict[str, list[str]]
    source_containers: list[str]


def safe_torch_load(path_or_file: Any) -> Any:
    """兼容不同 PyTorch 版本地加载可信制品。"""
    try:
        return torch.load(path_or_file, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path_or_file, map_location="cpu")


def fsync_file(path: Path) -> None:
    """将一个已经关闭的普通文件同步到磁盘。"""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """同步目录项；不支持目录 fsync 的文件系统上仍保留文件原子性。"""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 JSON 原子替换目标文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    """原子写入文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_torch_save(payload: Any, path: Path, *, validate: bool = True) -> None:
    """写临时 PTH、可选重载验证，然后原子发布。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        fsync_file(temporary)
        if validate:
            safe_torch_load(temporary)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _iter_extracted_files(base: Path, suffixes: set[str]) -> Iterator[tuple[str, Path]]:
    """枚举 base/subset 下普通文件。"""
    if not base.is_dir():
        return
    for subset_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        for path in sorted(subset_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                yield subset_dir.name, path


def discover_subsets(root: str | Path) -> dict[str, list[str]]:
    """发现三种模态各自可用的 ZIP 或已解压 subset。"""
    root = Path(root)
    result: dict[str, list[str]] = {}
    for modality, relative in MODALITY_RELATIVES.items():
        base = root / relative
        names: set[str] = set()
        if base.is_dir():
            names.update(path.stem for path in base.glob("*.zip") if path.is_file())
            names.update(path.name for path in base.iterdir() if path.is_dir())
        result[modality] = sorted(names)
    return result


def build_asset_index(root: str | Path, modality: str, subset: str) -> AssetIndex:
    """建立单个 subset 的 stem 索引，同时支持 ZIP 和解压目录。

    如果同名解压目录存在，则优先使用目录，不把 ZIP 和目录重复合并。
    """
    if modality not in MODALITY_RELATIVES:
        raise ValueError(f"Unknown Motion-X++ modality: {modality}")
    root = Path(root)
    base = root / MODALITY_RELATIVES[modality]
    suffixes = SUFFIXES_BY_MODALITY[modality]
    directory = base / subset
    archive_path = base / f"{subset}.zip"
    refs: list[AssetRef] = []
    containers: list[str] = []
    if directory.is_dir():
        containers.append(str(directory))
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                refs.append(
                    AssetRef(
                        modality,
                        subset,
                        path,
                        str(path.relative_to(directory)),
                        path.suffix.lower(),
                        False,
                    )
                )
    elif archive_path.is_file():
        containers.append(str(archive_path))
        with zipfile.ZipFile(archive_path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                suffix = Path(name).suffix.lower()
                if suffix in suffixes:
                    refs.append(AssetRef(modality, subset, archive_path, name, suffix, True))
    else:
        raise FileNotFoundError(
            f"Motion-X++ {modality} subset {subset!r} was not found under {base}"
        )

    grouped: dict[str, list[AssetRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.stem, []).append(ref)
    collisions = {
        stem: [ref.source_path for ref in values]
        for stem, values in grouped.items()
        if len(values) > 1
    }
    assets = {stem: values[0] for stem, values in grouped.items() if len(values) == 1}
    return AssetIndex(subset, modality, assets, collisions, containers)


def paired_asset_indices(
    root: str | Path, subset: str
) -> tuple[AssetIndex, AssetIndex, AssetIndex | None]:
    """返回 motion/text 以及可选 keypoints 索引。"""
    motion = build_asset_index(root, "motion", subset)
    text = build_asset_index(root, "text", subset)
    try:
        keypoints = build_asset_index(root, "keypoints", subset)
    except FileNotFoundError:
        keypoints = None
    return motion, text, keypoints


def _numpy_from_bytes(ref: AssetRef) -> Any:
    buffer = io.BytesIO(ref.read_bytes())
    if ref.suffix == ".npy":
        try:
            return np.load(buffer, allow_pickle=False)
        except ValueError as exc:
            # 仅兼容用户已授权的本地 Motion-X++ object/dict NPY。
            if "Object arrays cannot be loaded" not in str(exc):
                raise
            buffer.seek(0)
            value = np.load(buffer, allow_pickle=True)
            return value.item() if value.shape == () else value
    if ref.suffix == ".npz":
        with np.load(buffer, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if ref.suffix in {".pt", ".pth"}:
        return safe_torch_load(buffer)
    if ref.suffix == ".json":
        return json.loads(ref.read_bytes().decode("utf-8-sig"))
    raise MotionXppError(f"Unsupported NumPy/Torch asset: {ref.source_path}")


def _as_float_tensor(value: Any, field: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    except Exception as exc:
        raise MotionXppError(f"{field} cannot be converted to float32: {exc}") from exc
    if not torch.isfinite(tensor).all():
        raise MotionXppError(f"{field} contains NaN or Inf")
    return tensor.contiguous()


def _find_first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    lowered = {str(key).lower(): key for key in mapping}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return mapping[key]
    return None


def parse_motion_asset(ref: AssetRef) -> dict[str, Any]:
    """解析官方 [F,322]、NPZ 或分字段 dict 为统一 SMPL-X body 参数。"""
    value = _numpy_from_bytes(ref)
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    if isinstance(value, torch.Tensor | np.ndarray):
        array = _as_float_tensor(value, "motion")
        if array.ndim != 2 or array.shape[1] != 322:
            raise MotionXppError(
                f"{ref.source_path}: direct motion must be [F,322], got {tuple(array.shape)}"
            )
        result = {
            "global_orient": array[:, ROOT_ORIENT_SLICE],
            "body_pose": array[:, BODY_POSE_SLICE],
            "transl": array[:, TRANSL_SLICE],
            "betas": array[:, BETAS_SLICE],
            "fps": None,
            "gender": "neutral",
            "raw_shape": list(array.shape),
            "source_format": "smplx322",
        }
    elif isinstance(value, Mapping):
        nested = _find_first(value, ("motion", "poses", "smplx", "smplx322"))
        if nested is not None and not isinstance(nested, Mapping):
            array = _as_float_tensor(nested, "motion")
            if array.ndim != 2 or array.shape[1] != 322:
                raise MotionXppError(
                    f"{ref.source_path}: embedded motion must be [F,322], got {tuple(array.shape)}"
                )
            result = {
                "global_orient": array[:, ROOT_ORIENT_SLICE],
                "body_pose": array[:, BODY_POSE_SLICE],
                "transl": array[:, TRANSL_SLICE],
                "betas": array[:, BETAS_SLICE],
                "raw_shape": list(array.shape),
                "source_format": "dict_smplx322",
            }
        else:
            global_orient = _find_first(value, ("global_orient", "root_orient", "root_pose"))
            body_pose = _find_first(value, ("body_pose", "pose_body"))
            transl = _find_first(value, ("transl", "trans", "translation"))
            betas = _find_first(value, ("betas", "beta", "shape"))
            missing = [
                name
                for name, item in (
                    ("global_orient", global_orient),
                    ("body_pose", body_pose),
                    ("transl", transl),
                    ("betas", betas),
                )
                if item is None
            ]
            if missing:
                raise MotionXppError(f"{ref.source_path}: motion dict missing {', '.join(missing)}")
            result = {
                "global_orient": _as_float_tensor(global_orient, "global_orient"),
                "body_pose": _as_float_tensor(body_pose, "body_pose"),
                "transl": _as_float_tensor(transl, "transl"),
                "betas": _as_float_tensor(betas, "betas"),
                "raw_shape": None,
                "source_format": "parameter_dict",
            }
        fps = _find_first(value, ("fps", "frame_rate", "framerate", "mocap_frame_rate"))
        result["fps"] = float(np.asarray(fps).reshape(-1)[0]) if fps is not None else None
        gender = _find_first(value, ("gender", "sex"))
        result["gender"] = str(gender).lower() if gender is not None else "neutral"
    else:
        raise MotionXppError(f"{ref.source_path}: unsupported motion object {type(value).__name__}")

    global_orient = _as_float_tensor(result["global_orient"], "global_orient")
    body_pose = _as_float_tensor(result["body_pose"], "body_pose")
    transl = _as_float_tensor(result["transl"], "transl")
    betas = _as_float_tensor(result["betas"], "betas")
    if global_orient.ndim != 2 or global_orient.shape[1] != 3:
        raise MotionXppError(f"global_orient must be [F,3], got {tuple(global_orient.shape)}")
    frame_count = global_orient.shape[0]
    if body_pose.shape != (frame_count, 63):
        raise MotionXppError(f"body_pose must be [F,63], got {tuple(body_pose.shape)}")
    if transl.shape != (frame_count, 3):
        raise MotionXppError(f"transl must be [F,3], got {tuple(transl.shape)}")
    if betas.ndim == 1:
        if betas.numel() < 10:
            raise MotionXppError(f"betas must contain at least 10 values, got {betas.numel()}")
        betas = betas[:10]
    elif betas.ndim == 2:
        if betas.shape[1] < 10 or betas.shape[0] not in (1, frame_count):
            raise MotionXppError(
                f"betas must be [10], [1,>=10], or [F,>=10], got {tuple(betas.shape)}"
            )
        betas = betas[:, :10]
    else:
        raise MotionXppError(f"betas has invalid shape {tuple(betas.shape)}")
    if result.get("fps") is not None and (not np.isfinite(result["fps"]) or result["fps"] <= 0):
        raise MotionXppError(f"fps must be finite and positive, got {result['fps']}")
    gender = str(result.get("gender", "neutral")).lower()
    if gender not in {"neutral", "male", "female"}:
        gender = "neutral"
    return {
        **result,
        "global_orient": global_orient.contiguous(),
        "body_pose": body_pose.contiguous(),
        "transl": transl.contiguous(),
        "betas": betas.contiguous(),
        "gender": gender,
    }


def _stable_unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.strip().split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def _extract_text_values(value: Any) -> tuple[list[str], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if isinstance(value, str):
        return [value], metadata
    if isinstance(value, bytes):
        return [value.decode("utf-8-sig")], metadata
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _extract_text_values(value.item())
        return _extract_text_values(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        captions: list[str] = []
        for item in value:
            nested, nested_meta = _extract_text_values(item)
            captions.extend(nested)
            metadata.update(nested_meta)
        return captions, metadata
    if isinstance(value, Mapping):
        style = _find_first(value, ("style",))
        action = _find_first(value, ("action", "label"))
        segment = _find_first(value, ("segment", "time_segment", "interval"))
        if style is not None:
            metadata["style"] = style
        if action is not None:
            metadata["action"] = action
        if segment is not None:
            metadata["segment"] = segment
        captions: list[str] = []
        for key in ("caption", "captions", "text", "texts", "semantic_label", "description"):
            nested = _find_first(value, (key,))
            if nested is not None:
                found, _ = _extract_text_values(nested)
                captions.extend(found)
        if not captions and isinstance(action, str):
            generated = action.strip()
            if isinstance(style, str) and style.strip():
                generated = f"{generated} in a {style.strip()} style"
            captions.append(generated)
            metadata["caption_derived_from_action_style"] = True
        return captions, metadata
    return [], metadata


def parse_text_asset(ref: AssetRef) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """解析字符串、列表、JSON/NPY 文本并稳定去重。"""
    if ref.suffix == ".txt":
        raw = ref.read_bytes().decode("utf-8-sig")
        # 官方文件是一段文本；多行文件按非空行保留多 caption。
        value: Any = [line for line in raw.splitlines() if line.strip()] or [raw]
    elif ref.suffix == ".json":
        value = json.loads(ref.read_bytes().decode("utf-8-sig"))
    elif ref.suffix in {".npy", ".npz"}:
        value = _numpy_from_bytes(ref)
    else:
        raise MotionXppError(f"Unsupported text asset: {ref.source_path}")
    captions, metadata = _extract_text_values(value)
    captions = _stable_unique_strings(captions)
    if not captions:
        raise MotionXppError(f"{ref.source_path}: no non-empty semantic caption")
    records = [{"caption": caption} for caption in captions]
    for record in records:
        for key in ("style", "action", "segment"):
            if key in metadata:
                record[key] = metadata[key]
    return records, metadata


def parse_keypoint_asset(ref: AssetRef) -> dict[str, Any]:
    """审计 COCO-WholeBody JSON，不臆造缺失的图像/相机字段。"""
    value = _numpy_from_bytes(ref)
    if not isinstance(value, Mapping):
        raise MotionXppError(f"{ref.source_path}: keypoints must be a JSON/dict")
    annotations = value.get("annotations")
    images = value.get("images")
    if not isinstance(annotations, list) or not annotations:
        raise MotionXppError(f"{ref.source_path}: annotations must be non-empty")
    body_frames: list[torch.Tensor] = []
    for frame_index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping) or "body_kpts" not in annotation:
            raise MotionXppError(f"annotation {frame_index} is missing body_kpts")
        body = _as_float_tensor(annotation["body_kpts"], f"body_kpts[{frame_index}]")
        if body.shape != (17, 3):
            raise MotionXppError(
                f"body_kpts[{frame_index}] must be [17,3], got {tuple(body.shape)}"
            )
        body_frames.append(body)
    kp2d = torch.stack(body_frames).contiguous()
    width = value.get("width")
    height = value.get("height")
    if isinstance(images, list) and images:
        width = width if width is not None else images[0].get("width")
        height = height if height is not None else images[0].get("height")
    camera = value.get("camera_params") or value.get("cam_params") or value.get("K")
    return {
        "kp2d": kp2d,
        "frame_count": kp2d.shape[0],
        "coordinate_min": kp2d[..., :2].amin(dim=(0, 1)).tolist(),
        "coordinate_max": kp2d[..., :2].amax(dim=(0, 1)).tolist(),
        "confidence_min": float(kp2d[..., 2].min()),
        "confidence_max": float(kp2d[..., 2].max()),
        "confidence_values": sorted(float(v) for v in torch.unique(kp2d[..., 2])),
        "image_size": [int(width), int(height)] if width and height else None,
        "has_camera_intrinsics": camera is not None,
        "images_count": len(images) if isinstance(images, list) else None,
        "annotation_keys": sorted(str(key) for key in annotations[0]),
    }


def coordinate_rotation(source_up_axis: str) -> torch.Tensor:
    """返回 source 坐标到 GENMO AY/Y-up 的右手固定旋转。"""
    axis = source_up_axis.lower().strip()
    if axis == "y":
        return torch.eye(3, dtype=torch.float32)
    if axis == "z":
        # 与 gem.utils.motion_utils 的 az->ay 完全一致：绕 x 轴 -90°。
        return torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
    if axis == "x":
        # 绕 z 轴 +90°，把 +x 映射到 +y。
        return torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
    raise ValueError(f"--source-up-axis must be one of x/y/z, got {source_up_axis!r}")


def convert_coordinate_system(
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    source_up_axis: str,
    points3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """用同一固定旋转变换 root orientation、translation 和可选 3D 点。"""
    rotation = coordinate_rotation(source_up_axis).to(global_orient)
    root_matrix = quaternion_to_matrix(axis_angle_to_quaternion(global_orient))
    converted_root = matrix_to_axis_angle(rotation @ root_matrix)
    converted_transl = torch.einsum("ij,fj->fi", rotation, transl)
    converted_points = None
    if points3d is not None:
        if points3d.shape[-1] != 3:
            raise ValueError("points3d must end in xyz")
        converted_points = torch.einsum("ij,...j->...i", rotation, points3d)
    return (
        converted_root.contiguous(),
        converted_transl.contiguous(),
        None if converted_points is None else converted_points.contiguous(),
        rotation,
    )


def _slerp_quaternion(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """对批量 wxyz quaternion 做 shortest-path SLERP。"""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    safe = sin_theta.abs() > 1e-6
    denominator = torch.where(safe, sin_theta, torch.ones_like(sin_theta))
    while alpha.ndim < q0.ndim:
        alpha = alpha.unsqueeze(-1)
    weight0 = torch.where(safe, torch.sin((1.0 - alpha) * theta) / denominator, 1.0 - alpha)
    weight1 = torch.where(safe, torch.sin(alpha * theta) / denominator, alpha)
    result = weight0 * q0 + weight1 * q1
    return result / result.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def resample_motion(
    motion: Mapping[str, Any], source_fps: float, target_fps: float
) -> dict[str, Any]:
    """用旋转 SLERP 和线性位移把动作统一到目标 FPS。"""
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    global_orient = motion["global_orient"].float()
    body_pose = motion["body_pose"].float()
    transl = motion["transl"].float()
    betas = motion["betas"].float()
    source_len = global_orient.shape[0]
    if source_len < 2:
        raise MotionXppError("motion needs at least two source frames")
    target_len = int(round((source_len - 1) * target_fps / source_fps)) + 1
    if target_len == source_len and abs(source_fps - target_fps) < 1e-9:
        return {
            **motion,
            "global_orient": global_orient.contiguous().clone(),
            "body_pose": body_pose.contiguous().clone(),
            "transl": transl.contiguous().clone(),
            "betas": betas.contiguous().clone(),
            "source_fps": float(source_fps),
            "fps": float(target_fps),
        }
    positions = torch.linspace(0, source_len - 1, target_len)
    left = positions.floor().long().clamp(max=source_len - 1)
    right = (left + 1).clamp(max=source_len - 1)
    alpha = positions - left.float()
    all_pose = torch.cat([global_orient[:, None], body_pose.reshape(source_len, 21, 3)], 1)
    q = axis_angle_to_quaternion(all_pose)
    resampled_pose = quaternion_to_axis_angle(_slerp_quaternion(q[left], q[right], alpha))
    resampled_transl = transl[left] + (transl[right] - transl[left]) * alpha[:, None]
    if betas.ndim == 1:
        resampled_betas = betas.contiguous().clone()
    else:
        beta_left = betas[left.clamp(max=betas.shape[0] - 1)]
        beta_right = betas[right.clamp(max=betas.shape[0] - 1)]
        resampled_betas = beta_left + (beta_right - beta_left) * alpha[:, None]
    return {
        **motion,
        "global_orient": resampled_pose[:, 0].contiguous(),
        "body_pose": resampled_pose[:, 1:].reshape(target_len, 63).contiguous(),
        "transl": resampled_transl.contiguous(),
        "betas": resampled_betas.contiguous(),
        "source_fps": float(source_fps),
        "fps": float(target_fps),
    }


def motion_group(stem: str) -> str:
    """去除尾部 clip 编号，避免同一原序列片段跨 split。"""
    return re.sub(r"(?:_clip\d+)+$", "", stem, flags=re.IGNORECASE)


def deterministic_split(subset: str, stem: str, seed: int) -> str:
    """按 subset/source group 做稳定 98/1/1 hash split。"""
    group = f"{subset}:{motion_group(stem)}"
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / float(
        2**64
    )
    if value < 0.98:
        return "train"
    if value < 0.99:
        return "val"
    return "test"


def content_hash(
    pose: torch.Tensor,
    trans: torch.Tensor,
    betas: torch.Tensor,
    *,
    subset: str | None = None,
    source_path: str | None = None,
) -> tuple[str, str]:
    """返回纯动作 hash 及包含 source provenance 的 content hash。"""
    digest = hashlib.sha256()
    for tensor in (pose, trans, betas):
        array = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    motion_digest = digest.hexdigest()
    source_digest = hashlib.sha256()
    source_digest.update(motion_digest.encode())
    source_digest.update((subset or "").encode())
    source_digest.update((source_path or "").encode())
    return motion_digest, source_digest.hexdigest()


def anomaly_statistics(pose: torch.Tensor, trans: torch.Tensor, fps: float) -> dict[str, float]:
    """记录速度、位移和轴角幅值；不据此静默删除样本。"""
    speed = (trans[1:] - trans[:-1]).norm(dim=-1) * fps if trans.shape[0] > 1 else torch.zeros(1)
    return {
        "max_root_speed_mps": float(speed.max()),
        "root_displacement_m": float((trans[-1] - trans[0]).norm()),
        "max_axis_angle_rad": float(pose.reshape(-1, 3).norm(dim=-1).max()),
    }


def validate_record(record: Mapping[str, Any], motion_id: str = "<record>") -> None:
    """验证一个最终 motion shard record。"""
    required = ("pose", "trans", "beta", "gender", "fps", "text_data")
    missing = [key for key in required if key not in record]
    if missing:
        raise MotionXppError(f"{motion_id}: missing fields {missing}")
    pose = record["pose"]
    trans = record["trans"]
    beta = record["beta"]
    if not isinstance(pose, torch.Tensor) or pose.ndim != 2 or pose.shape[1] != 66:
        raise MotionXppError(f"{motion_id}: pose must be Tensor[F,66]")
    if not isinstance(trans, torch.Tensor) or trans.shape != (pose.shape[0], 3):
        raise MotionXppError(f"{motion_id}: trans must be Tensor[F,3]")
    if not isinstance(beta, torch.Tensor) or (
        beta.shape != (10,) and beta.shape != (pose.shape[0], 10)
    ):
        raise MotionXppError(f"{motion_id}: beta must be Tensor[10] or Tensor[F,10]")
    for name, tensor in (("pose", pose), ("trans", trans), ("beta", beta)):
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise MotionXppError(f"{motion_id}: {name} must be CPU float32")
        if not tensor.is_contiguous() or not torch.isfinite(tensor).all():
            raise MotionXppError(f"{motion_id}: {name} must be contiguous and finite")
    if pose.shape[0] < MIN_FRAMES:
        raise MotionXppError(f"{motion_id}: fewer than {MIN_FRAMES} frames")
    if not np.isfinite(float(record["fps"])) or float(record["fps"]) <= 0:
        raise MotionXppError(f"{motion_id}: invalid fps")
    text_data = record["text_data"]
    if not isinstance(text_data, list) or not text_data:
        raise MotionXppError(f"{motion_id}: text_data must be non-empty")
    for item in text_data:
        if not isinstance(item, Mapping) or not str(item.get("caption", "")).strip():
            raise MotionXppError(f"{motion_id}: invalid caption")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 行。"""
    result: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            result.append(value)
    return result


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """流式原子写入 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
