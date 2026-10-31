# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""文本来源身份、哈希与已发布 T5 特征的公共契约。

提供归档 ID、镜像母来源、官方 split 解析与原子 JSON 写入；读取历史预计算 T5
时继续核验其绑定的源 motion record，保留既有 60--300 帧源格式约束。该约束只适用
旧特征来源，不限制当前 BUMI 完整 qpos 数据；本模块不执行 272D/SMPL 转换。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import torch

OFFICIAL_FPS = 30.0
MIN_FRAMES = 60
MAX_FRAMES = 300
SMPL_POSE_DIM = 66
TEXT_HIDDEN_DIM = 1024
MAX_TEXT_TOKENS = 150
SPLITS = ("train", "val", "test")
_MIRROR_PREFIX = re.compile(r"^(?:mirror[_-]?|m[_-]|m(?=\d))", re.IGNORECASE)
_MIRROR_SUFFIX = re.compile(r"(?:[_-](?:mirror|mirrored))$", re.IGNORECASE)


class MotionMillionError(RuntimeError):
    """MotionMillion 数据或构建契约不成立时抛出。"""


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """流式计算普通文件 SHA256，避免把大归档一次性载入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """以 UTF-8、保留 Unicode 的格式原子发布 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def normalize_member_name(value: str) -> str:
    """把 tar member 名称规范成稳定的 POSIX 相对路径。"""
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise MotionMillionError(f"归档成员路径不安全: {value!r}")
    return path.as_posix()


def identifier_candidates(member_name: str) -> list[str]:
    """从官方不同归档前缀中生成可能的 motion/text ID。

    最终 ID 仍必须在官方 split/text 数据库中命中；本函数不会仅凭 basename 猜测并
    接收记录。
    """
    normalized = normalize_member_name(member_name)
    path = PurePosixPath(normalized)
    without_suffix = path.with_suffix("").as_posix()
    candidates = [without_suffix]
    markers = (
        "motion_data/vector_272/",
        "motion_272rpr/",
        "vector_272/",
        "texts/",
        "text/",
    )
    lowered = without_suffix.lower()
    for marker in markers:
        position = lowered.find(marker.lower())
        if position >= 0:
            candidates.append(without_suffix[position + len(marker) :])
    parts = PurePosixPath(without_suffix).parts
    for drop in range(1, min(4, len(parts))):
        candidates.append(PurePosixPath(*parts[drop:]).as_posix())
    candidates.append(PurePosixPath(without_suffix).name)
    result: list[str] = []
    for candidate in candidates:
        value = candidate.strip("/")
        if value and value not in result:
            result.append(value)
    return result


def mirror_base_id(motion_id: str) -> str:
    """生成仅用于 split 泄漏审计、但保留路径大小写的镜像归一化 ID。"""
    parts = []
    for part in motion_id.replace("\\", "/").split("/"):
        value = _MIRROR_PREFIX.sub("", part)
        value = _MIRROR_SUFFIX.sub("", value)
        if value.lower() not in {"mirror", "mirrored"}:
            parts.append(value)
    # tar/Unix 路径大小写敏感；官方数据中确实存在只在 Up/up 上不同的两个 ID。
    # 全量 lower 会把它们误合并为同一动作，进而制造虚假的跨 split 泄漏。
    return "/".join(parts)


def validate_motion_record(record: Mapping[str, Any]) -> None:
    """验证一个已转换 motion shard record。"""
    required = {
        "motion_id",
        "pose",
        "trans",
        "beta",
        "captions",
        "fps",
        "source_up_axis",
        "source_archive",
        "source_archive_sha256",
        "source_subset",
        "source_member",
        "source_sha256",
        "source_text_member",
        "source_text_sha256",
        "split",
    }
    missing = sorted(required - set(record))
    if missing:
        raise MotionMillionError(f"motion record 缺少字段: {missing}")
    pose = record["pose"]
    trans = record["trans"]
    beta = record["beta"]
    if not isinstance(pose, torch.Tensor) or pose.ndim != 2 or pose.shape[1] != SMPL_POSE_DIM:
        raise MotionMillionError("motion record pose 必须为 [F,66] Tensor")
    frames = int(pose.shape[0])
    if not isinstance(trans, torch.Tensor) or tuple(trans.shape) != (frames, 3):
        raise MotionMillionError("motion record trans 必须为 [F,3] Tensor")
    if not isinstance(beta, torch.Tensor) or tuple(beta.shape) != (10,):
        raise MotionMillionError("motion record beta 必须为 [10] Tensor")
    if frames < MIN_FRAMES or frames > MAX_FRAMES:
        raise MotionMillionError("motion record 帧数超出正式区间")
    if float(record["fps"]) != OFFICIAL_FPS:
        raise MotionMillionError("motion record FPS 必须为 30")
    if str(record["source_up_axis"]).lower() != "y":
        raise MotionMillionError("motion record source_up_axis 必须为 y")
    if str(record["source_subset"]) not in {"MotionGV", "MotionLLAMA", "MotionUnion"}:
        raise MotionMillionError("motion record source_subset 不属于官方三类来源")
    for key in ("source_archive_sha256", "source_sha256", "source_text_sha256"):
        value = str(record[key])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise MotionMillionError(f"motion record {key} 不是 SHA256")
    if str(record["split"]) not in SPLITS:
        raise MotionMillionError("motion record split 无效")
    captions = record["captions"]
    if not isinstance(captions, list) or not captions:
        raise MotionMillionError("motion record captions 必须为非空列表")
    if any(not isinstance(value, str) or not value.strip() for value in captions):
        raise MotionMillionError("motion record 包含空 caption")
    if not pose.is_contiguous() or not trans.is_contiguous():
        raise MotionMillionError("motion record tensor 必须连续")
    if (
        not torch.isfinite(pose).all()
        or not torch.isfinite(trans).all()
        or not torch.isfinite(beta).all()
    ):
        raise MotionMillionError("motion record tensor 包含 NaN 或 Inf")


def validate_embedding_record(record: Mapping[str, Any], *, caption_count: int) -> None:
    """验证紧凑 T5 embedding record。"""
    embeddings = record.get("embeddings")
    offsets = record.get("offsets")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise MotionMillionError("embedding record embeddings 必须为 [T,1024] Tensor")
    if embeddings.shape[1] != TEXT_HIDDEN_DIM or embeddings.dtype != torch.float16:
        raise MotionMillionError("embedding record 必须是 CPU FP16 [T,1024]")
    if embeddings.device.type != "cpu" or not embeddings.is_contiguous():
        raise MotionMillionError("embedding record 必须是连续 CPU Tensor")
    if not isinstance(offsets, torch.Tensor) or offsets.dtype != torch.int64:
        raise MotionMillionError("embedding offsets 必须是 int64 Tensor")
    if tuple(offsets.shape) != (caption_count + 1,):
        raise MotionMillionError("embedding offsets 长度与 caption 数不一致")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(embeddings.shape[0]):
        raise MotionMillionError("embedding offsets 首尾不合法")
    lengths = offsets[1:] - offsets[:-1]
    if (lengths <= 0).any() or (lengths > MAX_TEXT_TOKENS).any():
        raise MotionMillionError("embedding token 长度不在 [1,150]")
    if not torch.isfinite(embeddings).all():
        raise MotionMillionError("embedding record 包含 NaN 或 Inf")


def _split_from_member(member_name: str) -> str | None:
    normalized = normalize_member_name(member_name).lower()
    if "t2m_60_300" not in normalized:
        return None
    for split in SPLITS:
        if normalized.endswith(f"/{split}.txt") or normalized == f"{split}.txt":
            return split
    return None


def _normalize_split_id(value: str) -> str:
    motion_id = value.strip().replace("\\", "/")
    if motion_id.endswith(".npy"):
        motion_id = motion_id[:-4]
    while motion_id.startswith("./"):
        motion_id = motion_id[2:]
    if not motion_id or motion_id.startswith("/") or ".." in motion_id.split("/"):
        raise MotionMillionError(f"官方 split 包含非法 motion ID: {value!r}")
    return motion_id
