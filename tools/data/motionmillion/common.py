#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""MotionMillion 数据契约与 272D/SMPL 表示转换的共享实现。

本模块集中定义本项目接入 MotionMillion 时不能漂移的基础约定：官方发布动作固定
30 FPS、Y-up，输入是 22 关节的 272 维局部表示，GENMO 侧输出是 66 维 SMPL
轴角姿态与三维世界平移。转换严格遵循 MotionMillion 官方
``recover_from_local_rotation`` 的累积顺序，不通过关节位置重新拟合 SMPL。

模块同时提供原子写入、内容哈希、紧凑 sample index 和 manifest 校验函数，供下载、
构建、embedding、Dataset 与评测工具共同使用。所有路径契约都使用相对路径，数据根
目录可以整体迁移；任何维度、有限性、FPS 或 split 异常都会显式失败，禁止静默修补。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

SCHEMA_VERSION = 1
OFFICIAL_FPS = 30.0
MIN_FRAMES = 60
MAX_FRAMES = 300
MOTION_DIM = 272
SMPL_POSE_DIM = 66
SMPL_BODY_POSE_DIM = 63
TEXT_HIDDEN_DIM = 1024
MAX_TEXT_TOKENS = 150
RECORDS_PER_SHARD = 512
SPLITS = ("train", "val", "test")

ROOT_PLANAR_VELOCITY = slice(0, 2)
ROOT_HEADING_DELTA_6D = slice(2, 8)
LOCAL_JOINT_POSITION = slice(8, 74)
LOCAL_JOINT_VELOCITY = slice(74, 140)
LOCAL_JOINT_ROTATION_6D = slice(140, 272)

SAMPLE_INDEX_DTYPE = np.dtype(
    [
        ("shard_id", "<i4"),
        ("record_index", "<i4"),
        ("frames", "<i2"),
        ("window_index", "<i2"),
    ]
)


class MotionMillionError(RuntimeError):
    """MotionMillion 数据或构建契约不成立时抛出。"""


class MotionMillionFilteredError(MotionMillionError):
    """记录命中明确过滤边界时抛出。"""


def sha256_bytes(value: bytes) -> str:
    """返回字节内容的 SHA256。"""
    return hashlib.sha256(value).hexdigest()


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


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """原子发布 JSONL。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """写入、重载验证并原子发布可信 PTH 分片。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        safe_torch_load(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npy(path: str | Path, value: np.ndarray) -> None:
    """通过文件句柄原子发布 NumPy 数组，避免 ``np.save`` 自动改后缀。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    with temporary.open("rb") as handle:
        loaded = np.load(handle, allow_pickle=False)
        if loaded.shape != value.shape or loaded.dtype != value.dtype:
            raise MotionMillionError(f"原子 NPY 重载验证失败: {target}")
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def safe_torch_load(path_or_file: Any) -> Any:
    """兼容不同 PyTorch 版本地加载本工具生成的可信制品。"""
    try:
        return torch.load(path_or_file, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path_or_file, map_location="cpu")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


_MIRROR_PREFIX = re.compile(r"^(?:mirror[_-]?|m[_-]|m(?=\d))", re.IGNORECASE)
_MIRROR_SUFFIX = re.compile(r"(?:[_-](?:mirror|mirrored))$", re.IGNORECASE)


def mirror_base_id(motion_id: str) -> str:
    """生成仅用于 split 泄漏审计的镜像归一化 ID。"""
    parts = []
    for part in motion_id.replace("\\", "/").split("/"):
        value = _MIRROR_PREFIX.sub("", part)
        value = _MIRROR_SUFFIX.sub("", value)
        if value.lower() not in {"mirror", "mirrored"}:
            parts.append(value)
    return "/".join(parts).lower()


def _as_float_motion(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    motion = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if motion.ndim != 2 or tuple(motion.shape[1:]) != (MOTION_DIM,):
        raise MotionMillionError(
            f"MotionMillion motion 必须为 [F,{MOTION_DIM}]，实际为 {tuple(motion.shape)}"
        )
    if not torch.isfinite(motion).all():
        raise MotionMillionError("MotionMillion motion 包含 NaN 或 Inf")
    frames = int(motion.shape[0])
    if frames < MIN_FRAMES or frames > MAX_FRAMES:
        raise MotionMillionFilteredError(
            f"动作长度 {frames} 不在正式区间 [{MIN_FRAMES},{MAX_FRAMES}]"
        )
    return motion.contiguous()


def accumulate_heading_rotations(relative_rotations: torch.Tensor) -> torch.Tensor:
    """按官方 ``R_rel @ R_previous`` 顺序累积根 heading。"""
    if relative_rotations.ndim != 3 or relative_rotations.shape[-2:] != (3, 3):
        raise ValueError("relative_rotations 必须为 [F,3,3]")
    if relative_rotations.shape[0] == 0:
        raise ValueError("relative_rotations 不能为空")
    totals = [relative_rotations[0]]
    for relative in relative_rotations[1:]:
        totals.append(relative @ totals[-1])
    return torch.stack(totals, dim=0)


def recover_smpl_from_272(value: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
    """严格按官方语义把 272D 动作恢复为 GENMO 使用的 SMPL 参数。

    返回 ``pose[F,66]``、``trans[F,3]`` 和共享 ``beta[10]``。姿态的前 3 维是
    global orientation，后 63 维是 21 个 SMPL body joint；官方为兼容 SMPL+H
    额外补的两个零关节不进入 GENMO 的 66D pose。
    """
    motion = _as_float_motion(value)
    frames = int(motion.shape[0])
    local_rotations = rotation_6d_to_matrix(
        motion[:, LOCAL_JOINT_ROTATION_6D].reshape(frames, 22, 6)
    )
    heading_delta = rotation_6d_to_matrix(motion[:, ROOT_HEADING_DELTA_6D])
    heading = accumulate_heading_rotations(heading_delta)
    inverse_heading = heading.transpose(-1, -2)

    rotations = local_rotations.clone()
    rotations[:, 0] = inverse_heading @ rotations[:, 0]

    local_positions = motion[:, LOCAL_JOINT_POSITION].reshape(frames, 22, 3)
    # 官方 NumPy 实现以 ``np.zeros`` 建立速度缓冲（默认 float64），再做累计。
    # 保留该数值细节可避免 300 帧 FP32 cumsum 把亚微米误差累积到 parity 门外。
    root_velocity = torch.zeros((frames, 3), dtype=torch.float64)
    root_velocity[:, 0] = motion[:, ROOT_PLANAR_VELOCITY.start]
    root_velocity[:, 2] = motion[:, ROOT_PLANAR_VELOCITY.start + 1]
    if frames > 1:
        root_velocity[1:] = torch.einsum(
            "fij,fj->fi", inverse_heading[:-1].double(), root_velocity[1:]
        )
    translation = torch.cumsum(root_velocity, dim=0)
    translation[:, 1] = local_positions[:, 0, 1].double()

    pose = matrix_to_axis_angle(rotations).reshape(frames, SMPL_POSE_DIM)
    if not torch.isfinite(pose).all() or not torch.isfinite(translation).all():
        raise MotionMillionError("272D 恢复后的 SMPL 参数包含 NaN 或 Inf")
    return {
        "pose": pose.float().cpu().contiguous(),
        "trans": translation.float().cpu().contiguous(),
        "beta": torch.zeros(10, dtype=torch.float32),
    }


def _rotation_y(angle: torch.Tensor) -> torch.Tensor:
    """构造批量 Y 轴旋转矩阵。"""
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    return torch.stack(
        [
            cosine,
            zeros,
            sine,
            zeros,
            ones,
            zeros,
            -sine,
            zeros,
            cosine,
        ],
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def smpl_to_272(
    pose: np.ndarray | torch.Tensor,
    trans: np.ndarray | torch.Tensor,
    joint_positions_world: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """把生成的 SMPL 动作编码成官方 272D 评测表示。

    该方向需要 SMPL FK 得到的 22 个世界关节位置。heading 取 root forward 在 XZ
    平面的朝向，并把该朝向对齐到 +Z；这与恢复公式互逆。若 root forward 几乎竖直，
    表示没有稳定 heading，函数会显式拒绝该记录。
    """
    pose_tensor = torch.as_tensor(pose, dtype=torch.float32, device="cpu")
    trans_tensor = torch.as_tensor(trans, dtype=torch.float32, device="cpu")
    joints = torch.as_tensor(joint_positions_world, dtype=torch.float32, device="cpu")
    if pose_tensor.ndim != 2 or pose_tensor.shape[1] != SMPL_POSE_DIM:
        raise MotionMillionError(f"pose 必须为 [F,{SMPL_POSE_DIM}]")
    frames = int(pose_tensor.shape[0])
    if tuple(trans_tensor.shape) != (frames, 3) or tuple(joints.shape) != (frames, 22, 3):
        raise MotionMillionError("trans/joint_positions_world 与 pose 帧数或维度不一致")
    if not all(torch.isfinite(item).all() for item in (pose_tensor, trans_tensor, joints)):
        raise MotionMillionError("SMPL→272D 输入包含 NaN 或 Inf")

    rotations = axis_angle_to_matrix(pose_tensor.reshape(frames, 22, 3))
    root_forward = rotations[:, 0, :, 2]
    forward_norm = torch.linalg.vector_norm(root_forward[:, [0, 2]], dim=-1)
    if (forward_norm < 1.0e-6).any():
        raise MotionMillionError("root forward 几乎垂直，无法稳定提取 heading")
    facing_yaw = torch.atan2(root_forward[:, 0], root_forward[:, 2])
    heading = _rotation_y(-facing_yaw)

    heading_delta = torch.empty_like(heading)
    heading_delta[0] = heading[0]
    if frames > 1:
        heading_delta[1:] = heading[1:] @ heading[:-1].transpose(-1, -2)
    heading_delta_6d = matrix_to_rotation_6d(heading_delta)
    # 6D 落盘会经历一次正交化；后续字段使用同一量化后 heading，保证评测
    # adapter 生成的各通道彼此严格自洽，避免长序列中累计约 1e-6 m 漂移。
    encoded_heading = accumulate_heading_rotations(
        rotation_6d_to_matrix(heading_delta_6d)
    )

    local_rotations = rotations.clone()
    local_rotations[:, 0] = encoded_heading @ rotations[:, 0]

    centered_joints = joints.clone()
    centered_joints[:, :, 0] -= trans_tensor[:, None, 0]
    centered_joints[:, :, 2] -= trans_tensor[:, None, 2]
    local_positions = torch.einsum("fij,fkj->fki", encoded_heading, centered_joints)

    world_velocity = torch.zeros_like(trans_tensor)
    world_velocity[0] = trans_tensor[0]
    if frames > 1:
        world_velocity[1:] = trans_tensor[1:] - trans_tensor[:-1]
    local_root_velocity = world_velocity.clone()
    if frames > 1:
        # 恢复端严格使用 transpose；FP32 正交矩阵仍有约 1e-7 的数值残差。
        # 这里解线性方程而非再次假设 transpose==inverse，使适配器 round-trip
        # 在 300 帧累计后仍满足 1e-6 m 门限。
        local_root_velocity[1:] = torch.linalg.solve(
            encoded_heading[:-1].transpose(-1, -2), world_velocity[1:, :, None]
        ).squeeze(-1)

    local_joint_velocity = torch.zeros_like(local_positions)
    if frames > 1:
        local_joint_velocity[1:] = local_positions[1:] - local_positions[:-1]

    result = torch.cat(
        [
            local_root_velocity[:, [0, 2]],
            heading_delta_6d,
            local_positions.reshape(frames, -1),
            local_joint_velocity.reshape(frames, -1),
            matrix_to_rotation_6d(local_rotations).reshape(frames, -1),
        ],
        dim=-1,
    )
    if tuple(result.shape) != (frames, MOTION_DIM) or not torch.isfinite(result).all():
        raise MotionMillionError("SMPL→272D 输出契约失败")
    return result.float().cpu().contiguous()


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
    if not torch.isfinite(pose).all() or not torch.isfinite(trans).all():
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


def build_sample_index(shard_records: Sequence[Sequence[Mapping[str, Any]]], motion_frames: int) -> np.ndarray:
    """为 v1 分片记录生成一条动作对应一条样本的紧凑索引。

    120--300 帧动作由 Dataset 在每次读取时随机裁一个连续 120 帧窗口，而不是
    在索引中复制同一 motion。``window_index`` 在 v1 恒为 0，仅为后续动态长度版本
    预留，借此保证同一个 epoch 内一个动作不会被分配给多个 DDP rank。
    """
    if motion_frames <= 0:
        raise ValueError("motion_frames 必须为正数")
    rows: list[tuple[int, int, int, int]] = []
    for shard_id, records in enumerate(shard_records):
        for record_index, record in enumerate(records):
            frames = int(record["pose"].shape[0])
            rows.append((shard_id, record_index, frames, 0))
    return np.asarray(rows, dtype=SAMPLE_INDEX_DTYPE)


__all__ = [
    "LOCAL_JOINT_POSITION",
    "LOCAL_JOINT_ROTATION_6D",
    "LOCAL_JOINT_VELOCITY",
    "MAX_FRAMES",
    "MAX_TEXT_TOKENS",
    "MIN_FRAMES",
    "MOTION_DIM",
    "MotionMillionError",
    "MotionMillionFilteredError",
    "OFFICIAL_FPS",
    "RECORDS_PER_SHARD",
    "ROOT_HEADING_DELTA_6D",
    "ROOT_PLANAR_VELOCITY",
    "SAMPLE_INDEX_DTYPE",
    "SCHEMA_VERSION",
    "SMPL_POSE_DIM",
    "SPLITS",
    "TEXT_HIDDEN_DIM",
    "accumulate_heading_rotations",
    "atomic_save_npy",
    "atomic_torch_save",
    "atomic_write_json",
    "atomic_write_jsonl",
    "build_sample_index",
    "identifier_candidates",
    "mirror_base_id",
    "normalize_member_name",
    "read_json",
    "recover_smpl_from_272",
    "safe_torch_load",
    "sha256_bytes",
    "sha256_file",
    "smpl_to_272",
    "validate_embedding_record",
    "validate_motion_record",
]
