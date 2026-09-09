# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""MotionMillion 纯文本到 SMPL 动作训练数据集。

Dataset 只读取构建器已经验证并转换好的 motion/embedding release：动作分片保存
``pose[F,66]``、``trans[F,3]`` 和共享 beta，文本分片按有效 token 紧凑保存。
sample index 是 mmap 的结构化 NumPy 数组，因此百万级数据不会在每个 DDP rank 中
展开成巨大的 Python manifest。

训练样本固定为 120 帧：短动作补齐且保留有效长度 mask，长动作连续裁剪。每次从
该 motion 的 caption 中均匀选择一条，恢复为 ``text_embed[150,1024]`` 和
``text_attention_mask[150]``。本数据集不提供图像、2D、音乐或语音条件。
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gem.utils.net_utils import get_valid_mask
from gem.utils.pylogger import Log
from tools.data.motionmillion.common import (
    MAX_TEXT_TOKENS,
    SAMPLE_INDEX_DTYPE,
    SCHEMA_VERSION,
    TEXT_HIDDEN_DIM,
    MotionMillionError,
    read_json,
    safe_torch_load,
    validate_embedding_record,
    validate_motion_record,
)

from .base_dataset import BaseDataset
from .utils import pad_data


class MotionMillionDataset(BaseDataset):
    """按紧凑索引和 worker 本地 LRU 读取 MotionMillion SMPL+文本分片。"""

    def __init__(
        self,
        root: str | Path = "inputs/MotionMillion",
        motion_manifest_path: str | Path = (
            "inputs/MotionMillion/genmo_smpl_v1/manifests/train.json"
        ),
        embedding_manifest_path: str | Path = (
            "inputs/MotionMillion/t5_3b_v1_fp16/manifests/train.json"
        ),
        split: str = "train",
        motion_frames: int = 120,
        cam_augmentation: str = "static",
        limit_size: int | None = None,
        shard_cache_size: int = 2,
        random_seed: int = 20260909,
        source_up_axis: str = "y",
        random_crop: bool | None = None,
        mode: str = "default",
    ) -> None:
        if motion_frames <= 0:
            raise ValueError("motion_frames 必须为正数")
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size 必须为正数")
        if source_up_axis.lower() != "y":
            raise ValueError("MotionMillion 正式转换输出固定为 Y-up")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"无效 split: {split}")
        self.root = Path(root)
        self.motion_manifest_path = Path(motion_manifest_path)
        self.embedding_manifest_path = Path(embedding_manifest_path)
        self.split = split
        self.motion_frames = int(motion_frames)
        self.shard_cache_size = int(shard_cache_size)
        self.random_seed = int(random_seed)
        self.source_up_axis = source_up_axis.lower()
        self.random_crop = split == "train" if random_crop is None else bool(random_crop)
        self.mode = mode
        self.dataset_name = "MotionMillion"
        self._motion_cache: OrderedDict[str, Any] = OrderedDict()
        self._embedding_cache: OrderedDict[str, Any] = OrderedDict()
        self._rng_pid: int | None = None
        self._rng: np.random.RandomState | None = None
        super().__init__(cam_augmentation, limit_size)

    @staticmethod
    def _resolve(base: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else base / path

    def _load_dataset(self) -> None:
        if not self.motion_manifest_path.is_file():
            raise FileNotFoundError(f"motion manifest 不存在: {self.motion_manifest_path}")
        if not self.embedding_manifest_path.is_file():
            raise FileNotFoundError(f"embedding manifest 不存在: {self.embedding_manifest_path}")
        self.motion_root = self.motion_manifest_path.parent.parent
        self.embedding_root = self.embedding_manifest_path.parent.parent
        self.motion_manifest = read_json(self.motion_manifest_path)
        self.embedding_manifest = read_json(self.embedding_manifest_path)
        for name, manifest in (
            ("motion", self.motion_manifest),
            ("embedding", self.embedding_manifest),
        ):
            if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
                raise MotionMillionError(f"{name} manifest schema version 不兼容")
            if manifest.get("split") != self.split:
                raise MotionMillionError(
                    f"{name} manifest split={manifest.get('split')!r}，要求 {self.split!r}"
                )
        if self.motion_manifest.get("build_fingerprint") != self.embedding_manifest.get(
            "source_build_fingerprint"
        ):
            raise MotionMillionError("motion/embedding build fingerprint 不一致")
        if int(self.motion_manifest.get("motion_frames", -1)) != self.motion_frames:
            raise MotionMillionError(
                f"manifest motion_frames={self.motion_manifest.get('motion_frames')}，"
                f"Dataset 配置为 {self.motion_frames}"
            )
        if int(self.embedding_manifest.get("max_text_tokens", -1)) != MAX_TEXT_TOKENS:
            raise MotionMillionError("embedding manifest 不是 150-token v1 契约")
        if int(self.embedding_manifest.get("hidden_dim", -1)) != TEXT_HIDDEN_DIM:
            raise MotionMillionError("embedding hidden_dim 不是 1024")

        motion_shards = list(self.motion_manifest.get("shards", []))
        embedding_shards = list(self.embedding_manifest.get("shards", []))
        if len(motion_shards) != len(embedding_shards):
            raise MotionMillionError("motion/embedding shard 数量不一致")
        self.motion_shard_paths: list[Path] = []
        self.embedding_shard_paths: list[Path] = []
        for expected, (motion, embedding) in enumerate(zip(motion_shards, embedding_shards)):
            if int(motion.get("shard_id", -1)) != expected or int(
                embedding.get("shard_id", -1)
            ) != expected:
                raise MotionMillionError("motion/embedding shard_id 不连续或未对齐")
            if int(motion.get("record_count", -1)) != int(
                embedding.get("record_count", -2)
            ):
                raise MotionMillionError(f"shard {expected} record 数不一致")
            if embedding.get("source_motion_path") != motion.get("path"):
                raise MotionMillionError(f"shard {expected} source motion path 不一致")
            if embedding.get("source_motion_sha256") != motion.get("sha256"):
                raise MotionMillionError(f"shard {expected} source motion SHA256 不一致")
            self.motion_shard_paths.append(self._resolve(self.motion_root, str(motion["path"])))
            self.embedding_shard_paths.append(
                self._resolve(self.embedding_root, str(embedding["path"]))
            )

        index_path = self._resolve(
            self.motion_root, str(self.motion_manifest["sample_index_path"])
        )
        if not index_path.is_file():
            raise FileNotFoundError(f"sample index 不存在: {index_path}")
        self.sample_index = np.load(index_path, mmap_mode="r", allow_pickle=False)
        if self.sample_index.dtype != SAMPLE_INDEX_DTYPE or self.sample_index.ndim != 1:
            raise MotionMillionError(
                f"sample index dtype/shape 不符合 v1 契约: {self.sample_index.dtype}, "
                f"{self.sample_index.shape}"
            )
        if len(self.sample_index) != int(self.motion_manifest.get("sample_count", -1)):
            raise MotionMillionError("sample index 长度与 manifest sample_count 不一致")
        if len(self.sample_index) == 0:
            raise MotionMillionError("MotionMillion sample index 为空")
        max_shard = int(self.sample_index["shard_id"].max())
        if max_shard >= len(self.motion_shard_paths):
            raise MotionMillionError("sample index 包含越界 shard_id")
        self.sampling_summary = {
            "raw_sequences": int(self.motion_manifest["record_count"]),
            "hours": float(self.sample_index["frames"].sum()) / 30.0 / 3600.0,
            "duration_aware_sampling": True,
        }
        Log.info(
            f"[{self.dataset_name}] split={self.split}, "
            f"records={self.motion_manifest['record_count']}, samples={len(self.sample_index)}, "
            f"shards={len(self.motion_shard_paths)}"
        )

    def _get_idx2meta(self) -> None:
        # BaseDataset 只依赖该属性的长度；真正索引保留为 mmap 结构化数组。
        self.idx2meta = range(len(self.sample_index))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_motion_cache"] = OrderedDict()
        state["_embedding_cache"] = OrderedDict()
        state["_rng_pid"] = None
        state["_rng"] = None
        return state

    def _get_rng(self) -> np.random.RandomState:
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            # torch DataLoader 会为每个 worker 设置可复现 initial_seed。
            seed = (self.random_seed + int(torch.initial_seed())) % (2**32 - 1)
            self._rng = np.random.RandomState(seed)
            self._rng_pid = pid
        return self._rng

    def _cached_load(self, path: Path, cache: OrderedDict[str, Any]) -> list[dict[str, Any]]:
        key = str(path.resolve())
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        value = safe_torch_load(path)
        if not isinstance(value, list):
            raise MotionMillionError(f"shard 必须为 list: {path}")
        cache[key] = value
        while len(cache) > self.shard_cache_size:
            cache.popitem(last=False)
        return value

    def sample_shard_ids(self) -> np.ndarray:
        """返回 sampler 使用的只读 shard_id 视图。"""
        length = len(self) if self.limit_size is not None else len(self.sample_index)
        return np.asarray(self.sample_index["shard_id"][:length], dtype=np.int64)

    def _load_data(self, idx: int) -> dict[str, Any]:
        entry = self.sample_index[idx]
        shard_id = int(entry["shard_id"])
        record_index = int(entry["record_index"])
        motion_records = self._cached_load(
            self.motion_shard_paths[shard_id], self._motion_cache
        )
        embedding_records = self._cached_load(
            self.embedding_shard_paths[shard_id], self._embedding_cache
        )
        if record_index >= len(motion_records) or record_index >= len(embedding_records):
            raise MotionMillionError("sample index record_index 越界")
        motion = motion_records[record_index]
        embedding = embedding_records[record_index]
        validate_motion_record(motion)
        validate_embedding_record(embedding, caption_count=len(motion["captions"]))
        if str(motion["motion_id"]) != str(embedding.get("motion_id")):
            raise MotionMillionError("motion/embedding record motion_id 不一致")

        pose = motion["pose"].float()
        trans = motion["trans"].float()
        frames = int(pose.shape[0])
        target = self.motion_frames
        if frames > target:
            if self.random_crop:
                start = int(self._get_rng().randint(0, frames - target + 1))
            else:
                start = (frames - target) // 2
            pose = pose[start : start + target]
            trans = trans[start : start + target]
            valid_length = target
        else:
            start = 0
            valid_length = frames
            if frames < target:
                params = pad_data({"pose": pose, "trans": trans}, target)
                pose = params["pose"]
                trans = params["trans"]

        caption_count = len(motion["captions"])
        if self.random_crop:
            text_index = int(self._get_rng().randint(0, caption_count))
        else:
            text_index = 0
        offsets = embedding["offsets"]
        token_start = int(offsets[text_index])
        token_end = int(offsets[text_index + 1])
        token_count = token_end - token_start
        text_embed = torch.zeros(MAX_TEXT_TOKENS, TEXT_HIDDEN_DIM, dtype=torch.float32)
        text_embed[:token_count] = embedding["embeddings"][token_start:token_end].float()
        text_attention_mask = torch.zeros(MAX_TEXT_TOKENS, dtype=torch.bool)
        text_attention_mask[:token_count] = True

        beta = motion["beta"].float()
        beta_frames = beta.repeat(target, 1) if beta.ndim == 1 else beta[:target]
        return {
            "body_pose": pose[:, 3:66],
            "betas": beta_frames,
            "global_orient": pose[:, :3],
            "transl": trans,
            "data_name": "motionmillion",
            "motion_id": str(motion["motion_id"]),
            "source_subset": str(motion.get("source_subset", "unknown")),
            "source_archive": str(motion["source_archive"]),
            "caption": str(motion["captions"][text_index]),
            "text_index": text_index,
            "text_embed": text_embed,
            "text_attention_mask": text_attention_mask,
            "valid_length": valid_length,
            "crop_start": start,
        }

    def _process_data(self, data: dict[str, Any], idx: int) -> dict[str, Any]:
        metadata = {
            key: data[key]
            for key in (
                "motion_id",
                "source_subset",
                "source_archive",
                "caption",
                "text_index",
                "text_embed",
                "text_attention_mask",
                "valid_length",
                "crop_start",
            )
        }
        core = {
            "body_pose": data["body_pose"],
            "betas": data["betas"],
            "global_orient": data["global_orient"],
            "transl": data["transl"],
            "data_name": data["data_name"],
        }
        result = super()._process_data(core, idx)
        sequence_length = result["smpl_params_w"]["body_pose"].shape[0]
        result["length"] = int(metadata["valid_length"])
        result["caption"] = metadata["caption"]
        result["has_text"] = True
        result["text_embed"] = metadata["text_embed"]
        result["text_attention_mask"] = metadata["text_attention_mask"]
        result["meta"].update(
            {
                "dataset_id": "motionmillion",
                "mid": metadata["motion_id"],
                "motion_id": metadata["motion_id"],
                "text_ind": metadata["text_index"],
                "source_subset": metadata["source_subset"],
                "source_archive": metadata["source_archive"],
                "crop_start": metadata["crop_start"],
                "mode": self.mode,
            }
        )
        result["mask"]["valid"] = get_valid_mask(
            sequence_length, int(metadata["valid_length"])
        )
        result["mask"]["has_2d_mask"] = get_valid_mask(sequence_length, 0)
        result["mask"]["has_cam_mask"] = get_valid_mask(sequence_length, 0)
        result["mask"]["2d_only"] = False
        return result


__all__ = ["MotionMillionDataset"]
