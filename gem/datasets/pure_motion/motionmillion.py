# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""MotionMillion 纯文本到 SMPL 动作训练数据集。

Dataset 只读取构建器已经验证并转换好的 motion/embedding release：动作分片保存
``pose[F,66]``、``trans[F,3]`` 和共享 beta，文本分片按有效 token 紧凑保存。
sample index 是 mmap 的结构化 NumPy 数组，因此百万级数据不会在每个 DDP rank 中
展开成巨大的 Python manifest。

默认保持旧 120 帧裁剪路径；sequence_mode=full 保留完整 60—300 帧，在真实动作上
完成空间/相机增强及派生量计算之后尾部补齐到300。caption_sampling 独立控制文本
抽样，不再借用裁剪开关。文本始终为150个token；LRU中的原始记录不作原地修改。
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
    sha256_file,
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
        sequence_mode: str = "crop",
        pad_to_frames: int | None = None,
        caption_sampling: str | None = None,
    ) -> None:
        if motion_frames <= 0:
            raise ValueError("motion_frames 必须为正数")
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size 必须为正数")
        if source_up_axis.lower() != "y":
            raise ValueError("MotionMillion 正式转换输出固定为 Y-up")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"无效 split: {split}")
        if sequence_mode not in {"crop", "full"}:
            raise ValueError("sequence_mode 必须为 crop/full")
        if sequence_mode == "full" and random_crop is not None:
            raise ValueError("full 模式不接受 random_crop；文本抽样请设置 caption_sampling")
        self.sequence_mode = sequence_mode
        self.pad_to_frames = (300 if sequence_mode == "full" else motion_frames) if pad_to_frames is None else int(pad_to_frames)
        if sequence_mode == "full" and self.pad_to_frames != 300:
            raise ValueError("A0 full 模式必须 pad_to_frames=300")
        if sequence_mode == "crop" and self.pad_to_frames != motion_frames:
            raise ValueError("crop 模式 pad_to_frames 必须等于 motion_frames")
        self.root = Path(root)
        self.motion_manifest_path = Path(motion_manifest_path)
        self.embedding_manifest_path = Path(embedding_manifest_path)
        self.split = split
        self.motion_frames = int(motion_frames)
        self.shard_cache_size = int(shard_cache_size)
        self.random_seed = int(random_seed)
        self.source_up_axis = source_up_axis.lower()
        self.random_crop = split == "train" if random_crop is None else bool(random_crop)
        self.caption_sampling = caption_sampling or ("random" if self.random_crop else "first")
        if self.caption_sampling not in {"random", "first"}:
            raise ValueError("caption_sampling 必须为 random/first")
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
        if not self.motion_manifest.get("build_fingerprint") or self.motion_manifest.get("build_fingerprint") != self.embedding_manifest.get(
            "source_build_fingerprint"
        ):
            raise MotionMillionError("motion/embedding build fingerprint 不一致")
        release_motion_frames = int(self.motion_manifest.get("motion_frames", -1))
        if self.sequence_mode == "crop" and release_motion_frames != self.motion_frames:
            raise MotionMillionError(
                f"manifest motion_frames={self.motion_manifest.get('motion_frames')}，"
                f"Dataset 配置为 {self.motion_frames}"
            )
        if self.sequence_mode == "full":
            # v1 的 motion_frames 只参与构建身份和旧取样配置，分片本身保存完整动作。
            # 不重写 manifest/fingerprint；实际长度在索引及每条记录上继续核对。
            if release_motion_frames != self.motion_frames:
                raise MotionMillionError("full 模式 motion_frames 必须声明实际旧 release 值，不能伪装为 padding")
            if self.motion_manifest.get("fps") != 30 or self.motion_manifest.get("source_up_axis") != "y":
                raise MotionMillionError("full release 必须为 30 FPS / Y-up")
        if int(self.embedding_manifest.get("max_text_tokens", -1)) != MAX_TEXT_TOKENS:
            raise MotionMillionError("embedding manifest 不是 150-token v1 契约")
        if int(self.embedding_manifest.get("hidden_dim", -1)) != TEXT_HIDDEN_DIM:
            raise MotionMillionError("embedding hidden_dim 不是 1024")

        motion_shards = list(self.motion_manifest.get("shards", []))
        embedding_shards = list(self.embedding_manifest.get("shards", []))
        self._shard_record_counts = [int(row["record_count"]) for row in motion_shards]
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
        if max_shard >= len(self.motion_shard_paths) or (self.sample_index["shard_id"] < 0).any():
            raise MotionMillionError("sample index 包含越界 shard_id")
        if self.sequence_mode == "full":
            if ((self.sample_index["frames"] < 60) | (self.sample_index["frames"] > 300)).any():
                raise MotionMillionError("full sample index 帧数必须在 [60,300]")
            if (self.sample_index["window_index"] != 0).any():
                raise MotionMillionError("full 模式不接受分窗/重复采样索引")
            # 一个动作仅一条索引，保留现有 shard-aware 采样，不以长度增加权重。
            if len(self.sample_index) != sum(self._shard_record_counts):
                raise MotionMillionError("full sample index 必须覆盖每条完整动作一次")
            # 只排序一次索引，避免每个 shard 再扫描百万行；不读取/哈希大型分片。
            ordered = self.sample_index[np.argsort(self.sample_index["shard_id"], kind="stable")]
            offset = 0
            for shard_id, count in enumerate(self._shard_record_counts):
                rows = ordered[offset:offset + count]
                offset += count
                if not (rows["shard_id"] == shard_id).all():
                    raise MotionMillionError("full sample index shard记录数不一致")
                if not np.array_equal(np.sort(rows["record_index"]), np.arange(count)):
                    raise MotionMillionError("full sample index record_index 缺失、重复或越界")
        self.data_identity = {
            "schema_version": SCHEMA_VERSION,
            "split": self.split,
            "build_fingerprint": self.motion_manifest["build_fingerprint"],
            "release_motion_frames": release_motion_frames,
            "motion_manifest_sha256": sha256_file(self.motion_manifest_path),
            "embedding_manifest_sha256": sha256_file(self.embedding_manifest_path),
            "sample_index_sha256": sha256_file(index_path),
        }
        self.sampling_summary = {
            "raw_sequences": int(self.motion_manifest["record_count"]),
            "hours": float(self.sample_index["frames"].sum()) / 30.0 / 3600.0,
            "duration_aware_sampling": False,
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
        if len(motion_records) != self._shard_record_counts[shard_id] or len(embedding_records) != len(motion_records):
            raise MotionMillionError("实际 shard 记录数与 manifest 不一致")
        if record_index < 0 or record_index >= len(motion_records) or record_index >= len(embedding_records):
            raise MotionMillionError("sample index record_index 越界")
        motion = motion_records[record_index]
        embedding = embedding_records[record_index]
        validate_motion_record(motion)
        validate_embedding_record(embedding, caption_count=len(motion["captions"]))
        if str(motion["motion_id"]) != str(embedding.get("motion_id")):
            raise MotionMillionError("motion/embedding record motion_id 不一致")
        if motion["split"] != self.split or int(entry["frames"]) != len(motion["pose"]):
            raise MotionMillionError("sample index frames / record split 与实际动作不一致")

        pose = motion["pose"].float().clone()
        trans = motion["trans"].float().clone()
        frames = int(pose.shape[0])
        target = self.motion_frames
        if self.sequence_mode == "full":
            target = self.pad_to_frames
            start, valid_length = 0, frames
            pose = torch.cat([pose, pose[-1:].expand(target - frames, -1)], dim=0)
            trans = torch.cat([trans, trans[-1:].expand(target - frames, -1)], dim=0)
        elif frames > target:
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
        if self.caption_sampling == "random":
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
            "source_frames": frames,
            "sequence_mode": self.sequence_mode,
            "pad_to_frames": self.pad_to_frames,
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
                "source_frames",
                "sequence_mode",
                "pad_to_frames",
            )
        }
        core = {
            "body_pose": data["body_pose"],
            "betas": data["betas"],
            "global_orient": data["global_orient"],
            "transl": data["transl"],
            "data_name": data["data_name"],
        }
        if self.sequence_mode == "full":
            # 所有空间增强/相机轨迹/时间差分只看 F 帧；最后才使各字段 batchable。
            core = {key: value[:metadata["valid_length"]].clone() if torch.is_tensor(value) else value
                    for key, value in core.items()}
        result = super()._process_data(core, idx)
        if self.sequence_mode == "full":
            result = pad_full_sequence_fields(result, metadata["valid_length"], self.pad_to_frames)
        sequence_length = result["smpl_params_w"]["body_pose"].shape[0]
        result["length"] = int(metadata["valid_length"])
        result["valid_length"] = int(metadata["valid_length"])
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
                "source_frames": metadata["source_frames"],
                "valid_length": metadata["valid_length"],
                "sequence_mode": metadata["sequence_mode"],
                "pad_to_frames": metadata["pad_to_frames"],
                "text_index": metadata["text_index"],
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


def pad_full_sequence_fields(value, valid_length, pad_to_frames):
    """只用于 BaseDataset 输出：逐帧物理量末帧延拓，逐帧布尔 mask 补 False。

    在附加文本之前调用，所以即使 F=150 也不会误补齐 T5 token。末帧延拓保持旋转
    矩阵/6D 表示有效；派生速度只作为占位，不将 padding 算作真实观测。
    """
    if isinstance(value, dict):
        return {key: pad_full_sequence_fields(item, valid_length, pad_to_frames) for key, item in value.items()}
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == valid_length:
        tail_shape = (pad_to_frames - valid_length, *value.shape[1:])
        tail = value.new_zeros(tail_shape) if value.dtype == torch.bool else value[-1:].expand(tail_shape)
        return torch.cat([value, tail], dim=0)
    return value


__all__ = ["MotionMillionDataset"]
