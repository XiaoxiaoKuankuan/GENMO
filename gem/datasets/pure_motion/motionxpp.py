# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Motion-X++ 分片式 3D SMPL-X + semantic text 训练数据集。"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gem.utils.net_utils import get_valid_mask
from gem.utils.pylogger import Log
from tools.data.motionxpp.common import (
    MotionXppError,
    read_jsonl,
    safe_torch_load,
    validate_record,
)

from .base_dataset import BaseDataset
from .utils import interpolate_smpl_params, pad_data


class MotionXppDataset(BaseDataset):
    """按 manifest 和 worker 本地 LRU 延迟加载 Motion-X++ motion/T5 shards。"""

    def __init__(
        self,
        root: str | Path = "inputs/Motion-Xplusplus",
        manifest_path: str | Path = ("inputs/Motion-Xplusplus/genmo_support/manifests/train.jsonl"),
        embedding_manifest_path: str | Path = (
            "inputs/Motion-Xplusplus/t5_embeddings_v1_half/manifests/train.json"
        ),
        split: str = "train",
        motion_frames: int = 120,
        cam_augmentation: str = "static",
        condition_on_keypoints: bool = False,
        limit_size: int | None = None,
        shard_cache_size: int = 2,
        enable_speed_aug: bool = False,
        source_up_axis: str = "y",
        random_seed: int = 20260724,
        l_factor: float = 1.5,
        mode: str = "default",
    ) -> None:
        if motion_frames <= 0:
            raise ValueError("motion_frames must be positive")
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        if source_up_axis.lower() != "y":
            raise ValueError(
                "MotionXppDataset requires builder output already converted to AY/Y-up; "
                "set source_up_axis='y'"
            )
        if condition_on_keypoints:
            raise NotImplementedError(
                "Motion-X++ keypoint conditioning is disabled: the audited keypoint "
                "archives do not contain reliable image width/height and calibrated "
                "camera intrinsics/extrinsics. Rebuild a calibrated 2D contract before "
                "setting condition_on_keypoints=true; no synthetic K will be used."
            )
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.embedding_manifest_path = Path(embedding_manifest_path)
        self.split = split
        self.motion_frames = int(motion_frames)
        self.condition_on_keypoints = bool(condition_on_keypoints)
        self.shard_cache_size = int(shard_cache_size)
        self.enable_speed_aug = bool(enable_speed_aug)
        self.random_seed = int(random_seed)
        self.l_factor = float(l_factor)
        self.mode = mode
        self.dataset_name = "Motion-X++"
        self._motion_cache: OrderedDict[str, Any] = OrderedDict()
        self._embedding_cache: OrderedDict[str, Any] = OrderedDict()
        self._rng_pid: int | None = None
        self._rng: np.random.RandomState | None = None
        super().__init__(cam_augmentation, limit_size)

    def _load_dataset(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Motion-X++ motion manifest does not exist: {self.manifest_path}"
            )
        if not self.embedding_manifest_path.is_file():
            raise FileNotFoundError(
                f"Motion-X++ embedding manifest does not exist: {self.embedding_manifest_path}"
            )
        Log.info(f"[{self.dataset_name}] Loading manifests only ...")
        self.motion_root = self.manifest_path.parent.parent
        self.embedding_root = self.embedding_manifest_path.parent.parent
        self.motion_rows = read_jsonl(self.manifest_path)
        self.embedding_manifest = json.loads(
            self.embedding_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(self.embedding_manifest, dict):
            raise MotionXppError("Embedding manifest must be a JSON object")
        if self.embedding_manifest.get("split") != self.split:
            raise MotionXppError(
                f"Embedding manifest split={self.embedding_manifest.get('split')!r}, "
                f"dataset split={self.split!r}"
            )
        embedding_map = self.embedding_manifest.get("motion_to_shard")
        if not isinstance(embedding_map, dict):
            raise MotionXppError("Embedding manifest is missing motion_to_shard")
        self.embedding_map: dict[str, dict[str, Any]] = embedding_map
        ids = [str(row.get("motion_id", "")) for row in self.motion_rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise MotionXppError("Motion manifest has empty or duplicate motion_id values")
        missing = sorted(set(ids) - set(self.embedding_map))
        extra = sorted(set(self.embedding_map) - set(ids))
        if missing or extra:
            raise MotionXppError(
                "Motion/embedding manifest key mismatch: "
                f"missing embeddings={missing[:10]}, extra embeddings={extra[:10]}"
            )
        if not self.motion_rows:
            raise MotionXppError("Motion-X++ manifest contains no records")
        Log.info(
            f"[{self.dataset_name}] {len(self.motion_rows)} manifest records; "
            f"LRU cache={self.shard_cache_size}+{self.shard_cache_size} shards"
        )

    def _get_idx2meta(self) -> None:
        self.idx2meta: list[int] = []
        total_frames = 0
        for row_index, row in enumerate(self.motion_rows):
            frames = int(row.get("frames", 0))
            if frames < 25:
                continue
            total_frames += frames
            self.idx2meta.extend([row_index] * max(frames // self.motion_frames, 1))
        if not self.idx2meta:
            raise MotionXppError("No Motion-X++ records have at least 25 frames")
        Log.info(
            f"[{self.dataset_name}] {total_frames / 30 / 3600:.2f} hours -> "
            f"{len(self.idx2meta)} training samples"
        )

    def __getstate__(self) -> dict[str, Any]:
        """DataLoader worker pickle 时不复制已加载 shard 或 RNG 状态。"""
        state = self.__dict__.copy()
        state["_motion_cache"] = OrderedDict()
        state["_embedding_cache"] = OrderedDict()
        state["_rng_pid"] = None
        state["_rng"] = None
        return state

    def _get_rng(self) -> np.random.RandomState:
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            # PID 只用于让 worker 的 crop/caption 序列相互独立；基础 seed 可复现。
            self._rng = np.random.RandomState((self.random_seed + pid * 9973) % (2**32 - 1))
            self._rng_pid = pid
        return self._rng

    def _resolve(self, root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    def _cached_load(
        self,
        path: Path,
        cache: OrderedDict[str, Any],
    ) -> dict[str, Any]:
        key = str(path.resolve())
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        value = safe_torch_load(path)
        if not isinstance(value, dict):
            raise MotionXppError(f"Shard must contain a dict: {path}")
        cache[key] = value
        while len(cache) > self.shard_cache_size:
            cache.popitem(last=False)
        return value

    def _load_data(self, idx: int) -> dict[str, Any]:
        row_index = self.idx2meta[idx]
        row = self.motion_rows[row_index]
        motion_id = str(row["motion_id"])
        motion_path = self._resolve(self.motion_root, str(row["shard_path"]))
        motion_shard = self._cached_load(motion_path, self._motion_cache)
        record_key = str(row.get("record_key", motion_id))
        if record_key not in motion_shard:
            raise KeyError(f"{record_key!r} not found in {motion_path}")
        raw = motion_shard[record_key]
        validate_record(raw, motion_id)

        embedding_meta = self.embedding_map[motion_id]
        embedding_path = self._resolve(self.embedding_root, str(embedding_meta["shard_path"]))
        embedding_shard = self._cached_load(embedding_path, self._embedding_cache)
        embedding_key = str(embedding_meta.get("record_key", motion_id))
        if embedding_key not in embedding_shard:
            raise KeyError(f"{embedding_key!r} not found in {embedding_path}")
        all_embeddings = embedding_shard[embedding_key]
        text_data = raw["text_data"]
        if (
            not isinstance(all_embeddings, torch.Tensor)
            or all_embeddings.ndim != 3
            or tuple(all_embeddings.shape[1:]) != (50, 1024)
        ):
            raise MotionXppError(
                f"{motion_id}: embeddings must be [C,50,1024], "
                f"got {getattr(all_embeddings, 'shape', None)}"
            )
        if all_embeddings.shape[0] != len(text_data):
            raise MotionXppError(
                f"{motion_id}: caption/embedding count mismatch "
                f"{len(text_data)} != {all_embeddings.shape[0]}"
            )
        if not torch.isfinite(all_embeddings).all():
            raise MotionXppError(f"{motion_id}: embedding contains NaN or Inf")

        pose = raw["pose"].float()
        trans = raw["trans"].float()
        beta = raw["beta"].float()
        raw_len = pose.shape[0]
        if beta.ndim == 1:
            beta_frames = beta[:10].repeat(raw_len, 1)
        else:
            beta_frames = beta[:, :10]
        params = {
            "body_pose": pose[:, 3:66],
            "betas": beta_frames,
            "global_orient": pose[:, :3],
            "transl": trans,
        }
        rng = self._get_rng()
        target_len = self.motion_frames
        if self.enable_speed_aug:
            requested = rng.randint(
                max(2, int(target_len / self.l_factor)),
                max(3, int(target_len * self.l_factor) + 1),
            )
        else:
            requested = target_len
        crop_len = min(requested, raw_len)
        start = int(rng.randint(0, raw_len - crop_len + 1))
        params = {key: value[start : start + crop_len] for key, value in params.items()}
        if self.enable_speed_aug:
            params = interpolate_smpl_params(params, target_len)
            valid_length = target_len
        elif crop_len < target_len:
            params = pad_data(params, target_len)
            valid_length = crop_len
        else:
            valid_length = target_len
        text_index = int(rng.randint(0, len(text_data)))
        caption = str(text_data[text_index]["caption"])
        text_embed = all_embeddings[text_index].float().contiguous()
        if text_embed.shape != (50, 1024):
            raise MotionXppError(f"{motion_id}: selected text embedding must be [50,1024]")
        params.update(
            {
                "data_name": "motionxpp",
                "motion_id": motion_id,
                "text_index": text_index,
                "caption": caption,
                "text_embed": text_embed,
                "valid_length": valid_length,
                "source_subset": raw.get("source_subset", row.get("subset")),
            }
        )
        return params

    def _process_data(self, data: dict[str, Any], idx: int) -> dict[str, Any]:
        metadata = {
            "motion_id": data["motion_id"],
            "text_index": data["text_index"],
            "caption": data["caption"],
            "text_embed": data["text_embed"],
            "valid_length": int(data["valid_length"]),
            "source_subset": data["source_subset"],
        }
        core = {
            "body_pose": data["body_pose"],
            "betas": data["betas"],
            "global_orient": data["global_orient"],
            "transl": data["transl"],
            "data_name": data["data_name"],
        }
        result = super()._process_data(core, idx)
        length = result["smpl_params_w"]["body_pose"].shape[0]
        valid_length = metadata["valid_length"]
        result["length"] = valid_length
        result["caption"] = metadata["caption"]
        result["has_text"] = True
        result["text_embed"] = metadata["text_embed"]
        result["meta"].update(
            {
                "dataset_id": "motionxpp",
                "mid": metadata["motion_id"],
                "motion_id": metadata["motion_id"],
                "text_ind": metadata["text_index"],
                "source_subset": metadata["source_subset"],
                "mode": self.mode,
            }
        )
        result["mask"]["valid"] = get_valid_mask(length, valid_length)
        # SMPL-X 3D + semantic text。没有校准 2D 条件时绝不把零 kp 标成有效。
        result["mask"]["has_2d_mask"] = get_valid_mask(length, 0)
        result["mask"]["has_cam_mask"] = get_valid_mask(length, valid_length)
        result["mask"]["2d_only"] = False
        return result
