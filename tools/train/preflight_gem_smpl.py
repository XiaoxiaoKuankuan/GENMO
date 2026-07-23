#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Validate that a composed GEM-SMPL experiment is ready for server training."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_music_feature_tensor,
    validate_musicfeat_v2,
)

EXPECTED_TRAIN_DATASETS = (
    "amass_v11",
    "bedlam_v2",
    "h36m_v1",
    "3dpw_v1",
    "aistpp_train",
    "beat2_static_train",
    "humanml3d_static_train",
)
EXPECTED_VAL_DATASETS = (
    "emdb1_v1_fliptest",
    "emdb2_v1_fliptest",
    "3dpw_fliptest",
    "rich_all",
)
ORIGINAL_GEM_SMPL_SHA256 = "5131c329caaf4c26ac5cd84da9a2e6cdbff7f096fe2c97ce9bfb000dc553f1dd"


class PreflightError(RuntimeError):
    """Raised when a required training contract is not satisfied."""


def safe_torch_load(path: str | Path) -> Any:
    """Load a trusted local Torch artifact across PyTorch versions."""

    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compose_training_config(repo_root: Path, exp: str) -> DictConfig:
    """Compose the real training config through Hydra's config search path."""

    config_dir = (repo_root / "configs").resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory does not exist: {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"exp={exp}"])
    OmegaConf.resolve(cfg)
    return cfg


def validate_server_config(cfg: DictConfig) -> dict[str, Any]:
    """Validate the composed server experiment without hard-coding dataset discovery."""

    train_names = list(cfg.train_datasets.keys())
    val_names = list(cfg.test_datasets.keys())
    if train_names != list(EXPECTED_TRAIN_DATASETS):
        raise PreflightError(
            f"server train datasets are {train_names}; expected {list(EXPECTED_TRAIN_DATASETS)}"
        )
    if val_names != list(EXPECTED_VAL_DATASETS):
        raise PreflightError(
            f"server validation datasets are {val_names}; expected {list(EXPECTED_VAL_DATASETS)}"
        )
    yaml_text = OmegaConf.to_yaml(cfg)
    if "3dpw_occ_v1" in train_names or "metric_3dpw_occ" in yaml_text:
        raise PreflightError("gem_smpl_server must not include 3dpw_occ_v1 or metric_3dpw_occ")
    checks = {
        "regression_only": bool(cfg.network.model_cfg.regression_only),
        "pipeline_regression_only": bool(cfg.pipeline.args.regression_only),
        "encode_text": bool(cfg.network.model_cfg.denoiser.encode_text),
        "encoded_text_dim": int(cfg.network.model_cfg.denoiser.encoded_text_dim),
        "encoded_music_dim": int(cfg.pipeline.args.encoded_music_dim),
        "endecoder_feature_dim": int(cfg.endecoder.feat_dim),
        "text_encoder_load_llm": bool(cfg.model.model_cfg.text_encoder.load_llm),
        "text_encoder_version": str(cfg.model.model_cfg.text_encoder.llm_version),
        "train_batch_size": int(cfg.data.loader_opts.train.batch_size),
        "train_num_workers": int(cfg.data.loader_opts.train.num_workers),
        "precision": str(cfg.pl_trainer.precision),
        "max_steps": int(cfg.pl_trainer.max_steps),
        "gradient_clip_val": float(cfg.pl_trainer.gradient_clip_val),
    }
    expected = {
        "regression_only": False,
        "pipeline_regression_only": False,
        "encode_text": True,
        "encoded_text_dim": 1024,
        "encoded_music_dim": 35,
        "endecoder_feature_dim": 151,
        "text_encoder_load_llm": False,
        "text_encoder_version": "t5-3b",
        "train_batch_size": 4,
        "train_num_workers": 4,
        "precision": "16-mixed",
        "max_steps": 500000,
        "gradient_clip_val": 0.5,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise PreflightError(f"server config {key}={checks[key]!r}, expected {value!r}")
    return checks


def _require_paths(repo_root: Path, relative_paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for relative in relative_paths:
        path = repo_root / relative
        exists = path.exists()
        nonempty_directory = not path.is_dir() or any(path.iterdir())
        valid = exists and nonempty_directory
        results[relative] = {
            "path": str(path.resolve()),
            "exists": exists,
            "kind": "directory" if path.is_dir() else "file",
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "valid": valid,
        }
        if not valid:
            missing.append(relative)
    if missing:
        raise PreflightError(f"missing or empty required artifacts: {missing}")
    return results


def check_required_artifacts(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Check regression, validation and body-model artifacts required by the config."""

    return _require_paths(
        repo_root,
        (
            "inputs/AMASS/hmr4d_support/smplxpose_v2.pth",
            "inputs/BEDLAM/hmr4d_support/smplpose_v2.pth",
            "inputs/BEDLAM/hmr4d_support/mid_to_valid_range_all60.pt",
            "inputs/BEDLAM/hmr4d_support/mid_to_valid_range_maxspan60.pt",
            "inputs/BEDLAM/hmr4d_support/imgfeats/bedlam_all60",
            "inputs/BEDLAM/hmr4d_support/imgfeats/bedlam_maxspan60",
            "inputs/H36M/hmr4d_support/smplxpose_v1.pt",
            "inputs/H36M/hmr4d_support/vitfeat_h36m.pt",
            "inputs/3DPW/hmr4d_support/train_3dpw_gt_labels.pt",
            "inputs/3DPW/hmr4d_support/train_refit_smplx.pt",
            "inputs/3DPW/hmr4d_support/imgfeats/3dpw_train_smplx_refit",
            "inputs/EMDB/hmr4d_support",
            "inputs/RICH/hmr4d_support/rich_test_labels.pt",
            "inputs/RICH/hmr4d_support/rich_test_preproc.pt",
            "inputs/RICH/hmr4d_support/rich_test_vimo_preproc.pt",
            "gem/datasets/rich/resource/cam2params.pt",
            "inputs/3DPW/hmr4d_support/test_3dpw_gt_labels.pt",
            "inputs/checkpoints/body_models",
        ),
    )


def check_humanml3d_artifacts(
    motion_path: Path,
    embedding_path: Path,
    *,
    expected_motion_count: int = 23242,
) -> dict[str, Any]:
    """Validate HumanML3D motions and all precomputed T5 embedding contracts."""

    if not motion_path.is_file() or not embedding_path.is_file():
        raise PreflightError(
            f"HumanML3D artifact missing: motion={motion_path}, embedding={embedding_path}"
        )
    motions = safe_torch_load(motion_path)
    embeddings = safe_torch_load(embedding_path)
    if not isinstance(motions, dict) or not isinstance(embeddings, dict):
        raise PreflightError("HumanML3D motion and embedding artifacts must both be dicts")
    if len(motions) != expected_motion_count:
        raise PreflightError(
            f"HumanML3D motion count is {len(motions)}, expected {expected_motion_count}"
        )
    motion_keys, embedding_keys = set(motions), set(embeddings)
    if motion_keys != embedding_keys:
        raise PreflightError(
            "HumanML3D motion/embedding keys differ: "
            f"missing_embeddings={len(motion_keys - embedding_keys)}, "
            f"extra_embeddings={len(embedding_keys - motion_keys)}"
        )
    caption_count = 0
    dtype_counts: dict[str, int] = {}
    for motion_id, motion in motions.items():
        if not isinstance(motion, dict) or not isinstance(motion.get("text_data"), list):
            raise PreflightError(f"HumanML3D {motion_id}: missing text_data list")
        num_captions = len(motion["text_data"])
        value = embeddings[motion_id]
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise PreflightError(
                f"HumanML3D {motion_id}: embedding must be [N,50,1024], got "
                f"{getattr(value, 'shape', None)}"
            )
        if tuple(value.shape) != (num_captions, 50, 1024):
            raise PreflightError(
                f"HumanML3D {motion_id}: embedding shape {tuple(value.shape)} does not "
                f"match captions {(num_captions, 50, 1024)}"
            )
        if not value.is_floating_point():
            raise PreflightError(f"HumanML3D {motion_id}: embedding dtype must be floating")
        if not torch.isfinite(value).all():
            raise PreflightError(f"HumanML3D {motion_id}: embedding contains NaN or Inf")
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
        caption_count += num_captions
    result = {
        "motion_count": len(motions),
        "embedding_key_count": len(embeddings),
        "caption_count": caption_count,
        "embedding_dtype_counts": dtype_counts,
        "keys_equal": True,
        "all_embeddings_finite": True,
    }
    del motions, embeddings
    gc.collect()
    return result


def check_aist_artifacts(
    root: Path,
    *,
    expected_annot: int = 1020,
    expected_train: int = 980,
    expected_val: int = 20,
    expected_test: int = 20,
    expected_minitrain: int = 16,
) -> dict[str, Any]:
    """Validate official AIST++ split and aligned EDGE baseline35 contracts."""

    paths = {
        "annot": root / "annot_aist_30fps.pt",
        "train": root / "train.pt",
        "val": root / "val.pt",
        "test": root / "test.pt",
        "minitrain": root / "minitrain.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    music_root = root / "musicfeat_v2"
    if not music_root.is_dir():
        missing.append(str(music_root))
    if missing:
        raise PreflightError(f"AIST++ artifacts missing: {missing}")
    annot = safe_torch_load(paths["annot"])
    splits = {name: safe_torch_load(path) for name, path in paths.items() if name != "annot"}
    expected_counts = {
        "annot": expected_annot,
        "train": expected_train,
        "val": expected_val,
        "test": expected_test,
        "minitrain": expected_minitrain,
    }
    actual_counts = {"annot": len(annot), **{name: len(value) for name, value in splits.items()}}
    if actual_counts != expected_counts:
        raise PreflightError(
            f"AIST++ counts are {actual_counts}, expected {expected_counts}"
        )
    train, val, test = map(set, (splits["train"], splits["val"], splits["test"]))
    if train & val or train & test or val & test:
        raise PreflightError("AIST++ official train/val/test splits overlap")
    if train | val | test != set(annot):
        raise PreflightError("AIST++ split union does not equal annot keys")
    if not set(splits["minitrain"]) <= train:
        raise PreflightError("AIST++ minitrain is not a subset of train")
    feature_paths = list(music_root.glob("*_musicfeat_fps30.pt"))
    suffix = "_musicfeat_fps30.pt"
    feature_ids = {path.name.removesuffix(suffix) for path in feature_paths}
    if not set(annot) <= feature_ids:
        raise PreflightError(
            f"AIST++ is missing music features for {len(set(annot) - feature_ids)} official IDs"
        )
    for sequence, record in annot.items():
        feature_path = music_root / f"{sequence}{suffix}"
        features = load_music_feature_tensor(feature_path)
        validate_musicfeat_v2(features, feature_path)
        length = int(record["smpl_pose_global"].shape[0])
        if tuple(features.shape) != (length, 35):
            raise PreflightError(
                f"AIST++ {sequence}: music shape {tuple(features.shape)}, expected {(length, 35)}"
            )
    result = {
        **actual_counts,
        "music_feature_count": len(feature_ids),
        "extra_music_feature_count": len(feature_ids - set(annot)),
        "split_union_equals_annot": True,
        "all_music_features_valid": True,
    }
    del annot, splits
    gc.collect()
    return result


def check_beat2_artifacts(root: Path) -> dict[str, Any]:
    """Validate the BEAT2 index and every indexed motion/audio pair."""

    index_path = root / "all_splits.pth"
    if not index_path.is_file():
        raise PreflightError(f"BEAT2 index does not exist: {index_path}")
    payload = safe_torch_load(index_path)
    if not isinstance(payload, dict):
        raise PreflightError("BEAT2 all_splits.pth must contain a dict")
    counts: dict[str, int] = {}
    checked_pairs = 0
    for split in ("train", "val", "test", "minitrain"):
        items = payload.get(split)
        if not isinstance(items, list) or not items:
            raise PreflightError(f"BEAT2 {split} split must be a non-empty list")
        counts[split] = len(items)
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not {"video_id", "subset", "length"} <= set(item):
                raise PreflightError(f"BEAT2 {split}[{index}] has an invalid item contract")
            if not isinstance(item["video_id"], str) or not isinstance(item["subset"], str):
                raise PreflightError(f"BEAT2 {split}[{index}] video_id/subset must be strings")
            if int(item["length"]) <= 0:
                raise PreflightError(f"BEAT2 {split}[{index}] length must be positive")
            motion = root / item["subset"] / "smplxflame_30" / f"{item['video_id']}.npz"
            audio = root / item["subset"] / "wave16k" / f"{item['video_id']}.wav"
            if not motion.is_file() or not audio.is_file():
                raise PreflightError(
                    f"BEAT2 {split}[{index}] indexed pair missing: motion={motion}, audio={audio}"
                )
            checked_pairs += 1
    counts.update(
        {
            "checked_indexed_pairs": checked_pairs,
            "non_exact_beat2_train_count": counts["train"] != 1383,
        }
    )
    return counts


def _iter_numeric_values(value: Any, prefix: str = "root"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_numeric_values(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_numeric_values(child, f"{prefix}[{index}]")


def assert_all_finite(value: Any, context: str) -> None:
    """Recursively reject non-finite floating tensors and numeric arrays."""

    for path, tensor in _iter_numeric_values(value):
        if isinstance(tensor, torch.Tensor):
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise PreflightError(f"{context} {path} contains NaN or Inf")
        elif np.issubdtype(tensor.dtype, np.floating) and not np.isfinite(tensor).all():
            raise PreflightError(f"{context} {path} contains NaN or Inf")


def _mask_count(sample: dict[str, Any], name: str) -> int:
    value = sample["mask"].get(name)
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise PreflightError(f"mask.{name} must be a one-dimensional Tensor")
    return int(value.to(torch.bool).sum().item())


def _validate_smpl_fields(sample: dict[str, Any], dataset_name: str) -> None:
    for group in ("smpl_params_w", "smpl_params_c"):
        params = sample.get(group)
        if not isinstance(params, dict):
            raise PreflightError(f"{dataset_name}: missing {group} dict")
        for field, width in (("body_pose", 63), ("betas", 10), ("global_orient", 3), ("transl", 3)):
            tensor = params.get(field)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[-1] != width:
                raise PreflightError(
                    f"{dataset_name}: {group}.{field} must be [L,{width}], got "
                    f"{getattr(tensor, 'shape', None)}"
                )


def validate_dataset_sample(dataset_name: str, sample: Any) -> dict[str, Any]:
    """Validate one real dataset sample and its modality-specific condition contract."""

    if not isinstance(sample, dict):
        raise PreflightError(f"{dataset_name}: sample must be a dict")
    assert_all_finite(sample, dataset_name)
    raw_length = sample.get("length")
    length = int(raw_length.item()) if isinstance(raw_length, torch.Tensor) else int(raw_length)
    if length <= 0 or length > 120:
        raise PreflightError(f"{dataset_name}: sample length {length} is outside [1,120]")
    if not isinstance(sample.get("mask"), dict):
        raise PreflightError(f"{dataset_name}: sample mask must be a dict")
    for name in (
        "valid",
        "has_img_mask",
        "has_2d_mask",
        "has_cam_mask",
        "has_audio_mask",
        "has_music_mask",
    ):
        value = sample["mask"].get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) < length:
            raise PreflightError(f"{dataset_name}: mask.{name} must cover sample length {length}")
    _validate_smpl_fields(sample, dataset_name)

    conditions: dict[str, Any] = {}
    if dataset_name == "aistpp_train":
        music = sample.get("music_embed")
        if not isinstance(music, torch.Tensor) or music.shape[-1] != 35:
            raise PreflightError("aistpp_train: music_embed must end in 35")
        if _mask_count(sample, "has_music_mask") <= 0 or _mask_count(sample, "has_audio_mask") != 0:
            raise PreflightError("aistpp_train: invalid music/audio masks")
        if "text_embed" in sample and torch.count_nonzero(sample["text_embed"]) != 0:
            raise PreflightError("aistpp_train: text_embed must be absent or a zero placeholder")
        conditions = {"music_embed": list(music.shape), "text_embed_source": "collate_default"}
    elif dataset_name == "humanml3d_static_train":
        text = sample.get("text_embed")
        if not isinstance(text, torch.Tensor) or tuple(text.shape) != (50, 1024):
            raise PreflightError("humanml3d_static_train: text_embed must be [50,1024]")
        if not sample.get("caption") or not bool(sample.get("has_text")):
            raise PreflightError("humanml3d_static_train: caption/has_text is invalid")
        if _mask_count(sample, "has_music_mask") or _mask_count(sample, "has_audio_mask"):
            raise PreflightError("humanml3d_static_train: music/audio masks must be disabled")
        conditions = {"text_embed": list(text.shape), "caption": sample["caption"]}
    elif dataset_name == "beat2_static_train":
        audio = sample.get("audio_array")
        if int(sample.get("audio_fps", 0)) != 18000:
            raise PreflightError("beat2_static_train: audio_fps must be 18000")
        if not isinstance(audio, torch.Tensor) or audio.numel() < length * 600:
            raise PreflightError("beat2_static_train: audio does not cover the motion clip")
        if _mask_count(sample, "has_audio_mask") <= 0 or _mask_count(sample, "has_music_mask"):
            raise PreflightError("beat2_static_train: invalid audio/music masks")
        conditions = {"audio_samples": int(audio.numel()), "audio_fps": 18000}
    elif dataset_name == "amass_v11":
        if _mask_count(sample, "has_img_mask") or _mask_count(sample, "has_music_mask") or _mask_count(sample, "has_audio_mask"):
            raise PreflightError("amass_v11: image/music/audio conditions must be disabled")
        conditions = {"unconditioned_motion": True}
    elif dataset_name in {"bedlam_v2", "h36m_v1", "3dpw_v1"}:
        image = sample.get("f_imgseq")
        if not isinstance(image, torch.Tensor) or image.shape[-1] != 1024:
            raise PreflightError(f"{dataset_name}: f_imgseq must end in 1024")
        conditions = {
            "f_imgseq": list(image.shape),
            "has_img": _mask_count(sample, "has_img_mask"),
            "has_2d": _mask_count(sample, "has_2d_mask"),
            "has_cam": _mask_count(sample, "has_cam_mask"),
        }
    return {
        "length": length,
        "meta_data_name": sample.get("meta", {}).get("data_name"),
        "condition_checks": conditions,
        "all_finite": True,
    }


def validate_eval_sample(dataset_name: str, sample: Any) -> dict[str, Any]:
    """Validate finite validation data and metadata without rendering videos."""

    if not isinstance(sample, dict):
        raise PreflightError(f"{dataset_name}: validation sample must be a dict")
    assert_all_finite(sample, dataset_name)
    if not isinstance(sample.get("meta"), dict):
        raise PreflightError(f"{dataset_name}: validation sample has no metadata dict")
    return {
        "all_finite": True,
        "keys": sorted(sample.keys()),
        "meta_keys": sorted(sample["meta"].keys()),
    }


def inspect_datasets(
    dataset_configs: DictConfig,
    samples_per_dataset: int,
    *,
    validation: bool,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Instantiate each Hydra dataset independently and inspect deterministic samples."""

    lengths: dict[str, int] = {}
    results: dict[str, Any] = {}
    for name, dataset_cfg in dataset_configs.items():
        target = str(dataset_cfg.get("_target_", ""))
        print(f"[Dataset] {name}: {target}")
        dataset = instantiate(dataset_cfg)
        dataset_length = len(dataset)
        if dataset_length <= 0:
            raise PreflightError(f"{name}: dataset is empty")
        lengths[name] = dataset_length
        samples: list[dict[str, Any]] = []
        for index in range(min(samples_per_dataset, dataset_length)):
            random.seed(index)
            np.random.seed(index)
            torch.manual_seed(index)
            sample = dataset[index]
            samples.append(
                validate_eval_sample(name, sample)
                if validation
                else validate_dataset_sample(name, sample)
            )
        results[name] = {"target": target, "samples": samples}
        del dataset
        gc.collect()
    return lengths, results


def validate_mixed_batch(batch: Any, batch_size: int) -> dict[str, Any]:
    """Validate the collated, multimodal batch contract consumed by GEM."""

    if not isinstance(batch, dict) or int(batch.get("B", -1)) != batch_size:
        raise PreflightError(f"mixed batch B must be {batch_size}")
    assert_all_finite(batch, "mixed_batch")
    expected = {
        "text_embed": (batch_size, 50, 1024),
        "music_embed": (batch_size, 120, 35),
        "f_imgseq": (batch_size, 120, 1024),
    }
    shapes: dict[str, Any] = {"B": batch_size}
    for key, shape in expected.items():
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise PreflightError(
                f"mixed batch {key} shape is {getattr(value, 'shape', None)}, expected {shape}"
            )
        shapes[key] = list(value.shape)
    audio = batch.get("audio_array")
    if not isinstance(audio, torch.Tensor) or audio.shape[0] != batch_size or audio.numel() == 0:
        raise PreflightError("mixed batch audio_array is missing or empty")
    shapes["audio_array"] = list(audio.shape)
    length = batch.get("length")
    if not isinstance(length, torch.Tensor) or length.shape != (batch_size,):
        raise PreflightError("mixed batch length must be [B]")
    if not isinstance(batch.get("mask"), dict):
        raise PreflightError("mixed batch mask must be batchable dict")
    for key, value in batch["mask"].items():
        if isinstance(value, torch.Tensor) and value.shape[0] != batch_size:
            raise PreflightError(f"mixed batch mask.{key} has wrong batch dimension")
    meta = batch.get("meta")
    if not isinstance(meta, list) or len(meta) != batch_size:
        raise PreflightError("mixed batch meta must be a list of B dictionaries")
    sources = [item.get("data_name", item.get("dataset_id", "unknown")) for item in meta]
    shapes["sources"] = sources
    shapes["all_finite"] = True
    return shapes


def inspect_mixed_dataloader(
    cfg: DictConfig,
    batch_size: int,
    num_workers: int,
    samples_per_dataset: int,
) -> dict[str, Any]:
    """Instantiate the Hydra DataModule and inspect one real ConcatDataset batch."""

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    with open_dict(data_cfg):
        data_cfg.loader_opts.train.batch_size = batch_size
        data_cfg.loader_opts.train.num_workers = num_workers
        data_cfg.limit_each_trainset = max(batch_size, samples_per_dataset)
        if "val" in data_cfg.dataset_opts:
            del data_cfg.dataset_opts["val"]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    datamodule = instantiate(data_cfg, _recursive_=False)
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    result = validate_mixed_batch(batch, batch_size)
    del batch, loader, datamodule
    gc.collect()
    return result


def inspect_model(cfg: DictConfig) -> dict[str, Any]:
    """Instantiate the complete GEM model and validate its structural flags."""

    model = instantiate(cfg.model, _recursive_=False)
    endecoder_feature_dim = int(
        model.endecoder.FEATURE_DIMS[model.endecoder.encode_type]
    )
    result = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "regression_only": bool(model.pipeline.denoiser3d.regression_only),
        "encode_text": bool(cfg.network.model_cfg.denoiser.encode_text),
        "encoded_text_dim": int(cfg.network.model_cfg.denoiser.encoded_text_dim),
        "endecoder_feature_dim": endecoder_feature_dim,
    }
    if result["regression_only"] or not result["encode_text"]:
        raise PreflightError(f"instantiated model has invalid diffusion/text flags: {result}")
    if result["encoded_text_dim"] != 1024 or result["endecoder_feature_dim"] != 151:
        raise PreflightError(f"instantiated model dimensions are invalid: {result}")
    del model
    gc.collect()
    return result


def check_pretrained_checkpoint(path: Path) -> dict[str, Any]:
    """Load and inspect the optional full GEM-SMPL Lightning checkpoint."""

    if not path.is_file():
        raise PreflightError(f"pretrained checkpoint does not exist: {path}")
    checkpoint = safe_torch_load(path)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise PreflightError(f"pretrained checkpoint has no state_dict: {path}")
    keys = tuple(state.keys())
    has_music = any("music_embedder" in key for key in keys)
    has_text = any(
        marker in key for key in keys for marker in ("embed_text", "text_encoder_layers", "gate_cross_attn")
    )
    if not has_music or not has_text:
        raise PreflightError(
            f"pretrained checkpoint lacks full text/music weights: text={has_text}, music={has_music}"
        )
    result = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "state_dict_keys": len(state),
        "has_music_weights": has_music,
        "has_text_weights": has_text,
    }
    del checkpoint, state
    gc.collect()
    return result


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    """Create the preflight CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="gem_smpl_server")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    val_group = parser.add_mutually_exclusive_group()
    val_group.add_argument("--check-val", dest="check_val", action="store_true", default=True)
    val_group.add_argument("--skip-val", dest="check_val", action="store_false")
    parser.add_argument("--check-pretrained", action="store_true")
    parser.add_argument("--instantiate-model", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/preflight_gem_smpl/report.json")
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not args.repo_root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {args.repo_root}")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run every requested static, dataset, batch and model preflight check."""

    started = time.monotonic()
    repo_root = args.repo_root.resolve()
    cfg = compose_training_config(repo_root, args.exp)
    config_checks = validate_server_config(cfg)
    train_names = list(cfg.train_datasets.keys())
    val_names = list(cfg.test_datasets.keys()) if args.check_val else []
    report: dict[str, Any] = {
        "status": "running",
        "exp": args.exp,
        "repo_root": str(repo_root),
        "train_dataset_names": train_names,
        "train_dataset_lengths": {},
        "val_dataset_names": val_names,
        "val_dataset_lengths": {},
        "required_artifacts": {},
        "missing_artifacts": [],
        "optional_artifacts": {},
        "aist_counts": {},
        "humanml3d_counts": {},
        "beat2_counts": {},
        "beat2_missing_pair_policy": "indexed NPZ/WAV pairs are mandatory; train=1376 is accepted",
        "sample_validation_results": {},
        "mixed_batch_shapes": {},
        "model_instantiated": False,
        "model_validation": {},
        "pretrained_checkpoint": None,
        "config_checks": config_checks,
        "errors": [],
    }
    optional_occ = repo_root / "inputs/3DPW/hmr4d_support/imgfeats/3dpw_occ_train"
    report["optional_artifacts"] = {
        "optional_3dpw_occ_available": optional_occ.is_dir(),
        "path": str(optional_occ.resolve()),
        "required_by_server_config": False,
    }

    report["required_artifacts"] = check_required_artifacts(repo_root)
    report["humanml3d_counts"] = check_humanml3d_artifacts(
        repo_root / "inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth",
        repo_root / "inputs/HumanML3D_SMPL/t5_embeddings_v1_half/all_text_embed.pth",
    )
    report["aist_counts"] = check_aist_artifacts(repo_root / "inputs/AIST++")
    report["beat2_counts"] = check_beat2_artifacts(repo_root / "inputs/BEAT2")
    if args.check_pretrained:
        report["pretrained_checkpoint"] = check_pretrained_checkpoint(
            repo_root / "inputs/pretrained/gem_smpl.ckpt"
        )

    old_cwd = Path.cwd()
    try:
        # Dataset classes use repository-relative input roots.
        if old_cwd.resolve() != repo_root:
            import os

            os.chdir(repo_root)
        train_lengths, train_results = inspect_datasets(
            cfg.train_datasets, args.samples_per_dataset, validation=False
        )
        report["train_dataset_lengths"] = train_lengths
        report["sample_validation_results"]["train"] = train_results
        if args.check_val:
            val_lengths, val_results = inspect_datasets(
                cfg.test_datasets, args.samples_per_dataset, validation=True
            )
            report["val_dataset_lengths"] = val_lengths
            report["sample_validation_results"]["val"] = val_results
        report["mixed_batch_shapes"] = inspect_mixed_dataloader(
            cfg, args.batch_size, args.num_workers, args.samples_per_dataset
        )
        if args.instantiate_model:
            report["model_validation"] = inspect_model(cfg)
            report["model_instantiated"] = True
    finally:
        if Path.cwd() != old_cwd:
            import os

            os.chdir(old_cwd)
    report["status"] = "passed"
    report["elapsed_seconds"] = time.monotonic() - started
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point that always emits a JSON report."""

    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.monotonic()
    report: dict[str, Any]
    try:
        _validate_args(args)
        report = run_preflight(args)
    except Exception as exc:
        report = {
            "status": "failed",
            "exp": getattr(args, "exp", None),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_report(args.report, report)
        print(f"PREFLIGHT FAILED: {report['error']}", file=sys.stderr)
        return 1 if args.strict else 0
    _write_report(args.report, report)
    print("=" * 72)
    print(f"PREFLIGHT PASSED: {args.exp}")
    print(f"Train datasets: {report['train_dataset_lengths']}")
    print(f"Validation datasets: {report['val_dataset_lengths']}")
    print(f"Mixed batch sources: {report['mixed_batch_shapes'].get('sources', [])}")
    print(f"Model instantiated: {report['model_instantiated']}")
    print(f"Report: {args.report.resolve()}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
