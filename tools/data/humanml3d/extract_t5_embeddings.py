#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Extract T5-3B token embeddings for GENMO HumanML3D captions."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.network.utils import encode_text_batch  # noqa: E402

DEFAULT_INPUT = Path(
    "inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth"
)
DEFAULT_OUTPUT = Path(
    "inputs/HumanML3D_SMPL/t5_embeddings_v1_half/all_text_embed.pth"
)
DEFAULT_SHARD_DIR = Path("inputs/HumanML3D_SMPL/t5_embeddings_v1_half/shards")
DEFAULT_REPORT_DIR = Path("outputs/humanml3d_t5_report")

MAX_TEXT_LEN = 50
HIDDEN_DIM = 1024
OUTPUT_DTYPE = torch.float16
OUTPUT_DTYPE_NAME = "float16"
BYTES_PER_CAPTION = MAX_TEXT_LEN * HIDDEN_DIM * 2
MANIFEST_VERSION = 1


class T5EmbeddingBuildError(RuntimeError):
    """Raised when embedding extraction cannot satisfy the training contract."""


@dataclass
class BuildReports:
    """Mutable report payloads populated during extraction."""

    caption_counts: list[dict[str, Any]] = field(default_factory=list)
    invalid_records: list[dict[str, Any]] = field(default_factory=list)
    shard_rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)


def safe_torch_load(path: str | Path) -> Any:
    """Load trusted local Torch data across versions with optional weights_only."""
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_caption_metadata(
    input_path: str | Path,
    *,
    limit: int | None = None,
    strict: bool = False,
    keep_source_loaded: bool = False,
) -> tuple[dict[str, list[str]], dict[str, Any], list[dict[str, Any]], Any | None]:
    """Load motion records, copy ordered raw captions, then release large tensors."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"HumanML3D motion file does not exist: {input_path}")
    source_dataset = safe_torch_load(input_path)
    if not isinstance(source_dataset, dict):
        raise ValueError("HumanML3D motion artifact must contain a dict")
    invalid: list[dict[str, Any]] = []
    valid_ids: list[str] = []
    for motion_id in source_dataset:
        if not isinstance(motion_id, str) or not motion_id:
            issue = "motion key must be a non-empty string"
            invalid.append({"motion_id": repr(motion_id), "error": issue})
            if strict:
                del source_dataset
                gc.collect()
                raise T5EmbeddingBuildError(
                    f"Invalid record key {motion_id!r}: {issue}"
                )
            continue
        valid_ids.append(motion_id)

    sorted_ids = sorted(valid_ids)
    if limit is not None:
        sorted_ids = sorted_ids[:limit]

    captions_by_id: dict[str, list[str]] = {}
    for motion_id in sorted_ids:
        issue: str | None = None
        record = source_dataset[motion_id]
        if not isinstance(record, dict):
            issue = "motion record must be a dict"
        elif "text_data" not in record:
            issue = "motion record is missing text_data"
        elif not isinstance(record["text_data"], list) or not record["text_data"]:
            issue = "text_data must be a non-empty list"

        captions: list[str] = []
        if issue is None:
            for text_index, text_data in enumerate(record["text_data"]):
                if not isinstance(text_data, dict) or "caption" not in text_data:
                    issue = f"text_data[{text_index}] must be a dict containing caption"
                    break
                caption = text_data["caption"]
                if not isinstance(caption, str) or not caption.strip():
                    issue = f"text_data[{text_index}].caption must be a non-empty string"
                    break
                # Preserve the exact original string, including duplicates and ordering.
                captions.append(caption)
        if issue is not None:
            invalid.append({"motion_id": str(motion_id), "error": issue})
            if strict:
                del source_dataset
                gc.collect()
                raise T5EmbeddingBuildError(f"Invalid record {motion_id!r}: {issue}")
            continue
        captions_by_id[motion_id] = captions

    counts = [len(captions) for captions in captions_by_id.values()]
    stats = {
        "source_motion_record_count": len(source_dataset),
        "motion_record_count": len(captions_by_id),
        "total_caption_count": sum(counts),
        "min_captions_per_record": min(counts) if counts else 0,
        "max_captions_per_record": max(counts) if counts else 0,
        "mean_captions_per_record": sum(counts) / len(counts) if counts else 0.0,
        "subclip_record_count": sum("__seg_" in key for key in captions_by_id),
        "mirrored_record_count": sum(key.startswith("M") for key in captions_by_id),
        "limit": limit,
    }
    retained_source = source_dataset if keep_source_loaded else None
    if not keep_source_loaded:
        del source_dataset
        gc.collect()
    return captions_by_id, stats, invalid, retained_source


def compute_fingerprint(
    captions_by_id: dict[str, list[str]], model_name_or_path: str
) -> str:
    """Compute a length-delimited deterministic caption/model contract fingerprint."""
    digest = hashlib.sha256()

    def update(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)

    for motion_id, captions in captions_by_id.items():
        update(motion_id)
        digest.update(len(captions).to_bytes(8, byteorder="big", signed=False))
        for caption in captions:
            update(caption)
    update(model_name_or_path)
    update(f"max_text_len={MAX_TEXT_LEN}")
    update(f"hidden_dim={HIDDEN_DIM}")
    update(f"output_dtype={OUTPUT_DTYPE_NAME}")
    return digest.hexdigest()


def estimate_storage(total_caption_count: int, output_path: Path) -> dict[str, int | float]:
    """Estimate FP16 payload, peak shard/final storage and current free space."""
    payload = total_caption_count * BYTES_PER_CAPTION
    peak = int(payload * 2.2)
    ancestor = output_path.expanduser().resolve().parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    return {
        "estimated_payload_bytes": payload,
        "estimated_payload_gib": payload / (1024**3),
        "estimated_peak_disk_bytes": peak,
        "estimated_peak_disk_gib": peak / (1024**3),
        "available_disk_bytes": free,
        "available_disk_gib": free / (1024**3),
    }


def flatten_captions(
    captions_by_id: dict[str, list[str]], motion_ids: Sequence[str]
) -> tuple[list[str], list[tuple[str, int]]]:
    """Flatten captions without deduplication and retain exact owner/index mapping."""
    flat: list[str] = []
    owners: list[tuple[str, int]] = []
    for motion_id in motion_ids:
        for text_index, caption in enumerate(captions_by_id[motion_id]):
            flat.append(caption)
            owners.append((motion_id, text_index))
    return flat, owners


def restore_owner_embeddings(
    flat_embeddings: torch.Tensor,
    owners: Sequence[tuple[str, int]],
    captions_by_id: dict[str, list[str]],
    motion_ids: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Restore flat batch output to ordered per-motion token embedding tensors."""
    if flat_embeddings.ndim != 3 or flat_embeddings.shape[1:] != (
        MAX_TEXT_LEN,
        HIDDEN_DIM,
    ):
        raise ValueError(
            f"flat embeddings must be [C,{MAX_TEXT_LEN},{HIDDEN_DIM}], "
            f"got {tuple(flat_embeddings.shape)}"
        )
    if len(owners) != flat_embeddings.shape[0]:
        raise ValueError("owner count does not match flat embedding count")
    positions = {motion_id: [] for motion_id in motion_ids}
    for flat_index, (motion_id, text_index) in enumerate(owners):
        if motion_id not in positions:
            raise ValueError(f"owner references unexpected motion ID: {motion_id}")
        if text_index != len(positions[motion_id]):
            raise ValueError(f"owner ordering is invalid for {motion_id}")
        positions[motion_id].append(flat_index)
    output: dict[str, torch.Tensor] = {}
    for motion_id in motion_ids:
        expected = len(captions_by_id[motion_id])
        indices = positions[motion_id]
        if len(indices) != expected:
            raise ValueError(
                f"caption count mismatch for {motion_id}: owners={len(indices)}, expected={expected}"
            )
        value = flat_embeddings[indices].detach().cpu().to(OUTPUT_DTYPE).contiguous().clone()
        output[motion_id] = value
    return output


def validate_embedding_tensor(
    value: Any, expected_captions: int, motion_id: str
) -> None:
    """Validate one final/shard embedding tensor."""
    expected_shape = (expected_captions, MAX_TEXT_LEN, HIDDEN_DIM)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{motion_id}: embedding shape must be {expected_shape}, "
            f"got {getattr(value, 'shape', None)}"
        )
    if value.dtype != OUTPUT_DTYPE or value.device.type != "cpu":
        raise ValueError(f"{motion_id}: embedding must be CPU {OUTPUT_DTYPE_NAME}")
    if not value.is_contiguous():
        raise ValueError(f"{motion_id}: embedding must be contiguous")
    if not torch.isfinite(value).all():
        raise ValueError(f"{motion_id}: embedding contains NaN or Inf")


def validate_embedding_dict(
    value: Any,
    captions_by_id: dict[str, list[str]],
    expected_ids: Sequence[str] | None = None,
) -> tuple[int, int, int]:
    """Validate exact keys, per-record caption counts, shapes, dtype and finiteness."""
    if not isinstance(value, dict):
        raise ValueError("embedding artifact must contain a dict")
    ids = list(captions_by_id) if expected_ids is None else list(expected_ids)
    if set(value) != set(ids):
        missing = sorted(set(ids) - set(value))
        extra = sorted(set(value) - set(ids))
        raise ValueError(
            f"embedding key mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    caption_count = 0
    elements = 0
    for motion_id in ids:
        expected = len(captions_by_id[motion_id])
        validate_embedding_tensor(value[motion_id], expected, motion_id)
        caption_count += expected
        elements += value[motion_id].numel()
    return len(ids), caption_count, elements


def shard_layout(
    motion_ids: Sequence[str], motions_per_shard: int, shard_dir: Path
) -> list[dict[str, Any]]:
    """Build deterministic shard ranges over sorted motion IDs."""
    if motions_per_shard <= 0:
        raise ValueError("motions_per_shard must be positive")
    layout: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(motion_ids), motions_per_shard)):
        ids = list(motion_ids[start : start + motions_per_shard])
        layout.append(
            {
                "shard_index": shard_index,
                "start_motion_index": start,
                "end_motion_index": start + len(ids),
                "motion_ids": ids,
                "motion_count": len(ids),
                "output_file": str((shard_dir / f"shard_{shard_index:05d}.pth").resolve()),
            }
        )
    return layout


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _manifest_compatible(manifest: dict[str, Any], expected: dict[str, Any]) -> bool:
    fields = (
        "fingerprint",
        "model_name_or_path",
        "model_dtype",
        "output_dtype",
        "max_text_len",
        "hidden_dim",
        "motion_record_count",
        "total_caption_count",
        "motions_per_shard",
    )
    if not all(manifest.get(field) == expected.get(field) for field in fields):
        return False
    existing_shards = manifest.get("shards")
    expected_shards = expected.get("shards")
    if not isinstance(existing_shards, list) or len(existing_shards) != len(expected_shards):
        return False
    layout_fields = (
        "shard_index",
        "start_motion_index",
        "end_motion_index",
        "motion_ids",
        "motion_count",
        "output_file",
    )
    return all(
        all(current.get(field) == planned.get(field) for field in layout_fields)
        for current, planned in zip(existing_shards, expected_shards, strict=True)
    )


def make_manifest(
    captions_by_id: dict[str, list[str]],
    layout: list[dict[str, Any]],
    *,
    fingerprint: str,
    model_name_or_path: str,
    model_dtype: str,
    motions_per_shard: int,
) -> dict[str, Any]:
    """Create a pending shard manifest with all compatibility fields."""
    shards = []
    for item in layout:
        metadata = dict(item)
        metadata.update(
            {
                "caption_count": sum(len(captions_by_id[key]) for key in item["motion_ids"]),
                "output_size_bytes": 0,
                "status": "pending",
                "fingerprint": fingerprint,
                "model_name_or_path": model_name_or_path,
                "model_dtype": model_dtype,
                "output_dtype": OUTPUT_DTYPE_NAME,
                "max_text_len": MAX_TEXT_LEN,
                "hidden_dim": HIDDEN_DIM,
                "padding_zero_check_max_error": 0.0,
            }
        )
        shards.append(metadata)
    return {
        "manifest_version": MANIFEST_VERSION,
        "fingerprint": fingerprint,
        "model_name_or_path": model_name_or_path,
        "model_dtype": model_dtype,
        "output_dtype": OUTPUT_DTYPE_NAME,
        "max_text_len": MAX_TEXT_LEN,
        "hidden_dim": HIDDEN_DIM,
        "motion_record_count": len(captions_by_id),
        "total_caption_count": sum(len(value) for value in captions_by_id.values()),
        "motions_per_shard": motions_per_shard,
        "shards_cleaned": False,
        "shards": shards,
    }


def prepare_manifest(
    shard_dir: Path,
    expected_manifest: dict[str, Any],
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[dict[str, Any], bool]:
    """Load a compatible resume manifest or initialize clean shard state."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_dir / "manifest.json"
    existing_shards = sorted(shard_dir.glob("shard_*.pth"))
    existing_temps = sorted(shard_dir.glob("shard_*.pth.tmp"))
    existing_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_manifest = loaded
        except (OSError, json.JSONDecodeError):
            existing_manifest = None

    if resume and existing_manifest is not None:
        if _manifest_compatible(existing_manifest, expected_manifest):
            return existing_manifest, True
        if not overwrite:
            raise T5EmbeddingBuildError(
                "Existing shard manifest fingerprint/configuration is incompatible. "
                "Use --overwrite to discard incompatible shards."
            )
    elif resume and (existing_shards or existing_temps or manifest_path.exists()):
        if not overwrite:
            raise T5EmbeddingBuildError(
                "Existing shards have no valid compatible manifest; use --overwrite"
            )
    elif not resume and (existing_shards or existing_temps or manifest_path.exists()):
        if not overwrite:
            raise FileExistsError(
                f"Shard directory already contains extraction state: {shard_dir}; "
                "use --resume or --overwrite"
            )

    for path in [*existing_shards, *existing_temps]:
        path.unlink(missing_ok=True)
    if manifest_path.exists():
        manifest_path.unlink()
    _atomic_write_json(manifest_path, expected_manifest)
    return expected_manifest, False


def validate_shard_file(
    path: Path,
    captions_by_id: dict[str, list[str]],
    expected_ids: Sequence[str],
) -> tuple[int, int, int]:
    """Reload and fully validate a shard file."""
    shard = safe_torch_load(path)
    return validate_embedding_dict(shard, captions_by_id, expected_ids)


def write_shard_tmp(shard: dict[str, torch.Tensor], final_path: Path) -> Path:
    """Write a shard temporary file without publishing it."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(final_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(shard, temporary)
    return temporary


def publish_validated_shard(
    temporary_path: Path,
    final_path: Path,
    captions_by_id: dict[str, list[str]],
    expected_ids: Sequence[str],
) -> int:
    """Reload/validate a temporary shard and atomically publish it."""
    try:
        validate_shard_file(temporary_path, captions_by_id, expected_ids)
        os.replace(temporary_path, final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return final_path.stat().st_size


def _padding_error(
    encoded_text: torch.Tensor,
    raw_text: Sequence[str],
    tokenizer: Any,
    device: str,
) -> float:
    tokenize = getattr(tokenizer, "batch_encode_plus", None)
    if tokenize is None:
        tokenize = tokenizer
    encoded = tokenize(
        list(raw_text),
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_TEXT_LEN,
        truncation=True,
    )
    mask = encoded.attention_mask[:, :MAX_TEXT_LEN].to(device)
    padding = mask == 0
    if not padding.any():
        return 0.0
    values = encoded_text.masked_select(padding.unsqueeze(-1).expand_as(encoded_text))
    return float(values.abs().max().item()) if values.numel() else 0.0


def encode_shard(
    captions_by_id: dict[str, list[str]],
    motion_ids: Sequence[str],
    *,
    text_encoder: Any,
    tokenizer: Any,
    device: str,
    batch_size: int,
    strict: bool,
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Encode all captions in one shard and restore per-motion ordering."""
    flat_captions, owners = flatten_captions(captions_by_id, motion_ids)
    batches: list[torch.Tensor] = []
    max_padding_error = 0.0
    for batch_index, start in enumerate(range(0, len(flat_captions), batch_size)):
        batch_captions = flat_captions[start : start + batch_size]
        encoded = encode_text_batch(
            raw_text=batch_captions,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            device=device,
        )
        expected_shape = (len(batch_captions), MAX_TEXT_LEN, HIDDEN_DIM)
        if encoded.ndim != 3 or tuple(encoded.shape) != expected_shape:
            raise ValueError(
                f"T5 output shape must be {expected_shape}, got {tuple(encoded.shape)}"
            )
        if not torch.isfinite(encoded).all():
            raise ValueError(f"T5 output batch {batch_index} contains NaN or Inf")
        if batch_index < 10 or batch_index % 100 == 0:
            error = _padding_error(encoded, batch_captions, tokenizer, device)
            max_padding_error = max(max_padding_error, error)
            if strict and error > 1e-6:
                raise ValueError(
                    f"padding positions are non-zero in batch {batch_index}: max={error:.3e}"
                )
        cpu_half = encoded.detach().cpu().to(OUTPUT_DTYPE).contiguous()
        if not torch.isfinite(cpu_half).all():
            raise ValueError(f"FP16 conversion produced NaN or Inf in batch {batch_index}")
        batches.append(cpu_half)
        del encoded, cpu_half
    flat_embeddings = torch.cat(batches, dim=0).contiguous()
    del batches
    shard = restore_owner_embeddings(
        flat_embeddings, owners, captions_by_id, motion_ids
    )
    del flat_embeddings
    validate_embedding_dict(shard, captions_by_id, motion_ids)
    return shard, len(flat_captions), max_padding_error


def load_t5_model(
    model_name_or_path: str,
    *,
    cache_dir: Path | None,
    local_files_only: bool,
    device: str,
    model_dtype: str,
) -> tuple[Any, Any]:
    """Load and freeze T5 with the current GENMO tokenizer/model classes."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} was requested but CUDA is unavailable")
    from transformers import T5EncoderModel, T5Tokenizer

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, **kwargs)
        model = T5EncoderModel.from_pretrained(model_name_or_path, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load T5 encoder {model_name_or_path!r}. "
            "Provide a valid local directory/cache or fix Hugging Face access. "
            f"Original error: {exc}"
        ) from exc
    if int(model.config.d_model) != HIDDEN_DIM:
        raise RuntimeError(
            f"T5 hidden dimension is {model.config.d_model}, expected {HIDDEN_DIM}; "
            "do not use a smaller/different T5 model"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[model_dtype]
    model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        if not fieldnames:
            return
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    report_dir: Path,
    reports: BuildReports,
    manifest: dict[str, Any] | None,
) -> None:
    """Write extraction reports outside the final embedding dictionary."""
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(report_dir / "build_summary.json", reports.summary)
    _atomic_write_json(report_dir / "manifest_snapshot.json", manifest or {})
    _save_csv(report_dir / "motion_caption_counts.csv", reports.caption_counts)
    _atomic_write_json(report_dir / "invalid_records.json", reports.invalid_records)
    _save_csv(report_dir / "shard_summary.csv", reports.shard_rows)
    _atomic_write_json(report_dir / "validation_summary.json", reports.validation)


def finalize_shards(
    captions_by_id: dict[str, list[str]],
    manifest: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Validate all shards, merge exact keys, and atomically publish final output."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite final embedding output: {output_path}")
    all_embeddings: dict[str, torch.Tensor] = {}
    for shard_metadata in sorted(manifest["shards"], key=lambda item: item["shard_index"]):
        shard_path = Path(shard_metadata["output_file"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"completed shard file is missing: {shard_path}")
        expected_ids = shard_metadata["motion_ids"]
        validate_shard_file(shard_path, captions_by_id, expected_ids)
        shard = safe_torch_load(shard_path)
        duplicate = set(all_embeddings) & set(shard)
        if duplicate:
            raise ValueError(f"duplicate motion IDs across shards: {sorted(duplicate)[:10]}")
        for motion_id in expected_ids:
            all_embeddings[motion_id] = shard[motion_id]
        del shard
    records, captions, elements = validate_embedding_dict(all_embeddings, captions_by_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(all_embeddings, temporary)
        del all_embeddings
        gc.collect()
        reloaded = safe_torch_load(temporary)
        records, captions, elements = validate_embedding_dict(reloaded, captions_by_id)
        del reloaded
        gc.collect()
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "motion_record_count": records,
        "total_caption_count": captions,
        "total_embedding_elements": elements,
        "output_size_bytes": output_path.stat().st_size,
        "key_set_exact": True,
        "all_finite": True,
        "output_dtype": OUTPUT_DTYPE_NAME,
        "output_device": "cpu",
    }


def cleanup_shards(shard_dir: Path, manifest: dict[str, Any]) -> None:
    """Delete only validated shard PTH files after final output validation."""
    for shard_metadata in manifest["shards"]:
        Path(shard_metadata["output_file"]).unlink(missing_ok=True)
        shard_metadata["status"] = "cleaned_after_finalization"
    manifest["shards_cleaned"] = True
    _atomic_write_json(shard_dir / "manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    """Create the HumanML3D T5 extraction CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-name-or-path", default="t5-3b")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--motions-per-shard", type=int, default=256)
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cleanup-shards", action="store_true")
    parser.add_argument("--limit", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--estimate-only", action="store_true")
    mode.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--keep-source-loaded", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"HumanML3D input does not exist: {args.input}")
    if args.batch_size <= 0 or args.motions_per_shard <= 0:
        raise ValueError("--batch-size and --motions-per-shard must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.finalize_only and args.cleanup_shards and not args.overwrite and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite final output: {args.output}")
    if not args.estimate_only and args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite final output: {args.output}")
    if (
        not args.estimate_only
        and not args.finalize_only
        and args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(f"CUDA device {args.device!r} was requested but CUDA is unavailable")


def _print_estimate(stats: dict[str, Any], estimate: dict[str, Any]) -> None:
    print(f"motion records: {stats['motion_record_count']}")
    print(f"total captions: {stats['total_caption_count']}")
    print(f"estimated embedding payload bytes: {estimate['estimated_payload_bytes']}")
    print(f"estimated embedding payload GiB: {estimate['estimated_payload_gib']:.3f}")
    print(f"available output disk bytes: {estimate['available_disk_bytes']}")
    print(f"available output disk GiB: {estimate['available_disk_gib']:.3f}")
    print(f"estimated shard + final tmp peak bytes: {estimate['estimated_peak_disk_bytes']}")
    print(f"estimated shard + final tmp peak GiB: {estimate['estimated_peak_disk_gib']:.3f}")


def main(argv: list[str] | None = None) -> int:
    """Extract, resume, estimate, or finalize HumanML3D T5 embeddings."""
    started = time.monotonic()
    args = build_parser().parse_args(argv)
    _validate_args(args)
    reports = BuildReports()
    manifest: dict[str, Any] | None = None
    retained_source: Any | None = None
    try:
        captions_by_id, stats, invalid, retained_source = extract_caption_metadata(
            args.input,
            limit=args.limit,
            strict=args.strict,
            keep_source_loaded=args.keep_source_loaded,
        )
        reports.invalid_records.extend(invalid)
        if invalid:
            raise T5EmbeddingBuildError(
                f"Found {len(invalid)} invalid selected motion records; "
                "no motion may be silently lost"
            )
        reports.caption_counts.extend(
            {
                "motion_id": motion_id,
                "caption_count": len(captions),
                "mirrored": motion_id.startswith("M"),
                "subclip": "__seg_" in motion_id,
            }
            for motion_id, captions in captions_by_id.items()
        )
        fingerprint = compute_fingerprint(captions_by_id, args.model_name_or_path)
        estimate = estimate_storage(stats["total_caption_count"], args.output)
        motion_ids = list(captions_by_id)
        layout = shard_layout(motion_ids, args.motions_per_shard, args.shard_dir)
        reports.summary.update(
            {
                "status": "starting",
                "input_path": str(args.input.resolve()),
                "output_path": str(args.output.resolve()),
                "model_name_or_path": args.model_name_or_path,
                "model_dtype": args.model_dtype,
                "output_dtype": OUTPUT_DTYPE_NAME,
                "max_text_len": MAX_TEXT_LEN,
                "hidden_dim": HIDDEN_DIM,
                **stats,
                "shard_count": len(layout),
                "completed_shard_count": 0,
                "resumed_shard_count": 0,
                "encoded_caption_count": 0,
                "total_embedding_elements": stats["total_caption_count"]
                * MAX_TEXT_LEN
                * HIDDEN_DIM,
                **estimate,
                "output_size_bytes": 0,
                "elapsed_seconds": 0.0,
                "device": args.device,
                "batch_size": args.batch_size,
                "motions_per_shard": args.motions_per_shard,
                "fingerprint": fingerprint,
                "padding_zero_check_max_error": 0.0,
                "limit": args.limit,
            }
        )
        _print_estimate(stats, estimate)
        if args.estimate_only:
            reports.summary["status"] = "estimate_only_complete"
            reports.summary["elapsed_seconds"] = time.monotonic() - started
            write_reports(args.report_dir, reports, None)
            return 0
        if estimate["available_disk_bytes"] < estimate["estimated_peak_disk_bytes"]:
            raise RuntimeError(
                "Insufficient disk space for shards plus final temporary output: "
                f"need approximately {estimate['estimated_peak_disk_gib']:.3f} GiB, "
                f"available {estimate['available_disk_gib']:.3f} GiB"
            )

        expected_manifest = make_manifest(
            captions_by_id,
            layout,
            fingerprint=fingerprint,
            model_name_or_path=args.model_name_or_path,
            model_dtype=args.model_dtype,
            motions_per_shard=args.motions_per_shard,
        )
        manifest, resumed_manifest = prepare_manifest(
            args.shard_dir,
            expected_manifest,
            resume=args.resume or args.finalize_only,
            overwrite=args.overwrite,
        )

        if not args.finalize_only:
            text_encoder, tokenizer = load_t5_model(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
                device=args.device,
                model_dtype=args.model_dtype,
            )
            for shard_index, shard_metadata in enumerate(manifest["shards"]):
                expected_ids = shard_metadata["motion_ids"]
                final_path = Path(shard_metadata["output_file"])
                resumed = False
                if resumed_manifest and shard_metadata.get("status") == "complete":
                    try:
                        validate_shard_file(final_path, captions_by_id, expected_ids)
                        resumed = True
                    except Exception as exc:
                        print(
                            f"[Shard {shard_index:05d}] existing shard invalid; regenerating: {exc}"
                        )
                        final_path.unlink(missing_ok=True)
                if resumed:
                    reports.summary["resumed_shard_count"] += 1
                    reports.summary["completed_shard_count"] += 1
                    reports.shard_rows.append({**shard_metadata, "resumed": True})
                    print(f"[Shard {shard_index + 1}/{len(layout)}] resumed {final_path.name}")
                    continue

                shard_output, encoded_count, padding_error = encode_shard(
                    captions_by_id,
                    expected_ids,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    device=args.device,
                    batch_size=args.batch_size,
                    strict=args.strict,
                )
                temporary = write_shard_tmp(shard_output, final_path)
                del shard_output
                gc.collect()
                output_size = publish_validated_shard(
                    temporary, final_path, captions_by_id, expected_ids
                )
                shard_metadata.update(
                    {
                        "caption_count": encoded_count,
                        "output_size_bytes": output_size,
                        "status": "complete",
                        "padding_zero_check_max_error": padding_error,
                    }
                )
                reports.summary["encoded_caption_count"] += encoded_count
                reports.summary["completed_shard_count"] += 1
                reports.summary["padding_zero_check_max_error"] = max(
                    reports.summary["padding_zero_check_max_error"], padding_error
                )
                reports.shard_rows.append({**shard_metadata, "resumed": False})
                _atomic_write_json(args.shard_dir / "manifest.json", manifest)
                print(
                    f"[Shard {shard_index + 1}/{len(layout)}] encoded {encoded_count} captions "
                    f"-> {final_path.name}"
                )
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            del text_encoder, tokenizer
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        else:
            for shard_metadata in manifest["shards"]:
                validate_shard_file(
                    Path(shard_metadata["output_file"]),
                    captions_by_id,
                    shard_metadata["motion_ids"],
                )
                reports.summary["completed_shard_count"] += 1
                reports.summary["resumed_shard_count"] += 1
                reports.shard_rows.append({**shard_metadata, "resumed": True})

        if retained_source is not None:
            del retained_source
            retained_source = None
            gc.collect()
        validation = finalize_shards(
            captions_by_id, manifest, args.output, overwrite=args.overwrite
        )
        reports.validation.update(validation)
        reports.summary.update(validation)
        reports.summary["status"] = "complete"
        reports.summary["elapsed_seconds"] = time.monotonic() - started
        if args.cleanup_shards:
            cleanup_shards(args.shard_dir, manifest)
        write_reports(args.report_dir, reports, manifest)
    except Exception as exc:
        reports.summary.setdefault("status", "failed")
        reports.summary["status"] = "failed"
        reports.summary["error"] = f"{type(exc).__name__}: {exc}"
        reports.summary["elapsed_seconds"] = time.monotonic() - started
        try:
            write_reports(args.report_dir, reports, manifest)
        except Exception as report_exc:
            print(f"WARNING: failed to write T5 reports: {report_exc}", file=sys.stderr)
        raise
    print("=" * 72)
    print("HumanML3D T5 token embedding 提取完成")
    print(f"  motion records: {reports.summary['motion_record_count']}")
    print(f"  total captions: {reports.summary['total_caption_count']}")
    print(f"  shards:         {reports.summary['shard_count']}")
    print(f"  resumed shards: {reports.summary['resumed_shard_count']}")
    print(f"  output bytes:   {reports.summary['output_size_bytes']}")
    print(f"  padding error:  {reports.summary['padding_zero_check_max_error']:.3e}")
    print(f"  output:          {args.output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
