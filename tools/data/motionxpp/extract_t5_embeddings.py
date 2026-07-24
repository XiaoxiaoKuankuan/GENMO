#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""从 Motion-X++ motion manifest 流式提取 T5-3B FP16 文本 embedding 分片。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.network.utils import encode_text_batch  # noqa: E402
from tools.data.motionxpp.common import (  # noqa: E402
    MotionXppError,
    atomic_torch_save,
    atomic_write_json,
    read_jsonl,
    safe_torch_load,
)

DEFAULT_OUTPUT_ROOT = Path("inputs/Motion-Xplusplus/t5_embeddings_v1_half")
MAX_TEXT_LEN = 50
HIDDEN_DIM = 1024
OUTPUT_DTYPE = torch.float16
EMBEDDING_VERSION = 1


def manifest_fingerprint(
    manifest_path: Path,
    model_name_or_path: str,
    motions_per_shard: int,
) -> str:
    """对源 manifest 内容和 T5 输出契约做确定性 fingerprint。"""
    digest = hashlib.sha256(manifest_path.read_bytes())
    for value in (
        str(model_name_or_path),
        f"motions_per_shard={motions_per_shard}",
        f"max_text_len={MAX_TEXT_LEN}",
        f"hidden_dim={HIDDEN_DIM}",
        "dtype=float16",
        f"version={EMBEDDING_VERSION}",
    ):
        digest.update(value.encode())
    return digest.hexdigest()


def _resolve_shard_path(source_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else source_root / path


def _load_caption_records(
    rows: Sequence[Mapping[str, Any]],
    source_root: Path,
    *,
    strict: bool,
) -> tuple[OrderedDict[str, list[str]], list[dict[str, Any]]]:
    """每次仅持有当前 source motion shard，复制 caption 后立即释放动作。"""
    captions: OrderedDict[str, list[str]] = OrderedDict()
    invalid: list[dict[str, Any]] = []
    cached_path: Path | None = None
    cached_shard: Any = None
    for row in rows:
        motion_id = str(row["motion_id"])
        path = _resolve_shard_path(source_root, str(row["shard_path"]))
        try:
            if path != cached_path:
                del cached_shard
                cached_shard = None
                cached_shard = safe_torch_load(path)
                cached_path = path
                if not isinstance(cached_shard, dict):
                    raise MotionXppError(f"Motion shard is not a dict: {path}")
            key = str(row.get("record_key", motion_id))
            record = cached_shard[key]
            text_data = record.get("text_data")
            if not isinstance(text_data, list) or not text_data:
                raise MotionXppError("text_data must be a non-empty list")
            values: list[str] = []
            for index, item in enumerate(text_data):
                caption = item.get("caption") if isinstance(item, Mapping) else None
                if not isinstance(caption, str) or not caption.strip():
                    raise MotionXppError(f"text_data[{index}].caption is empty")
                values.append(caption)
            expected = int(row.get("caption_count", len(values)))
            if len(values) != expected:
                raise MotionXppError(
                    f"caption count mismatch: manifest={expected}, record={len(values)}"
                )
            captions[motion_id] = values
        except Exception as exc:
            issue = {
                "motion_id": motion_id,
                "shard_path": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            invalid.append(issue)
            if strict:
                raise MotionXppError(f"Invalid caption source for {motion_id}: {exc}") from exc
    del cached_shard
    return captions, invalid


def validate_embedding_tensor(value: Any, expected_captions: int, motion_id: str) -> None:
    """验证单条 `[C,50,1024]` CPU FP16 embedding。"""
    expected = (expected_captions, MAX_TEXT_LEN, HIDDEN_DIM)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
        raise MotionXppError(
            f"{motion_id}: embedding must be {expected}, got {getattr(value, 'shape', None)}"
        )
    if value.dtype != OUTPUT_DTYPE or value.device.type != "cpu":
        raise MotionXppError(f"{motion_id}: embedding must be CPU float16")
    if not value.is_contiguous() or not torch.isfinite(value).all():
        raise MotionXppError(f"{motion_id}: embedding must be contiguous and finite")


def encode_caption_records(
    captions_by_id: Mapping[str, list[str]],
    *,
    encode_batch: Callable[[Sequence[str]], torch.Tensor],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """批量编码 caption，再恢复为每个 motion 对应的三维 tensor。

    该函数通过注入 ``encode_batch`` 使单元测试不需要下载或加载 T5。
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    flat: list[str] = []
    owners: list[tuple[str, int]] = []
    for motion_id, captions in captions_by_id.items():
        for caption_index, caption in enumerate(captions):
            flat.append(caption)
            owners.append((motion_id, caption_index))
    batches: list[torch.Tensor] = []
    for start in range(0, len(flat), batch_size):
        values = encode_batch(flat[start : start + batch_size])
        expected = (min(batch_size, len(flat) - start), MAX_TEXT_LEN, HIDDEN_DIM)
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected:
            raise MotionXppError(
                f"T5 output must be {expected}, got {getattr(values, 'shape', None)}"
            )
        if not torch.isfinite(values).all():
            raise MotionXppError("T5 output contains NaN or Inf")
        half = values.detach().cpu().to(torch.float16).contiguous()
        if not torch.isfinite(half).all():
            raise MotionXppError("FP16 conversion produced NaN or Inf")
        batches.append(half)
    flat_embedding = (
        torch.cat(batches, 0)
        if batches
        else torch.empty(0, MAX_TEXT_LEN, HIDDEN_DIM, dtype=torch.float16)
    )
    result: dict[str, torch.Tensor] = {}
    positions: dict[str, list[int]] = {key: [] for key in captions_by_id}
    for flat_index, (motion_id, _) in enumerate(owners):
        positions[motion_id].append(flat_index)
    for motion_id, captions in captions_by_id.items():
        result[motion_id] = flat_embedding[positions[motion_id]].contiguous().clone()
        validate_embedding_tensor(result[motion_id], len(captions), motion_id)
    return result


def _encode_with_loaded_t5(
    captions: Sequence[str],
    *,
    encoder: Any,
    tokenizer: Any,
    device: str,
) -> torch.Tensor:
    """适配 encode_text_batch 到可注入的单参数 batch callable。"""
    return encode_text_batch(
        raw_text=list(captions),
        text_encoder=encoder,
        tokenizer=tokenizer,
        device=device,
    )


def _load_t5(
    model_name_or_path: str,
    *,
    device: str,
    model_dtype: str,
    local_files_only: bool,
    cache_dir: Path | None,
) -> tuple[Any, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} requested but CUDA is unavailable")
    from transformers import T5EncoderModel, T5Tokenizer

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, **kwargs)
        encoder = T5EncoderModel.from_pretrained(model_name_or_path, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load T5-3B from {model_name_or_path!r}. "
            "Provide a valid local path/cache or fix Hugging Face access. "
            f"Original error: {exc}"
        ) from exc
    if int(encoder.config.d_model) != HIDDEN_DIM:
        raise RuntimeError(
            f"T5 hidden dimension is {encoder.config.d_model}; expected {HIDDEN_DIM}"
        )
    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[model_dtype]
    encoder.eval().requires_grad_(False)
    encoder = encoder.to(device=device, dtype=dtype)
    return encoder, tokenizer


def _validate_shard(path: Path, captions_by_id: Mapping[str, list[str]]) -> int:
    value = safe_torch_load(path)
    if not isinstance(value, dict) or list(value) != list(captions_by_id):
        raise MotionXppError(f"Embedding shard key mismatch: {path}")
    for motion_id, captions in captions_by_id.items():
        validate_embedding_tensor(value[motion_id], len(captions), motion_id)
    return path.stat().st_size


def _caption_fingerprint(global_fingerprint: str, captions_by_id: Mapping[str, list[str]]) -> str:
    digest = hashlib.sha256(global_fingerprint.encode())
    for motion_id, captions in captions_by_id.items():
        digest.update(motion_id.encode())
        for caption in captions:
            encoded = caption.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def extract_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    """按 motions_per_shard 流式读取 motion shards 并发布 embedding shards。"""
    started = time.monotonic()
    source_manifest = Path(args.manifest).expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Motion manifest does not exist: {source_manifest}")
    source_root = source_manifest.parent.parent
    split = source_manifest.stem
    rows = read_jsonl(source_manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise MotionXppError(f"No motion rows selected from {source_manifest}")
    ids = [str(row["motion_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise MotionXppError("Motion manifest contains duplicate motion_id values")
    output_root = Path(args.output_root)
    manifest_path = output_root / "manifests" / f"{split}.json"
    fingerprint = manifest_fingerprint(
        source_manifest, args.model_name_or_path, args.motions_per_shard
    )
    if args.limit is not None:
        fingerprint = hashlib.sha256(f"{fingerprint}:limit={args.limit}".encode()).hexdigest()
    existing: dict[str, Any] | None = None
    if manifest_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Embedding manifest exists: {manifest_path}; use --resume")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise MotionXppError(
                f"Existing embedding manifest fingerprint differs: {manifest_path}"
            )

    plans: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(rows), args.motions_per_shard)):
        selected = rows[start : start + args.motions_per_shard]
        plans.append(
            {
                "shard_index": shard_index,
                "rows": selected,
                "motion_ids": [str(row["motion_id"]) for row in selected],
                "path": (
                    Path("shards") / split / f"motionxpp_t5_{split}_{shard_index:05d}.pth"
                ).as_posix(),
                "status": "pending",
            }
        )
    manifest: dict[str, Any] = {
        "version": EMBEDDING_VERSION,
        "split": split,
        "source_manifest": str(source_manifest),
        "source_root": str(source_root),
        "fingerprint": fingerprint,
        "model_name_or_path": args.model_name_or_path,
        "max_text_len": MAX_TEXT_LEN,
        "hidden_dim": HIDDEN_DIM,
        "dtype": "float16",
        "motion_count": len(rows),
        "motions_per_shard": args.motions_per_shard,
        "shards": [],
        "motion_to_shard": {},
        "invalid_records": [],
    }
    if existing is not None:
        # 只借用完成状态，布局必须完全一致。
        existing_layout = [
            (item.get("shard_index"), item.get("path"), item.get("motion_ids"))
            for item in existing.get("shards", [])
        ]
        planned_layout = [(item["shard_index"], item["path"], item["motion_ids"]) for item in plans]
        if existing_layout != planned_layout[: len(existing_layout)]:
            raise MotionXppError(
                "Existing embedding shard layout is not a valid completed prefix "
                "of the current plan"
            )

    encoder: Any | None = None
    tokenizer: Any | None = None
    resumed_count = 0
    encoded_count = 0
    try:
        for plan_index, plan in enumerate(plans):
            captions_by_id, invalid = _load_caption_records(
                plan["rows"], source_root, strict=args.strict
            )
            if invalid:
                manifest["invalid_records"].extend(invalid)
                raise MotionXppError(
                    f"Selected shard {plan_index} contains {len(invalid)} invalid records; "
                    "embedding/motion key parity cannot be preserved"
                )
            output_path = output_root / plan["path"]
            meta_path = output_path.with_suffix(".meta.json")
            shard_fingerprint = _caption_fingerprint(fingerprint, captions_by_id)
            resumed = False
            if output_path.exists() or meta_path.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"Embedding shard state exists: {output_path}; use --resume"
                    )
                if output_path.is_file():
                    if meta_path.is_file():
                        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                        if metadata.get("fingerprint") != shard_fingerprint:
                            raise MotionXppError(
                                f"Embedding shard fingerprint differs: {output_path}"
                            )
                        size = _validate_shard(output_path, captions_by_id)
                        resumed = True
                        resumed_count += 1
                    else:
                        # 没有 caption fingerprint sidecar 时不能证明 PTH 对应当前文本；
                        # 删除本工具自己的不完整输出并重新编码。
                        output_path.unlink()
                else:
                    # 只有 meta 说明上次尚未成功发布 embedding PTH。
                    meta_path.unlink(missing_ok=True)
            if not resumed:
                if encoder is None or tokenizer is None:
                    encoder, tokenizer = _load_t5(
                        args.model_name_or_path,
                        device=args.device,
                        model_dtype=args.model_dtype,
                        local_files_only=args.local_files_only,
                        cache_dir=args.cache_dir,
                    )
                shard = encode_caption_records(
                    captions_by_id,
                    encode_batch=partial(
                        _encode_with_loaded_t5,
                        encoder=encoder,
                        tokenizer=tokenizer,
                        device=args.device,
                    ),
                    batch_size=args.batch_size,
                )
                atomic_torch_save(shard, output_path)
                size = _validate_shard(output_path, captions_by_id)
                atomic_write_json(
                    meta_path,
                    {
                        "fingerprint": shard_fingerprint,
                        "global_fingerprint": fingerprint,
                        "motion_ids": list(captions_by_id),
                        "caption_counts": {
                            key: len(value) for key, value in captions_by_id.items()
                        },
                        "output_size_bytes": size,
                    },
                )
                encoded_count += sum(len(value) for value in captions_by_id.values())
                del shard
            shard_meta = {
                "shard_index": plan["shard_index"],
                "path": plan["path"],
                "motion_ids": list(captions_by_id),
                "caption_counts": {key: len(value) for key, value in captions_by_id.items()},
                "motion_count": len(captions_by_id),
                "caption_count": sum(len(value) for value in captions_by_id.values()),
                "output_size_bytes": size,
                "fingerprint": shard_fingerprint,
                "status": "complete",
                "resumed": resumed,
            }
            manifest["shards"].append(shard_meta)
            for motion_id, captions in captions_by_id.items():
                manifest["motion_to_shard"][motion_id] = {
                    "shard_path": plan["path"],
                    "record_key": motion_id,
                    "caption_count": len(captions),
                }
            atomic_write_json(manifest_path, manifest)
            del captions_by_id
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(
                f"[T5 {plan_index + 1}/{len(plans)}] "
                f"{'resumed' if resumed else 'encoded'} {plan['path']}"
            )
    finally:
        if encoder is not None:
            del encoder
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    manifest["status"] = "complete"
    manifest["resumed_shards"] = resumed_count
    manifest["encoded_caption_count_this_run"] = encoded_count
    manifest["total_caption_count"] = sum(int(item["caption_count"]) for item in manifest["shards"])
    manifest["elapsed_seconds"] = time.monotonic() - started
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """创建 T5 提取 CLI。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--motions-per-shard", type=int, default=256)
    parser.add_argument("--model-name-or-path", default="t5-3b")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.motions_per_shard <= 0:
        raise ValueError("--batch-size and --motions-per-shard must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")


def main(argv: list[str] | None = None) -> int:
    """提取并打印摘要。"""
    args = build_parser().parse_args(argv)
    _validate_args(args)
    manifest = extract_embeddings(args)
    print("=" * 72)
    print("Motion-X++ T5 embedding 提取完成")
    print(f"  split:             {manifest['split']}")
    print(f"  motions:           {manifest['motion_count']}")
    print(f"  captions:          {manifest['total_caption_count']}")
    print(f"  shards:            {len(manifest['shards'])}")
    print(f"  resumed shards:    {manifest['resumed_shards']}")
    print(
        f"  output manifest:   {Path(args.output_root) / 'manifests' / (manifest['split'] + '.json')}"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
