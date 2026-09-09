#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""为 MotionMillion motion shard 生成对齐的 T5-3B 150-token FP16 特征。

工具逐个读取 motion shard，只编码其中的 caption，并发布相同 split、shard_id 和
record 顺序的 embedding shard。磁盘格式不保存每条 caption 的 150-token padding：
每个 record 使用一个连续 ``embeddings[有效 token 总数,1024]`` Tensor 和
``offsets[caption数+1]``，Dataset 选中 caption 后才补齐并生成 attention mask。

正式运行固定 T5 模型 revision、隐藏维 1024、最大 150 token 和 FP16 输出。每个
shard 都带 caption fingerprint、模型 revision、token 截断统计与 SHA256，可中断恢复；
训练配置读取预计算特征，不会在每个 DDP rank 常驻 T5-3B。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    MAX_TEXT_TOKENS,
    SCHEMA_VERSION,
    SPLITS,
    TEXT_HIDDEN_DIM,
    MotionMillionError,
    atomic_torch_save,
    atomic_write_json,
    read_json,
    safe_torch_load,
    sha256_file,
    validate_embedding_record,
    validate_motion_record,
)

DEFAULT_MOTION_ROOT = Path("/data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1")
DEFAULT_OUTPUT_ROOT = Path("/data0/user/liwei/datasets/MotionMillion/t5_3b_v1_fp16")
EMBEDDING_VERSION = 1


def _caption_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        motion_id = str(record["motion_id"])
        digest.update(motion_id.encode("utf-8"))
        for caption in record["captions"]:
            payload = str(caption).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _fingerprint_model_files(model_root: Path) -> list[dict[str, Any]]:
    """指纹化实际模型文件，排除 HF local-dir 的易变下载缓存元数据。"""
    model_files = []
    for path in sorted(value for value in model_root.rglob("*") if value.is_file()):
        relative_path = path.relative_to(model_root)
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        model_files.append(
            {
                "path": relative_path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return model_files


def compact_caption_embeddings(
    captions: Sequence[str],
    *,
    encode_batch: Callable[[Sequence[str]], tuple[torch.Tensor, torch.Tensor, Sequence[int]]],
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """编码 caption 并剥离 padding，返回紧凑 Tensor、offset 和 token 统计。"""
    if not captions or any(not isinstance(value, str) or not value.strip() for value in captions):
        raise MotionMillionError("caption 列表必须非空且不能包含空字符串")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    compact: list[torch.Tensor] = []
    offsets = [0]
    counters = Counter()
    for start in range(0, len(captions), batch_size):
        batch = list(captions[start : start + batch_size])
        embeddings, attention_mask, raw_lengths = encode_batch(batch)
        expected = (len(batch), MAX_TEXT_TOKENS, TEXT_HIDDEN_DIM)
        if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != expected:
            raise MotionMillionError(
                f"T5 输出必须为 {expected}，实际 {getattr(embeddings, 'shape', None)}"
            )
        if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != (
            len(batch),
            MAX_TEXT_TOKENS,
        ):
            raise MotionMillionError("T5 attention_mask shape 不符合 [B,150]")
        if len(raw_lengths) != len(batch):
            raise MotionMillionError("raw token length 数量与 batch 不一致")
        if not torch.isfinite(embeddings).all():
            raise MotionMillionError("T5 输出包含 NaN 或 Inf")
        mask = attention_mask.detach().cpu().bool()
        for index in range(len(batch)):
            valid = int(mask[index].sum().item())
            if valid <= 0 or valid > MAX_TEXT_TOKENS:
                raise MotionMillionError(f"caption 有效 token 数非法: {valid}")
            selected = embeddings[index, :valid].detach().cpu().to(torch.float16).contiguous()
            compact.append(selected)
            offsets.append(offsets[-1] + valid)
            raw_length = int(raw_lengths[index])
            counters["captions"] += 1
            counters["valid_tokens"] += valid
            counters["raw_tokens"] += raw_length
            if raw_length > MAX_TEXT_TOKENS:
                counters["truncated_captions"] += 1
    packed = torch.cat(compact, dim=0).contiguous()
    if not torch.isfinite(packed).all():
        raise MotionMillionError("FP16 转换产生 NaN 或 Inf")
    result = {
        "embeddings": packed,
        "offsets": torch.tensor(offsets, dtype=torch.int64),
    }
    validate_embedding_record(result, caption_count=len(captions))
    return result, {key: int(value) for key, value in counters.items()}


def compact_motion_caption_embeddings(
    motion_records: Sequence[Mapping[str, Any]],
    *,
    encode_batch: Callable[[Sequence[str]], tuple[torch.Tensor, torch.Tensor, Sequence[int]]],
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    """把一个 motion shard 的 caption 合批编码，再按动作边界无损拆回。

    MotionMillion 平均每个动作约 21 条 caption；若逐动作调用 T5，会让 64 的配置批量
    长期只吃到约三分之一。这里保持原 motion/caption 顺序，把整个 shard 展平后按真实
    batch size 编码，最后利用 caption token offset 恢复每个动作的紧凑格式。
    """
    caption_counts = [len(record["captions"]) for record in motion_records]
    if not motion_records or any(count <= 0 for count in caption_counts):
        raise MotionMillionError("motion shard 必须非空，且每条动作至少有一条 caption")
    captions = [
        str(caption)
        for record in motion_records
        for caption in record["captions"]
    ]
    packed, counters = compact_caption_embeddings(
        captions,
        encode_batch=encode_batch,
        batch_size=batch_size,
    )
    embeddings = packed["embeddings"]
    caption_offsets = packed["offsets"]
    results: list[dict[str, torch.Tensor]] = []
    caption_cursor = 0
    for caption_count in caption_counts:
        token_start = int(caption_offsets[caption_cursor])
        caption_end = caption_cursor + caption_count
        token_end = int(caption_offsets[caption_end])
        results.append(
            {
                "embeddings": embeddings[token_start:token_end].contiguous(),
                "offsets": (
                    caption_offsets[caption_cursor : caption_end + 1] - token_start
                ).clone(),
            }
        )
        caption_cursor = caption_end
    if caption_cursor != len(captions):
        raise MotionMillionError("caption 合批拆分后的内部计数不一致")
    return results, counters


def _load_t5(
    model_name_or_path: str,
    *,
    requested_revision: str,
    cache_dir: Path | None,
    device: str,
    local_files_only: bool,
) -> tuple[Any, Any, str, list[dict[str, Any]]]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求 {device}，但 CUDA 不可用")
    try:
        from huggingface_hub import HfApi, snapshot_download
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise RuntimeError("缺少 transformers，无法加载 T5-3B") from exc
    requested_path = Path(model_name_or_path).expanduser()
    if requested_path.is_dir():
        model_root = requested_path.resolve()
        resolved_revision = requested_revision
    else:
        try:
            info = HfApi().model_info(model_name_or_path, revision=requested_revision)
            resolved_revision = str(info.sha)
            model_root = Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    revision=resolved_revision,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    local_files_only=local_files_only,
                )
            ).resolve()
        except Exception as exc:
            raise MotionMillionError(
                f"无法固定 T5 模型快照 {model_name_or_path!r}@{requested_revision}: {exc}"
            ) from exc
    if requested_revision != "main" and not requested_path.is_dir() and resolved_revision != requested_revision:
        raise MotionMillionError(
            f"T5 实际 revision={resolved_revision}，请求 revision={requested_revision}"
        )
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_root, local_files_only=True)
        encoder = T5EncoderModel.from_pretrained(model_root, local_files_only=True)
    except Exception as exc:
        raise MotionMillionError(
            f"无法加载 T5-3B {model_name_or_path!r}@{requested_revision}: {exc}"
        ) from exc
    if int(encoder.config.d_model) != TEXT_HIDDEN_DIM:
        raise MotionMillionError(
            f"T5 hidden dim={encoder.config.d_model}，要求 {TEXT_HIDDEN_DIM}"
        )
    model_files = _fingerprint_model_files(model_root)
    if not model_files:
        raise MotionMillionError(f"T5 模型快照没有可指纹化文件: {model_root}")
    encoder.eval().requires_grad_(False)
    encoder = encoder.to(device=device, dtype=torch.float16)
    return encoder, tokenizer, resolved_revision, model_files


def _make_t5_encoder(
    encoder: Any,
    tokenizer: Any,
    *,
    device: str,
) -> Callable[[Sequence[str]], tuple[torch.Tensor, torch.Tensor, Sequence[int]]]:
    def encode(captions: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, Sequence[int]]:
        raw = tokenizer(
            list(captions),
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        raw_ids = raw["input_ids"] if isinstance(raw, dict) else raw.input_ids
        raw_lengths = [len(value) for value in raw_ids]
        tokenized = tokenizer(
            list(captions),
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_TEXT_TOKENS,
            truncation=True,
        )
        input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
        attention_mask = (
            tokenized["attention_mask"]
            if isinstance(tokenized, dict)
            else tokenized.attention_mask
        )
        with torch.no_grad():
            output = encoder(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).last_hidden_state
        return output.detach().cpu(), attention_mask.detach().cpu(), raw_lengths

    return encode


def _embedding_fingerprint(
    source_manifest: Mapping[str, Any],
    *,
    model_name_or_path: str,
    resolved_revision: str,
    model_files: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "embedding_version": EMBEDDING_VERSION,
        "source_build_fingerprint": source_manifest["build_fingerprint"],
        "model_name_or_path": model_name_or_path,
        "resolved_revision": resolved_revision,
        "model_files": list(model_files),
        "max_text_tokens": MAX_TEXT_TOKENS,
        "hidden_dim": TEXT_HIDDEN_DIM,
        "dtype": "float16",
        "storage": "compact_valid_tokens_with_offsets",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _select_source_shards(
    source_manifest: Mapping[str, Any],
    *,
    limit_shards: int | None,
    worker_rank: int,
    worker_world_size: int,
) -> list[Mapping[str, Any]]:
    """为多 GPU 离线提取器确定互不重叠、可复现的 source shard 集合。"""
    selected = list(source_manifest["shards"])
    if limit_shards is not None:
        selected = selected[:limit_shards]
    if worker_world_size > 1:
        selected = [
            item
            for item in selected
            if int(item["shard_id"]) % worker_world_size == worker_rank
        ]
    return selected


def _validate_embedding_shard(
    path: Path,
    motion_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = safe_torch_load(path)
    if not isinstance(records, list) or len(records) != len(motion_records):
        raise MotionMillionError(f"embedding shard record 数不一致: {path}")
    rows = []
    for index, (embedding, motion) in enumerate(zip(records, motion_records)):
        if str(embedding.get("motion_id")) != str(motion["motion_id"]):
            raise MotionMillionError(f"embedding/motion record 顺序不一致: {path}:{index}")
        validate_embedding_record(embedding, caption_count=len(motion["captions"]))
        rows.append(
            {
                "motion_id": str(motion["motion_id"]),
                "record_index": index,
                "caption_count": len(motion["captions"]),
                "token_count": int(embedding["embeddings"].shape[0]),
            }
        )
    return rows


def extract_embeddings(
    args: argparse.Namespace,
    *,
    injected_encoder: Callable[
        [Sequence[str]], tuple[torch.Tensor, torch.Tensor, Sequence[int]]
    ]
    | None = None,
    injected_revision: str | None = None,
) -> dict[str, Any]:
    """逐 motion shard 生成严格对齐的紧凑 embedding release。"""
    started = time.monotonic()
    motion_root = Path(args.motion_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    release_path = motion_root / "dataset_release.json"
    if not release_path.is_file():
        raise FileNotFoundError(f"缺少 MotionMillion motion release: {release_path}")
    motion_release = read_json(release_path)
    output_root.mkdir(parents=True, exist_ok=True)
    worker_rank = int(getattr(args, "worker_rank", 0))
    worker_world_size = int(getattr(args, "worker_world_size", 1))
    if worker_world_size <= 0 or not 0 <= worker_rank < worker_world_size:
        raise ValueError(
            f"非法 worker 拓扑: rank={worker_rank}, world_size={worker_world_size}"
        )
    distributed_worker = worker_world_size > 1

    if injected_encoder is None:
        encoder, tokenizer, resolved_revision, model_files = _load_t5(
            args.model_name_or_path,
            requested_revision=args.model_revision,
            cache_dir=args.cache_dir,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        encode_batch = _make_t5_encoder(encoder, tokenizer, device=args.device)
    else:
        encode_batch = injected_encoder
        resolved_revision = injected_revision or args.model_revision
        model_files = [
            {
                "path": "injected-test-encoder",
                "size_bytes": 0,
                "sha256": "0" * 64,
            }
        ]

    manifests: dict[str, dict[str, Any]] = {}
    global_counters = Counter()
    total_source_records = 0
    for split in SPLITS:
        source = read_json(motion_root / "manifests" / f"{split}.json")
        selected = _select_source_shards(
            source,
            limit_shards=args.limit_shards,
            worker_rank=worker_rank,
            worker_world_size=worker_world_size,
        )
        total_source_records += sum(int(item["record_count"]) for item in selected)
    processed_records = 0
    for split in SPLITS:
        source_manifest_path = motion_root / "manifests" / f"{split}.json"
        source_manifest = read_json(source_manifest_path)
        fingerprint = _embedding_fingerprint(
            source_manifest,
            model_name_or_path=args.model_name_or_path,
            resolved_revision=resolved_revision,
            model_files=model_files,
        )
        contract_path = output_root / "contracts" / f"{split}.json"
        if contract_path.is_file():
            if not args.resume:
                raise FileExistsError(f"embedding contract 已存在: {contract_path}；请使用 --resume")
            if read_json(contract_path).get("embedding_fingerprint") != fingerprint:
                raise MotionMillionError(f"embedding contract fingerprint 不一致: {contract_path}")
        else:
            atomic_write_json(
                contract_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "embedding_version": EMBEDDING_VERSION,
                    "embedding_fingerprint": fingerprint,
                    "source_manifest": str(source_manifest_path),
                    "source_build_fingerprint": source_manifest["build_fingerprint"],
                    "model_name_or_path": args.model_name_or_path,
                    "requested_revision": args.model_revision,
                    "resolved_revision": resolved_revision,
                    "model_files": model_files,
                    "max_text_tokens": MAX_TEXT_TOKENS,
                    "hidden_dim": TEXT_HIDDEN_DIM,
                    "dtype": "float16",
                    "storage": "compact_valid_tokens_with_offsets",
                },
            )

        output_shards: list[dict[str, Any]] = []
        selected_source_shards = _select_source_shards(
            source_manifest,
            limit_shards=args.limit_shards,
            worker_rank=worker_rank,
            worker_world_size=worker_world_size,
        )
        for source_shard in selected_source_shards:
            shard_id = int(source_shard["shard_id"])
            motion_path = motion_root / str(source_shard["path"])
            if sha256_file(motion_path) != source_shard["sha256"]:
                raise MotionMillionError(f"motion shard SHA256 不一致: {motion_path}")
            motion_records = safe_torch_load(motion_path)
            if not isinstance(motion_records, list):
                raise MotionMillionError(f"motion shard 必须为 list: {motion_path}")
            for record in motion_records:
                validate_motion_record(record)

            relative = (
                Path("shards")
                / split
                / f"motionmillion_t5_{split}_{shard_id:06d}.pth"
            )
            output_path = output_root / relative
            meta_path = output_path.with_suffix(".meta.json")
            caption_fingerprint = _caption_fingerprint(motion_records)
            if output_path.exists() or meta_path.exists():
                if not args.resume or not output_path.is_file() or not meta_path.is_file():
                    raise FileExistsError(f"embedding shard 状态不完整或未启用 resume: {output_path}")
                metadata = read_json(meta_path)
                if (
                    metadata.get("embedding_fingerprint") != fingerprint
                    or metadata.get("caption_fingerprint") != caption_fingerprint
                ):
                    raise MotionMillionError(f"已有 embedding shard fingerprint 不一致: {output_path}")
                if sha256_file(output_path) != metadata.get("sha256"):
                    raise MotionMillionError(f"已有 embedding shard SHA256 不一致: {output_path}")
                rows = _validate_embedding_shard(output_path, motion_records)
                output_shards.append(metadata)
                global_counters.update(metadata.get("token_statistics", {}))
                global_counters["resumed_motion_records"] += len(rows)
                processed_records += len(rows)
                continue

            embedding_records: list[dict[str, Any]] = []
            compact_records, counters = compact_motion_caption_embeddings(
                motion_records,
                encode_batch=encode_batch,
                batch_size=args.batch_size,
            )
            shard_counters = Counter(counters)
            for motion_record, compact in zip(motion_records, compact_records):
                embedding_record = {
                    "motion_id": str(motion_record["motion_id"]),
                    **compact,
                }
                validate_embedding_record(
                    embedding_record,
                    caption_count=len(motion_record["captions"]),
                )
                embedding_records.append(embedding_record)
            atomic_torch_save(embedding_records, output_path)
            rows = _validate_embedding_shard(output_path, motion_records)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "embedding_version": EMBEDDING_VERSION,
                "embedding_fingerprint": fingerprint,
                "caption_fingerprint": caption_fingerprint,
                "split": split,
                "shard_id": shard_id,
                "path": relative.as_posix(),
                "source_motion_path": str(source_shard["path"]),
                "source_motion_sha256": source_shard["sha256"],
                "record_count": len(rows),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "token_statistics": {key: int(value) for key, value in shard_counters.items()},
                "records": rows,
            }
            atomic_write_json(meta_path, metadata)
            output_shards.append(metadata)
            global_counters.update(shard_counters)
            global_counters["encoded_motion_records"] += len(rows)
            processed_records += len(rows)
            elapsed = max(time.monotonic() - started, 1.0e-9)
            record_rate = processed_records / elapsed
            caption_rate = global_counters["captions"] / elapsed
            eta_seconds = (total_source_records - processed_records) / max(
                record_rate, 1.0e-9
            )
            print(
                f"[MotionMillion T5] split={split}, shard={shard_id}, "
                f"records={processed_records}/{total_source_records}, "
                f"records/s={record_rate:.3f}, captions/s={caption_rate:.3f}, "
                f"ETA={eta_seconds / 3600:.2f}h",
                flush=True,
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "embedding_version": EMBEDDING_VERSION,
            "embedding_fingerprint": fingerprint,
            "source_build_fingerprint": source_manifest["build_fingerprint"],
            "split": split,
            "model_name_or_path": args.model_name_or_path,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_revision,
            "max_text_tokens": MAX_TEXT_TOKENS,
            "hidden_dim": TEXT_HIDDEN_DIM,
            "dtype": "float16",
            "record_count": sum(int(item["record_count"]) for item in output_shards),
            "shards": [
                {
                    key: item[key]
                    for key in (
                        "shard_id",
                        "path",
                        "source_motion_path",
                        "source_motion_sha256",
                        "record_count",
                        "size_bytes",
                        "sha256",
                    )
                }
                for item in output_shards
            ],
        }
        # 多 GPU worker 只原子写自己负责的 shard/meta，避免多个进程竞争覆盖最终
        # manifest。全部 worker 结束后再用单进程 ``--resume`` 复核 SHA/顺序并发布。
        if not distributed_worker:
            atomic_write_json(output_root / "manifests" / f"{split}.json", manifest)
        manifests[split] = manifest

    release = {
        "schema_version": SCHEMA_VERSION,
        "embedding_version": EMBEDDING_VERSION,
        "motion_release": str(release_path),
        "source_build_fingerprint": motion_release["build_fingerprint"],
        "model_name_or_path": args.model_name_or_path,
        "requested_revision": args.model_revision,
        "resolved_revision": resolved_revision,
        "model_files": model_files,
        "max_text_tokens": MAX_TEXT_TOKENS,
        "hidden_dim": TEXT_HIDDEN_DIM,
        "dtype": "float16",
        "storage": "compact_valid_tokens_with_offsets",
        "mode": "distributed_worker" if distributed_worker else "complete_release",
        "worker_rank": worker_rank,
        "worker_world_size": worker_world_size,
        "manifests": {
            split: {
                "path": f"manifests/{split}.json",
                "record_count": manifests[split]["record_count"],
            }
            for split in SPLITS
        },
        "counters": {key: int(value) for key, value in global_counters.items()},
        "elapsed_seconds": time.monotonic() - started,
    }
    if distributed_worker:
        atomic_write_json(
            output_root / "workers" / f"rank_{worker_rank:03d}.json",
            release,
        )
    else:
        atomic_write_json(output_root / "embedding_release.json", release)
    return release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-name-or-path", default="t5-3b")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="正式运行必须给出不可变 Hugging Face commit SHA 或本地模型版本标识",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--worker-rank",
        type=int,
        default=0,
        help="多 GPU 离线提取 worker 序号；world-size>1 时按 shard_id 取模分工",
    )
    parser.add_argument("--worker-world-size", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or (args.limit_shards is not None and args.limit_shards <= 0):
        raise SystemExit("batch-size 和 limit-shards 必须为正数")
    if args.worker_world_size <= 0 or not 0 <= args.worker_rank < args.worker_world_size:
        raise SystemExit("worker-rank 必须位于 [0, worker-world-size) 内")
    report = extract_embeddings(args)
    print(
        "MotionMillion T5 "
        + ("worker complete: " if args.worker_world_size > 1 else "release complete: ")
        + ", ".join(
            f"{split}={value['record_count']}"
            for split, value in report["manifests"].items()
        )
    )


if __name__ == "__main__":
    main()
