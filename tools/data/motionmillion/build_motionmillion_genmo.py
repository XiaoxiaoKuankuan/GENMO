#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""把官方 MotionMillion 272D 归档流式转换为 GENMO SMPL 训练分片。

构建器先把官方 ``version1/t2m_60_300`` split 与文本 tar 建成可审计 SQLite
元数据索引，再逐个读取 ``motion_272rpr`` tar member。每条动作必须同时命中官方
split 和非空文本，随后按官方恢复公式转换为 ``pose[F,66] + trans[F,3] +
beta[10]``。工具不会把全部 308 GB 数据解压成第二份副本。

训练输出按 split 每 512 条原子发布一个 PTH motion shard，并生成只含整数的 mmap
sample index。每个 shard 旁的 meta JSON 足以支持中断恢复：恢复只接受完全一致的
build fingerprint、已验证 shard 和连续 shard 编号，禁止混用不同数据 revision、
转换参数或半成品。
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import sqlite3
import sys
import tarfile
import time
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    MAX_FRAMES,
    MIN_FRAMES,
    OFFICIAL_FPS,
    RECORDS_PER_SHARD,
    SAMPLE_INDEX_DTYPE,
    SCHEMA_VERSION,
    SPLITS,
    MotionMillionError,
    MotionMillionFilteredError,
    atomic_save_npy,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    identifier_candidates,
    mirror_base_id,
    normalize_member_name,
    read_json,
    recover_smpl_from_272,
    safe_torch_load,
    sha256_bytes,
    sha256_file,
    validate_motion_record,
)

DEFAULT_RAW_ROOT = Path("/data0/user/liwei/datasets/MotionMillion/raw_hf")
DEFAULT_OUTPUT_ROOT = Path("/data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1")
BUILD_VERSION = 3


def _source_subset(archive_relative: str, motion_id: str) -> str:
    """从官方三大来源名中提取稳定的 subset 标签。"""
    combined = f"{archive_relative}/{motion_id}".lower()
    for subset in ("MotionGV", "MotionLLAMA", "MotionUnion"):
        if subset.lower() in combined:
            return subset
    return "unknown"


def _archive_identifier_candidates(
    member_name: str,
    archive_relative: str,
) -> list[str]:
    """补回官方分卷 tar 省略的顶层来源命名空间。

    MotionGV、MotionLLAMA、MotionUnion 及其镜像归档位于
    ``motion_272rpr/<来源>/<分卷>.tar.gz``，但 tar member 通常从 ``folder0``、
    ``finedance`` 等下一层开始；官方 split ID 则保留 ``MotionGV/`` 等来源前缀。
    根目录下的 PhantomDance 归档已经把数据集名写在 member 中，不需要补前缀。
    所有候选最终仍必须唯一命中 split/text SQLite，不能凭路径推断直接接收。
    """
    candidates = identifier_candidates(member_name)
    archive_parts = Path(archive_relative).parts
    if len(archive_parts) < 3 or archive_parts[0] != "motion_272rpr":
        return candidates
    namespace = archive_parts[1]
    prefixed = [
        f"{namespace}/{candidate}"
        for candidate in candidates
        if candidate != namespace and not candidate.startswith(f"{namespace}/")
    ]
    return list(dict.fromkeys([*candidates, *prefixed]))


def _open_tar(path: Path) -> tarfile.TarFile:
    try:
        return tarfile.open(path, mode="r:*")
    except (tarfile.TarError, OSError) as exc:
        raise MotionMillionError(f"无法打开或校验 tar 归档 {path}: {exc}") from exc


def _read_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if not member.isfile():
        raise MotionMillionError(f"归档成员不是普通文件: {member.name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise MotionMillionError(f"无法读取归档成员: {member.name}")
    return handle.read()


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


def _create_metadata_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS split_entries (
            motion_id TEXT PRIMARY KEY,
            split TEXT NOT NULL CHECK(split IN ('train','val','test')),
            mirror_base_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS split_mirror_idx
            ON split_entries(mirror_base_id);
        CREATE TABLE IF NOT EXISTS split_exclusions (
            motion_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            mirror_base_id TEXT NOT NULL,
            canonical_split TEXT NOT NULL,
            reason TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS text_entries (
            motion_id TEXT PRIMARY KEY,
            captions_json TEXT NOT NULL,
            source_member TEXT NOT NULL,
            source_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS built_entries (
            motion_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            source_archive TEXT NOT NULL,
            source_member TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS observed_entries (
            motion_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            source_archive TEXT NOT NULL,
            source_member TEXT NOT NULL,
            error_type TEXT,
            error TEXT
        );
        """
    )
    return connection


def _metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _set_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
    )


def _resolve_candidate(
    connection: sqlite3.Connection,
    candidates: Sequence[str],
    *,
    require_text: bool,
) -> tuple[str, str] | None:
    matches: list[tuple[str, str]] = []
    for candidate in candidates:
        if require_text:
            row = connection.execute(
                """
                SELECT s.motion_id, s.split
                FROM split_entries AS s
                JOIN text_entries AS t ON t.motion_id = s.motion_id
                WHERE s.motion_id = ?
                """,
                (candidate,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT motion_id, split FROM split_entries WHERE motion_id = ?",
                (candidate,),
            ).fetchone()
        if row is not None and (str(row[0]), str(row[1])) not in matches:
            matches.append((str(row[0]), str(row[1])))
    if len(matches) > 1:
        raise MotionMillionError(f"归档路径同时匹配多个官方 ID: {matches}")
    return matches[0] if matches else None


def _index_splits(connection: sqlite3.Connection, split_tar: Path) -> dict[str, int]:
    counts = Counter()
    seen_in_call: dict[str, str] = {}
    with _open_tar(split_tar) as archive:
        matched_files = 0
        for member in archive:
            split = _split_from_member(member.name)
            if split is None or not member.isfile():
                continue
            matched_files += 1
            text = _read_tar_member(archive, member).decode("utf-8-sig")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                motion_id = _normalize_split_id(line)
                previous = seen_in_call.get(motion_id)
                if previous is not None and previous != split:
                    raise MotionMillionError(
                        f"官方 split 泄漏: {motion_id!r} 同时属于 {previous}/{split}"
                    )
                seen_in_call[motion_id] = split
                try:
                    connection.execute(
                        "INSERT INTO split_entries(motion_id, split, mirror_base_id) VALUES (?, ?, ?)",
                        (motion_id, split, mirror_base_id(motion_id)),
                    )
                except sqlite3.IntegrityError as exc:
                    raise MotionMillionError(
                        f"官方 split 重复 ID: {motion_id!r}，文件 {member.name}:{line_number}"
                    ) from exc
                counts[split] += 1
        if matched_files != len(SPLITS):
            raise MotionMillionError(
                f"应在 split.tar.gz 中找到 train/val/test 三个 t2m_60_300 文件，实际 {matched_files}"
            )
    connection.commit()
    return {split: int(counts[split]) for split in SPLITS}


def _index_texts(connection: sqlite3.Connection, texts_tar: Path) -> dict[str, int]:
    counts = Counter()
    with _open_tar(texts_tar) as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".txt"):
                continue
            payload = _read_tar_member(archive, member)
            if len(payload) > 4 * 1024 * 1024:
                raise MotionMillionError(f"单条文本文件异常大: {member.name}")
            resolved = _resolve_candidate(
                connection,
                identifier_candidates(member.name),
                require_text=False,
            )
            if resolved is None:
                counts["not_in_official_split"] += 1
                continue
            motion_id, _ = resolved
            raw_lines = payload.decode("utf-8-sig").splitlines()
            captions = [line.strip() for line in raw_lines if line.strip()]
            counts["blank_caption_lines"] += len(raw_lines) - len(captions)
            if not captions:
                counts["empty"] += 1
                continue
            counts["duplicate_captions"] += len(captions) - len(set(captions))
            try:
                connection.execute(
                    """
                    INSERT INTO text_entries(motion_id, captions_json, source_member, source_sha256)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        motion_id,
                        json.dumps(captions, ensure_ascii=False),
                        normalize_member_name(member.name),
                        sha256_bytes(payload),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise MotionMillionError(f"文本归档重复 motion ID: {motion_id!r}") from exc
            counts["motions"] += 1
            counts["captions"] += len(captions)
            if counts.get("caption_count_min", 0) == 0:
                counts["caption_count_min"] = len(captions)
            else:
                counts["caption_count_min"] = min(
                    counts["caption_count_min"], len(captions)
                )
            counts["caption_count_max"] = max(
                counts.get("caption_count_max", 0), len(captions)
            )
            if counts["motions"] % 10000 == 0:
                connection.commit()
    connection.commit()
    return {key: int(value) for key, value in sorted(counts.items())}


def _quarantine_cross_split_mirrors(
    connection: sqlite3.Connection,
    output_root: Path,
) -> dict[str, Any]:
    """以非镜像原动作的官方 split 为准，隔离落到其他 split 的镜像条目。

    MotionMillion 官方 ``t2m_60_300`` 会把部分原动作及镜像增强随机分到不同 split。
    为同时保持原动作的官方 split 和零镜像泄漏，本分支不重分配条目：保留 canonical
    原动作所在 split 及同 split 变体，把其他 split 变体写入可审计排除表后移出训练
    eligibility。若一个 base 找不到唯一原动作 split，则拒绝猜测并阻断构建。
    """
    leaking_bases = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT mirror_base_id
            FROM split_entries
            GROUP BY mirror_base_id
            HAVING COUNT(DISTINCT split) > 1
            ORDER BY mirror_base_id
            """
        )
    ]
    excluded_by_split = Counter()
    retained_in_leaking_groups = Counter()
    for base_id in leaking_bases:
        rows = [
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                """
                SELECT motion_id, split FROM split_entries
                WHERE mirror_base_id = ? ORDER BY split, motion_id
                """,
                (base_id,),
            )
        ]
        canonical_splits = {
            split for motion_id, split in rows if motion_id == base_id
        }
        if len(canonical_splits) != 1:
            raise MotionMillionError(
                "镜像跨 split 组无法确定唯一非镜像原动作 split，拒绝猜测: "
                f"base_id={base_id!r}, canonical_splits={sorted(canonical_splits)}, "
                f"rows={rows}"
            )
        canonical_split = next(iter(canonical_splits))
        for motion_id, split in rows:
            if split == canonical_split:
                retained_in_leaking_groups[split] += 1
                continue
            connection.execute(
                """
                INSERT INTO split_exclusions
                (motion_id, split, mirror_base_id, canonical_split, reason)
                VALUES (?, ?, ?, ?, 'mirror_cross_split')
                """,
                (motion_id, split, base_id, canonical_split),
            )
            excluded_by_split[split] += 1
    connection.execute(
        """
        DELETE FROM split_entries
        WHERE motion_id IN (SELECT motion_id FROM split_exclusions)
        """
    )
    connection.commit()
    report_path = output_root / "reports" / "mirror_cross_split_exclusions.jsonl"
    atomic_write_jsonl(
        report_path,
        (
            {
                "motion_id": row[0],
                "official_split": row[1],
                "mirror_base_id": row[2],
                "canonical_split": row[3],
                "reason": row[4],
            }
            for row in connection.execute(
                """
                SELECT motion_id, split, mirror_base_id, canonical_split, reason
                FROM split_exclusions ORDER BY mirror_base_id, split, motion_id
                """
            )
        ),
    )
    eligible_by_split = {
        split: int(
            connection.execute(
                "SELECT COUNT(*) FROM split_entries WHERE split = ?", (split,)
            ).fetchone()[0]
        )
        for split in SPLITS
    }
    return {
        "policy": "keep_canonical_original_split_and_quarantine_cross_split_variants",
        "leaking_base_group_count": len(leaking_bases),
        "excluded_entry_count": int(sum(excluded_by_split.values())),
        "excluded_by_official_split": {
            split: int(excluded_by_split[split]) for split in SPLITS
        },
        "retained_in_leaking_groups_by_split": {
            split: int(retained_in_leaking_groups[split]) for split in SPLITS
        },
        "eligible_by_split": eligible_by_split,
        "report_path": str(report_path),
    }


def prepare_metadata_database(
    raw_root: Path,
    output_root: Path,
    *,
    resume: bool,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    """构建或严格复用 split/text SQLite 元数据索引。"""
    split_tar = raw_root / "split.tar.gz"
    texts_tar = raw_root / "texts.tar.gz"
    if not split_tar.is_file() or not texts_tar.is_file():
        raise FileNotFoundError(
            f"缺少 metadata 阶段文件: {split_tar} 或 {texts_tar}"
        )
    source = {
        "split_tar": str(split_tar),
        "split_sha256": sha256_file(split_tar),
        "texts_tar": str(texts_tar),
        "texts_sha256": sha256_file(texts_tar),
    }
    database_path = output_root / "reports" / "metadata.sqlite3"
    if database_path.exists() and not resume:
        raise FileExistsError(f"元数据数据库已存在: {database_path}；请使用 --resume 或新输出目录")
    if database_path.exists():
        connection = _create_metadata_database(database_path)
        existing = _metadata_value(connection, "source")
        if existing != json.dumps(source, ensure_ascii=False, sort_keys=True):
            connection.close()
            raise MotionMillionError("已有 metadata.sqlite3 的 split/text fingerprint 不一致")
        counts = json.loads(_metadata_value(connection, "counts") or "{}")
        return connection, {"source": source, "counts": counts, "resumed": True}

    connection = _create_metadata_database(database_path)
    try:
        split_counts = _index_splits(connection, split_tar)
        mirror_exclusions = _quarantine_cross_split_mirrors(
            connection, output_root
        )
        text_counts = _index_texts(connection, texts_tar)
        missing_text = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM split_entries AS s
                LEFT JOIN text_entries AS t ON t.motion_id = s.motion_id
                WHERE t.motion_id IS NULL
                """
            ).fetchone()[0]
        )
        remaining_mirror_leakage = list(
            connection.execute(
                """
                SELECT mirror_base_id, GROUP_CONCAT(DISTINCT split), COUNT(*)
                FROM split_entries
                GROUP BY mirror_base_id
                HAVING COUNT(DISTINCT split) > 1
                LIMIT 100
                """
            )
        )
        if remaining_mirror_leakage:
            raise MotionMillionError(
                "镜像隔离后仍检测到 base ID 跨 split，前 100 条: "
                f"{remaining_mirror_leakage}"
            )
        counts = {
            "official_split": split_counts,
            "eligible_split": mirror_exclusions["eligible_by_split"],
            "mirror_cross_split_exclusions": mirror_exclusions,
            "text": text_counts,
            "split_entries_without_text": missing_text,
        }
        _set_metadata(connection, "source", source)
        _set_metadata(connection, "counts", counts)
        connection.commit()
        return connection, {"source": source, "counts": counts, "resumed": False}
    except Exception:
        connection.close()
        raise


def discover_motion_archives(raw_root: Path, archive_pattern: str | None = None) -> list[Path]:
    motion_root = raw_root / "motion_272rpr"
    if not motion_root.is_dir():
        raise FileNotFoundError(f"MotionMillion motion_272rpr 不存在: {motion_root}")
    archives = sorted(
        path
        for path in motion_root.rglob("*")
        if path.is_file() and (path.name.endswith(".tar.gz") or path.suffix == ".tar")
    )
    if not archives:
        raise MotionMillionError(f"{motion_root} 下没有 motion tar 归档")
    if archive_pattern:
        archives = [
            path
            for path in archives
            if fnmatch.fnmatch(path.relative_to(raw_root).as_posix(), archive_pattern)
        ]
        if not archives:
            raise MotionMillionError(f"没有归档匹配 --archive-pattern={archive_pattern!r}")
    return archives


def _load_motion_array(payload: bytes, member_name: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise MotionMillionError(f"无法读取 NumPy 动作 {member_name}: {exc}") from exc
    if isinstance(value, np.lib.npyio.NpzFile):
        keys = list(value.files)
        if len(keys) != 1:
            value.close()
            raise MotionMillionError(f"NPZ 动作必须只有一个数组: {member_name}, keys={keys}")
        array = np.asarray(value[keys[0]])
        value.close()
        return array
    return np.asarray(value)


def _load_existing_shards(
    output_root: Path,
    *,
    build_fingerprint: str,
    resume: bool,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    shard_meta: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    built_ids: set[str] = set()
    for split in SPLITS:
        meta_dir = output_root / "shards" / "motion" / split
        existing = sorted(meta_dir.glob("*.meta.json")) if meta_dir.is_dir() else []
        if existing and not resume:
            raise FileExistsError(f"已有 motion shard: {existing[0]}；请使用 --resume")
        for expected_index, meta_path in enumerate(existing):
            metadata = read_json(meta_path)
            if int(metadata.get("shard_id", -1)) != expected_index:
                raise MotionMillionError(f"已有 shard 编号不连续: {meta_path}")
            if metadata.get("build_fingerprint") != build_fingerprint:
                raise MotionMillionError(f"已有 shard build fingerprint 不一致: {meta_path}")
            shard_path = output_root / str(metadata["path"])
            if not shard_path.is_file() or sha256_file(shard_path) != metadata.get("sha256"):
                raise MotionMillionError(f"已有 shard 文件缺失或 SHA256 不一致: {shard_path}")
            records = safe_torch_load(shard_path)
            if not isinstance(records, list) or len(records) != int(metadata["record_count"]):
                raise MotionMillionError(f"已有 shard 无法重载: {shard_path}")
            for record in records:
                validate_motion_record(record)
                motion_id = str(record["motion_id"])
                if motion_id in built_ids:
                    raise MotionMillionError(f"已有 shard 重复 motion ID: {motion_id}")
                built_ids.add(motion_id)
            shard_meta[split].append(metadata)
    return shard_meta, built_ids


def _flush_motion_shard(
    *,
    split: str,
    records: list[dict[str, Any]],
    shard_id: int,
    output_root: Path,
    build_fingerprint: str,
) -> dict[str, Any]:
    for record in records:
        validate_motion_record(record)
        if record["split"] != split:
            raise MotionMillionError("待写 shard 中存在 split 不一致记录")
    relative = Path("shards") / "motion" / split / f"motionmillion_{split}_{shard_id:06d}.pth"
    path = output_root / relative
    meta_path = path.with_suffix(".meta.json")
    if path.exists() or meta_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 shard 状态: {path}")
    atomic_torch_save(records, path)
    reloaded = safe_torch_load(path)
    if not isinstance(reloaded, list) or len(reloaded) != len(records):
        raise MotionMillionError(f"新 shard 重载失败: {path}")
    rows = []
    for record_index, record in enumerate(reloaded):
        validate_motion_record(record)
        rows.append(
            {
                "motion_id": str(record["motion_id"]),
                "record_index": record_index,
                "frames": int(record["pose"].shape[0]),
                "caption_count": len(record["captions"]),
                "source_archive": str(record["source_archive"]),
                "source_member": str(record["source_member"]),
                "source_sha256": str(record["source_sha256"]),
            }
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "build_version": BUILD_VERSION,
        "build_fingerprint": build_fingerprint,
        "split": split,
        "shard_id": shard_id,
        "path": relative.as_posix(),
        "record_count": len(rows),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "records": rows,
    }
    atomic_write_json(meta_path, metadata)
    return metadata


def _build_fingerprint(raw_root: Path, metadata_report: Mapping[str, Any], args: argparse.Namespace) -> str:
    import hashlib

    download_manifest = raw_root / "download_manifest_full.json"
    resolved_revision = None
    if download_manifest.is_file():
        resolved_revision = read_json(download_manifest).get("resolved_revision")
    payload = {
        "build_version": BUILD_VERSION,
        "schema_version": SCHEMA_VERSION,
        "resolved_revision": resolved_revision,
        "split_sha256": metadata_report["source"]["split_sha256"],
        "texts_sha256": metadata_report["source"]["texts_sha256"],
        "fps": OFFICIAL_FPS,
        "min_frames": MIN_FRAMES,
        "max_frames": MAX_FRAMES,
        "records_per_shard": args.records_per_shard,
        "motion_frames": args.motion_frames,
        "limit": args.limit,
        "only_split": getattr(args, "only_split", None),
        "archive_pattern": getattr(args, "archive_pattern", None),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _records_from_shards(output_root: Path, shard_metadata: Sequence[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    for metadata in shard_metadata:
        records = safe_torch_load(output_root / str(metadata["path"]))
        if not isinstance(records, list):
            raise MotionMillionError(f"motion shard 不是 list: {metadata['path']}")
        yield from records


def _write_split_release(
    output_root: Path,
    split: str,
    shard_metadata: Sequence[Mapping[str, Any]],
    build_fingerprint: str,
    motion_frames: int,
) -> dict[str, Any]:
    total_records = sum(int(item["record_count"]) for item in shard_metadata)
    sample_count = total_records
    sample_index = np.empty(sample_count, dtype=SAMPLE_INDEX_DTYPE)
    cursor = 0
    total_frames = 0
    for shard_id, item in enumerate(shard_metadata):
        if int(item["shard_id"]) != shard_id:
            raise MotionMillionError(f"{split} shard_id 不连续")
        for row in item["records"]:
            frames = int(row["frames"])
            sample_index[cursor] = (
                shard_id,
                int(row["record_index"]),
                frames,
                0,
            )
            total_frames += frames
            cursor += 1
    if cursor != sample_count:
        raise MotionMillionError("sample index 内部计数不一致")
    index_relative = Path("indices") / f"{split}.npy"
    atomic_save_npy(output_root / index_relative, sample_index)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_version": BUILD_VERSION,
        "build_fingerprint": build_fingerprint,
        "split": split,
        "fps": OFFICIAL_FPS,
        "source_up_axis": "y",
        "motion_frames": motion_frames,
        "record_count": total_records,
        "sample_count": sample_count,
        # 时长只统计真正通过转换、校验并写入 shard 的原始有效帧；不会把
        # 60--119 帧样本训练时补齐到 120 的 padding 计入数据集时长。
        "total_frames": total_frames,
        "duration_seconds": total_frames / OFFICIAL_FPS,
        "duration_hours": total_frames / OFFICIAL_FPS / 3600.0,
        "sample_index_path": index_relative.as_posix(),
        "sample_index_dtype": SAMPLE_INDEX_DTYPE.descr,
        "shards": [
            {
                key: item[key]
                for key in (
                    "shard_id",
                    "path",
                    "record_count",
                    "size_bytes",
                    "sha256",
                )
            }
            for item in shard_metadata
        ],
    }
    atomic_write_json(output_root / "manifests" / f"{split}.json", manifest)
    return manifest


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """执行 MotionMillion metadata 闭环、流式转换和 release 发布。"""
    started = time.monotonic()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"MotionMillion raw root 不存在: {raw_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    connection, metadata_report = prepare_metadata_database(
        raw_root, output_root, resume=args.resume
    )
    if getattr(args, "metadata_only", False):
        eligible_motion_count = sum(
            int(value)
            for value in metadata_report["counts"]["eligible_split"].values()
        )
        eligible_caption_count = int(
            metadata_report["counts"]["text"].get("captions", 0)
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "mode": "metadata_only",
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "metadata": metadata_report,
            "space_budget": {
                "required_free_bytes": int(1.5 * 1024**4),
                "converted_motion_reserved_bytes": 100 * 1024**3,
                "dense_all_caption_upper_bound_bytes": (
                    eligible_caption_count * 150 * 1024 * 2
                ),
                "one_caption_per_motion_dense_reference_bytes": (
                    eligible_motion_count * 150 * 1024 * 2
                ),
                "eligible_motion_count": eligible_motion_count,
                "eligible_caption_count": eligible_caption_count,
                "embedding_storage": "compact_valid_tokens_with_offsets",
            },
        }
        download_manifest = raw_root / "download_manifest_metadata.json"
        if download_manifest.is_file():
            downloaded = read_json(download_manifest)
            report["download"] = {
                key: downloaded.get(key)
                for key in (
                    "repo_id",
                    "resolved_revision",
                    "remote_repository_file_count",
                    "remote_repository_total_bytes",
                    "remote_motion_file_count",
                    "remote_motion_total_bytes",
                )
            }
        atomic_write_json(output_root / "reports" / "metadata_audit.json", report)
        connection.close()
        return report
    build_fingerprint = _build_fingerprint(raw_root, metadata_report, args)
    fingerprint_path = output_root / "reports" / "build_contract.json"
    if fingerprint_path.is_file():
        existing_fingerprint = read_json(fingerprint_path).get("build_fingerprint")
        if existing_fingerprint != build_fingerprint:
            connection.close()
            raise MotionMillionError("已有 build contract 与本次参数不一致")
    else:
        atomic_write_json(
            fingerprint_path,
            {
                "schema_version": SCHEMA_VERSION,
                "build_version": BUILD_VERSION,
                "build_fingerprint": build_fingerprint,
                "raw_root": str(raw_root),
                "output_root": str(output_root),
                "metadata": metadata_report,
                "fps": OFFICIAL_FPS,
                "source_up_axis": "y",
                "min_frames": MIN_FRAMES,
                "max_frames": MAX_FRAMES,
                "motion_frames": args.motion_frames,
                "records_per_shard": args.records_per_shard,
                "limit": args.limit,
            },
        )

    shard_metadata, built_ids = _load_existing_shards(
        output_root,
        build_fingerprint=build_fingerprint,
        resume=args.resume,
    )
    for motion_id in sorted(built_ids):
        record = connection.execute(
            "SELECT split FROM split_entries WHERE motion_id = ?", (motion_id,)
        ).fetchone()
        if record is None:
            connection.close()
            raise MotionMillionError(f"已有 shard ID 不再属于官方 split: {motion_id}")
        connection.execute(
            "INSERT OR IGNORE INTO built_entries VALUES (?, ?, ?, ?)",
            (motion_id, str(record[0]), "resumed", "resumed"),
        )
    connection.commit()

    buffers: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    counters = Counter()
    archive_reports: list[dict[str, Any]] = []
    accepted_this_run = 0
    rate_window: deque[tuple[float, int]] = deque(maxlen=10)
    only_split = getattr(args, "only_split", None)
    if only_split is None:
        target_records = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM split_entries AS s
                JOIN text_entries AS t ON t.motion_id = s.motion_id
                """
            ).fetchone()[0]
        )
    else:
        target_records = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM split_entries AS s
                JOIN text_entries AS t ON t.motion_id = s.motion_id
                WHERE s.split = ?
                """,
                (only_split,),
            ).fetchone()[0]
        )

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        metadata = _flush_motion_shard(
            split=split,
            records=buffers[split],
            shard_id=len(shard_metadata[split]),
            output_root=output_root,
            build_fingerprint=build_fingerprint,
        )
        shard_metadata[split].append(metadata)
        buffers[split] = []

    try:
        archives = discover_motion_archives(
            raw_root, getattr(args, "archive_pattern", None)
        )
        for archive_index, archive_path in enumerate(archives):
            archive_started = time.monotonic()
            archive_relative = archive_path.relative_to(raw_root).as_posix()
            archive_hash = sha256_file(archive_path)
            archive_counter = Counter()
            with _open_tar(archive_path) as archive:
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith((".npy", ".npz")):
                        continue
                    if args.limit is not None and len(built_ids) + accepted_this_run >= args.limit:
                        break
                    archive_counter["motion_members"] += 1
                    motion_id: str | None = None
                    normalized_member = normalize_member_name(member.name)
                    try:
                        resolved = _resolve_candidate(
                            connection,
                            _archive_identifier_candidates(member.name, archive_relative),
                            require_text=True,
                        )
                        if resolved is None:
                            archive_counter["not_in_closed_release"] += 1
                            continue
                        motion_id, split = resolved
                        only_split = getattr(args, "only_split", None)
                        if only_split is not None and split != only_split:
                            archive_counter["other_split"] += 1
                            continue
                        if motion_id in built_ids:
                            archive_counter["resumed_skip"] += 1
                            continue
                        duplicate = connection.execute(
                            "SELECT 1 FROM built_entries WHERE motion_id = ?", (motion_id,)
                        ).fetchone()
                        if duplicate is not None:
                            raise MotionMillionError(f"多个归档成员解析为同一 ID: {motion_id}")
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO observed_entries
                            (motion_id, status, source_archive, source_member, error_type, error)
                            VALUES (?, 'observed', ?, ?, NULL, NULL)
                            """,
                            (motion_id, archive_relative, normalized_member),
                        )
                        payload = _read_tar_member(archive, member)
                        array = _load_motion_array(payload, member.name)
                        converted = recover_smpl_from_272(array)
                        text_row = connection.execute(
                            "SELECT captions_json, source_member, source_sha256 FROM text_entries WHERE motion_id = ?",
                            (motion_id,),
                        ).fetchone()
                        if text_row is None:
                            raise MotionMillionError("内部错误：已解析 ID 没有文本")
                        captions = json.loads(str(text_row[0]))
                        record = {
                            "motion_id": motion_id,
                            "pose": converted["pose"],
                            "trans": converted["trans"],
                            "beta": converted["beta"],
                            "captions": captions,
                            "fps": OFFICIAL_FPS,
                            "source_up_axis": "y",
                            "source_subset": _source_subset(archive_relative, motion_id),
                            "source_archive": archive_relative,
                            "source_archive_sha256": archive_hash,
                            "source_member": normalized_member,
                            "source_sha256": sha256_bytes(payload),
                            "source_text_member": str(text_row[1]),
                            "source_text_sha256": str(text_row[2]),
                            "split": split,
                        }
                        validate_motion_record(record)
                        buffers[split].append(record)
                        connection.execute(
                            "INSERT INTO built_entries VALUES (?, ?, ?, ?)",
                            (motion_id, split, archive_relative, record["source_member"]),
                        )
                        connection.execute(
                            "UPDATE observed_entries SET status = 'accepted' WHERE motion_id = ?",
                            (motion_id,),
                        )
                        accepted_this_run += 1
                        counters[f"accepted_{split}"] += 1
                        archive_counter["accepted"] += 1
                        if len(buffers[split]) >= args.records_per_shard:
                            flush(split)
                            connection.commit()
                        if accepted_this_run and accepted_this_run % args.progress_every == 0:
                            now = time.monotonic()
                            rate_window.append((now, accepted_this_run))
                            if len(rate_window) > 1:
                                rate = (rate_window[-1][1] - rate_window[0][1]) / max(
                                    rate_window[-1][0] - rate_window[0][0], 1.0e-9
                                )
                            else:
                                rate = accepted_this_run / max(now - started, 1.0e-9)
                            remaining = max(
                                target_records - len(built_ids) - accepted_this_run, 0
                            )
                            eta_seconds = remaining / max(rate, 1.0e-9)
                            print(
                                f"[MotionMillion] accepted={accepted_this_run}, "
                                f"records_per_second={rate:.3f}, "
                                f"eligible_remaining_upper_bound={remaining}, "
                                f"ETA={eta_seconds / 3600:.2f}h, "
                                f"archive={archive_index + 1}/{len(archives)}",
                                flush=True,
                            )
                    except MotionMillionFilteredError as exc:
                        if motion_id is not None:
                            connection.execute(
                                """
                                INSERT OR REPLACE INTO observed_entries
                                (motion_id, status, source_archive, source_member, error_type, error)
                                VALUES (?, 'rejected', ?, ?, ?, ?)
                                """,
                                (
                                    motion_id,
                                    archive_relative,
                                    normalized_member,
                                    type(exc).__name__,
                                    str(exc),
                                ),
                            )
                        counters["filtered_length"] += 1
                        if args.strict:
                            raise
                    except Exception as exc:
                        if motion_id is not None:
                            connection.execute(
                                """
                                INSERT OR REPLACE INTO observed_entries
                                (motion_id, status, source_archive, source_member, error_type, error)
                                VALUES (?, 'rejected', ?, ?, ?, ?)
                                """,
                                (
                                    motion_id,
                                    archive_relative,
                                    normalized_member,
                                    type(exc).__name__,
                                    str(exc),
                                ),
                            )
                        counters["rejected_error"] += 1
                        if args.strict:
                            raise MotionMillionError(
                                f"严格模式下转换失败: {archive_relative}!/{member.name}: {exc}"
                            ) from exc
            archive_reports.append(
                {
                    "path": archive_relative,
                    "sha256": archive_hash,
                    "size_bytes": archive_path.stat().st_size,
                    "elapsed_seconds": time.monotonic() - archive_started,
                    "counts": dict(archive_counter),
                }
            )
            # 归档边界强制落盘未满 512 条的尾 shard，并在数据库提交后原子
            # 更新恢复清单。这样不会把两个来源归档混进同一 shard；中断恢复时
            # 只复用已经有完整 meta、SHA256 和 build fingerprint 的产物。
            for split in SPLITS:
                flush(split)
            connection.commit()
            atomic_write_json(
                output_root / "conversion_progress.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "in_progress",
                    "build_fingerprint": build_fingerprint,
                    "completed_archive_count": len(archive_reports),
                    "total_archive_count": len(archives),
                    "accepted_record_count": len(built_ids) + accepted_this_run,
                    "archives": archive_reports,
                    "shards": shard_metadata,
                },
            )
            if args.limit is not None and len(built_ids) + accepted_this_run >= args.limit:
                break

        for split in SPLITS:
            flush(split)
        connection.commit()

        manifests = {
            split: _write_split_release(
                output_root,
                split,
                shard_metadata[split],
                build_fingerprint,
                args.motion_frames,
            )
            for split in SPLITS
        }
        rejected_query = connection.execute(
            """
            SELECT motion_id, source_archive, source_member, error_type, error
            FROM observed_entries WHERE status = 'rejected'
            ORDER BY motion_id
            """
        )
        atomic_write_jsonl(
            output_root / "reports" / "rejected_samples.jsonl",
            (
                {
                    "motion_id": row[0],
                    "source_archive": row[1],
                    "source_member": row[2],
                    "error_type": row[3],
                    "error": row[4],
                }
                for row in rejected_query
            ),
        )

        full_scope = (
            args.limit is None
            and getattr(args, "only_split", None) is None
            and getattr(args, "archive_pattern", None) is None
        )
        if full_scope:
            unavailable_query = connection.execute(
                """
                SELECT s.motion_id, s.split,
                       CASE WHEN t.motion_id IS NULL THEN 'missing_text' ELSE 'missing_motion' END
                FROM split_entries AS s
                LEFT JOIN text_entries AS t ON t.motion_id = s.motion_id
                LEFT JOIN built_entries AS b ON b.motion_id = s.motion_id
                LEFT JOIN observed_entries AS o ON o.motion_id = s.motion_id
                WHERE t.motion_id IS NULL
                   OR (b.motion_id IS NULL AND o.motion_id IS NULL)
                ORDER BY s.split, s.motion_id
                """
            )
        else:
            # pilot 只看一个归档/train/前 N 条，未观察到的动作不能被误报为
            # 官方缺失；只有 metadata 已能证明的 missing_text 属于 unavailable。
            unavailable_query = connection.execute(
                """
                SELECT s.motion_id, s.split, 'missing_text'
                FROM split_entries AS s
                LEFT JOIN text_entries AS t ON t.motion_id = s.motion_id
                WHERE t.motion_id IS NULL
                ORDER BY s.split, s.motion_id
                """
            )
        atomic_write_jsonl(
            output_root / "reports" / "unavailable_by_release.jsonl",
            (
                {"motion_id": row[0], "split": row[1], "reason": row[2]}
                for row in unavailable_query
            ),
        )
        if full_scope:
            unavailable_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM split_entries AS s
                    LEFT JOIN text_entries AS t ON t.motion_id = s.motion_id
                    LEFT JOIN built_entries AS b ON b.motion_id = s.motion_id
                    LEFT JOIN observed_entries AS o ON o.motion_id = s.motion_id
                    WHERE t.motion_id IS NULL
                       OR (b.motion_id IS NULL AND o.motion_id IS NULL)
                    """
                ).fetchone()[0]
            )
        else:
            unavailable_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM split_entries AS s
                    LEFT JOIN text_entries AS t ON t.motion_id = s.motion_id
                    WHERE t.motion_id IS NULL
                    """
                ).fetchone()[0]
            )
        unresolved_count = 0
        if not full_scope:
            unresolved_query = connection.execute(
                """
                SELECT s.motion_id, s.split, 'excluded_by_build_scope'
                FROM split_entries AS s
                JOIN text_entries AS t ON t.motion_id = s.motion_id
                LEFT JOIN built_entries AS b ON b.motion_id = s.motion_id
                LEFT JOIN observed_entries AS o ON o.motion_id = s.motion_id
                WHERE b.motion_id IS NULL AND o.motion_id IS NULL
                ORDER BY s.split, s.motion_id
                """
            )
            unresolved_path = output_root / "reports" / "unresolved_by_scope.jsonl"
            atomic_write_jsonl(
                unresolved_path,
                (
                    {"motion_id": row[0], "split": row[1], "reason": row[2]}
                    for row in unresolved_query
                ),
            )
            unresolved_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM split_entries AS s
                    JOIN text_entries AS t ON t.motion_id = s.motion_id
                    LEFT JOIN built_entries AS b ON b.motion_id = s.motion_id
                    LEFT JOIN observed_entries AS o ON o.motion_id = s.motion_id
                    WHERE b.motion_id IS NULL AND o.motion_id IS NULL
                    """
                ).fetchone()[0]
            )
        release = {
            "schema_version": SCHEMA_VERSION,
            "build_version": BUILD_VERSION,
            "build_fingerprint": build_fingerprint,
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "fps": OFFICIAL_FPS,
            "source_up_axis": "y",
            "motion_frames": args.motion_frames,
            "records_per_shard": args.records_per_shard,
            "only_split": getattr(args, "only_split", None),
            "archive_pattern": getattr(args, "archive_pattern", None),
            "metadata": metadata_report,
            "manifests": {
                split: {
                    "path": f"manifests/{split}.json",
                    "record_count": manifests[split]["record_count"],
                    "sample_count": manifests[split]["sample_count"],
                    "total_frames": manifests[split]["total_frames"],
                    "duration_seconds": manifests[split]["duration_seconds"],
                    "duration_hours": manifests[split]["duration_hours"],
                }
                for split in SPLITS
            },
            "total_frames": sum(manifests[split]["total_frames"] for split in SPLITS),
            "duration_seconds": sum(
                manifests[split]["duration_seconds"] for split in SPLITS
            ),
            "duration_hours": sum(manifests[split]["duration_hours"] for split in SPLITS),
            "accepted_this_run": accepted_this_run,
            "resumed_record_count": len(built_ids),
            "unavailable_by_release_count": unavailable_count,
            "unresolved_by_scope_count": unresolved_count,
            "rejected_count": int(
                connection.execute(
                    "SELECT COUNT(*) FROM observed_entries WHERE status = 'rejected'"
                ).fetchone()[0]
            ),
            "counters": dict(counters),
            "archives": archive_reports,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(output_root / "dataset_release.json", release)
        atomic_write_json(
            output_root / "conversion_progress.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "build_fingerprint": build_fingerprint,
                "completed_archive_count": len(archive_reports),
                "total_archive_count": len(archives),
                "accepted_record_count": sum(
                    manifest["record_count"] for manifest in manifests.values()
                ),
                "archives": archive_reports,
                "shards": shard_metadata,
            },
        )
        return release
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--records-per-shard", type=int, default=RECORDS_PER_SHARD)
    parser.add_argument("--motion-frames", type=int, default=120)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="只审计 split/text 与远端清单，不要求或读取 motion 归档",
    )
    parser.add_argument("--only-split", choices=SPLITS)
    parser.add_argument(
        "--archive-pattern",
        help="相对 raw root 的 fnmatch；pilot 可限定 motion_272rpr/MotionGV/*.tar.gz",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.records_per_shard <= 0 or args.motion_frames <= 0 or args.progress_every <= 0:
        raise SystemExit("records-per-shard、motion-frames、progress-every 必须为正数")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit 必须为正数")
    report = build_dataset(args)
    if args.metadata_only:
        print(
            "MotionMillion metadata audit complete: "
            f"report={args.output_root / 'reports' / 'metadata_audit.json'}"
        )
        return
    counts = report["manifests"]
    print(
        "MotionMillion GENMO build complete: "
        + ", ".join(f"{split}={counts[split]['record_count']}" for split in SPLITS)
    )


if __name__ == "__main__":
    main()
