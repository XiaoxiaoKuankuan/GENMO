#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""流式物化 MotionMillion 官方验证集 evaluator 所需的最小数据视图。

正式训练数据保持 tar 归档只读，不把 308 GB 原始包整体再解压。本工具读取 GENMO
转换 manifest 中已经闭环的 ``source_archive/source_member/source_sha256``，每个归档
只顺序扫描一次，仅把官方 val split 中符合当前官方 loader 资格（60--200 帧）的原始
272D 数组写入 ``official_evaluator/dataset/MotionMillion``。文本、split、mean/std 与
每条输入指纹同时冻结到 ``eligibility.json``，供 20-seed 评测严格复用。

小批链路检查可以显式限定来源归档与样本数，按 seed 和 motion_id 的 SHA256 排序
选取固定集合。此模式标记为 subset_smoke，不冒充完整验证集；默认仍准备完整集合。

这里的 200 帧上限不是训练过滤规则；它来自固定官方代码 commit 的
``dataset/dataset_TM_eval_motionmillion.py``。若未来官方代码改变资格规则，应升级本工具
schema，而不能在一次报告中静默更改样本集合。
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    MOTION_DIM,
    SCHEMA_VERSION,
    atomic_save_npy,
    atomic_write_json,
    normalize_member_name,
    read_json,
    safe_torch_load,
    sha256_bytes,
    sha256_file,
)

OFFICIAL_CODE_COMMIT = "8a2a7dfa66ecb6a1533d3d9cb49c743a697e1e1c"
OFFICIAL_EVAL_MIN_FRAMES = 60
OFFICIAL_EVAL_MAX_FRAMES = 200


def select_targets(
    targets: dict[str, dict[str, Any]],
    *,
    max_samples: int | None = None,
    selection_seed: int = 42,
    source_archives: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """确定性选择小批验证集合，显式记录范围，避免把有限来源检查当成正式评测。"""
    archives = sorted(set(source_archives or []))
    available = {row["source_archive"] for row in targets.values()}
    if set(archives) - available:
        raise ValueError(f"指定归档没有合格 val 样本: {sorted(set(archives) - available)}")
    candidates = {
        key: row for key, row in targets.items()
        if not archives or row["source_archive"] in archives
    }
    if max_samples is not None and not 1 <= max_samples <= len(candidates):
        raise ValueError(f"样本数必须在 1–{len(candidates)}，实际 {max_samples}")
    ids = sorted(candidates)
    if max_samples is not None:
        ids = sorted(ids, key=lambda key: (sha256_bytes(f"{selection_seed}:{key}".encode()), key))
        ids = sorted(ids[:max_samples])
    selection = {
        "scope": "subset_smoke" if archives or max_samples is not None else "full_validation",
        "selection_seed": selection_seed,
        "max_samples": max_samples,
        "source_archives": archives,
        "full_eligible_count": len(targets),
        "candidate_count": len(candidates),
        "selected_count": len(ids),
        "selected_ids_sha256": sha256_bytes(("\n".join(ids) + "\n").encode()),
    }
    return {key: candidates[key] for key in ids}, selection


def load_official_statistics(raw_root: Path, name: str) -> tuple[Path, np.ndarray]:
    """兼容官方代码视图和真实 HF release 的统计量路径，不重新估计或替换数值。"""
    candidates = [raw_root / "mean_std" / "vector_272" / name,
                  raw_root / "mean_std" / name.capitalize()]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"缺少官方 {name}: {candidates}")
    if len({sha256_file(path) for path in existing}) != 1:
        raise ValueError(f"两种路径的官方 {name} 指纹不一致: {existing}")
    array = np.load(existing[0], allow_pickle=False)
    if array.shape != (MOTION_DIM,) or not np.isfinite(array).all():
        raise ValueError(f"官方 {name} shape/finite 异常")
    if name == "std.npy" and np.any(array <= 0):
        raise ValueError("官方 std 必须全部为正数")
    return existing[0], array


def _atomic_write_text(path: Path, value: str, *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"拒绝覆盖 evaluator 文本: {path}")
        if path.read_text(encoding="utf-8") != value:
            raise ValueError(f"已有 evaluator 文本内容漂移: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_or_verify_array(path: Path, value: np.ndarray, *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"拒绝覆盖 evaluator 数组: {path}")
        existing = np.load(path, mmap_mode="r", allow_pickle=False)
        if existing.shape != value.shape or existing.dtype != value.dtype:
            raise ValueError(f"已有 evaluator 数组 shape/dtype 漂移: {path}")
        if not np.array_equal(existing, value):
            raise ValueError(f"已有 evaluator 数组内容漂移: {path}")
        return
    atomic_save_npy(path, value)


def _load_targets(motion_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = motion_root / "manifests" / "val.json"
    manifest = read_json(manifest_path)
    targets: dict[str, dict[str, Any]] = {}
    for shard in manifest["shards"]:
        shard_path = motion_root / str(shard["path"])
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"val motion shard SHA256 不一致: {shard_path}")
        records = safe_torch_load(shard_path)
        for record in records:
            frames = int(record["pose"].shape[0])
            if not OFFICIAL_EVAL_MIN_FRAMES <= frames <= OFFICIAL_EVAL_MAX_FRAMES:
                continue
            motion_id = normalize_member_name(str(record["motion_id"]))
            if motion_id in targets:
                raise ValueError(f"val manifest 重复 motion_id: {motion_id}")
            targets[motion_id] = {
                "motion_id": motion_id,
                "frames": frames,
                "captions": list(record["captions"]),
                "source_archive": str(record["source_archive"]),
                "source_member": normalize_member_name(str(record["source_member"])),
                "source_sha256": str(record["source_sha256"]),
            }
    if not targets:
        raise ValueError("官方 val 资格集合为空")
    return targets


def prepare_official_eval(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = Path(args.raw_root).expanduser().resolve()
    motion_root = Path(args.motion_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_root = output_root / "dataset" / "MotionMillion"
    eligibility_path = output_root / "eligibility.json"
    if eligibility_path.exists() and not args.resume:
        raise FileExistsError(f"已有 evaluator 数据身份: {eligibility_path}；请使用 --resume")

    statistics = {name: load_official_statistics(raw_root, name)
                  for name in ("mean.npy", "std.npy")}
    targets, selection = select_targets(
        _load_targets(motion_root),
        max_samples=getattr(args, "max_samples", None),
        selection_seed=getattr(args, "selection_seed", 42),
        source_archives=getattr(args, "source_archive", None),
    )
    if eligibility_path.exists():
        previous = read_json(eligibility_path)
        if previous.get("selection", selection) != selection:
            raise ValueError("已有 evaluator 选择协议与当前请求不同")
        if {row["motion_id"] for row in previous["records"]} != set(targets):
            raise ValueError("已有 evaluator 样本集合与当前请求不同")
    by_archive: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for target in targets.values():
        by_archive[target["source_archive"]][target["source_member"]] = target

    written: dict[str, dict[str, Any]] = {}
    for archive_relative, member_targets in sorted(by_archive.items()):
        archive_path = raw_root / archive_relative
        if not archive_path.is_file():
            raise FileNotFoundError(f"缺少来源归档: {archive_path}")
        remaining = set(member_targets)
        print(f"提取 {archive_relative}: {len(remaining)} 个 val 样本", flush=True)
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive:
                normalized = normalize_member_name(member.name)
                target = member_targets.get(normalized)
                if target is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"无法读取归档成员: {archive_relative}!/{normalized}")
                payload = handle.read()
                if sha256_bytes(payload) != target["source_sha256"]:
                    raise ValueError(f"来源 member SHA256 漂移: {archive_relative}!/{normalized}")
                loaded = np.load(io.BytesIO(payload), allow_pickle=False)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    keys = list(loaded.files)
                    if len(keys) != 1:
                        loaded.close()
                        raise ValueError(f"官方 NPZ 必须只有一个数组: {normalized}")
                    array = np.asarray(loaded[keys[0]])
                    loaded.close()
                else:
                    array = np.asarray(loaded)
                expected = (target["frames"], MOTION_DIM)
                if array.shape != expected or not np.isfinite(array).all():
                    raise ValueError(
                        f"官方 evaluator 动作 shape/finite 异常: {target['motion_id']}, "
                        f"expected={expected}, actual={array.shape}"
                    )
                relative = Path("motion_data") / "vector_272" / f"{target['motion_id']}.npy"
                destination = dataset_root / relative
                _save_or_verify_array(destination, array, resume=args.resume)
                text_relative = Path("texts") / f"{target['motion_id']}.txt"
                _atomic_write_text(
                    dataset_root / text_relative,
                    "\n".join(str(value) for value in target["captions"]) + "\n",
                    resume=args.resume,
                )
                written[target["motion_id"]] = {
                    **target,
                    "motion_path": relative.as_posix(),
                    "motion_sha256": sha256_file(destination),
                    "text_path": text_relative.as_posix(),
                    "text_sha256": sha256_file(dataset_root / text_relative),
                }
                remaining.discard(normalized)
                if not remaining:
                    break

    missing = sorted(set(targets) - set(written))
    if missing:
        raise ValueError(f"有 {len(missing)} 条官方 val member 未找到，前 20 条: {missing[:20]}")

    mean_std_rows = {}
    for name in ("mean.npy", "std.npy"):
        source, array = statistics[name]
        relative = Path("mean_std") / "vector_272" / name
        _save_or_verify_array(dataset_root / relative, array, resume=args.resume)
        mean_std_rows[name] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(dataset_root / relative),
            "source_path": str(source),
        }

    ordered = [written[motion_id] for motion_id in sorted(written)]
    split_relative = Path("split/version1/t2m_60_300/val.txt")
    _atomic_write_text(
        dataset_root / split_relative,
        "".join(f"{row['motion_id']}\n" for row in ordered),
        resume=args.resume,
    )
    motion_manifest_path = motion_root / "manifests" / "val.json"
    report = {
        "schema_version": SCHEMA_VERSION,
        "official_code_commit": OFFICIAL_CODE_COMMIT,
        "official_loader_source": "dataset/dataset_TM_eval_motionmillion.py",
        "selection": selection,
        "eligibility": {
            "split": "val",
            "min_frames_inclusive": OFFICIAL_EVAL_MIN_FRAMES,
            "max_frames_inclusive": OFFICIAL_EVAL_MAX_FRAMES,
            "record_count": len(ordered),
            "drop_last": True,
            "batch_size": 32,
        },
        "raw_root": str(raw_root),
        "motion_release_manifest": str(motion_manifest_path),
        "motion_release_manifest_sha256": sha256_file(motion_manifest_path),
        "dataset_root": str(dataset_root),
        "split_path": split_relative.as_posix(),
        "split_sha256": sha256_file(dataset_root / split_relative),
        "mean_std": mean_std_rows,
        "records": ordered,
    }
    atomic_write_json(eligibility_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, help="只准备固定数量的小批样本，标记为 subset_smoke")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--source-archive", action="append", help="限定 raw-root 下的相对归档路径，可重复")
    return parser


def main() -> None:
    report = prepare_official_eval(build_parser().parse_args())
    print(
        "MotionMillion official evaluator data prepared: "
        f"eligible_val={report['eligibility']['record_count']}"
    )


if __name__ == "__main__":
    main()
