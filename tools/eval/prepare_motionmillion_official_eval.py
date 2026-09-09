#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""流式物化 MotionMillion 官方验证集 evaluator 所需的最小数据视图。

正式训练数据保持 tar 归档只读，不把 308 GB 原始包整体再解压。本工具读取 GENMO
转换 manifest 中已经闭环的 ``source_archive/source_member/source_sha256``，每个归档
只顺序扫描一次，仅把官方 val split 中符合当前官方 loader 资格（60--200 帧）的原始
272D 数组写入 ``official_evaluator/dataset/MotionMillion``。文本、split、mean/std 与
每条输入指纹同时冻结到 ``eligibility.json``，供 20-seed 评测严格复用。

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

    targets = _load_targets(motion_root)
    by_archive: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for target in targets.values():
        by_archive[target["source_archive"]][target["source_member"]] = target

    written: dict[str, dict[str, Any]] = {}
    for archive_relative, member_targets in sorted(by_archive.items()):
        archive_path = raw_root / archive_relative
        if not archive_path.is_file():
            raise FileNotFoundError(f"缺少来源归档: {archive_path}")
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

    missing = sorted(set(targets) - set(written))
    if missing:
        raise ValueError(f"有 {len(missing)} 条官方 val member 未找到，前 20 条: {missing[:20]}")

    mean_std_rows = {}
    for name in ("mean.npy", "std.npy"):
        source = raw_root / "mean_std" / "vector_272" / name
        if not source.is_file():
            raise FileNotFoundError(f"缺少官方 mean/std: {source}")
        array = np.load(source, allow_pickle=False)
        if array.shape != (MOTION_DIM,) or not np.isfinite(array).all():
            raise ValueError(f"官方 {name} shape/finite 异常")
        relative = Path("mean_std") / "vector_272" / name
        _save_or_verify_array(dataset_root / relative, array, resume=args.resume)
        mean_std_rows[name] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(dataset_root / relative),
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
    return parser


def main() -> None:
    report = prepare_official_eval(build_parser().parse_args())
    print(
        "MotionMillion official evaluator data prepared: "
        f"eligible_val={report['eligibility']['record_count']}"
    )


if __name__ == "__main__":
    main()
