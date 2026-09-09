#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""冻结 MotionMillion 官方 evaluator 代码、权重、统计量和资格集合身份。

工具拒绝 dirty 官方代码 checkout，核对指定 commit，并对 evaluator checkpoint、
关键官方 Python 源码、mean/std 与 ``eligibility.json`` 计算 SHA256。最终 fingerprint
必须写入每个 seed 的量化报告，防止在 20 次运行间悄悄更换 evaluator 或验证样本。
工具不下载 Google Drive 权重，也不接触任何凭据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
)
from tools.eval.prepare_motionmillion_official_eval import (  # noqa: E402
    OFFICIAL_CODE_COMMIT,
)

OFFICIAL_SOURCE_FILES = (
    "models/evaluator_wrapper_motionmillion_rpr272.py",
    "dataset/dataset_TM_eval_motionmillion.py",
    "utils/eval_trans.py",
)


def _git(code_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(code_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def fingerprint_evaluator(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    code_root = Path(args.code_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    eligibility_path = Path(args.eligibility).expanduser().resolve()
    expected_checkpoint = root / "checkpoints" / "evaluator" / "epoch=199.ckpt"
    if checkpoint != expected_checkpoint:
        raise ValueError(
            f"官方 wrapper 固定读取 {expected_checkpoint}，不能指纹化其他 checkpoint"
        )
    commit = _git(code_root, "rev-parse", "HEAD")
    if commit != args.official_commit:
        raise ValueError(f"官方 evaluator commit 不一致: {commit} != {args.official_commit}")
    dirty = _git(code_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("官方 evaluator tracked 工作树非干净，拒绝生成身份")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少官方 evaluator checkpoint: {checkpoint}")
    eligibility = read_json(eligibility_path)
    dataset_root = Path(eligibility["dataset_root"])
    expected_dataset_root = root / "dataset" / "MotionMillion"
    if dataset_root != expected_dataset_root:
        raise ValueError(
            f"官方 wrapper 固定读取 {expected_dataset_root}，实际 eligibility={dataset_root}"
        )

    files = []
    for relative in OFFICIAL_SOURCE_FILES:
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"官方 evaluator 源码缺失: {path}")
        files.append({"role": "source", "path": str(path), "sha256": sha256_file(path)})
    files.append(
        {"role": "checkpoint", "path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    )
    files.append(
        {
            "role": "eligibility",
            "path": str(eligibility_path),
            "sha256": sha256_file(eligibility_path),
        }
    )
    for name, row in sorted(eligibility["mean_std"].items()):
        path = dataset_root / row["path"]
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ValueError(f"官方 {name} SHA256 与 eligibility 不一致")
        files.append({"role": name, "path": str(path), "sha256": digest})

    identity: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "official_repository": "https://github.com/VankouF/MotionMillion-Codes",
        "official_commit": commit,
        "root": str(root),
        "code_root": str(code_root),
        "dataset_root": str(dataset_root),
        "eligible_val_records": int(eligibility["eligibility"]["record_count"]),
        "files": files,
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity["evaluator_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    atomic_write_json(Path(args.output), identity)
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--official-commit", default=OFFICIAL_CODE_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    identity = fingerprint_evaluator(build_parser().parse_args())
    print(f"MotionMillion evaluator fingerprint={identity['evaluator_fingerprint']}")


if __name__ == "__main__":
    main()
