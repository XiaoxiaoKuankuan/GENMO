#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""分阶段、可恢复地下载并审计官方 MotionMillion gated 数据集。

脚本只从 Hugging Face 官方 ``InternRobotics/MotionMillion`` 数据仓库读取文件，先
解析不可变 commit revision 和远端文件树，再按 metadata/full 两阶段调用
``snapshot_download``。访问令牌只能由 ``hf auth login`` 或 ``HF_TOKEN`` 提供，
命令行没有 token 参数，避免凭据进入 shell history、进程列表或项目日志。

每次执行都会原子更新下载 manifest，记录远端大小、LFS/Xet 标识、本地大小与
SHA256。full 阶段默认要求目标文件系统至少保留 1.5 TiB 可用空间，并逐个校验已下载
归档；脚本不会解压数据，也不会启动后续转换或训练。
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    MotionMillionError,
    atomic_write_json,
    read_json,
    sha256_file,
)

DEFAULT_REPO_ID = "InternRobotics/MotionMillion"
DEFAULT_OUTPUT_ROOT = Path("/data0/user/liwei/datasets/MotionMillion/raw_hf")
MIN_FREE_TIB = 1.5

METADATA_PATTERNS = [
    "README.md",
    "split.tar.gz",
    "texts.tar.gz",
    "mean_std/**",
    "data_process/**",
    ".gitattributes",
]
FULL_PATTERNS = [*METADATA_PATTERNS, "motion_272rpr/**"]


def _remote_file_row(entry: Any) -> dict[str, Any] | None:
    """把 huggingface_hub 的 repo tree entry 规整为稳定 JSON 行。"""
    path = getattr(entry, "path", None)
    entry_type = getattr(entry, "type", None)
    if not isinstance(path, str) or entry_type not in {"file", None}:
        return None
    size = getattr(entry, "size", None)
    lfs = getattr(entry, "lfs", None)
    lfs_oid = getattr(lfs, "sha256", None) if lfs is not None else None
    if lfs_oid is None and isinstance(lfs, dict):
        lfs_oid = lfs.get("sha256") or lfs.get("oid")
    return {
        "path": path,
        "remote_size_bytes": int(size) if size is not None else None,
        "remote_blob_id": getattr(entry, "blob_id", None),
        "lfs_sha256": lfs_oid,
    }


def _matches_stage(path: str, stage: str) -> bool:
    """对远端清单使用与 snapshot allow-pattern 等价的前缀判断。"""
    if path in {"README.md", "split.tar.gz", "texts.tar.gz", ".gitattributes"}:
        return True
    if path.startswith("mean_std/") or path.startswith("data_process/"):
        return True
    return stage == "full" and path.startswith("motion_272rpr/")


def _check_free_space(path: Path, minimum_tib: float) -> dict[str, int]:
    """返回磁盘容量并在空间不足时阻断 full 下载。"""
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    minimum = int(minimum_tib * 1024**4)
    if usage.free < minimum:
        raise MotionMillionError(
            f"目标文件系统可用 {usage.free / 1024**4:.3f} TiB，"
            f"低于要求的 {minimum_tib:.3f} TiB"
        )
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _validate_archive(path: Path) -> dict[str, int | bool]:
    """完整遍历 tar header，确认归档可读且没有不安全成员路径。"""
    if not (path.name.endswith(".tar.gz") or path.name.endswith(".tgz") or path.suffix == ".tar"):
        return {"tar_checked": False, "tar_members": 0}
    members = 0
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                candidate = Path(member.name)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise MotionMillionError(f"归档包含不安全路径: {path}!/{member.name}")
                members += 1
    except (tarfile.TarError, OSError) as exc:
        raise MotionMillionError(f"tar 完整性检查失败: {path}: {exc}") from exc
    return {"tar_checked": True, "tar_members": members}


def _verify_downloaded_row(local_root: Path, row: dict[str, Any], *, skip_sha256: bool) -> None:
    """下载一个文件后立即完成大小、LFS/SHA256 与 tar 完整性检查。"""
    local_path = local_root / row["path"]
    if not local_path.is_file():
        raise MotionMillionError(f"下载结束后缺少文件: {row['path']}")
    row["local_size_bytes"] = local_path.stat().st_size
    expected = row.get("remote_size_bytes")
    if expected is not None and int(expected) != int(row["local_size_bytes"]):
        raise MotionMillionError(
            f"下载文件大小不一致: {row['path']}，远端 {expected}，本地 {row['local_size_bytes']}"
        )
    row["sha256"] = None if skip_sha256 else sha256_file(local_path)
    lfs_sha = row.get("lfs_sha256")
    if lfs_sha and row["sha256"] and str(lfs_sha).removeprefix("sha256:") != row["sha256"]:
        raise MotionMillionError(f"LFS SHA256 不一致: {row['path']}")
    row.update(_validate_archive(local_path))


def download_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """解析官方 revision，分阶段下载并生成可追溯 manifest。"""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "缺少 huggingface_hub；请先在 GENMO 环境安装与服务器兼容的版本"
        ) from exc

    output_root = Path(args.output_root).expanduser().resolve()
    disk = _check_free_space(output_root, args.minimum_free_tib if args.stage == "full" else 0)
    api = HfApi()
    try:
        info = api.dataset_info(args.repo_id, revision=args.revision)
    except Exception as exc:
        raise MotionMillionError(
            "无法访问 MotionMillion。请先由用户本人接受 gated 协议，再执行 hf auth login；"
            f"原始错误: {exc}"
        ) from exc
    resolved_revision = str(info.sha)
    metadata_manifest = output_root / "download_manifest_metadata.json"
    if args.stage == "full" and metadata_manifest.is_file():
        previous = read_json(metadata_manifest)
        if previous.get("repo_id") != args.repo_id:
            raise MotionMillionError("已有 metadata manifest 的 repo_id 与本次 full 下载不一致")
        if previous.get("resolved_revision") != resolved_revision:
            raise MotionMillionError(
                "metadata/full 解析到不同 revision，拒绝混用；请用 metadata manifest 中的 "
                "resolved_revision 重新执行 full"
            )
    if args.stage == "full":
        for state_path in (
            output_root / "download_progress_full.json",
            output_root / "download_manifest_full.json",
        ):
            if not state_path.is_file():
                continue
            previous = read_json(state_path)
            expected = {
                "repo_id": args.repo_id,
                "resolved_revision": resolved_revision,
            }
            actual = {key: previous.get(key) for key in expected}
            if actual != expected:
                raise MotionMillionError(
                    f"已有 {state_path.name} 与本次 full 下载身份不一致，拒绝混用: "
                    f"expected={expected}, actual={actual}"
                )

    remote_rows: list[dict[str, Any]] = []
    repository_rows: list[dict[str, Any]] = []
    try:
        for entry in api.list_repo_tree(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            recursive=True,
            expand=True,
        ):
            row = _remote_file_row(entry)
            if row is not None:
                repository_rows.append(dict(row))
                selected = _matches_stage(row["path"], args.stage)
                if (
                    selected
                    and args.stage == "full"
                    and args.motion_pattern
                    and row["path"].startswith("motion_272rpr/")
                ):
                    selected = fnmatch.fnmatch(row["path"], args.motion_pattern)
                if selected:
                    remote_rows.append(row)
    except Exception as exc:
        raise MotionMillionError(f"读取官方文件树失败: {exc}") from exc
    remote_rows.sort(key=lambda row: row["path"])
    if not remote_rows:
        raise MotionMillionError("官方文件树中没有匹配当前下载阶段的文件")

    started = time.monotonic()
    try:
        # metadata 小文件先形成闭环；full 阶段再逐个 motion 归档下载、立即校验，
        # 因而中断后 Hugging Face cache 可以恢复，且损坏归档不会流入转换器。
        snapshot_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            local_dir=output_root,
            allow_patterns=METADATA_PATTERNS,
        )
        local_root = Path(snapshot_path).resolve()
        if args.stage == "full":
            motion_rows = [row for row in remote_rows if row["path"].startswith("motion_272rpr/")]
            completed_motion_bytes = 0
            motion_started = time.monotonic()
            for index, row in enumerate(motion_rows, start=1):
                file_started = time.monotonic()
                snapshot_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    revision=resolved_revision,
                    local_dir=output_root,
                    allow_patterns=[row["path"]],
                )
                _verify_downloaded_row(local_root, row, skip_sha256=args.skip_sha256)
                elapsed = max(time.monotonic() - file_started, 1.0e-9)
                speed = int(row["local_size_bytes"]) / elapsed / 1024**2
                completed_motion_bytes += int(row["local_size_bytes"])
                aggregate_rate = completed_motion_bytes / max(
                    time.monotonic() - motion_started, 1.0e-9
                )
                remaining_bytes = sum(
                    int(value["remote_size_bytes"] or 0) for value in motion_rows[index:]
                )
                eta_seconds = remaining_bytes / max(aggregate_rate, 1.0e-9)
                # 每个归档验证通过后立即原子更新进度清单。即使进程随后中断，
                # 恢复任务也能区分“已校验归档”和 HF cache 中尚未完成的文件。
                atomic_write_json(
                    output_root / "download_progress_full.json",
                    {
                        "schema_version": 1,
                        "status": "in_progress",
                        "repo_id": args.repo_id,
                        "resolved_revision": resolved_revision,
                        "motion_pattern": args.motion_pattern,
                        "completed_motion_file_count": index,
                        "total_motion_file_count": len(motion_rows),
                        "completed_motion_bytes": completed_motion_bytes,
                        "files": motion_rows[:index],
                    },
                )
                print(
                    f"[MotionMillion download] archive={index}/{len(motion_rows)}, "
                    f"MiB/s={speed:.2f}, ETA={eta_seconds / 3600:.2f}h, "
                    f"path={row['path']}",
                    flush=True,
                )
    except Exception as exc:
        if isinstance(exc, MotionMillionError):
            raise
        raise MotionMillionError(f"MotionMillion 下载失败或未完成: {exc}") from exc

    for row in remote_rows:
        if "local_size_bytes" not in row:
            _verify_downloaded_row(local_root, row, skip_sha256=args.skip_sha256)

    manifest = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "stage": args.stage,
        "local_root": str(local_root),
        "disk_before_download": disk,
        "minimum_free_tib": args.minimum_free_tib if args.stage == "full" else 0,
        "motion_pattern": args.motion_pattern,
        "sha256_verified": not args.skip_sha256,
        "file_count": len(remote_rows),
        "remote_total_bytes": sum(
            int(row["remote_size_bytes"] or 0) for row in remote_rows
        ),
        "elapsed_seconds": time.monotonic() - started,
        "files": remote_rows,
        "remote_repository_file_count": len(repository_rows),
        "remote_repository_total_bytes": sum(
            int(row["remote_size_bytes"] or 0) for row in repository_rows
        ),
        "remote_motion_file_count": sum(
            row["path"].startswith("motion_272rpr/") for row in repository_rows
        ),
        "remote_motion_total_bytes": sum(
            int(row["remote_size_bytes"] or 0)
            for row in repository_rows
            if row["path"].startswith("motion_272rpr/")
        ),
    }
    atomic_write_json(output_root / f"download_manifest_{args.stage}.json", manifest)
    if args.stage == "full":
        completed_motion_rows = [
            row for row in remote_rows if row["path"].startswith("motion_272rpr/")
        ]
        atomic_write_json(
            output_root / "download_progress_full.json",
            {
                "schema_version": 1,
                "status": "complete",
                "repo_id": args.repo_id,
                "resolved_revision": resolved_revision,
                "motion_pattern": args.motion_pattern,
                "completed_motion_file_count": len(completed_motion_rows),
                "total_motion_file_count": len(completed_motion_rows),
                "completed_motion_bytes": sum(
                    int(row["local_size_bytes"]) for row in completed_motion_rows
                ),
                "files": completed_motion_rows,
            },
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        default="main",
        help="首次可用 main，manifest 会记录解析后的不可变 commit SHA",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=("metadata", "full"), default="metadata")
    parser.add_argument("--minimum-free-tib", type=float, default=MIN_FREE_TIB)
    parser.add_argument(
        "--motion-pattern",
        help="full 阶段可用 fnmatch 限定 pilot 归档；正式全量不设置",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="仅用于快速清单诊断；正式 full release 禁止使用",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "full" and args.skip_sha256:
        raise SystemExit("正式 full 下载不允许 --skip-sha256")
    report = download_dataset(args)
    print(
        f"MotionMillion download {report['stage']} complete: "
        f"files={report['file_count']}, revision={report['resolved_revision']}"
    )


if __name__ == "__main__":
    main()
