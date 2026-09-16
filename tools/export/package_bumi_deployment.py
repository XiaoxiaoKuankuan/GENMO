#!/usr/bin/env python3
"""在完整 GENMO 仓库中发布无需 checkpoint 的 BUMI 音乐运行资产包。

本工具接收已经导出并验证的 ONNX、TensorRT engine、运动学和统计，核验
源 checkpoint 的 SHA256 后复制六项GENMO运行资产，生成相对路径清单，再调用部署端同一个
检查器验证跨文件契约。输出通过同一文件系统上的临时目录原子发布；不覆盖已有目录。
临时目录无论成功或失败都会自动清理。checkpoint 本身只在原仓库读取，不进入部署包。

本工具不导出网络、不构建 engine、不启动 Redis 或控制器。它只属于完整仓库的发布
流程，最小部署分支不需要保留本文件，也不需要训练框架或模型构造器。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gem.runtime.bumi_deployment_bundle import (  # noqa: E402
    BUMI_DEPLOYMENT_CONTRACT,
    BUMI_DEPLOYMENT_LEGACY_CONTRACT,
    file_sha256,
    load_bumi_deployment_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "onnx", "engine", "kinematics", "stats", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gmt-policy", type=Path, help="仅兼容旧v1打包；新部署不要传此参数")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    onnx = args.onnx.expanduser().resolve(strict=True)
    engine = args.engine.expanduser().resolve(strict=True)
    sources = {
        "onnx": onnx,
        "onnx_metadata": onnx.with_suffix(onnx.suffix + ".json").resolve(strict=True),
        "engine": engine,
        "engine_metadata": (engine.parent / "engine.json").resolve(strict=True),
        "kinematics": args.kinematics.expanduser().resolve(strict=True),
        "stats": args.stats.expanduser().resolve(strict=True),
    }
    if args.gmt_policy is not None:
        sources["gmt_policy"] = args.gmt_policy.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite deployment bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = {
        "onnx": "onnx",
        "onnx_metadata": "onnx",
        "engine": "engine",
        "engine_metadata": "engine",
        "kinematics": "assets",
        "stats": "assets",
        "gmt_policy": "gmt",
    }
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        ).strip()
    )
    with tempfile.TemporaryDirectory(prefix=".bumi-package-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        records = {}
        for name, source in sources.items():
            relative = Path(groups[name]) / source.name
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records[name] = {
                "path": relative.as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": file_sha256(target),
            }
        payload = {
            "contract_version": BUMI_DEPLOYMENT_LEGACY_CONTRACT
            if args.gmt_policy is not None else BUMI_DEPLOYMENT_CONTRACT,
            "source_checkpoint_sha256": file_sha256(checkpoint),
            "source_checkpoint_name": checkpoint.name,
            "source_git_commit": source_commit,
            "source_git_dirty": source_dirty,
            "assets": records,
        }
        manifest = staging / "deployment.json"
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        load_bumi_deployment_manifest(manifest)
        staging.rename(output)
    print(output / "deployment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
