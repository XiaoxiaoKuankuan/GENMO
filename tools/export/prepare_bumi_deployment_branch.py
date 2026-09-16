#!/usr/bin/env python3
"""在独立 Git worktree 中裁剪 BUMI 部署分支，保留可审计的实际 Python 依赖闭包。

本工具只接受 deploy/bumi-music-only-gmt 分支的独立 worktree，要求受跟踪文件无未提交
改动。它从 Console、Bridge、检查器和部署回归测试出发，递归解析包内绝对/相对导入，
保留必要 __init__、许可证、记录、部署说明和环境锁；删除操作仅通过 git rm 针对明确
列出的受跟踪文件执行，不递归清理未跟踪模型、虚拟环境或用户结果，不修改原仓库。

默认输出候选清单；--apply 才进行用户已授权的最小分支裁剪。训练/导出依赖若仍落入
闭包则中止，避免生成表面精简但运行时依赖完整训练仓库的目录。仓库历史记录完整保留。
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_RUNTIME_INIT = '''# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI music-only 独立部署的运行时包入口。

部署入口显式导入所需的 BUMI 子模块，本文件不自动装载文本、人体、视频或多模态引擎，
避免这些功能把训练框架和模型构造器重新带入部署目录。完整仓库的包级公共接口保持原状；
此精简仅属于 deploy/bumi-music-only-gmt 分支。轨迹生成、后处理与通信仍复用原实现。
"""
'''
SEEDS = (
    "scripts/demo/demo_music_bumi_console.py",
    "scripts/demo/demo_bumi_gmt_bridge.py",
    "scripts/demo/check_bumi_deployment.py",
    "tests/bumi/test_bumi_deployment_bundle.py",
    "tests/bumi/test_bumi_online_deployment.py",
    "tests/bumi/test_gmt_policy_source.py",
    "tests/bumi/conftest.py",
)


def dependency_closure(root: Path) -> set[str]:
    """对项目内模块的导入做闭包；外部依赖由独立环境锁和干净环境执行检查覆盖。"""
    pending = [root / item for item in SEEDS]
    visited: set[Path] = set()

    def add_module(parts: list[str]) -> None:
        if not parts or parts[0] not in {"gem", "scripts", "tests", "tools"}:
            return
        candidate = root.joinpath(*parts)
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if path.is_file() and path not in visited:
                pending.append(path)

    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        visited.add(path)
        relative = path.relative_to(root)
        package = list(relative.parts[:-1])
        for count in range(1, len(package) + 1):
            init = root.joinpath(*package[:count], "__init__.py")
            if init.is_file() and init not in visited:
                pending.append(init)
        source = (
            DEPLOYMENT_RUNTIME_INIT
            if relative.as_posix() == "gem/runtime/__init__.py"
            else path.read_text(encoding="utf-8")
        )
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                base = package[: len(package) - node.level + 1] if node.level else []
                module = base + (node.module.split(".") if node.module else [])
                add_module(module)
                for alias in node.names:
                    if alias.name != "*":
                        add_module(module + alias.name.split("."))
    return {str(path.relative_to(root)) for path in visited}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    root = args.worktree.expanduser().resolve(strict=True)
    if root == ROOT or not (root / ".git").is_file():
        raise ValueError("target must be a separate Git worktree")

    def git(*command: str) -> str:
        return subprocess.check_output(["git", *command], cwd=root, text=True)

    if git("branch", "--show-current").strip() != "deploy/bumi-music-only-gmt":
        raise ValueError("target branch must be deploy/bumi-music-only-gmt")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("target has uncommitted tracked changes; preserve them before trimming")
    tracked = set(git("ls-files", "-z").rstrip("\0").split("\0"))
    code = dependency_closure(root)
    forbidden = (
        "gem/model/",
        "gem/network/",
        "gem/dataset/",
        "gem/data/",
        "tools/export/",
        "gem/runtime/bumi_music_onnx.py",
    )
    invalid = sorted(path for path in code if path.startswith(forbidden))
    if invalid:
        raise RuntimeError(f"training/export dependency remains: {invalid}")
    keep = code | {
        "README.md",
        ".gitignore",
        "AGENTS.md",
        "记录文本.md",
        "pyproject.toml",
        "docs/BUMI_MUSIC_DEPLOYMENT.md",
        "docs/BUMI_GMT_GENMO_INTERFACE.md",
        "docs/GENMO_CONTROLLER_ADAPTATION.md",
    }
    keep.update(
        path
        for path in tracked
        if path.startswith("requirements/deployment/")
        or Path(path).name.upper().startswith(("LICENSE", "NOTICE", "COPYING"))
    )
    remove = sorted(tracked - keep)
    report = {
        "description": "BUMI music-only 部署分支依赖闭包；模型文件不纳入Git",
        "source_commit": git("rev-parse", "HEAD").strip(),
        "python_files": sorted(code),
        "retained_files": sorted(keep & tracked),
        "removed_tracked_count": len(remove),
        "replaced_initializers": ["gem/runtime/__init__.py"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.apply:
        for start in range(0, len(remove), 100):
            subprocess.run(
                ["git", "rm", "-q", "--", *remove[start : start + 100]], cwd=root, check=True
            )
        (root / "gem/runtime/__init__.py").write_text(DEPLOYMENT_RUNTIME_INIT, encoding="utf-8")
        (root / ".gitignore").write_text(
            ".venv/\nmodels/\noutputs/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n.ruff_cache/\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text(
            "# BUMI music-only GENMO＋GMT 部署\n\n"
            "本分支只运行原 GENMO 仓库导出的 BUMI 模型。模型资产位于 `models/bumi_v5_s350000/`，"
            "不包含训练 checkpoint、模型导出或训练代码。\n\n"
            "- [环境安装、模型检查和三个终端启动](docs/BUMI_MUSIC_DEPLOYMENT.md)\n"
            "- [GMT 中直接用于 GENMO 接入的改动](docs/BUMI_GMT_GENMO_INTERFACE.md)\n"
            "- [适配其他 GMT、SONIC 与通用控制器](docs/GENMO_CONTROLLER_ADAPTATION.md)\n"
            "- [依赖闭包清单](DEPLOYMENT_FILES.json)\n"
            "- [实现与实际验收记录](记录文本.md)\n\n"
            "默认 DDIM20、CFG2.5、seed42，生成30 Hz、GMT参考50 Hz。"
            "迁移时复制代码和完整模型目录，重新创建环境并运行模型检查；"
            "目标GPU或TensorRT环境不兼容时在原仓库重新构建engine。\n",
            encoding="utf-8",
        )
        (root / "DEPLOYMENT_FILES.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
