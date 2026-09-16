#!/usr/bin/env python3
"""使用 deployment.ini 启动既有 BUMI GENMO、Bridge 和检查器。

本入口为 run.sh 提供 Python 调度：读取统一持久配置，把配置转换为原脚本的参数，
然后通过 exec 替换当前进程，保留 Ctrl-C、Console 标准输入和原脚本退出码。它不启动
GMT、Redis、仿真或实机，不修改 policy。show-config 只打印生效配置；check 仅执行
模型安装验收；check-gmt 才只读在线 GMT 的 ROS 参数和策略契约。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gem.runtime.bumi_deployment_config import (  # noqa: E402
    deployment_command,
    load_deployment_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("genmo", "bridge", "check", "check-gmt", "show-config"))
    parser.add_argument("--config", type=Path, default=ROOT / "deployment.ini")
    args = parser.parse_args(argv)
    config = load_deployment_config(args.config)
    if args.role == "show-config":
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str))
        return 0
    script, options = deployment_command(config, args.role)
    print(f"[部署配置] {config.path}", flush=True)
    os.execv(
        sys.executable, [sys.executable, "-B", "-u", str(ROOT / "scripts/demo" / script), *options]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
