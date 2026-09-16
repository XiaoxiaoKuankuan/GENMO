#!/usr/bin/env bash
# BUMI 部署统一启动入口：bash run.sh bridge / genmo / check / check-gmt / show-config。
# 所有模型、GPU、端口和容器设置从同目录 deployment.ini 读取，不在终端拼接路径参数。
# 使用部署目录自身的虚拟环境，保持进程信号和交互终端，不启动或修改 GMT 控制器。
set -euo pipefail
BUMI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$BUMI_ROOT/.venv/bin/python" ]]; then
    echo '尚未创建运行环境，请先执行 bash install.sh。' >&2
    exit 1
fi
exec "$BUMI_ROOT/.venv/bin/python" -B -u "$BUMI_ROOT/scripts/demo/run_bumi_deployment.py" "$@"
