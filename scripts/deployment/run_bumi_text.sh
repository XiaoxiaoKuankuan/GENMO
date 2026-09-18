#!/usr/bin/env bash
# BUMI文本部署包启动入口：显式调用包内虚拟环境，不要求activate或进入uv shell。
# 本文件在打包后位于根目录run.sh；额外参数传给同一常驻控制台。
set -euo pipefail
deployment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$deployment_root"
if [[ ! -x .venv/bin/python || ! -f deployment.json ]]; then
    echo '请在生成的部署包内先运行 bash install.sh。' >&2
    exit 1
fi
exec .venv/bin/python -B -u -m gem.runtime.bumi_text_launcher "$@"
