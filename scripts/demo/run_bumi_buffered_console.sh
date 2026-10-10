#!/usr/bin/env bash
# 当前Stage1 s595000部署的完整缓存Stage1控制台，整段生成后只提交一个最终块。
# 模型与地址从部署根目录deployment.ini读取，仅对本次进程显式选择运行方式。
# 沿用三终端操作和原bumi>命令，不受runtime.mode=preview影响，不修改配置文件。
# 本入口不启动GMT/Redis，不构建引擎，不执行额外验收；额外参数透传给对应Python入口。
set -euo pipefail
BUMI_DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$BUMI_DEPLOY_ROOT/run.sh" buffered-console "$@"
