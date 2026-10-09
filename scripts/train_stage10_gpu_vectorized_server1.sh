#!/usr/bin/env bash
# 服务器1 GPU向量化第二阶段入口：8卡、每卡8环境、全局160条真实转移。
# 复用已维护的八卡进程、空闲检查和NCCL设置，仅选择独立GPU配置；原CPU入口不变。
# 默认只运行1个接受轮次，完成全部正确性/吞吐验收前不自动开启10000轮。
# 新run必须用新输出目录；恢复只接受本配置自身完整checkpoint，不读取有限测试产物。
# 使用：bash scripts/train_stage10_gpu_vectorized_server1.sh --output-dir 新目录 --stop-after-iteration 1
set -euo pipefail
TASK_VECTOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export STAGE10_8GPU_CONFIG="${STAGE10_8GPU_CONFIG:-$TASK_VECTOR_ROOT/configs/closedloop/stage10_8gpu_server1_gpu_vectorized.yaml}"
exec bash "$TASK_VECTOR_ROOT/scripts/train_stage10_8gpu_server1.sh" "$@"
