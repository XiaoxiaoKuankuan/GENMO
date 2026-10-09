#!/usr/bin/env bash
# 第二阶段性能分支的服务器1有限八卡验收入口，不调用正式一万轮启动器。
# 沿用既有训练启动脚本验证过的四项NCCL设置和PhysX动态库路径，避免直接torchrun
# 漏掉单机通信环境而误触服务器不可用的IB/GDR路径。仅允许下列显式有限验收入口，
# 包括固定旧rollout学习、GPU多环境采集/奖励/重放；各有独立墙钟上限。
# 输出路径由后续Python参数显式指定，Python入口在创建CUDA上下文前检查八卡空闲。
# 不删除正式数据、不修改已发布checkpoint；测试失败保持非零退出并保留诊断。
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${GENMO_PYTHON:-/home/user/liwei/GENMO/.venv/bin/python}"
TASK_MODE="${1:-}"
[[ $# -ge 1 ]] || { printf '需要模式：nccl | learning | saved-learning | sampling-graph | vector-collection | vector-replay | vector-rewards | replay | dual | kernels | journal\n' >&2; exit 2; }
shift
case "$TASK_MODE" in
  nccl) TASK_TOOL=check_stage10_eight_gpu_collectives.py; TASK_SECONDS=150 ;;
  learning) TASK_TOOL=profile_stage10_learning_v4.py; TASK_SECONDS=1800 ;;
  saved-learning) TASK_TOOL=verify_stage10_saved_learning.py; TASK_SECONDS=1800 ;;
  sampling-graph) TASK_TOOL=verify_stage10_sampling_graph.py; TASK_SECONDS=900 ;;
  vector-collection) TASK_TOOL=verify_stage10_vector_collection.py; TASK_SECONDS=1800 ;;
  vector-replay) TASK_TOOL=compare_stage10_cpu_gpu_replay.py; TASK_SECONDS=1200 ;;
  vector-rewards) TASK_TOOL=verify_stage10_vector_rewards.py; TASK_SECONDS=900 ;;
  replay) TASK_TOOL=replay_stage10_runtime_v4.py; TASK_SECONDS=2400 ;;
  dual) TASK_TOOL=profile_stage10_dual_collector.py; TASK_SECONDS=1800 ;;
  kernels) TASK_TOOL=profile_stage10_optional_kernels.py; TASK_SECONDS=900 ;;
  journal) TASK_TOOL=profile_stage10_journal_v4.py; TASK_SECONDS=300 ;;
  *) printf '未知测试模式：%s\n' "$TASK_MODE" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export LD_LIBRARY_PATH="/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -c 0
cd "$TASK_REPO_ROOT"
exec timeout --signal=TERM --kill-after=60 "$TASK_SECONDS" "$TASK_PYTHON" -B -m torch.distributed.run \
  --standalone --nproc_per_node=8 "tools/$TASK_TOOL" "$@"
