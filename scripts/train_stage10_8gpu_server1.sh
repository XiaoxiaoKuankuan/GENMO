#!/usr/bin/env bash
# 第十步服务器1八卡共同训练启动脚本：八个进程使用同一套Actor/Critic参数和同步梯度。
# rank0从完整训练池采样并独占冻结GMT/CPU PhysX采集，八卡分担本轮优化，rank0写唯一
# 运行账本和checkpoint；这不是八个独立实验，也不承诺八个物理环境的采集加速。
# 为便于先验证正常启动，默认只运行到第1个接受更新；持续首段需显式传入
# --stop-after-iteration N。续训使用同一目录加--resume latest，恢复已发布checkpoint，
# 同步模型、优化器动量和步数；采样器、BC、随机状态和累计预算由rank0恢复。
# GENMO_PYTHON、CUDA_VISIBLE_DEVICES和STAGE10_8GPU_CONFIG可以覆盖解释器、八张卡
# 和显式配置路径；配置内模型、数据及GMT资产仍使用服务器1已经核验的正式路径。
# NCCL设置沿用本机单节点八卡通信条件；PhysX动态库路径仅在此进程树中补充，不改系统。
# 本脚本不删除运行产物，启动验收应由调用方使用明确临时目录并在核验后精确清理。
set -euo pipefail

TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${GENMO_PYTHON:-/home/user/liwei/GENMO/.venv/bin/python}"
TASK_CONFIG="${STAGE10_8GPU_CONFIG:-$TASK_REPO_ROOT/configs/closedloop/stage10_8gpu_server1.yaml}"
TASK_OUTPUT=""
TASK_STOP=1
TASK_RESUME=""

usage() {
  cat <<'EOF'
用法：bash scripts/train_stage10_8gpu_server1.sh --output-dir 目录 [--stop-after-iteration 1] [--resume latest]
  --output-dir PATH           同一8卡作业唯一输出目录；新run用新目录，恢复用原目录。
  --stop-after-iteration N     在第N个接受更新后停止，默认1；上限由所选配置决定。
  --resume latest|PATH        从同一run最新发布checkpoint恢复，继续累计预算。
  -h, --help                 显示帮助，不启动训练。
默认1轮用于启动验收，续训时显式指定更高停止轮次；配置中的运行截止时间优先生效。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
      TASK_OUTPUT="$2"
      shift 2
      ;;
    --stop-after-iteration)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      TASK_STOP="$2"
      shift 2
      ;;
    --resume)
      [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
      TASK_RESUME="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) printf '不支持的参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$TASK_OUTPUT" ]] || { usage >&2; exit 2; }
[[ "$TASK_STOP" =~ ^[1-9][0-9]*$ ]] || { printf '停止轮次必须是正整数。\n' >&2; exit 2; }
[[ -x "$TASK_PYTHON" ]] || { printf 'Python不可执行：%s\n' "$TASK_PYTHON" >&2; exit 2; }
[[ -f "$TASK_CONFIG" ]] || { printf '配置不存在：%s\n' "$TASK_CONFIG" >&2; exit 2; }

# 在切换到仓库前解析相对输出路径，避免用户指定目录随工作目录变化。
TASK_OUTPUT="$(realpath -m -- "$TASK_OUTPUT")"
if [[ -n "$TASK_RESUME" && "$TASK_RESUME" != latest ]]; then
  TASK_RESUME="$(realpath -m -- "$TASK_RESUME")"
fi
TASK_RESUME_ARGS=()
if [[ -n "$TASK_RESUME" ]]; then
  TASK_RESUME_ARGS=(--resume "$TASK_RESUME")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a TASK_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#TASK_GPU_IDS[@]} -eq 8 ]] || { printf 'CUDA_VISIBLE_DEVICES必须列出8张GPU。\n' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export LD_LIBRARY_PATH="/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$TASK_REPO_ROOT"
printf '八卡共同更新同一模型；rank0单GMT采集\n配置=%s\n输出=%s\n停止轮次=%s\nGPU=%s\n' \
  "$TASK_CONFIG" "$TASK_OUTPUT" "$TASK_STOP" "$CUDA_VISIBLE_DEVICES"
exec "$TASK_PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train_closedloop_stage10_8gpu.py --config "$TASK_CONFIG" \
  --output-dir "$TASK_OUTPUT" --stop-after-iteration "$TASK_STOP" "${TASK_RESUME_ARGS[@]}"
