#!/usr/bin/env bash
# Stage10 的八卡统一运行环境入口。
# 本脚本在Python/Kit启动前设置Isaac动态库和单机NCCL环境，防止只在进程内部
# 修改LD_LIBRARY_PATH而无法影响动态链接器。默认训练边界仍由Python入口限定
# 为两个Actor外层轮次；配置中的长期预算不会自动启动。必须显式传入config和独立output-dir。
# 正式RoboFFT使用stage2_continuous_rolling_robofft.yaml的已标定副本；旧DPPO只作显式独立对照。
# 当前第二阶段统一从本脚本启动；不覆盖已有模型、日志，不抢占已有GPU作业。
# 需要统一Python3.11环境继承Isaac依赖并安装支持RTX 6000D的Torch/Triton。
set -euo pipefail
# 在启动八个Python进程前拒绝遗漏配置；具体路径与算法合同由Python入口统一校验。
task_has_config=false
task_help=false
for task_argument in "$@"; do
  case "$task_argument" in
    --config|--config=*) task_has_config=true ;;
    --help|-h) task_help=true ;;
  esac
done
if [[ "$task_has_config" == false && "$task_help" == false ]]; then
  echo '必须显式传入 --config；正式 RoboFFT 使用 stage2_continuous_rolling_robofft.yaml 的已标定副本。' >&2
  exit 2
fi
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${STAGE10_PYTHON:-/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/data0/user/liwei/closedloop_stage8_runtime/IsaacLab}"
export LD_LIBRARY_PATH="/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export TORCH_NCCL_BLOCKING_WAIT=1
cd -- "$task_repo_root"
exec "$task_python" -B -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train_closedloop_stage10.py "$@"
