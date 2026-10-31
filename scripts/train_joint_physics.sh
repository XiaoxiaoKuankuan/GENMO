#!/usr/bin/env bash
# 从零联合扩散正式启动/恢复包装；只调用本方法入口，不读取旧Stage2状态。
# 第一个参数为已核验配置，其余原样传给CLI。JOINT_GPUS=1使用单进程，>1使用torchrun；
# CUDA_VISIBLE_DEVICES由操作者显式指定。脚本不选择空闲GPU、不停止已有进程、不后台启动。
# --resume只接受完整joint checkpoint；--output必须指向新的独立目录。精度由配置决定，课程和
# AdamW从零或完整恢复推进；运行本脚本就是显式启动正式训练，不属于自动验收步骤。
# --continue-from仅用于固定物理权重断点显式接入梯度占比课程，保留原全局步数和完整状态。
set -euo pipefail
joint_cfg="${1:?需要本方法配置路径}"
shift
joint_python="${JOINT_PYTHON:-python}"
joint_gpus="${JOINT_GPUS:-1}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if (( joint_gpus > 1 )); then
  export NCCL_CUMEM_HOST_ENABLE=0
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
  export TORCH_NCCL_BLOCKING_WAIT=1
  exec "$joint_python" -m torch.distributed.run --standalone --nproc_per_node="$joint_gpus" \
    tools/train_joint_physics.py --config "$joint_cfg" --mode train "$@"
fi
exec "$joint_python" tools/train_joint_physics.py --config "$joint_cfg" --mode train "$@"
