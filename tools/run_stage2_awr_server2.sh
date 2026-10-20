#!/usr/bin/env bash
# Server-2专属AWR启动包装：显式独立仓库、原Isaac Python与八卡通信环境。
# 所有参数传给独立入口；不安装包、不改editable指向、不停止其他作业。
# run/resume/evaluate/ablate用torchrun；preflight/stop为单进程只读或运行内停止标志。
set -euo pipefail
AWR_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AWR_PYTHON="${AWR_PYTHON:-/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${AWR_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export ISAACLAB_PATH=/data0/user/liwei/closedloop_stage8_runtime/IsaacLab
export LD_LIBRARY_PATH="/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ "${1:-}" == "preflight" || "${1:-}" == "stop" ]]; then
    exec "${AWR_PYTHON}" -B "${AWR_REPO}/tools/train_stage2_awr.py" "$@"
fi
if [[ "${1:-}" == "verify" ]]; then
    shift
    exec "${AWR_PYTHON}" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
        "${AWR_REPO}/tools/verify_stage2_awr_distributed.py" "$@"
fi
if [[ "${1:-}" == "complete-collection" ]]; then
    shift
    exec "${AWR_PYTHON}" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
        "${AWR_REPO}/tools/complete_stage2_awr_collection.py" "$@"
fi
if [[ "${1:-}" == "retention" ]]; then
    shift
    exec "${AWR_PYTHON}" -B "${AWR_REPO}/tools/validate_stage2_awr_generation.py" "$@"
fi
exec "${AWR_PYTHON}" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    "${AWR_REPO}/tools/train_stage2_awr.py" "$@"
