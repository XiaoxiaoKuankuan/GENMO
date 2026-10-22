#!/usr/bin/env bash
# 原DPPO三项疑问验证的独立八卡入口。
# 输入capture或replay及对应Python参数；沿用服务器2已验收的Isaac动态库、NCCL与
# FP32执行环境，在独立输出目录采集或重放真实数据，不启动长期正式训练。
# capture必须经来源登记克隆恢复；replay只读取固定学习状态并保存诊断候选。
set -euo pipefail
STAGE2_DIAGNOSTIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE2_DIAGNOSTIC_MODE="${1:?需要capture或replay}"
shift
case "$STAGE2_DIAGNOSTIC_MODE" in
  capture) STAGE2_DIAGNOSTIC_ENTRY=tools/capture_stage2_dppo_case.py ;;
  replay) STAGE2_DIAGNOSTIC_ENTRY=tools/validate_stage2_dppo_suspicions.py ;;
  *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ISAACLAB_PATH=/data0/user/liwei/closedloop_stage8_runtime/IsaacLab
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo TORCH_NCCL_BLOCKING_WAIT=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGE2_DIAGNOSTIC_SYSROOT=/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="$STAGE2_DIAGNOSTIC_SYSROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$STAGE2_DIAGNOSTIC_ROOT"
exec /data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B -m torch.distributed.run \
  --standalone --nproc_per_node=8 "$STAGE2_DIAGNOSTIC_ENTRY" "$@"
