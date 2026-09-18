#!/usr/bin/env bash
# BUMI音乐生成舞蹈的八卡训练入口：读取正式四来源数据及其统计量指纹，从零训练35万step。
# 默认使用服务器1已发布的UMR前70%+自建PASS数据；其他数据版本可通过BUMI_DATASET_ROOT、
# BUMI_EXP_CONFIG和GENMO_PYTHON覆盖。--smoke仅执行两步真实八卡训练及一次短验证，
# 所有checkpoint、日志和Hydra产物写系统临时目录，退出时按精确路径清理。
# 正式模式为每次运行创建独立输出目录，既有checkpoint和训练日志不会被覆盖；可在tmux中运行。
set -euo pipefail

TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_REPO_ROOT"
TASK_PYTHON="${GENMO_PYTHON:-/home/user/liwei/GENMO/.venv/bin/python}"
TASK_DATA="${BUMI_DATASET_ROOT:-/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_v1}"
TASK_CONFIG="${BUMI_EXP_CONFIG:-gem_bumi_music_only_umr70_mine_scratch_350k}"
TASK_MODE="train"
if [[ "${1:-}" == "--smoke" ]]; then TASK_MODE="smoke"; shift; fi

export AISTPP_BUMI_ROOT="$TASK_DATA/AIST++"
export AIOZ_GDANCE_BUMI_ROOT="$TASK_DATA/AIOZ-GDANCE"
export FINEDANCE_BUMI_ROOT="$TASK_DATA/FineDance"
export MINE_BUMI_ROOT="$TASK_DATA/Mine"
export BUMI_KINEMATICS_PATH="$TASK_REPO_ROOT/configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
export BUMI_MUSIC_QPOS30_STATS_PATH="$TASK_DATA/stats/qpos30_train_stats.json"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export BUMI_EXPECTED_TRAIN_SEQUENCES
BUMI_EXPECTED_TRAIN_SEQUENCES="$($TASK_PYTHON -B - "$TASK_DATA" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
report = json.loads((root / 'conversion_report.json').read_text())
assert report['status'] == 'passed'
assert (root / 'stats/qpos30_train_stats.json').is_file()
print(sum(v['splits']['train'] for v in report['datasets'].values()))
PY
)"

TASK_EXTRA=()
if [[ "$TASK_MODE" == "smoke" ]]; then
  TASK_OUTPUT="$(mktemp -d /tmp/genmo-bumi-eight-gpu.XXXXXX)"
  trap 'case "$TASK_OUTPUT" in /tmp/genmo-bumi-eight-gpu.*) rm -rf -- "$TASK_OUTPUT"; test ! -e "$TASK_OUTPUT";; *) exit 99;; esac' EXIT
  TASK_EXTRA=(pl_trainer.max_steps=2 pl_trainer.val_check_interval=2
    +pl_trainer.limit_val_batches=1 callbacks={} data.samples_per_epoch=4096)
else
  TASK_BASE="${BUMI_OUTPUT_BASE:-/data0/user/liwei/GENMO_outputs}"
  TASK_OUTPUT="$TASK_BASE/bumi_umr70_mine_s350k_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
  test ! -e "$TASK_OUTPUT"
  mkdir -p "$TASK_OUTPUT"
fi
printf '模式=%s\n代码=%s\n数据=%s\n训练序列=%s\n输出=%s\n' \
  "$TASK_MODE" "$(git rev-parse HEAD)" "$TASK_DATA" "$BUMI_EXPECTED_TRAIN_SEQUENCES" "$TASK_OUTPUT"
"$TASK_PYTHON" -B scripts/train.py exp="$TASK_CONFIG" output_dir="$TASK_OUTPUT" \
  hydra.run.dir="$TASK_OUTPUT/hydra" "${TASK_EXTRA[@]}" "$@" 2>&1 | tee "$TASK_OUTPUT/launch.log"
