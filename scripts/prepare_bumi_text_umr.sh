#!/usr/bin/env bash
# 服务器2 MotionMillion UMR -> BUMI 文本动作预处理筛选启动器。
#
# 复用 prepare_bumi_text.py filter-umr，默认核对559924条输入，用16个CPU常驻worker
# 逐帧检查并把报告写入/data0/user/liwei。原动作不修改、不裁剪，不启动训练或渲染。
# 可通过 BUMI_* 环境变量改路径，通过命令行追加 --limit/--workers/--output/--resume。
# 小样本验证必须显式指定 /tmp 下独立 --output；完成后由调用方清理精确测试目录。
# 同一路径续跑会核对配置/代码/资产和源文件指纹，变更规则时请使用新报告目录。

set -euo pipefail
BUMI_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUMI_PYTHON="${BUMI_PYTHON:-/home/user/miniconda3/envs/ykj_umr/bin/python}"
BUMI_UMR_ROOT="${BUMI_UMR_ROOT:-/home/user/ykj/code/UMR}"
BUMI_INPUT_ROOT="${BUMI_INPUT_ROOT:-/data2/user/motion_dataset/millionmotion/umr_change/all}"
BUMI_SOURCE_ROOT="${BUMI_SOURCE_ROOT:-/data2/user/motion_dataset/millionmotion/pre_change/all}"
BUMI_REPORT_ROOT="${BUMI_REPORT_ROOT:-/data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_v1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
cd -- "$BUMI_REPO_ROOT"
exec "$BUMI_PYTHON" -u -B tools/data/bumi/prepare_bumi_text.py filter-umr \
  --input-root "$BUMI_INPUT_ROOT" \
  --source-root "$BUMI_SOURCE_ROOT" \
  --robot-xml "$BUMI_UMR_ROOT/assets/bumi3/mjcf/bumi3_retarget.xml" \
  --retarget-config "$BUMI_UMR_ROOT/robot_configs/humanoid_retarget_bumi3_example.json" \
  --batch-config "$BUMI_UMR_ROOT/humanoid_retarget_defaults_batch_bumi3.json" \
  --asset-manifest "$BUMI_UMR_ROOT/assets/bumi3/asset_manifest.json" \
  --expected-records "${BUMI_EXPECTED_RECORDS:-559924}" \
  --workers "${BUMI_WORKERS:-16}" \
  --output "$BUMI_REPORT_ROOT" "$@"

