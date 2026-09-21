#!/usr/bin/env bash
# 服务器2 MotionMillion/HumanML3D UMR -> BUMI 文本动作预处理筛选启动器。
#
# 复用 prepare_bumi_text.py filter-umr，默认核对559924条输入，用16个CPU常驻worker
# 逐帧检查并把报告写入/data0/user/liwei。原动作不修改、不裁剪，不启动训练或渲染。
# 可通过 BUMI_* 环境变量改路径，通过命令行追加 --limit/--workers/--output/--resume。
# 小样本验证必须显式指定 /tmp 下独立 --output；完成后由调用方清理精确测试目录。
# 同一路径续跑会核对配置/代码/资产和源文件指纹，变更规则时请使用新报告目录。
# BUMI_DATASET=humanml3d选择HumanML3D交付与补齐源包；机器人输出仍为Z-up。
# BUMI_BUILD_TRAINING=1在全量完成后绑定PASS文本、编码T5、构建分片并计算train统计量。
# 所有正式产物默认位于/data0/user/liwei；不修改data2原始机器人动作、不启动模型训练。
# BUMI_DATASET=bones_seed使用50Hz源人体到30Hz机器人契约，绑定原始SMPL目录及官方文本。

set -euo pipefail
BUMI_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUMI_PYTHON="${BUMI_PYTHON:-/home/user/miniconda3/envs/ykj_umr/bin/python}"
BUMI_UMR_ROOT="${BUMI_UMR_ROOT:-/home/user/ykj/code/UMR}"
BUMI_DATASET="${BUMI_DATASET:-motionmillion}"
BUMI_EXTRA_ARGS=()
if [[ "$BUMI_DATASET" == humanml3d ]]; then
  BUMI_INPUT_ROOT="${BUMI_INPUT_ROOT:-/data2/user/liwei/hml3d_umr}"
  BUMI_SOURCE_ROOT="${BUMI_SOURCE_ROOT:-/data0/user/liwei/datasets/humanml3d_umr_source_v1}"
  BUMI_REPORT_ROOT="${BUMI_REPORT_ROOT:-/data0/user/liwei/dataset_reports/humanml3d_umr_bumi3_latest}"
  BUMI_EXPECTED_RECORDS="${BUMI_EXPECTED_RECORDS:-23242}"
  BUMI_EXTRA_ARGS=(--recorded-output-root "${BUMI_RECORDED_OUTPUT_ROOT:-/home/user/hml3d_umr/out_umr/bumi3}"
                   --recorded-robot-xml "${BUMI_RECORDED_ROBOT_XML:-/home/user/UMR/assets/bumi3/mjcf/bumi3_retarget.xml}")
elif [[ "$BUMI_DATASET" == bones_seed ]]; then
  BUMI_UMR_ROOT="${BUMI_BONES_UMR_ROOT:-/home/user/ykj/code/UMR-main}"
  BUMI_INPUT_ROOT="${BUMI_INPUT_ROOT:-/data0/user/liwei/datasets/BONES-SEED-SMPL/data/ykj/umr_change}"
  BUMI_SOURCE_ROOT="${BUMI_SOURCE_ROOT:-/data0/user/liwei/datasets/BONES-SEED-SMPL/data/ykj/pre_change}"
  BUMI_REPORT_ROOT="${BUMI_REPORT_ROOT:-/data0/user/liwei/dataset_reports/bones_seed_umr_bumi3_latest}"
  BUMI_EXPECTED_RECORDS="${BUMI_EXPECTED_RECORDS:-131454}"
  BUMI_EXTRA_ARGS=(--metadata-csv "${BUMI_METADATA_CSV:-/data0/user/liwei/datasets/BONES-SEED/metadata/seed_metadata_v004.csv}"
                   --original-source-root "${BUMI_ORIGINAL_SOURCE_ROOT:-/data0/user/liwei/datasets/BONES-SEED-SMPL/data/smpl_filtered}")
elif [[ "$BUMI_DATASET" == motionmillion ]]; then
  BUMI_INPUT_ROOT="${BUMI_INPUT_ROOT:-/data2/user/motion_dataset/millionmotion/umr_change/all}"
  BUMI_SOURCE_ROOT="${BUMI_SOURCE_ROOT:-/data2/user/motion_dataset/millionmotion/pre_change/all}"
  BUMI_REPORT_ROOT="${BUMI_REPORT_ROOT:-/data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_latest}"
  BUMI_EXPECTED_RECORDS="${BUMI_EXPECTED_RECORDS:-559924}"
else
  echo "未知数据集: $BUMI_DATASET" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd -- "$BUMI_REPO_ROOT"
CUDA_VISIBLE_DEVICES="" "$BUMI_PYTHON" -u -B tools/data/bumi/prepare_bumi_text.py filter-umr \
  --dataset "$BUMI_DATASET" \
  --input-root "$BUMI_INPUT_ROOT" \
  --source-root "$BUMI_SOURCE_ROOT" \
  --robot-xml "$BUMI_UMR_ROOT/assets/bumi3/mjcf/bumi3_retarget.xml" \
  --retarget-config "$BUMI_UMR_ROOT/robot_configs/humanoid_retarget_bumi3_example.json" \
  --batch-config "$BUMI_UMR_ROOT/humanoid_retarget_defaults_batch_bumi3.json" \
  --asset-manifest "$BUMI_UMR_ROOT/assets/bumi3/asset_manifest.json" \
  --expected-records "$BUMI_EXPECTED_RECORDS" \
  --workers "${BUMI_WORKERS:-16}" \
  --output "$BUMI_REPORT_ROOT" "${BUMI_EXTRA_ARGS[@]}" "$@"

if [[ "${BUMI_BUILD_TRAINING:-0}" == 1 ]]; then
  [[ "$BUMI_DATASET" == humanml3d ]] || { echo '自动文本清单构建仅支持HumanML3D' >&2; exit 2; }
  BUMI_RELEASE_ROOT="${BUMI_RELEASE_ROOT:-/data0/user/liwei/datasets/bumi_text_humanml3d_umr_pass_latest}"
  BUMI_T5_ROOT="${BUMI_T5_ROOT:-${BUMI_RELEASE_ROOT}_t5}"
  BUMI_T5_MODEL="${BUMI_T5_MODEL:-/data0/user/liwei/models/t5-3b_bed96aab}"
  BUMI_ENCODE_PYTHON="${BUMI_ENCODE_PYTHON:-/data0/user/liwei/envs/GENMO-cu128/bin/python}"
  "$BUMI_PYTHON" -u -B tools/data/bumi/prepare_bumi_text.py humanml-conversion \
    --quality-report "$BUMI_REPORT_ROOT" --output "$BUMI_REPORT_ROOT/conversion.json"
  "$BUMI_ENCODE_PYTHON" -u -B tools/data/bumi/encode_text_features.py \
    --source "$BUMI_REPORT_ROOT/conversion.json" --output "$BUMI_T5_ROOT" \
    --t5-model "$BUMI_T5_MODEL" --device "${BUMI_TEXT_DEVICE:-cuda:0}"
  "$BUMI_PYTHON" -u -B tools/data/bumi/prepare_bumi_text.py build \
    --source "$BUMI_T5_ROOT/conversion.json" --output "$BUMI_RELEASE_ROOT" \
    --quality-report "$BUMI_REPORT_ROOT"
  "$BUMI_PYTHON" -u -B tools/data/bumi/prepare_bumi_text.py stats \
    --root "$BUMI_RELEASE_ROOT" --dataset humanml3d --output "$BUMI_RELEASE_ROOT/train_stats.json"
  "$BUMI_PYTHON" -u -B tools/data/bumi/prepare_bumi_text.py preflight \
    --root "$BUMI_RELEASE_ROOT" --dataset humanml3d --limit 0 --output "$BUMI_REPORT_ROOT/training_preflight.json"
fi
