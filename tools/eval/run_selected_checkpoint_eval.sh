#!/usr/bin/env bash
# 本脚本用于按固定顺序评测一组 GENMO 人体模型 checkpoint，并在每个候选模型上执行
# EMDB_1、EMDB_2、3DPW 和 RICH 四个数据集的统一测试。脚本会保存独立日志、运行环境、
# 成功/失败标记和原训练配置，最后调用汇总工具生成跨 checkpoint 排名与最佳模型软链接。
# 默认仅复用已完成的结果；设置 FORCE=1 可强制重跑，传入 step 参数可只评测指定模型。
set -Eeuo pipefail

# ============================================================
# GENMO 候选 checkpoint 顺序评测器
#
# Usage:
#   GPU=0 bash tools/eval/run_selected_checkpoint_eval.sh
#
# Evaluate one checkpoint first:
#   GPU=0 bash tools/eval/run_selected_checkpoint_eval.sh 120000
#
# Force rerun:
#   FORCE=1 GPU=0 bash tools/eval/run_selected_checkpoint_eval.sh
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/gem_mixed/gem_smpl_server_8gpu_bs128_scratch_500k/version_2}"
EVAL_DIR="${EVAL_DIR:-${RUN_DIR}/eval_selected}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-4}"

if [[ "$#" -gt 0 ]]; then
    STEPS=("$@")
else
    STEPS=(
        120000
        150000
        180000
        200000
        230000
    )
fi

EXPECTED_DATASETS=(
    "EMDB_1"
    "EMDB_2"
    "3DPW"
    "RICH"
)

echo "============================================================"
echo "GENMO checkpoint evaluation"
echo "ROOT_DIR : ${ROOT_DIR}"
echo "RUN_DIR  : ${RUN_DIR}"
echo "EVAL_DIR : ${EVAL_DIR}"
echo "GPU      : ${GPU}"
echo "STEPS    : ${STEPS[*]}"
echo "============================================================"

if [[ ! -d "${ROOT_DIR}/.venv" ]]; then
    echo "[ERROR] 未找到虚拟环境：${ROOT_DIR}/.venv"
    exit 1
fi

if [[ ! -d "${RUN_DIR}/checkpoints" ]]; then
    echo "[ERROR] 未找到checkpoint目录：${RUN_DIR}/checkpoints"
    exit 1
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/.venv/bin/activate"
cd "${ROOT_DIR}"

mkdir -p "${EVAL_DIR}"

# 保存评测环境记录
{
    echo "date: $(date --iso-8601=seconds)"
    echo "root_dir: ${ROOT_DIR}"
    echo "run_dir: ${RUN_DIR}"
    echo "eval_dir: ${EVAL_DIR}"
    echo "gpu_env: ${GPU}"
    echo "steps: ${STEPS[*]}"
    echo
    echo "git_commit:"
    git rev-parse HEAD 2>/dev/null || true
    echo
    echo "git_status:"
    git status --short 2>/dev/null || true
} > "${EVAL_DIR}/evaluation_metadata.txt"

python - <<'PY' > "${EVAL_DIR}/environment.txt"
import platform
import torch
import pytorch_lightning as pl

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("lightning:", pl.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count_visible:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda_device_{i}:", torch.cuda.get_device_name(i))
PY

# 备份原训练配置，便于以后核查
if [[ -f "${RUN_DIR}/hparams.yaml" ]]; then
    cp -f "${RUN_DIR}/hparams.yaml" "${EVAL_DIR}/training_hparams.yaml"
fi

if [[ -f "${RUN_DIR}/meta.yaml" ]]; then
    cp -f "${RUN_DIR}/meta.yaml" "${EVAL_DIR}/training_meta.yaml"
fi

SUCCESS_COUNT=0
FAILED_COUNT=0

for STEP in "${STEPS[@]}"; do
    NAME="$(printf 's%06d' "${STEP}")"
    CKPT="${RUN_DIR}/checkpoints/${NAME}.ckpt"
    OUT_DIR="${EVAL_DIR}/${NAME}"
    LOG_FILE="${OUT_DIR}/eval.log"
    SUCCESS_FILE="${OUT_DIR}/SUCCESS"
    FAILED_FILE="${OUT_DIR}/FAILED"

    echo
    echo "============================================================"
    echo "[EVAL] checkpoint=${NAME}"
    echo "[PATH] ${CKPT}"
    echo "============================================================"

    if [[ ! -s "${CKPT}" ]]; then
        echo "[ERROR] checkpoint不存在或为空：${CKPT}"
        mkdir -p "${OUT_DIR}"
        echo "checkpoint missing: ${CKPT}" > "${FAILED_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    if [[ -f "${SUCCESS_FILE}" && "${FORCE}" != "1" ]]; then
        echo "[SKIP] 已经成功评测：${NAME}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        continue
    fi

    # Hydra 的 job_logging 会在进入 Python main() 前打开：
    #   ${output_dir}/${hydra.job.name}.log
    # 因此必须提前创建 runtime 目录。
    mkdir -p "${OUT_DIR}" "${OUT_DIR}/runtime" "${OUT_DIR}/hydra"
    rm -f "${SUCCESS_FILE}" "${FAILED_FILE}"

    # FORCE=1 或上次失败时，重新覆盖日志
    : > "${LOG_FILE}"

    {
        echo "checkpoint=${CKPT}"
        echo "step=${STEP}"
        echo "start_time=$(date --iso-8601=seconds)"
        echo "gpu=${GPU}"
        echo
    } | tee -a "${LOG_FILE}"

    CKPT_ABS="$(realpath "${CKPT}")"

    # 说明：
    # 1. 使用原训练对应的 exp=gem_smpl_server；
    # 2. task=test 会调用 trainer.test；
    # 3. 删除 train/val dataset，避免评测时重复加载所有训练集；
    # 4. 将 test_datasets 映射到 DataModule 的 test 节点；
    # 5. 单GPU、batch_size=1，保证各checkpoint评测口径完全一致。
    set +e

    CUDA_VISIBLE_DEVICES="${GPU}" \
    HYDRA_FULL_ERROR=1 \
    PYTHONUNBUFFERED=1 \
    python scripts/train.py \
        exp=gem_smpl_server \
        task=test \
        seed=42 \
        ckpt_path="${CKPT_ABS}" \
        use_wandb=false \
        pl_trainer.devices=1 \
        pl_trainer.strategy=auto \
        +pl_trainer.enable_checkpointing=false \
        '~data.dataset_opts.train' \
        '~data.loader_opts.train' \
        '~data.dataset_opts.val' \
        '~data.loader_opts.val' \
        '+data.dataset_opts.test=${test_datasets}' \
        '+data.loader_opts.test.batch_size=1' \
        "+data.loader_opts.test.num_workers=${TEST_NUM_WORKERS}" \
        output_dir="${OUT_DIR}/runtime" \
        hydra.run.dir="${OUT_DIR}/hydra" \
        2>&1 | tee -a "${LOG_FILE}"

    EXIT_CODE="${PIPESTATUS[0]}"

    set -e

    {
        echo
        echo "end_time=$(date --iso-8601=seconds)"
        echo "exit_code=${EXIT_CODE}"
    } >> "${LOG_FILE}"

    if [[ "${EXIT_CODE}" -ne 0 ]]; then
        echo "[FAILED] ${NAME}，退出码=${EXIT_CODE}"
        echo "exit_code=${EXIT_CODE}" > "${FAILED_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    # 检查四个数据集是否全部输出
    MISSING_METRIC_BLOCK=0

    for DATASET in "${EXPECTED_DATASETS[@]}"; do
        if grep -Eq "\[Metrics\][[:space:]]+${DATASET}:" "${LOG_FILE}"; then
            echo "[OK] ${NAME}: 找到 ${DATASET} 指标"
        else
            echo "[ERROR] ${NAME}: 缺少 ${DATASET} 指标"
            MISSING_METRIC_BLOCK=1
        fi
    done

    if [[ "${MISSING_METRIC_BLOCK}" -ne 0 ]]; then
        echo "missing one or more metric blocks" > "${FAILED_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    # 检查明显运行错误
    if grep -Eq "Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory" "${LOG_FILE}"; then
        echo "[ERROR] ${NAME}: 日志中检测到异常"
        echo "runtime exception found in log" > "${FAILED_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    touch "${SUCCESS_FILE}"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

    echo "[SUCCESS] ${NAME}"
done

echo
echo "============================================================"
echo "Evaluation finished"
echo "success=${SUCCESS_COUNT}"
echo "failed=${FAILED_COUNT}"
echo "============================================================"

if [[ "${SUCCESS_COUNT}" -gt 0 ]]; then
    python tools/eval/summarize_selected_checkpoint_eval.py \
        --run-dir "${RUN_DIR}" \
        --eval-dir "${EVAL_DIR}" \
        --steps "${STEPS[@]}"
fi

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
    echo "[WARNING] 存在评测失败项，请检查：${EVAL_DIR}"
    exit 2
fi

echo "[DONE] 全部checkpoint评测和汇总完成。"
