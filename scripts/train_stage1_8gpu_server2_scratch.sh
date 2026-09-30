#!/usr/bin/env bash
# 服务器2 Stage1 八卡随机初始化训练入口：只启动本次600000优化步的新训练，不加载旧权重。
# 本脚本复用现有持久DDP执行器，固定当前四库及新train统计量，先核对配置、统计SHA、
# 四库训练清单/元数据身份和Mine高度恢复版本，再启动八个同步训练rank。P=0概率15%，
# 非零P均匀取6～30；网络、历史代理、损失和四库采样权重不变。
# 默认使用服务器2已核验的Torch2.7.1+cu128环境及GPU0～7。NCCL选项仅作用于本进程树，
# 不改变系统环境。GPU已有计算任务、数据身份不符或输出目录非空时拒绝新训练。
# --check-only只进行启动前只读核验，不创建模型、不更新参数、不创建输出目录。
# 正式日志由外层tmux调用重定向到独立console文件；checkpoint和TensorBoard由rank0写入。
# 需要断点恢复时使用tools/train_closedloop_stage1.py的--resume-checkpoint入口，不能将
# 本脚本的“新训练”检查绕过后写入已有目录；具体完整恢复命令见对应训练说明。
set -euo pipefail

TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${GENMO_PYTHON:-/data0/user/liwei/envs/GENMO-cu128/bin/python}"
TASK_CONFIG="$TASK_REPO_ROOT/configs/closedloop/stage1_server2_scratch_600k.yaml"
TASK_CHECK_ONLY=0
if [[ $# -gt 0 ]]; then
  case "$1" in
    --check-only) TASK_CHECK_ONLY=1 ;;
    -h|--help)
      printf '用法：bash scripts/train_stage1_8gpu_server2_scratch.sh [--check-only]\n'
      exit 0
      ;;
    *) printf '不支持的参数：%s\n' "$1" >&2; exit 2 ;;
  esac
  shift
fi
[[ $# -eq 0 ]] || { printf '只允许一个可选参数。\n' >&2; exit 2; }
[[ -x "$TASK_PYTHON" && -f "$TASK_CONFIG" ]] || { printf '解释器或配置不存在。\n' >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a TASK_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#TASK_GPU_IDS[@]} -eq 8 ]] || { printf '此配置要求8张GPU。\n' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export BUMI_CLOSEDLOOP_90505_ROOT=/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1
export BUMI_CLOSEDLOOP_90505_STATS_PATH=/data0/user/liwei/GENMO_assets/stage1_scratch600k_20260930/qpos30_train_stats.json

cd "$TASK_REPO_ROOT"
"$TASK_PYTHON" -B - "$TASK_CONFIG" <<'PY'
"""只读核验随机初始化、新统计量和当前数据身份，避免误启动旧权重或旧Mine版本。"""
import hashlib
import json
import os
from pathlib import Path
import sys

from gem.closedloop.training import load_stage1_config, load_stage1_data_config

config = load_stage1_config(sys.argv[1])
data = load_stage1_data_config(config)
assert config.mode == "train" and config.trainer.enabled
assert not config.trainer.require_warm_start
for key in ("warm_start_checkpoint", "stage1_checkpoint", "resume_checkpoint"):
    assert config.get(key) is None, f"新训练禁止加载已有权重或训练状态：{key}"
assert config.train.max_steps == config.scheduler.total_steps == 600000
assert config.sample_contract.prefix_min_frames == 6
assert config.sample_contract.prefix_max_frames == 30
assert config.sample_contract.prefix_zero_probability == 0.15
assert config.data_loader.batch_size == 256 and config.trainer.gradient_accumulation == 1
assert len(set(os.environ["CUDA_VISIBLE_DEVICES"].split(","))) == 8
output = Path(str(config.runtime.output_dir))
assert not output.exists() or not any(output.iterdir()), f"新训练目录必须为空：{output}"

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

stats_path = Path(str(config.endecoder.stats_path))
assert stats_path == Path(str(data.qpos30_stats.path))
assert sha256(stats_path) == config.input_identity.stats_sha256, "当前train统计量SHA不符"
stats = json.loads(stats_path.read_text())
assert not stats["is_placeholder"] and stats["feature_dim"] == 30
assert stats["kinematics_sha256"] == sha256(config.endecoder.kinematics_path)
counts = {}
for entry in data.datasets.train.values():
    root = Path(str(entry.root))
    assert root.parent == Path(str(config.input_identity.dataset_root))
    manifest = root / "manifests/train.jsonl"
    info_path = root / "meta/dataset_info.json"
    fingerprint = stats["dataset_fingerprints"][str(entry.dataset_name)]
    assert sha256(manifest) == fingerprint["train_manifest_sha256"], f"训练清单改变：{root}"
    assert sha256(info_path) == fingerprint["dataset_info_sha256"], f"元数据改变：{root}"
    count = sum(bool(line.strip()) for line in manifest.read_text().splitlines())
    assert count == fingerprint["sequences"]
    counts[str(entry.dataset_name)] = count
    if entry.dataset_name == "mine_bumi":
        info = json.loads(info_path.read_text())
        assert info["ground_semantics"] == config.input_identity.mine_ground_semantics
        assert info["height_restoration_version"] == config.input_identity.mine_height_restoration_version
assert counts == {"aistpp_bumi": 351, "aioz_gdance_bumi": 3745, "finedance_bumi": 102, "mine_bumi": 89}
print(json.dumps({"status": "passed", "initialization": "random_initialization",
                  "loaded_checkpoint": None, "optimizer_restored": False,
                  "initial_global_step": 0, "max_steps": 600000,
                  "stats_sha256": sha256(stats_path), "train_sequences": counts,
                  "output_dir": str(output)}, ensure_ascii=False), flush=True)
PY

if [[ "$TASK_CHECK_ONLY" -eq 1 ]]; then
  exit 0
fi
TASK_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$TASK_GPU_PROCESSES" ]] || { printf 'GPU已有计算进程，未启动新训练：\n%s\n' "$TASK_GPU_PROCESSES" >&2; exit 2; }
exec "$TASK_PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train_closedloop_stage1.py --config "$TASK_CONFIG"
