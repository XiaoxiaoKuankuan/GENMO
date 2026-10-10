#!/usr/bin/env bash
# 当前Stage1 s595000的CUDA掩码插件、TensorRT FP32构建与七资产打包入口。
# 默认读取已导出的s595000 ONNX、配套元数据、训练统计和运动学资源，不加载训练checkpoint。
# 构建关闭TF32/FP16，生成models/bumi_stage1_s595000/deployment.json供在线/缓存入口共用。
# 用现有nvcc/匹配TensorRT SDK编译IPluginV3，派生图内部将四个bool掩码编码为0/1 INT32。
# Where/Cast/逻辑比较使用独立CUDA内核，保留原分支选择；外部仍是Stage1十一输入/两输出。
# 插件/图改写源码/共享库和原模型身份进入缓存；旧失败缓存保留，新策略不复用旧engine。
# 构建前保存network_lowering.json，记录常量形状折叠、插件节点和派生图SHA，失败也可查看。
# 缺少或版本不符的ONNX构建库用已有uv或pip安装export.lock；不自动安装CUDA/SDK或更换驱动。
# 本脚本只构建和打包，不执行推理对齐、预热、pytest、demo、GMT仿真或实机。
# 源路径可用BUMI_STAGE1_*环境变量覆盖；额外参数透传构建器，例如--overwrite或--workspace-gib。
# 构建脚本不会清理旧模型资产，也不会更改deployment.ini或GMT代码。
set -euo pipefail
BUMI_DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -x "$BUMI_DEPLOY_ROOT/.venv/bin/python" ]]; then
    echo '缺少部署虚拟环境，请先执行 bash install.sh。' >&2
    exit 1
fi
if ! "$BUMI_DEPLOY_ROOT/.venv/bin/python" -B -c 'import importlib.util, importlib.metadata, sys; sys.exit(importlib.util.find_spec("onnx") is None or importlib.metadata.version("onnx") != "1.18.0")'; then
    echo '安装锁定的Stage1 ONNX构建依赖；不执行模型推理或验收。'
    if command -v uv >/dev/null; then
        "$(command -v uv)" --no-config pip install --python "$BUMI_DEPLOY_ROOT/.venv/bin/python" \
            -r "$BUMI_DEPLOY_ROOT/requirements/deployment/export.lock"
    elif [[ -x "$BUMI_DEPLOY_ROOT/.tools/uv" ]]; then
        "$BUMI_DEPLOY_ROOT/.tools/uv" --no-config pip install --python "$BUMI_DEPLOY_ROOT/.venv/bin/python" \
            -r "$BUMI_DEPLOY_ROOT/requirements/deployment/export.lock"
    elif "$BUMI_DEPLOY_ROOT/.venv/bin/python" -B -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("pip") is None)'; then
        "$BUMI_DEPLOY_ROOT/.venv/bin/python" -B -m pip install \
            -r "$BUMI_DEPLOY_ROOT/requirements/deployment/export.lock"
    else
        echo '缺少uv/pip安装器；请先运行 bash install.sh，原环境保留。' >&2
        exit 1
    fi
fi
BUMI_STAGE1_SOURCE_ROOT="${BUMI_STAGE1_SOURCE_ROOT:-/home/weili/bumi-closedloop-worktrees/GENMO}"
BUMI_STAGE1_SOURCE_ASSETS="$BUMI_STAGE1_SOURCE_ROOT/inputs/checkpoints/stage1_server2_s595000_20261008"
BUMI_STAGE1_SOURCE_ONNX="${BUMI_STAGE1_ONNX:-$BUMI_STAGE1_SOURCE_ROOT/outputs/onnx/stage1_server2_s595000_20261008/bumi_stage1_denoiser_s595000.onnx}"
exec "$BUMI_DEPLOY_ROOT/.venv/bin/python" -B -u \
    "$BUMI_DEPLOY_ROOT/tools/export/build_bumi_music_tensorrt.py" \
    --onnx "$BUMI_STAGE1_SOURCE_ONNX" \
    --onnx-metadata "${BUMI_STAGE1_ONNX_METADATA:-$BUMI_STAGE1_SOURCE_ONNX.json}" \
    --stats "${BUMI_STAGE1_STATS:-$BUMI_STAGE1_SOURCE_ASSETS/qpos30_train_stats.json}" \
    --kinematics "${BUMI_STAGE1_KINEMATICS:-$BUMI_STAGE1_SOURCE_ASSETS/bumi_kinematics_robot_retargeter_fe934_v1.json}" \
    --output-dir "${BUMI_STAGE1_OUTPUT_DIR:-$BUMI_DEPLOY_ROOT/models/bumi_stage1_s595000}" \
    --device "${BUMI_STAGE1_DEVICE:-cuda:0}" --precision fp32 "$@"
