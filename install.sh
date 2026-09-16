#!/usr/bin/env bash
# BUMI music-only Ubuntu x86_64 / RTX 4090 一键安装入口。
# 自动准备 uv、Python 3.10 虚拟环境和锁定的 Python/TensorRT 依赖；缺失时仅通过 APT
# 安装 curl、FFmpeg、Redis 等系统工具。无需预先安装 python3-pip、python3.10-venv 或 Git。
# 已有正确的 TensorRT 环境直接复用；新环境安装虚拟环境内的官方库，不更换系统驱动。
# 最后执行模型哈希、GPU/engine 兼容性和真实单步推理检查，不连接 GMT、不发送动作。
# 只允许在精简部署目录运行，防止误修改完整训练仓库的 .venv；临时下载脚本自动清理。
set -euo pipefail
BUMI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$BUMI_ROOT"
if [[ $# -gt 0 ]]; then
    echo '用法：bash install.sh；运行配置请用编辑器修改 deployment.ini。' >&2
    exit 2
fi
if [[ -f setup.cfg || -d gem/model ]]; then
    echo '请在 GENMO-deploy-bumi 精简部署目录运行安装器，保留完整仓库的训练环境。' >&2
    exit 1
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 || ! -f /etc/debian_version ]]; then
    echo '本安装器支持 Ubuntu/Debian x86_64；当前发布基线为 Ubuntu 22.04 / RTX 4090。' >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L; then
    echo '请先使 NVIDIA 驱动正常识别 GPU；安装器不自动更换驱动。' >&2
    exit 1
fi

BUMI_PACKAGES=()
if ! command -v ffmpeg >/dev/null || ! command -v ffplay >/dev/null; then
    BUMI_PACKAGES+=(ffmpeg)
fi
if ! command -v redis-server >/dev/null || ! command -v redis-cli >/dev/null; then
    BUMI_PACKAGES+=(redis-server)
fi
if ! command -v uv >/dev/null && [[ ! -x .tools/uv ]] && ! command -v curl >/dev/null && ! command -v wget >/dev/null; then
    BUMI_PACKAGES+=(curl ca-certificates)
fi
if [[ ${#BUMI_PACKAGES[@]} -gt 0 ]]; then
    BUMI_SUDO=()
    if [[ $(id -u) -ne 0 ]]; then BUMI_SUDO=(sudo); fi
    echo "安装缺失的系统工具：${BUMI_PACKAGES[*]}（需要系统管理员权限）。"
    "${BUMI_SUDO[@]}" apt-get update
    "${BUMI_SUDO[@]}" apt-get install -y --no-install-recommends "${BUMI_PACKAGES[@]}"
fi

if command -v uv >/dev/null; then
    BUMI_UV="$(command -v uv)"
elif [[ -x .tools/uv ]]; then
    BUMI_UV="$BUMI_ROOT/.tools/uv"
else
    BUMI_INSTALLER="$(mktemp /tmp/bumi-uv-installer.XXXXXXXX.sh)"
    trap 'rm -f -- "$BUMI_INSTALLER"' EXIT
    if command -v curl >/dev/null; then
        curl --fail --location --retry 3 https://astral.sh/uv/0.12.0/install.sh -o "$BUMI_INSTALLER"
    else
        wget -O "$BUMI_INSTALLER" https://astral.sh/uv/0.12.0/install.sh
    fi
    UV_UNMANAGED_INSTALL="$BUMI_ROOT/.tools" sh "$BUMI_INSTALLER"
    BUMI_UV="$BUMI_ROOT/.tools/uv"
fi

if [[ -e .venv ]]; then
    if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -B -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))'; then
        echo '现有 .venv 不是可用的 Python 3.10 环境；请保留旧目录并在新部署目录安装。' >&2
        exit 1
    fi
else
    "$BUMI_UV" --no-config venv --python 3.10 .venv
fi
"$BUMI_UV" --no-config pip install --python .venv/bin/python --index-strategy unsafe-best-match \
    -r requirements/deployment/runtime.lock

if .venv/bin/python -B gem/runtime/tensorrt_environment.py; then
    echo '已有匹配的 TensorRT，复用当前安装。'
else
    # 先尝试已有同版系统运行库；失败才安装虚拟环境内的完整库，避免重复下载约 2.74 GB。
    "$BUMI_UV" --no-config pip install --python .venv/bin/python --no-deps \
        -r requirements/deployment/tensorrt-bindings.lock
    if ! .venv/bin/python -B gem/runtime/tensorrt_environment.py; then
        "$BUMI_UV" --no-config pip install --python .venv/bin/python --no-deps \
            -r requirements/deployment/tensorrt-runtime.lock
        .venv/bin/python -B gem/runtime/tensorrt_environment.py
    fi
fi
bash "$BUMI_ROOT/run.sh" check
echo '安装与模型检查通过。先启动 GMT，然后分别执行 bash run.sh bridge 和 bash run.sh genmo。'
