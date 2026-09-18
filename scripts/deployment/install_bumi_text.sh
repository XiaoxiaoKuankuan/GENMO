#!/usr/bin/env bash
# BUMI文本部署包一键环境准备（Ubuntu22.04/x86_64，Python3.10）。
# uv管理Python及独立.venv；不会安装训练框架，不执行GPU推理或修改驱动。
# TensorRT模型须与本机GPU/库兼容；只安装依赖不会使不兼容engine自动变得兼容。
set -euo pipefail
deployment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$deployment_root"
[[ -f deployment.json && -f requirements.lock ]] || { echo '本入口仅在导出的部署包根目录运行。' >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo '当前锁定平台仅支持Linux x86_64。' >&2; exit 1; }
if ! command -v uv >/dev/null; then
    installer_path="$(mktemp -t bumi-text-uv.XXXXXXXX)"
    trap 'rm -f -- "$installer_path"' EXIT
    command -v curl >/dev/null || { echo '需要curl下载uv，请先安装curl。' >&2; exit 1; }
    curl --fail --location --silent --show-error https://astral.sh/uv/0.12.0/install.sh -o "$installer_path"
    env UV_NO_MODIFY_PATH=1 sh "$installer_path"
    export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.10 .venv; fi
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.lock
selected_backend="$(.venv/bin/python -B -c 'import configparser; c=configparser.ConfigParser(); c.read("deployment.ini"); print(c["runtime"]["backend"])')"
if [[ "$selected_backend" == tensorrt ]]; then
    uv pip install --python .venv/bin/python --no-deps -r tensorrt.lock
fi
.venv/bin/python -B -c 'from gem.runtime.bumi_text_runtime import read_bundle; read_bundle("deployment.json"); print("部署资产检查通过")'
system_packages=()
if ! command -v ffmpeg >/dev/null; then
    system_packages+=(ffmpeg)
fi
if ! .venv/bin/python -B -c 'import ctypes.util,sys; sys.exit(not ctypes.util.find_library("GL"))'; then
    system_packages+=(libgl1)
fi
if (( ${#system_packages[@]} )); then
    echo "准备安装缺少的显示/视频系统依赖：${system_packages[*]}，sudo可能要求本机密码。"
    command -v apt-get >/dev/null && command -v sudo >/dev/null || { echo '请由管理员安装上述系统包后重试。' >&2; exit 1; }
    sudo apt-get update
    sudo apt-get install -y "${system_packages[@]}"
fi
echo '依赖准备完成。请在deployment.ini配置本地T5目录；bash run.sh启动。尚未执行GPU或图形验收。'
