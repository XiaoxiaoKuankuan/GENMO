#!/usr/bin/env bash
# BUMI Stage1 Ubuntu x86_64 / RTX 4090 一键安装入口。
# 自动准备 uv、Python 3.10 虚拟环境和锁定的 Python/TensorRT 依赖；缺失时仅通过 APT
# 安装 curl、FFmpeg、Redis 等系统工具。无需预先安装 python3-pip、python3.10-venv 或 Git。
# 已有正确的 TensorRT 环境直接复用；新环境安装虚拟环境内的官方库，不更换系统驱动。
# 固定安装 MuJoCo 3.2.3；检查器校验机器人资源和关节契约，不会打开窗口或启动 GMT。
# 仅安装依赖，不自动执行模型检查或推理；之后手工构建Stage1引擎并使用所需启动脚本。
# 同时安装ONNX图改写构建依赖；插件编译使用已有CUDA nvcc及匹配TensorRT SDK，安装器不改CUDA。
# 只允许在精简部署目录运行；--venv显式复用已有Python 3.10环境，并在修改前完整备份。
# 共享环境先逐项核对已安装的锁定依赖，拒绝覆盖不同版本或另一CUDA族的TensorRT。
# --tensorrt-cuda-major 12使用独立CUDA12依赖锁，13保留原默认；不安装驱动或CUDA Toolkit。
# 部署目录的.venv仅链接指定环境，不修改原GENMO的editable安装；临时下载脚本自动清理。
set -euo pipefail
BUMI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$BUMI_ROOT"
BUMI_VENV="$BUMI_ROOT/.venv"
BUMI_SHARED_ENV=false
BUMI_CUDA_MAJOR=13
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            [[ $# -ge 2 && -n "$2" ]] || { echo '--venv缺少路径。' >&2; exit 2; }
            BUMI_VENV="$(realpath -e -- "$2")"
            BUMI_SHARED_ENV=true
            shift 2 ;;
        --tensorrt-cuda-major)
            [[ $# -ge 2 && "$2" =~ ^(12|13)$ ]] || { echo 'CUDA major必须为12或13。' >&2; exit 2; }
            BUMI_CUDA_MAJOR="$2"
            shift 2 ;;
        --help|-h)
            echo '用法：bash install.sh [--venv 已有环境路径] [--tensorrt-cuda-major 12|13]'
            exit 0 ;;
        *) echo "未知安装参数：$1" >&2; exit 2 ;;
    esac
done
if [[ "$BUMI_SHARED_ENV" == true ]]; then
    if [[ ! -x "$BUMI_VENV/bin/python" ]]; then
        echo '--venv必须指向已有可用环境，安装器不会替换它。' >&2
        exit 1
    fi
    if [[ -e .venv || -L .venv ]] && [[ "$(realpath -m .venv)" != "$BUMI_VENV" ]]; then
        echo '部署目录已有另一份.venv，保留原目录并停止。' >&2
        exit 1
    fi
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
BUMI_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
BUMI_MIN_DRIVER=535
if [[ "$BUMI_CUDA_MAJOR" == 13 ]]; then BUMI_MIN_DRIVER=580; fi
if [[ ! "$BUMI_DRIVER" =~ ^([0-9]+)\. ]] || (( BASH_REMATCH[1] < BUMI_MIN_DRIVER )); then
    echo "CUDA $BUMI_CUDA_MAJOR需要R$BUMI_MIN_DRIVER及以上驱动，当前$BUMI_DRIVER；安装器不更换驱动。" >&2
    exit 1
fi
if [[ "$BUMI_SHARED_ENV" == true ]]; then
    "$BUMI_VENV/bin/python" -B - "$BUMI_CUDA_MAJOR" <<'PY'
import importlib.metadata as metadata
import re
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 10):
    raise SystemExit('共享环境必须为Python 3.10，原环境保留。')
normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
installed = {normalize(d.metadata['Name']): d.version for d in metadata.distributions()}
conflicts = []
for filename in ('runtime.lock', 'export.lock'):
    for line in (Path('requirements/deployment') / filename).read_text().splitlines():
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s]+)', line.strip())
        if match:
            name, expected = match.groups()
            actual = installed.get(normalize(name))
            if actual is not None and actual != expected:
                conflicts.append(f'{name}: 已有{actual}，部署要求{expected}')
other = '13' if sys.argv[1] == '12' else '12'
if any(name.startswith('tensorrt-cu' + other) for name in installed):
    conflicts.append(f'共享环境已有CUDA {other} TensorRT，拒绝混装。')
if conflicts:
    raise SystemExit('共享环境存在版本冲突，未修改：\n' + '\n'.join(conflicts))
print('共享环境锁定依赖核对通过；仅补充缺项。')
PY
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

if [[ -e "$BUMI_VENV" ]]; then
    if [[ ! -x "$BUMI_VENV/bin/python" ]] || ! "$BUMI_VENV/bin/python" -B -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))'; then
        echo '现有 .venv 不是可用的 Python 3.10 环境；请保留旧目录并在新部署目录安装。' >&2
        exit 1
    fi
else
    "$BUMI_UV" --no-config venv --python 3.10 "$BUMI_VENV"
fi
if [[ "$BUMI_SHARED_ENV" == true ]]; then
    BUMI_BACKUP_ROOT="${BUMI_INSTALL_BACKUP_ROOT:-$(dirname -- "$BUMI_ROOT")/migration}"
    BUMI_BACKUP="$BUMI_BACKUP_ROOT/genmo-venv-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p -- "$BUMI_BACKUP"
    echo "完整备份共享环境到：$BUMI_BACKUP/.venv"
    cp -a --reflink=auto -- "$BUMI_VENV" "$BUMI_BACKUP/.venv"
    "$BUMI_UV" --no-config pip freeze --python "$BUMI_VENV/bin/python" > "$BUMI_BACKUP/packages.txt"
    "$BUMI_VENV/bin/python" -B - "$BUMI_VENV" > "$BUMI_BACKUP/environment.json" <<'PY'
import importlib.metadata as metadata
import json
import sys
print(json.dumps({'venv': sys.argv[1], 'python': sys.version,
                  'packages': {d.metadata['Name']: d.version for d in metadata.distributions()}},
                 ensure_ascii=False, indent=2))
PY
    if [[ ! -e .venv && ! -L .venv ]]; then ln -s -- "$BUMI_VENV" .venv; fi
fi
"$BUMI_UV" --no-config pip install --python "$BUMI_VENV/bin/python" --index-strategy unsafe-best-match \
    -r requirements/deployment/runtime.lock -r requirements/deployment/export.lock

BUMI_BINDINGS_LOCK=requirements/deployment/tensorrt-bindings.lock
BUMI_RUNTIME_LOCK=requirements/deployment/tensorrt-runtime.lock
if [[ "$BUMI_CUDA_MAJOR" == 12 ]]; then
    BUMI_BINDINGS_LOCK=requirements/deployment/tensorrt-cu12-bindings.lock
    BUMI_RUNTIME_LOCK=requirements/deployment/tensorrt-cu12-runtime.lock
fi
export NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1
if "$BUMI_VENV/bin/python" -B gem/runtime/tensorrt_environment.py --cuda-major "$BUMI_CUDA_MAJOR"; then
    echo '已有匹配的 TensorRT，复用当前安装。'
else
    # 先尝试已有同版系统运行库；失败才安装虚拟环境内的完整库，避免重复下载约 2.74 GB。
    "$BUMI_UV" --no-config pip install --python "$BUMI_VENV/bin/python" --no-deps \
        -r "$BUMI_BINDINGS_LOCK"
    if ! "$BUMI_VENV/bin/python" -B gem/runtime/tensorrt_environment.py --cuda-major "$BUMI_CUDA_MAJOR"; then
        "$BUMI_UV" --no-config pip install --python "$BUMI_VENV/bin/python" --no-deps \
            -r "$BUMI_RUNTIME_LOCK"
        "$BUMI_VENV/bin/python" -B gem/runtime/tensorrt_environment.py --cuda-major "$BUMI_CUDA_MAJOR"
    fi
fi
"$BUMI_VENV/bin/python" -B - <<'PY'
from pathlib import Path
import gem
expected = Path.cwd() / 'gem' / '__init__.py'
if Path(gem.__file__).resolve() != expected.resolve():
    raise SystemExit(f'部署gem导入来源错误：{gem.__file__}')
print(f'部署代码导入来源：{gem.__file__}')
PY
"$BUMI_UV" --no-config pip check --python "$BUMI_VENV/bin/python"
echo '依赖安装完成；未执行模型推理验收。接着运行 bash scripts/export/build_bumi_stage1_engine.sh。'
