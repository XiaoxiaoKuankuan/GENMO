"""统一检查系统安装和虚拟环境 wheel 安装的 TensorRT 实际运行库。

旧环境使用系统 libnvinfer，新电脑的一键安装将同版库放在虚拟环境，不要求单独配置
NVIDIA APT 源。CUDA 13 的 Python 包改名为 nvidia-cuda-runtime，库移动到 nvidia/cu13，
因此加载 TensorRT 前显式预加载该目录的 libcudart，避免旧版 TensorRT wheel 找不到它。
CUDA 12 的 PyTorch 仍使用自己的不同 SONAME 运行库，不覆盖系统 CUDA 或驱动。

版本查询优先读取 Linux 进程已经实际映射的 libnvinfer 路径，不能用系统 ldconfig 中
另一套库冒充当前 Python binding 所链接的版本。发现同时加载不同路径时明确拒绝。
本模块只依赖标准库，也可直接作为安装器探针执行；不会创建 CUDA 上下文或运行模型。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import re
from importlib import metadata
from pathlib import Path


def prepare_tensorrt_libraries() -> None:
    try:
        runtime = metadata.distribution("nvidia-cuda-runtime")
    except metadata.PackageNotFoundError:
        return
    path = Path(runtime.locate_file("nvidia/cu13/lib/libcudart.so.13"))
    if path.is_file():
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def loaded_nvinfer_paths(maps: str) -> list[Path]:
    paths = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and re.fullmatch(r"libnvinfer\.so(?:\.[0-9]+)*", Path(fields[5]).name):
            paths.add(Path(fields[5]).resolve())
    return sorted(paths)


def linked_tensorrt_version() -> str:
    maps = Path("/proc/self/maps")
    loaded = loaded_nvinfer_paths(maps.read_text()) if maps.is_file() else []
    if len(loaded) > 1:
        raise RuntimeError(f"同时加载了多份 libnvinfer，无法确定实际 ABI：{loaded}")
    library_path = str(loaded[0]) if loaded else ctypes.util.find_library("nvinfer")
    if not library_path:
        raise RuntimeError("libnvinfer 未加载；请先运行部署安装脚本并导入 TensorRT")
    library = ctypes.CDLL(library_path)
    get_version = library.getInferLibVersion
    get_version.restype = ctypes.c_int32
    encoded = int(get_version())
    if encoded <= 0:
        raise RuntimeError(f"libnvinfer 返回无效版本：{encoded}")
    return f"{encoded // 10000}.{encoded % 10000 // 100}.{encoded % 100}"


def main() -> int:
    prepare_tensorrt_libraries()
    import tensorrt as trt

    actual = linked_tensorrt_version()
    if trt.__version__ != "10.13.3.9" or actual != "10.13.3":
        raise RuntimeError(
            f"部署锁要求 TensorRT 10.13.3.9，实际 binding={trt.__version__}, lib={actual}"
        )
    print(json.dumps({"tensorrt": trt.__version__, "libnvinfer": actual}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
