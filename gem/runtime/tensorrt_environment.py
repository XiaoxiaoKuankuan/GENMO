"""统一检查系统安装和虚拟环境 wheel 安装的 TensorRT 实际运行库。

旧环境使用系统 libnvinfer，新电脑的一键安装将同版库放在虚拟环境，不要求单独配置
NVIDIA APT 源。CUDA 13 的 Python 包改名为 nvidia-cuda-runtime，库移动到 nvidia/cu13，
因此加载 TensorRT 前显式预加载该目录的 libcudart，避免旧版 TensorRT wheel 找不到它。
CUDA 12 的 PyTorch 仍使用自己的不同 SONAME 运行库，不覆盖系统 CUDA 或驱动。

版本查询优先读取 Linux 进程已经实际映射的 libnvinfer 路径，不能用系统 ldconfig 中
另一套库冒充当前 Python binding 所链接的版本。发现同时加载不同路径时明确拒绝。
本模块只依赖标准库，也可直接作为安装器探针执行；不会创建 CUDA 上下文或运行模型。
CUDA12部署按已安装的TensorRT绑定族预加载nvidia/cuda_runtime的libcudart.so.12，
CUDA13保持原路径；两种TensorRT绑定同时存在时拒绝运行。安装探针可显式指定CUDA族，
同时报告实际加载的libnvinfer和cudart路径，避免把系统另一套库当成当前wheel环境。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import re
from importlib import metadata
from pathlib import Path


def installed_tensorrt_cuda_major() -> int | None:
    """按绑定发行包识别CUDA族，不把PyTorch的CUDA12运行库误认为TensorRT族。"""
    families = []
    for major in (12, 13):
        try:
            metadata.distribution(f"tensorrt-cu{major}-bindings")
        except metadata.PackageNotFoundError:
            continue
        families.append(major)
    if len(families) > 1:
        raise RuntimeError("同时安装了CUDA12和CUDA13 TensorRT绑定，拒绝混用")
    return families[0] if families else None


def prepare_tensorrt_libraries(cuda_major: int | None = None) -> None:
    installed = installed_tensorrt_cuda_major()
    if installed is not None and cuda_major is not None and installed != cuda_major:
        raise RuntimeError(f"TensorRT CUDA族不匹配：已有{installed}，要求{cuda_major}")
    major = cuda_major or installed
    if major == 12:
        name, relative = "nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime/lib/libcudart.so.12"
    else:
        # 无wheel绑定时保留原系统CUDA13安装的预加载行为。
        name, relative = "nvidia-cuda-runtime", "nvidia/cu13/lib/libcudart.so.13"
    try:
        runtime = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return
    path = Path(runtime.locate_file(relative))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-major", type=int, choices=(12, 13))
    args = parser.parse_args()
    prepare_tensorrt_libraries(args.cuda_major)
    import tensorrt as trt

    actual = linked_tensorrt_version()
    if trt.__version__ != "10.13.3.9" or actual != "10.13.3":
        raise RuntimeError(
            f"部署锁要求 TensorRT 10.13.3.9，实际 binding={trt.__version__}, lib={actual}"
        )
    maps = Path("/proc/self/maps").read_text()
    cuda_paths = sorted({line.split(maxsplit=5)[5]
                         for line in maps.splitlines()
                         if len(line.split(maxsplit=5)) == 6
                         and re.fullmatch(r"libcudart\.so(?:\.[0-9]+)*",
                                          Path(line.split(maxsplit=5)[5]).name)})
    print(json.dumps({"tensorrt": trt.__version__, "libnvinfer": actual,
                      "tensorrt_cuda_major": installed_tensorrt_cuda_major(),
                      "libnvinfer_paths": [str(p) for p in loaded_nvinfer_paths(maps)],
                      "libcudart_paths": cuda_paths}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
