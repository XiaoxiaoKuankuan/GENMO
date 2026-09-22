"""TensorRT 公共运行库和执行上下文。

负责版本检查、GPU 指纹、线程互斥、CUDA 流执行和可选图捕获。模型后端必须提供
自身的清单与 I/O 校验，公共层不预设音乐输入、151D 人体输出或滑窗采样方式。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from gem.runtime.tensorrt_environment import linked_tensorrt_version, prepare_tensorrt_libraries


def _parse_version(value: str) -> tuple[int, int, int]:
    parts = str(value).split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise RuntimeError(f"cannot parse TensorRT version {value!r}")
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return int(parts[0]), int(parts[1]), patch


def validate_tensorrt_installation(trt_module: object) -> str:
    """Require the Python binding and linked runtime to share major/minor ABI."""
    binding = str(getattr(trt_module, "__version__", ""))
    runtime = linked_tensorrt_version()
    if _parse_version(binding)[:2] != _parse_version(runtime)[:2]:
        raise RuntimeError(
            "TensorRT Python binding/runtime mismatch: "
            f"binding={binding}, libnvinfer={runtime}. Install a binding matching "
            "the local libnvinfer major/minor version."
        )
    return runtime


def gpu_fingerprint(device: torch.device | str = "cuda:0") -> dict[str, object]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TensorRT physical deployment requires CUDA")
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory": properties.total_memory,
        "torch_cuda": torch.version.cuda,
    }


class TensorRTStepRunner:
    """管理执行上下文与 CUDA 张量；具体输入、输出和清单校验由模型后端实现。"""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | str = "cuda:0",
        use_cuda_graph: bool = True,
        require_manifest: bool = True,
    ) -> None:
        self.engine_path = Path(engine_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TensorRTStepRunner requires a CUDA device")
        try:
            prepare_tensorrt_libraries()
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT Python bindings are missing. Install bindings matching the "
                "deployment machine's libnvinfer before starting physical mode."
            ) from exc
        self.linked_tensorrt_version = validate_tensorrt_installation(trt)
        self.trt = trt
        self.manifest = self._validate_manifest(require_manifest)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self._lock = threading.Lock()
        self._buffers: dict[str, torch.Tensor] = {}
        self._input_names: set[str] = set()
        self._output_name: str | None = None
        self._allocate_and_bind()
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        if use_cuda_graph:
            self._try_capture_cuda_graph()

    @staticmethod
    def _torch_dtype(np_dtype: np.dtype) -> torch.dtype:
        mapping = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.bool_): torch.bool,
        }
        try:
            return mapping[np_dtype]
        except KeyError as exc:
            raise RuntimeError(f"unsupported TensorRT tensor dtype: {np_dtype}") from exc

    def _execute(self) -> None:
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

    def _try_capture_cuda_graph(self) -> None:
        try:
            self._execute()
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._execute()
            self.cuda_graph = graph
        except Exception:
            self.cuda_graph = None

    def _validate_manifest(self, required):
        raise NotImplementedError("模型后端必须校验自己的引擎清单")

    def _allocate_and_bind(self):
        raise NotImplementedError("模型后端必须绑定自己的输入输出契约")
