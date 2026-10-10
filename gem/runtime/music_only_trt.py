# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Stage1部署共用的TensorRT资源管理、文件指纹及音频帧数工具。

本文件仅保留部署需要的CUDA环境、ABI检查、持久张量绑定和可选CUDA Graph。
具体Stage1输入输出契约和元数据校验由BUMI运行器提供，底层不定义旧五输入网络。
CUDA Graph捕获前将所有缓冲初始化为有限值，捕获后每次调用仍更新全部输入。
TensorRT内部为四个掩码分配INT32缓冲，子类在反序列化前加载对应CUDA选择插件；
模型外部仍接收bool掩码，不增加输入输出，保留持久缓冲和同一CUDA stream。
DDIM、历史条件、前缀回填和动作解码由Stage1生成器处理，不保留旧SMPL151D或
纯音乐overlap-add生成器。音频帧数继续按30Hz半开时间区间计算。
"""

from __future__ import annotations

import hashlib
import math
import threading
from pathlib import Path

import numpy as np
import torch

from gem.runtime.bumi_music_contract import SOURCE_FPS
from gem.runtime.tensorrt_environment import linked_tensorrt_version, prepare_tensorrt_libraries


def validate_tensorrt_installation(trt_module) -> str:
    """检查Python绑定与实际链接运行库的主次版本，拒绝ABI混用。"""
    binding = str(trt_module.__version__)
    runtime = linked_tensorrt_version()
    if binding.split(".")[:2] != runtime.split(".")[:2]:
        raise RuntimeError(f"TensorRT ABI不匹配: binding={binding}, libnvinfer={runtime}")
    return runtime


def sha256_file(path: str | Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def gpu_fingerprint(device: torch.device | str = "cuda:0") -> dict[str, object]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TensorRT需要CUDA设备")
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory": properties.total_memory,
        "torch_cuda": torch.version.cuda,
    }


def exact_motion_frame_count(feature_frames: int, duration_sec: float | None,
                             *, fps: int = SOURCE_FPS) -> int:
    """保持原控制台半开区间帧数，例如30秒对应900帧。"""
    if feature_frames <= 0 or fps <= 0:
        raise ValueError("特征帧数与fps必须为正数")
    if duration_sec is None:
        return int(feature_frames)
    if not math.isfinite(float(duration_sec)) or duration_sec <= 0:
        raise ValueError("播放时长必须为有限正数")
    requested = max(1, int(math.floor(float(duration_sec) * fps + 1e-7)))
    if requested > feature_frames:
        raise ValueError(f"音频只有{feature_frames}帧特征，少于请求的{requested}帧")
    return requested


class TensorRTStepRunner:
    """只管理TensorRT上下文和持久张量，子类提供唯一网络契约。"""

    REQUIRED_INPUTS = {}
    REQUIRED_OUTPUTS = {}
    REQUIRED_DTYPES = {}

    def __init__(self, engine_path, *, device="cuda:0", use_cuda_graph=True,
                 require_manifest=True):
        self.engine_path = Path(engine_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TensorRT运行器需要CUDA")
        torch.cuda.set_device(self.device)
        prepare_tensorrt_libraries()
        import tensorrt as trt

        self.trt = trt
        self.linked_tensorrt_version = validate_tensorrt_installation(trt)
        self.manifest = self._validate_manifest(require_manifest)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT无法加载引擎: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT无法创建执行上下文")
        self._lock = threading.Lock()
        self._buffers = {}
        self._allocate_and_bind()
        self.cuda_graph = None
        if use_cuda_graph:
            self._try_capture_cuda_graph()

    def _validate_manifest(self, required):
        raise NotImplementedError

    def _allocate_and_bind(self):
        trt = self.trt
        actual_inputs, actual_outputs = {}, {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
            if any(x <= 0 for x in shape):
                raise RuntimeError(f"TensorRT必须固定形状: {name}={shape}")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            expected_dtype = self.REQUIRED_DTYPES.get(name)
            if expected_dtype is None or dtype != np.dtype(expected_dtype):
                raise RuntimeError(f"TensorRT类型不匹配: {name}={dtype}")
            torch_dtype = {np.dtype("float32"): torch.float32,
                           np.dtype("int64"): torch.int64,
                           np.dtype("int32"): torch.int32,
                           np.dtype("bool"): torch.bool}[dtype]
            tensor = torch.zeros(shape, dtype=torch_dtype, device=self.device)
            self._buffers[name] = tensor
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT张量绑定失败: {name}")
            target = (actual_inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                      else actual_outputs)
            target[name] = shape
        if actual_inputs != self.REQUIRED_INPUTS or actual_outputs != self.REQUIRED_OUTPUTS:
            raise RuntimeError(f"TensorRT接口不匹配: inputs={actual_inputs}, outputs={actual_outputs}")
        # 捕获时使用有音乐、无历史/前缀的合法条件；不能执行未初始化掩码或时间步。
        self._buffers["music_valid"].fill_(True)
        self._buffers["future_valid"].fill_(True)
        self._buffers["guidance_scale"].fill_(2.5)
        self._buffers["diffusion_timestep"].fill_(999)

    def _execute(self):
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT执行失败")

    def _try_capture_cuda_graph(self):
        try:
            self._execute()
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._execute()
            self.cuda_graph = graph
        except Exception:
            # 捕获不支持时仍使用同一TensorRT引擎，不切换后端。
            self.cuda_graph = None
