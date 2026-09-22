"""BUMI 文本按契约120或300帧 TensorRT 单步后端。

复用原部署的运行库发现、GPU指纹和执行上下文，实现文本六输入与动作/接触双输出。
仅接受与ONNX、来源checkpoint、GPU和运行库匹配的engine；不连接任何控制器。
"""

import json

import numpy as np
import torch

from gem.runtime.bumi_text_contract import sha256_file
from gem.runtime.bumi_text_runtime import INPUTS, OUTPUTS, io_shapes
from gem.runtime.music_only_trt import TensorRTStepRunner, gpu_fingerprint

ENGINE_SCHEMA = "genmo.bumi_text_engine.v1"


class TextTensorRTStep(TensorRTStepRunner):
    REQUIRED_INPUTS = INPUTS

    def __init__(self, engine_path, *, onnx_metadata, device="cuda:0"):
        self.inputs, self.outputs = io_shapes(onnx_metadata["model_contract"])
        self.REQUIRED_INPUTS = self.inputs
        self.onnx_metadata = onnx_metadata
        super().__init__(engine_path, device=device, use_cuda_graph=False, require_manifest=True)

    def _validate_manifest(self, required):
        value = json.loads(self.engine_path.with_suffix(".json").read_text())
        expected = dict(
            schema=ENGINE_SCHEMA,
            engine_sha256=sha256_file(self.engine_path),
            onnx_sha256=self.onnx_metadata["onnx_sha256"],
            source_checkpoint_sha256=self.onnx_metadata["source_checkpoint_sha256"],
            gpu=gpu_fingerprint(self.device),
        )
        for key, item in expected.items():
            if value.get(key) != item:
                raise ValueError(f"TensorRT engine {key} 与当前模型/设备不符")
        for key, actual in [
            ("tensorrt_version", str(self.trt.__version__)),
            ("libnvinfer_version", self.linked_tensorrt_version),
        ]:
            if value[key].split(".")[:2] != actual.split(".")[:2]:
                raise ValueError(f"TensorRT {key} 不兼容")
        return value

    def _allocate_and_bind(self):
        expected = {**self.inputs, **self.outputs}
        seen = set()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if name not in expected or shape != expected[name]:
                raise ValueError(f"TensorRT接口错误: {name} {shape}")
            is_input = self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
            if is_input != (name in INPUTS):
                raise ValueError("TensorRT输入输出方向错误")
            dtype = self._torch_dtype(np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name))))
            if name in OUTPUTS and dtype != torch.float32:
                raise ValueError("TensorRT输出必须为FP32")
            value = torch.empty(shape, dtype=dtype, device=self.device)
            self._buffers[name] = value
            self.context.set_tensor_address(name, value.data_ptr())
            seen.add(name)
        if seen != set(expected):
            raise ValueError("TensorRT缺少文本输入或接触输出")

    def __call__(self, *args):
        if len(args) != len(INPUTS):
            raise ValueError("文本TensorRT需要六个输入")
        with self._lock:
            for name, value in zip(INPUTS, args):
                if tuple(value.shape) != self.inputs[name]:
                    raise ValueError(f"TensorRT {name} 形状错误")
                self._buffers[name].copy_(
                    value.to(device=self.device, dtype=self._buffers[name].dtype)
                )
            self._execute()
            return tuple(self._buffers[name].clone() for name in OUTPUTS)
