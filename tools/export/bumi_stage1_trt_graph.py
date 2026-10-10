"""将当前Stage1原始ONNX转换为仅供TensorRT编译的独立掩码插件图。

原始ONNX及元数据只读；模型权重、输入输出名称/尺寸、归一化、CFG和前缀语义保持。
由于TensorRT IPluginV3不支持BOOL，部署专用图将全部布尔张量改为0/1 INT32。
Where、Cast和布尔比较/逻辑通过BumiStage1Mask CUDA插件执行，形状搬运仍由原算子
处理。运行器在持久缓冲拷贝时将四个bool掩码转换为INT32，外部Stage1调用接口不变。

先用ONNX静态形状推导建立类型/尺寸表，再仅折叠小型静态形状/逻辑常量，移除
Expand等尺寸路径中的常量Where，避免插件被误用作宿主形状计算。未进行模型前向，
不使用ReferenceEvaluator、ONNX Runtime或数值验收；浮点Add/Sub/Mul/Div不折叠。
保留动态Where的原分支选择，尤其历史有效性、已知坐标、注意力负无穷和最终补零。
生成派生图和network_lowering报告，并校验公开尺寸与私有engine类型；不声称数值已对齐。

同文件提供插件构建：使用目标GPU架构、当前实际映射的libnvinfer和匹配SDK头文件，
通过nvcc生成共享库；编译身份与.so哈希进入缓存。此过程只在用户手工运行构建入口
时发生，不自动构建engine或运行推理；旧融合规避策略和旧GENMO网络均不支持。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from gem.runtime.bumi_music_contract import (
    BUMI_ONNX_INPUTS, BUMI_ONNX_OUTPUTS, BUMI_TRT_INPUT_DTYPES,
)
from gem.runtime.bumi_stage1_plugin import (
    PLUGIN_FILENAME, PLUGIN_NAME, PLUGIN_NAMESPACE, PLUGIN_VERSION, plugin_sha256,
)
from gem.runtime.tensorrt_environment import loaded_nvinfer_paths

_OPS = {"Where": 1, "Cast": 2, "Equal": 4, "Less": 5, "Greater": 6,
        "And": 7, "Or": 8, "Not": 9}
_SMALL_LIMIT = 4096


def _attributes(node, helper):
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _small_shape(shape):
    return all(int(value) >= 0 for value in shape) and math.prod(shape) <= _SMALL_LIMIT


def _static_value(node, constants, shapes, onnx):
    """只解释已知小常量及静态Shape，不执行包含模型输入的图或浮点算术。"""
    op, attrs = node.op_type, _attributes(node, onnx.helper)
    if op == "Constant":
        tensor = attrs.get("value")
        if tensor is None or not _small_shape(tuple(tensor.dims)):
            return None
        return onnx.numpy_helper.to_array(tensor)
    if op == "Shape":
        shape = shapes.get(node.input[0])
        if shape is None or any(value < 0 for value in shape):
            return None
        return np.asarray(shape, dtype=np.int64)[int(attrs.get("start", 0)):attrs.get("end")]
    whitelist = {"Identity", "Cast", "Gather", "Unsqueeze", "Squeeze", "Concat",
                 "Reshape", "Transpose", "Slice", "ConstantOfShape", "Expand",
                 "Equal", "Less", "Greater", "And", "Or", "Not", "Where", "Add", "Sub", "Mul"}
    if op not in whitelist or any(name and name not in constants for name in node.input):
        return None
    values = [constants.get(name) for name in node.input]
    x = values[0]
    if op in {"Add", "Sub", "Mul"}:
        if any(value.dtype.kind not in "iu" for value in values):
            return None
        return {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply}[op](*values)
    if op == "Identity":
        return x
    if op == "Cast":
        return x.astype(onnx.helper.tensor_dtype_to_np_dtype(attrs["to"]))
    if op == "Gather":
        return np.take(x, values[1], axis=int(attrs.get("axis", 0)))
    if op == "Unsqueeze":
        axes = values[1].tolist() if len(values) > 1 else attrs["axes"]
        rank = x.ndim + len(axes)
        for axis in sorted(int(axis) % rank for axis in axes):
            x = np.expand_dims(x, axis)
        return x
    if op == "Squeeze":
        axes = values[1].tolist() if len(values) > 1 and values[1] is not None else attrs.get("axes")
        return np.squeeze(x, axis=None if axes is None else tuple(int(axis) for axis in axes))
    if op == "Concat":
        return np.concatenate(values, axis=int(attrs["axis"]))
    if op == "Reshape":
        shape = [int(value) for value in values[1]]
        if not attrs.get("allowzero", 0):
            shape = [x.shape[index] if value == 0 else value for index, value in enumerate(shape)]
        return np.reshape(x, shape)
    if op == "Transpose":
        return np.transpose(x, attrs.get("perm"))
    if op == "Slice":
        starts, ends = values[1], values[2]
        axes = values[3] if len(values) > 3 and values[3] is not None else np.arange(len(starts))
        steps = values[4] if len(values) > 4 and values[4] is not None else np.ones(len(starts), dtype=np.int64)
        selections = [slice(None)] * x.ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            selections[int(axis)] = slice(int(start), int(end), int(step))
        return x[tuple(selections)]
    if op in {"ConstantOfShape", "Expand"}:
        shape = tuple(int(value) for value in (x if op == "ConstantOfShape" else values[1]))
        if not _small_shape(shape):
            return None
        if op == "Expand":
            shape = np.broadcast_shapes(x.shape, shape)
            return np.broadcast_to(x, shape) if _small_shape(shape) else None
        value = onnx.numpy_helper.to_array(attrs["value"]) if "value" in attrs else np.asarray([0], dtype=np.float32)
        return np.full(shape, value.reshape(-1)[0], dtype=value.dtype)
    if op == "Where":
        return np.where(*values)
    if op == "Not":
        return np.logical_not(x)
    operation = {"Equal": np.equal, "Less": np.less, "Greater": np.greater,
                 "And": np.logical_and, "Or": np.logical_or}.get(op)
    return operation(*values) if operation else None


def _prune_graph(graph):
    needed = {value.name for value in graph.output}
    kept = []
    for node in reversed(graph.node):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    initializers = [tensor for tensor in graph.initializer if tensor.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    live = needed | {name for node in graph.node for name in node.output}
    values = [value for value in graph.value_info if value.name in live]
    del graph.value_info[:]
    graph.value_info.extend(values)


def lower_stage1_onnx(source, destination, trt):
    """生成编译专用图；原文件不改写，不加载checkpoint，不执行模型推理。"""
    import onnx

    # ONNX shape inference属于编译期类型/尺寸推导，不是模型运行或数值对齐。
    model = onnx.shape_inference.infer_shapes(onnx.load(str(source)), strict_mode=True)
    graph = model.graph
    if model.functions or graph.sparse_initializer or any(
            attribute.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in graph.node for attribute in node.attribute):
        raise ValueError("Stage1编译器只接受当前固定图，不支持函数/控制流/稀疏权重")
    types, shapes = {}, {}
    for value in [*graph.input, *graph.output, *graph.value_info]:
        tensor = value.type.tensor_type
        types[value.name] = tensor.elem_type
        if tensor.HasField("shape"):
            shapes[value.name] = tuple(dimension.dim_value if dimension.HasField("dim_value") else -1
                                       for dimension in tensor.shape.dim)
    constants = {}
    for tensor in graph.initializer:
        types[tensor.name], shapes[tensor.name] = tensor.data_type, tuple(tensor.dims)
        if _small_shape(tuple(tensor.dims)):
            constants[tensor.name] = onnx.numpy_helper.to_array(tensor)
    folded, rewritten = [], []
    for node in graph.node:
        value = _static_value(node, constants, shapes, onnx)
        if value is not None and _small_shape(value.shape) and len(node.output) == 1:
            constants[node.output[0]] = value
            shapes[node.output[0]] = value.shape
            types[node.output[0]] = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
            if node.op_type != "Constant":
                folded.append({"node": node.name, "operator": node.op_type,
                               "output": node.output[0], "shape": list(value.shape)})
                node = onnx.helper.make_node("Constant", [], list(node.output), name=node.name,
                    value=onnx.numpy_helper.from_array(np.asarray(value)))
        rewritten.append(node)
    del graph.node[:]
    graph.node.extend(rewritten)
    _prune_graph(graph)
    for tensor in graph.initializer:
        if tensor.data_type == onnx.TensorProto.BOOL:
            tensor.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(tensor).astype(np.int32), tensor.name))
    for value in [*graph.input, *graph.output, *graph.value_info]:
        if value.type.tensor_type.elem_type == onnx.TensorProto.BOOL:
            value.type.tensor_type.elem_type = onnx.TensorProto.INT32
    plugin_counts = Counter()
    plugin_nodes = []
    output_types = {onnx.TensorProto.FLOAT: int(trt.float32),
                    onnx.TensorProto.INT32: int(trt.int32), onnx.TensorProto.INT64: int(trt.int64)}
    for node in graph.node:
        # 同时覆盖ConstantOfShape等算子的张量属性，避免常量中残留BOOL表示。
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == onnx.TensorProto.BOOL:
                attr.t.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(attr.t).astype(np.int32)))
        if node.op_type not in _OPS:
            continue
        if len(node.output) != 1:
            raise ValueError(f"Stage1插件操作必须有单个输出: {node.name}")
        original = node.op_type
        operation = _OPS[original]
        attrs = _attributes(node, onnx.helper)
        dtype = types.get(node.output[0])
        if original == "Cast":
            dtype = int(attrs["to"])
            if dtype == onnx.TensorProto.BOOL:
                operation = 3
        if dtype == onnx.TensorProto.BOOL:
            dtype = onnx.TensorProto.INT32
        if dtype not in output_types:
            raise ValueError(f"Stage1插件输出类型不支持: {node.name}={dtype}")
        replacement = onnx.helper.make_node(PLUGIN_NAME, list(node.input), list(node.output),
            name=node.name, domain=PLUGIN_NAMESPACE, operation=operation,
            output_type=output_types[dtype], plugin_namespace=PLUGIN_NAMESPACE,
            plugin_version=PLUGIN_VERSION)
        node.CopyFrom(replacement)
        plugin_counts[original] += 1
        plugin_nodes.append({"node": node.name, "operator": original,
                             "output": node.output[0], "operation": operation})
    if not plugin_counts["Where"] or not plugin_counts["Cast"]:
        raise ValueError("没有匹配当前Stage1的动态Where/Cast，拒绝使用其他网络")
    for value in graph.input:
        shape = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        actual_type = str(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)))
        if shape != BUMI_ONNX_INPUTS.get(value.name) or actual_type != BUMI_TRT_INPUT_DTYPES.get(value.name):
            raise ValueError(f"Stage1派生图输入形状/类型不匹配: {value.name}")
    for value in graph.output:
        if ([dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
                != BUMI_ONNX_OUTPUTS.get(value.name) or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
            raise ValueError(f"Stage1派生图输出不匹配: {value.name}")
    if any(value.type.tensor_type.elem_type == onnx.TensorProto.BOOL
           for value in [*graph.input, *graph.output, *graph.value_info]):
        raise ValueError("Stage1派生图仍含BOOL类型")
    if any(node.op_type in _OPS for node in graph.node):
        raise ValueError("Stage1派生图仍含原生Where/Cast/布尔算子")
    model.opset_import.add(domain=PLUGIN_NAMESPACE, version=1)
    destination = Path(destination)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        onnx.save_model(model, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source_onnx_sha256": plugin_sha256(source),
            "onnx_version": onnx.__version__, "numpy_version": np.__version__,
            "derived_onnx": destination.name, "derived_onnx_sha256": plugin_sha256(destination),
            "mask_encoding": "int32_zero_or_one", "input_dtypes": BUMI_TRT_INPUT_DTYPES,
            "folded_constants": folded, "plugin_counts": dict(plugin_counts),
            "plugin_nodes": plugin_nodes, "inference_validation": "not_run"}


def compile_stage1_plugin(source, output, *, nvcc, include_dir, trt_version, lib_version, gpu,
                          overwrite=False):
    """仅在用户运行构建时编译.so；编译器/头文件/实际ABI/架构均进入身份。"""
    source, include_dir = Path(source).resolve(strict=True), Path(include_dir).resolve(strict=True)
    compiler = shutil.which(str(nvcc)) or (str(Path(nvcc).resolve()) if Path(nvcc).is_file() else None)
    if compiler is None:
        raise FileNotFoundError("缺少nvcc；使用--nvcc指定CUDA Toolkit编译器，不自动安装或更换CUDA")
    compiler = Path(compiler).resolve(strict=True)
    header = (include_dir / "NvInferVersion.h").read_text(encoding="utf-8")
    components = []
    for name in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        match = re.search(rf"^#define TRT_{name}_ENTERPRISE\s+(\d+)", header, re.MULTILINE)
        if match is None:
            match = re.search(rf"^#define NV_TENSORRT_{name}\s+(\d+)", header, re.MULTILINE)
        if match is None:
            raise ValueError(f"无法读取TensorRT SDK版本: {include_dir}")
        components.append(match.group(1))
    if ".".join(components) != trt_version or ".".join(components[:3]) != lib_version:
        raise ValueError(f"TensorRT SDK与实际运行库版本不匹配: {'.'.join(components)} != {trt_version}")
    libraries = loaded_nvinfer_paths(Path("/proc/self/maps").read_text())
    if len(libraries) != 1:
        raise RuntimeError("插件编译必须使用当前进程唯一的libnvinfer实际映射路径")
    library = libraries[0].resolve(strict=True)
    identity = {"link_policy": "exact_library_host_linker_v1",
                "source_sha256": plugin_sha256(source), "nvcc": str(compiler),
                "nvcc_sha256": plugin_sha256(compiler), "tensorrt_version": trt_version,
                "libnvinfer_version": lib_version,
                "libnvinfer": str(library), "gpu": gpu,
                "sdk_sha256": {name: plugin_sha256(include_dir / name)
                               for name in ("NvInfer.h", "NvInferRuntime.h", "NvInferRuntimePlugin.h",
                                            "NvInferRuntimeBase.h", "NvInferPluginBase.h", "NvInferVersion.h")}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(output) / "plugins" / key
    directory.mkdir(parents=True, exist_ok=True)
    target, metadata = directory / PLUGIN_FILENAME, directory / "plugin_build.json"
    if target.is_file() and metadata.is_file():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("identity") == identity and record.get("sha256") == plugin_sha256(target):
            return target, record
        if not overwrite:
            raise RuntimeError(f"插件缓存身份/哈希不匹配，保留原文件；显式--overwrite或更换输出目录: {directory}")
    elif target.exists() or metadata.exists():
        if not overwrite:
            raise RuntimeError(f"插件编译缓存不完整，请保留目录并检查或显式--overwrite: {directory}")
    arch = "".join(str(value) for value in gpu["compute_capability"])
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}.so")
    command = [str(compiler), "-std=c++17", "-O2", "-shared", "-Xcompiler=-fPIC",
               "--fmad=false", "-gencode", f"arch=compute_{arch},code=[sm_{arch},compute_{arch}]",
               "-I", str(include_dir), str(source), "-o", str(temporary),
               # 带版本号的.so不能作为nvcc位置输入；直接交给主机链接器，仍绑定实际映射库。
               "-Xlinker", str(library),
               "-Xlinker", "-rpath", "-Xlinker", str(library.parent),
               "-Xlinker", "-rpath", "-Xlinker", str(compiler.parent.parent / "lib64")]
    print(f"编译Stage1 CUDA掩码插件: nvcc={compiler}, sm={arch}, SDK={trt_version}", flush=True)
    try:
        subprocess.run(command, check=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    record = {"identity": identity, "command": command, "sha256": plugin_sha256(target),
              "inference_validation": "not_run"}
    pending = metadata.with_name(metadata.name + f".tmp.{os.getpid()}")
    try:
        pending.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(metadata)
    finally:
        pending.unlink(missing_ok=True)
    return target, record
