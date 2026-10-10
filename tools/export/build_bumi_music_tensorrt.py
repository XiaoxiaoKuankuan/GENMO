#!/usr/bin/env python3
"""为当前Stage1构建独立CUDA掩码插件、FP32 TensorRT engine及七资产部署包。

针对用户三次构建中的Myelin内部失败，放弃debug融合边界和Select广播规避策略。
先编译IPluginV3 CUDA实现，再生成仅供TensorRT的派生ONNX：将布尔值编码为0/1
INT32，Where/Cast/逻辑比较由插件执行；原始ONNX、元数据、权重和训练目录只读。
外部仍为Stage1十一输入字典及两个FP32输出，运行器负责四个bool掩码的INT32拷贝。
固定batch1/T120/H50，时间步仍INT64，关闭TF32/FP16，原DDIM与在线/buffered流程不变。

插件、派生图、代码版本和原模型身份都进入缓存及报告；运行时加载绑定的.so。
构建前保存network_lowering.json，失败不发布新的deployment.json，不清理旧缓存。
成功后打包原ONNX/其元数据、engine/其元数据、统计、运动学、插件，共七项资产。
本入口只在用户手工执行时编译、改写和构建；不加载训练checkpoint，不执行模型
推理预热、数值对齐、pytest、demo、仿真或实机。构建可用性及数值/性能验收另行确认。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION
from gem.runtime.bumi_deployment_bundle import BUMI_DEPLOYMENT_CONTRACT
from gem.runtime.bumi_music_contract import (
    BUMI_ENGINE_CONTRACT, BUMI_ONNX_INPUTS, BUMI_ONNX_INPUT_DTYPES, BUMI_TRT_INPUT_DTYPES,
    BUMI_ONNX_OUTPUTS, BUMI_ONNX_OUTPUT_DTYPES, BUMI_STAGE1_PRECISION_POLICY,
    BUMI_STAGE1_SAMPLING_CONTRACT_VERSION, BUMI_STAGE1_TRT_BUILD_POLICY,
    validate_stage1_engine_build_options,
)
from gem.runtime.bumi_music_deploy import bumi_engine_cache_key, validate_stage1_metadata
from gem.runtime.bumi_stage1_plugin import PLUGIN_FILENAME, load_stage1_plugin
from gem.runtime.music_only_trt import gpu_fingerprint, sha256_file, validate_tensorrt_installation
from gem.runtime.tensorrt_environment import prepare_tensorrt_libraries
from bumi_stage1_trt_graph import compile_stage1_plugin, lower_stage1_onnx


def atomic_json(path, payload):
    """只替换本工具的目标JSON，避免读者看到不完整清单。"""
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def network_contract(network, trt):
    """检查十一输入/两输出尺寸和INT32掩码的私有engine类型，不运行模型。"""
    for tensors, shapes, dtypes in (
        ([network.get_input(i) for i in range(network.num_inputs)],
         BUMI_ONNX_INPUTS, BUMI_TRT_INPUT_DTYPES),
        ([network.get_output(i) for i in range(network.num_outputs)],
         BUMI_ONNX_OUTPUTS, BUMI_ONNX_OUTPUT_DTYPES),
    ):
        if {tensor.name: list(tensor.shape) for tensor in tensors} != shapes:
            raise ValueError("TensorRT解析网络不是当前Stage1固定接口")
        for tensor in tensors:
            actual = str(np.dtype(trt.nptype(tensor.dtype)))
            if actual != dtypes[tensor.name]:
                raise ValueError(f"TensorRT输入输出类型不符: {tensor.name}={actual}")
    remaining = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if (tensor is not None and tensor.dtype == trt.bool
                    and tensor.is_execution_tensor and not tensor.is_shape_tensor):
                remaining.append({"layer": layer.name, "tensor": tensor.name})
    if remaining:
        raise ValueError(f"派生图解析后仍包含原生BOOL执行路径，停止构建: {remaining}")


def copy_asset(source, destination, expected_sha, overwrite):
    """按完整指纹复制，源文件不移动，旧模型目录不清理。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) == expected_sha:
            return destination
        if not overwrite:
            raise FileExistsError(f"拒绝覆盖不同资产: {destination}；换输出目录或显式--overwrite")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_sha:
            raise RuntimeError(f"复制期间资产内容改变: {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-metadata", type=Path)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--optimization-level", type=int, default=0,
                        help="0..5；掩码插件策略始终启用，等级不表示已关闭整个Myelin后端")
    parser.add_argument("--nvcc", default=os.environ.get("BUMI_STAGE1_NVCC")
                        or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    parser.add_argument("--trt-include-dir", type=Path,
                        default=os.environ.get("BUMI_STAGE1_TRT_INCLUDE", "/usr/include/x86_64-linux-gnu"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.workspace_gib) or args.workspace_gib <= 0:
        parser.error("--workspace-gib必须为有限正数")
    if not 0 <= args.optimization_level <= 5:
        parser.error("--optimization-level必须为0..5")
    workspace_bytes = int(args.workspace_gib * 1024**3)
    if workspace_bytes <= 0:
        parser.error("--workspace-gib换算后必须至少为1字节")
    onnx = args.onnx.expanduser().resolve(strict=True)
    metadata_path = (args.onnx_metadata or onnx.with_suffix(onnx.suffix + ".json")).expanduser().resolve(strict=True)
    stats = args.stats.expanduser().resolve(strict=True)
    kin = args.kinematics.expanduser().resolve(strict=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_stage1_metadata(metadata)
    onnx_sha, metadata_sha = sha256_file(onnx), sha256_file(metadata_path)
    stats_sha, kin_sha = sha256_file(stats), sha256_file(kin)
    if metadata.get("onnx_sha256") != onnx_sha or metadata.get("onnx_size_bytes") != onnx.stat().st_size:
        raise ValueError("ONNX文件与Stage1导出元数据不一致")
    identity = metadata["asset_identity"]
    if identity.get("stats_sha256") != stats_sha or identity.get("kinematics_sha256") != kin_sha:
        raise ValueError("统计量/运动学资源与Stage1模型不一致")
    if json.loads(stats.read_text(encoding="utf-8")).get("kinematics_sha256") != kin_sha:
        raise ValueError("stats自身绑定的运动学资源不一致")
    checkpoint_sha = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or any(
            value not in "0123456789abcdef" for value in checkpoint_sha):
        raise ValueError("Stage1导出元数据缺少checkpoint来源SHA256")

    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("构建Stage1 TensorRT引擎需要CUDA")
    torch.cuda.set_device(device)
    prepare_tensorrt_libraries()
    import tensorrt as trt

    lib_version = validate_tensorrt_installation(trt)
    gpu = gpu_fingerprint(device)
    output = args.output_dir.expanduser().resolve()
    plugin_source = Path(__file__).with_name("bumi_stage1_mask_plugin.cu")
    graph_source = Path(__file__).with_name("bumi_stage1_trt_graph.py")
    compiled_plugin, plugin_build = compile_stage1_plugin(
        plugin_source, output, nvcc=args.nvcc, include_dir=args.trt_include_dir,
        trt_version=str(trt.__version__), lib_version=lib_version, gpu=gpu,
        overwrite=args.overwrite)
    build_options = validate_stage1_engine_build_options({
        "policy": BUMI_STAGE1_TRT_BUILD_POLICY,
        "optimization_level": args.optimization_level, "workspace_bytes": workspace_bytes,
        "plugin_source_sha256": sha256_file(plugin_source),
        "plugin_library_sha256": plugin_build["sha256"],
        "lowering_source_sha256": sha256_file(graph_source),
    })
    cache_key = bumi_engine_cache_key(
        onnx_sha256=onnx_sha, checkpoint_sha256=checkpoint_sha, metadata_sha256=metadata_sha,
        tensorrt_version=trt.__version__, precision="fp32", gpu=gpu, build_options=build_options)
    cache = output / "engines" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    engine = cache / "bumi_stage1_denoiser.engine"
    engine_metadata = cache / "engine.json"
    plugin = copy_asset(compiled_plugin, cache / PLUGIN_FILENAME,
                        build_options["plugin_library_sha256"], args.overwrite)
    plugin_record = {"path": plugin.name, "sha256": sha256_file(plugin)}
    engine_info = None
    if engine.exists() or engine_metadata.exists():
        if not (engine.is_file() and engine_metadata.is_file()) and not args.overwrite:
            raise RuntimeError("引擎缓存不完整；请显式--overwrite重建")
        if engine.is_file() and engine_metadata.is_file() and not args.overwrite:
            candidate = json.loads(engine_metadata.read_text(encoding="utf-8"))
            if (candidate.get("cache_key") != cache_key
                    or candidate.get("contract_version") != BUMI_ENGINE_CONTRACT
                    or candidate.get("build_options") != build_options
                    or candidate.get("plugin_library") != plugin_record
                    or candidate.get("engine_sha256") != sha256_file(engine)):
                raise RuntimeError("引擎缓存身份不匹配；请显式--overwrite重建")
            engine_info = candidate
    if engine_info is None:
        derived = cache / "bumi_stage1_denoiser.trt.onnx"
        print(f"Stage1源ONNX（只读）: {onnx}", flush=True)
        lowering = lower_stage1_onnx(onnx, derived, trt)
        if lowering["source_onnx_sha256"] != onnx_sha:
            raise RuntimeError("Stage1源ONNX在图转换期间改变，停止构建，不发布新清单")
        lowering.update({"build_options": build_options, "plugin_build": plugin_build,
                         "inference_validation": "not_run"})
        lowering_path = cache / "network_lowering.json"
        atomic_json(lowering_path, lowering)
        print(f"Stage1 TensorRT构建策略: {build_options}", flush=True)
        print(f"静态常量折叠={len(lowering['folded_constants'])}；独立CUDA插件={lowering['plugin_counts']}；"
              "掩码编码=0/1 INT32；公开名称/尺寸保持11输入/2输出", flush=True)
        print(f"网络改写报告: {lowering_path}", flush=True)
        load_stage1_plugin(plugin, plugin_record["sha256"], trt)
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = trt.OnnxParser(network, logger)
        if not onnx_parser.parse_from_file(str(derived)):
            errors = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
            raise RuntimeError(f"Stage1派生ONNX解析失败；报告={lowering_path}:\n" + errors)
        network_contract(network, trt)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        config.builder_optimization_level = args.optimization_level
        for flag in (trt.BuilderFlag.TF32, trt.BuilderFlag.FP16, trt.BuilderFlag.INT8):
            config.clear_flag(flag)
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                "Stage1 FP32引擎构建失败，未发布新的deployment.json；"
                f"策略={build_options['policy']}，插件={lowering['plugin_counts']}；"
                f"报告={lowering_path}；请保留完整TensorRT日志，不能据此认为数值或构建已通过")
        temporary = engine.with_name(engine.name + f".tmp.{os.getpid()}")
        try:
            temporary.write_bytes(bytes(serialized))
            temporary.replace(engine)
        finally:
            temporary.unlink(missing_ok=True)
        engine_info = {
            "contract_version": BUMI_ENGINE_CONTRACT,
            "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
            "cache_key": cache_key, "engine": engine.name, "engine_sha256": sha256_file(engine),
            "onnx_sha256": onnx_sha, "onnx_metadata_sha256": metadata_sha,
            "checkpoint_sha256": checkpoint_sha, "stats_sha256": stats_sha,
            "kinematics_sha256": kin_sha, "tensorrt_version": trt.__version__,
            "libnvinfer_version": lib_version, "gpu": gpu, "precision": "fp32",
            "precision_policy": BUMI_STAGE1_PRECISION_POLICY, "build_options": build_options,
            "plugin_library": plugin_record, "derived_onnx_sha256": lowering["derived_onnx_sha256"],
            "network_lowering_report": lowering_path.name,
            "network_lowering_report_sha256": sha256_file(lowering_path),
            "input_shapes": BUMI_ONNX_INPUTS, "input_dtypes": BUMI_TRT_INPUT_DTYPES,
            "source_input_dtypes": BUMI_ONNX_INPUT_DTYPES,
            "output_shapes": BUMI_ONNX_OUTPUTS, "output_dtypes": BUMI_ONNX_OUTPUT_DTYPES,
            "build_seconds": time.perf_counter() - started,
            "workspace_gib": args.workspace_gib, "optimization_level": args.optimization_level,
            "inference_validation": "not_run",
        }
        atomic_json(engine_metadata, engine_info)
    assets = {
        "onnx": copy_asset(onnx, output / "bumi_stage1_denoiser.onnx", onnx_sha, args.overwrite),
        "onnx_metadata": copy_asset(metadata_path, output / "bumi_stage1_denoiser.onnx.json",
                                    metadata_sha, args.overwrite),
        "stats": copy_asset(stats, output / "assets" / "qpos30_train_stats.json", stats_sha, args.overwrite),
        "kinematics": copy_asset(kin, output / "assets" / "bumi_kinematics.json", kin_sha, args.overwrite),
        "engine": engine, "engine_metadata": engine_metadata, "engine_plugin": plugin,
    }
    payload = {
        "contract_version": BUMI_DEPLOYMENT_CONTRACT,
        "source_checkpoint_sha256": checkpoint_sha, "global_step": metadata.get("global_step"),
        "sampling_contract_version": BUMI_STAGE1_SAMPLING_CONTRACT_VERSION,
        "assets": {name: {"path": str(path.relative_to(output)),
                          "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                   for name, path in assets.items()},
        "inference_validation": "not_run",
    }
    manifest = output / "deployment.json"
    atomic_json(manifest, payload)
    print(f"Stage1 FP32 engine: {engine}")
    print(f"Stage1 CUDA mask plugin: {plugin}")
    print(f"Stage1 deployment manifest: {manifest}")
    print("本构建入口只编译/打包，不运行推理验收；数值检查使用scripts/demo/validate_bumi_stage1_engine.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
