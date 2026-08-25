#!/usr/bin/env python3
"""构建并指纹化 BUMI qpos30/contact 音乐去噪 TensorRT 引擎。

脚本只接受固定 batch=1、120 帧、30D 运动、2D 接触和 EDGE35 音乐的 ONNX 图；构建前
检查五个输入和两个输出的名称/形状，FP16 模式仍强制输出 float32 以减小 DDIM 累积误差。
引擎放在由 ONNX/checkpoint SHA256、TensorRT/libnvinfer、精度和当前 GPU 共同决定的缓存
目录中，并写入 ``engine.json``。运行时会再次核验这些字段，不能把计划文件复制到不同
GPU 或与另一个 checkpoint 混用。已有完整缓存默认复用，只有显式 ``--overwrite`` 才重建。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.feature_codec import (  # noqa: E402
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.runtime.bumi_music_deploy import (  # noqa: E402
    BUMI_ENGINE_CONTRACT,
    BUMI_MOTION_DIM,
    BumiTensorRTStepRunner,
    bumi_engine_cache_key,
)
from gem.runtime.bumi_music_onnx import BUMI_ONNX_CONTRACT_VERSION  # noqa: E402
from gem.runtime.music_only_trt import (  # noqa: E402
    gpu_fingerprint,
    sha256_file,
    validate_tensorrt_installation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--optimization-level", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _network_contract(network: object, trt: object) -> None:
    expected = BumiTensorRTStepRunner.REQUIRED_INPUTS
    actual: dict[str, tuple[int, ...]] = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        actual[str(tensor.name)] = tuple(int(value) for value in tensor.shape)
    if actual != expected:
        raise RuntimeError(f"BUMI TensorRT input contract mismatch: {actual}")
    outputs = {
        str(network.get_output(index).name): network.get_output(index)
        for index in range(network.num_outputs)
    }
    expected_outputs = BumiTensorRTStepRunner.REQUIRED_OUTPUTS
    actual_outputs = {
        name: tuple(int(value) for value in tensor.shape) for name, tensor in outputs.items()
    }
    if actual_outputs != expected_outputs:
        raise RuntimeError(f"BUMI TensorRT output contract mismatch: {actual_outputs}")
    for output in outputs.values():
        output.dtype = trt.float32


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    onnx_path = args.onnx.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    onnx_metadata_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    if not onnx_metadata_path.is_file():
        raise FileNotFoundError(f"BUMI ONNX metadata is missing: {onnx_metadata_path}")
    onnx_metadata = json.loads(onnx_metadata_path.read_text(encoding="utf-8"))
    if onnx_metadata.get("contract_version") != BUMI_ONNX_CONTRACT_VERSION:
        raise ValueError("TensorRT build requires the current qpos30/contact BUMI ONNX contract")
    if onnx_metadata.get("representation_contract_version") != BUMI_REPRESENTATION_CONTRACT_VERSION:
        raise ValueError("TensorRT build requires the current BUMI qpos30 representation")
    if args.workspace_gib <= 0.0:
        raise ValueError("--workspace-gib must be > 0")
    if not 0 <= args.optimization_level <= 5:
        raise ValueError("--optimization-level must be in 0..5")
    if not torch.cuda.is_available():
        raise RuntimeError("BUMI TensorRT engine building requires CUDA")
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT Python binding is missing; install the version matching libnvinfer"
        ) from exc
    libnvinfer_version = validate_tensorrt_installation(trt)
    gpu = gpu_fingerprint(args.device)
    onnx_sha = sha256_file(onnx_path)
    checkpoint_sha = sha256_file(checkpoint)
    if (onnx_metadata.get("checkpoint") or {}).get("sha256") != checkpoint_sha:
        raise ValueError("ONNX metadata checkpoint SHA does not match --checkpoint")
    cache_key = bumi_engine_cache_key(
        onnx_sha256=onnx_sha,
        checkpoint_sha256=checkpoint_sha,
        tensorrt_version=trt.__version__,
        precision=args.precision,
        gpu=gpu,
    )
    output_dir = args.output_dir.expanduser().resolve() / cache_key
    engine_path = output_dir / "bumi_music_denoiser.engine"
    manifest_path = output_dir / "engine.json"
    if engine_path.is_file() and manifest_path.is_file() and not args.overwrite:
        print(engine_path)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("BUMI TensorRT ONNX parse failed:\n" + "\n".join(errors))
    _network_contract(network, trt)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
    config.builder_optimization_level = int(args.optimization_level)
    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("this GPU does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the BUMI serialized engine")
    temporary = engine_path.with_suffix(".engine.tmp")
    temporary.write_bytes(bytes(serialized))
    temporary.replace(engine_path)
    elapsed = time.perf_counter() - started
    manifest = {
        "contract_version": BUMI_ENGINE_CONTRACT,
        "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "cache_key": cache_key,
        "engine": engine_path.name,
        "engine_sha256": sha256_file(engine_path),
        "onnx": str(onnx_path),
        "onnx_sha256": onnx_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "tensorrt_version": trt.__version__,
        "libnvinfer_version": libnvinfer_version,
        "precision": args.precision,
        "workspace_gib": args.workspace_gib,
        "optimization_level": args.optimization_level,
        "gpu": gpu,
        "build_seconds": elapsed,
        "input_shape": [1, 120, BUMI_MOTION_DIM],
        "output_shapes": {
            name: list(shape) for name, shape in BumiTensorRTStepRunner.REQUIRED_OUTPUTS.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(engine_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
