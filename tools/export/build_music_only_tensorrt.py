#!/usr/bin/env python3
"""Build and fingerprint the fixed-shape music-only TensorRT engine."""

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

from gem.runtime.music_only_trt import (  # noqa: E402
    engine_cache_key,
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    onnx_path = args.onnx.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if args.workspace_gib <= 0:
        raise ValueError("--workspace-gib must be > 0")
    if not 0 <= args.optimization_level <= 5:
        raise ValueError("--optimization-level must be in 0..5")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT engine building requires CUDA")
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT Python bindings are missing. Install the version matching "
            "the deployment machine's libnvinfer before building the engine."
        ) from exc
    libnvinfer_version = validate_tensorrt_installation(trt)

    gpu = gpu_fingerprint(args.device)
    cache_key = engine_cache_key(
        onnx_sha256=sha256_file(onnx_path),
        checkpoint_sha256=sha256_file(checkpoint),
        tensorrt_version=trt.__version__,
        precision=args.precision,
        gpu=gpu,
    )
    output_dir = args.output_dir.expanduser().resolve() / cache_key
    engine_path = output_dir / "music_only_denoiser.engine"
    manifest_path = output_dir / "engine.json"
    if engine_path.exists() and manifest_path.exists() and not args.overwrite:
        print(engine_path)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3)
    )
    config.builder_optimization_level = int(args.optimization_level)
    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("this GPU does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    for output_index in range(network.num_outputs):
        network.get_output(output_index).dtype = trt.float32

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the serialized engine")
    temporary = engine_path.with_suffix(".engine.tmp")
    temporary.write_bytes(bytes(serialized))
    temporary.replace(engine_path)
    elapsed = time.perf_counter() - started
    manifest = {
        "contract_version": "gem_music_only_trt_engine_v1",
        "cache_key": cache_key,
        "engine": engine_path.name,
        "engine_sha256": sha256_file(engine_path),
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tensorrt_version": trt.__version__,
        "libnvinfer_version": libnvinfer_version,
        "precision": args.precision,
        "workspace_gib": args.workspace_gib,
        "optimization_level": args.optimization_level,
        "gpu": gpu,
        "build_seconds": elapsed,
        "input_shape": [1, 120, 151],
        "output_shape": [1, 120, 151],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(engine_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
