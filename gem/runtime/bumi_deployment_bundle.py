"""读取并完整核验无需训练 checkpoint 的 BUMI 音乐部署资产包。

部署清单只允许引用其所在目录内的相对路径，并对 ONNX、TensorRT engine、两份元数据、
统计、运动学和 GMT policy 逐个核验字节数及 SHA256。源 checkpoint 的 SHA256 是原仓库
导出时形成的来源记录，部署机器不重新读取 checkpoint；但它必须与 ONNX 元数据及
engine 元数据一致，且所有实际执行的模型文件仍须通过完整哈希校验。

本模块不导入训练模型、不下载文件、不连接 Redis，也不启动 CUDA 推理。实际图接口、
TensorRT 动态库和 GPU 的检查继续由对应运行器完成。原仓库旧 checkpoint 启动方式
通过 resolve_console_assets 保留，与清单模式互斥，防止两个模型版本的资产混合使用。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gem.runtime.bumi_music_contract import (
    BUMI_ONNX_CONTRACT_VERSION,
    BUMI_ONNX_INPUTS,
    BUMI_ONNX_OUTPUTS,
)

BUMI_DEPLOYMENT_CONTRACT = "genmo.bumi_music_deployment.v1"
ASSET_NAMES = (
    "onnx",
    "onnx_metadata",
    "engine",
    "engine_metadata",
    "kinematics",
    "stats",
    "gmt_policy",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


@dataclass(frozen=True)
class BumiDeploymentBundle:
    manifest_path: Path
    payload: dict[str, Any]
    paths: dict[str, Path]
    hashes: dict[str, str]
    source_checkpoint_sha256: str


def load_bumi_deployment_manifest(path: str | Path) -> BumiDeploymentBundle:
    """验证完整包及跨文件来源关系；失败时不创建任何执行器。"""
    manifest = Path(path).expanduser().resolve(strict=True)
    root = manifest.parent
    payload = _json(manifest)
    if payload.get("contract_version") != BUMI_DEPLOYMENT_CONTRACT:
        raise ValueError("unsupported BUMI deployment manifest contract")
    source_sha = _sha(payload.get("source_checkpoint_sha256"), "source_checkpoint_sha256")
    assets = payload.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(ASSET_NAMES):
        raise ValueError(f"deployment assets must contain exactly {ASSET_NAMES}")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in ASSET_NAMES:
        record = assets[name]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"invalid deployment asset: {name}")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"{name}: asset path must be relative and stay inside the bundle")
        actual = (root / relative).resolve(strict=True)
        if not actual.is_relative_to(root) or not actual.is_file():
            raise ValueError(f"{name}: asset escapes bundle or is not a regular file")
        expected_sha = _sha(record.get("sha256"), f"assets.{name}.sha256")
        size = record.get("size_bytes")
        if type(size) is not int or size < 1 or actual.stat().st_size != size:
            raise ValueError(f"{name}: size_bytes mismatch")
        actual_sha = file_sha256(actual)
        if actual_sha != expected_sha:
            raise ValueError(f"{name}: SHA256 mismatch")
        paths[name], hashes[name] = actual, actual_sha
    if len(set(paths.values())) != len(paths):
        raise ValueError("deployment assets must refer to distinct files")
    if paths["engine_metadata"] != paths["engine"].parent / "engine.json":
        raise ValueError("engine_metadata must be engine.json beside the engine")

    onnx = _json(paths["onnx_metadata"])
    engine = _json(paths["engine_metadata"])
    if onnx.get("contract_version") != BUMI_ONNX_CONTRACT_VERSION:
        raise ValueError("ONNX metadata has an incompatible contract")
    if (
        onnx.get("input_contract") != BUMI_ONNX_INPUTS
        or onnx.get("output_contract") != BUMI_ONNX_OUTPUTS
    ):
        raise ValueError("ONNX metadata shape contract must be fixed BUMI 120/30/35/2")
    if onnx.get("sequence_length") != 120 or onnx.get("fixed_batch") != 1:
        raise ValueError("ONNX metadata must describe batch=1 and 120 frames")
    if (onnx.get("checkpoint") or {}).get("sha256") != source_sha:
        raise ValueError("ONNX source checkpoint SHA256 mismatch")
    for name in ("kinematics", "stats"):
        if (onnx.get(name) or {}).get("sha256") != hashes[name]:
            raise ValueError(f"ONNX {name} SHA256 mismatch")
    # 本发布格式只接受自包含 ONNX。外部权重图需先在原仓库重新导出，不能漏拷权重。
    artifacts = onnx.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("deployment requires one self-contained ONNX artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict) or artifact.get("path") != paths["onnx"].name:
        raise ValueError("ONNX artifact filename mismatch")
    if (
        artifact.get("sha256") != hashes["onnx"]
        or artifact.get("size_bytes") != paths["onnx"].stat().st_size
    ):
        raise ValueError("ONNX artifact SHA256/size mismatch")
    if engine.get("checkpoint_sha256") != source_sha or engine.get("onnx_sha256") != hashes["onnx"]:
        raise ValueError("engine source checkpoint/ONNX SHA256 mismatch")
    if engine.get("engine_sha256") != hashes["engine"]:
        raise ValueError("engine artifact SHA256 mismatch")
    if (
        engine.get("input_shape") != BUMI_ONNX_INPUTS["noisy_motion"]
        or engine.get("output_shapes") != BUMI_ONNX_OUTPUTS
    ):
        raise ValueError("engine shape contract mismatch")
    # 导入运行时常量，不导入模型构造器；保持协议单一来源。
    from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION
    from gem.runtime.bumi_music_deploy import BUMI_ENGINE_CONTRACT

    if engine.get("contract_version") != BUMI_ENGINE_CONTRACT:
        raise ValueError("engine metadata contract mismatch")
    if any(
        value.get("representation_contract_version") != BUMI_REPRESENTATION_CONTRACT_VERSION
        for value in (onnx, engine)
    ):
        raise ValueError("deployment motion representation mismatch")
    stats = _json(paths["stats"])
    if stats.get("kinematics_sha256") != hashes["kinematics"]:
        raise ValueError("stats/kinematics identity mismatch")
    return BumiDeploymentBundle(manifest, payload, paths, hashes, source_sha)


def resolve_console_assets(args: Any) -> tuple[Any, BumiDeploymentBundle | None]:
    """原地补齐清单路径；旧 CLI 缺参时尽早给出明确错误，不改变其资产校验。"""
    manifest = getattr(args, "deployment_manifest", None)
    overrides = ("checkpoint", "onnx", "onnx_metadata", "engine", "kinematics", "stats")
    if manifest is not None:
        conflicting = [
            f"--{name.replace('_', '-')}"
            for name in overrides
            if getattr(args, name, None) is not None
        ]
        if conflicting:
            raise ValueError(
                "--deployment-manifest cannot be combined with " + ", ".join(conflicting)
            )
        bundle = load_bumi_deployment_manifest(manifest)
        for name in overrides:
            if name != "checkpoint":
                setattr(args, name, bundle.paths[name])
        args.checkpoint = None
        return args, bundle
    missing = [
        f"--{name}"
        for name in ("checkpoint", "onnx", "kinematics", "stats")
        if getattr(args, name, None) is None
    ]
    if getattr(args, "backend", "tensorrt") == "tensorrt" and getattr(args, "engine", None) is None:
        missing.append("--engine")
    if missing:
        raise ValueError(
            "legacy startup requires " + ", ".join(missing) + "; or use --deployment-manifest"
        )
    return args, None
