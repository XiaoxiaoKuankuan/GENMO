"""无 checkpoint 的 BUMI 发布清单与旧启动接口回归测试。

测试在 pytest 的独立临时目录中构造极小二进制资产，不启动 CUDA、Redis、GMT 或网络。
覆盖目录搬迁、文件篡改、跨资产身份错配、路径越界/软链接、元数据契约及新旧 CLI 互斥。
这些测试只证明清单和参数边界；真实图推理、采样和 C++ 通信由单独验收执行。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION
from gem.runtime.bumi_deployment_bundle import (
    BUMI_DEPLOYMENT_CONTRACT,
    BUMI_DEPLOYMENT_LEGACY_CONTRACT,
    file_sha256,
    load_bumi_deployment_manifest,
    resolve_console_assets,
)
from gem.runtime.bumi_music_contract import (
    BUMI_ONNX_CONTRACT_VERSION,
    BUMI_ONNX_INPUTS,
    BUMI_ONNX_OUTPUTS,
)
from gem.runtime.bumi_music_deploy import BUMI_ENGINE_CONTRACT
from scripts.demo.demo_music_bumi_console import build_parser


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _update_record(manifest: Path, name: str, update: dict | None = None) -> None:
    payload = json.loads(manifest.read_text())
    path = manifest.parent / payload["assets"][name]["path"]
    if update is not None:
        data = json.loads(path.read_text())
        data.update(update)
        _write(path, data)
    payload["assets"][name].update(size_bytes=path.stat().st_size, sha256=file_sha256(path))
    _write(manifest, payload)


@pytest.fixture
def deployment_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    root.mkdir()
    paths = {
        name: root / filename
        for name, filename in {
            "onnx": "model.onnx",
            "onnx_metadata": "model.onnx.json",
            "engine": "engine/model.engine",
            "engine_metadata": "engine/engine.json",
            "kinematics": "kinematics.json",
            "stats": "stats.json",
        }.items()
    }
    for name in ("onnx", "engine"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes(("test " + name).encode())
    _write(paths["kinematics"], {"test": "kinematics"})
    _write(paths["stats"], {"kinematics_sha256": file_sha256(paths["kinematics"])})
    source_sha = "a" * 64
    _write(
        paths["onnx_metadata"],
        {
            "contract_version": BUMI_ONNX_CONTRACT_VERSION,
            "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
            "checkpoint": {"sha256": source_sha, "path": "/unavailable/training/s350000.ckpt"},
            "input_contract": BUMI_ONNX_INPUTS,
            "output_contract": BUMI_ONNX_OUTPUTS,
            "sequence_length": 120,
            "fixed_batch": 1,
            "kinematics": {"sha256": file_sha256(paths["kinematics"])},
            "stats": {"sha256": file_sha256(paths["stats"])},
            "artifacts": [
                {
                    "path": paths["onnx"].name,
                    "size_bytes": paths["onnx"].stat().st_size,
                    "sha256": file_sha256(paths["onnx"]),
                }
            ],
        },
    )
    _write(
        paths["engine_metadata"],
        {
            "contract_version": BUMI_ENGINE_CONTRACT,
            "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
            "checkpoint_sha256": source_sha,
            "onnx_sha256": file_sha256(paths["onnx"]),
            "engine_sha256": file_sha256(paths["engine"]),
            "input_shape": [1, 120, 30],
            "output_shapes": BUMI_ONNX_OUTPUTS,
        },
    )
    manifest = root / "deployment.json"
    _write(
        manifest,
        {
            "contract_version": BUMI_DEPLOYMENT_CONTRACT,
            "source_checkpoint_sha256": source_sha,
            "assets": {
                name: {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for name, path in paths.items()
            },
        },
    )
    return manifest


def test_relocated_bundle_needs_no_checkpoint(
    deployment_manifest: Path, tmp_path: Path, monkeypatch
) -> None:
    moved = tmp_path / "中文 空格" / "bundle"
    shutil.copytree(deployment_manifest.parent, moved)
    shutil.rmtree(deployment_manifest.parent)
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--deployment-manifest", str(moved / "deployment.json")])
    args, bundle = resolve_console_assets(args)
    assert args.checkpoint is None
    assert bundle.source_checkpoint_sha256 == "a" * 64
    assert args.onnx.is_relative_to(moved)
    assert not list(moved.rglob("*.ckpt"))


def test_v2_bundle_is_controller_independent(deployment_manifest):
    bundle = load_bumi_deployment_manifest(deployment_manifest)
    assert "gmt_policy" not in bundle.paths
    assert len(bundle.paths) == 6


def test_v1_seven_asset_bundle_still_loads(deployment_manifest):
    policy = deployment_manifest.parent / "legacy_gmt.onnx"
    policy.write_bytes(b"legacy policy")
    data = json.loads(deployment_manifest.read_text())
    data["contract_version"] = BUMI_DEPLOYMENT_LEGACY_CONTRACT
    data["assets"]["gmt_policy"] = {
        "path": policy.name,
        "size_bytes": policy.stat().st_size,
        "sha256": file_sha256(policy),
    }
    _write(deployment_manifest, data)
    assert load_bumi_deployment_manifest(deployment_manifest).paths["gmt_policy"] == policy
    policy.write_bytes(b"legacy tamper")
    with pytest.raises(ValueError):
        load_bumi_deployment_manifest(deployment_manifest)


@pytest.mark.parametrize(
    "name",
    ["onnx", "engine", "onnx_metadata", "engine_metadata", "stats", "kinematics"],
)
def test_tampered_asset_rejected(deployment_manifest: Path, name: str) -> None:
    bundle = load_bumi_deployment_manifest(deployment_manifest)
    path = bundle.paths[name]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_bumi_deployment_manifest(deployment_manifest)


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("engine_metadata", {"checkpoint_sha256": "b" * 64}),
        ("engine_metadata", {"onnx_sha256": "b" * 64}),
        ("engine_metadata", {"input_shape": [1, 120, 151]}),
        ("engine_metadata", {"contract_version": "smpl"}),
        ("onnx_metadata", {"checkpoint": {"sha256": "b" * 64}}),
        ("onnx_metadata", {"kinematics": {"sha256": "b" * 64}}),
        ("onnx_metadata", {"sequence_length": 60}),
        ("onnx_metadata", {"input_contract": {"wrong": [1]}}),
        ("onnx_metadata", {"artifacts": []}),
    ],
)
def test_cross_asset_mismatch_rejected_even_with_updated_file_hash(
    deployment_manifest: Path, name: str, change: dict
) -> None:
    _update_record(deployment_manifest, name, change)
    with pytest.raises(ValueError):
        load_bumi_deployment_manifest(deployment_manifest)


@pytest.mark.parametrize("value", ["../outside.onnx", "/tmp/outside.onnx"])
def test_asset_path_escape_rejected(deployment_manifest: Path, value: str) -> None:
    payload = json.loads(deployment_manifest.read_text())
    payload["assets"]["onnx"]["path"] = value
    _write(deployment_manifest, payload)
    with pytest.raises(ValueError, match="relative"):
        load_bumi_deployment_manifest(deployment_manifest)


def test_symlink_escape_rejected(deployment_manifest: Path, tmp_path: Path) -> None:
    path = deployment_manifest.parent / "model.onnx"
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_bumi_deployment_manifest(deployment_manifest)


@pytest.mark.parametrize(
    "option", ["checkpoint", "onnx", "onnx-metadata", "engine", "kinematics", "stats"]
)
def test_manifest_rejects_legacy_path_overrides(deployment_manifest: Path, option: str) -> None:
    args = build_parser().parse_args(
        ["--deployment-manifest", str(deployment_manifest), f"--{option}", "wrong"]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_console_assets(args)


def test_legacy_cli_preserves_asset_paths() -> None:
    argv = ["--backend", "onnx"]
    for name in ("checkpoint", "onnx", "kinematics", "stats"):
        argv.extend([f"--{name}", f"old/{name}"])
    args, bundle = resolve_console_assets(build_parser().parse_args(argv))
    assert bundle is None
    assert args.checkpoint == Path("old/checkpoint")


def test_legacy_missing_paths_fail_before_model_loading() -> None:
    with pytest.raises(ValueError, match="legacy startup requires"):
        resolve_console_assets(build_parser().parse_args([]))


@pytest.mark.parametrize("policy", ["legacy", "attention_norm_heads_fp32_v1"])
def test_engine_precision_policy_is_part_of_validated_identity(
    deployment_manifest: Path, monkeypatch, policy: str
) -> None:
    from types import SimpleNamespace

    from gem.runtime import bumi_music_deploy as runtime

    bundle = load_bumi_deployment_manifest(deployment_manifest)
    gpu = {
        "name": "test GPU",
        "compute_capability": [8, 9],
        "total_memory": 100,
        "torch_cuda": "12.4",
    }
    args = dict(
        onnx_sha256=bundle.hashes["onnx"],
        checkpoint_sha256=bundle.source_checkpoint_sha256,
        tensorrt_version="10.13.3.9",
        precision="fp16",
        gpu=gpu,
    )
    legacy = runtime.bumi_engine_cache_key(**args)
    assert legacy == runtime.bumi_engine_cache_key(**args, precision_policy="legacy")
    key = runtime.bumi_engine_cache_key(**args, precision_policy=policy)
    if policy != "legacy":
        assert key != legacy
    metadata = json.loads(bundle.paths["engine_metadata"].read_text())
    metadata.update(
        precision_policy=policy,
        precision="fp16",
        gpu=gpu,
        cache_key=key,
        tensorrt_version="10.13.3.9",
        libnvinfer_version="10.13.3",
    )
    _write(bundle.paths["engine_metadata"], metadata)
    monkeypatch.setattr(runtime, "gpu_fingerprint", lambda _: gpu)
    runner = runtime.BumiTensorRTStepRunner.__new__(runtime.BumiTensorRTStepRunner)
    runner.engine_path = bundle.paths["engine"]
    runner.device = "cuda:0"
    runner.trt = SimpleNamespace(__version__="10.13.3.9")
    runner.linked_tensorrt_version = "10.13.3"
    assert runner._validate_manifest(True)["cache_key"] == key
    metadata["precision_policy"] = "other_policy"
    _write(bundle.paths["engine_metadata"], metadata)
    with pytest.raises(RuntimeError, match="cache fingerprint"):
        runner._validate_manifest(True)
