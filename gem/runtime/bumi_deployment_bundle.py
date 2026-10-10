"""当前Stage1部署包的完整资源清单与来源绑定。

清单v4只接受当前Stage1十一输入ONNX、FP32 TensorRT engine、对应两份元数据、
qpos30统计、BUMI运动学和CUDA掩码插件，共七个资产。每个文件核验路径、大小和SHA256。
训练checkpoint不随包部署，只保留来源哈希；拒绝旧清单与checkpoint回退。
ONNX元数据使用当前Stage1的asset_identity/interface结构，不再读取旧纯音乐字段。
运行时仍检查engine的ABI与GPU；本模块不执行推理、不连接Redis/ROS、不构建引擎。
构建工具在完成engine后原子发布deployment.json，避免启动脚本加载未完成资产。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gem.runtime.bumi_music_contract import (
    BUMI_ENGINE_CONTRACT, BUMI_ONNX_INPUTS, BUMI_ONNX_INPUT_DTYPES,
    BUMI_ONNX_OUTPUTS, BUMI_ONNX_OUTPUT_DTYPES, BUMI_TRT_INPUT_DTYPES, BUMI_STAGE1_PRECISION_POLICY,
    BUMI_STAGE1_SAMPLING_CONTRACT_VERSION,
    validate_stage1_engine_build_options,
)

BUMI_DEPLOYMENT_CONTRACT = "genmo.bumi_stage1_deployment.v4"
ASSET_NAMES = ("onnx", "onnx_metadata", "engine", "engine_metadata", "stats", "kinematics", "engine_plugin")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label}必须是小写SHA256")
    return value


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"元数据必须是JSON对象: {path}")
    return value


@dataclass(frozen=True)
class BumiDeploymentBundle:
    manifest_path: Path
    payload: dict[str, Any]
    paths: dict[str, Path]
    hashes: dict[str, str]
    source_checkpoint_sha256: str


def load_bumi_deployment_manifest(path) -> BumiDeploymentBundle:
    """加载唯一Stage1清单，严格拒绝旧模型或配套资产混用。"""
    requested = Path(path).expanduser()
    if not requested.is_file():
        raise FileNotFoundError(
            f"Stage1部署清单不存在: {requested}；先运行bash scripts/export/build_bumi_stage1_engine.sh")
    manifest = requested.resolve(strict=True)
    root = manifest.parent
    payload = _json(manifest)
    if payload.get("contract_version") != BUMI_DEPLOYMENT_CONTRACT:
        raise ValueError("只支持Stage1 v4插件部署清单；请构建当前Stage1 engine")
    if payload.get("sampling_contract_version") != BUMI_STAGE1_SAMPLING_CONTRACT_VERSION:
        raise ValueError("Stage1部署采样身份不匹配")
    source_sha = _sha(payload.get("source_checkpoint_sha256"), "source_checkpoint_sha256")
    assets = payload.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(ASSET_NAMES):
        raise ValueError(f"Stage1清单必须包含{ASSET_NAMES}")
    paths, hashes = {}, {}
    for name, record in assets.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"资产记录无效: {name}")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"资产必须是包内相对路径: {name}")
        actual = (root / relative).resolve(strict=True)
        if not actual.is_relative_to(root) or not actual.is_file():
            raise ValueError(f"资产路径越界: {name}")
        size = record.get("size_bytes")
        if type(size) is not int or size <= 0 or actual.stat().st_size != size:
            raise ValueError(f"资产大小不匹配: {name}")
        expected = _sha(record.get("sha256"), f"assets.{name}.sha256")
        if file_sha256(actual) != expected:
            raise ValueError(f"资产SHA256不匹配: {name}")
        paths[name], hashes[name] = actual, expected
    if len(set(paths.values())) != len(paths):
        raise ValueError("不同资产不能引用同一文件")
    if paths["engine_metadata"] != paths["engine"].parent / "engine.json":
        raise ValueError("engine.json必须位于engine旁")
    onnx = _json(paths["onnx_metadata"])
    engine = _json(paths["engine_metadata"])
    from gem.runtime.bumi_music_deploy import validate_stage1_metadata

    validate_stage1_metadata(onnx)
    if onnx.get("checkpoint_sha256") != source_sha:
        raise ValueError("Stage1 ONNX来源checkpoint哈希不匹配")
    if onnx.get("onnx_sha256") != hashes["onnx"]:
        raise ValueError("Stage1 ONNX元数据哈希不匹配")
    if onnx.get("onnx_size_bytes") != paths["onnx"].stat().st_size:
        raise ValueError("Stage1 ONNX元数据文件大小不匹配")
    identity = onnx["asset_identity"]
    for name in ("stats", "kinematics"):
        if identity.get(name + "_sha256") != hashes[name]:
            raise ValueError(f"Stage1 ONNX配套{name}哈希不匹配")
        if engine.get(name + "_sha256") != hashes[name]:
            raise ValueError(f"Stage1 engine配套{name}哈希不匹配")
    from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION

    expected_engine = {
        "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "contract_version": BUMI_ENGINE_CONTRACT, "checkpoint_sha256": source_sha,
        "onnx_sha256": hashes["onnx"], "onnx_metadata_sha256": hashes["onnx_metadata"],
        "engine_sha256": hashes["engine"], "input_shapes": BUMI_ONNX_INPUTS,
        "input_dtypes": BUMI_TRT_INPUT_DTYPES, "source_input_dtypes": BUMI_ONNX_INPUT_DTYPES,
        "output_shapes": BUMI_ONNX_OUTPUTS,
        "output_dtypes": BUMI_ONNX_OUTPUT_DTYPES, "precision": "fp32",
        "precision_policy": BUMI_STAGE1_PRECISION_POLICY,
    }
    for key, expected in expected_engine.items():
        if engine.get(key) != expected:
            raise ValueError(f"Stage1 engine元数据不匹配: {key}")
    validate_stage1_engine_build_options(engine.get("build_options"))
    from gem.runtime.bumi_stage1_plugin import engine_plugin_path

    plugin_path, plugin_sha = engine_plugin_path(paths["engine"], engine)
    if plugin_path != paths["engine_plugin"] or plugin_sha != hashes["engine_plugin"]:
        raise ValueError("Stage1 engine插件与部署清单不一致")
    stats = _json(paths["stats"])
    if stats.get("kinematics_sha256") != hashes["kinematics"]:
        raise ValueError("Stage1 stats/kinematics身份不匹配")
    return BumiDeploymentBundle(manifest, payload, paths, hashes, source_sha)


def resolve_console_assets(args):
    """仅从Stage1部署清单解析资源，不接受旧checkpoint或混合资产路径。"""
    manifest = getattr(args, "deployment_manifest", None)
    if manifest is None:
        raise ValueError("必须提供--deployment-manifest；不支持旧checkpoint启动方式")
    bundle = load_bumi_deployment_manifest(manifest)
    for name, path in bundle.paths.items():
        setattr(args, name, path)
    return args, bundle
