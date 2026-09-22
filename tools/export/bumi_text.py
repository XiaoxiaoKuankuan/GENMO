#!/usr/bin/env python3
"""BUMI 文本模型的导出、TensorRT构建、数值验证及部署资产打包入口。

按checkpoint契约导出120或300帧单步去噪图并记录真实length输入；外部权重文件逐一指纹绑定。
构建沿用音乐部署的TensorRT环境检查和敏感层FP32策略，GPU工作只在显式build执行。
validate固定有效噪声比较单步和完整DDIM；package不携带训练checkpoint或数据集。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gem.runtime.bumi_text_contract import resolve_assets, sha256_file
from gem.runtime.bumi_text_runtime import (
    BUNDLE_SCHEMA,
    EXPORT_SCHEMA,
    INPUTS,
    OUTPUTS,
    BumiTextSampler,
    OnnxTextStep,
    io_shapes,
    load_checkpoint_step,
    read_export_metadata,
)


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖既有记录: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def sample_inputs(device="cpu", frames=97, tensor_frames=300):
    generator = torch.Generator(device=device).manual_seed(42)
    return (
        torch.randn(1, tensor_frames, 30, generator=generator, device=device),
        torch.tensor([500], device=device),
        torch.randn(1, 150, 1024, generator=generator, device=device),
        (torch.arange(150, device=device)[None] < 17),
        torch.tensor([frames], device=device),
        torch.tensor([2.5], device=device),
    )


def export(checkpoint, output, device="cpu"):
    import onnx

    checkpoint, output = Path(checkpoint).resolve(strict=True), Path(output).resolve()
    if output.exists() or (output.parent.exists() and any(output.parent.iterdir())):
        raise FileExistsError("ONNX要求新的空输出目录，避免覆盖其他外部权重")
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper, contract, diffusion = load_checkpoint_step(checkpoint, device)
    assets = resolve_assets(contract, checkpoint=checkpoint)
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(
            wrapper,
            sample_inputs(device, tensor_frames=contract["sequence"]["pad_to_frames"]),
            str(output),
            opset_version=17,
            dynamo=False,
            input_names=list(INPUTS),
            output_names=list(OUTPUTS),
            external_data=True,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    onnx.checker.check_model(str(output))
    graph = onnx.load(str(output), load_external_data=False)
    names = set()
    for tensor in graph.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                names.add(item.value)
    external = {}
    for name in sorted(names):
        path = (output.parent / name).resolve(strict=True)
        if not path.is_relative_to(output.parent):
            raise ValueError("ONNX外部权重路径越界")
        external[name] = {"sha256": sha256_file(path)}
    meta = dict(
        schema=EXPORT_SCHEMA,
        onnx_sha256=sha256_file(output),
        external_data=external,
        source_checkpoint_sha256=sha256_file(checkpoint),
        model_contract=contract,
        diffusion_config=diffusion,
        inputs=io_shapes(contract)[0],
        outputs=io_shapes(contract)[1],
    )
    meta["asset_locations"] = {name: str(path) for name, path in assets.items()}
    write_json(output.with_suffix(output.suffix + ".json"), meta)
    # 验证真实图没有丢失length/text mask等输入。
    OnnxTextStep(output)
    return meta


def constrain_sensitive_layers(network, trt):
    """沿用v5注意力/归一化/头部FP32方案；MLP允许FP16。"""
    arithmetic = {
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.UNARY,
        trt.LayerType.ACTIVATION,
        trt.LayerType.SOFTMAX,
        trt.LayerType.NORMALIZATION,
        trt.LayerType.REDUCE,
        trt.LayerType.SCALE,
    }
    names = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type not in arithmetic or "/mlp/" in layer.name:
            continue
        outputs = [
            j
            for j in range(layer.num_outputs)
            if layer.get_output(j).dtype in (trt.float32, trt.float16)
        ]
        if outputs:
            layer.precision = trt.float32
            for j in outputs:
                layer.set_output_type(j, trt.float32)
            names.append(layer.name)
    return names


def build_engine(onnx_path, output, device="cuda:0", precision="fp16"):
    from gem.runtime.bumi_text_tensorrt import ENGINE_SCHEMA, TextTensorRTStep
    from gem.runtime.tensorrt_core import gpu_fingerprint, validate_tensorrt_installation
    from gem.runtime.tensorrt_environment import prepare_tensorrt_libraries

    meta = read_export_metadata(onnx_path)
    output = Path(output).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("拒绝覆盖已有engine")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT构建需要显式授权的CUDA环境")
    prepare_tensorrt_libraries()
    import tensorrt as trt

    library = validate_tensorrt_installation(trt)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(Path(onnx_path).resolve())):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    for count, getter, expected in [
        (network.num_inputs, network.get_input, io_shapes(meta["model_contract"])[0]),
        (network.num_outputs, network.get_output, io_shapes(meta["model_contract"])[1]),
    ]:
        if {getter(i).name: tuple(getter(i).shape) for i in range(count)} != expected:
            raise ValueError("TensorRT图接口不符")
    for i in range(network.num_outputs):
        network.get_output(i).dtype = trt.float32
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024**3)
    cfg.builder_optimization_level = 5
    constrained = []
    cfg.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        constrained = constrain_sensitive_layers(network, trt)
    with torch.cuda.device(device):
        serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError("TensorRT构建失败")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pending")
    temporary.write_bytes(bytes(serialized))
    temporary.replace(output)
    record = dict(
        schema=ENGINE_SCHEMA,
        engine_sha256=sha256_file(output),
        onnx_sha256=meta["onnx_sha256"],
        source_checkpoint_sha256=meta["source_checkpoint_sha256"],
        gpu=gpu_fingerprint(device),
        tensorrt_version=trt.__version__,
        libnvinfer_version=library,
        precision=precision,
        precision_policy="attention_norm_heads_fp32_v1",
        constrained_layers=constrained,
    )
    write_json(output.with_suffix(".json"), record)
    TextTensorRTStep(output, onnx_metadata=meta, device=device)
    return record


def validate(checkpoint, onnx_path, output, *, engine=None, device="cpu", ddim_steps=50):
    wrapper, contract, diffusion = load_checkpoint_step(checkpoint, device)
    meta = read_export_metadata(onnx_path)
    if (
        meta["source_checkpoint_sha256"] != sha256_file(checkpoint)
        or meta["model_contract"] != contract
    ):
        raise ValueError("验证必须使用同一checkpoint和资产")
    candidate = OnnxTextStep(onnx_path)
    if engine is not None:
        from gem.runtime.bumi_text_tensorrt import TextTensorRTStep

        candidate = TextTensorRTStep(engine, onnx_metadata=meta, device=device)
    limits = dict(
        step_motion=0.005,
        step_contact=0.01,
        ddim_motion=0.02,
        ddim_contact=0.02,
        root_position_m=0.05,
        joint_angle_rad=0.02,
        root_rotation_rad=0.02,
    )
    report = dict(
        schema="genmo.bumi_text_parity.v1",
        checkpoint_sha256=sha256_file(checkpoint),
        onnx_sha256=meta["onnx_sha256"],
        engine_sha256=None if engine is None else sha256_file(engine),
        ddim_steps=ddim_steps,
        seed=42,
        frames=[],
        passed=True,
        tensorrt_numerical_limits=limits,
        limits_are_quality_guarantees=False,
    )
    left, right = (
        BumiTextSampler(wrapper, diffusion, ddim_steps),
        BumiTextSampler(candidate, diffusion, ddim_steps),
    )
    from gem.robots.bumi.endecoder import BumiEndecoder

    assets = resolve_assets(contract, checkpoint=checkpoint)
    decoder = BumiEndecoder(assets["kinematics"], assets["stats"], sequence_mode="full").to(device)
    sequence = contract["sequence"]
    for frames in sorted({sequence["min_frames"], 60, 97, 120, 183, 240, 299, 300}):
        if not sequence["min_frames"] <= frames <= sequence["max_frames"]:
            continue
        inputs = sample_inputs(device, frames, sequence["pad_to_frames"])
        with torch.no_grad():
            a, b = wrapper(*inputs), candidate(*inputs)
            noise = torch.randn(
                1,
                frames,
                30,
                generator=torch.Generator(device=device).manual_seed(42),
                device=device,
            )
            aa = left.generate(
                inputs[2], inputs[3], frames, noise=noise, tensor_frames=sequence["pad_to_frames"]
            )
            bb = right.generate(
                inputs[2], inputs[3], frames, noise=noise, tensor_frames=sequence["pad_to_frames"]
            )
            qa = decoder.compose_qpos(decoder.decode(aa[0]))
            qb = decoder.compose_qpos(decoder.decode(bb[0]))
        errors = {
            name: float((x - y).abs().max())
            for name, x, y in [
                ("step_motion", a[0], b[0]),
                ("step_contact", a[1], b[1]),
                ("ddim_motion", aa[0], bb[0]),
                ("ddim_contact", aa[1], bb[1]),
                ("root_position_m", qa[..., :3], qb[..., :3]),
                ("joint_angle_rad", qa[..., 7:], qb[..., 7:]),
            ]
        }
        errors["root_rotation_rad"] = float(
            2 * torch.acos((qa[..., 3:7] * qb[..., 3:7]).sum(-1).abs().clamp(0, 1)).max()
        )
        if engine is None:
            passed = all(torch.allclose(x, y, rtol=1e-3, atol=1e-4) for x, y in zip(a, b)) and all(
                torch.allclose(x, y, rtol=2e-3, atol=2e-3) for x, y in zip(aa, bb)
            )
        else:
            passed = all(0 <= errors[key] <= limit for key, limit in limits.items())
        report["frames"].append(dict(frames=frames, max_abs_errors=errors, passed=bool(passed)))
        report["passed"] &= bool(passed)
    write_json(output, report)
    if not report["passed"]:
        raise RuntimeError("数值对照未通过，请检查报告；不可标记部署验收成功")
    return report


def package(
    onnx_path,
    output,
    *,
    engine=None,
    validation_report=None,
    robot_manifest=ROOT / "assets/bumi_viewer/manifest.json",
):
    meta = read_export_metadata(onnx_path)
    assets = resolve_assets(meta["model_contract"], **meta.get("asset_locations", {}))
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("部署包必须使用新目录")
    output.mkdir(parents=True)
    files = {}
    software = {}

    def add(key, source, relative, *, software_only=False):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        (software if software_only else files)[key] = dict(
            path=relative, sha256=sha256_file(target)
        )

    onnx_path = Path(onnx_path).resolve()
    add("onnx", onnx_path, "models/model.onnx")
    add(
        "onnx_metadata", onnx_path.with_suffix(onnx_path.suffix + ".json"), "models/model.onnx.json"
    )
    for name in meta["external_data"]:
        add("external:" + name, onnx_path.parent / name, "models/" + name)
    for name, path in assets.items():
        add(name, path, "models/" + name + ".json")
    if engine is not None:
        engine = Path(engine).resolve()
        info = json.loads(engine.with_suffix(".json").read_text())
        if info.get("onnx_sha256") != meta["onnx_sha256"] or info.get(
            "engine_sha256"
        ) != sha256_file(engine):
            raise ValueError("engine与ONNX不匹配")
        add("engine", engine, "models/model.engine")
        add("engine_metadata", engine.with_suffix(".json"), "models/model.json")
    from gem.runtime.bumi_preview import validate_robot_assets

    validate_robot_assets(robot_manifest, assets["kinematics"])
    robot_manifest = Path(robot_manifest)
    robot = json.loads(robot_manifest.read_text())
    add("robot_manifest", robot_manifest, "assets/bumi_viewer/manifest.json")
    for name in robot["files"]:
        add("robot:" + name, robot_manifest.parent / name, "assets/bumi_viewer/" + name)
    if validation_report is not None:
        report = json.loads(Path(validation_report).read_text())
        if (
            not report.get("passed")
            or report.get("onnx_sha256") != meta["onnx_sha256"]
            or report.get("engine_sha256") != (None if engine is None else sha256_file(engine))
        ):
            raise ValueError("部署验证报告不匹配")
        add("validation", validation_report, "validation.json")
    # 维护显式运行依赖闭包，不复制训练包、checkpoint、dataset或导出工具。
    paths = [
        "LICENSE",
        "gem/__init__.py",
        "gem/runtime/__init__.py",
        "gem/runtime/bumi_text_contract.py",
        "gem/runtime/bumi_text_runtime.py",
        "gem/runtime/bumi_text_tensorrt.py",
        "gem/runtime/tensorrt_core.py",
        "gem/runtime/tensorrt_environment.py",
        "gem/runtime/text_encoding.py",
        "gem/runtime/bumi_preview.py",
        "gem/runtime/bumi_text_viewer.py",
        "gem/runtime/bumi_text_launcher.py",
        "gem/runtime/text_motion_web/__init__.py",
        "gem/runtime/text_motion_web/worker.py",
        "gem/runtime/text_motion_web/storage.py",
        "gem/utils/sequence_contract.py",
        "gem/utils/rotation_conversions.py",
        "scripts/demo/demo_bumi_text.py",
        "scripts/demo/bumi_mujoco_viewer.py",
    ]
    paths += [
        "gem/robots/bumi/" + name + ".py"
        for name in [
            "__init__",
            "endecoder",
            "kinematics",
            "feature_codec",
            "contacts",
            "postprocess",
        ]
    ]
    paths += [
        "gem/diffusion_utils/" + name + ".py"
        for name in ["gaussian_diffusion", "nn", "losses", "respace", "model_util"]
    ]
    # 导出后端不构造Transformer，因此无需timm/训练网络源码。
    for name in paths:
        add("code:" + name, ROOT / name, name, software_only=True)
    for source, name in [
        ("requirements/bumi_text_runtime.lock", "requirements.lock"),
        ("requirements/bumi_text_tensorrt.lock", "tensorrt.lock"),
        ("scripts/deployment/install_bumi_text.sh", "install.sh"),
        ("scripts/deployment/run_bumi_text.sh", "run.sh"),
        ("configs/deployment/bumi_text.ini", "deployment.ini"),
        ("docs/BUMI_TEXT_DEPLOY.md", "README.md"),
    ]:
        add("runtime:" + name, ROOT / source, name, software_only=True)
    if engine is None:
        config = output / "deployment.ini"
        config.write_text(config.read_text().replace("backend = tensorrt", "backend = onnx"))
        software["runtime:deployment.ini"]["sha256"] = sha256_file(config)
    value = dict(
        schema=BUNDLE_SCHEMA,
        source_checkpoint_sha256=meta["source_checkpoint_sha256"],
        model_contract=meta["model_contract"],
        files=files,
        software_inventory=software,
        validation_status="passed" if validation_report else "not_validated",
    )
    write_json(output / "deployment.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["export", "build", "validate", "package"]:
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        if command in ["export", "validate"]:
            p.add_argument("--checkpoint", type=Path, required=True)
        if command != "export":
            p.add_argument("--onnx", type=Path, required=True)
        if command in ["validate", "package"]:
            p.add_argument("--engine", type=Path)
        if command in ["export", "build", "validate"]:
            p.add_argument("--device", default="cuda:0" if command == "build" else "cpu")
        if command == "build":
            p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
        if command == "validate":
            p.add_argument("--ddim-steps", type=int, default=50)
        if command == "package":
            p.add_argument("--validation-report", type=Path)
    a = parser.parse_args()
    if a.command == "export":
        result = export(a.checkpoint, a.output, a.device)
    elif a.command == "build":
        result = build_engine(a.onnx, a.output, a.device, a.precision)
    elif a.command == "validate":
        result = validate(
            a.checkpoint,
            a.onnx,
            a.output,
            engine=a.engine,
            device=a.device,
            ddim_steps=a.ddim_steps,
        )
    else:
        result = package(a.onnx, a.output, engine=a.engine, validation_report=a.validation_report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
