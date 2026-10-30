#!/usr/bin/env python3
"""导出 GENMO Stage2 十帧参考 Adapter，并在 CPU 核对 ONNX 与 PyTorch。

读取指定 checkpoint 的 Adapter 权重及根通道开关，从配套 SuperMimic RobotSpec
取得21关节顺序。输入必须已按 InputSpec 拼接、归一化为[B,2654]，输出为有界的
[B,10,27]名义参考相对残差；图包含三隐藏层MLP、Tanh、旋转向量范数限幅和根通道
开关。动态batch不能通过训练入口中的len(x)追踪，故直接复用net与decode。

此文件不训练、不调用WM/Critic、不占用CUDA。输入特征处理、历史/音乐状态、SO(3)
姿态组合、导数halo、Goal570及冻结Actor仍在控制程序中。旧参考投影也不包含在图内；
元数据保留checkpoint的执行配置，导出网络不代表替换完整运行时。输出只能写入新目录，
保存模型、具名输入布局和多batch数值核验报告，不覆盖已有模型或训练记录。
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gem.closedloop.reference_adapter.policy import ReferenceAdapter


class ExportAdapter(nn.Module):
    """复用训练网络和限幅，仅补入VectorControl在网络外执行的根姿态开关。"""

    def __init__(self, adapter, root_orientation=True):
        super().__init__()
        self.adapter = adapter
        self.root_orientation = root_orientation

    def forward(self, adapter_input):
        residual = self.adapter.decode(self.adapter.net(adapter_input))
        if not self.root_orientation:
            residual = torch.cat((residual[..., :3], residual[..., 3:6]*0,
                                  residual[..., 6:]), dim=-1)
        return residual


def export_and_check(model, destination):
    """固定opset17、动态batch导出；核对零/随机/大幅输入及真实输出限幅。"""
    model = model.cpu().eval().requires_grad_(False)
    dimension = model.adapter.input_spec.dimension
    torch.onnx.export(
        model, torch.zeros(1, dimension), str(destination),
        input_names=["adapter_input"], output_names=["residual_plan"],
        dynamic_axes={"adapter_input": {0: "batch"}, "residual_plan": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    results = []
    bounds = model.adapter.bounds.detach().numpy()
    for batch in (1, 16):
        for case, scale in (("zero", 0.), ("random", 1.), ("large_input", 100.)):
            x = rng.standard_normal((batch, dimension)).astype(np.float32)*scale
            with torch.inference_mode():
                expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(["residual_plan"], {"adapter_input": x})[0]
            if actual.shape != (batch, 10, 27) or not np.isfinite(actual).all():
                raise AssertionError("ONNX输出形状或有限性错误")
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
            if not (np.abs(actual) <= bounds+1e-7).all():
                raise AssertionError("ONNX残差超出训练幅值上限")
            if np.linalg.norm(actual[..., 3:6], axis=-1).max() > .05+1e-7:
                raise AssertionError("ONNX旋转向量范数超出0.05rad")
            if model.adapter.disable_root_xy and np.count_nonzero(actual[..., :2]):
                raise AssertionError("关闭的根XY通道必须严格为零")
            if not model.root_orientation and np.count_nonzero(actual[..., 3:6]):
                raise AssertionError("关闭的根姿态通道必须严格为零")
            results.append(dict(batch=batch, case=case,
                                max_abs_error=float(np.abs(actual-expected).max())))
    return dict(passed=True, provider="CPUExecutionProvider", atol=1e-6, rtol=1e-5,
                seed=42, cases=results, finite_and_bounds_passed=True,
                scope="网络导出数值一致性；不代表物理评价或整套部署验证")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT/"configs/closedloop/adapter_sim2sim_mujoco.yaml")
    parser.add_argument("--checkpoint", type=Path, help="覆盖配置中的Adapter checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="尚不存在的正式导出目录")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    checkpoint = (args.checkpoint or Path(config["adapter_checkpoint"])).expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"请使用新的导出目录，已有文件不覆盖：{output}")
    sys.path.insert(0, str(Path(config["supermimic"]["repo"]).expanduser().resolve()))
    from supermimic.robots import get_robot_spec

    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training = saved["config"]
    settings = training.get("alternating", {})
    spec = get_robot_spec("bumi3_4340")
    adapter = ReferenceAdapter(spec)
    adapter.load_state_dict(saved["adapter"], strict=True)
    adapter.disable_root_xy = not settings.get("root_xy", False)
    model = ExportAdapter(adapter, root_orientation=settings.get("root_orientation", True))
    output.mkdir(parents=True)
    report = export_and_check(model, output/"adapter.onnx")
    fields, offset = [], 0
    for field in adapter.input_spec.fields:
        stop = offset+math.prod(field.shape)
        fields.append(dict(**asdict(field), start=offset, stop=stop))
        offset = stop
    metadata = dict(
        schema="genmo-reference-adapter-onnx-v1",
        exported_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        accepted_updates=saved.get("state", {}).get("accepted_updates"),
        onnx_sha256=hashlib.sha256((output/"adapter.onnx").read_bytes()).hexdigest(),
        opset=17, robot=spec.robot_id,
        architecture=[adapter.input_spec.dimension, 512, 512, 256, 270],
        activation="SiLU", dtype="float32",
        input=dict(name="adapter_input", shape=["batch", adapter.input_spec.dimension],
                   fields=fields, normalization_in_graph=False),
        output=dict(name="residual_plan", shape=["batch", 10, 27],
                    frame_offsets=list(range(1, 11)),
                    channels=["root_dx", "root_dy", "root_dz", "root_rx", "root_ry", "root_rz",
                              *spec.joint_names],
                    bounds=adapter.bounds.tolist(), root_xy=not adapter.disable_root_xy,
                    root_orientation=model.root_orientation, rotation_norm_max_rad=.05,
                    translation_frame="episode_fixed_initial_nominal_heading",
                    rotation_composition="R_corrected = R_nominal @ Exp(rotation_vector)",
                    semantics="relative_to_nominal_not_incremental; q_corrected=q_nominal+joint_residual"),
        control_hz=50, music_hz=30,
        included=["MLP", "Tanh_bounds", "rotation_vector_norm_bound", "root_channel_switches"],
        outside_graph=["InputSpec_feature_assembly_and_normalization", "music_and_history_state",
                       "SO3_reference_composition", "committed_anchors", "halo=2*d10-d9",
                       "reference_derivatives", "Goal570", "SuperMimic_Actor"],
        checkpoint_reference_execution=training.get("reference_execution"),
        checkpoint_projection=settings.get("projection"),
        checkpoint_constraint_beta=settings.get("constraint_beta", 1.),
        runtime_assets=dict(supermimic=config["supermimic"], genmo=config["genmo"],
                            music_baseline=config["music_baseline"]),
        versions=dict(torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__),
    )
    for name, value in (("metadata.json", metadata), ("parity_report.json", report)):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"已导出：{output/'adapter.onnx'}")
    print(f"输入[B,{adapter.input_spec.dimension}] → 输出[B,10,27]；动态batch 1/16核对通过")
    print(f"最大绝对误差：{max(x['max_abs_error'] for x in report['cases']):.9g}")
    print(f"输入字段、关节序、根开关：{output/'metadata.json'}")


if __name__ == "__main__":
    main()
