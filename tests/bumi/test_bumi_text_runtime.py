"""BUMI 文本推理、ONNX 和可搬运部署包的 CPU 契约测试。

用合成数据训练配置构造32维两层随机网络，绝不加载真实训练权重或启动GPU。
将零初始化输出头改为非零随机权重，避免padding/length测试因常数输出而虚假通过。
真实T5在产物测试中显式替换为合成特征；ONNX仍实际导出并通过CPU Runtime执行。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate

from tests.bumi.test_bumi_text_fullseq import text_release as text_release, config
from gem.runtime.bumi_text_runtime import (
    load_checkpoint_step,
    BumiTextSampler,
    ResidentBumiTextEngine,
    OnnxTextStep,
    read_bundle,
)
from tools.export.bumi_text import export, sample_inputs, package


@pytest.fixture
def small_checkpoint(text_release, tmp_path):
    cfg = config(text_release[0])
    model = instantiate(cfg.model, _recursive_=False).cpu().eval()
    # 每个输出头都非零，测试真实注意力依赖。
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "gate_" in name:
                parameter.fill_(0.5)
            if "pred_x_start" in name or "static_conf" in name:
                parameter.normal_(std=0.02)
    payload = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(payload)
    path = tmp_path / "synthetic.ckpt"
    torch.save(payload, path)
    return path


@pytest.mark.parametrize("frames", [60, 97, 183, 299])
def test_real_step_padding_and_sampler(small_checkpoint, frames):
    step, _, diffusion = load_checkpoint_step(small_checkpoint)
    inputs = list(sample_inputs(frames=frames))
    with torch.no_grad():
        a = step(*inputs)
        short = inputs.copy()
        short[0] = inputs[0][:, :frames]
        b = step(*short)
        inputs[0][:, frames:] = torch.randn_like(inputs[0][:, frames:]) * 100
        c = step(*inputs)
    assert a[0][:, :frames].std() > 1e-6
    for x, y, z in zip(a, b, c):
        torch.testing.assert_close(x[:, :frames], y, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(x[:, :frames], z[:, :frames], atol=2e-5, rtol=2e-5)
    sampler = BumiTextSampler(step, diffusion, 2)
    old = sampler.diffusion
    assert not sampler.set_ddim_steps(2) and sampler.diffusion is old
    noise = torch.randn(1, frames, 30)
    p = sampler.generate(inputs[2], inputs[3], frames, noise=noise)
    q = sampler.generate(inputs[2], inputs[3], frames, noise=noise, tensor_frames=frames)
    for x, y in zip(p, q):
        torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)
    assert sampler.set_ddim_steps(3) and sampler.diffusion is not old


def test_cpu_onnx_length_and_movable_bundle(small_checkpoint, tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    output = tmp_path / "export" / "model.onnx"
    export(small_checkpoint, output)
    candidate = OnnxTextStep(output)
    step, _, diffusion = load_checkpoint_step(small_checkpoint)
    predictions = []
    for frames in [60, 97, 120, 183, 240, 299, 300]:
        inputs = sample_inputs(frames=frames)
        with torch.no_grad():
            a, b = step(*inputs), candidate(*inputs)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-3)
        predictions.append(b[0][:, :60])
    assert not torch.allclose(predictions[0], predictions[-1]), "length必须是运行时输入"
    noise = torch.randn(1, 97, 30)
    inputs = sample_inputs()
    for x, y in zip(
        BumiTextSampler(step, diffusion, 2).generate(inputs[2], inputs[3], 97, noise=noise),
        BumiTextSampler(candidate, diffusion, 2).generate(inputs[2], inputs[3], 97, noise=noise),
    ):
        torch.testing.assert_close(x, y, atol=2e-3, rtol=2e-3)
    package(output, tmp_path / "bundle")
    bundle = tmp_path / "moved"
    (tmp_path / "bundle").rename(bundle)
    # 配置可编辑；模型资产仍严格校验。包内没有训练checkpoint。
    with (bundle / "deployment.ini").open("a") as file:
        file.write("\n# user comment\n")
    read_bundle(bundle / "deployment.json")
    assert not list(bundle.rglob("*.ckpt"))
    script = """
import sys
from gem.runtime.bumi_text_runtime import ResidentBumiTextEngine
engine=ResidentBumiTextEngine(deployment_manifest='deployment.json',backend='onnx',device='cpu',ddim_steps=2)
engine.initialize()
assert not any(name.split('.')[0] in {'hydra','lightning','pytorch_lightning'} for name in sys.modules)
engine.close()
"""
    env = dict(
        os.environ, PYTHONPATH=str(bundle), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=bundle, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_engine_outputs_only_requested_frames(small_checkpoint, tmp_path, monkeypatch):
    engine = ResidentBumiTextEngine(
        small_checkpoint, device="cpu", ddim_steps=2, output_root=tmp_path / "jobs"
    )
    inputs = sample_inputs(frames=240)
    monkeypatch.setattr(engine, "encode_prompt", lambda prompt: (inputs[2], inputs[3]))
    try:
        result = engine.generate(dict(prompt="test-only", num_frames=240, seed=42, postproc=False))
        assert result["ok"], result
        from gem.runtime.bumi_text_viewer import check_bumi_motion

        check_bumi_motion(Path(result["output_dir"]) / "motion.npz", 240)
        assert not engine.generate(dict(prompt="test-only", num_frames=301))["ok"]
    finally:
        engine.close()
