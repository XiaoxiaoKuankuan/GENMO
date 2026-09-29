"""Stage1 ONNX 条件包装与真实导出回归测试。

复用已有小型真实Actor夹具，验证合并CFG与原双分支、P=0/P=12、末端padding和
空历史的行为一致；主动启用历史和前缀残差权重，防止零初始化掩盖条件错误。
另执行真实ONNX导出和CPU Runtime的完整DDIM比较，检查已知物理前缀每步保持。
所有文件只写入pytest临时目录，不读取正式训练模型，不声称验证生成质量。
"""

# ruff: noqa: F811

from __future__ import annotations

import pytest
import torch

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.runtime.closedloop_stage1_onnx import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    Stage1GuidedDenoiser,
    Stage1OnnxSampler,
    make_inputs,
)
from tests.closedloop.test_stage1_actor import actor_factory  # noqa: F401


@pytest.mark.parametrize(
    "prefix,start,empty_history,cfg",
    [(0, 45, False, 2.5), (12, 45, False, 2.5), (12, 170, True, 1.0)],
)
def test_guided_wrapper_matches_actor(actor_factory, prefix, start, empty_history, cfg):
    actor, batch = actor_factory(prefix=prefix, starts=(start,))
    actor.eval()
    with torch.no_grad():
        actor.history_encoder.out_proj.weight.normal_(std=0.01)
        actor.prefix_encoder.out_proj.weight.normal_(std=0.01)
    conditions = {k: batch[k] for k in STAGE1_CONDITION_KEYS}
    if empty_history:
        conditions["proprio_history_valid"] = torch.zeros_like(conditions["proprio_history_valid"])
    noise, timestep = torch.randn(1, 120, 30), torch.tensor([950])
    adapted = actor.adapt_conditions(conditions)
    cond = actor._denoise(noise, timestep, adapted, actor.encode_conditions(adapted))
    uncond = actor._denoise(
        noise, timestep, adapted, actor.encode_conditions(adapted, torch.ones(1, dtype=torch.bool))
    )
    predicted = Stage1GuidedDenoiser(actor)(*make_inputs(conditions, noise, timestep, cfg))
    reference = actor._constrain(
        uncond["pred_x_start"] + cfg * (cond["pred_x_start"] - uncond["pred_x_start"]), adapted
    )
    torch.testing.assert_close(predicted[0], reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        predicted[1],
        uncond["static_conf_logits"]
        + cfg * (cond["static_conf_logits"] - uncond["static_conf_logits"]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_export_and_ddim_runtime(actor_factory, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    actor, batch = actor_factory(prefix=12, starts=(45,))
    actor.eval()
    conditions = {k: batch[k] for k in STAGE1_CONDITION_KEYS}
    noise = torch.randn(1, 120, 30)
    path = tmp_path / "stage1.onnx"
    torch.onnx.export(
        Stage1GuidedDenoiser(actor).eval(),
        make_inputs(conditions, noise, torch.tensor([999]), 2.5),
        str(path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=18,
        dynamo=False,
    )
    runtime = Stage1OnnxSampler(path, actor, device="cpu")
    expected = actor.sample(conditions, steps=2, guidance_scale=2.5, noise=noise)
    actual = runtime.sample(conditions, steps=2, guidance_scale=2.5, noise=noise, return_trace=True)
    torch.testing.assert_close(actual["qpos30"], expected["qpos30"], atol=5e-5, rtol=5e-5)
    mask = conditions["known_qpos30_mask"]
    assert torch.equal(actual["qpos30"][mask], conditions["known_qpos30"][mask])
    known = actor.endecoder.normalize(conditions["known_qpos30"])
    assert all(torch.equal(x[mask], known[mask]) for x in actual["trace"])
