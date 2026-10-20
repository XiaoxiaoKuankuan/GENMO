"""验证离线分项计时只计算梯度，不更新正式模型。

使用小型真实可微网络执行AWR/BC两个梯度分支，检查计时归约、裁剪后参数不变及
优化器step未调用。该测试不冒充八卡性能；正式性能数值由完成运行后的独立工具提供。
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def test_backward_profile_never_updates_model_or_optimizer():
    path = Path(__file__).resolve().parents[3]/"tools/profile_stage2_awr_backward.py"
    spec = importlib.util.spec_from_file_location("backward_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actor = torch.nn.Linear(2, 1)
    previous = {k: v.clone() for k, v in actor.state_dict().items()}
    optimizer = torch.optim.AdamW(actor.parameters())
    def backward(indices=None, step=None):
        loss = actor(torch.ones(2, 2)).square().mean()
        loss.backward()
        return float(loss.detach())
    trainer = SimpleNamespace(actor=actor, optimizer=optimizer, backward_awr=backward,
        backward_bc=lambda coefficient: backward(), bc_weight=.1,
        distributed=SimpleNamespace(gradients=lambda model: None), config={"gradient_clip": 1.})
    report = module.timed_backward(trainer, [0, 1], 0, lambda: None)
    assert report["optimizer_updates"] == 0 and not optimizer.state
    assert all(torch.equal(previous[k], value) for k, value in actor.state_dict().items())
    assert report["total_without_optimizer_seconds"] >= report["stage1_bc_forward_backward_seconds"] > 0
