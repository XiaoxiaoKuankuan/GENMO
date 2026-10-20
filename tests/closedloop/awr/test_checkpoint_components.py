"""真实断点组件验收工具的恢复与篡改回归。

先对小型可微网络执行实际AdamW更新，再验证工具恢复参数、优化矩与调度器完全相同，
工具自身不追加优化步。CPU测试不冒称原Stage1完整模型或八卡CUDA续训验收。
"""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch


def test_checkpoint_components_restore_actual_optimizer_without_update():
    path = Path(__file__).resolve().parents[3]/"tools/verify_stage2_awr_checkpoint.py"
    spec = importlib.util.spec_from_file_location("checkpoint_components", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    actor = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    actor(torch.ones(1, 2)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    payload = dict(actor=copy.deepcopy(actor.state_dict()), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), config=dict(lr=1e-4, warmup_steps=0), total_steps=1,
        steps=1, epoch=1, cursor=0, data_identity="test", bc_weight=.1)
    with torch.no_grad():
        actor.weight.zero_()
    report = tool.restore_components(actor, payload)
    assert report["optimizer_updates"] == 0
    assert report["adam_step_min"] == report["adam_step_max"] == 1
    torch.testing.assert_close(actor.weight, payload["actor"]["weight"], atol=0, rtol=0)
    invalid = copy.deepcopy(payload)
    invalid["optimizer"]["state"][0]["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError):
        tool.restore_components(actor, invalid)


def test_resume_acceptance_uses_saved_boundary_and_restores_original_actor(tmp_path, episode_factory, monkeypatch):
    from gem.closedloop.awr.dataset import ReplayDataset
    from gem.closedloop.awr.distributed import Distributed
    from tests.closedloop.awr.test_training import ArithmeticBC
    path = Path(__file__).resolve().parents[3]/"tools"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location("resume_acceptance", path/"verify_stage2_awr_resume.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    actor, _, make = episode_factory
    data = ReplayDataset([make(episode_id="a"), make(episode_id="b")], unit_weights=True)
    data.rows, data.sample_ids, data.sources = data.rows[:3], data.sample_ids[:3], data.sources[:3]
    data.returns, data.weights = data.returns[:3], torch.tensor([.5, 1., 2.], dtype=torch.float64)
    d = Distributed()
    original = d.assert_models_equal(actor)
    report = tool.verify_resume(actor, data, ArithmeticBC, dict(lr=1e-5), d, tmp_path/"checkpoint")
    assert report["diagnostic_optimizer_updates"] == 3
    assert report["actor_optimizer_scheduler_bc_rng_exact"]
    assert report["original_actor_restored"] == original
