"""Stage1 完整训练 checkpoint 的恢复、拒绝错误与原子发布验证。

本文件复用仓库真实 Stage1 Actor 的小型测试夹具，测试重点是状态持久化协议：原音乐模型
的 weights-only warm start 不会被当作续训；新训练文件同时绑定模型接口、资产身份、优化器
参数名顺序、scheduler、AMP、数据与采样契约，以及每个 rank 的随机状态和采样位置。
通过连续优化与保存后恢复的逐参数完全一致，验证恢复的不只是 global_step 显示值。

所有 checkpoint 只写入 pytest 的临时目录，损坏测试只修改内存副本，不操作正式模型或数据。
这些单元测试不替代服务器1真实完整模型的单卡、8卡 DDP、10+5 步恢复验收。
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from gem.closedloop.checkpoint import (
    load_stage1_checkpoint,
    load_training_checkpoint,
    save_stage1_checkpoint,
    save_training_checkpoint,
)
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory


def _runtime(step=2):
    return {
        "world_size": 1,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "precision": "fp32",
        "data_fingerprint": "test-only-manifest-sha256",
        "sampling": {"seed": 42, "epoch_size_per_rank": 100, "weights": [0.2, 0.35, 0.25, 0.2]},
        "training_contract": {"optimizer": "AdamW", "scheduler_total_steps": 100},
        "rank_states": [
            {
                "rank": 0,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": [],
                },
                "sampler_epoch": 0,
                "sampler_offset": step * 2,
            }
        ],
        "completed_steps": step,
    }


def _components(actor, *, amp=False, reverse=False):
    parameters = list(actor.parameters())
    if reverse:
        parameters.reverse()
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - step / 100)
    scaler = torch.amp.GradScaler("cpu") if amp else None
    return optimizer, scheduler, scaler


def _step(actor, optimizer, scheduler, scaler=None):
    optimizer.zero_grad(set_to_none=True)
    # 所有参数均参与非线性目标，保证各层动量都确实初始化且影响恢复后的下一次更新。
    loss = sum(parameter.square().sum() for parameter in actor.parameters())
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()


def _save(actor, path, optimizer, scheduler, scaler=None):
    return save_training_checkpoint(
        actor,
        path,
        config={"unit_test": True},
        global_step=2,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        runtime_state=_runtime(),
        warm_start_report={"mode": "weights_only_warm_start", "source_global_step_for_provenance_only": 30},
    )


@pytest.mark.parametrize("amp", [False, True])
def test_resume_preserves_full_state_and_next_update_exactly(actor_factory, tmp_path, amp):
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    optimizer, scheduler, scaler = _components(actor, amp=amp)
    for _ in range(2):
        _step(actor, optimizer, scheduler, scaler)
    path = tmp_path / "step_00000002.pt"
    payload = _save(actor, path, optimizer, scheduler, scaler)
    restored = copy.deepcopy(actor)
    with torch.no_grad():
        next(restored.parameters()).add_(0.5)
    restored_optimizer, restored_scheduler, restored_scaler = _components(restored, amp=amp)
    expected = {key: value for key, value in _runtime().items() if key not in {"rank_states", "completed_steps"}}
    report = load_training_checkpoint(
        restored,
        path,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        expected_runtime=expected,
    )
    assert report["mode"] == "stage1_training_resume"
    assert report["global_step"] == 2 and report["optimizer_restored"]
    assert report["scheduler_restored"] and report["scaler_restored"] is amp
    assert report["runtime_state"]["rank_states"][0]["sampler_offset"] == 4
    assert report["warm_start_report"]["mode"] == "weights_only_warm_start"
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    if amp:
        assert restored_scaler.state_dict() == scaler.state_dict()
    _step(actor, optimizer, scheduler, scaler)
    _step(restored, restored_optimizer, restored_scheduler, restored_scaler)
    for name, expected_value in actor.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected_value, atol=0, rtol=0)
    for left, right in zip(optimizer.state.values(), restored_optimizer.state.values()):
        for name, expected_value in left.items():
            torch.testing.assert_close(right[name], expected_value, atol=0, rtol=0)
    # 训练文件可显式作为纯 Actor 权重读取，但不会因此恢复 optimizer/global_step。
    weight_report = load_stage1_checkpoint(restored, payload)
    assert weight_report["global_step_restored"] is False


@pytest.mark.parametrize(
    "failure",
    ["actor", "dtype", "asset", "optimizer_shape", "optimizer_missing", "scheduler", "scaler", "runtime", "rank", "rng", "invalid_rng", "step"],
)
def test_resume_rejects_incompatibility_before_any_state_mutation(actor_factory, tmp_path, failure):
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    optimizer, scheduler, scaler = _components(actor)
    _step(actor, optimizer, scheduler)
    payload = _save(actor, tmp_path / "source.pt", optimizer, scheduler)
    if failure == "actor":
        payload["actor_interface_config"]["history_steps"] = 500
    elif failure == "dtype":
        name = next(iter(payload["state_dict"]))
        payload["state_dict"][name] = payload["state_dict"][name].double()
    elif failure == "asset":
        payload["asset_identity"]["stats_sha256"] = "different"
    elif failure == "optimizer_shape":
        next(iter(payload["optimizer_state_dict"]["state"].values()))["exp_avg"] = torch.zeros(1)
    elif failure == "optimizer_missing":
        next(iter(payload["optimizer_state_dict"]["state"].values())).pop("exp_avg")
    elif failure == "scheduler":
        payload["scheduler"]["state_dict"].pop("last_epoch")
    elif failure == "scaler":
        payload["scaler"] = {"class": "unexpected", "state_dict": {}}
    elif failure == "runtime":
        payload["runtime_state"]["data_fingerprint"] = "different"
    elif failure == "rank":
        payload["runtime_state"]["rank_states"][0]["rank"] = 1
    elif failure == "rng":
        payload["runtime_state"]["rank_states"][0]["rng"].pop("python")
    elif failure == "invalid_rng":
        payload["runtime_state"]["rank_states"][0]["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    else:
        payload["global_step"] = 3
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(RuntimeError):
        load_training_checkpoint(
            actor, payload, optimizer=optimizer, scheduler=scheduler,
            expected_runtime={"data_fingerprint": "test-only-manifest-sha256"},
        )
    for name, value in before.items():
        torch.testing.assert_close(actor.state_dict()[name], value, atol=0, rtol=0)
    assert optimizer.state_dict()["param_groups"] == optimizer_before["param_groups"]
    assert scheduler.state_dict() == scheduler_before


def test_resume_rejects_reordered_optimizer_parameters_and_weights_only(actor_factory, tmp_path):
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    optimizer, scheduler, _ = _components(actor)
    payload = _save(actor, tmp_path / "full.pt", optimizer, scheduler)
    reverse_optimizer, reverse_scheduler, _ = _components(actor, reverse=True)
    with pytest.raises(RuntimeError, match="layout/class"):
        load_training_checkpoint(actor, payload, optimizer=reverse_optimizer, scheduler=reverse_scheduler)
    weights = save_stage1_checkpoint(actor, tmp_path / "weights.pt", config={}, global_step=2)
    with pytest.raises(RuntimeError, match="weights-only is not resumable"):
        load_training_checkpoint(actor, weights, optimizer=optimizer, scheduler=scheduler)


def test_unused_scheduler_and_scaler_are_explicitly_none(actor_factory, tmp_path):
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    optimizer, scheduler, _ = _components(actor)
    payload = _save(actor, tmp_path / "without_optional.pt", optimizer, None)
    assert payload["scheduler"] is None and payload["scaler"] is None
    report = load_training_checkpoint(actor, payload, optimizer=optimizer)
    assert report["scheduler_used"] is False and report["scaler_used"] is False
    with pytest.raises(RuntimeError, match="scheduler class/presence"):
        load_training_checkpoint(actor, payload, optimizer=optimizer, scheduler=scheduler)


def test_atomic_save_refuses_overwrite_and_cleans_failed_temporary(actor_factory, tmp_path, monkeypatch):
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    optimizer, scheduler, _ = _components(actor)
    path = tmp_path / "step_00000002.pt"
    _save(actor, path, optimizer, scheduler)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _save(actor, path, optimizer, scheduler)
    assert path.read_bytes() == original

    def fail(payload, handle):
        handle.write(b"incomplete")
        raise OSError("simulated full disk")

    monkeypatch.setattr(torch, "save", fail)
    failed_path = tmp_path / "step_00000003.pt"
    with pytest.raises(OSError, match="full disk"):
        _save(actor, failed_path, optimizer, scheduler)
    assert not failed_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))
