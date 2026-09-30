"""有界学习率探测的 CPU 状态一致性及数值解析测试。

使用带有非空 AdamW 动量的真实小网络，将每个候选与独立、同一起点的一步更新
逐元素比较，验证探测不是累积更新，最终模型和完整优化器状态属于同一个被选
候选。故障测试覆盖预算预占、KL/回调异常、缓冲区污染、固定梯度及随机状态回滚。
另外用 FP32 权重 1.0 的真实零变化小学习率和初值零参数核验舍入/ULP 聚合统计。
本文件不加载 GENMO 大模型、不采样音乐、不执行 GPU 或物理仿真，不把单元测试
当作训练效果证据；测试数据仅在内存中构造，不写正式训练产物。
"""
from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.lr_calibration import LearningRateCalibrationRejected, calibrated_optimizer_step


class SmallActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = torch.nn.Linear(3, 2)
        self.history_encoder = torch.nn.Linear(2, 2, bias=False)
        self.scalar = torch.nn.Parameter(torch.tensor(.01))
        self.register_buffer("fixed_stats", torch.tensor([2., 3.]))
        self.register_buffer("scratch", torch.tensor(7), persistent=False)
        self.register_buffer("optional", None, persistent=False)


def gradients(actor, scale=1.):
    for index, parameter in enumerate(actor.parameters()):
        parameter.grad = (torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape) + 1 + index) * scale


def prepared():
    torch.manual_seed(71)
    actor = SmallActor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.mul_(.03)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=2e-4, betas=(.8, .95), eps=1e-8, weight_decay=.03)
    for step in range(2):
        gradients(actor, .2 + .3 * step)
        optimizer.step()
    gradients(actor, .7)
    return actor, optimizer


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.device == right.device
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def snapshot(actor, optimizer):
    return {"model": copy.deepcopy(actor.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "gradients": {name: None if value.grad is None else value.grad.clone() for name, value in actor.named_parameters()},
            "scratch": actor.scratch.clone(), "optional": None if actor.optional is None else actor.optional.clone(),
            "modes": {name: module.training for name, module in actor.named_modules()}}


def independent_step(actor, optimizer, lr):
    other = copy.deepcopy(actor)
    mapping = {id(original): replacement for original, replacement in zip(actor.parameters(), other.parameters())}
    groups = [{**{key: copy.deepcopy(value) for key, value in group.items() if key != "params"},
               "params": [mapping[id(parameter)] for parameter in group["params"]]} for group in optimizer.param_groups]
    opt = torch.optim.AdamW(groups, lr=lr)
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for group in opt.param_groups:
        group["lr"] = lr
    for (_, original), (_, parameter) in zip(actor.named_parameters(), other.named_parameters()):
        parameter.grad = None if original.grad is None else original.grad.clone()
    opt.step()
    return other, opt


def test_candidates_are_independent_real_adamw_steps_and_largest_valid_is_restored():
    actor, optimizer = prepared()
    initial = snapshot(actor, optimizer)
    candidates = [1e-9, 3e-9, 1e-8]
    expected = {lr: independent_step(actor, optimizer, lr) for lr in candidates}
    reserved, observed, events = [], [], []
    def reserve():
        reserved.append(optimizer.param_groups[0]["lr"])
        for state, original in zip(optimizer.state.values(), initial["optimizer"]["state"].values()):
            assert_same(state["step"], original["step"])
    def evaluate():
        lr = optimizer.param_groups[0]["lr"]
        comparison_actor, comparison_optimizer = expected[lr]
        assert_same(actor.state_dict(), comparison_actor.state_dict())
        assert_same(optimizer.state_dict(), comparison_optimizer.state_dict())
        observed.append(lr)
        return {"mean_joint_kl": {1e-9: .001, 3e-9: .015, 1e-8: .03}[lr],
                "per_denoising_step": [{"step_index": 0, "mean_joint_kl": .01}]}
    report = calibrated_optimizer_step(actor, optimizer, evaluate, candidates, .02,
                                        reserve_attempt=reserve, progress=events.append)
    assert reserved == observed == candidates
    assert report["selected_lr"] == report["accepted_lr"] == 3e-9
    assert report["attempt_count"] == 3 and report["accepted_updates"] == 1
    assert report["selected_candidate_index"] == 1
    assert [item["accepted"] for item in report["candidates"]] == [True, True, False]
    assert_same(actor.state_dict(), expected[3e-9][0].state_dict())
    assert_same(optimizer.state_dict(), expected[3e-9][1].state_dict())
    assert_same(snapshot(actor, optimizer)["gradients"], initial["gradients"])
    assert actor.scratch == 7 and actor.optional is None
    assert [event["event"] for event in events] == ["candidate_start", "candidate_end"] * 3 + ["selected"]
    assert events[1]["candidate"] == report["candidates"][0]
    json.dumps(report, allow_nan=False)
    for candidate in report["candidates"]:
        change = candidate["parameter_change"]
        expected_actor = expected[candidate["lr"]][0]
        differences = [(value.double() - initial["model"][name].double()).flatten() for name, value in expected_actor.named_parameters()]
        flat = torch.cat(differences)
        assert change["parameter_count"] == flat.numel()
        assert change["changed_count"] == int((flat != 0).sum())
        assert change["l2"] == pytest.approx(float(flat.norm()))
        assert change["max_abs"] == float(flat.abs().max())
        assert sum(module["changed_count"] for module in change["per_module"].values()) == change["changed_count"]
        assert change["nonzero_gradient_count"] == flat.numel()
        assert change["unchanged_with_nonzero_gradient_count"] + change["changed_with_nonzero_gradient_count"] == flat.numel()


def test_all_rejected_restores_model_momentum_lr_buffers_gradients_and_report():
    actor, optimizer = prepared()
    initial = snapshot(actor, optimizer)
    attempts = []
    with pytest.raises(LearningRateCalibrationRejected) as caught:
        calibrated_optimizer_step(actor, optimizer, lambda: {"mean_joint_kl": .03},
                                  [1e-9, 3e-9, 1e-8], .02, reserve_attempt=lambda: attempts.append(1))
    assert len(attempts) == 3
    assert_same(snapshot(actor, optimizer), initial)
    report = caught.value.report
    assert report == caught.value.lr_calibration_report
    assert report["attempt_count"] == 3 and report["accepted_updates"] == 0
    assert report["rolled_back_to_initial_state"] and report["selected_lr"] is None
    assert len(report["candidates"]) == 3


def test_budget_failure_after_a_good_candidate_rolls_back_and_never_performs_unreserved_step():
    actor, optimizer = prepared()
    initial = snapshot(actor, optimizer)
    attempts, evaluations = [], []
    def reserve():
        attempts.append(optimizer.param_groups[0]["lr"])
        if len(attempts) == 2:
            raise OverflowError("attempt budget exhausted")
    def evaluate():
        evaluations.append(1)
        return {"mean_joint_kl": .001}
    with pytest.raises(OverflowError, match="budget exhausted") as caught:
        calibrated_optimizer_step(actor, optimizer, evaluate, [1e-9, 3e-9], .02, reserve_attempt=reserve)
    assert len(attempts) == 2 and len(evaluations) == 1
    assert_same(snapshot(actor, optimizer), initial)
    report = caught.value.lr_calibration_report
    assert report["attempt_count"] == 1 and report["candidates"][0]["accepted"]
    assert not report["candidates"][1]["attempted"] and report["selected_lr"] is None


@pytest.mark.parametrize("fault", ["exception", "nonfinite", "negative", "missing", "parameter", "gradient", "buffer", "new_buffer", "none_buffer"])
def test_kl_faults_restore_initial_state_even_after_an_earlier_acceptable_candidate(fault):
    actor, optimizer = prepared()
    initial = snapshot(actor, optimizer)
    calls = []
    def evaluate():
        calls.append(1)
        if len(calls) == 1:
            return {"mean_joint_kl": .001}
        if fault == "exception":
            raise RuntimeError("injected evaluation failure")
        if fault == "nonfinite":
            return {"mean_joint_kl": float("nan")}
        if fault == "negative":
            return {"mean_joint_kl": -.1}
        if fault == "missing":
            return {"other": 1.}
        if fault == "parameter":
            actor.scalar.add_(1.)
        if fault == "gradient":
            actor.scalar.grad.add_(1.)
        if fault == "buffer":
            actor.scratch.add_(1)
        if fault == "new_buffer":
            actor.register_buffer("unexpected", torch.ones(1))
        if fault == "none_buffer":
            actor.optional = torch.ones(2)
        return {"mean_joint_kl": .001}
    with pytest.raises((RuntimeError, ValueError, FloatingPointError)) as caught:
        calibrated_optimizer_step(actor, optimizer, evaluate, [1e-9, 3e-9], .02)
    assert len(calls) == 2
    assert_same(snapshot(actor, optimizer), initial)
    assert "unexpected" not in actor._buffers
    assert caught.value.lr_calibration_report["attempt_count"] == 2


@pytest.mark.parametrize("event", ["candidate_start", "candidate_end", "selected"])
def test_progress_mutating_parameters_is_detected_and_rolled_back(event):
    actor, optimizer = prepared()
    initial = snapshot(actor, optimizer)
    def progress(message):
        if message["event"] == event:
            with torch.no_grad():
                actor.scalar.add_(1.)
    with pytest.raises(RuntimeError, match="callback changed"):
        calibrated_optimizer_step(actor, optimizer, lambda: {"mean_joint_kl": .001}, [1e-8], .02, progress=progress)
    assert_same(snapshot(actor, optimizer), initial)


def test_fp32_real_zero_change_is_rejected_and_nonzero_gradient_rounding_is_counted():
    actor = SmallActor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.fill_(1.)
    gradients(actor)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-12, weight_decay=0.)
    initial = snapshot(actor, optimizer)
    with pytest.raises(LearningRateCalibrationRejected) as caught:
        calibrated_optimizer_step(actor, optimizer, lambda: {"mean_joint_kl": 0.}, [1e-12], .02)
    assert_same(snapshot(actor, optimizer), initial)
    candidate = caught.value.report["candidates"][0]
    change = candidate["parameter_change"]
    assert not candidate["accepted"] and candidate["status"] == "rejected_zero_parameter_change"
    assert change["changed_count"] == change["l2"] == change["max_abs"] == 0
    assert change["unchanged_with_nonzero_gradient_count"] == change["parameter_count"]
    assert change["fp32_ulp"]["zero_change_fraction"] == 1.
    assert change["fp32_ulp"]["max_delta_ulp"] == 0.
    assert change["fp32_ulp"]["min_positive_delta_ulp"] is None


def test_fp32_zero_initial_values_have_finite_separate_ulp_diagnostics():
    actor = SmallActor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    gradients(actor)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-9, weight_decay=0.)
    report = calibrated_optimizer_step(actor, optimizer, lambda: {"mean_joint_kl": .001}, [1e-9], .02)
    change = report["candidates"][0]["parameter_change"]
    ulp = change["fp32_ulp"]
    assert ulp["initial_zero_count"] == change["parameter_count"]
    assert ulp["mean_delta_ulp_excluding_initial_zero"] is None
    assert not ulp["used_as_acceptance_gate"]
    assert ulp["max_delta_ulp"] > 0
    json.dumps(report, allow_nan=False)


def test_diagnostics_and_callbacks_reuse_identical_rng_and_restore_all_cpu_rngs():
    actor, optimizer = prepared()
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    draws = []
    def evaluate():
        draws.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
        actor.eval()
        return {"mean_joint_kl": .001}
    report = calibrated_optimizer_step(actor, optimizer, evaluate, [1e-9, 3e-9, 1e-8], .02)
    assert draws[0] == draws[1] == draws[2]
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0] and after_numpy[2:] == before[1][2:]
    np.testing.assert_array_equal(after_numpy[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert actor.training and report["rng_state_restored"]


@pytest.mark.parametrize("candidates,limit", [([], .02), ([1e-9] * 2, .02), ([3e-9, 1e-9], .02),
    ([1e-9, 2e-9, 3e-9, 4e-9], .02), ([0.], .02), ([-1e-9], .02), ([1e-5], .02),
    ([float("nan")], .02), ([True], .02), ([1e-9], 0.), ([1e-9], float("inf"))])
def test_invalid_candidate_configuration_is_rejected_before_any_mutation(candidates, limit):
    actor, optimizer = prepared()
    initial = snapshot(actor, optimizer)
    calls = []
    with pytest.raises(ValueError):
        calibrated_optimizer_step(actor, optimizer, lambda: calls.append(1), candidates, limit, reserve_attempt=lambda: calls.append(1))
    assert not calls
    assert_same(snapshot(actor, optimizer), initial)


def test_two_groups_with_equal_lr_preserve_distinct_adam_hyperparameters():
    actor = SmallActor()
    parameters = list(actor.parameters())
    optimizer = torch.optim.AdamW([{"params": parameters[:2], "weight_decay": .1},
                                  {"params": parameters[2:], "weight_decay": 0.}], lr=1e-6)
    gradients(actor)
    optimizer.step()
    gradients(actor, .2)
    expected_actor, expected_optimizer = independent_step(actor, optimizer, 3e-9)
    initial = copy.deepcopy(optimizer.state_dict())
    report = calibrated_optimizer_step(actor, optimizer, lambda: {"mean_joint_kl": .001}, [1e-9, 3e-9], .02)
    assert report["selected_lr"] == 3e-9
    assert [group["weight_decay"] for group in optimizer.param_groups] == [.1, 0.]
    assert [group["lr"] for group in optimizer.param_groups] == [3e-9, 3e-9]
    assert_same(actor.state_dict(), expected_actor.state_dict())
    assert_same(optimizer.state_dict(), expected_optimizer.state_dict())
    for key, value in optimizer.state_dict()["state"].items():
        assert value["step"] == initial["state"][key]["step"] + 1


def test_different_initial_group_lrs_are_rejected():
    actor = SmallActor()
    parameters = list(actor.parameters())
    optimizer = torch.optim.AdamW([{"params": parameters[:2], "lr": 1e-6}, {"params": parameters[2:], "lr": 1e-8}])
    gradients(actor)
    with pytest.raises(ValueError, match="same learning rate"):
        calibrated_optimizer_step(actor, optimizer, lambda: {"mean_joint_kl": .001}, [1e-9], .02)
