"""实际Stage1+SM资产的整链验收测试源，本轮未执行。

必须通过GENMO_SM_TEST_CONFIG提供完整实际资产配置；缺项则skip，不创建小模型代替
真实Bumi验收。测试覆盖零分支parity、checkpoint重算和多决策梯度，物理方向另走审计CLI。
"""

import os

import pytest


def test_real_genmo_sm_differentiable_chain():
    path = os.getenv("GENMO_SM_TEST_CONFIG")
    if path is None:
        pytest.skip("需要实际Stage1/SM资产；本轮不加载或测试")
    from gem.closedloop.mb_control.application import read_config, build_components, read_tasks, task_inputs
    from gem.closedloop.mb_control.gradient_audit import tensor_fixture, audit_gradients
    cfg = read_config(path)
    device = os.getenv("GENMO_SM_TEST_DEVICE", "cuda:0")
    actor, adapter, scheduler, rewards, critic, engine, original = build_components(cfg, device)
    music, initial = task_inputs(read_tasks(cfg["tasks"])[0], adapter, device)
    result = audit_gradients(actor, critic, engine, tensor_fixture(adapter, music, initial),
                             [scheduler.bootstrap_latency_ticks] * engine.decisions, finite_difference=True)
    assert result["zero_forecast_stage1_max_abs"] <= 1e-5
    assert result["checkpoint_gradient_max_abs"] <= 1e-5
    assert result["grad_no_grad_forward_abs"] <= 1e-5
    assert result["first_noise_gradient_finite"] and result["first_noise_gradient_norm"] > 0
    assert result["accepted_commits"] >= 2
    assert result["frozen_parameters_and_buffers_unchanged"]
