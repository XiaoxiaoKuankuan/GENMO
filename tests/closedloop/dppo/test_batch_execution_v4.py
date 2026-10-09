"""验证新的样本矩阵批量计算契约，不以放宽概率门槛消除布局差异。

测试使用真实 Stage1Actor 的缩小架构，激活全部条件分支，覆盖同一步来自不同链、
混合去噪步与尾批；另外检查自定义 Linear 的反向、参数别名和冻结状态。真实模型
GPU 验收由独立工具执行，本测试结果不能解释为八卡闭环已验收。
"""
import copy
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo.batch_execution import (SampleMatrixLinear, set_sample_linear, batch_accounting,
                                                MicrobatchGradientAccumulator)
from gem.closedloop.dppo.policy import DPPODiffusionPolicy, masked_joint_log_prob
from gem.closedloop.dppo.tensor_cache import ConditionGraphCache, RolloutTensorCache
from gem.closedloop.dppo.updater_v2 import _parameters
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.performance import PhaseProfiler, activate, deactivate, measure
from tests.closedloop.test_stage1_actor import actor_factory, _conditions, _activate_branches


def test_sample_linear_derivative_and_parameter_identity():
    layer = torch.nn.Linear(5, 7).double()
    before = dict(layer.named_parameters())
    reference = copy.deepcopy(layer)
    set_sample_linear(layer, True)
    assert all(p is before[n] for n, p in layer.named_parameters())
    x = torch.randn(3, 4, 5, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(layer, x)
    layer(x).square().sum().backward()
    reference(x.detach()).square().sum().backward()
    for a, b in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    set_sample_linear(layer, False)
    assert type(layer) is torch.nn.Linear


def test_microbatch_accumulator_preserves_cancelling_gradient_and_freeze():
    layer=torch.nn.Linear(1,1,bias=True)
    layer.bias.requires_grad_(False)
    accumulator=MicrobatchGradientAccumulator(layer)
    for contribution in (1e8,1.,-1e8):
        layer.weight.grad=torch.full_like(layer.weight,contribution)
        accumulator.add()
        assert layer.weight.grad is None
    accumulator.finish()
    torch.testing.assert_close(layer.weight.grad,torch.ones_like(layer.weight),rtol=0,atol=0)
    assert layer.bias.grad is None and not layer.bias.requires_grad


def test_microbatch_accumulator_does_not_create_absent_gradients():
    layer=torch.nn.Linear(2,3)
    accumulator=MicrobatchGradientAccumulator(layer)
    accumulator.add();accumulator.finish()
    assert all(p.grad is None for p in layer.parameters())


@pytest.mark.parametrize('size', [2, 4, 8, 16, 32])
def test_real_batch_same_step_and_mixed_tail(actor_factory, size):
    torch.set_num_threads(1)
    actor, batch = actor_factory(starts=(45, 47))
    _activate_branches(actor)
    policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1')
    contexts = _conditions(batch)
    rows = []
    for i in range(2):
        context = {k: v[i:i+1] for k, v in contexts.items()}
        trace = policy.sample_rollout(context, generator=torch.Generator().manual_seed(14+i))
        rows.append(SimpleNamespace(context=context, chain=trace['chain'][0],
            old_log_prob=trace['old_log_probs'][0], free_mask=trace['free_mask'][0],
            transition_valid=True, metadata=dict(sampler_trace=trace, remaining_music_seconds=10.)))
    cache = RolloutTensorCache(rows, dict(advantages=torch.tensor([1., -.3])), 'cpu')
    pairs = [(row, step) for step in range(20) for row in rows]
    graph = ConditionGraphCache(policy, cache)
    graph.prime(rows)
    for start in range(0, len(pairs), size):
        group = pairs[start:start+size]
        selected, steps = zip(*group)
        parameters, mask = _parameters(policy, selected, steps, 'cpu', cache, graph)
        p = masked_joint_log_prob(cache.get('chain', selected, [s+1 for s in steps]),
                                 parameters['mean'], parameters['std'], mask)
        delta = p-cache.get('old_log_prob', selected, steps)
        assert float(delta.abs().max()) <= 1e-4
        assert float(torch.expm1(delta).abs().max()) <= 1e-3
        (-p.sum()/40).backward()
        report = batch_accounting(len(group), cfg=True)
        assert report['padding_rows'] == 0 and report['useful_row_fraction'] == 1
    graph.backward()
    for name in ('history_encoder', 'prefix_encoder', 'music_embedder'):
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in getattr(actor, name).parameters())
    cache.close()


def test_deferred_intermediate_nan_fails_before_phase_publication(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    policy = DPPODiffusionPolicy(actor, defer_checks=True)
    with pytest.raises(ValueError, match='finite'):
        with policy_phase(policy):
            state = torch.full((1, 120, 30), float('nan'))
            policy._state(state, state, 'intermediate')
            # 后续恢复有限输出不能掩盖中间去噪步的故障。
            policy._state(torch.zeros_like(state), state, 'last')
    assert policy._phase_signature is None
    with policy_phase(policy):
        assert policy._parameter_signature() is policy._phase_signature
    assert policy._phase_signature is None


def test_profiler_quiet_mode_counts_without_hotloop_events():
    profiler = PhaseProfiler(detailed=False)
    token = activate(profiler)
    try:
        with measure('compute.actor_minibatch_forward_backward'):
            for _ in range(8):
                with measure('policy.transition', gpu=True):
                    pass
    finally:
        deactivate(token)
    report = profiler.report()
    assert not report['hotloop_timing_enabled']
    assert report['stages']['policy.transition']['calls'] == 8
    assert report['stages']['policy.transition']['host_seconds'] == 0
    assert report['stages']['compute.actor_minibatch_forward_backward']['host_seconds'] > 0


def test_new_execution_contract_survives_cfg_switch(actor_factory):
    actor, _ = actor_factory()
    policy = DPPODiffusionPolicy(actor, numerical_layout='sample_matrix_bmm_fp32.v1')
    policy.cfg_batch = True
    assert policy.kernel_config['version'].endswith('.v4')
    assert policy.kernel_config['execution_contract'] == policy.numerical_layout
