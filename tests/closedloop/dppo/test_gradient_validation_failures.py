"""完整链梯度验收入口的纯CPU异常传播与独立Normal公式回归。

本文件复用有解析概率比的双条件网络和真实RolloutTensorCache，执行工具中的
完整Normal前反向、生产概率重算及五种流式反向组合。正常路径另用独立损失
核对参考梯度，并检查参数/RNG不变及结束后梯度释放；不通过复制生产公式
或替换反向为常数来证明数学正确性。

故障路径以记录调用顺序的CPU collective替身注入本rank前反向、参考复制、
比较异常以及另一rank报告的异常，断言所有本地阶段先同步失败，且失败后
不进入下一次梯度SUM，部分梯度也会释放。替身仅模拟控制协议与相同rank
的梯度相加，不声称完成真实Gloo/NCCL通信、CUDA OOM恢复、八卡数值核或
物理收益验收；生产2e-5阈值保持，所有测试不初始化CUDA、不写正式资产。
"""
import math

import pytest
import torch

from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from tests.closedloop.dppo.test_condition_chunk_backward import (
    CPUCollective, analytic_case, independent_loss,
)
import tools.validate_stage2_full_chain as validation


class TracedCPUCollective(CPUCollective):
    """明确记录本地异常传播边界；可模拟第二rank失败或不同诊断误差。"""

    def __init__(self, *, remote_failure_at=None, after_sum=None, peer_metrics=False):
        self.world_size = 2 if remote_failure_at is not None or peer_metrics else 1
        self.remote_failure_at = remote_failure_at
        self.after_sum = after_sum
        self.peer_metrics = peer_metrics
        self.local_phases = 0
        self.sum_calls = 0
        self.events = []
        self.gradients = []

    def all_gather_object(self, value):
        if value is None or isinstance(value, str):
            self.local_phases += 1
            self.events.append(('local_phase', self.local_phases, value))
            if self.local_phases == self.remote_failure_at:
                return [value, 'rank 1: RuntimeError: injected peer failure']
        elif isinstance(value, dict):
            self.events.append(('metrics', value))
            if self.peer_metrics:
                return [value, dict(relative_gradient_l2=1e-6,
                                    maximum_coordinate_error=2e-6,
                                    recompute_max_error=3e-6)]
        else:
            self.events.append(('gather', value))
        return [value] * self.world_size

    def sum_gradients(self, actor):
        self.sum_calls += 1
        self.events.append(('sum', self.sum_calls))
        with torch.no_grad():
            for parameter in actor.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(self.world_size)
        self.gradients.append({name: parameter.grad.clone()
            for name, parameter in actor.named_parameters() if parameter.grad is not None})
        if self.after_sum is not None:
            self.after_sum(actor, self.sum_calls)


@pytest.fixture
def case(request):
    policy, rows, targets = analytic_case(getattr(request, 'param', 5))
    # 工具的独立Normal路径从未缓存的条件接口进入，解析夹具补同一公开接口。
    policy.transition_parameters = lambda context, state, steps, prepared=None: (
        policy._transition_prepared(
            policy.prepare_conditions(context) if prepared is None else prepared, state, steps))
    cache = RolloutTensorCache(rows, targets, 'cpu')
    try:
        # 原第0/1链都在有利侧被裁剪；选第2/3链的非零梯度检验避免零比零。
        yield policy, rows[2:], cache, dict(advantages=targets['advantages'][2:4])
    finally:
        cache.close()


def assert_clean(policy):
    assert all(parameter.grad is None for parameter in policy.actor.parameters())
    assert not torch.cuda.is_initialized()


def assert_failure_synced(collective, text):
    assert collective.events[-1][0] == 'local_phase'
    assert text in collective.events[-1][2]


@pytest.mark.parametrize('case', [4, 5, 10], indirect=True)
def test_normal_formula_variants_and_final_cleanup(case):
    policy, rows, cache, targets = case
    independent_loss(policy, rows[:2], targets, denominator=2).backward()
    expected = {name: parameter.grad.clone() for name, parameter in policy.actor.named_parameters()}
    assert sum(float(value.square().sum()) for value in expected.values()) > 0
    policy.actor.zero_grad(set_to_none=True)
    before = {name: value.clone() for name, value in policy.actor.state_dict().items()}
    rng = torch.get_rng_state().clone()
    collective = TracedCPUCollective()
    report = validation.gradient_validation(policy, rows, cache, collective)
    assert report['passed'] and report['global_chains'] == 2
    assert collective.sum_calls == 6
    assert [(row['microbatch'], row['condition_chunk_chains']) for row in report['variants']] == [
        (1, None), (3, 1024), (8, 1024), (128, None), (128, 1)]
    for gradients in collective.gradients:
        assert gradients.keys() == expected.keys()
        for name, value in gradients.items():
            torch.testing.assert_close(value, expected[name], rtol=1e-11, atol=1e-12)
    for variant in report['variants']:
        assert math.isfinite(variant['relative_gradient_l2'])
        assert variant['relative_gradient_l2'] <= 2e-5
        assert variant['recompute_max_error'] < 1e-12
    for name, value in policy.actor.state_dict().items():
        assert torch.equal(value, before[name])
    assert torch.equal(rng, torch.get_rng_state())
    assert_clean(policy)


@pytest.mark.parametrize('phase', ['independent-forward', 'independent-backward', 'records', 'streamed'])
def test_local_failure_never_enters_next_gradient_sum(case, monkeypatch, phase):
    policy, rows, cache, _ = case
    collective = TracedCPUCollective()

    def fail(*args, **kwargs):
        next(policy.actor.parameters()).grad = torch.ones_like(next(policy.actor.parameters()))
        raise RuntimeError('injected ' + phase)

    hook = None
    if phase == 'independent-forward':
        monkeypatch.setattr(validation, '_parameters', fail)
    elif phase == 'independent-backward':
        hook = policy.actor.output.register_hook(fail)
    else:
        monkeypatch.setattr(validation,
            'evaluate_chain_records' if phase == 'records' else 'backward_chain_surrogate', fail)
    try:
        with pytest.raises(RuntimeError, match='rank 0: RuntimeError: injected ' + phase):
            validation.gradient_validation(policy, rows, cache, collective)
    finally:
        if hook is not None:
            hook.remove()
    assert collective.sum_calls == (0 if phase.startswith('independent') else 1)
    assert_failure_synced(collective, 'injected ' + phase)
    assert_clean(policy)


def test_reference_copy_failure_is_synchronized_and_cleaned(case, monkeypatch):
    policy, rows, cache, _ = case

    def fail_clone(tensor, *args, **kwargs):
        raise RuntimeError('injected reference allocation')

    def after_sum(actor, count):
        assert count == 1
        monkeypatch.setattr(torch.Tensor, 'clone', fail_clone)

    collective = TracedCPUCollective(after_sum=after_sum)
    with pytest.raises(RuntimeError, match='rank 0: RuntimeError: injected reference allocation'):
        validation.gradient_validation(policy, rows, cache, collective)
    assert collective.sum_calls == 1
    assert_failure_synced(collective, 'injected reference allocation')
    assert_clean(policy)


@pytest.mark.parametrize('mode', ['missing-gradient', 'threshold', 'nonfinite', 'allocation'])
def test_compare_failure_is_synchronized_before_next_variant(case, monkeypatch, mode):
    policy, rows, cache, _ = case

    def fail_double(tensor, *args, **kwargs):
        raise RuntimeError('injected comparison allocation')

    def after_sum(actor, count):
        if count != 2:
            return
        parameter = next(actor.parameters())
        if mode == 'missing-gradient':
            parameter.grad = None
        elif mode == 'allocation':
            monkeypatch.setattr(torch.Tensor, 'double', fail_double)
        else:
            with torch.no_grad():
                parameter.grad.fill_(float('nan')) if mode == 'nonfinite' else parameter.grad.mul_(10)

    collective = TracedCPUCollective(after_sum=after_sum)
    with pytest.raises(RuntimeError, match='rank 0:'):
        validation.gradient_validation(policy, rows, cache, collective)
    assert collective.sum_calls == 2
    assert_failure_synced(collective, 'rank 0:')
    assert_clean(policy)


@pytest.mark.parametrize('local_phase, expected_sums', [(2, 0), (3, 1), (4, 1), (5, 2), (6, 2)])
def test_peer_failure_propagates_before_next_sum(case, local_phase, expected_sums):
    policy, rows, cache, _ = case
    collective = TracedCPUCollective(remote_failure_at=local_phase)
    with pytest.raises(RuntimeError, match='rank 1: RuntimeError: injected peer failure'):
        validation.gradient_validation(policy, rows, cache, collective)
    assert collective.sum_calls == expected_sums
    assert collective.events[-1] == ('local_phase', local_phase, None)
    assert_clean(policy)


def test_all_ranks_report_the_same_worst_passed_error(case):
    policy, rows, cache, _ = case
    collective = TracedCPUCollective(peer_metrics=True)
    report = validation.gradient_validation(policy, rows, cache, collective)
    assert report['passed'] and report['global_chains'] == 4
    assert collective.sum_calls == 6
    for variant in report['variants']:
        assert variant['relative_gradient_l2'] == 1e-6
        assert variant['maximum_coordinate_error'] == 2e-6
        assert variant['recompute_max_error'] == 3e-6
    assert_clean(policy)
