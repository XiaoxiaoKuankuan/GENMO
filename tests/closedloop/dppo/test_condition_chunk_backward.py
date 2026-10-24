"""完整链PPO条件图分块的独立CPU数学、生命周期和真实小网络回归。

解析用例将每条链设为已知联合概率比，以独立torch.distributions.Normal完整图
计算参考loss/梯度；生产路径仍使用真实RolloutTensorCache、ConditionGraphCache、
Gaussian统计和流式反向。覆盖正负/零优势、有利侧裁剪、未裁剪尾部、保护坐标、
四/五/十步合同、非整除尾块及大于本地链数的全局分母，避免测试只是镜像循环。
记录每链条件编码次数与条件图释放边界，默认1024链限制另以1025链验证。

真实小型Stage1网络复用既有临时资产夹具，对照None无界参考与2/2/1链块；
它验证条件编码器和去噪器共同反传、参数/RNG不变，不替代正式1024隐维网络、
八卡SUM、GPU峰值或物理收益验收。所有测试仅用CPU，资产只写pytest basetemp。
离线工具边界诊断另以同一解析网络检查1025条有效链的选择、全参数对照、RNG/
梯度清理、样本不足拒绝及CLI默认关闭；CPU峰值明确为None，不伪造GPU测量。
"""
from types import SimpleNamespace
import weakref

import pytest
import torch

import gem.closedloop.dppo.chain_ppo as chain_module
from gem.closedloop.dppo.chain_ppo import evaluate_chain_records, backward_chain_surrogate
from gem.closedloop.dppo.denoising_contract import steps_for_count
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.tensor_cache import ConditionGraphCache, RolloutTensorCache
from tests.closedloop.test_stage1_actor import actor_factory
from tests.closedloop.dppo.test_short_denoising import short_case
from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
from gem.closedloop.dppo.rollout_helpers import split_trace


class AnalyticPolicy:
    """可独立算Normal参考的双条件分支网络；不复刻生产裁剪或反向实现。"""
    batched_conditions = True

    def __init__(self, count):
        self.trainable_steps = steps_for_count(count)
        self.actor = torch.nn.Module()
        self.actor.encoder = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        self.actor.output = torch.nn.Parameter(torch.tensor([.11, -.07], dtype=torch.float64))
        with torch.no_grad():
            self.actor.encoder.weight.copy_(torch.tensor([[.2, -.03], [.04, .15]]))
        self.actor.eval()
        self.encoded = []

    def _parameter_signature(self):
        return tuple((id(p), p._version, p.requires_grad) for p in self.actor.parameters())

    def prepare_conditions(self, context):
        self.encoded.extend(context['chain_id'].flatten().tolist())
        feature = self.actor.encoder(context['feature'])[:, None, :].expand(-1, 2, -1)
        return dict(conditional=feature, unconditional=feature * .25,
                    adapted={name: context[name] for name in
                             ('known_x', 'known_mask', 'future_valid', 'context_valid')})

    def mean(self, feature, steps):
        condition = self.actor.encoder(feature)[:, None, :].expand(-1, 2, -1)
        return condition * 1.125 + self.actor.output * (steps[:, None, None] + 1) / 20

    def _transition_prepared(self, prepared, state, steps):
        mean = (prepared['conditional'] + .5 * prepared['unconditional']
                + self.actor.output * (steps[:, None, None] + 1) / 20)
        return dict(mean=mean, std=mean.new_ones((len(mean), 1, 1)))


def analytic_case(count=5, chains=7):
    policy = AnalyticPolicy(count)
    steps = policy.trainable_steps
    total = max(steps) + 2
    ratios = torch.tensor([1.5, .5, .5, 1.5, 1., 1.05, 1.2], dtype=torch.float64)
    advantages = torch.tensor([1., -1., 1., -1., 0., .7, -.2], dtype=torch.float64)
    rows = []
    with torch.no_grad():
        for i in range(chains):
            feature = torch.tensor([[.5 + .001 * i, -.2]], dtype=torch.float64)
            mask = torch.tensor([[True, False], [i % 2 == 0, True]])
            context = dict(feature=feature, chain_id=torch.tensor([[i]]),
                           known_x=torch.zeros(1, 2, 2, dtype=torch.float64),
                           known_mask=(~mask)[None], future_valid=torch.ones(1, 2, dtype=torch.bool),
                           context_valid=torch.ones(1, 2, dtype=torch.bool))
            values = torch.zeros(total + 1, 2, 2, dtype=torch.float64)
            for step in steps:
                mean = policy.mean(feature, torch.tensor([step]))[0]
                square = mean.square().masked_fill(~mask, 0.).sum()
                # 固定旧均值0、方差1时，每步log-ratio为x·mu-|mu|²/2。
                scale = (ratios[i % 7].log() / count + .5 * square) / square
                values[step + 1] = torch.where(mask, scale * mean, 1e6)
            rows.append(SimpleNamespace(context=context, identity={}, chain=values,
                old_log_prob=torch.zeros(total, dtype=torch.float64), free_mask=mask,
                metadata=dict(critic_context={}, sampler_trace=dict(kernel_config={'analytic': True},
                    sampling_noise=torch.zeros(1, total + 1, 2, 2, dtype=torch.float64),
                    old_means=torch.zeros(1, total, 2, 2, dtype=torch.float64),
                    old_stds=torch.ones(1, total, 1, 1, dtype=torch.float64)))))
    targets = dict(advantages=advantages[torch.arange(chains) % 7], valid=torch.ones(chains, dtype=torch.bool))
    return policy, rows, targets


def independent_loss(policy, rows, targets, denominator):
    features = torch.cat([row.context['feature'] for row in rows])
    mask = torch.stack([row.free_mask for row in rows])
    terms = []
    for step in policy.trainable_steps:
        mean = policy.mean(features, torch.full((len(rows),), step))
        value = torch.stack([row.chain[step + 1] for row in rows])
        old = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
        new = torch.distributions.Normal(mean, torch.ones_like(mean))
        terms.append((new.log_prob(value) - old.log_prob(value)).masked_fill(~mask, 0.).sum((-2, -1)))
    ratio = torch.stack(terms, 1).exp()
    advantage = targets['advantages'][:, None]
    return -torch.minimum(ratio * advantage, ratio.clamp(.9, 1.1) * advantage).sum() / (denominator * len(policy.trainable_steps))


def observe_condition_lifetime(monkeypatch):
    state = dict(live=0, sizes=[])
    previous_totals = []

    class ObservedConditions(ConditionGraphCache):
        def prime(self, rows):
            assert state['live'] == 0, '上一块条件图必须先反传并释放'
            assert all(value() is None for value in previous_totals), '上块微批图仍持有FP64条件梯度池'
            state['live'] += 1
            state['sizes'].append(len(rows))
            return super().prime(rows)

        def backward(self):
            previous_totals[:] = [weakref.ref(value) for value in self.bank['totals'].values()]
            super().backward()
            assert self.finished and self.bank is None and not self.entries
            state['live'] -= 1

    monkeypatch.setattr(chain_module, 'ConditionGraphCache', ObservedConditions)
    return state


@pytest.mark.parametrize('count', [4, 5, 10])
@pytest.mark.parametrize('chunk', [None, 1, 2, 4])
def test_blocked_gradients_match_independent_normal_with_tail_and_global_denominator(count, chunk, monkeypatch):
    policy, rows, targets = analytic_case(count)
    cache = RolloutTensorCache(rows, targets, 'cpu')
    try:
        expected = independent_loss(policy, rows, targets, denominator=11)
        expected.backward()
        reference = {name: p.grad.clone() for name, p in policy.actor.named_parameters()}
        policy.actor.zero_grad(set_to_none=True)
        records = evaluate_chain_records(policy, rows, cache, microbatch=3)
        policy.encoded.clear()
        state = observe_condition_lifetime(monkeypatch)
        rng = torch.get_rng_state().clone()
        report = backward_chain_surrogate(policy, rows, cache, records, clip=.1,
            global_chain_count=11, microbatch=3, condition_chunk_chains=chunk)
        torch.testing.assert_close(report['loss'], expected.detach(), rtol=1e-8, atol=1e-9)
        assert report['recompute_max_error'] < 1e-12
        for name, p in policy.actor.named_parameters():
            torch.testing.assert_close(p.grad, reference[name], rtol=1e-11, atol=1e-12)
        assert policy.encoded == list(range(len(rows)))
        assert sum(state['sizes']) == len(rows) and state['live'] == 0
        assert max(state['sizes']) <= (len(rows) if chunk is None else chunk)
        assert torch.equal(rng, torch.get_rng_state())
        # 继续向同一参数梯度累加，模拟另一个本地链分片；不允许块内清空grad。
        second = backward_chain_surrogate(policy, rows, cache, records, clip=.1,
            global_chain_count=11, microbatch=3, condition_chunk_chains=chunk)
        torch.testing.assert_close(second['loss'], report['loss'], rtol=0, atol=0)
        for name, p in policy.actor.named_parameters():
            torch.testing.assert_close(p.grad, 2 * reference[name], rtol=1e-11, atol=1e-12)
    finally:
        cache.close()


def test_default_limits_condition_graph_to_1024_chains_and_preserves_tail(monkeypatch):
    policy, rows, targets = analytic_case(chains=1025)
    cache = RolloutTensorCache(rows, targets, 'cpu')
    try:
        records = evaluate_chain_records(policy, rows, cache, microbatch=128)
        backward_chain_surrogate(policy, rows, cache, records, clip=.1,
            global_chain_count=1029, microbatch=128, condition_chunk_chains=None)
        reference = {name: p.grad.clone() for name, p in policy.actor.named_parameters()}
        policy.actor.zero_grad(set_to_none=True)
        policy.encoded.clear()
        state = observe_condition_lifetime(monkeypatch)
        backward_chain_surrogate(policy, rows, cache, records, clip=.1,
            global_chain_count=1029, microbatch=128)
        assert state == dict(live=0, sizes=[1024, 1])
        assert policy.encoded == list(range(1025))
        for name, p in policy.actor.named_parameters():
            torch.testing.assert_close(p.grad, reference[name], rtol=1e-11, atol=1e-12)
    finally:
        cache.close()


@pytest.mark.parametrize('chunk', [0, -1, True, 1.5])
def test_invalid_condition_chunk_is_rejected(chunk):
    with pytest.raises(ValueError, match='positive integer or None'):
        backward_chain_surrogate(None, [], None, None, clip=.1,
            global_chain_count=1, microbatch=3, condition_chunk_chains=chunk)


def test_tail_recompute_mismatch_and_empty_rank_keep_existing_contract():
    policy, rows, targets = analytic_case()
    cache = RolloutTensorCache(rows, targets, 'cpu')
    try:
        records = evaluate_chain_records(policy, rows, cache, microbatch=3)
        records[-1, -1, 2] += 1e-3
        with pytest.raises(FloatingPointError, match='recomputation changed probabilities'):
            backward_chain_surrogate(policy, rows, cache, records, clip=.1,
                global_chain_count=11, microbatch=3, condition_chunk_chains=2)
        policy.actor.zero_grad(set_to_none=True)
        result = backward_chain_surrogate(policy, [], cache, records[:0], clip=.1,
            global_chain_count=11, microbatch=3, condition_chunk_chains=2)
        assert result['loss'] == result['recompute_max_error'] == 0
        assert all(p.grad is None for p in policy.actor.parameters())
    finally:
        cache.close()


def test_real_small_network_five_steps_matches_unbounded_condition_graph(actor_factory):
    policy, context = short_case(actor_factory, 30, 60)
    privileged = tracking_context(context)
    trace = policy.sample_rollout(context, generator=torch.Generator().manual_seed(17))
    rows = []
    for i in (0, 1, 0, 1, 0):
        rows.append(SimpleNamespace(context={k: v[i:i+1].clone() for k, v in context.items()}, identity={},
            chain=trace['chain'][i].clone(), old_log_prob=trace['old_log_probs'][i].clone(),
            free_mask=trace['free_mask'][i].clone(), metadata=dict(
                critic_context={k: v[i:i+1].clone() for k, v in privileged.items()},
                sampler_trace=split_trace(trace, i, 2))))
    targets = dict(advantages=torch.tensor([1., -1., .4, -.3, 0.]), valid=torch.ones(5, dtype=torch.bool))
    cache = RolloutTensorCache(rows, targets, 'cpu')
    try:
        signature = policy._parameter_signature()
        rng = torch.get_rng_state().clone()
        with policy_phase(policy):
            records = evaluate_chain_records(policy, rows, cache, microbatch=3)
            reference_report = backward_chain_surrogate(policy, rows, cache, records, clip=.1,
                global_chain_count=8, microbatch=3, condition_chunk_chains=None)
        reference = {name: p.grad.clone() for name, p in policy.actor.named_parameters() if p.grad is not None}
        policy.actor.zero_grad(set_to_none=True)
        with policy_phase(policy):
            report = backward_chain_surrogate(policy, rows, cache, records, clip=.1,
                global_chain_count=8, microbatch=3, condition_chunk_chains=2)
        delta = sum((p.grad.double() - reference[name].double()).square().sum()
                    for name, p in policy.actor.named_parameters() if name in reference)
        scale = sum(value.double().square().sum() for value in reference.values())
        assert float((delta / scale).sqrt()) < 2e-4  # CPU native GEMM；八卡固定核另验收。
        torch.testing.assert_close(report['loss'], reference_report['loss'], rtol=0, atol=0)
        assert report['recompute_max_error'] <= 1e-4
        assert policy._parameter_signature() == signature
        assert torch.equal(rng, torch.get_rng_state())
        assert policy.assert_frozen_identity()['checked']
        assert not torch.cuda.is_initialized()
    finally:
        cache.close()


class CPUCollective:
    """单rank CPU控制流夹具；不把此空SUM实现称为真实多卡验收。"""
    rank = 0
    world_size = 1
    device = torch.device('cpu')

    def all_gather_object(self, value):
        return [value]

    def sum_tensor(self, value):
        return value.detach().clone()

    def sum_gradients(self, actor):
        pass


@pytest.mark.parametrize('chains', [1026, 1027])
def test_boundary_diagnostic_requires_1025_eligible_and_clears_gradients(chains, monkeypatch):
    from tools.validate_stage2_full_chain import condition_chunk_boundary_validation
    # CPU执行不得因宿主机恰好有GPU而初始化CUDA RNG。
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    policy, rows, targets = analytic_case(chains=chains)
    manifest = [dict(owner_rank=0, local_index=i, valid=i != 0, has_free=i != 1) for i in range(chains)]
    cache = RolloutTensorCache(rows, targets, 'cpu')
    for parameter in policy.actor.parameters():
        parameter.grad = torch.ones_like(parameter)
    rng = torch.get_rng_state().clone()
    try:
        if chains == 1026:
            with pytest.raises(ValueError, match='1025 eligible chains per rank'):
                condition_chunk_boundary_validation(policy, rows, cache, manifest, CPUCollective())
        else:
            report = condition_chunk_boundary_validation(policy, rows, cache, manifest, CPUCollective())
            assert report['passed'] and report['global_chains'] == 1025
            assert report['available_chains_by_rank'] == report['selected_chains_by_rank'] == [1025]
            assert [value['chains_per_condition_block'] for value in report['variants']] == [[1025], [1024, 1]]
            assert all(value['relative_gradient_l2'] <= 2e-5 for value in report['variants'])
            assert all(value['absolute_loss_difference'] == 0 for value in report['variants'])
            assert all(value['peak_allocated_bytes'] is None for value in report['variants'])
            assert all(value['identical'] for value in report['state_unchanged_by_rank'])
        assert all(parameter.grad is None for parameter in policy.actor.parameters())
        assert torch.equal(rng, torch.get_rng_state())
        assert not torch.cuda.is_initialized()
    finally:
        cache.close()


def test_condition_boundary_cli_is_explicit_and_disabled_by_default():
    from tools.validate_stage2_full_chain import argument_parser
    args = ['--case', '/tmp/unread_case', '--output-dir', '/tmp/uncreated_output']
    assert not argument_parser().parse_args(args).audit_condition_chunk_boundary
    assert argument_parser().parse_args(args + ['--audit-condition-chunk-boundary']).audit_condition_chunk_boundary
    assert not torch.cuda.is_initialized()


def test_boundary_diagnostic_failure_restores_rng_and_clears_partial_gradients(monkeypatch):
    import tools.validate_stage2_full_chain as diagnostic
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    policy, rows, targets = analytic_case(chains=1025)
    manifest = [dict(owner_rank=0, local_index=i, valid=True, has_free=True) for i in range(1025)]
    cache = RolloutTensorCache(rows, targets, 'cpu')
    rng = torch.get_rng_state().clone()

    def fail_backward(*args, **kwargs):
        torch.rand(3)
        for parameter in policy.actor.parameters():
            parameter.grad = torch.ones_like(parameter)
        raise RuntimeError('injected boundary failure')

    monkeypatch.setattr(diagnostic, 'backward_chain_surrogate', fail_backward)
    try:
        with pytest.raises(RuntimeError, match='injected boundary failure'):
            diagnostic.condition_chunk_boundary_validation(policy, rows, cache, manifest, CPUCollective())
        assert all(parameter.grad is None for parameter in policy.actor.parameters())
        assert torch.equal(rng, torch.get_rng_state())
        assert not torch.cuda.is_initialized()
    finally:
        cache.close()
