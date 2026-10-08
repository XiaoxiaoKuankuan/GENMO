"""第二阶段条件缓存、CFG合批和逐步方差日程的CPU数值/梯度验收。

本文件复用现有小型真实Stage1Actor夹具，不复制网络、不读取正式checkpoint，
也不启动GPU、Isaac或长训练。核对单链条件只编码一次而随机链与旧逐步路径一致；
训练prepared条件保持历史/前缀/音乐梯度，且输入或Actor原地变化使旧缓存失效；
CFG仅沿batch拼接，保持逐坐标已知前缀及padding，并用独立Normal密度检查概率；
显式std地板日程只改变随机标准差，不改变DDIM均值，其日程和执行方式进入核身份。
临时数据只写pytest指定的独立basetemp，由测试调用者在结束后精确删除。
"""
import copy

import pytest
import torch

from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from tests.closedloop.test_stage1_actor import _activate_branches, _conditions
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_single_chain_encoding_and_scalar_reference_are_identical(actor_factory, monkeypatch):
    actor, batch = actor_factory(starts=(45,))
    _activate_branches(actor)
    conditions = _conditions(batch)
    calls = []
    original = actor.adapt_conditions
    def observe(value):
        calls.append(1)
        return original(value)
    monkeypatch.setattr(actor, 'adapt_conditions', observe)
    policy = DPPODiffusionPolicy(actor, steps=3)
    trace = policy.sample_rollout(conditions, generator=torch.Generator().manual_seed(41))
    assert len(calls) == 1
    original_transition = policy.transition_parameters
    monkeypatch.setattr(policy, 'transition_parameters',
        lambda context, state, step, **kwargs: original_transition(context, state, step))
    reference = policy.sample_rollout(conditions, generator=torch.Generator().manual_seed(41))
    for name in ('chain', 'old_means', 'old_stds', 'old_log_probs', 'contact'):
        torch.testing.assert_close(trace[name], reference[name], rtol=0, atol=0)


def test_prepared_gradient_matches_uncached_and_rejects_stale_cache(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    _activate_branches(actor)
    conditions = _conditions(batch)
    state = torch.randn_like(conditions['known_qpos30'])
    following = torch.randn_like(state)
    policy = DPPODiffusionPolicy(actor, steps=3)
    reference = copy.deepcopy(actor)
    prepared = policy.prepare_conditions(conditions)
    assert prepared['conditional'].requires_grad
    (-policy.evaluate_log_probs(conditions, state, following, 1, prepared=prepared).sum()).backward()
    other = DPPODiffusionPolicy(reference, steps=3)
    (-other.evaluate_log_probs(conditions, state, following, 1).sum()).backward()
    for (name, parameter), (_, expected) in zip(actor.named_parameters(), reference.named_parameters()):
        if expected.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected.grad, atol=1e-5, rtol=1e-5)
    for prefix in ('history_encoder', 'prefix_encoder', 'music_embedder'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for name, p in actor.named_parameters() if name.startswith(prefix))
    fresh = policy.prepare_conditions(conditions)
    with torch.no_grad():
        next(actor.parameters()).add_(1e-5)
    with pytest.raises(ValueError, match='Actor version'):
        policy.transition_parameters(conditions, state, 0, prepared=fresh)
    with torch.no_grad():
        sampled_cache = policy.prepare_conditions(conditions)
    with pytest.raises(ValueError, match='grad mode'):
        policy.transition_parameters(conditions, state, 0, prepared=sampled_cache)
    fresh = policy.prepare_conditions(conditions)
    conditions['music_features'].add_(.01)
    with pytest.raises(ValueError, match='inputs'):
        policy.transition_parameters(conditions, state, 0, prepared=fresh)


def test_cfg_batch_replay_has_independent_density_and_shared_masks(actor_factory):
    actor, batch = actor_factory(starts=(45,), prefix=6)
    _activate_branches(actor)
    conditions = _conditions(batch)
    conditions['future_valid'][:, 100:] = False
    conditions['music_valid'][:, 100:] = False
    policy = DPPODiffusionPolicy(actor, steps=3, cfg_batch=True)
    trace = policy.sample_rollout(conditions, generator=torch.Generator().manual_seed(71))
    assert policy.kernel_config['cfg_forward'] == 'batched'
    for step in range(policy.steps):
        current = policy.evaluate_log_probs(conditions, trace['chain'][:, step], trace['chain'][:, step+1], step)
        torch.testing.assert_close(current, trace['old_log_probs'][:, step], atol=1e-4, rtol=0)
        independent = torch.distributions.Normal(trace['old_means'][:, step].double(),
            trace['old_stds'][:, step].double()).log_prob(trace['chain'][:, step+1].double())
        independent = independent.masked_fill(~trace['free_mask'], 0).sum((-2, -1))
        torch.testing.assert_close(current, independent, atol=1e-8, rtol=0)
    state = trace['chain'][:, 0]
    fused = policy.transition_parameters(conditions, state, 0)
    policy.cfg_batch = False
    separate = policy.transition_parameters(conditions, state, 0)
    torch.testing.assert_close(fused['mean'], separate['mean'], atol=2e-6, rtol=2e-5)
    assert torch.equal(fused['free_mask'], separate['free_mask'])
    assert torch.count_nonzero(trace['chain'][..., 100:, :]) == 0


def test_std_schedule_changes_noise_not_mean_and_is_identified(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    conditions = _conditions(batch)
    state = torch.randn_like(conditions['known_qpos30'])
    ordinary = DPPODiffusionPolicy(actor, steps=3)
    scheduled = DPPODiffusionPolicy(actor, steps=3, std_schedule=[.2, .1, .01])
    for step in range(3):
        first = ordinary.transition_parameters(conditions, state, step)
        second = scheduled.transition_parameters(conditions, state, step)
        torch.testing.assert_close(first['mean'], second['mean'], atol=0, rtol=0)
        torch.testing.assert_close(first['base_std'], second['base_std'], atol=0, rtol=0)
        assert second['std'].item() >= [.2, .1, .01][step]-1e-8
    assert ordinary.kernel_config['version'].endswith('.v1')
    assert scheduled.kernel_config['effective_std_floors'] == [.2, .1, .01]
    assert scheduled.kernel_config['version'].endswith('.v2')
    with pytest.raises(ValueError, match='std_schedule'):
        DPPODiffusionPolicy(actor, steps=3, std_schedule=[.001, 0., .01])


def test_cfg_batch_backward_preserves_all_trainable_condition_branches(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    _activate_branches(actor)
    reference = copy.deepcopy(actor)
    conditions = _conditions(batch)
    state = torch.randn_like(conditions['known_qpos30'])
    weight = torch.randn_like(state)
    for model, fused in ((actor, True), (reference, False)):
        policy = DPPODiffusionPolicy(model, steps=3, cfg_batch=fused)
        prepared = policy.prepare_conditions(conditions)
        output = policy.transition_parameters(conditions, state, 1, prepared=prepared)
        (output['mean'] * weight).sum().backward()
    for (name, actual), (_, expected) in zip(actor.named_parameters(), reference.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=3e-5)
