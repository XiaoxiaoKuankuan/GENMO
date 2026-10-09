"""KL 检查位置的显式算法对照与旧采样数据不变性验收。

post_step_full 保留更新后的检查；pre_step_plus_final 只在当前 PPO 前向计算的
解析 KL 上判断是否丢弃尚未执行的更新，不能伪称验收了更新后策略。用真实 Adam
和小型高斯策略计数验证 BC、更新预算和 KL 前向调用，最终全量检查仍由训练事务
执行，不能使用旧参数版本或不存在的末次 KL 缓存。
"""
import copy
import pytest
import torch

from gem.closedloop.dppo.updater_v2 import actor_update_v2, analytic_kl_local
from tests.closedloop.dppo.test_updater_v2 import BatchGaussianPolicy, rows, Anchor
from gem.closedloop.dppo.trainer import trainable_actor_parameters


@pytest.mark.parametrize('mode', ['post_step_full', 'pre_step_plus_final'])
def test_kl_mode_call_count_full_final_and_fixed_old(mode):
    policy = BatchGaussianPolicy()
    samples = rows(policy)
    original = [r.old_log_prob.clone() for r in samples]
    optimizer = torch.optim.Adam(trainable_actor_parameters(policy.actor), lr=.008)
    cache = {}
    report = actor_update_v2(policy, optimizer, samples, {'advantages':torch.tensor([1., -.3, .4, -.7])},
        actor_minibatch_internal_transitions=4, denoising_microbatch=3,
        soft_kl_limit=None, kl_check_mode=mode, kl_cache_sink=cache)
    assert report['optimizer_steps'] == 4
    assert report['hard_kl_pending']
    assert all(torch.equal(r.old_log_prob, p) for r, p in zip(samples, original))
    assert sum(r['post_kl_extra_internal_forwards'] for r in report['steps']) == (16 if mode == 'post_step_full' else 0)
    if mode == 'pre_step_plus_final':
        assert not cache
        assert all(r['post_minibatch_kl'] is None for r in report['steps'])
    final = analytic_kl_local(policy, samples, reuse_cache=cache.get('cache'))
    full = analytic_kl_local(policy, samples)
    assert final['mean_joint_kl'] == full['mean_joint_kl']
    assert final['fresh_internal_forwards'] == (4 if mode == 'post_step_full' else 8)


def test_pre_check_discards_step_before_bc_adam_and_budget():
    policy = BatchGaussianPolicy()
    samples = rows(policy)
    anchor, attempts, cache = Anchor(), [], {'cache':'must_be_discarded'}
    optimizer = torch.optim.Adam(trainable_actor_parameters(policy.actor), lr=.03)
    report = actor_update_v2(policy, optimizer, samples, {'advantages':torch.ones(4)},
        actor_minibatch_internal_transitions=4, denoising_microbatch=3,
        epoch_orders=[[0,1,2,3], [0,1,2,3]], soft_kl_limit=1e-15,
        kl_check_mode='pre_step_plus_final', kl_cache_sink=cache,
        bc=anchor, reserve_attempt=lambda: attempts.append(1))
    assert report['optimizer_steps'] == anchor.calls == len(attempts) == 1
    assert report['early_stop_reason'] == 'pre_minibatch_soft_kl'
    assert not report['discarded_pending_minibatch']['optimizer_step_executed']
    assert all(p.grad is None for p in policy.actor.parameters())
    assert all(float(state['step']) == 1 for state in optimizer.state.values())
    assert not cache


def test_post_mode_respects_current_pre_kl_even_when_previous_batch_passed(monkeypatch):
    """不同批分布使上一批通过、当前批已超限时，不能继续调用 BC/Adam。"""
    from gem.closedloop.dppo import updater_v2
    policy = BatchGaussianPolicy()
    samples = rows(policy)
    original = [r.old_log_prob.clone() for r in samples]
    anchor, attempts = Anchor(), []
    optimizer = torch.optim.Adam(trainable_actor_parameters(policy.actor), lr=.03)
    actual_kl = updater_v2.analytic_kl_local
    post_calls = []
    def previous_batch_below_soft(*args, **kwargs):
        report = actual_kl(*args, **kwargs)
        post_calls.append(report)
        return dict(report, mean_joint_kl=0.)
    monkeypatch.setattr(updater_v2, 'analytic_kl_local', previous_batch_below_soft)
    report = actor_update_v2(policy, optimizer, samples, {'advantages': torch.ones(4)},
        actor_minibatch_internal_transitions=4, denoising_microbatch=3,
        epoch_orders=[[0,1,2,3], [0,1,2,3]], soft_kl_limit=1e-15,
        kl_check_mode='post_step_full', bc=anchor, reserve_attempt=lambda: attempts.append(1))
    assert report['optimizer_steps'] == anchor.calls == len(attempts) == len(post_calls) == 1
    assert report['early_stop_reason'] == 'pre_minibatch_soft_kl'
    assert report['discarded_pending_minibatch']['mean_joint_kl'] > 1e-15
    assert all(float(state['step']) == 1 for state in optimizer.state.values())
    assert all(p.grad is None for p in policy.actor.parameters())
    assert all(torch.equal(row.old_log_prob, old) for row, old in zip(samples, original))
