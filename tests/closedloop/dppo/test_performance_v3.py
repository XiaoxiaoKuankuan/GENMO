"""第二阶段性能改造的计时和数值探针验收。

使用 CPU 小型真实 Actor 与可控计时替身，验证记录器不改变随机数或梯度、嵌套区间
不会被伪装为可加总关键路径、候选自身误差与旧标量路径误差分开保存。探针必须覆盖
全部去噪步骤，并在结束后恢复模型模式和已有梯度。临时产物使用外部 pytest basetemp，
本文件不连接服务器、不启动 GMT，也不读取或覆盖正式训练目录。
"""
import copy
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo.execution_profile import probe_profiles
from gem.closedloop.dppo.performance import PhaseProfiler, activate, deactivate, measure
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.tensor_cache import ConditionGraphCache, RolloutTensorCache
from gem.closedloop.dppo.updater_v2 import _parameters, probability_check_local, analytic_kl_local
from gem.closedloop.dppo.updater_v2 import balanced_epoch_order
from tests.closedloop.test_stage1_actor import _activate_branches, _conditions
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory


def test_performance_counts_and_preserves_rng_and_gradients():
    rng = torch.get_rng_state().clone()
    parameter = torch.tensor(2., requires_grad=True)
    profiler = PhaseProfiler()
    token = activate(profiler)
    try:
        with measure('outer'):
            for _ in range(3):
                with measure('inner', gpu=True):
                    (parameter.square()/3).backward()
    finally:
        deactivate(token)
    report = profiler.report()
    assert report['stages']['outer']['calls'] == 1
    assert report['stages']['inner']['calls'] == 3
    assert report['intervals'] == 'inclusive_nested_do_not_sum'
    assert report['stages']['inner']['cuda_seconds'] == 0
    torch.testing.assert_close(parameter.grad, torch.tensor(4.))
    assert torch.equal(rng, torch.get_rng_state())


def test_profile_reports_self_and_cross_all_twenty_steps(actor_factory):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actor, batch = actor_factory(starts=(45,))
        _activate_branches(actor)
        actor.train()
        parameter = next(p for p in actor.parameters() if p.requires_grad)
        parameter.grad = torch.ones_like(parameter)
        saved = parameter.grad
        policy = DPPODiffusionPolicy(actor, steps=20)
        actor.train()
        reports = probe_profiles(policy, _conditions(batch), maximum_microbatch=1)
        assert len(reports) == 2
        for row in reports:
            assert row['backward_denoising_steps'] == 20
            assert len(row['self_consistency']['per_step_logprob_error']) == 20
            assert len(row['cross_execution']['layer_max_abs_error']['mean']) == 20
            assert row['independent_gaussian_error'] <= 1e-8
            assert row['finite_gradients']
        assert actor.training
        assert parameter.grad is saved
        assert not policy.cfg_batch
    finally:
        torch.set_num_threads(previous_threads)


def test_tensor_cache_full_and_explicit_blocks_match_original_and_reject_foreign_rows():
    from tests.closedloop.dppo.test_updater_v2 import BatchGaussianPolicy, rows
    policy = BatchGaussianPolicy()
    samples = rows(policy, 4)
    targets = dict(advantages=torch.arange(4, dtype=torch.float64), returns=torch.arange(4).float())
    full = RolloutTensorCache(samples, targets, 'cpu')
    blocked = RolloutTensorCache(samples, targets, 'cpu', max_device_bytes=full.bytes//2)
    try:
        assert blocked.mode == 'pinned_blocks'
        for cache in (full, blocked):
            selected, steps = [samples[3], samples[0], samples[3]], [0, 1, 1]
            torch.testing.assert_close(cache.get('chain', selected, steps),
                torch.stack([row.chain[step] for row, step in zip(selected, steps)]), rtol=0, atol=0)
            assert probability_check_local(policy, samples, tensor_cache=cache)['passed']
            expected = analytic_kl_local(policy, samples)
            actual = analytic_kl_local(policy, samples, tensor_cache=cache)
            assert actual == expected
            with pytest.raises(ValueError, match='different rollout'):
                cache.get('chain', [copy.copy(samples[0])], [0])
    finally:
        full.close()
        blocked.close()
    with pytest.raises(RuntimeError, match='released'):
        full.context([samples[0]])


def test_fixed_shape_and_condition_gradient_cache_match_every_trainable_module(actor_factory):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actor, batch = actor_factory(starts=(0, 45))
        _activate_branches(actor)
        expected_actor = copy.deepcopy(actor)
        conditions = _conditions(batch)
        policy = DPPODiffusionPolicy(actor, steps=20, cfg_batch=True, execution_batch_size=4)
        reference = DPPODiffusionPolicy(expected_actor, steps=20, cfg_batch=True, execution_batch_size=4)
        samples = []
        for i in range(2):
            context = {key: value[i:i+1] for key, value in conditions.items()}
            trace = policy.sample_rollout(context, generator=torch.Generator().manual_seed(40+i))
            samples.append(SimpleNamespace(context=context, chain=trace['chain'][0], old_log_prob=trace['old_log_probs'][0],
                free_mask=trace['free_mask'][0], transition_valid=True,
                metadata=dict(sampler_trace=trace, remaining_music_seconds=10.)))
        cache = RolloutTensorCache(samples, dict(advantages=torch.tensor([1., -.3])), 'cpu')
        assert probability_check_local(policy, samples, denoising_microbatch=4, tensor_cache=cache)['passed']
        weights = torch.linspace(-.3, .7, 120*30).reshape(1, 120, 30)
        for row in samples:
            for step in range(20):
                result, _ = _parameters(reference, [row], [step], 'cpu')
                (result['mean']*weights/40).sum().backward()
        graph = ConditionGraphCache(policy, cache)
        for row in samples:
            for start in range(0, 20, 4):
                result, _ = _parameters(policy, [row]*4, list(range(start, start+4)), 'cpu', cache, graph)
                (result['mean']*weights/40).sum().backward()
        graph.backward()
        modules = set()
        for (name, actual), (_, expected) in zip(actor.named_parameters(), expected_actor.named_parameters()):
            if expected.grad is None:
                assert actual.grad is None, name
            else:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-4, atol=2e-5, msg=name)
                if actual.grad.abs().sum() > 0:
                    modules.add(name.split('.')[0])
        assert {'history_encoder', 'prefix_encoder', 'music_embedder'} <= modules
        for model in (actor, expected_actor):
            torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-9, weight_decay=0.).step()
        for actual, expected in zip(actor.parameters(), expected_actor.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        with pytest.raises(ValueError, match='optimizer step|completed'):
            graph.prepare([samples[0]], cache.context([samples[0]]))
        cache.close()
    finally:
        torch.set_num_threads(previous_threads)


def test_fixed_shape_probe_keeps_strict_gate_and_restores_temporary_optimizer(actor_factory):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actor, batch = actor_factory(starts=(45,))
        _activate_branches(actor)
        before = copy.deepcopy(actor.state_dict())
        policy = DPPODiffusionPolicy(actor, steps=20)
        reports = probe_profiles(policy, _conditions(batch), maximum_microbatch=4,
                                 fixed_execution_shape=4, optimizer_probe=True)
        selected = next(row for row in reports if row['microbatch'] == 4 and row['cfg_batch'])
        assert selected['passed'], selected
        assert selected['max_logprob_error'] <= 1e-4
        assert selected['max_ratio_error'] <= 1e-3
        assert selected['optimizer_probe']['learning_rate'] == 5e-9
        assert selected['optimizer_probe']['finite']
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert policy.execution_batch_size is None
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('counts', [[20]*8, [0, 1, 4, 20, 2, 0, 3, 1]])
def test_balanced_epoch_keeps_all_data_and_rng_restore(counts):
    manifest = [dict(owner_rank=rank, local_index=i) for rank, count in enumerate(counts) for i in range(count)]
    generator = torch.Generator().manual_seed(93)
    state = generator.get_state()
    order = balanced_epoch_order(list(range(len(manifest))), manifest, generator)
    assert sorted(order) == list(range(len(manifest)))
    generator.set_state(state)
    assert balanced_epoch_order(list(range(len(manifest))), manifest, generator) == order
    if counts == [20]*8:
        for start in (0, 80):
            assert [sum(manifest[i]['owner_rank'] == rank for i in order[start:start+80]) for rank in range(8)] == [10]*8


def test_final_kl_reuses_only_unchanged_version_and_matches_authoritative():
    from tests.closedloop.dppo.test_updater_v2 import BatchGaussianPolicy, rows
    policy = BatchGaussianPolicy()
    samples = rows(policy, 4)
    with torch.no_grad():
        policy.actor.denoiser.weight.add_(.01)
    _, cache = analytic_kl_local(policy, samples, global_indices=[1, 3], return_cache=True)
    calls = []
    original = policy.transition_parameters
    def counted(context, state, steps):
        calls.append(len(state))
        return original(context, state, steps)
    policy.transition_parameters = counted
    actual = analytic_kl_local(policy, samples, reuse_cache=cache)
    assert sum(calls) == 4 and actual['reused_upper_transitions'] == 2
    expected = analytic_kl_local(policy, samples)
    for name in expected:
        if name not in ('reused_upper_transitions', 'fresh_internal_forwards'):
            assert actual[name] == expected[name]
    with torch.no_grad():
        policy.actor.denoiser.weight.add_(.001)
    with pytest.raises(ValueError, match='KL cache'):
        analytic_kl_local(policy, samples, reuse_cache=cache)
    _, fresh = analytic_kl_local(policy, samples, return_cache=True)
    samples[0].chain.add_(.1)
    with pytest.raises(ValueError, match='KL cache'):
        analytic_kl_local(policy, samples, reuse_cache=fresh)


def test_batched_bc_samplewise_loss_and_independent_rng_restore(actor_factory):
    from tests.closedloop.dppo.test_trainer import _anchor
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actor, batch = actor_factory(starts=(45,))
        anchor = _anchor(actor, batch)
        legacy = copy.deepcopy(anchor.state_dict())
        anchor.batch_forward = True
        with pytest.raises(ValueError, match='execution/RNG'):
            anchor.validate_state_dict(legacy)
        saved = copy.deepcopy(anchor.state_dict())
        global_rng = torch.get_rng_state().clone()
        first = anchor.backward(actor, .1)
        grads = {name: p.grad.clone() for name, p in actor.named_parameters() if p.grad is not None}
        assert first['batch_size'] == 2 and first['bc_update_steps'] == 4
        assert len(first['weighted_gradient_norm_per_microbatch']) == 1
        assert all(row['auxiliary_warmup_factor'] == pytest.approx(.3) for row in first['warmup_factors'])
        assert any('static_conf' in name and value.abs().sum() > 0 for name, value in grads.items())
        assert torch.equal(torch.get_rng_state(), global_rng)
        anchor.load_state_dict(saved)
        actor.zero_grad(set_to_none=True)
        second = anchor.backward(actor, .1)
        assert first == second
        for name, p in actor.named_parameters():
            if name in grads:
                torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


def test_full_learning_probe_restores_weights_and_reports_all_encoder_gradients(actor_factory):
    from gem.closedloop.dppo.execution_profile import compare_learning_execution
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actor, batch = actor_factory(starts=(45,))
        _activate_branches(actor)
        policy = DPPODiffusionPolicy(actor, steps=20, cfg_batch=True, execution_batch_size=4)
        saved = copy.deepcopy(actor.state_dict())
        report = compare_learning_execution(policy, _conditions(batch))
        assert report['max_logprob_difference'] <= 1e-4
        assert report['max_parameter_difference'] < 1e-7
        assert report['max_optimizer_state_difference'] < 1e-6
        assert any('history' in name for name in report['modules'])
        assert len(report['modules']) > 3
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(old_threads)
