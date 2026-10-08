"""多 minibatch PPO、本地分片和规范位置表修复的确定性 CPU 验收。

使用可批处理的小型高斯策略保留真实自由坐标联合概率、两步旧采样链和固定优势，
对照一次完整 minibatch 的标量梯度与 microbatch 累计，检查后续更新真实重算概率、
BC 每 step 的全局频率、KL 软停止以及完整链指标。两个真实 Gloo 进程覆盖不均匀分片、
空卡、仅监督头与未使用参数，使用 SGD 避免 Adam 首步掩盖错误的归一化系数。

另外验证缓存价值版本和全局 GAE 统计、位置表所有别名的规范修复。测试仅在 pytest
指定临时目录生成通信/结果文件，不读取正式训练产物、不启动 GPU/GMT/长期训练。
"""
from __future__ import annotations

import copy
import random
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.policy import masked_joint_log_prob
from gem.closedloop.dppo.position_repair import audit_position_tables, canonical_position_tables
from gem.closedloop.dppo.returns import normalize_advantages_global
from gem.closedloop.dppo.trainer import (actor_update_v2, analytic_kl_local, critic_update_local,
    fixed_targets, populate_values, probability_check_local, trainable_actor_parameters)
from tests.closedloop.dppo.test_distributed_training import GaussianActor, SmallCritic, _assert_nested_close
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory


class BatchGaussianPolicy:
    steps = 2
    timestep_map = (999, 0)
    kernel_config = {'test': 'v2_batch_gaussian'}

    def __init__(self):
        self.actor = GaussianActor()

    def transition_parameters(self, context, state, step):
        batch = state.shape[0]
        indices = torch.as_tensor(step, device=state.device).expand(batch)
        mean = self.actor.denoiser(context['feature']).reshape(batch, 1, 1) * (indices[:, None, None] + 1)
        return dict(mean=mean.expand_as(state), std=mean.new_full((batch, 1, 1), .7),
                    free_mask=context['future_valid'][..., None] & ~context['known_qpos30_mask'])


class Anchor:
    def __init__(self):
        self.calls = 0

    def backward(self, actor, weight):
        self.calls += 1
        loss = weight * (actor.bc_only.square() + .3 * actor.denoiser.weight.square().mean())
        loss.backward()
        return dict(batch_size=2, calls=self.calls, loss=float(loss.detach()))


def rows(policy, count=4):
    generator = torch.Generator().manual_seed(195)
    result = []
    for index in range(count):
        known = torch.ones((1, 120, 30), dtype=torch.bool)
        known[:, 0, :index % 3 + 1] = False
        context = dict(feature=torch.tensor([[.5 + index * .3]]),
                       future_valid=torch.ones((1, 120), dtype=torch.bool), known_qpos30_mask=known)
        chain = torch.zeros((3, 120, 30))
        means, stds, logp = [], [], []
        with torch.no_grad():
            for step in range(2):
                parameters = policy.transition_parameters(context, chain[step:step + 1], step)
                chain[step + 1] = parameters['mean'][0] + .7 * torch.randn((120, 30), generator=generator)
                means.append(parameters['mean'].clone())
                stds.append(parameters['std'].clone())
                logp.append(masked_joint_log_prob(chain[step + 1:step + 2], parameters['mean'], parameters['std'], ~known)[0])
        result.append(SimpleNamespace(context=context, chain=chain, free_mask=~known[0],
            old_log_prob=torch.stack(logp), transition_valid=True,
            metadata=dict(remaining_music_seconds=float(index + 1), sampler_trace=dict(kernel_config=policy.kernel_config,
                timestep_map=torch.tensor(policy.timestep_map), old_means=torch.stack(means, 1), old_stds=torch.stack(stds, 1)))))
    return result


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def run_actor(*, micro=3, distributed=None, count=4, bc=True, soft=None, reduction='joint_sum'):
    policy = BatchGaussianPolicy()
    all_rows = rows(policy, count)
    old_probabilities = torch.stack([item.old_log_prob.clone() for item in all_rows])
    values = torch.tensor([1., -.3, .4, -.7][:count], dtype=torch.float64)
    if distributed is None:
        local, targets, manifest = all_rows, {'advantages': values}, None
    else:
        # 3+1 或 1+0，覆盖非均匀分片和无本地样本但仍需参加 collective 的 rank。
        owner_ids = [[i for i in range(count) if i < 3], [i for i in range(count) if i >= 3]]
        manifest = [dict(owner_rank=rank, local_index=index, valid=True, has_free=True)
                    for rank, ids in enumerate(owner_ids) for index in range(len(ids))]
        own = owner_ids[distributed.rank]
        local = [all_rows[i] for i in own]
        targets = {'advantages': values[own]}
    optimizer = torch.optim.SGD(trainable_actor_parameters(policy.actor), lr=.008, momentum=.7)
    anchor = Anchor() if bc and (distributed is None or distributed.rank == 0) else None
    report = actor_update_v2(policy, optimizer, local, targets, global_manifest=manifest, distributed=distributed,
        ppo_epochs=2, actor_minibatch_internal_transitions=4, denoising_microbatch=micro,
        max_optimizer_steps=4, generator=torch.Generator().manual_seed(19), bc=anchor, bc_weight=.17,
        clip=.001, gamma_denoising=.9, grad_clip_norm=2., soft_kl_limit=soft,
        verify_initial_probability=True, objective_logprob_reduction=reduction)
    return dict(state=copy.deepcopy(policy.actor.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()), report=report,
        kl=analytic_kl_local(policy, local, global_manifest=manifest, distributed=distributed, denoising_microbatch=micro),
        old_probabilities=old_probabilities, after_probabilities=torch.stack([item.old_log_prob for item in all_rows]),
        bc_calls=0 if anchor is None else anchor.calls)


def test_real_multistep_reuses_fixed_old_statistics_and_bc_each_step():
    result = run_actor()
    report = result['report']
    assert report['optimizer_steps'] == 4
    assert result['bc_calls'] == 4 and report['bc_global_samples'] == 8
    assert report['hard_kl_pending'] and report['old_statistics_fixed']
    assert report['steps'][0]['mean_ratio'] == pytest.approx(1., abs=1e-10)
    assert any(abs(step['mean_ratio'] - 1.) > 1e-4 for step in report['steps'][1:])
    assert any(step['clip_fraction'] > 0 for step in report['steps'][1:])
    torch.testing.assert_close(result['old_probabilities'], result['after_probabilities'], rtol=0, atol=0)
    for order in report['epoch_orders']:
        assert sorted(order) == [0, 1, 2, 3]
    assert result['kl']['mean_chain_joint_kl'] == pytest.approx(2 * result['kl']['mean_joint_kl'])
    assert result['kl']['max_chain_joint_kl'] >= result['kl']['mean_chain_joint_kl']
    contribution = report['steps'][0]['gradient_contributions']
    assert contribution['shared']['cosine'] is not None
    assert contribution['bc_only']['ppo_norm'] == 0


@pytest.mark.parametrize('micro', [1, 2, 3, 7])
def test_microbatch_changes_memory_not_optimizer_normalization(micro):
    expected = run_actor(micro=4)
    actual = run_actor(micro=micro)
    _assert_nested_close(actual['state'], expected['state'])
    _assert_nested_close(actual['optimizer'], expected['optimizer'])
    _assert_nested_close(actual['kl'], expected['kl'])


def test_soft_stop_is_minibatch_diagnostic_and_hard_gate_still_pending():
    result = run_actor(soft=1e-12)
    assert result['report']['optimizer_steps'] == 1
    assert result['report']['early_stop_reason'] == 'minibatch_soft_kl'
    assert result['report']['hard_kl_pending']
    assert result['report']['steps'][0]['post_kl_extra_internal_forwards'] == 4


def test_alternative_coordinate_objective_does_not_rewrite_joint_probability():
    result = run_actor(reduction='free_coordinate_mean')
    assert result['report']['objective_logprob_reduction'] == 'free_coordinate_mean'
    torch.testing.assert_close(result['old_probabilities'], result['after_probabilities'], atol=0, rtol=0)
    assert result['kl']['joint_kl_scope'] == 'sum_free_coordinates_per_internal_transition_then_mean'


def run_critic(distributed=None):
    critic = SmallCritic()
    all_rows = rows(BatchGaussianPolicy())
    target = torch.tensor([1., -.2, .7, 1.3])
    if distributed is None:
        local, local_target, manifest = all_rows, target, None
    else:
        ids = [0, 1, 2] if distributed.rank == 0 else [3]
        local, local_target = [all_rows[i] for i in ids], target[ids]
        manifest = [dict(owner_rank=0 if i < 3 else 1, local_index=i if i < 3 else 0,
                         valid=True, has_free=True) for i in range(4)]
    optimizer = torch.optim.SGD(critic.parameters(), lr=.03, momentum=.7)
    report = critic_update_local(critic, optimizer, local, {'returns': local_target}, global_manifest=manifest,
        distributed=distributed, steps=3, batch_size=2, generator=torch.Generator().manual_seed(913))
    return dict(state=critic.state_dict(), optimizer=optimizer.state_dict(), report=report)


def _worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory / 'rendezvous').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        distributed = DistributedCollectives(rank, 2, device='cpu')
        raw = [torch.tensor([1., 3., 5.]), torch.tensor([7.])][rank]
        target = normalize_advantages_global({'advantages_raw': raw, 'valid': torch.ones_like(raw, dtype=torch.bool)},
                                              distributed=distributed)
        result = dict(actor=run_actor(distributed=distributed), empty=run_actor(distributed=distributed, count=1),
                      critic=run_critic(distributed), normalized=target)
        torch.save(result, directory / f'rank_{rank}.pt')
        dist.barrier()  # 两份结果均完成后再销毁Gloo，避免一方退出时另一方仍在保存。
    finally:
        dist.destroy_process_group()


def _spawn_bounded(worker, directory):
    context = mp.spawn(worker, args=(str(directory),), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('Bounded Gloo test did not exit within 60 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


def test_two_rank_local_shards_match_single_global_multistep(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo is unavailable')
    _spawn_bounded(_worker, tmp_path)
    ranks = [torch.load(tmp_path / f'rank_{rank}.pt', weights_only=False) for rank in range(2)]
    expected = dict(actor=run_actor(), empty=run_actor(count=1), critic=run_critic())
    for rank, output in enumerate(ranks):
        for name in expected:
            for field in ('state', 'optimizer'):
                _assert_nested_close(output[name][field], expected[name][field])
            if name != 'critic':
                _assert_nested_close(output[name]['kl'], expected[name]['kl'])
                assert output[name]['bc_calls'] == (expected[name]['bc_calls'] if rank == 0 else 0)
                for actual, reference in zip(output[name]['report']['steps'], expected[name]['report']['steps']):
                    for field in ('ppo_loss', 'ppo_only_gradient_norm', 'total_gradient_norm', 'gradient_contributions'):
                        _assert_nested_close(actual[field], reference[field])
            else:
                _assert_nested_close(output[name]['report'], expected[name]['report'])
    combined = torch.cat([entry['normalized']['advantages'] for entry in ranks])
    reference = torch.tensor([1., 3., 5., 7.], dtype=torch.float64)
    torch.testing.assert_close(combined, (reference - reference.mean()) / reference.std(unbiased=False))


def test_value_reuse_requires_exact_version_and_never_recomputes():
    critic = SmallCritic()
    row = rows(BatchGaussianPolicy(), 1)[0]
    row.next_context = None
    row.terminated, row.truncated = True, False
    row.rewards = [1., 2.]
    row.executed_control_steps = 2
    row.identity = dict(backend_session_id='a', episode_id=1, policy_version=0)
    populate_values([row], critic, 'cpu', critic_version='critic:7')
    original = row.old_value
    with torch.no_grad():
        critic.value_head.bias.add_(10)
    result = fixed_targets([row], critic, 'cpu', reuse_values=True, critic_version='critic:7', normalize=False)
    assert row.old_value == original
    assert not result['advantages_normalized']
    with pytest.raises(ValueError, match='version'):
        fixed_targets([row], critic, 'cpu', reuse_values=True, critic_version='critic:8')


def test_canonical_fixed_table_repair_restores_all_aliases_and_frozen_layout():
    from gem.network.base_arch.embeddings.pe import PositionalEncoding
    actor = torch.nn.Module()
    actor.sequence_pos_encoder = PositionalEncoding(8, max_len=12)
    actor.embed_timestep = torch.nn.Module()
    actor.embed_timestep.sequence_pos_encoder = actor.sequence_pos_encoder
    actor.learned = torch.nn.Parameter(torch.ones(2))
    canonical = canonical_position_tables(actor)
    actor.requires_grad_(True)
    with torch.no_grad():
        actor.sequence_pos_encoder.pe[2, 0, 3] += .01
    audit = audit_position_tables(actor, canonical=canonical)
    assert audit['modified_before_repair'] and audit['table_aliases_count'] == 2
    assert all(entry['changed_values'] == 1 for entry in audit['tables'])
    audit_position_tables(actor, canonical=canonical, repair=True)
    assert not audit_position_tables(actor)['modified_before_repair']
    assert trainable_actor_parameters(actor) == [actor.learned]


def test_initial_probability_gate_rejects_bad_old_logprob():
    policy = BatchGaussianPolicy()
    sample = rows(policy, 1)
    sample[0].old_log_prob[0] += .01
    with pytest.raises(FloatingPointError, match='probability mismatch'):
        probability_check_local(policy, sample, denoising_microbatch=2)


def test_probability_sentinel_uses_global_indices_without_truncating_manifest():
    policy = BatchGaussianPolicy()
    samples = rows(policy, 4)
    samples[3].old_log_prob[0] += .01
    manifest = [dict(owner_rank=0, local_index=i, valid=True, has_free=True) for i in range(4)]
    assert probability_check_local(policy, samples, global_manifest=manifest, global_indices=[0, 1])['passed']
    with pytest.raises(FloatingPointError):
        probability_check_local(policy, samples, global_manifest=manifest)


def test_manifest_must_cover_local_shard_and_optimizer_excludes_frozen_parameters():
    policy = BatchGaussianPolicy()
    samples = rows(policy, 2)
    manifest = [dict(owner_rank=0, local_index=0, valid=True, has_free=True)]
    with pytest.raises(ValueError, match='complete local shard'):
        analytic_kl_local(policy, samples, global_manifest=manifest)
    policy.actor.always_unused.requires_grad_(False)
    optimizer = torch.optim.SGD(policy.actor.parameters(), lr=.001)
    with pytest.raises(ValueError, match='exclude frozen'):
        actor_update_v2(policy, optimizer, samples, {'advantages': torch.tensor([1., -1.])})


def test_real_actor_prepared_conditions_support_multiple_backward_steps(actor_factory):
    from gem.closedloop.dppo.policy import DPPODiffusionPolicy
    from tests.closedloop.test_stage1_actor import _conditions, _activate_branches
    from tests.closedloop.dppo.test_trainer import _row
    actor, batch = actor_factory(starts=(45,))
    _activate_branches(actor)
    policy = DPPODiffusionPolicy(actor, steps=3)
    samples = [_row(policy.sample_rollout(_conditions(batch), generator=torch.Generator().manual_seed(seed)))
               for seed in (19, 23)]
    baseline = copy.deepcopy(actor)
    fixed_before = {name: parameter.clone() for name, parameter in actor.named_parameters() if not parameter.requires_grad}
    contact_before = copy.deepcopy(actor.denoiser.static_conf_head.state_dict())
    results = []
    for micro in (1, 4):
        current = copy.deepcopy(baseline)
        current_policy = DPPODiffusionPolicy(current, steps=3)
        optimizer = torch.optim.SGD(trainable_actor_parameters(current), lr=2e-8)
        result = actor_update_v2(current_policy, optimizer, samples,
            {'advantages': torch.tensor([1., -1.])}, actor_minibatch_internal_transitions=3,
            denoising_microbatch=micro, epoch_orders=[[0, 1], [1, 0]], soft_kl_limit=None,
            gradient_diagnostics=False)
        assert result['optimizer_steps'] == 4
        for name, parameter in current.named_parameters():
            if name in fixed_before:
                assert parameter.grad is None
                torch.testing.assert_close(parameter, fixed_before[name], rtol=0, atol=0)
        for name, value in current.denoiser.static_conf_head.state_dict().items():
            torch.testing.assert_close(value, contact_before[name], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in current.denoiser.static_conf_head.parameters())
        results.append(current.state_dict())
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], atol=2e-7, rtol=2e-6)


def test_repair_exports_weights_only_and_preserves_source(actor_factory, tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from gem.closedloop.checkpoint import save_stage1_checkpoint
    from gem.closedloop.dppo.checkpoint import VERSION
    from gem.closedloop.dppo.position_repair import repair_stage2_weights, file_sha256
    actor, _ = actor_factory(starts=(45,))
    architecture = tmp_path / 'architecture.pt'
    save_stage1_checkpoint(actor, architecture, config={'test_only': True}, global_step=123)
    stats = Path(actor.endecoder.stats_path)
    kinematics = Path(actor.endecoder.kinematics.kinematics_path)
    mutated = copy.deepcopy(actor)
    with torch.no_grad():
        mutated.denoiser.sequence_pos_encoder.pe[2, 0, 3] += .01
        mutated.history_encoder.out_proj.bias.add_(.02)
    source = tmp_path / 'stage2_full.pt'
    torch.save(dict(version=VERSION, actor=mutated.state_dict(), actor_optimizer={'must_be_discarded': True},
        state={'iteration': 2897}, identity=dict(actor_interface=dict(actor.interface_config),
            assets=dict(checkpoint=file_sha256(architecture), stats=file_sha256(stats), kinematics=file_sha256(kinematics)))), source)
    # 保持测试统计量明确标记为 placeholder；只替换模型工厂，不伪造生产数据身份。
    monkeypatch.setattr('gem.closedloop.dppo.trainer.load_actor',
                        lambda _config: (copy.deepcopy(actor), OmegaConf.create({'test_only': True}), {}))
    initial_sha = file_sha256(source)
    audit = repair_stage2_weights(source, architecture, stats=stats, kinematics=kinematics)
    assert audit['position_tables']['modified_before_repair']
    assert not audit['position_tables']['repaired']
    output = tmp_path / 'repaired_actor.pt'
    report = repair_stage2_weights(source, architecture, stats=stats, kinematics=kinematics, output=output)
    assert report['position_tables']['repaired']
    assert file_sha256(source) == initial_sha
    result = torch.load(output, weights_only=False)
    assert 'actor_optimizer' not in result and 'critic_optimizer' not in result
    assert result['global_step'] == 0 and result['warm_start_report']['source_iteration'] == 2897
    for key, value in mutated.state_dict().items():
        expected = actor.state_dict()[key] if key.endswith('sequence_pos_encoder.pe') else value
        torch.testing.assert_close(result['state_dict'][key], expected, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        repair_stage2_weights(source, architecture, stats=stats, kinematics=kinematics, output=output)


def _failure_worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory / 'failure_rendezvous').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=20))
    try:
        distributed = DistributedCollectives(rank, 2, device='cpu')
        outcomes = {}
        for phase in ('forward', 'bc', 'optimizer'):
            policy = BatchGaussianPolicy()
            sample = rows(policy, 2)[rank:rank + 1]
            manifest = [dict(owner_rank=i, local_index=0, valid=True, has_free=True) for i in range(2)]
            anchor = Anchor() if rank == 0 else None
            optimizer = torch.optim.SGD(policy.actor.parameters(), lr=.002)
            def broken(*_args, **_kwargs):
                raise RuntimeError(f'injected_{phase}')
            if phase == 'forward' and rank == 1:
                policy.transition_parameters = broken
            elif phase == 'bc' and rank == 0:
                anchor.backward = broken
            elif phase == 'optimizer' and rank == 1:
                optimizer.step = broken
            try:
                actor_update_v2(policy, optimizer, sample, {'advantages': torch.tensor([1.])},
                    global_manifest=manifest, distributed=distributed, actor_minibatch_internal_transitions=4,
                    bc=anchor, soft_kl_limit=None, gradient_diagnostics=False)
            except RuntimeError as error:
                outcomes[phase] = str(error)
            else:
                raise AssertionError('Injected rank-local failure was not propagated')
            distributed.barrier()
        torch.save(outcomes, directory / f'failure_{rank}.pt')
        dist.barrier()  # 故障传播已结束；这里只同步成功的证据保存，不吞掉测试异常。
    finally:
        dist.destroy_process_group()


def test_rank_local_failures_propagate_before_following_collective(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo is unavailable')
    _spawn_bounded(_failure_worker, tmp_path)
    outputs = [torch.load(tmp_path / f'failure_{rank}.pt', weights_only=False) for rank in range(2)]
    assert outputs[0] == outputs[1]
    for phase, message in outputs[0].items():
        assert f'injected_{phase}' in message and 'Cooperative local phase failed' in message


def test_bc_state_validation_is_atomic_and_has_no_rng_or_dataset_side_effects():
    from gem.closedloop.dppo.trainer import SupervisedAnchor
    anchor = SupervisedAnchor.__new__(SupervisedAnchor)
    anchor.batch_size, anchor.bc_update_steps = 2, 3
    anchor.generator = torch.Generator().manual_seed(7)
    anchor.torch_rng = torch.Generator().manual_seed(8).get_state()
    anchor.numpy_rng = np.random.RandomState(9).get_state()
    anchor.python_rng = random.Random(10).getstate()
    anchor.cuda_rng, anchor.cuda_rng_device = None, None
    state = copy.deepcopy(anchor.state_dict())
    global_torch, global_numpy, global_python = torch.get_rng_state(), np.random.get_state(), random.getstate()
    assert anchor.validate_state_dict(state)
    corrupted = copy.deepcopy(state)
    corrupted['python_rng'] = ('invalid',)
    with pytest.raises((TypeError, ValueError)):
        anchor.load_state_dict(corrupted)
    torch.testing.assert_close(anchor.generator.get_state(), state['generator'], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_torch, atol=0, rtol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], global_numpy[1])
    assert random.getstate() == global_python and anchor.bc_update_steps == 3


def test_music_sampler_state_validation_never_copies_catalog(tmp_path, monkeypatch):
    from gem.closedloop.dppo.full_dataset import FullMusicCatalog, FullMusicSampler
    from tests.closedloop.dppo.test_full_dataset import make_catalog_data
    catalog = FullMusicCatalog(make_catalog_data(tmp_path))
    sampler = FullMusicSampler(catalog, seed=7)
    sampler.next_task()
    state = sampler.state_dict()
    def forbidden(*_args):
        raise AssertionError('Catalog must not be copied for sampler-state validation')
    monkeypatch.setattr(FullMusicCatalog, '__deepcopy__', forbidden, raising=False)
    assert sampler.validate_state_dict(state)
    corrupted = copy.deepcopy(state)
    corrupted['rng'] = {'bit_generator': 'invalid'}
    with pytest.raises(ValueError):
        sampler.load_state_dict(corrupted)
    assert sampler.state_dict() == state
