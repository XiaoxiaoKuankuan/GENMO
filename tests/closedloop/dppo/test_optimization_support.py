"""当前 DPPO 公共概率检查、价值快照和条件缓存的 CPU 验收。

小型高斯策略保留真实自由坐标联合概率和旧链，检查零更新概率、缓存价值版本、
预训练位置表及多次反向的条件编码。Anchor 和有界双进程夹具供当前集成测试复用。
原旧优化器的专属测试已经随旧方法撤下；当前更新器由 training_integration 和
八卡 systems/learning 验收负责。测试不读取正式数据，不启动物理训练。
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
from gem.closedloop.dppo.trainer import fixed_targets, populate_values, trainable_actor_parameters
from gem.closedloop.dppo.optimization_support import probability_check_local
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
