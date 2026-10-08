"""可选小 Critic 单卡广播与逐步 x0 诊断的有限 CPU 验收。

两个真实 Gloo 进程验证 3+1 不均匀分片及 rank 0 空分片：全局索引、固定目标和 Adam
状态与原单卡更新一致，广播后所有副本能继续相同更新。测试检查辅助入口只汇集条件、
时间及 returns，没有读取随机链；同时不把通信时间的单元测试数值当成 GPU 提速结论。

x0 夹具明确区分 pred_x_start 与核均值，检查真实输出位移、自由坐标加权分母、不同
微批一致性、前缀/padding 排除，以及条件或链被改写后的拒绝机制。还用真实小型
Stage1Actor 验证输出接口。所有进程有时间上限，文件只写 pytest 临时目录，不读取
正式数据、不运行 GMT、不访问服务器、不改变正式训练默认值。
"""
from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.optional_diagnostics import (
    capture_x0_reference, critic_update_root_broadcast, x0_change_local,
)
from gem.closedloop.dppo.updater_v2 import critic_update_local
from tests.closedloop.dppo.test_distributed_training import SmallCritic, _assert_nested_close
from tests.closedloop.dppo.test_updater_v2 import BatchGaussianPolicy, _spawn_bounded, rows
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory


class X0Policy(BatchGaussianPolicy):
    def transition_parameters(self, context, state, step):
        output = super().transition_parameters(context, state, step)
        # x0 与核均值明确不同，防止测试把 mean 位移误记为 x0 位移。
        output['pred_x_start'] = 3 * output['mean'] + self.actor.bc_only * (~output['free_mask'])
        return output


class CriticInput(SimpleNamespace):
    @property
    def chain(self):
        raise AssertionError('Root Critic entry must never access the diffusion chain')


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _critic_case(count, distributed=None):
    source = rows(BatchGaussianPolicy(), count)
    source = [CriticInput(context=row.context, metadata=row.metadata,
                         free_mask=row.free_mask, transition_valid=True) for row in source]
    values = torch.tensor([1., -.2, .7, 1.3][:count])
    critic = SmallCritic()
    optimizer = torch.optim.AdamW(critic.parameters(), lr=.007, weight_decay=.03)
    # 非空 Adam 动量验证广播完整状态，而非只同步参数或从空 Adam 重启。
    critic_update_local(critic, optimizer, source, {'returns': values}, steps=1, batch_size=2,
                        generator=torch.Generator().manual_seed(71))
    if distributed is None:
        local, target, manifest = source, values, None
    else:
        assignments = [[0, 1, 2], [3]] if count == 4 else [[], [0]]
        indices = assignments[distributed.rank]
        local, target = [source[i] for i in indices], values[indices]
        manifest = [dict(owner_rank=rank, local_index=index, valid=True, has_free=True)
                    for rank, items in enumerate(assignments) for index in range(len(items))]
    generator = torch.Generator().manual_seed(313)
    before_generator = generator.get_state().clone()
    report = critic_update_root_broadcast(critic, optimizer, local, {'returns': target},
        global_manifest=manifest, distributed=distributed, steps=3, batch_size=2, generator=generator)
    result = dict(state=copy.deepcopy(critic.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                  report=report, generator_changed=not torch.equal(before_generator, generator.get_state()))
    # 广播后的状态继续一步，确认 optimizer 的 step/momentum/param_groups 一致。
    critic_update_local(critic, optimizer, source, {'returns': values}, steps=1, batch_size=2,
                        generator=torch.Generator().manual_seed(99))
    result['continued_state'] = copy.deepcopy(critic.state_dict())
    result['continued_optimizer'] = copy.deepcopy(optimizer.state_dict())
    return result


def _worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory / 'rendezvous').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        collective = DistributedCollectives(rank, 2, device='cpu')
        result = {count: _critic_case(count, collective) for count in (4, 1)}
        policy = X0Policy()
        source = rows(policy)
        indices = [0, 1, 2] if rank == 0 else [3]
        local = [source[i] for i in indices]
        manifest = [dict(owner_rank=0 if i < 3 else 1, local_index=i if i < 3 else 0,
                         valid=True, has_free=True) for i in range(4)]
        reference = capture_x0_reference(policy, local, global_manifest=manifest, distributed=collective)
        with torch.no_grad():
            policy.actor.denoiser.weight.add_(.02)
        result['x0'] = x0_change_local(policy, local, reference, global_manifest=manifest, distributed=collective)
        torch.save(result, directory / f'rank_{rank}.pt')
        dist.barrier()  # 所有证据落盘后共同销毁Gloo，保持原有超时和数值检查。
    finally:
        dist.destroy_process_group()


def test_root_critic_broadcast_matches_global_update_and_empty_root(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo is unavailable')
    _spawn_bounded(_worker, tmp_path)
    output = [torch.load(tmp_path / f'rank_{rank}.pt', weights_only=False) for rank in range(2)]
    for count in (4, 1):
        expected = _critic_case(count)
        for rank, results in enumerate(output):
            actual = results[count]
            for field in ('state', 'optimizer', 'continued_state', 'continued_optimizer'):
                _assert_nested_close(actual[field], expected[field])
            for field in ('before', 'after', 'losses', 'gradient_norms'):
                _assert_nested_close(actual['report'][field], expected['report'][field])
            assert actual['generator_changed'] == (rank == 0 and count > 1)
            assert actual['report']['optimizer_state_broadcast']
            assert actual['report']['critic_payload_bytes'] > 0
            assert all(value >= 0 for value in actual['report']['timings'].values())
    policy = X0Policy()
    source = rows(policy)
    reference = capture_x0_reference(policy, source)
    with torch.no_grad():
        policy.actor.denoiser.weight.add_(.02)
    expected = x0_change_local(policy, source, reference)
    for result in output:
        _assert_nested_close(result['x0'], expected)


@pytest.mark.parametrize('microbatch', [1, 3, 8])
def test_x0_is_actual_prediction_difference_with_free_coordinate_denominator(microbatch):
    policy = X0Policy()
    source = rows(policy)
    before_rows = copy.deepcopy(source)
    reference = capture_x0_reference(policy, source, denoising_microbatch=microbatch, policy_version_label='actor:9')
    unchanged = x0_change_local(policy, source, reference, denoising_microbatch=microbatch)
    assert all(step['max_absolute_x0_change'] == 0 for step in unchanged['per_denoising_step'])
    with torch.no_grad():
        policy.actor.denoiser.weight.add_(.02)
        policy.actor.bc_only.add_(1000.)  # 只有已知前缀改变，必须排除。
    report = x0_change_local(policy, source, reference, denoising_microbatch=microbatch)
    features = torch.tensor([row.context['feature'].item() for row in source]).double()
    counts = torch.tensor([row.free_mask.sum() for row in source]).double()
    for step in report['per_denoising_step']:
        expected = .06 * (step['step_index'] + 1) * features
        assert step['free_coordinate_x0_rms'] == pytest.approx(float((expected.square() @ counts / counts.sum()).sqrt()), rel=2e-6)
        assert step['max_absolute_x0_change'] == pytest.approx(float(expected.max()), rel=2e-6)
        assert step['free_coordinate_count'] == int(counts.sum())
    assert report['policy_version_label'] == 'actor:9'
    assert report['reference_extra_internal_forwards'] == report['comparison_extra_internal_forwards'] == 8
    for before, after in zip(before_rows, source):
        _assert_nested_close(vars(before), vars(after))


@pytest.mark.parametrize('changed', ['condition', 'chain', 'mask'])
def test_x0_reference_rejects_changes_to_fixed_inputs(changed):
    policy = X0Policy()
    source = rows(policy)
    reference = capture_x0_reference(policy, source)
    if changed == 'condition':
        source[0].context['feature'].add_(1.)
    elif changed == 'chain':
        source[0].chain[0, 0, 0] += .01
    else:
        source[0].free_mask[0, 1] = ~source[0].free_mask[0, 1]
    with pytest.raises(ValueError, match='fixed chain|free-coordinate mask|free mask'):
        x0_change_local(policy, source, reference)


def test_x0_reference_works_with_real_stage1_actor(actor_factory):
    from gem.closedloop.dppo.policy import DPPODiffusionPolicy
    from tests.closedloop.test_stage1_actor import _conditions
    actor, batch = actor_factory(starts=(45,))
    policy = DPPODiffusionPolicy(actor, steps=2)
    trace = policy.sample_rollout(_conditions(batch), generator=torch.Generator().manual_seed(41))
    source = [SimpleNamespace(context=trace['conditions'], chain=trace['chain'][0],
        free_mask=trace['free_mask'][0], old_log_prob=trace['old_log_probs'][0], transition_valid=True,
        metadata={'sampler_trace': trace})]
    reference = capture_x0_reference(policy, source, denoising_microbatch=1)
    before = copy.deepcopy(actor.state_dict())
    report = x0_change_local(policy, source, reference, denoising_microbatch=2)
    assert all(step['max_absolute_x0_change'] < 3e-6 for step in report['per_denoising_step'])
    _assert_nested_close(actor.state_dict(), before)


def test_root_critic_does_not_change_learning_rate_or_allow_unbounded_steps():
    critic = SmallCritic()
    optimizer = torch.optim.AdamW(critic.parameters(), lr=.004)
    source = rows(BatchGaussianPolicy(), 1)
    with pytest.raises(ValueError, match='1 to 80'):
        critic_update_root_broadcast(critic, optimizer, source, {'returns': torch.tensor([1.])}, steps=81)
    report = critic_update_root_broadcast(critic, optimizer, source, {'returns': torch.tensor([1.])}, steps=1)
    assert optimizer.param_groups[0]['lr'] == .004
    assert report['optimizer_steps'] == 1
