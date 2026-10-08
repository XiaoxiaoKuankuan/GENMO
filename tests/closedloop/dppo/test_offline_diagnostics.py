"""显式离线诊断的只读边界和既有 rollout 文件契约测试。

测试证明逐步 Actor 反向不会执行参数更新、不会遗留梯度或模式变化；Critic 各候选从
同一独立状态开始，原模型与 Adam 状态保持不变。真实小 Actor 的采样只用于临时夹具，
随后经原 RolloutWriter 发布，再检验读取时的 SHA、顺序和 fixed_targets 绑定。
所有文件位于 pytest 系统临时目录，不读取正式 checkpoint 或启动 GMT／物理环境。
"""
from __future__ import annotations

import copy

import pytest
import torch

from gem.closedloop.dppo.buffer import UpperTransition
from gem.closedloop.dppo.offline_diagnostics import (
    critic_step_comparison, denoising_gradient_diagnostic, load_immutable_rollouts,
)
from gem.closedloop.dppo.run_management import RolloutWriter
from gem.closedloop.dppo.updater_v2 import analytic_kl_local
from tests.closedloop.dppo.test_distributed_training import SmallCritic, _assert_nested_close
from tests.closedloop.dppo.test_updater_v2 import BatchGaussianPolicy, rows
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_step_gradient_diagnostic_restores_original_gradients_modes_and_weights():
    policy = BatchGaussianPolicy()
    samples = rows(policy)
    actor = policy.actor
    actor.train()
    actor.denoiser.eval()
    actor.denoiser.weight.grad = torch.ones_like(actor.denoiser.weight) * .123
    gradient = actor.denoiser.weight.grad
    before = copy.deepcopy(actor.state_dict())
    modes = {name: module.training for name, module in actor.named_modules()}
    report = denoising_gradient_diagnostic(policy, samples, {'advantages': torch.tensor([1., -.3, .4, -.7])})
    assert report['optimizer_steps'] == 0
    assert len(report['per_denoising_step']) == 2
    assert all(row['gradient_norm'] > 0 for row in report['per_denoising_step'])
    assert actor.denoiser.weight.grad is gradient
    assert {name: module.training for name, module in actor.named_modules()} == modes
    _assert_nested_close(actor.state_dict(), before)


def test_critic_comparison_uses_independent_snapshots_and_same_rng_prefix():
    critic = SmallCritic()
    optimizer = torch.optim.AdamW(critic.parameters(), lr=.001, weight_decay=0.)
    before, optimizer_before = copy.deepcopy(critic.state_dict()), copy.deepcopy(optimizer.state_dict())
    sample = rows(BatchGaussianPolicy())
    report = critic_step_comparison(critic, optimizer.state_dict(), sample,
        {'returns': torch.tensor([1., -.2, .7, 1.3])}, steps=(1, 2, 3), batch_size=2)
    assert [case['steps'] for case in report['cases']] == [1, 2, 3]
    losses = [case['training_batch']['losses'] for case in report['cases']]
    assert losses[0] == losses[1][:1] == losses[2][:1]
    assert losses[1] == losses[2][:2]
    _assert_nested_close(critic.state_dict(), before)
    _assert_nested_close(optimizer.state_dict(), optimizer_before)


def test_kl_kernel_statistics_share_the_existing_forward():
    policy = BatchGaussianPolicy()
    sample = rows(policy)
    report = analytic_kl_local(policy, sample)
    assert report['mean_joint_kl'] == 0
    assert report['free_coordinate_count'] == {'min': 1, 'max': 3, 'mean': 1.75}
    for row in report['per_denoising_step']:
        assert row['mean_std'] == pytest.approx(.7)
        assert row['mean_old_std'] == row['mean_base_std']
        assert row['mean_normalized_mean_shift_rms'] == 0


def test_offline_reader_checks_real_writer_sha_and_target_order(actor_factory, tmp_path):
    from gem.closedloop.dppo.policy import DPPODiffusionPolicy
    from tests.closedloop.test_stage1_actor import _conditions
    actor, batch = actor_factory(starts=(45,))
    policy = DPPODiffusionPolicy(actor, steps=2)
    trace = policy.sample_rollout(_conditions(batch), generator=torch.Generator().manual_seed(14))
    writer = RolloutWriter(tmp_path / 'rollout', policy_version=3, chunk_size=2)
    paths = []
    for index in range(2):
        row = UpperTransition(identity=dict(run_id='offline-test', backend_session_id='one', episode_id=1,
            decision_id=index, policy_version=3), context=trace['conditions'], next_context=None,
            chain=trace['chain'][0], old_log_prob=trace['old_log_probs'][0], free_mask=trace['free_mask'][0],
            rewards=torch.tensor([1.], dtype=torch.float64), old_value=.1, next_value=0.,
            control_tick_begin=index * 12, control_tick_end=(index + 1) * 12, executed_control_steps=1,
            executed_physics_steps=4, terminated=True, metadata={'sampler_trace': trace})
        paths.append(writer.append(row))
    manifest = writer.finish()
    targets = tmp_path / 'fixed_targets.pt'
    torch.save(dict(advantages=torch.tensor([-.5, .5]), returns=torch.tensor([1., 2.]),
                    valid=torch.ones(2, dtype=torch.bool)), targets)
    selected, fixed, report = load_immutable_rollouts([manifest], [targets], max_chains=1)
    assert len(selected) == 1 and selected[0].identity['decision_id'] == 0
    assert fixed['returns'].tolist() == [1.]
    assert report['available_chains'] == 2 and report['selected_chains'] == 1
    paths[0].write_bytes(b'corrupted-test-record')
    with pytest.raises(ValueError, match='size/SHA'):
        load_immutable_rollouts([manifest], [targets], max_chains=1)
