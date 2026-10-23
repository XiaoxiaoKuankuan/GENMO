"""RoboFFT 审查修正的诊断覆盖、数学不变性与显式启动入口回归。

真实小型 Stage1Actor 生成 compact 动作并只准备一次旧 MC；比较有界哨兵诊断和
旧的逐步全量诊断得到的每个参数及 Adam 状态，确保诊断不改变优化。另在固定旧
损失下比较 MC 微批 1/2/4 与单遍 Autograd 的值和完整条件编码器梯度，覆盖重排。
命令行测试只检查参数解析和 Bash 转发，不启动 CUDA、Isaac 或正式训练。
所有运行产物限于 pytest 临时目录；CPU 数学测试不代表生产 GPU 性能或物理收益。
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tests.closedloop.test_stage1_actor import actor_factory
from tests.closedloop.dppo.test_robofft import (
    cpu_threads, make_policy, make_rows, settings, gradients, local_distributed,
)
from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.robofft import updater
from gem.closedloop.robofft.surrogate import action_objective

ROOT = Path(__file__).resolve().parents[3]


def test_fixed_sentinel_is_bounded_unique_and_does_not_draw_rng():
    state = torch.random.get_rng_state().clone()
    eligible = [i for i in range(1003) if i % 3]
    chosen = updater.fixed_sentinel_indices(eligible)
    assert len(chosen) == len(set(chosen)) == 64
    assert set(chosen).issubset(eligible)
    assert chosen == updater.fixed_sentinel_indices(eligible)
    assert updater.fixed_sentinel_indices([]) == []
    assert updater.fixed_sentinel_indices([2, 7]) == [2, 7]
    assert torch.equal(state, torch.random.get_rng_state())


def test_bounded_diagnostics_preserve_every_parameter_and_adam(actor_factory, monkeypatch):
    policy, context = make_policy(actor_factory)
    base, targets, original_cache, _ = make_rows(policy, context)
    original_cache.close()
    rows = [copy.deepcopy(base[i % 2]) for i in range(5)]
    targets = dict(targets, advantages=torch.tensor([1., -.7, .2, -.4, 99.], dtype=torch.float64))
    manifest = [dict(valid=i < 4, has_free=True, owner_rank=0, local_index=i) for i in range(5)]
    cache = RolloutTensorCache(rows, targets, 'cpu')
    mc_rng = torch.Generator().manual_seed(444)
    updater.prepare_mc(policy, rows, cache, mc_rng, microbatch=2)
    fixed_rng = mc_rng.get_state().clone()
    old = [row.mc_record.old_loss.clone() for row in rows]
    initial = copy.deepcopy(policy.actor.state_dict())
    actual_diagnostics = updater.diagnostics
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs.get('indices'))
        return actual_diagnostics(*args, **kwargs)

    monkeypatch.setattr(updater, 'diagnostics', observed)
    optimizer = make_actor_optimizer(policy.actor, settings())
    order_rng = torch.Generator().manual_seed(76)
    report = updater.actor_update(policy, optimizer, rows, targets, cache, manifest,
        distributed=local_distributed(), generator=order_rng, ppo_epochs=2,
        minibatch_actions=2, microbatch=1, diagnostic_sentinel_actions=1)
    expected = copy.deepcopy(policy.actor.state_dict())
    expected_adam = copy.deepcopy(optimizer.state_dict())
    assert calls[0] is None and calls[-1] is None and len(calls) == 6
    sentinel, = report['diagnostic_policy']['sentinel_manifest_indices']
    for step, indices in zip(report['steps'], calls[1:-1]):
        order = report['orders'][step['epoch']]
        begin = ((step['optimizer_step'] - 1) % 2) * 2
        assert set(indices) == set(order[begin:begin+2]) | {sentinel}
        assert len(indices) == len(set(indices)) <= 3
        assert 4 not in indices
        post = step['post_step_surrogate']
        assert post['scope'] == 'current_minibatch_union_fixed_sentinel'
        assert post['covered_actions'] == len(indices)
    assert report['final_surrogate']['covered_actions'] == 4
    assert report['final_surrogate']['scope'] == 'full_rollout'

    # 恢复相同参数/空 Adam/重排 RNG，只把逐步诊断改回旧的全量，目标和 MC 保持不变。
    def legacy_full(*args, **kwargs):
        kwargs.pop('indices', None)
        kwargs.pop('scope', None)
        return actual_diagnostics(*args, **kwargs)

    monkeypatch.setattr(updater, 'diagnostics', legacy_full)
    policy.actor.load_state_dict(initial)
    optimizer = make_actor_optimizer(policy.actor, settings())
    replay_order = torch.Generator().manual_seed(76)
    replay = updater.actor_update(policy, optimizer, rows, targets, cache, manifest,
        distributed=local_distributed(), generator=replay_order, ppo_epochs=2,
        minibatch_actions=2, microbatch=1, diagnostic_sentinel_actions=1)
    assert report['orders'] == replay['orders']
    assert torch.equal(order_rng.get_state(), replay_order.get_state())
    for name, value in policy.actor.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0, msg=name)
    for index, values in optimizer.state_dict()['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, expected_adam['state'][index][name], atol=0, rtol=0)
    for row, value in zip(rows, old):
        row.mc_record.validate()
        assert torch.equal(row.mc_record.old_loss, value)
    assert torch.equal(fixed_rng, mc_rng.get_state())
    cache.close()


@pytest.mark.parametrize('mc_microbatch', [1, 2, 4])
def test_mc_microbatch_matches_single_graph_after_reordering(actor_factory, mc_microbatch):
    policy, context = make_policy(actor_factory)
    rows, targets, cache, _ = make_rows(policy, context)
    updater.prepare_mc(policy, rows, cache, torch.Generator().manual_seed(10))
    old = torch.stack([r.mc_record.old_loss for r in rows])
    fixed = {key: torch.stack([getattr(r.mc_record, key) for r in rows])
             for key in ('eps', 'sampling_indices', 'original_timesteps')}
    full = policy.losses(context, cache.get('action_x0', rows), **fixed)
    metrics = action_objective(full, old, targets['advantages'])
    metrics['loss'].mean().backward()
    expected = gradients(policy.actor)
    policy.actor.zero_grad(set_to_none=True)
    policy.robofft['mc_microbatch'] = mc_microbatch
    result = updater.objective_batch(policy, rows[::-1], cache, clip=.01, global_count=2, backward=True)
    actual = gradients(policy.actor)
    torch.testing.assert_close(result[0] / 2, metrics['loss'].mean(), atol=2e-6, rtol=2e-6)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-4, msg=name)
    assert sum(float(x.square().sum()) for x in actual.values()) > 0
    assert torch.equal(old, torch.stack([r.mc_record.old_loss for r in rows]))
    cache.close()


@pytest.mark.parametrize('command', [
    [sys.executable, '-B', str(ROOT/'tools/train_closedloop_stage10.py')],
    ['bash', str(ROOT/'scripts/train_stage10_8gpu_server1.sh')],
    [sys.executable, '-B', str(ROOT/'tools/run_stage2_training_supervisor.py')],
])
def test_training_entries_reject_missing_config_before_start(command, tmp_path):
    result = subprocess.run([*command, '--output-dir', str(tmp_path/'unused')],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and '--config' in result.stderr
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('filename', ['stage2_continuous_rolling_robofft.yaml',
                                     'stage2_continuous_rolling_train10.yaml'])
def test_wrapper_forwards_explicit_algorithm_without_default(filename, tmp_path):
    stub = tmp_path/'python_stub'
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    config = str(ROOT/'configs/closedloop'/filename)
    result = subprocess.run(['bash', str(ROOT/'scripts/train_stage10_8gpu_server1.sh'),
        '--config='+config, '--output-dir', str(tmp_path/'unused')],
        env=dict(os.environ, STAGE10_PYTHON=str(stub)), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0
    assert result.stdout.splitlines().count('--config='+config) == 1
    assert '--nproc_per_node=8' in result.stdout
    assert not (tmp_path/'unused').exists()
