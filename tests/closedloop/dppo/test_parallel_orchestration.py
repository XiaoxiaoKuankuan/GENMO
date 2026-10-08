"""八采集器主流程的配置、保存边界和整轮回滚集成检查。

使用真实 PyTorch 小模型、Adam 状态与持久预算，替换昂贵的物理采集和网络前向，
注入 Critic 后异常、Actor 后异常及最终 KL 超限。检查模型、优化器和随机数确实
退回整轮起点，而已经持久化的优化尝试数保持不变；不把 helper 的单元通过当作
真实八卡物理执行验收。所有文件仅位于 pytest tmp_path，不读取正式训练模型。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo import parallel_training as training
from gem.closedloop.dppo.execution_profile import select_profile
from gem.closedloop.dppo.run_management import TrainingBudget
from tools.train_closedloop_stage10 import configuration


class Solo:
    rank, world_size, device = 0, 1, torch.device('cpu')

    def broadcast_object(self, value):
        return value


def test_formal_configuration_and_outer_save_boundaries():
    root = Path(__file__).resolve().parents[3]
    config = configuration(root/'configs/closedloop/stage10_8gpu_server1_v2.yaml')
    settings = config['stage9']
    assert settings['actor_lr'] == 5e-9
    assert 'actor_lr_candidates' not in settings
    assert settings['rollout_upper_steps'] == 160
    assert settings['ppo_epochs'] == 2
    assert settings['actor_minibatch_internal_transitions'] == 1600
    assert settings['bc_batch'] == 2
    assert [i for i in range(1, 601) if training.checkpoint_due(i)] == [300, 600]
    assert training.checkpoint_due(1, normal_end=True)
    assert not training.checkpoint_due(4)


def test_profile_selects_common_pass_without_changing_tolerances():
    ranks = [[dict(microbatch=m, cfg_batch=c, passed=m <= cap and not c)
              for m in (4, 2, 1) for c in (True, False)] for cap in (4, 2)]
    assert select_profile(ranks)['microbatch'] == 2
    assert select_profile(ranks)['cfg_batch'] is False
    with pytest.raises(RuntimeError):
        select_profile(ranks, required=dict(microbatch=4, cfg_batch=False))


@pytest.mark.parametrize('failure', ['critic', 'actor', 'kl'])
def test_whole_iteration_rollback_preserves_spent_budget(tmp_path, monkeypatch, failure):
    actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    aopt = torch.optim.AdamW(actor.parameters(), lr=5e-9)
    copt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    budget = TrainingBudget(tmp_path/'budget.json', dict(accepted_iterations=2, optimizer_attempts=8,
        generations=20, control_steps=100, physics_steps=400))
    config = configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage10_8gpu_server1_v2.yaml')
    context = SimpleNamespace(distributed=Solo(), initial_iteration=0, stage=config['stage10'],
        settings=config['stage9'], policy=SimpleNamespace(actor=actor), actor=actor, critic=critic,
        actor_optimizer=aopt, critic_optimizer=copt, bc=None, budget=budget, session=tmp_path,
        profile={'microbatch':1}, generators={k:torch.Generator().manual_seed(13) for k in ('actor','critic')})
    old_actor, old_critic = copy.deepcopy(actor.state_dict()), copy.deepcopy(critic.state_dict())
    rng = torch.get_rng_state().clone()
    def step(model, optimizer):
        optimizer.zero_grad(); model(torch.randn(3, 2)).sum().backward(); optimizer.step()
    def fake_critic(*args, **kwargs):
        step(critic, copt)
        if failure == 'critic':
            raise FloatingPointError('injected critic failure')
        return {'optimizer_steps':80}
    def fake_actor(*args, **kwargs):
        kwargs['reserve_attempt']()
        step(actor, aopt)
        if failure == 'actor':
            raise FloatingPointError('injected actor failure')
        return {'optimizer_steps':1}
    monkeypatch.setattr(training, 'probability_check_local', lambda *a, **k: {'passed':True})
    monkeypatch.setattr(training, 'critic_update_local', fake_critic)
    monkeypatch.setattr(training, 'actor_update_v2', fake_actor)
    monkeypatch.setattr(training, 'analytic_kl_local', lambda *a, **k: {
        'mean_joint_kl':1., 'per_denoising_step':[{'mean_joint_kl':1.}]})
    monkeypatch.setattr(training, 'broadcast_state', lambda value, _: value)
    with pytest.raises((FloatingPointError, RuntimeError)):
        training._update(context, SimpleNamespace(transitions=[]), {}, [], 1)
    for name, value in actor.state_dict().items():
        assert torch.equal(value, old_actor[name])
    for name, value in critic.state_dict().items():
        assert torch.equal(value, old_critic[name])
    assert not aopt.state and not copt.state
    assert torch.equal(torch.get_rng_state(), rng)
    assert budget.state_dict()['used']['optimizer_attempts'] == (0 if failure == 'critic' else 1)
    assert json.loads((tmp_path/'rejected_000001.json').read_text())['rolled_back']
