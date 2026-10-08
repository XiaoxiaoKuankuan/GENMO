"""八采集器主流程的配置、保存边界和整轮回滚集成检查。

使用真实 PyTorch 小模型、Adam 状态与持久预算，替换昂贵的物理采集和网络前向，
注入 Critic 后异常、Actor 后异常及最终 KL 超限。检查模型、优化器和随机数确实
退回整轮起点，而已经持久化的优化尝试数保持不变；不把 helper 的单元通过当作
真实八卡物理执行验收。所有文件仅位于 pytest tmp_path，不读取正式训练模型。
普通非周期轮次另外经过真实小型 PPO 更新器，检查每个 step 的 PPO／BC 分解梯度
都由根调用启用，而不是只在首轮或每一百轮完整概率诊断时才输出。
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
from gem.closedloop.dppo.parallel_support import begin_lease
from tools.train_closedloop_stage10 import configuration


class Solo:
    rank, world_size, device = 0, 1, torch.device('cpu')

    def broadcast_object(self, value):
        return value

    def all_gather_object(self, value):
        return [value]


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


def test_global_collection_credit_rejected_before_any_rank_reservation(tmp_path):
    budget = TrainingBudget(tmp_path/'global.json', dict(accepted_iterations=2, optimizer_attempts=8,
        generations=10, control_steps=100, physics_steps=400))
    credits = [dict(generations=6, control_steps=10, physics_steps=40)]*2
    before = budget.state_dict()
    with pytest.raises(RuntimeError, match='complete parallel collection lease'):
        begin_lease(Solo(), None, budget, tmp_path, 'collect', credits)
    assert budget.state_dict() == before
    assert not (tmp_path/'resource_lease.json').exists()


def test_calibration_measures_selected_cfg_after_numeric_fallback(tmp_path, monkeypatch):
    from tools import train_closedloop_stage10 as entry
    events = []
    env = SimpleNamespace(latency_budget_s=.26, config={'stage9':{'seed':42}},
        reset_task=lambda *a, **k: events.append('reset'),
        preview_context=lambda: ({'value':torch.zeros(1)}, None))
    context = SimpleNamespace(env=env, generators={}, base_config={'timing':{'calibration_warmup':1,'calibration_samples':2}},
        settings={'episode_seconds':10.,'denoising_microbatch':4}, distributed=Solo(), stage={'seed':42},
        state={}, policy=SimpleNamespace(cfg_batch=True), catalog=SimpleNamespace(
            samples={'train':{'Mine':[{}]}}, load_music=lambda _: None), budget=None)
    monkeypatch.setattr(training, '_new_phase', lambda *args: (tmp_path, 'profile', SimpleNamespace(reserve=lambda *a, **k:None)))
    monkeypatch.setattr(training, 'probe_profiles', lambda *a, **k: [dict(microbatch=1,cfg_batch=False,passed=True)])
    monkeypatch.setattr(training, 'finish_lease', lambda *args: {'used':{}})
    def calibrate(*args):
        assert context.policy.cfg_batch is False
        events.append('calibrate_separate_cfg')
        env.latency_budget_s=.4
        return dict(durations=[.3,.31],latency_budget_s=.4)
    monkeypatch.setattr(entry, 'calibrate', calibrate)
    training._calibrate_and_profile(context)
    assert events == ['reset','calibrate_separate_cfg']
    assert env.latency_budget_s == .4
    assert context.state['execution_profile']['microbatch'] == 1


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
    rejected = json.loads((tmp_path/'rejected_000001.json').read_text())
    assert rejected['rolled_back']
    assert rejected['probability_check'] == dict(passed=True, scope='full_rollout_before_first_step')
    assert rejected['timings']['critic_seconds'] > 0
    assert (rejected['timings']['actor_seconds'] > 0) == (failure != 'critic')
    assert (rejected['timings']['kl_seconds'] > 0) == (failure == 'kl')


def test_ordinary_iteration_records_decomposed_gradients_on_every_actor_step(tmp_path):
    from tests.closedloop.dppo.test_distributed_training import SmallCritic
    from tests.closedloop.dppo.test_updater_v2 import Anchor, BatchGaussianPolicy, rows

    class SingleRank(Solo):
        def sum_gradients(self, module):
            pass

        def sum_tensor(self, value):
            return value

        def gather_rows(self, values, positions, count):
            assert sorted(positions) == list(range(count))
            return values[torch.tensor(positions).argsort()].double()

    config = configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage10_8gpu_server1_v2.yaml')
    config['stage9'].update(actor_minibatch_internal_transitions=4, critic_steps=2)
    policy, critic, anchor = BatchGaussianPolicy(), SmallCritic(), Anchor()
    samples = rows(policy)
    manifest = [dict(owner_rank=0, local_index=index, valid=True, has_free=True) for index in range(len(samples))]
    budget = TrainingBudget(tmp_path/'budget.json', dict(accepted_iterations=2, optimizer_attempts=8,
        generations=20, control_steps=100, physics_steps=400))
    # 使用真实状态接口让主循环也执行根进程不可变快照，而不绕过事务边界。
    anchor.state_dict = lambda: {'calls':anchor.calls}
    context = SimpleNamespace(distributed=SingleRank(), initial_iteration=0, stage=config['stage10'],
        settings=config['stage9'], policy=policy, actor=policy.actor, critic=critic,
        actor_optimizer=torch.optim.AdamW(policy.actor.parameters(), lr=5e-9, weight_decay=0.),
        critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=0.),
        bc=anchor, budget=budget, session=tmp_path, profile={'microbatch':3},
        generators={key:torch.Generator().manual_seed(13) for key in ('actor','critic')})
    targets = dict(advantages=torch.tensor([1., -.3, .4, -.7]), returns=torch.tensor([1., -.2, .7, 1.3]))
    report = training._update(context, SimpleNamespace(transitions=samples), targets, manifest, 2)
    assert report['probability_check']['scope'] == 'rank_sentinel_all_denoising_steps'
    assert report['actor']['optimizer_steps'] == 4
    assert report['actor']['bc_global_samples'] == 8
    for step in report['actor']['steps']:
        gradients = step['gradient_contributions']
        assert gradients['all']['ppo_norm'] > 0
        assert gradients['all']['weighted_bc_norm'] > 0
        assert gradients['shared']['cosine'] is not None
    assert budget.state_dict()['used']['optimizer_attempts'] == 4
