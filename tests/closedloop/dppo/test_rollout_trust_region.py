"""逐rollout学习率和有意义更新边界的独立CPU回归。

输入为非空分组Adam状态、小型真实Stage1扩散网络及可控的KL拒绝替身。先核对名义
LR在新批次重试且组倍率/动量不丢失，再通过真实流式PPO反传与BC验证：额度接近
上限但仍有有效步长时继续；后续有限KL拒绝只恢复待接受小批次，BC控制器、游标和RNG
不能虚增。数值异常仍抛出，不能当作正常额度耗尽。

解析KL替身仅用于覆盖控制分支，不代表真实信任域或物理收益。全部临时通信文件
位于pytest tmp_path，不启动GPU、PhysX、正式训练或下载权重；八卡真实数据比较
由显式离线入口单独执行，不将这些CPU断言冒充物理验收。
"""
import copy

import pytest
import torch
import torch.distributed as dist

from gem.closedloop.dppo import updater
from gem.closedloop.dppo.actor_backtracking import backtracked_step
from gem.closedloop.dppo.actor_optimizer import actor_base_lr, actor_learning_rates, set_actor_base_lr
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from tests.closedloop.dppo.test_actor_backtracking import Collective, equal_state
from tests.closedloop.dppo.test_actor_optimizer_groups import prepared
from tests.closedloop.dppo.test_clipped_dppo import case,close_cpu_group
from tests.closedloop.test_stage1_actor import actor_factory


@pytest.mark.parametrize('raise_error', [False, True])
def test_new_rollout_retries_nominal_lr_without_resetting_adam_or_group_ratios(monkeypatch, raise_error):
    actor, optimizer, controller = prepared()
    set_actor_base_lr(optimizer, 7.63e-9)
    adam = copy.deepcopy(optimizer.state_dict()['state'])
    def inner(policy, opt, *args, **kwargs):
        assert actor_base_lr(opt) == pytest.approx(1e-6)
        assert actor_learning_rates(opt)['transformer_00'] == pytest.approx(1e-7)
        assert equal_state(opt.state_dict()['state'], adam)
        set_actor_base_lr(opt, 1.25e-7)
        if raise_error:
            raise RuntimeError('injected failure after temporary LR')
        return {}
    monkeypatch.setattr(updater, '_actor_update', inner)
    if raise_error:
        with pytest.raises(RuntimeError, match='temporary LR'):
            updater.actor_update(None, optimizer, trust_region={'maximum_lr': 1e-6})
    else:
        report = updater.actor_update(None, optimizer, trust_region={'maximum_lr': 1e-6})
        assert report['last_trial_learning_rate'] == pytest.approx(1.25e-7)
    assert actor_base_lr(optimizer) == pytest.approx(1e-6)
    assert equal_state(optimizer.state_dict()['state'], adam)


def test_minimum_trial_restores_nonempty_adam_and_never_executes_tiny_steps():
    actor, optimizer, controller = prepared()
    before = copy.deepcopy((actor.state_dict(), optimizer.state_dict(), controller.state_dict()))
    attempts = []
    result = backtracked_step(actor, optimizer, controller,
        evaluate=lambda: dict(chain_joint_kl={'max': 2.}), limits={'chain_joint_max': 1.},
        distributed=Collective(), minimum_lr=2.5e-7, max_attempts=8,
        reserve_attempt=lambda: attempts.append(actor_base_lr(optimizer)))
    assert attempts == pytest.approx([1e-6, 5e-7, 2.5e-7])
    assert result['attempt_count'] == 3 and result['finite_trust_region_exhausted']
    assert result['rejection_reason'] == 'minimum_trial_lr'
    assert equal_state(before, (actor.state_dict(), optimizer.state_dict(), controller.state_dict()))


def test_last_rejected_trial_is_distinct_from_last_accepted_lr(monkeypatch):
    _, optimizer, _ = prepared()
    def inner(*args, **kwargs):
        return dict(steps=[dict(learning_rate=2e-7)],discarded_pending_minibatch=dict(
            trust_region=dict(attempts=[dict(learning_rate=1e-7),dict(learning_rate=5e-8)])))
    monkeypatch.setattr(updater, '_actor_update', inner)
    result = updater.actor_update(None, optimizer, trust_region={'maximum_lr': 1e-6})
    assert result['last_accepted_learning_rate'] == pytest.approx(2e-7)
    assert result['last_trial_learning_rate'] == pytest.approx(5e-8)
    assert actor_base_lr(optimizer) == pytest.approx(1e-6)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_kl_cannot_be_reported_as_normal_rollout_exhaustion(value):
    actor, optimizer, controller = prepared()
    before = copy.deepcopy((actor.state_dict(), optimizer.state_dict(), controller.state_dict()))
    result = backtracked_step(actor, optimizer, controller,
        evaluate=lambda: dict(chain_joint_kl={'max': value}), limits={'chain_joint_max': 1.},
        distributed=Collective(), minimum_lr=5e-7, max_attempts=2)
    assert not result['accepted'] and not result['finite_trust_region_exhausted']
    assert equal_state(before, (actor.state_dict(), optimizer.state_dict(), controller.state_dict()))


class Anchor:
    def __init__(self):
        self.calls = 0
        self.batch_size = 2
        self.generator = torch.Generator().manual_seed(802)
    def state_dict(self):
        return dict(calls=self.calls, batch_size=self.batch_size, rng=self.generator.get_state())
    def load_state_dict(self, state):
        self.calls, self.batch_size = state['calls'], state['batch_size']
        self.generator.set_state(state['rng'])
    def backward(self, actor, weight):
        self.calls += 1
        torch.rand(3, generator=self.generator)
        torch.rand(4)
        loss = sum(p.square().mean() for p in actor.parameters() if p.requires_grad)
        (loss * weight).backward()
        return dict(loss=float(loss.detach()), batch_size=self.batch_size,
                    samples=[[{'dataset_id': 'fixture'}] * self.batch_size])


@pytest.mark.parametrize('boundary', ['near_budget', 'finite_rejected', 'numerical'])
def test_real_streamed_update_keeps_accepted_steps_and_restores_discarded_BC(tmp_path, actor_factory, monkeypatch, boundary):
    from gem.closedloop.dppo import actor_backtracking
    policy, rows, targets = case(actor_factory)
    optimizer = torch.optim.AdamW([p for p in policy.actor.parameters() if p.requires_grad],
                                  lr=1e-6, weight_decay=0.)
    anchor = Anchor()
    controller = BCGradientController()
    # 近额度用例只隔离“额度比例是否误触发停止”，小网络用自身的可测安全步长，
    # 避免真实PPO裁剪饱和抢先触发另一条已独立覆盖的停止分支。
    nominal = 1e-8 if boundary == 'near_budget' else 1e-6
    manifest = [dict(owner_rank=0, local_index=i, valid=True, has_free=True) for i in range(len(rows))]
    saved_noise = torch.get_rng_state().clone()
    original = actor_backtracking.backtracked_step
    attempts = 0
    attempt_counts = []
    accepted = None
    def controlled(*args, **kwargs):
        nonlocal attempts, accepted
        attempts += 1
        evaluate = kwargs['evaluate']
        def check():
            report = evaluate()
            if boundary == 'near_budget':
                report['chain_joint_kl']['max'] = kwargs['limits']['chain_joint_max'] * .96
            elif attempts > 1:
                if boundary == 'numerical':
                    raise FloatingPointError('injected probability anomaly')
                report['chain_joint_kl']['max'] = kwargs['limits']['chain_joint_max'] * 2
            return report
        kwargs['evaluate'] = check
        result = original(*args, **kwargs)
        attempt_counts.append(result['attempt_count'])
        if result['accepted']:
            accepted = copy.deepcopy((policy.actor.state_dict(), optimizer.state_dict()['state'], anchor.state_dict(),
                                      controller.state_dict(), torch.get_rng_state()))
        return result
    monkeypatch.setattr(actor_backtracking, 'backtracked_step', controlled)
    cache = RolloutTensorCache(rows, targets, 'cpu')
    dist.init_process_group('gloo', init_method=(tmp_path/'gloo').as_uri(), rank=0, world_size=1)
    try:
        call = lambda: updater.actor_update(policy, optimizer, rows, targets, cache, manifest, controller,
            distributed=DistributedCollectives(0, 1), anchor=anchor, generator=torch.Generator().manual_seed(5),
            minibatch_chains=1, microbatch=3, bc_samples_per_rollout=4, detailed_diagnostics=True,
            soft_kl=1e6, trust_region=dict(maximum_lr=nominal, growth=1.25, margin=.8,
                max_attempts=2, minimum_trial_lr_ratio=.0625),
            kl_limits={f'chain_joint_{k}': 1e6 for k in ('mean', 'p95', 'p99', 'max')})
        if boundary == 'numerical':
            with pytest.raises(FloatingPointError, match='whole iteration rollback'):
                call()
            return
        report = call()
        expected_steps = len(rows) if boundary == 'near_budget' else 1
        assert report['optimizer_steps'] == expected_steps
        assert report['skipped_optimizer_steps'] == len(rows)-expected_steps
        assert report['early_stop_reason'] == (None if boundary == 'near_budget'
                                                else 'rollout_no_useful_trust_region_step')
        assert report['steps'][0]['parameter_update']['gradient_projection_measured']
        assert equal_state((policy.actor.state_dict(), optimizer.state_dict()['state'], anchor.state_dict(),
                            controller.state_dict(), torch.get_rng_state()), accepted)
        assert anchor.calls == controller.updates == expected_steps
        assert all(float(state['step']) == expected_steps for state in optimizer.state.values())
        assert report['executed_optimizer_steps'] == sum(attempt_counts)
        if boundary == 'finite_rejected':
            assert attempt_counts[-1] == 2
            assert report['rejected_step_candidates'] == sum(attempt_counts)-1
        assert actor_base_lr(optimizer) == pytest.approx(nominal)
        assert policy.assert_frozen_identity()['checked']
    finally:
        cache.close()
        close_cpu_group()
        torch.set_rng_state(saved_noise)
