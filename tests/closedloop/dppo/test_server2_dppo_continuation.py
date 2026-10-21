"""Server-2 正式 DPPO 协议迁移的数学、数据身份及恢复测试。

使用真实小 Stage1 Actor/四步 PPO 更新器证明关闭 BC 后没有监督调用或控制器
更新；使用带非空 Adam 的小网络证明只继承父 Actor 学习状态，奖励、Critic 和
旧物理轨迹不冒充同协议恢复。配置测试同时保留旧v1复现入口、新v3奖励身份、
原训练音乐评估、取消yaw终止及显式200轮边界。所有文件只写pytest临时目录。
"""
import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from gem.closedloop.dppo.continuation import DisabledBC, initialize_actor_continuation
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.training_checkpoint import SCHEMA, file_sha256
from gem.closedloop.dppo.actor_lr import ActorLRController
from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.updater import actor_update
from tests.closedloop.dppo.test_clipped_dppo import case
from tests.closedloop.dppo.test_actor_backtracking import equal_state
from tests.closedloop.test_stage1_actor import actor_factory

ROOT = Path(__file__).resolve().parents[3]


def test_formal_original_dppo_and_new_reward_configuration():
    c = configuration(ROOT/'configs/closedloop/stage2_dppo_server2_formal.yaml')
    s, t = c['stage10'], c['stage9']
    assert s['derived']['planned_actor_steps'] == 4
    assert t['actor_objective'] == 'clipped_dppo'
    assert t['objective_logprob_reduction'] == 'free_coordinate_mean'
    assert not t['bc_enabled'] and t['bc_batch'] == t['bc_samples_per_rollout'] == t['bc_weight'] == 0
    assert t['reward']['tracking']['source_std']['body_pos'] == .3
    assert t['reward']['tracking']['reward_std']['body_pos'] == .15
    assert c['termination']['yaw_error_rad'] is None
    assert c['termination']['non_yaw_orientation_error_rad'] == .6
    assert s['evaluation']['split'] == s['evaluation']['original_split'] == 'train'
    assert s['evaluation']['selection_policy'] == 'reward_only'
    assert not s['critic']['complete_mc']['enforce_quality_gate']
    assert s['run_control']['max_training_iterations'] == 200
    old = configuration(ROOT/'configs/closedloop/stage2_credit_server2.yaml')
    assert old['stage9']['reward']['version'].endswith('.v1')
    assert old['stage9']['bc_batch'] == 256


def test_disabled_bc_real_four_step_ppo_has_no_supervision(actor_factory, tmp_path):
    dist.init_process_group('gloo', init_method=(tmp_path/'gloo').as_uri(), rank=0, world_size=1)
    policy, rows, targets = case(actor_factory)
    optimizer = torch.optim.AdamW([p for p in policy.actor.parameters() if p.requires_grad], lr=1e-7, weight_decay=0.)
    controller = BCGradientController()
    before = copy.deepcopy(controller.state_dict())
    manifest = [dict(owner_rank=0, local_index=i, valid=True, has_free=True) for i in range(len(rows))]
    cache = RolloutTensorCache(rows, targets, 'cpu')
    try:
        result = actor_update(policy, optimizer, rows, targets, cache, manifest, controller,
            distributed=DistributedCollectives(0, 1, device='cpu'), anchor=DisabledBC(),
            generator=torch.Generator().manual_seed(4), minibatch_chains=5, microbatch=3,
            bc_enabled=False, bc_samples_per_rollout=0,
            backtracking={'max_attempts': 3, 'factor': .5},
            kl_limits={'mean_coordinate': 1., 'maximum_step_mean_coordinate': 1., 'p95_coordinate': 1.},
            kl_microbatch=3)
        assert result['optimizer_steps'] == 1
        assert result['actual_global_bc_samples'] == result['computed_global_bc_samples'] == 0
        assert controller.state_dict() == before
        assert result['steps'][0]['gradient_contributions']['all']['weighted_bc_norm'] == 0
    finally:
        cache.close()
        dist.destroy_process_group()


def test_parent_actor_adam_preserved_and_bad_asset_rejected(tmp_path):
    actor = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(actor.parameters(), lr=3e-7, weight_decay=0.)
    actor(torch.ones(1, 2)).sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    cfg = configuration(ROOT/'configs/closedloop/stage2_dppo_server2_formal.yaml')
    lr = ActorLRController(3e-7, cfg['stage9']['actor_lr_adaptation'])
    assets, kernel = {'actor': 'fixture'}, {'kernel': '15+4+1'}
    state = dict(iteration=22, rollouts_completed=22, actor_updates=87, policy_version=22)
    payload = dict(schema=SCHEMA, actor=actor.state_dict(), actor_optimizer=opt.state_dict(),
        actor_fingerprint=_model_fingerprint(actor), scheduler=lr.state_dict(), state=state,
        identity=dict(assets=assets, kernel=kernel, reward={'version': 'old'},
                      physical_configuration={'termination': {'yaw': 1.5}}))
    path = tmp_path/'parent.pt'; torch.save(payload, path)
    spec = dict(path=str(path), sha256=file_sha256(path), mode='actor_adam_new_reward')
    child = torch.nn.Linear(2, 1); child_opt = torch.optim.AdamW(child.parameters(), lr=3e-7, weight_decay=0.)
    report = initialize_actor_continuation(spec, actor=child, optimizer=child_opt,
        lr_controller=lr, assets=assets, kernel=kernel)
    assert equal_state(actor.state_dict(), child.state_dict())
    assert equal_state(opt.state_dict(), child_opt.state_dict())
    assert report['parent_progress'] == state
    assert not report['critic_restored'] and not report['old_reward_labels_reused']
    with pytest.raises(ValueError, match='assets'):
        initialize_actor_continuation(spec, actor=child, optimizer=child_opt,
            lr_controller=lr, assets={'actor': 'wrong'}, kernel=kernel)


def test_reward_only_selection_records_failure_without_blocking(monkeypatch):
    import gem.closedloop.dppo.periodic_monitor as module
    monkeypatch.setattr(module, 'paired_summary', lambda *args: {'new_failures': 1})
    base = dict(physical_failure_count=0, mean_executed_seconds=100.,
                episode_metrics=[{'activity_ratio': 1.}])
    current = dict(physical_failure_count=1, mean_executed_seconds=50.,
        episode_metrics=[{'activity_ratio': .1}], source_balanced_reward=20.,
        plan_sha256='x', actor_identity={}, evaluation_identity={'training_identity': {
            'evaluation': {'selection_policy': 'reward_only'}}})
    result = module.choose_best(current, base)
    assert result['eligible'] and result['paired_new_failures'] == 1
    assert not result['activity_retention_passed']


def test_parent_new_reward_baseline_is_allowed_only_before_new_actor_update():
    from types import SimpleNamespace
    from gem.closedloop.dppo.training import _requires_initial_evaluation
    args = SimpleNamespace(collect_only=False)
    counters = dict(iteration=22, actor_updates=87, policy_version=22)
    state = dict(counters, continuation_parent={'parent_progress': counters.copy()})
    assert _requires_initial_evaluation(args, state, False)
    state['actor_updates'] += 1
    with pytest.raises(ValueError, match='durable initial'):
        _requires_initial_evaluation(args, state, False)
