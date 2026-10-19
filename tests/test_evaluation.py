"""第十步独立评估计划、执行统计和只读网络边界的 CPU 验收。

完整的合成 val/test catalog 包含多个来源和不同配对样本，执行替身每次返回实际控制
区间级奖励证据，使测试可以验证评估超过原先四次决策、按多 seed 完成歌曲/显式
窗口、保存并独立重算逐来源统计。替身不代表物理质量，测试不调用 GPU 或优化器。
故障用例确认训练 split、策略对象不一致、参数污染和基础设施异常不能成为通过
报告；临时 JSON 全部位于 pytest tmp_path，由统一 TemporaryDirectory 清理。
session 完成标记覆盖缺失、摘要 SHA 篡改、错误状态与退出码，确保仅有摘要文件的
未完成会话不被当作成功；允许不完整验收时也必须明确返回 incomplete。
"""
from __future__ import annotations

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.evaluation import build_evaluation_tasks, evaluate_policy, aggregate_evaluation


class Catalog:
    def __init__(self, root):
        self.data_root = root
        self.identity = {'version': 'fake_full_catalog', 'all_manifest_sha256': 'full12', 'train_eval_groups_disjoint': True}
        self.samples = {}
        for split in ('train', 'val', 'test'):
            self.samples[split] = {}
            for source in ('AIST++', 'Mine'):
                self.samples[split][source] = tuple({'dataset': source, 'split': split, 'group_id': f'{split}:{source}:{i}',
                    'manifest_sha256': f'{split}:{source}', 'row': {'sample_id': str(i), 'split': split,
                    'num_frames': 150+i*30, 'fps': 30, 'source_audio_sha256': f'{split}:{source}:{i}'}} for i in range(3))
    def load_music(self, sample):
        return np.zeros((sample['row']['num_frames'], 35), dtype=np.float32)


class Env:
    def __init__(self, policy, *, failure=False, rejection=False, fault=None):
        self.policy = policy
        self.config = {'stage9': {'episode_seconds': 30.}}
        self.comparison_noise_index = None
        self.resets, self.seen_weights = [], []
        self.failure, self.rejection, self.fault = failure, rejection, fault
    def reset_task(self, sample, music, *, seed, phase, music_start_frame):
        self.resets.append((sample['dataset'], sample['row']['sample_id'], seed, phase, music_start_frame))
        self.sample = sample
        self.steps, self.decisions = 0, 0
        self.full_steps = ((len(music)-music_start_frame)*50)//30
        self.limit = min(self.full_steps, int(self.config['stage9']['episode_seconds']*50))
        self.noise_start = self.comparison_noise_index
        random.random(); np.random.rand(); torch.rand(())
    def step(self, deterministic=False):
        if self.fault == 'infra' and self.decisions == 2:
            raise OSError('simulated transport fault')
        self.seen_weights.append(self.policy.actor.weight.detach().clone())
        if self.fault == 'weights':
            self.policy.actor.weight.add_(1.)
        count = min(25, self.limit-self.steps)
        components = {}
        for name, weight, score in [('track', 2.5, 1.), ('music', 0., 0.), ('stable', 0., 0.), ('alive', .5, 1.),
                                    ('cmd', -.15, 0.), ('torque', -.1, 0.), ('contact', -.2, 0.), ('joint_limit', -.5, 0.)]:
            components[name] = {'score': score, 'gate': 1., 'activity_gate': 1., 'weight': weight,
                                'weighted_rate': score*weight, 'integrated_reward': .02*score*weight,
                                'valid': True, 'raw': {'joint_position_rmse_rad': .05} if name=='track' else {'root_height_error_m': .01} if name=='stable' else {}}
        base = sum(item['integrated_reward'] for item in components.values())
        from tests.closedloop.dppo.test_data_learning import actual_step
        from gem.closedloop.dppo.joint_limits import executed_boundaries,generated_boundaries
        boundary_row=actual_step();names=boundary_row['physical_diagnostics']['joint_names']
        limits=executed_boundaries(boundary_row,np.zeros((2,2,21),dtype=np.int64))[0]
        limits.update(actual_cost=0.,reference_cost=0.)
        generated_limits=generated_boundaries(np.zeros((60,28)),np.ones((60,21),bool),
            dict(joint_names=names,joint_hard_limits_rad=np.tile([-2,2],(21,1)),
                 joint_soft_limits_rad=np.tile([-1.8,1.8],(21,1))),names)
        details = [{'version': 'stage2.gmt8_reference_limits.v1', 'tick': 600+12*(self.steps+i+1), 'dt_s': .02,
                    'transition_valid': True, 'reward': base, 'components': copy.deepcopy(components),
                    'joint_limit_diagnostics':copy.deepcopy(limits),'diagnostics':{'physical_state':{'root_height_error_m':.01}},
                    'activity': {'valid': True, 'actual_activity_rad_s': 1., 'target_activity_rad_s': 1.,
                                 'gate': 1., 'intensity_score': 1., 'window_count': min(25, self.steps+i+1),
                                 'window_complete': self.steps+i+1>=25}} for i in range(count)]
        rewards = [base]*count
        penalty = -.5 if self.rejection and self.decisions==0 else 0.
        self.steps += count
        self.decisions += 1
        terminal = self.steps >= self.limit
        failed = self.failure and terminal
        if failed:
            penalty -= 5.
        if rewards:
            rewards[-1] += penalty
        return {'rewards': rewards, 'executed_control_steps': count, 'executed_physics_steps': count*4,
                'terminated': terminal and (failed or self.limit==self.full_steps),
                'truncated': terminal and self.limit<self.full_steps and not failed,
                'reason': 'root_height' if failed else 'music_end' if terminal and self.limit==self.full_steps else 'collection_limit' if terminal else None,
                'metadata': {'reward_details': details, 'event_reward': penalty if count==0 else 0., 'event_penalty_total': penalty,
                    'generated': {'prefix_frames': 21,'generation_horizon_frames':60,'joint_limit_diagnostics':generated_limits},
                    'terminal_snapshot': {'terminated': failed},
                    'rejection': {'policy_penalty': True, 'code': 'invalid_qpos'} if self.rejection and self.decisions==1 else None,
                    'latency_seconds': .31, 'events': [], 'raw_sample_path': 'raw/sample.pt'}}


def policy():
    actor = torch.nn.Linear(2, 2)
    actor.weight.grad = torch.ones_like(actor.weight)
    return SimpleNamespace(actor=actor)


def test_plan_uses_complete_eval_pool_deterministically_and_crosses_all_seeds(tmp_path):
    catalog = Catalog(tmp_path)
    first = build_evaluation_tasks(catalog, eval_count=4, seeds=(11, 22), selection_seed=7)
    second = build_evaluation_tasks(catalog, eval_count=4, seeds=(11, 22), selection_seed=7)
    assert first == second
    assert first['complete_pool_count'] == 6 and first['selected_sample_count'] == 4 and len(first['tasks']) == 8
    assert {task['dataset'] for task in first['tasks']} == {'AIST++', 'Mine'}
    assert len({task['task_id'] for task in first['tasks']}) == 8
    assert len({task['noise_index_start'] for task in first['tasks']}) == 8
    all_tasks = build_evaluation_tasks(catalog, split='test', eval_count='all', seeds=(11, 22))
    assert len(all_tasks['tasks']) == 12
    assert all(task['sample']['row']['split']=='test' for task in all_tasks['tasks'])
    assert not all_tasks['pretraining_unseen_claim'] and not all_tasks['training_buffer_allowed']


@pytest.mark.parametrize('kwargs', [{'split': 'train'}, {'eval_count': 7}, {'eval_count': 0}, {'eval_count': True},
                                  {'seeds': (1, 1)}, {'seeds': ()}, {'start_mode': 'center'}, {'episode_seconds': 0}])
def test_invalid_eval_plan_settings_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_evaluation_tasks(Catalog(tmp_path), **kwargs)


def test_full_music_evaluation_uses_passed_policy_multisource_multiseed_and_recomputes(tmp_path):
    catalog, loaded = Catalog(tmp_path), policy()
    env = Env(loaded)
    critic = torch.nn.Linear(3, 1)
    plan = build_evaluation_tasks(catalog, eval_count=2, seeds=(11, 22))
    before = copy.deepcopy(loaded.actor.state_dict())
    gradient = loaded.actor.weight.grad.clone()
    report = evaluate_policy(env, loaded, plan, tmp_path/'evaluation', catalog=catalog,
                             actor_identity={'checkpoint_sha256': 'actual_loaded_weights'}, frozen_modules={'critic': critic})
    assert report['status'] == 'passed' and report['completed_episode_count'] == 4
    assert report['actor_identity']['checkpoint_sha256'] == 'actual_loaded_weights'
    assert report['networks_unchanged'] == {'actor': True, 'critic': True}
    assert not report['updates_performed'] and not report['training_buffer_used']
    assert loaded.actor.training and critic.training
    torch.testing.assert_close(loaded.actor.weight.grad, gradient, atol=0, rtol=0)
    for key, value in loaded.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert all(torch.equal(value, before['weight']) for value in env.seen_weights)
    from gem.closedloop.dppo.compact_evidence import read_episode
    episodes = [read_episode(tmp_path/'evaluation'/item['path']) for item in report['episode_manifests']]
    assert all(len(episode['decisions']) > 4 for episode in episodes)
    assert all(episode['end_reason']=='music_end' for episode in episodes)
    assert aggregate_evaluation(episodes) == report['aggregate']
    assert report['aggregate']['overall']['music_completion_count'] == 4
    assert report['aggregate']['overall']['raw_errors']['joint_position_rmse_rad']['mean'] == pytest.approx(.05)
    assert set(report['aggregate']['by_source']) == {'AIST++', 'Mine'}
    assert set(report['aggregate']['by_seed']) == {'11', '22'}
    assert all(reset[3]=='evaluation' for reset in env.resets)
    assert env.config['stage9']['episode_seconds']==30 and env.comparison_noise_index is None


def test_explicit_window_center_and_administrative_truncation_are_recorded(tmp_path):
    catalog, loaded = Catalog(tmp_path), policy()
    plan = build_evaluation_tasks(catalog, eval_count=2, seeds=(11,), start_mode='center', episode_seconds=3.)
    env = Env(loaded)
    report = evaluate_policy(env, loaded, plan, tmp_path/'evaluation', catalog=catalog)
    assert report['aggregate']['overall']['administrative_truncation_count'] == 2
    assert report['aggregate']['overall']['music_completion_count'] == 0
    assert report['aggregate']['overall']['executed_seconds'] == pytest.approx(6.)
    assert all(reset[4]>0 for reset in env.resets)


def test_events_actual_failure_and_finite_reference_rejection_are_separate(tmp_path):
    catalog, loaded = Catalog(tmp_path), policy()
    plan = build_evaluation_tasks(catalog, eval_count=1, seeds=(11,))
    report = evaluate_policy(Env(loaded, failure=True, rejection=True), loaded, plan, tmp_path/'evaluation', catalog=catalog)
    overall = report['aggregate']['overall']
    assert overall['physical_failure_count']==1 and overall['finite_invalid_reference_count']==1
    assert overall['infrastructure_failure_count']==0 and overall['rejection_count']==1
    continuous = sum(item['integrated_reward_sum'] for item in overall['components'].values())
    assert overall['reward_sum']==pytest.approx(continuous-5.-.5)


def test_infrastructure_fault_is_invalid_and_reported_without_policy_failure(tmp_path):
    catalog, loaded = Catalog(tmp_path), policy()
    plan = build_evaluation_tasks(catalog, eval_count=2, seeds=(11,))
    with pytest.raises(OSError, match='transport') as caught:
        evaluate_policy(Env(loaded, fault='infra'), loaded, plan, tmp_path/'evaluation', catalog=catalog)
    report = caught.value.evaluation_report
    assert report['status']=='failed' and report['completed_episode_count']==1
    assert report['aggregate']['overall']['physical_failure_count']==0
    assert report['aggregate']['overall']['infrastructure_failure_count']==1
    assert report['aggregate']['overall']['reward_sum']==0
    assert (tmp_path/'evaluation/report.json').is_file()


def test_changed_actor_is_caught_and_cannot_publish_passed_evaluation(tmp_path):
    catalog, loaded = Catalog(tmp_path), policy()
    plan = build_evaluation_tasks(catalog, eval_count=1, seeds=(11,))
    with pytest.raises(RuntimeError, match='changed model') as caught:
        evaluate_policy(Env(loaded, fault='weights'), loaded, plan, tmp_path/'evaluation', catalog=catalog)
    assert not caught.value.evaluation_report['networks_unchanged']['actor']


def test_another_policy_object_or_tampered_catalog_plan_is_rejected(tmp_path):
    catalog, loaded = Catalog(tmp_path), policy()
    plan = build_evaluation_tasks(catalog, eval_count=1, seeds=(11,))
    with pytest.raises(ValueError, match='explicitly supplied'):
        evaluate_policy(Env(policy()), loaded, plan, tmp_path/'wrong', catalog=catalog)
    altered = copy.deepcopy(plan)
    altered['tasks'][0]['seed'] += 1
    with pytest.raises(ValueError, match='identity mismatch'):
        evaluate_policy(Env(loaded), loaded, altered, tmp_path/'tampered', catalog=catalog)


def test_evaluation_restores_rng_and_rejects_overwriting_existing_evidence(tmp_path):
    catalog, loaded = Catalog(tmp_path), policy()
    plan = build_evaluation_tasks(catalog, eval_count=1, seeds=(11,))
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    evaluate_policy(Env(loaded), loaded, plan, tmp_path/'evaluation', catalog=catalog)
    assert random.getstate()==before[0]
    after = np.random.get_state()
    assert after[0]==before[1][0] and after[2:]==before[1][2:]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    with pytest.raises(FileExistsError):
        evaluate_policy(Env(loaded), loaded, plan, tmp_path/'evaluation', catalog=catalog)
