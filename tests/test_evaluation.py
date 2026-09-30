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
        for name, weight, score in [('track', 2.5, 1.), ('music', 2., .3), ('stable', 1., 1.), ('alive', .5, 1.),
                                    ('cmd', -.15, 0.), ('torque', -.1, 0.), ('contact', -.2, 0.), ('joint_limit', -.5, 0.)]:
            components[name] = {'score': score, 'gate': 1., 'activity_gate': 1., 'weight': weight,
                                'weighted_rate': score*weight, 'integrated_reward': .02*score*weight,
                                'valid': True, 'raw': {'joint_position_rmse_rad': .05} if name=='track' else {'root_height_error_m': .01} if name=='stable' else {}}
        base = sum(item['integrated_reward'] for item in components.values())
        details = [{'version': 'stage9.execution_reward.v2', 'tick': 600+12*(self.steps+i+1), 'dt_s': .02,
                    'transition_valid': True, 'reward': base, 'components': copy.deepcopy(components),
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
                    'generated': {'prefix_frames': 21}, 'terminal_snapshot': {'terminated': failed},
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
    episodes = [json.loads((tmp_path/'evaluation'/item['path']).read_text()) for item in report['episode_manifests']]
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


def _stage10_archive(root):
    """构建三个已接受真实格式小转移及一次续训，用于独立产物审计故障测试。"""
    import hashlib
    from gem.closedloop.dppo.buffer import UpperTransition
    from gem.closedloop.dppo.returns import compute_gae
    from tools.eval.audit_closedloop_stage10 import sha256
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    sources = ['AIST++', 'AIOZ-GDANCE', 'FineDance', 'Mine']
    catalog = {'manifest_sha256': {f'{source}/{split}': f'{source}:{split}' for source in sources for split in ['train', 'val', 'test']},
               'sample_counts': {split: {source: 1 for source in sources} for split in ['train', 'val', 'test']}}
    records = [{'dataset': source, 'split': split, 'sample_id': '0', 'group_id': f'{source}:{split}', 'num_frames': 300,
                'manifest_sha256': f'{source}:{split}', 'motion_payload_sha256': f'm:{source}:{split}',
                'music_feature_sha256': f'f:{source}:{split}', 'source_motion_sha256': f'original:{source}:{split}',
                'audio_sha256': f'a:{source}:{split}', 'audio_verified': True}
               for source in sources for split in ['train', 'val', 'test']]
    content = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    data = {'status': 'passed', 'complete_manifest_scan': True, 'identity': catalog, 'sample_count': len(records),
            'require_audio': True, 'data_content_sha256': content, 'records': records}
    contract = {'denoising_steps': 2, 'eta': .1, 'std_floor': .001, 'guidance_scale': 2.5,
                'gamma_upper': .99, 'lambda_upper': .95, 'kl_stop_joint': .02}
    identity = {'stage10': 'genmo.closedloop.stage10.v1', 'dataset': catalog,
                'data_content_sha256': content, 'training_contract': contract}
    write(root/'run.json', {'schema': 'genmo.closedloop.stage10.run.v1', 'mode': 'train', 'identity': identity})
    limits = {'accepted_iterations': 4, 'optimizer_attempts': 12, 'generations': 100, 'control_steps': 1000, 'physics_steps': 4000}
    def budget(index):
        used = {'accepted_iterations': index, 'optimizer_attempts': index, 'generations': index, 'control_steps': 2*index, 'physics_steps': 8*index}
        return {'schema': 'genmo.closedloop.stage10.budget.v1', 'used': used, 'limits': limits,
                'phases': {'train': {'generations': index, 'control_steps': 2*index, 'physics_steps': 8*index},
                           'update': {'accepted_iterations': index, 'optimizer_attempts': index}}}
    session_refs, checkpoints = {'s1': [], 's2': []}, {}
    for index in range(1, 4):
        session = 's1' if index<=2 else 's2'
        directory = root/'sessions'/session/'iterations'/f'{index:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        context = {'music_features': torch.zeros(1, 120, 35), 'music_valid': torch.ones(1, 120, dtype=torch.bool),
            'proprio_history': torch.zeros(1, 50, 48), 'proprio_history_valid': torch.ones(1, 50, dtype=torch.bool),
            'proprio_history_times': (torch.arange(50, dtype=torch.float64)/50)[None],
            'known_qpos30': torch.zeros(1, 120, 30), 'known_qpos30_mask': torch.zeros(1, 120, 30, dtype=torch.bool),
            'future_valid': torch.ones(1, 120, dtype=torch.bool),
            'future_times': (torch.arange(120, dtype=torch.float64)/30)[None], 'decision_time': torch.tensor([1.], dtype=torch.float64)}
        item = UpperTransition(identity={'run_id': 'r', 'backend_session_id': f'worker:{session}', 'episode_id': f'e:{index}',
                  'decision_id': 0, 'policy_version': index-1, 'env_id': 'env:0', 'request_id': f'request:{index}',
                  'plan_id': f'plan:{index}', 'parent_plan_id': None}, context=context, next_context=context,
            chain=torch.zeros(3, 120, 30), old_log_prob=torch.zeros(2, dtype=torch.float64), free_mask=torch.ones(120, 30, dtype=torch.bool),
            rewards=torch.tensor([.1, .2], dtype=torch.float64), old_value=.5, next_value=.7,
            control_tick_begin=600, control_tick_end=624, executed_control_steps=2, executed_physics_steps=8,
            truncated=True, metadata={'training_task': {'dataset': 'AIST++', 'sample_id': '0', 'split': 'train',
                'music_start_frame': 0, 'manifest_sha256': 'AIST++:train'}, 'sampler_trace': {'kernel_config': {
                'steps': 2, 'eta': .1, 'std_floor': .001, 'guidance_scale': 2.5,
                'log_prob_reduction': 'joint_sum_fp64', 'cfg_policy': 'music_only_shared_history_and_prefix'}}})
        record_path = directory/'rollout/chunk_000000/transition_000000000.pt'
        record_path.parent.mkdir(parents=True)
        torch.save(item, record_path)
        record = {'path': record_path.name, 'sha256': sha256(record_path), 'size_bytes': record_path.stat().st_size,
                  'identity': {key: item.identity[key] for key in ('run_id', 'backend_session_id', 'episode_id', 'decision_id', 'policy_version')},
                  'executed_control_steps': 2, 'executed_physics_steps': 8}
        chunk_path = record_path.parent/'manifest.json'
        write(chunk_path, {'schema': 'genmo.closedloop.stage10.rollout_chunk.v1', 'policy_version': index-1,
                           'records': [record], 'record_count': 1})
        manifest_path = directory/'rollout/manifest.json'
        write(manifest_path, {'schema': 'genmo.closedloop.stage10.rollout.v1', 'complete': True, 'policy_version': index-1,
            'transition_count': 1, 'executed_control_steps': 2, 'executed_physics_steps': 8,
            'chunks': [{'path': 'chunk_000000/manifest.json', 'sha256': sha256(chunk_path), 'record_count': 1}]})
        targets = compute_gae([item.rewards], [.5], [.7], [2], [True], [False])
        targets.update(old_values=torch.tensor([.5], dtype=torch.float64), next_values=torch.tensor([.7], dtype=torch.float64))
        torch.save(targets, directory/'fixed_targets.pt')
        calibration = {'accepted_updates': 1, 'base_state_restored_per_candidate': True, 'gradients_reused': True,
                       'attempt_count': 1, 'selected_lr': 1e-8, 'candidates': [{'lr': 1e-8, 'kl': {'mean_joint_kl': .001},
                       'parameter_change': {'changed_count': 1}, 'accepted': True}]}
        checkpoint = root/'checkpoints'/f'{index:06d}.pt'
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save({'version': 'genmo.closedloop.stage9.full_state.v1', 'actor': {'weight': torch.tensor([index])},
            'critic': {}, 'actor_optimizer': {'state': {}, 'param_groups': [{'lr': 1e-8}]}, 'critic_optimizer': {},
            'state': {'iteration': index, 'policy_version': index, 'actor_updates': index, 'critic_updates': index*20,
                      'buffer_size': 0, 'pending_plan': False, 'selected_actor_lr': 1e-8, 'session_id': session, 'budget': budget(index)},
            'identity': identity, 'rng': {}, 'samplers': {'music': {'split': 'train', 'catalog_identity': catalog}},
            'config': {}, 'optimizer_layout': {}, 'restore_environment': 'new_worker_session_and_reset'}, checkpoint)
        checkpoints[index] = checkpoint
        summary_path = directory/'summary.json'
        summary = {'iteration': index, 'status': 'accepted', 'policy_version_before': index-1, 'policy_version_after': index,
            'collected_upper_transitions': 1, 'collection': {'full_train_pool': True, 'control_steps': 2},
            'rollout_manifest': str(manifest_path.relative_to(root)), 'targets_path': str((directory/'fixed_targets.pt').relative_to(root)),
            'probability_check': {'passed': True, 'max_abs_log_probability_difference': 0., 'max_abs_ratio_minus_one': 0.,
                                  'max_abs_independent_gaussian_difference': 0.},
            'actor': {'parameters_changed': True, 'critic_unchanged': True, 'optimizer_steps': 1, 'ppo_only_gradient_norm': 1., 'lr_calibration': calibration},
            'critic': {'parameters_changed': True, 'actor_unchanged': True}, 'kl': {'mean_joint_kl': .001}, 'kl_limit': .02,
            'gmt_frozen': {'policy_unchanged': True, 'runtime_parameters_unchanged': True,
                'execution_journal': {'backend_session_id': f'worker:{session}', 'executed_seq': 15*index, 'acked_seq': 15*index}},
            'source_unchanged': {'unchanged': True}, 'budget': budget(index), 'checkpoint': str(checkpoint.relative_to(root))}
        write(summary_path, summary)
        session_refs[session].append(str(summary_path.relative_to(root)))
        publication = {'schema': 'genmo.closedloop.stage10.checkpoint_publication.v1', 'iteration': index,
            'path': str(checkpoint.relative_to(root)), 'sha256': sha256(checkpoint), 'size_bytes': checkpoint.stat().st_size,
            'session_id': session, 'metadata': {'iteration_summary': str(summary_path.relative_to(root))}, 'budget': budget(index)}
        publication_path = root/'checkpoints/publications'/f'{index:09d}-{session}.json'
        write(publication_path, publication)
        write(root/'latest.json', {**publication, 'publication': str(publication_path.relative_to(root))})
    for session in ('s1', 's2'):
        directory = root/'sessions'/session
        write(directory/'data_audit.json', data)
        resume = None if session=='s1' else {'training_resume': True, 'checkpoint': str(checkpoints[2]), 'sha256': sha256(checkpoints[2]),
            'restored_full_state': True, 'old_buffer_discarded': True, 'initial_iteration': 2, 'new_backend_session_id': 'worker:s2'}
        write(directory/'summary.json', {'status': 'passed', 'exit_code': 0, 'session_id': session, 'mode': 'train',
            'initial_iteration': 0 if session=='s1' else 2, 'final_iteration': 2 if session=='s1' else 3,
            'iterations': session_refs[session], 'evaluations': [], 'resume': resume,
            'data_audit': str((directory/'data_audit.json').relative_to(root)),
            'source_unchanged': {'unchanged': True}, 'original_assets_unchanged': True,
            'worker_shutdown': {'gmt': {'policy_unchanged': True, 'runtime_parameters_unchanged': True, 'process_exit_code': 0}},
            'budget': budget(2 if session=='s1' else 3)})
        write(directory/'completion.json', {'schema': 'genmo.closedloop.stage10.session_completion.v1',
            'summary_sha256': sha256(directory/'summary.json'), 'status': 'passed', 'exit_code': 0})
    write(root/'budget.json', budget(3))
    return root


def test_stage10_audit_requires_multiple_fresh_policies_and_resume_then_real_update(tmp_path):
    from tools.eval.audit_closedloop_stage10 import audit_run
    root = _stage10_archive(tmp_path/'run')
    report = audit_run(root)
    assert report['status']=='passed', report
    checks = {check['name']: check for check in report['checks']}
    assert checks['multiple_fresh_policy_iterations']['details']['fresh_policy_versions']==[0, 1, 2]
    assert checks['resume_then_optimize']['details'][0]['final_iteration']==3
    assert checks['iteration:1']['details']['fixed_targets']['max_abs_differences']['returns'] < 1e-12


@pytest.mark.parametrize('key', ['run_id', 'backend_session_id', 'episode_id', 'decision_id', 'policy_version'])
@pytest.mark.parametrize('mutation', ['missing', 'changed'])
def test_rollout_published_identity_keys_cannot_be_missing_or_changed(tmp_path, key, mutation):
    from tools.eval.audit_closedloop_stage10 import audit_run, sha256
    root = _stage10_archive(tmp_path/'run')
    directory = root/'sessions/s1/iterations/000001/rollout'
    chunk_path = directory/'chunk_000000/manifest.json'
    chunk = json.loads(chunk_path.read_text())
    if mutation=='missing':
        del chunk['records'][0]['identity'][key]
    else:
        chunk['records'][0]['identity'][key] = 'changed_identity'
    chunk_path.write_text(json.dumps(chunk))
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['chunks'][0]['sha256'] = sha256(chunk_path)
    manifest_path.write_text(json.dumps(manifest))
    result = audit_run(root)
    assert result['status']=='failed'
    first = next(check for check in result['checks'] if check['name']=='iteration:1')
    assert first['status']=='failed' and 'identity' in first['error']


@pytest.mark.parametrize('fault', ['old_value', 'probability', 'kl', 'budget', 'resume_without_update', 'same_worker', 'wrong_policy', 'checkpoint_bytes', 'ledger_rollback', 'malformed_session'])
def test_stage10_audit_rejects_drift_and_false_resume_proofs(tmp_path, fault):
    from tools.eval.audit_closedloop_stage10 import audit_run
    root = _stage10_archive(tmp_path/'run')
    iteration = root/'sessions/s2/iterations/000003/summary.json'
    if fault=='old_value':
        path = iteration.parent/'fixed_targets.pt'
        values = torch.load(path, weights_only=False)
        values['old_values'][0] += 1.
        torch.save(values, path)
    elif fault=='checkpoint_bytes':
        publication = json.loads(next((root/'checkpoints/publications').glob('000000001-*.json')).read_text())
        with (root/publication['path']).open('ab') as stream:
            stream.write(b'changed immutable checkpoint')
    elif fault=='ledger_rollback':
        path = root/'budget.json'
        value = json.loads(path.read_text())
        value['used']['optimizer_attempts'] = 2
        value['phases']['update']['optimizer_attempts'] = 2
        path.write_text(json.dumps(value))
    elif fault=='malformed_session':
        (root/'sessions/s2/summary.json').write_text('{broken JSON')
    elif fault in {'resume_without_update', 'same_worker'}:
        path = root/'sessions/s2/summary.json'
        value = json.loads(path.read_text())
        if fault=='resume_without_update':
            value['final_iteration'] = value['initial_iteration']
        else:
            value['resume']['new_backend_session_id'] = 'worker:s1'
        path.write_text(json.dumps(value))
    else:
        value = json.loads(iteration.read_text())
        if fault=='probability':
            value['probability_check']['max_abs_ratio_minus_one'] = .1
        elif fault=='kl':
            value['kl']['mean_joint_kl'] = 1.
        elif fault=='budget':
            value['budget']['used']['optimizer_attempts'] = 0
        else:
            value['policy_version_before'] = 0
        iteration.write_text(json.dumps(value))
    result = audit_run(root, allow_incomplete=True)
    assert result['status']=='failed'
    assert result['failed_checks']>0


@pytest.mark.parametrize('fault', ['missing', 'sha256', 'status', 'exit_code', 'schema', 'summary_changed'])
def test_stage10_session_completion_marker_required_and_bound(tmp_path, fault):
    from tools.eval.audit_closedloop_stage10 import audit_run
    root = _stage10_archive(tmp_path/'run')
    marker = root/'sessions/s2/completion.json'
    if fault=='missing':
        marker.unlink()
    elif fault=='summary_changed':
        path = marker.parent/'summary.json'
        value = json.loads(path.read_text())
        value['unbound_new_field'] = 'changed after completion'
        path.write_text(json.dumps(value))
    else:
        value = json.loads(marker.read_text())
        key, replacement = {'sha256': ('summary_sha256', 'wrong-sha'), 'status': ('status', 'failed'),
                            'exit_code': ('exit_code', 1), 'schema': ('schema', 'unknown')}[fault]
        value[key] = replacement
        marker.write_text(json.dumps(value))
    strict = audit_run(root)
    assert strict['status']=='failed'
    session_check = next(item for item in strict['checks'] if item['name']=='session:s2')
    assert session_check['status']=='failed'
    if fault=='missing':
        incomplete = audit_run(root, allow_incomplete=True)
        assert incomplete['status']=='incomplete'
        assert next(item for item in incomplete['checks'] if item['name']=='session:s2')['status']=='not_run'


def test_stage10_evaluation_audit_recomputes_episode_aggregate(tmp_path):
    from tools.eval.audit_closedloop_stage10 import audit_evaluation
    catalog = Catalog(tmp_path)
    catalog.identity['sample_counts'] = {split: {source: len(samples) for source, samples in groups.items()}
                                        for split, groups in catalog.samples.items()}
    loaded = policy()
    plan = build_evaluation_tasks(catalog, eval_count='all', seeds=(1, 2), episode_seconds=3.)
    evaluate_policy(Env(loaded), loaded, plan, tmp_path/'evaluation', catalog=catalog,
                    actor_identity={'checkpoint': '/explicit/loaded/stage10.pt', 'sha256': 'confirmed_by_loader'})
    lookup = {(source, 'val', str(sample['row']['sample_id'])): {'manifest_sha256': sample['manifest_sha256']}
              for source, samples in catalog.samples['val'].items() for sample in samples}
    result = audit_evaluation(tmp_path.resolve(), tmp_path/'evaluation/report.json', lookup, catalog.identity)
    assert result['episodes']==12 and result['distinct_paired_samples']==6
    assert result['unique_audio_count']==6
