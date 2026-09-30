"""第十步全清单逐任务配对比较的 CPU 合成证据验收。

用两份不同 checkpoint 的完整 merge 发布结构验证同任务/同噪声比较；合成 episode
覆盖双方成功、新失败、失败恢复、双方失败，并让失败改变实际控制时长，确认任务
平均与控制加权奖励率不混淆。奖励直接使用记录值及原 aggregate_evaluation，无
模型执行或 GPU。篡改父计划、实现身份、checkpoint、episode、完成标记或聚合均
必须拒绝；零控制步事件惩罚保持为完整任务且连续指标缺测。所有文件在 tmp_path，
通过调用方 TemporaryDirectory 清理，不写正式训练结果或既有源码。
"""
from __future__ import annotations

import copy
import json

import pytest

from gem.closedloop.dppo.evaluation import _digest, aggregate_evaluation
from tools.eval.audit_closedloop_stage10 import sha256
from tools.eval.compare_closedloop_stage10 import compare_evaluations, main, MERGE_VERSION


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _publish(root, report):
    _write(root/'report.json', report)
    audit = {'schema': MERGE_VERSION, 'status': 'passed', 'report_sha256': sha256(root/'report.json'),
             'shards': [{'status': 'passed'}]}
    _write(root/'independent_audit.json', audit)
    _write(root/'completion.json', {'schema': MERGE_VERSION+'.completion', 'status': 'passed',
        'report_sha256': sha256(root/'report.json'), 'audit_sha256': sha256(root/'independent_audit.json')})


def _fixture(root):
    tasks = [{'task_id': f'task{index}', 'dataset': 'AIST++' if index<2 else 'Mine',
              'sample_id': str(index//2), 'split': 'val', 'seed': (42, 1729)[index%2],
              'music_start_frame': 0, 'noise_index_start': 100+index, 'full_music_seconds': 10.,
              'remaining_music_seconds': 10.} for index in range(4)]
    plan = {'requested_eval_count': 'all', 'selected_sample_count': 2, 'complete_pool_count': 2,
            'split': 'val', 'seeds': [42, 1729], 'tasks': tasks, 'task_count': 4, 'episode_seconds': .08}
    plan['plan_sha256'] = _digest(plan)
    for side, failures in [('initial', [False, False, True, True]), ('final', [False, True, False, True])]:
        directory = root/side
        directory.mkdir(parents=True)
        checkpoint = directory/'model.pt'
        checkpoint.write_bytes(side.encode())
        episodes, references = [], []
        for index, (task, failed) in enumerate(zip(tasks, failures)):
            count = 1 if failed else 4
            per_control = .06 if side=='initial' else .08
            controls = [{'dt_s': .02, 'reward': per_control-(5. if failed and step==count-1 else 0.),
                'components': {'track': {'integrated_reward': per_control, 'score': .8}},
                'raw_errors': {'joint_position_rmse_rad': .1 if side=='initial' else .09},
                'activity': {'valid': True, 'actual_activity_rad_s': 1., 'target_activity_rad_s': 1.2,
                             'gate': 1., 'intensity_score': .9}} for step in range(count)]
            reward = sum(row['reward'] for row in controls)
            decision = {'controls': controls, 'executed_control_steps': count, 'executed_physics_steps': 4*count,
                'reward_sum': reward, 'zero_step_event_reward': 0., 'rejection': None,
                'latency_seconds': .3, 'prefix_frames': 21, 'events': []}
            episode = copy.deepcopy(task)|{'transition_valid': True, 'physical_failure': failed,
                'truncated': not failed, 'end_reason': 'root_height' if failed else 'collection_limit',
                'episode_seconds_limit': .08, 'reward_sum': reward, 'decisions': [decision]}
            path = directory/'episodes'/f'{index:06d}.json'
            _write(path, episode)
            references.append({'path': str(path.relative_to(directory)), 'sha256': sha256(path),
                               'task_id': task['task_id'], 'transition_valid': True})
            episodes.append(episode)
        _write(directory/'selection.json', plan)
        files = [{'relative_path': 'frozen_eval.py', 'sha256': 'implementation-source'}]
        report = {'merge_schema': MERGE_VERSION, 'status': 'passed',
            'merge_assertions': {'exact_full_parent_task_set': True, 'duplicate_tasks': 0,
                'original_task_identity_and_noise_preserved': True, 'all_shards_independently_audited': True},
            'updates_performed': False, 'training_buffer_used': False, 'actor_gradients_unchanged': True,
            'rng_restored': True, 'networks_unchanged': {'actor': True, 'critic': True},
            'network_fingerprints_before': {'actor': side}, 'network_fingerprints_after': {'actor': side},
            'actor_identity': {'checkpoint': str(checkpoint), 'sha256': sha256(checkpoint), 'iteration': 0 if side=='initial' else 4},
            'selection_path': 'selection.json', 'plan_sha256': plan['plan_sha256'],
            'requested_task_count': 4, 'completed_episode_count': 4, 'episode_manifests': references,
            'extension_provenance': {'included_in_original_checkpoint_identity': False, 'files': files,
                'source_manifest_sha256': _digest([(row['relative_path'], row['sha256']) for row in files])},
            'episode_seconds_limit': .08, 'training_identity': {'reward': 'same', 'dataset': 'same'},
            'deterministic': False, 'aggregate': aggregate_evaluation(episodes)}
        _publish(directory, report)
    return root/'initial/report.json', root/'final/report.json'


def test_complete_task_comparison_preserves_four_failure_groups_and_metric_weights(tmp_path):
    paths = _fixture(tmp_path)
    result = compare_evaluations(*paths)
    assert result['status']=='passed' and result['paired_task_count']==4
    assert set(result['overall']['failure_pair_counts'].values())=={1}
    overall = result['overall']
    assert overall['actual_execution_aggregate']['initial']['physical_failure_count']==2
    assert overall['actual_execution_aggregate']['final']['physical_failure_count']==2
    assert overall['task_weighted_metrics']['episode_reward']['delta']['mean']==pytest.approx(.05)
    assert overall['task_weighted_metrics']['executed_seconds']['delta']['mean']==0.
    task_rate = overall['task_weighted_metrics']['reward_per_executed_second']['initial']['mean']
    exposure_rate = overall['actual_execution_aggregate']['initial']['reward_per_executed_second']
    assert task_rate != pytest.approx(exposure_rate)
    assert result['by_failure_pair']['new_failure']['task_weighted_metrics']['episode_reward']['delta']['mean']<0
    assert result['by_failure_pair']['failure_recovered']['task_weighted_metrics']['executed_seconds']['delta']['mean']>0
    assert 'survivor bias' in result['metric_semantics']['actual_execution']
    assert len(result['by_seed'])==2 and len(result['by_source'])==2


@pytest.mark.parametrize('fault', ['same_checkpoint', 'checkpoint_bytes', 'parent_plan', 'implementation',
                                  'training_identity', 'duration', 'noise', 'episode_sha', 'aggregate',
                                  'missing_completion', 'missing_episode', 'duplicate_task', 'gradient_changed'])
def test_comparison_rejects_incompatible_or_tampered_evidence(tmp_path, fault):
    initial_path, final_path = _fixture(tmp_path)
    directory = final_path.parent
    report = json.loads(final_path.read_text())
    if fault=='same_checkpoint':
        report['actor_identity'] = json.loads(initial_path.read_text())['actor_identity']
    elif fault=='checkpoint_bytes':
        (directory/'model.pt').write_bytes(b'changed model')
    elif fault=='parent_plan':
        plan_path = directory/'selection.json'
        plan = json.loads(plan_path.read_text())
        plan['changed_identity'] = True
        plan['plan_sha256'] = _digest({key:value for key,value in plan.items() if key!='plan_sha256'})
        report['plan_sha256'] = plan['plan_sha256']
        _write(plan_path, plan)
    elif fault=='implementation':
        report['extension_provenance']['files'][0]['sha256'] = 'changed'
        report['extension_provenance']['source_manifest_sha256'] = _digest([('frozen_eval.py', 'changed')])
    elif fault=='training_identity':
        report['training_identity']['reward'] = 'new_reward'
    elif fault=='duration':
        report['episode_seconds_limit'] = .1
    elif fault=='noise':
        path = directory/report['episode_manifests'][0]['path']
        episode = json.loads(path.read_text()); episode['noise_index_start'] += 1
        _write(path, episode)
        report['episode_manifests'][0]['sha256'] = sha256(path)
    elif fault=='episode_sha':
        report['episode_manifests'][0]['sha256'] = 'wrong-sha'
    elif fault=='aggregate':
        report['aggregate']['overall']['reward_sum'] += 1.
    elif fault=='missing_episode':
        (directory/report['episode_manifests'][0]['path']).unlink()
    elif fault=='duplicate_task':
        report['episode_manifests'][1] = report['episode_manifests'][0]
    elif fault=='gradient_changed':
        report['actor_gradients_unchanged'] = False
    _publish(directory, report)
    if fault=='missing_completion':
        (directory/'completion.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        compare_evaluations(initial_path, final_path)


def test_zero_control_event_remains_a_paired_task_and_is_not_missing_reward(tmp_path):
    initial_path, final_path = _fixture(tmp_path)
    directory = final_path.parent
    report = json.loads(final_path.read_text())
    path = directory/report['episode_manifests'][1]['path']
    episode = json.loads(path.read_text())
    episode['reward_sum'] = -5.
    episode['decisions'][0].update(controls=[], executed_control_steps=0, executed_physics_steps=0,
                                   reward_sum=-5., zero_step_event_reward=-5.)
    _write(path, episode)
    report['episode_manifests'][1]['sha256'] = sha256(path)
    episodes = [json.loads((directory/row['path']).read_text()) for row in report['episode_manifests']]
    report['aggregate'] = aggregate_evaluation(episodes)
    _publish(directory, report)
    result = compare_evaluations(initial_path, final_path)
    assert result['paired_task_count']==4
    assert result['pairs'][1]['final']['episode_reward']==-5.
    assert result['pairs'][1]['final']['reward_per_executed_second'] is None
    assert result['pairs'][1]['final']['component_integral.track']==0.
    assert result['pairs'][1]['delta']['raw_error.joint_position_rmse_rad'] is None


def test_comparison_cli_publishes_new_report_without_overwriting(tmp_path):
    paths = _fixture(tmp_path/'inputs')
    output = tmp_path/'comparison.json'
    args = ['--initial-report', str(paths[0]), '--final-report', str(paths[1]), '--output', str(output)]
    assert main(args)==0
    assert json.loads(output.read_text())['status']=='passed'
    with pytest.raises(FileExistsError):
        main(args)
