"""精简训练曲线的统计语义、真实事件文件与历史重建验收。

使用不均衡 rank 时长、冻结去噪步骤、同曲多种子、零执行失败和不同来源误差，
检查分母与来源权重、原始平方误差到 RMS 的变换、行政结束不算失败、配对区间
不污染 RNG、缺测不补零。真实 TensorBoard 事件在 pytest 临时目录中读回，检查
一个任务只发现一个 run、训练/验证同轮奖励不混合，稀疏 MC 保持真实记录频率。
历史重建用带 SHA 的完整小报告验证 0..N 横轴、旧日志保留以及缺轮/篡改拒绝；
不启动仿真、GPU或正式训练，不在生产运行目录生成测试产物。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

from gem.closedloop.dppo.tensorboard_metrics import (
    COMPONENT_TAGS, EVENT_DEFINITIONS, ERROR_TAGS, SOURCES, DashboardWriter, checked_scalars,
    evaluation_scalars, evaluation_view, mc_scalars, training_scalars,
)
from tools.eval.export_stage10_tensorboard import export_run


def training_record(step=1):
    collections = []
    for duration, rate, failures in [(1., 1., 0), (9., 10., 2)]:
        by_source = {source: dict(executed_seconds=duration, reward_sum=duration*rate) for source in SOURCES}
        collections.append(dict(by_source=by_source, executed_seconds=4*duration,
                                control_steps=int(4*duration*50), physical_failures=failures))
    actor_steps = [dict(internal_sample_visits=count, bc_global_samples=bc, ppo_loss=loss,
                        bc_loss=loss, total_gradient_norm=2., learning_rate=1e-7,
                        gradient_contributions={'all': dict(ppo_norm=2., weighted_bc_norm=.2)})
                   for count, bc, loss in [(1, 3, 1.), (3, 1, 3.)]]
    return dict(mode='dppo_bc', state={'iteration': step}, collection_by_rank=collections,
        actor={'steps': actor_steps}, critic={'before': {'mse': 9., 'explained_variance': .2}, 'after': {'mse': 1.}},
        trainable4_kl={ 'maximum_step_mean_coordinate_kl': .2,
            'per_denoising_step': [dict(step=i, mean_coordinate_kl={'mean': .2},
                                       objective_clip_fraction=.1) for i in range(15,19)]},
        wall_seconds=20., time_utc='2026-10-16T00:00:00+00:00')


def evaluation_report(iteration=0, gain=0., error=.01, activity=1.):
    tasks, episodes = [], []
    for source in SOURCES:
        for music in range(2):
            for seed in (42, 1729):
                task_id = f'{source}:{music}:{seed}'
                tasks.append(dict(task_id=task_id, sample={'group_id': f'music{music}'}))
                control = dict(dt_s=.02, activity={'valid': True, 'actual_activity_rad_s': activity},
                    raw_errors={key: error for key in ERROR_TAGS},
                    components={name: {'integrated_reward': .02, 'weight': 1.} for name in COMPONENT_TAGS})
                episodes.append(dict(task_id=task_id, dataset=source, transition_valid=True,
                    decisions=[{'controls': [control, copy.deepcopy(control)]}], reward_sum=(2.+gain)*.04,
                    physical_failure=False, end_reason='physical_safety_limit',
                    motion_diagnostic={'music_alignment': .7}))
    return dict(status='passed', iteration=iteration, plan={'tasks': tasks}, plan_sha256='fixed_plan',
        evaluation_identity={'physics': 'fixed'}, episodes=episodes,
        aggregate={'overall': {'executed_seconds': len(tasks)*.04, 'rejection_count': 0}},
        stage1_retention={'source_balanced_loss': 4.-gain})


def test_compact_evaluation_preserves_dashboard_values_and_checks_sources(tmp_path):
    from gem.closedloop.dppo.compact_evidence import write_episode, summary_only
    from gem.closedloop.dppo.evaluation import _publish_json
    original=evaluation_report()
    compact=summary_only(original)
    compact['source_episode_manifests']=[]
    for i,episode in enumerate(original['episodes']):
        path=tmp_path/f'{i:06d}.json'
        write_episode(path,episode,publish_json=_publish_json)
        compact['source_episode_manifests'].append(dict(path=str(path),task_id=episode['task_id'],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    expected=evaluation_scalars(original)
    assert evaluation_scalars(compact)==expected
    assert evaluation_scalars(compact,compact)==expected
    reference=compact['source_episode_manifests'][0]
    Path(reference['path']).write_text(Path(reference['path']).read_text()+'\n')
    with pytest.raises(ValueError,match='JSON SHA'):
        evaluation_scalars(compact)


def test_training_uses_actual_rank_denominators_and_excludes_frozen_steps():
    values = training_scalars(training_record())
    assert values['Reward/total_per_second'] == pytest.approx(9.1)
    assert values['Termination/physical_failures_per_1000s'] == pytest.approx(50.)
    assert values['Policy/trainable_kl'] == pytest.approx(.2)
    assert 'Policy/clipped_objective_fraction' not in values  # joint-score没有真实PPO目标裁剪
    assert values['Loss/joint_score'] == pytest.approx(2.5)
    assert values['Loss/bc'] == pytest.approx(1.5)
    assert values['Policy/bc_to_ppo_gradient_ratio'] == pytest.approx(.1)
    assert values['Perf/controls_per_second'] == 100.
    assert sum(values['Sampling/time_share_'+s] for s in SOURCES) == pytest.approx(1.)


def test_missing_and_rejected_updates_do_not_fabricate_zero_learning_curves():
    record = training_record()
    record['mode'] = 'kl_rejected_rollback'
    assert training_scalars(record) == {}
    record['mode'] = 'dppo_bc'
    record['actor']['steps'] = []
    assert 'Loss/ppo' not in training_scalars(record)
    for collection in record['collection_by_rank']:
        del collection['by_source']['Mine']
    assert 'Reward/total_per_second' not in training_scalars(record)
    with pytest.raises(ValueError):
        checked_scalars({'budget/random_uuid/count': 5.})
    with pytest.raises(ValueError):
        checked_scalars({'Loss/ppo': float('nan')})


def test_validation_rms_and_paired_gain_keep_failure_and_motion_meaning():
    baseline = evaluation_report()
    current = evaluation_report(1, gain=.25, error=.04, activity=.8)
    current['episodes'][0]['physical_failure'] = True
    current['episodes'][0]['end_reason'] = 'physical_failure'
    values = evaluation_scalars(current, baseline)
    assert values['Tracking/root_position_rms_m'] == pytest.approx(.2)
    assert values['Tracking/root_position_p95_m'] == pytest.approx(.2)
    assert values['Motion/activity_retention'] == pytest.approx(.8)
    assert values['Termination/new_failure_count'] == 1
    assert values['Termination/physical_failure_fraction'] == 1/16
    assert values['Termination/window_limit_fraction'] == 15/16
    assert values['Termination/music_end_fraction'] == 0
    assert values['Reward/paired_gain_per_second'] == pytest.approx(.25)
    assert values['Reward/paired_gain_ci95_low'] == pytest.approx(.25)
    assert values['Reward/paired_gain_ci95_high'] == pytest.approx(.25)
    assert 'Reward/music' not in values


def test_validation_balances_sources_and_rms_is_not_average_root_error():
    report = evaluation_report()
    for episode in report['episodes']:
        value = .01 if episode['dataset'] == SOURCES[0] else .09
        for control in episode['decisions'][0]['controls']:
            control['raw_errors']['anchor_pos_squared_error'] = value
        if episode['dataset'] == SOURCES[0]:
            episode['decisions'][0]['controls'] *= 10
            episode['reward_sum'] *= 10
    values = evaluation_scalars(report)
    assert values['Tracking/root_position_rms_m'] == pytest.approx(np.sqrt(.07))
    assert values['Reward/total_per_second'] == pytest.approx(2.)


def test_bootstrap_does_not_touch_training_numpy_rng_and_keeps_music_clusters():
    baseline = evaluation_report()
    report = evaluation_report(1)
    for episode in report['episodes']:
        episode['reward_sum'] += .04 if ':0:' in episode['task_id'] else -.04
    before = np.random.get_state()
    result = evaluation_scalars(report, baseline)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert result['Reward/paired_gain_ci95_low'] < 0 < result['Reward/paired_gain_ci95_high']
    assert result == evaluation_scalars(report, evaluation_view(baseline))
    report['plan_sha256'] = 'different'
    with pytest.raises(ValueError, match='Baseline'):
        evaluation_scalars(report, baseline)


def test_zero_execution_failure_is_counted_without_invented_reward_rate():
    report = evaluation_report(1)
    episode = report['episodes'][0]
    episode.update(physical_failure=True, end_reason='physical_failure', reward_sum=-5.)
    episode['decisions'] = []
    values = evaluation_scalars(report, evaluation_report())
    assert values['Termination/physical_failure_fraction'] == 1/16
    assert 'Reward/total_per_second' not in values
    assert 'Reward/paired_gain_per_second' not in values


def test_real_events_keep_one_run_and_separate_training_validation_reward(tmp_path):
    writer = DashboardWriter(tmp_path/'dashboard')
    writer.write('training', 1, training_scalars(training_record()))
    writer.write('validation', 0, evaluation_scalars(evaluation_report()))
    writer.write('validation', 1, evaluation_scalars(evaluation_report(1, gain=.25)))
    writer.write('critic_mc', 1, mc_scalars({'after': dict(explained_variance=.8, mean_prediction=5., mean_target=7.)}))
    writer.close()
    dashboard = tmp_path/'dashboard'
    multiplexer = EventMultiplexer().AddRunsFromDirectory(str(dashboard)).Reload()
    assert list(multiplexer.Runs()) == ['.']
    events = EventAccumulator(str(dashboard)).Reload()
    tags = events.Tags()['scalars']
    assert tags and set(tags) <= EVENT_DEFINITIONS.keys()
    groups = {tag.split('/')[0] for tag in tags}
    assert groups == {'Reward', 'Tracking', 'Motion', 'Termination', 'Policy', 'Loss', 'Sampling', 'Perf'}
    assert len(EVENT_DEFINITIONS) <= 70
    assert all(events.Scalars(tag) for tag in tags)
    train = events.Scalars('Reward/train_total_per_second')
    validation = events.Scalars('Reward/validation_total_per_second')
    assert [(row.step, row.value) for row in train] == [(1, pytest.approx(9.1))]
    assert [(row.step, row.value) for row in validation] == [(0, 2.), (1, 2.25)]
    assert 'Reward/total_per_second' not in tags
    assert len(events.Scalars('Loss/critic_mc_ev')) == 1
    assert not any((dashboard/scope).exists() for scope in ('training', 'validation', 'critic_mc'))
    records = [json.loads(line) for line in (tmp_path/'dashboard/scalars.jsonl').read_text().splitlines()]
    assert records[1]['step'] == 0
    assert all('/state/' not in tag for row in records for tag in row['scalars'])


def make_completed_run(root):
    session = root/'sessions'/'session1'
    session.mkdir(parents=True)
    (root/'metrics').mkdir()
    artifacts = []
    for step in (0, 1):
        path = session/f'eval{step}.json'
        raw = json.dumps(evaluation_report(step, gain=step*.2)).encode()
        path.write_bytes(raw)
        artifacts.append(dict(iteration=step, path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest()))
        if step == 0:
            (root/'evaluation_baseline.json').write_bytes(raw)
    (session/'summary.json').write_text(json.dumps(dict(status='passed', exit_code=0, session_id='session1',
        initial_state={'iteration': 0}, final_state={'iteration': 1}, evaluation_artifacts=artifacts)))
    (root/'metrics/session1.jsonl').write_text(json.dumps(training_record())+'\n')
    (root/'curves.jsonl').write_text(json.dumps({'metrics': {'training/budget/uuid/unused': 1.}})+'\n')
    return root


def test_export_rebuilds_original_iterations_and_leaves_source_untouched(tmp_path):
    root = make_completed_run(tmp_path/'original')
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    summary = export_run(root, tmp_path/'clean')
    assert summary['accepted_iterations'] == 1
    assert summary['fixed_evaluations'] == 2
    assert summary['tensorboard_runs'] == 1
    events = EventAccumulator(str(tmp_path/'clean')).Reload()
    assert len(events.Scalars('Reward/train_total_per_second')) == 1
    assert [row.step for row in events.Scalars('Reward/validation_total_per_second')] == [0, 1]
    assert summary['final']['iteration'] == 1
    assert summary['final']['scalars']['Reward/paired_gain_per_second'] == pytest.approx(.2)
    assert {p: p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
    assert not (tmp_path/'clean.partial').exists()
    with pytest.raises(FileExistsError):
        export_run(root, tmp_path/'clean')


def test_export_rejects_missing_iteration_and_tampered_evaluation(tmp_path):
    root = make_completed_run(tmp_path/'original')
    (root/'metrics/session1.jsonl').write_text('')
    with pytest.raises(ValueError, match='coverage'):
        export_run(root, tmp_path/'missing')
    (root/'metrics/session1.jsonl').write_text(json.dumps(training_record())+'\n')
    (root/'sessions/session1/eval1.json').write_text('{}')
    with pytest.raises(ValueError, match='SHA'):
        export_run(root, tmp_path/'tampered')


def test_joint_score_curves_are_not_mislabeled_as_clipped_ppo():
    record = training_record()
    record['actor']['objective'] = 'joint_score_pg'
    record['actor']['steps'][-1]['backtracking'] = {'backtracks': 2}
    values = training_scalars(record)
    assert values['Loss/joint_score'] == pytest.approx(2.5)
    assert values['Policy/backtracks'] == 2
    assert 'Loss/ppo' not in values
    assert 'Policy/clipped_objective_fraction' not in values
    record['trainable4_kl']['per_denoising_step'].append(dict(step=19))
    with pytest.raises(ValueError, match='four trainable'):
        training_scalars(record)
