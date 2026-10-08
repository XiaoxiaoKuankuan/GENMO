"""第二阶段固定32任务周期验证的CPU计划、证据与选择验收。

本文件使用既有evaluate_policy及可计数执行替身，在pytest临时目录形成真实
逐episode JSON及SHA。检查四库均衡固定选样、多seed分片不变、完整覆盖与重复/
漏片/篡改拒绝、原控制奖励统计重算、best observed和best saved分别维护，以及
显式periodic_subset配对模式不会绕过默认完整val审计。嵌套指标使用SummaryWriter
替身核对tag/step并写临时JSONL；固定初始随机链漂移复用真实小型Actor夹具，
不启动GPU、Isaac、正式checkpoint下载或长训练。
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.evaluation import evaluate_policy
from gem.closedloop.dppo.full_dataset import SOURCES
from gem.closedloop.dppo.periodic_monitor import (
    build_balanced_plan,
    choose_best,
    fixed_chain_drift,
    load_reference_traces,
    log_metrics,
    merge_periodic_reports,
    shard_plan,
)
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from tests.closedloop.test_stage1_actor import _activate_branches, _conditions
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory
from tests.test_evaluation import Env, policy
from tools.eval.compare_closedloop_stage10 import compare_evaluations


class Catalog:
    def __init__(self, root):
        self.data_root, self.identity = root, {'version': 'periodic_fake_four_source'}
        self.samples = {'val': {source: tuple(dict(dataset=source, split='val', group_id=f'{source}:{i}',
            row=dict(sample_id=str(i), split='val', fps=30, num_frames=6,
                     source_audio_sha256=f'{source}:{i}')) for i in range(5)) for source in SOURCES}}
    def load_music(self, sample):
        return np.zeros((sample['row']['num_frames'], 35), dtype=np.float32)


def evaluate_ranks(tmp_path, *, failure=False):
    catalog, loaded = Catalog(tmp_path), policy()
    plan = build_balanced_plan(catalog)
    paths = []
    for rank in range(2):
        child = shard_plan(plan, rank, 2)
        root = tmp_path/f'rank_{rank}'
        evaluate_policy(Env(loaded, failure=failure), loaded, child, root, catalog=catalog,
                        actor_identity={'in_memory_model': True, 'iteration': 100})
        paths.append(root/'report.json')
    return plan, paths


def test_balanced_plan_is_fixed_and_shards_preserve_all_task_noise(tmp_path):
    catalog = Catalog(tmp_path)
    plan = build_balanced_plan(catalog)
    assert plan == build_balanced_plan(catalog)
    assert plan['complete_pool_count'] == 20 and plan['selected_sample_count'] == 16
    assert plan['task_count'] == 32 and plan['seeds'] == [42, 1729] and plan['episode_seconds'] == 10.
    assert all(sum(t['dataset'] == source for t in plan['tasks']) == 8 for source in SOURCES)
    tasks = [task for rank in range(8) for task in shard_plan(plan, rank, 8)['tasks']]
    assert len({t['task_id'] for t in tasks}) == 32
    assert sorted(t['task_id'] for t in tasks) == sorted(t['task_id'] for t in plan['tasks'])
    assert all(task in plan['tasks'] for task in tasks)
    with pytest.raises(ValueError, match='parent'):
        shard_plan(shard_plan(plan, 0, 8), 0, 1)
    catalog.samples['val'][SOURCES[0]] = catalog.samples['val'][SOURCES[0]][:3]
    with pytest.raises(ValueError, match='Insufficient'):
        build_balanced_plan(catalog)


def test_periodic_merge_reproduces_32_episodes_and_rejects_missing_or_duplicate(tmp_path):
    plan, paths = evaluate_ranks(tmp_path)
    result = merge_periodic_reports(plan, paths[::-1], evaluation_identity={'reward': 'test'})
    assert result['task_count'] == 32 and result['physical_failure_count'] == 0
    assert len(result['episode_metrics']) == 32
    assert result['mean_executed_seconds'] == pytest.approx(.2)
    assert result['source_balanced_reward'] == pytest.approx(result['aggregate']['overall']['episode_reward']['mean'])
    assert result['assertions']['full_heldout_acceptance'] is False
    assert merge_periodic_reports(plan, paths, baseline=result, evaluation_identity={'reward': 'test'})['paired_baseline']['new_failures'] == 0
    with pytest.raises(ValueError, match='rank'):
        merge_periodic_reports(plan, paths[:1])
    with pytest.raises(ValueError, match='rank'):
        merge_periodic_reports(plan, [paths[0], paths[0]])
    reference = json.loads(paths[0].read_text())['episode_manifests'][0]
    episode_path = paths[0].parent/reference['path']
    value = json.loads(episode_path.read_text())
    value['reward_sum'] += 1.
    episode_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='SHA'):
        merge_periodic_reports(plan, paths)


def test_best_observed_is_not_mislabelled_saved_and_baseline_safety_gate(tmp_path):
    plan, paths = evaluate_ranks(tmp_path)
    baseline = merge_periodic_reports(plan, paths)
    improved = copy.deepcopy(baseline)
    improved['source_balanced_reward'] += 1.
    for row in improved['episode_metrics']:
        row['reward'] += 1.
    best = choose_best(improved, baseline, iteration=100)
    assert best['eligible'] and best['best_observed']['iteration'] == 100 and best['best_saved'] is None
    saved = choose_best(improved, baseline, saved=True, iteration=300, **{
        name: best[name] for name in ('best_observed', 'best_saved')})
    assert saved['best_observed']['iteration'] == 100 and saved['best_saved']['iteration'] == 300
    shortened = copy.deepcopy(improved)
    shortened['mean_executed_seconds'] -= .01
    assert not choose_best(shortened, baseline)['eligible']
    failed = copy.deepcopy(improved)
    failed['physical_failure_count'] = 1
    failed['episode_metrics'][0]['physical_failure'] = 1
    assert not choose_best(failed, baseline)['eligible']


def test_subset_comparison_is_explicit_and_rechecks_origin_sha(tmp_path):
    plan, paths = evaluate_ranks(tmp_path/'evaluation')
    merged = merge_periodic_reports(plan, paths, evaluation_identity={'reward': 'fixed', 'runtime': 'same'})
    initial, final = tmp_path/'initial.json', tmp_path/'final.json'
    for path in (initial, final):
        path.write_text(json.dumps(merged))
    result = compare_evaluations(initial, final, mode='periodic_subset')
    assert result['paired_task_count'] == 32 and result['full_heldout_acceptance'] is False
    assert result['mode'] == 'periodic_subset'
    with pytest.raises((ValueError, FileNotFoundError)):
        compare_evaluations(initial, final)  # 不能把subset静默当fullval。
    damaged = copy.deepcopy(merged)
    damaged['episode_metrics'][0]['reward'] += 1
    final.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match='aggregate'):
        compare_evaluations(initial, final, mode='periodic_subset')


def test_nested_metrics_keep_tags_and_skip_missing_values(tmp_path):
    logged = []
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: logged.append((tag, value, step)), flush=lambda: None)
    path = tmp_path/'metrics.jsonl'
    values = log_metrics(writer, path, 100, {'reward': 2., 'source': {'Mine': {'loss': .3}}, 'missing': None}, prefix='val')
    assert values == {'val/reward': 2., 'val/source/Mine/loss': .3}
    assert len(logged) == 2 and all(row[2] == 100 for row in logged)
    assert json.loads(path.read_text())['metrics'] == values
    with pytest.raises(ValueError, match='Nonfinite'):
        log_metrics(writer, path, 101, {'bad': float('nan')})


def test_fixed_initial_chain_drift_needs_no_upper_transition(actor_factory, tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actor, batch = actor_factory(starts=(45,))
        _activate_branches(actor)
        loaded = DPPODiffusionPolicy(actor, steps=3)
        trace = loaded.sample_rollout(_conditions(batch), generator=torch.Generator().manual_seed(72))
        raw = tmp_path/'initial_eval_raw.pt'
        torch.save({'trace': trace}, raw)
        reference = load_reference_traces([raw])
        first = fixed_chain_drift(loaded, reference)
        assert first['internal_transition_count'] == 3 and first['mean_joint_kl'] == pytest.approx(0., abs=1e-12)
        with torch.no_grad():
            next(actor.music_embedder.parameters()).add_(.01)
        second = fixed_chain_drift(loaded, reference)
        assert second['mean_joint_kl'] > 0 and second['reference_sha256'] == first['reference_sha256']
    finally:
        torch.set_num_threads(previous)
