"""当前第二阶段32任务报告配对比较的CPU回归。

使用当前evaluate_policy和periodic_monitor生成四来源、双seed的实际格式小型
报告；执行器是CPU替身，不声称进行物理验收。覆盖四类成功/失败配对、提前失败
时长、任务均值与执行加权指标、原文件SHA、任务/奖励身份及拒绝覆盖输出。
仅验证当前报告，不构建已删除的旧分片发布或旧checkpoint兼容夹具。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gem.closedloop.dppo.evaluation import evaluate_policy
from gem.closedloop.dppo.full_dataset import SOURCES
from gem.closedloop.dppo.periodic_monitor import build_balanced_plan, merge_periodic_reports, shard_plan
from tests.closedloop.dppo.test_periodic_monitor import Catalog
from tests.test_evaluation import Env, policy
from tools.eval.compare_closedloop_stage10 import compare_evaluations, main


class PairedEnv(Env):
    def __init__(self, loaded, side):
        super().__init__(loaded)
        self.side = side

    def reset_task(self, sample, music, **kwargs):
        source = SOURCES.index(sample['dataset'])
        self.failure = source >= 2 if self.side == 'initial' else source % 2 == 1
        super().reset_task(sample, music, **kwargs)
        if self.failure:
            self.limit = min(self.limit, 5)


def reports(root):
    catalog, loaded = Catalog(root), policy()
    plan = build_balanced_plan(catalog)
    paths = []
    for side in ('initial', 'final'):
        child_root = root/side
        evaluate_policy(PairedEnv(loaded, side), loaded, shard_plan(plan, 0, 1), child_root,
                        catalog=catalog, actor_identity={'in_memory_model': True})
        report = merge_periodic_reports(plan, [child_root/'report.json'],
                                       evaluation_identity={'reward': 'current', 'physics': 'cpu_fixture'})
        path = child_root/'merged.json'
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def test_current_comparison_preserves_every_failure_and_execution_duration(tmp_path):
    result = compare_evaluations(*reports(tmp_path))
    assert result['paired_task_count'] == 32 and result['mode'] == 'periodic_subset'
    assert result['full_heldout_acceptance'] is False
    assert result['overall']['failure_pair_counts'] == dict.fromkeys(
        ('both_without_failure', 'new_failure', 'failure_recovered', 'both_failed'), 8)
    assert len(result['by_source']) == 4 and len(result['by_seed']) == 2
    for side in ('initial', 'final'):
        aggregate = result['overall']['actual_execution_aggregate'][side]
        assert aggregate['physical_failure_count'] == 16
        assert aggregate['executed_seconds'] == pytest.approx(4.8)
    for name, sign in [('new_failure', -1), ('failure_recovered', 1)]:
        metrics = result['by_failure_pair'][name]['task_weighted_metrics']
        assert sign*metrics['episode_reward']['delta']['mean'] > 0
        assert sign*metrics['executed_seconds']['delta']['mean'] > 0
    metrics = result['overall']['task_weighted_metrics']
    assert metrics['episode_reward']['delta']['mean'] == pytest.approx(0)
    assert metrics['reward_per_executed_second']['initial']['mean'] != pytest.approx(
        result['overall']['actual_execution_aggregate']['initial']['reward_per_executed_second'])


@pytest.mark.parametrize('fault', ['old_schema', 'plan', 'reward', 'aggregate', 'episode_bytes',
                                  'missing_episode', 'duplicate_episode', 'model_identity'])
def test_current_comparison_rejects_tampered_or_retired_evidence(tmp_path, fault):
    initial, final = reports(tmp_path)
    report = json.loads(final.read_text())
    if fault == 'old_schema':
        report['schema'] = 'genmo.closedloop.stage10.evaluation_merge.v1'
    elif fault == 'plan':
        report['plan']['tasks'][0]['seed'] += 1
    elif fault == 'reward':
        report['evaluation_identity']['reward'] = 'different'
    elif fault == 'aggregate':
        report['episode_metrics'][0]['reward'] += 1
    elif fault == 'episode_bytes':
        path = Path(report['source_episode_manifests'][0]['path'])
        path.write_text(path.read_text()+'\n')
    elif fault == 'missing_episode':
        report['source_episode_manifests'].pop()
    elif fault == 'duplicate_episode':
        report['source_episode_manifests'][0] = report['source_episode_manifests'][1]
    else:
        report['actor_identity']['model_fingerprint'] = ''
    final.write_text(json.dumps(report))
    with pytest.raises((ValueError, FileNotFoundError)):
        compare_evaluations(initial, final)


def test_current_comparison_cli_refuses_overwriting(tmp_path):
    initial, final = reports(tmp_path)
    output = tmp_path/'comparison.json'
    args = ['--initial-report', str(initial), '--final-report', str(final), '--output', str(output)]
    assert main(args) == 0
    assert json.loads(output.read_text())['paired_task_count'] == 32
    with pytest.raises(FileExistsError):
        main(args)
