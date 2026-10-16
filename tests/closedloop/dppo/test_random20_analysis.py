"""二十轮只读分析的跨rank覆盖和外部显存计量回归。

仅使用内存采样器checkpoint替身和临时CSV。验证重叠真实执行区间必须求并集，
按歌曲时间三等分统计，中后段不能由抽样次数伪造；GPU监控单位显式换算GiB。
共同执行时长对照保留零步事件罚分，不计入另一方尚未执行区间，也不重复乘dt。
不载入生产权重、不生成训练数据或启动物理，完整二十轮仍由实际运行报告核验。
"""
from types import SimpleNamespace
import pytest
from tools.eval.analyze_stage2_random20 import (
    execution_coverage, gpu_memory, reward_timelines, common_duration_pairing,
)
from gem.closedloop.dppo.full_dataset import SOURCES


def test_cross_rank_real_time_coverage_merges_duplicates_and_keeps_music_thirds():
    catalog = SimpleNamespace(samples={'train': {s: [{'row': {'sample_id': 'song', 'num_frames': 90}}]
                                               for s in SOURCES}})
    ranks = []
    for spans in ([[0, 600], [1200, 1800]], [[300, 900]]):
        ranks.append(dict(collector=dict(schema='genmo.stage10.inprocess_collector.time_sampling.v1',
            time_pool={'standing_draws': 1, 'motion_draws': 2},
            states=[dict(sampler={'intervals': {SOURCES[0] + '/song': spans}}), None])))
    result = execution_coverage({'rank_states': ranks}, catalog)
    source = result['by_source'][SOURCES[0]]
    assert source['covered_seconds_by_third'] == [1., .5, 1.]
    assert source['total_unique_seconds'] == 2.5
    assert source['total_unique_fraction'] == pytest.approx(5/6)
    assert result['issued_task_counts'] == dict(standing=2, paired_motion=4)
    assert result['by_source'][SOURCES[1]]['total_unique_seconds'] == 0
    ranks[0]['collector']['schema'] = 'old'
    with pytest.raises(ValueError, match='identity'):
        execution_coverage({'rank_states': ranks}, catalog)


def test_polled_gpu_peak_reads_actual_csv_units(tmp_path):
    path = tmp_path/'gpu.csv'
    path.write_text('timestamp, index, memory.used [MiB], utilization.gpu [%], power.draw [W]\n'
                    '2026/10/16 18:00:00, 0, 1024 MiB, 90 %, 100 W\n'
                    '2026/10/16 18:00:05, 0, 2048 MiB, 10 %, 80 W\n')
    assert gpu_memory(path)['per_gpu_GiB'] == {'0': 2.}


def episode(rewards, event=0.):
    return dict(task_id='song-seed', dataset='FineDance', sample_id='song', seed=42,
        music_start_frame=0, full_music_seconds=10., reward_sum=sum(rewards)+event,
        decisions=[dict(controls=[dict(reward=r, dt_s=.02) for r in rewards], zero_step_event_reward=0.),
                   dict(controls=[], zero_step_event_reward=event)])


def test_common_elapsed_time_excludes_later_controls_but_keeps_boundary_event():
    before = reward_timelines([episode([1., 2., 100.])])
    after = reward_timelines([episode([2., 4.], event=-5.)])
    result = common_duration_pairing(before, after)
    row = result['pairs'][0]
    assert row['common_controls'] == 2
    assert row['initial_reward'] == 3.
    assert row['final_reward'] == 1.
    assert row['delta']['common_prefix_reward_per_second'] == -50.
    assert result['music_group_interval']['paired_mean'] == -50.
    assert result['initial_all_time_weighted_rate'] == 75.


def test_zero_elapsed_task_is_explicitly_excluded_from_rate_only():
    result = common_duration_pairing(reward_timelines([episode([1.])]),
                                   reward_timelines([episode([], event=-5.)]))
    assert result['excluded_zero_common_controls'] == ['song-seed']
    assert result['music_group_interval'] is None
    assert result['final_all_time_weighted_rate'] is None


@pytest.mark.parametrize('fault', ['clock', 'integral', 'identity'])
def test_common_elapsed_comparison_rejects_misaligned_evidence(fault):
    original = episode([1.])
    changed = episode([1.])
    if fault == 'clock':
        changed['decisions'][0]['controls'][0]['dt_s'] = .04
    elif fault == 'integral':
        changed['reward_sum'] = 2.
    else:
        changed['music_start_frame'] = 1
    with pytest.raises(ValueError):
        common_duration_pairing(reward_timelines([original]), reward_timelines([changed]))
