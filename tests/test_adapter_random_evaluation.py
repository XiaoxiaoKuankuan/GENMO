"""随机音乐评价的纯CPU合同回归，不启动仿真、不加载训练模型。

构造临时音频字节验证固定种子复现、音乐组及音频双重去重；构造逐歌曲物理报告
验证任意声明样本数、混合来源分组、失败保留和配对身份检查。测试目录由pytest
调用方指定系统临时路径，所有数据均为合成夹具，不冒充真实动力学成绩。
"""

import copy
import importlib.util
from pathlib import Path
import random

import pytest


def module(name):
    path = Path(__file__).resolve().parents[1]/'tools/eval'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_random_selection_unique_reproducible(tmp_path):
    sample = module('prepare_random_adapter_evaluation').sample_unique
    pool = []
    for index in range(8):
        path = tmp_path/f'{index}.wav'
        path.write_bytes(str(index//2).encode())
        pool.append(dict(id=str(index), song_id=str(index), audio_path=str(path)))
    a = sample(pool, 4, random.Random(20261031), set(), set())
    b = sample(pool, 4, random.Random(20261031), set(), set())
    assert a == b
    assert len({t['audio_sha256'] for t in a}) == 4
    with pytest.raises(ValueError):
        sample(pool, 5, random.Random(42), set(), set())


def report():
    episodes = []
    for index in range(128):
        episodes.append(dict(task=str(index), song=str(index), split='train' if index<64 else 'evaluation',
            source='synthetic', adapter_training_seen=index<32, planned_steps=100,
            evaluation_budget_steps=100, initial_state_sha256=str(index), physical_xi_sha256='fixed',
            steps=100, full_music_complete=True, reason='music_end',
            full_horizon_tracking=.1, root_position_rmse_m=.3, jerk_rms=100.,
            music={'f1': .3}, sm_terms={}))
    return dict(episodes=episodes, full_song_evaluation=True, sm_mdp_contract={})


def test_mixed128_groups_failure_and_pairing():
    compare = module('summarize_adapter_train_matrix').compare_reports
    zero = report()
    candidate = copy.deepcopy(zero)
    candidate['episodes'][0].update(steps=10, full_music_complete=False, reason='physical_failure')
    result = compare([zero, zero], [candidate], expected_songs=128, allowed_splits=['train','evaluation'])
    assert result['overall']['songs'] == 128
    assert result['overall']['adapter_completion'] == 127/128
    assert result['groups']['adapter_training_seen']['True']['songs'] == 32
    assert result['groups']['split']['evaluation']['songs'] == 64
    candidate['episodes'][0]['physical_xi_sha256'] = 'changed'
    with pytest.raises(ValueError, match='配对身份'):
        compare([zero], [candidate], expected_songs=128, allowed_splits=['train','evaluation'])


def test_repeated_songs_rejected():
    compare = module('summarize_adapter_train_matrix').compare_reports
    zero = report()
    zero['episodes'][1]['song'] = zero['episodes'][0]['song']
    with pytest.raises(ValueError, match='重复歌曲'):
        compare([zero], [zero], expected_songs=128, allowed_splits=['train','evaluation'])


def test_p90_nearest_rank():
    aggregate = module('summarize_adapter_train_matrix').aggregate_rows
    rows = [dict(baseline_completion=1., adapter_completion=1., metrics={'jerk_rms': dict(
        baseline=1., adapter=float(i), delta=float(i-1), baseline_repeat_range=0., candidate_repeat_range=0.)})
        for i in range(1,11)]
    assert aggregate(rows)['metrics']['jerk_rms']['adapter_p90'] == 9.
