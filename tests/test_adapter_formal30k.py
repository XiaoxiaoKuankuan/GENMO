"""正式三万次八卡配置、学习率边界及128首评价口径的纯CPU回归。

逐项核对四种参数组合与双种子、相同冻结资产/训练缓存、实际Adam预算和阶段降率。
用合成128首清单验证127首可评价与1首原初态失败的精确分母；验证报告保持原始
数据划分、已见/未见分组、失败计数以及两次零基线只平均汇总而不捏造episode。
所有测试数据只写pytest临时目录，不启动服务器、CUDA或仿真，不代表真实质量收益。
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'gem/closedloop/reference_adapter'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eight_configs_are_paired_two_factor_30k_matrix():
    configs = [yaml.safe_load((ROOT/f'configs/closedloop/stage2_formal30k_20261031/gpu{i}.yaml').read_text())
               for i in range(8)]
    combinations = set()
    for i, cfg in enumerate(configs):
        a, train = cfg['alternating'], cfg['training']
        multiplier = 3 if i//2 in (1, 3) else 1
        assert a['precision']['execution_jerk_weight'] == pytest.approx(.10504519010557042*multiplier)
        assert a['precision']['stance_foot_weight'] == pytest.approx(.41037086737445894*multiplier)
        combinations.add((multiplier, train['adapter_lr'], train['seed']))
        assert cfg['bptt_steps'] == 32 and not a['critic_enabled']
        assert a['precision']['root_tracking_bonus_weight'] == 1.2
        assert cfg['vector']['target_accepted_updates'] == 30000 and cfg['vector']['num_envs'] == 2048
        assert train['seed'] == cfg['vector']['sampling_seed'] == 42+i%2
        assert train['adapter_lr_milestones'] == [10000, 20000] and train['adapter_lr_gamma'] == .5
        assert a['evaluation_num_envs'] == 127 and a['fixed_music_evaluation']['requested_song_count'] == 128
        assert a['evaluation_updates'][-1] == 30000 and a['full_song_from_update'] == 100
        for key in ('tasks', 'genmo', 'supermimic', 'sm_mdp', 'rewards', 'reference_execution'):
            assert cfg[key] == configs[0][key]
    assert combinations == {(s, lr, seed) for s in (1, 3) for lr in (1e-5, 5e-6) for seed in (42, 43)}


def test_lr_uses_completed_adam_not_warmup_and_keeps_old_default():
    lr = load_module('alternating_entry').adapter_learning_rate
    cfg = dict(adapter_lr=1e-5, adapter_lr_milestones=[10000, 20000], adapter_lr_gamma=.5)
    assert [lr(cfg, n) for n in (0, 9999, 10000, 19999, 20000, 29999)] == [1e-5, 1e-5, 5e-6, 5e-6, 2.5e-6, 2.5e-6]
    assert lr(dict(adapter_lr=1e-5), 25000) == 1e-5


def panel_fixture(tmp_path):
    selected = [dict(id=str(i), song_id=str(i), global_music_group_id=str(i),
        split='train' if i<117 else 'validation', adapter_training_seen=i<35) for i in range(128)]
    tasks = selected[:-1]
    coverage = dict(requested=128, evaluable=127, unavailable=[{'task': '127', 'model_evaluated': False}])
    paths = {}
    for key, data in [('selection_manifest', selected), ('tasks', tasks), ('reference_coverage', coverage)]:
        path = tmp_path/(key+'.json'); path.write_text(json.dumps(data)); paths[key] = str(path)
    return dict(paths, requested_song_count=128), tasks


def test_panel_preserves_failed_reference_and_rejects_silent_replacement(tmp_path):
    module = load_module('fixed_music_panel')
    cfg, tasks = panel_fixture(tmp_path)
    meta = module.panel_metadata(cfg, tasks)
    assert meta['requested_song_count'] == 128 and meta['paired_evaluable_songs'] == 127
    assert meta['reference_generation_failures'][0]['model_evaluated'] is False
    with pytest.raises(ValueError, match='覆盖'):
        module.panel_metadata(cfg, tasks[:-1])
    changed = copy.deepcopy(tasks); changed[0]['adapter_training_seen'] = False
    with pytest.raises(ValueError, match='身份'):
        module.panel_metadata(cfg, changed)


def test_panel_rejects_duplicate_music_even_with_distinct_tasks(tmp_path):
    module = load_module('fixed_music_panel')
    cfg, tasks = panel_fixture(tmp_path)
    path = Path(cfg['selection_manifest']); selected = json.loads(path.read_text())
    selected[-1]['global_music_group_id'] = selected[0]['global_music_group_id']
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match='独立音乐'):
        module.panel_metadata(cfg, tasks)


def test_seen_unseen_summary_keeps_original_split_and_physical_failures():
    module = load_module('fixed_music_panel')
    episodes = [dict(task=str(i), split='train', adapter_training_seen=i<35, steps=10,
        full_music_complete=i!=80, constraint_failure=False, root_position_rmse_m=.1 if i<35 else .4,
        full_horizon_tracking=.09, sm_terms={'track': .1}, music={'f1': .3}) for i in range(127)]
    report = dict(episodes=episodes, num_envs=127, requested_song_count=128, paired_evaluable_songs=127)
    seen, unseen = module.panel_subset(report, True), module.panel_subset(report, False)
    assert seen['measured_evaluation_envs'] == 35 and unseen['measured_evaluation_envs'] == 92
    assert unseen['completion_rate'] == pytest.approx(91/92)
    assert unseen['mean_root_position_rmse_m'] == .4
    assert all(e['split'] == 'train' for e in unseen['episodes'])
    second = dict(unseen, mean_root_position_rmse_m=.6)
    average = module.mean_baseline_reports([unseen, second])
    assert average['mean_root_position_rmse_m'] == .5 and average['baseline_repetitions'] == 2
    assert 'episodes' not in average


def test_development_best_label_never_claims_pristine_test():
    choose = load_module('alternating_entry').select_validation_best
    zero = dict(completion_rate=1., mean_root_position_rmse_m=.3, mean_full_horizon_tracking=.1, mean_jerk_rms=100.)
    result = choose(None, zero, dict(zero, mean_root_position_rmse_m=.2), {'path': 'initial.pt'},
        adapter_updates=100, scope='full_song', selection_population='adapter_unseen_development_panel')
    assert result['selection_split'] == 'adapter_unseen_development_panel' and result['adapter_updates'] == 100
