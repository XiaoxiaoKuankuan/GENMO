"""当前 Stage10 入口、真实规模和配套 GMT 身份的 CPU 集成检查。

读取实际主配置与诊断配置，确认同一实现绑定正确资产、原功能工作树和统一 Python，
并验证 8192 链名义容量的 minibatch/BC 推导。旧训练配置必须明确拒绝，不能隐式转换为当前
方法。独立检查 CLI 帮助和源码身份清单，覆盖入口迁移时容易遗漏的脚本路径。
配套 GMT 检查只执行纯目录配置函数，不导入 Isaac，不创建 GPU 上下文或运行仿真。
"""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from gem.closedloop.dppo.training import _sources
from gem.closedloop.dppo.training_config import VERSION, configuration

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'tests/closedloop/fixtures/dppo_contract.yaml'


def test_main_scale_is_16384_nominal_chains_with_four_clipped_dppo_updates():
    c = configuration(CONFIG)
    s, t = c['stage10'], c['stage10']['training']
    assert s['version'] == VERSION
    assert s['derived'] == dict(world_size=8, real_chains=16384, local_chains=2048,
        planned_actor_steps=4, local_ppo_internal_visits=8192, local_trainable_kl_visits=8192,
        bc_per_step=256, bc_per_rollout=1024,maximum_actor_steps=4,
        count_semantics='nominal_capacity_only_actual_manifest_controls_optimization')
    assert c['runtime']['num_envs']==c['runtime']['generation_batch']==128
    assert s['critic']['global_batch']==2048 and s['scale']['actor_minibatch_chains']==4096
    assert t['denoising_microbatch']==t['kl_microbatch']==128
    assert s['critic']['complete_mc']['training_samples_per_source']==24
    assert (t['denoising_steps'], t['ppo_steps_per_chain'], t['ppo_epochs']) == (20, 4, 1)
    assert t['frozen_prefix_steps']==15 and t['ppo_step_strategy']=='intermediate4'
    assert t['actor_lr'] == 3e-7 and t['ppo_clip'] == .01
    assert c['runtime']['timing_contract'] == 'modeled_deployment.v3'
    assert (c['timing']['control_ticks'], c['timing']['physics_ticks']) == (12, 3)


@pytest.mark.parametrize('path', [CONFIG],
                         ids=lambda path: path.stem)
def test_diagnostics_share_current_code_and_existing_asset_paths(path):
    c = configuration(path)
    p = c['paths']
    assert p['genmo_repo'] == '/home/user/liwei/GENMO-bumi-closedloop'
    assert p['gmt_repo'] == '/home/user/liwei/legged_lab_gmt'
    assert p['genmo_python'] == p['isaac_python'] == (
        '/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python')
    assert Path(p['checkpoint']).name == 's595000.pt'
    assert Path(p['gmt_checkpoint']).name == 'model_140500.pt'
    assert c['stage10']['assets']['checkpoint'] == configuration(CONFIG)['stage10']['assets']['checkpoint']


@pytest.mark.parametrize('version', ['genmo.closedloop.stage10.v2', 'genmo.closedloop.stage10.v3', 'genmo.closedloop.stage10.continuous_horizon.v1'])
def test_superseded_training_configuration_is_rejected(tmp_path, version):
    c = yaml.safe_load(CONFIG.read_text())
    c['stage10']['version'] = version
    path = tmp_path / 'old.yaml'; path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='configuration required'):
        configuration(path)


@pytest.mark.parametrize('key,value', [('random_start',False),('sampling_contract',None),
    ('initialization','standing'),('standing_start_probability',1.),('time_bin_seconds',.015)])
def test_training_rejects_incomplete_random_state_configuration(tmp_path,key,value):
    c=yaml.safe_load(CONFIG.read_text())
    c['stage10']['dataset'][key]=value
    path=tmp_path/'incomplete_random_state.yaml';path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='sampling|initialization|time bins'):
        configuration(path)


def test_cli_help_is_available_before_loading_gpu_or_isaac():
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/train_closedloop_stage10.py'), '--help'],
        capture_output=True, text=True, timeout=30, check=True)
    assert '--warmup-rollouts' in result.stdout and '--stop-after-iteration' in result.stdout
    subprocess.run(['bash', '-n', str(ROOT / 'scripts/train_stage10_8gpu_server1.sh')], check=True)


@pytest.mark.parametrize('value', [0, None, [], [1, 2]])
def test_unused_effect_iteration_field_is_explicitly_rejected(tmp_path, value):
    c = yaml.safe_load(CONFIG.read_text())
    c['stage10']['evaluation']['effect_check_iterations'] = value
    path = tmp_path / 'unused.yaml'
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='effect_check_iterations is unsupported'):
        configuration(path)


@pytest.mark.parametrize('split,count,unique', [('train',4,False),('val','all_unique',False),('test',0,True),
    ('val',{'AIST++':2},True)])
def test_invalid_heldout_selection_rejected_before_gpu_start(tmp_path,split,count,unique):
    c=yaml.safe_load(CONFIG.read_text())
    c['stage10']['evaluation'].update(split=split,samples_per_source=count,unique_music=unique)
    path=tmp_path/'invalid_selection.yaml';path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='Held-out evaluation'):
        configuration(path)


def test_source_identity_covers_shell_entry_and_shared_execution_module(tmp_path):
    genmo, gmt = tmp_path / 'genmo', tmp_path / 'gmt'
    names = ['tools/train_closedloop_stage10.py', 'scripts/train_stage10_8gpu_server1.sh',
             'gem/closedloop/dppo/execution_state.py']
    for name in names:
        path = genmo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('original')
    c = dict(paths=dict(genmo_repo=str(genmo), gmt_repo=str(gmt)))
    original = _sources(c)
    for name in names:
        path = genmo / name; path.write_text('modified')
        assert _sources(c) != original
        path.write_text('original')


def test_current_gmt_identity_keeps_rank_asset_conversion_separate(tmp_path):
    gmt = Path(os.environ.get('STAGE10_TEST_GMT_REPO', ROOT.parent / 'legged_lab_gmt'))
    path = gmt / ('source/NoetixRobot/NoetixRobot/tasks/mimic/'
                  'mimic_noetix_bumi4340_mha_sonic/closedloop/config.py')
    if not path.is_file():
        pytest.skip('配套 GMT 源码未提供；服务器集成检查必须提供 STAGE10_TEST_GMT_REPO')
    source = path.read_text(); tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'configure_worker_asset_conversion')
    module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    scope = {'Path': Path}
    exec(compile(module, str(path), 'exec'), scope)
    observed = []
    for rank in (0, 7):
        directory = tmp_path / f'rank{rank:02d}'
        spawn = SimpleNamespace(usd_dir='default')
        cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=spawn)))
        settings = dict(stage10=dict(version=VERSION),
                        runtime=dict(rank=rank, asset_conversion_dir=str(directory)))
        scope['configure_worker_asset_conversion'](cfg, settings)
        observed.append(spawn.usd_dir)
        assert spawn.usd_dir == str(directory.resolve())
    assert observed[0] != observed[1]
