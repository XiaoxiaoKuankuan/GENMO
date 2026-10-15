"""精简sim2sim配置的默认补齐、路径迁移与独立加载回归。

使用仓库中的日常YAML和临时配置验证：省略固定参数后仍保留确定性DDIM和完整
运行设置；更换GMT根目录只迁移GMT资产，不影响GENMO权重；配置复制到别处或
更换启动目录不会误解相对路径。另核验Python符号链接入口和模型名称原样保留，
防止路径规范化破坏虚拟环境或混淆模型选择。测试只解析配置，不加载任何
真实权重、声音设备、GPU或动力学；所有写入均使用pytest临时目录。
"""
from pathlib import Path

import pytest
import yaml

from gem.closedloop.sim2sim.config import GENMO_ROOT,load_config


CONFIG=GENMO_ROOT/'configs/closedloop/sim2sim_mujoco.yaml'


def test_minimal_config_supplies_fixed_runtime_and_deterministic_sampling():
    config=load_config(CONFIG)
    assert config['schema']=='genmo.gmt.mujoco_sim2sim.v1'
    assert config['generation_frames']==config['sampling']['generation_frames']==60
    assert config['sampling']['denoising_steps']==20 and config['sampling']['eta']==0.
    assert config['runtime']['continue_on_failure'] and config['runtime']['audio']
    assert config['runtime']['warmup_control_steps']==50 and config['runtime']['viewer_fps']==25
    assert config['models'][0]['checkpoint']==config['paths']['stage1_checkpoint']
    assert Path(config['paths']['genmo_repo'])==GENMO_ROOT
    config['sampling']['std_schedule'][0]=99.
    assert load_config(CONFIG)['sampling']['std_schedule'][0]==.001


def test_gmt_move_and_different_launch_directory_keep_correct_asset_bases(tmp_path,monkeypatch):
    raw=yaml.safe_load(CONFIG.read_text())
    gmt=tmp_path/'新的GMT目录';raw['paths']['gmt_repo']=str(gmt)
    raw['runtime']['audio']=False
    path=tmp_path/'另一份配置.yaml';path.write_text(yaml.safe_dump(raw,allow_unicode=True))
    before=load_config(CONFIG)
    monkeypatch.chdir(tmp_path)
    config=load_config(path)
    for key in ('gmt_policy','gmt_checkpoint','gmt_agent','mjcf','urdf','isaac_contract'):
        assert Path(config['paths'][key])==gmt/raw['paths'][key]
    assert config['models']==before['models'] and config['playlist']==before['playlist']
    assert not config['runtime']['audio'] and config['runtime']['calibration_samples']==12


def test_absolute_paths_and_python_symlink_entry_are_preserved(tmp_path):
    executable=tmp_path/'真正解释器';executable.touch()
    link=tmp_path/'虚拟环境python';link.symlink_to(executable)
    raw=yaml.safe_load(CONFIG.read_text())
    raw['paths']['gmt_validation_python']=str(link)
    raw['models'][1]['checkpoint']=str(tmp_path/'新模型.pt')
    path=tmp_path/'配置.yaml';path.write_text(yaml.safe_dump(raw,allow_unicode=True))
    config=load_config(path)
    assert config['paths']['gmt_validation_python']==str(link)
    assert config['models'][1]['checkpoint']==str(tmp_path/'新模型.pt')
    assert config['models'][1]['name']==raw['models'][1]['name']


@pytest.mark.parametrize('content',['[]','runtime: []','schema: unsupported'])
def test_invalid_config_structure_is_rejected(tmp_path,content):
    path=tmp_path/'invalid.yaml';path.write_text(content)
    with pytest.raises(ValueError):load_config(path)
