"""独立物理评价的模型来源与配置隔离回归。

验证评价配置相对正式配置仅关闭训练父模型初始化，并强制观察入口拒绝可能在
显式评价Actor加载后覆盖模型的配置；同时检查禁止恢复训练和覆盖已有运行目录。
这些测试保护物理对照的模型身份，不运行Isaac、不加载真实checkpoint或修改训练。
所有临时文件由pytest的显式独立basetemp及外层TemporaryDirectory负责清理。
"""
from pathlib import Path

import pytest
import yaml

from tools.evaluate_stage2_dppo_consumed import validate_arguments

FORMAL=Path('configs/closedloop/stage2_dppo_server2_formal.yaml')
EVALUATION=Path('configs/closedloop/stage2_dppo_server2_diagnostic_eval.yaml')


def test_evaluation_configuration_changes_only_parent_initialization():
    formal=yaml.safe_load(FORMAL.read_text())
    evaluation=yaml.safe_load(EVALUATION.read_text())
    assert formal['stage10']['actor_initialization']['mode']=='actor_adam_new_reward'
    formal['stage10']['actor_initialization']=None
    assert formal==evaluation


def test_rejects_configuration_that_would_replace_explicit_actor(tmp_path):
    with pytest.raises(ValueError,match='覆盖显式评价Actor'):
        validate_arguments(['--output-dir',str(tmp_path/'new'),'--evaluation-actor','candidate.pt',
                            '--config',str(FORMAL),'--evaluate-only'])


def test_evaluation_only_and_no_output_overwrite(tmp_path):
    output=tmp_path/'new'
    argv=['--output-dir',str(output),'--evaluation-actor','candidate.pt','--config',str(EVALUATION)]
    assert validate_arguments(argv+['--evaluate-only'])==output
    with pytest.raises(ValueError):validate_arguments(argv)
    with pytest.raises(ValueError):validate_arguments(argv+['--evaluate-only','--resume','latest'])
    output.mkdir()
    with pytest.raises(ValueError):validate_arguments(argv+['--evaluate-only'])
