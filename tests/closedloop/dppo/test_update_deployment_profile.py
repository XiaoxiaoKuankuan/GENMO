"""部署标定回写工具的纯CPU验证：只改 deployment_profile 段、摘要与训练入口一致、拒绝不一致标定。

复制固定 dppo_contract.yaml 测试夹具 到临时目录，构造八rank标定报告；检查回写后注释与其余配置
逐字保留、样本按rank顺序拼接、kernel摘要能通过训练入口的同一校验，且rank间kernel不一致
时在写文件前拒绝。不运行Isaac、不访问GPU。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from gem.closedloop.dppo.training import _validate_calibrated_execution
from tools.update_deployment_profile import update

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'tests/closedloop/fixtures/dppo_contract.yaml'


def _report(profile, kernel):
    calibrations = []
    for rank in range(8):
        calibrations.append(dict(durations=[[.2 + .001*rank + .0001*i for i in range(8)]],
            numerical_execution=copy.deepcopy(profile['numerical_execution']), kernel=copy.deepcopy(kernel),
            assets=copy.deepcopy(profile['measured_assets'])))
    return dict(status='passed', scope='single_request_deployment_calibration_not_training', calibrations=calibrations)


def test_profile_rewrite_preserves_other_text_and_matches_training_check(tmp_path):
    config = tmp_path / 'stage10.yaml'
    original = CONFIG.read_text()
    config.write_text(original)
    profile = yaml.safe_load(original)['timing']['deployment_profile']
    kernel = dict(version='candidate', steps=20, numerical_execution=dict(precision='bf16_backbone'))
    report = tmp_path / 'deployment_calibration.json'
    report.write_text(json.dumps(_report(profile, kernel)))
    result = update(config, report)
    text = config.read_text()
    assert text.splitlines()[0] == original.splitlines()[0]
    assert text.split('  deployment_profile:')[0] == original.split('  deployment_profile:')[0]
    parsed = yaml.safe_load(text)['timing']['deployment_profile']
    assert parsed['samples_seconds'][:2] == [.2, .2001] and len(parsed['samples_seconds']) == 64
    assert parsed['provenance'][0]['path'] == str(report.resolve())
    _validate_calibrated_execution(parsed, profile['numerical_execution'], profile['measured_assets'], kernel)
    assert result['behavior_kernel_sha256'] == parsed['behavior_kernel_sha256']


def test_inconsistent_rank_kernel_is_rejected_before_writing(tmp_path):
    config = tmp_path / 'stage10.yaml'
    config.write_text(CONFIG.read_text())
    profile = yaml.safe_load(CONFIG.read_text())['timing']['deployment_profile']
    data = _report(profile, dict(version='a'))
    data['calibrations'][3]['kernel'] = dict(version='b')
    report = tmp_path / 'deployment_calibration.json'
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='disagree on kernel'):
        update(config, report)
    assert config.read_text() == CONFIG.read_text()
