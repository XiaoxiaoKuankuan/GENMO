"""检查sim2sim跨Torch版本校验的接受边界与拒绝边界。

使用小型真实Tensor和冻结设置模拟服务器/本机版本差异，验证只改变Torch版本
标签能够通过；修改权重、DDIM系数、来源摘要或其他采样参数仍然拒绝。测试不
加载正式模型、不使用GPU、不启动仿真，也不生成训练或部署产物。
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.sim2sim.kernel_compatibility import frozen_content_fingerprint, verify_source_kernel


def fixture_policy():
    model = torch.nn.Module()
    model.register_parameter('weight', torch.nn.Parameter(torch.tensor([[1., 2.]])))
    settings = dict(kernel=dict(numerical_execution=dict(torch_version='local', precision='fp32')),
                    eta=.1)
    policy = SimpleNamespace(frozen_policy=SimpleNamespace(actor=model),
                             _frozen_coefficients={'alpha': torch.tensor([.2, .8])},
                             _frozen_settings=lambda: json.dumps(settings, sort_keys=True, separators=(',', ':')))
    policy.kernel_config = dict(numerical_execution=dict(torch_version='local', precision='fp32'), eta=.1,
                                frozen_prefix=dict(steps=15, content_sha256=frozen_content_fingerprint(policy)))
    source = copy.deepcopy(policy.kernel_config)
    source['numerical_execution']['torch_version'] = 'server'
    source['frozen_prefix']['content_sha256'] = frozen_content_fingerprint(policy, torch_version='server')
    return policy, source


def test_cross_version_checks_full_content_and_preserves_inputs():
    policy, source = fixture_policy()
    before_source, before_local = copy.deepcopy(source), copy.deepcopy(policy.kernel_config)
    report = verify_source_kernel(source, policy)
    assert report['frozen_content_version_metadata_rebased']
    assert source == before_source and policy.kernel_config == before_local


def test_same_version_requires_original_digest():
    policy, _ = fixture_policy()
    assert not verify_source_kernel(copy.deepcopy(policy.kernel_config), policy)['frozen_content_version_metadata_rebased']
    source = copy.deepcopy(policy.kernel_config)
    source['frozen_prefix']['content_sha256'] = 'invalid'
    with pytest.raises(ValueError, match='随机核'):
        verify_source_kernel(source, policy)


@pytest.mark.parametrize('mutation', ['weight', 'coefficient', 'source_digest', 'eta', 'precision', 'frozen_steps'])
def test_cross_version_still_rejects_real_changes(mutation):
    policy, source = fixture_policy()
    if mutation == 'weight':
        with torch.no_grad():
            policy.frozen_policy.actor.weight.add_(1.)
    elif mutation == 'coefficient':
        policy._frozen_coefficients['alpha'].add_(.1)
    elif mutation == 'source_digest':
        source['frozen_prefix']['content_sha256'] = 'invalid'
    elif mutation == 'eta':
        source['eta'] = .2
    elif mutation == 'precision':
        source['numerical_execution']['precision'] = 'bf16'
    else:
        source['frozen_prefix']['steps'] = 16
    with pytest.raises(ValueError, match='随机核'):
        verify_source_kernel(source, policy)
