"""统一世界窗口的变长与空片段学习边界验收。

使用真实Buffer/GAE/cache接口验证空环境与空rank不构造伪动作，不调用Critic，
不跨Actor版本拼接优势；固定世界时间的配置没有每环境配额，BC总量可以按实际
optimizer尾批分配而无需整除世界规模。本测试不替代真实PhysX和八卡梯度验收。
"""
from pathlib import Path
import pytest
import torch
import yaml
from gem.closedloop.dppo.memory_execution import DeviceRolloutBuffer
from gem.closedloop.dppo.dual_collector import fixed_fragment_targets
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.training_config import configuration


def test_empty_local_world_buffer_has_no_virtual_transition_or_critic_call():
    class Critic:
        def __call__(self,*args):raise AssertionError('empty rank must not call Critic')
    buffer=DeviceRolloutBuffer([[],[]],None,'cpu')
    rows,targets=fixed_fragment_targets([[],[]],Critic(),'cpu',critic_version=3)
    assert len(buffer)==len(rows)==0 and targets['returns'].shape==(0,)
    cache=RolloutTensorCache(rows,targets,'cpu')
    assert cache.report()['bytes']==0 and cache.report()['upper_transitions']==0
    cache.close();buffer.clear()


def test_current_config_accepts_nondivisible_bc_total_without_environment_quota(tmp_path):
    root=Path(__file__).resolve().parents[3]
    config=yaml.safe_load((root/'configs/closedloop/stage10_server1.yaml').read_text())
    config['stage10']['training']['bc_samples_per_rollout']=517
    path=tmp_path/'candidate.yaml';path.write_text(yaml.safe_dump(config))
    resolved=configuration(path)
    assert 'decisions_per_environment' not in resolved['stage10']['scale']
    assert resolved['stage10']['training']['bc_samples_per_rollout']==517
    config['stage10']['scale']['decisions_per_environment']=8
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='quotas'):configuration(path)
