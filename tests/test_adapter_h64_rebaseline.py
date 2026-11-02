"""H64八组重设及历史基线隔离机制的CPU回归，所有夹具只写pytest临时目录。

验证四种参数组合各有双种子、实际Adam降率时机及u2600固定身份；使用含buffer的
真实Torch网络，验证临时换权重正常结束、评价异常和部分加载异常时均完整恢复，
且不替换Parameter对象。缓存测试防止模型、歌曲、物理或评价预算变动后静默复用。
这些测试不启动仿真，不表示新方案超过旧模型；质量必须由完整真实物理评价判断。
"""

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'gem/closedloop/reference_adapter'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_h64_matrix_and_early_lr_boundaries():
    combinations = set()
    lr_at = module('alternating_entry').adapter_learning_rate
    for gpu in range(8):
        c = yaml.safe_load((ROOT/f'configs/closedloop/stage2_h64_rebaseline_20261102/gpu{gpu}.yaml').read_text())
        a, t = c['alternating'], c['training']
        stability = (gpu//2)%2
        combinations.add((stability, t['adapter_lr'], t['seed']))
        assert c['bptt_steps'] == 64 and c['vector']['num_envs'] == 2048
        assert c['vector']['target_accepted_updates'] == 30000
        assert t['seed'] == c['vector']['sampling_seed'] == 42+gpu%2
        assert a['precision']['root_xy_weight'] == 1.4910286903381347
        assert a['precision']['root_tracking_bonus_weight'] == 0
        assert a['precision']['execution_jerk_weight'] == pytest.approx(.10504519010557042*stability)
        assert a['precision']['stance_foot_weight'] == pytest.approx(.41037086737445894*stability)
        assert a['fixed_music_evaluation']['adapter_baseline']['source_adapter_updates'] == 2600
        assert a['evaluation_num_envs'] == 127 and a['full_song_from_update'] == 100
        assert not a['critic_enabled'] and not a['root_xy']
        assert t['adapter_lr_milestones'] == [1000, 3000, 10000, 20000]
        assert [lr_at(t, n)/t['adapter_lr'] for n in (999, 1000, 2999, 3000, 10000, 20000)] == [1, .5, .5, .25, .125, .0625]
    assert combinations == {(s, lr, seed) for s in (0, 1) for lr in (1e-5, 5e-6) for seed in (42, 43)}


@pytest.mark.parametrize('failure', ['none', 'evaluation', 'load'])
def test_temporary_baseline_always_restores_parameters_and_buffers(tmp_path, failure):
    net = torch.nn.Linear(2, 2)
    net.register_buffer('bounds', torch.tensor([1., 2.]))
    original = {k: v.clone() for k, v in net.state_dict().items()}
    parameter_ids = [id(p) for p in net.parameters()]
    baseline = {k: v+3 for k, v in original.items()}
    if failure == 'load':
        baseline['bias'] = torch.ones(3)
    path = tmp_path/'old.pt'
    torch.save({'adapter': baseline}, path)
    def execute():
        with module('fixed_music_panel').temporary_adapter_checkpoint(net, path):
            assert all(torch.equal(net.state_dict()[k], v) for k, v in baseline.items())
            if failure == 'evaluation':
                raise RuntimeError('evaluation failed')
    if failure == 'none':
        execute()
    else:
        with pytest.raises(RuntimeError):
            execute()
    assert all(torch.equal(net.state_dict()[k], v) for k, v in original.items())
    assert [id(p) for p in net.parameters()] == parameter_ids


def test_baseline_cache_identity_and_conditions(tmp_path):
    m = module('fixed_music_panel')
    path = tmp_path/'identity.pt'; path.write_bytes(b'fixed identity')
    settings = dict(label='u2600', checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_adapter_updates=2600)
    identity = m.adapter_baseline_identity(settings)
    metadata = dict(evaluable_tasks_sha256='songs', requested_song_count=128)
    report = dict(metadata, adapter_baseline_identity=identity, num_envs=127, max_steps=None, sm_mdp_contract={'physics': 1})
    def check(r):
        m.validate_adapter_baseline_cache(r, identity, metadata, num_envs=127, max_steps=None, mdp_contract={'physics': 1})
    check(report)
    for key, value in [('num_envs', 64), ('max_steps', 256), ('evaluable_tasks_sha256', 'other'),
                       ('sm_mdp_contract', {'physics': 2}), ('adapter_baseline_identity', dict(identity, checkpoint_sha256='other'))]:
        changed = copy.deepcopy(report); changed[key] = value
        with pytest.raises(ValueError, match='不同'):
            check(changed)
    path.write_bytes(b'replaced model')
    with pytest.raises(ValueError, match='SHA'):
        m.adapter_baseline_identity(settings)
