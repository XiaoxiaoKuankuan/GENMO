"""内存维护快照的严格源码变更边界及完整学习状态检查。

验证维护入口只接受原函数体原样包入指定finally的唯一变换，奖励、计算、模块
其他函数或异常处理有任何额外变化均拒绝。用含模型、Adam、BC、RNG和进度的
小快照检查完整读回逐值比对；不修改正式run，不把小快照当作真实恢复验收。
服务器上的真实维护另有全2.6GB状态读回、父SHA和源码AST证据。
"""
import copy

import numpy as np
import pytest
import torch

from tools.repair_host_copy_checkpoint import same_state, verify_cleanup_only


OLD = '''def move(x):
    """原说明。"""
    groups = {}; converted = {}
    def visit(a):
        if a: visit(a - 1)
    visit(x)
    return x * 2
'''
NEW = '''def move(x):
    """原说明。"""
    groups = {}; converted = {}
    try:
        def visit(a):
            if a: visit(a - 1)
        visit(x)
        return x * 2
    finally:
        groups.clear()
        converted.clear()
        visit = rebuild = None
'''
CLEANUP = 'groups.clear()\nconverted.clear()\nvisit = rebuild = None'


def test_exact_cleanup_proof():
    assert verify_cleanup_only(OLD, NEW, 'move', CLEANUP)['original_statements_unchanged']


@pytest.mark.parametrize('changed', [NEW.replace('x * 2', 'x * 3'),
    NEW.replace('groups.clear()', 'groups.pop(0)'), NEW+'\ny = 1\n',
    NEW.replace('    finally:', '    except Exception:\n        pass\n    finally:')])
def test_extra_code_changes_rejected(changed):
    with pytest.raises(ValueError):
        verify_cleanup_only(OLD, changed, 'move', CLEANUP)


def test_full_state_roundtrip_and_mutation_rejection(tmp_path):
    original = dict(actor={'w': torch.randn(3, 2)}, critic={'v': torch.ones(2)},
        optimizer={'state': {0: {'step': torch.tensor(270.), 'exp_avg': torch.randn(3, 2)}}},
        bc={'samples': 2048}, rng=[np.arange(5, dtype=np.uint32), torch.arange(10)],
        state={'iteration': 34, 'actor_updates': 270})
    path = tmp_path/'checkpoint.pt'
    torch.save(original, path)
    restored = torch.load(path, weights_only=False, mmap=True)
    same_state(original, restored)
    for change in ('actor', 'optimizer', 'rng', 'state'):
        candidate = copy.deepcopy(restored)
        if change == 'actor':
            candidate['actor']['w'][0, 0] += 1
        elif change == 'optimizer':
            candidate['optimizer']['state'][0]['step'] += 1
        elif change == 'rng':
            candidate['rng'][0][0] += 1
        else:
            candidate['state']['iteration'] += 1
        with pytest.raises(ValueError):
            same_state(original, candidate)
