"""GPU合批搬运的结构等价、引用寿命和异常路径回归。

从两个仓库抽取实际生产函数，避免启动Isaac或创建机器人世界。用独立递归CPU
复制验证数值、容器、共享对象与dtype；关闭自动循环GC后用弱引用检查输入张量
能否立即释放，覆盖正常、收集、拼接和重建异常。CUDA用例另外测量连续调用后的
真实allocated字节，不把reserved缓存当作泄漏，也不调用gc掩盖引用滞留。
全部测试只使用pytest临时目录；不加载训练checkpoint，不证明物理训练收益。
"""
import ast
import gc
import os
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch


@pytest.fixture(params=['genmo', 'gmt'])
def copier(request):
    root = Path(__file__).resolve().parents[3]
    if request.param == 'genmo':
        path = root/'gem/closedloop/dppo/vector_generation.py'
        name = 'bulk_cpu_copy'
    else:
        root = Path(os.environ.get('GMT_REPO', str(root.parent/'legged_lab_gmt')))
        path = root/'source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop/vector_diagnostics.py'
        name = 'to_host'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name], request.param


def assert_tree(actual, expected):
    if torch.is_tensor(expected):
        if isinstance(actual, np.ndarray):
            actual = torch.from_numpy(actual)
        assert actual.device.type == 'cpu'
        assert actual.dtype == expected.dtype
        assert not actual.requires_grad
        torch.testing.assert_close(actual, expected.detach().cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree(a, b)
    else:
        assert actual == expected


def test_values_aliases_and_no_source_retention(copier):
    copy, _ = copier
    source = torch.arange(24., requires_grad=True).reshape(4, 6).detach().requires_grad_()
    tree = dict(a=source, alias=source, nested=[source.T, (source.double(), None)],
                empty=torch.empty(0), integer=torch.tensor(17), mask=source > 10, text='保持')
    result = copy(tree)
    assert_tree(result, tree)
    assert result['a'] is result['alias']
    source_ref = weakref.ref(source)
    enabled = gc.isenabled()
    gc.disable()
    try:
        del source, tree
        assert source_ref() is None
        # 输出仍有完整CPU数据，不能通过销毁返回值来“修复”泄漏。
        assert float(result['a'][0, 1]) == 1.
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize('failure', ['collect', 'transfer', 'rebuild'])
def test_exception_does_not_leave_recursive_cuda_owners(copier, monkeypatch, failure):
    copy, _ = copier

    class Broken(dict):
        def values(self):
            yield self['a']
            if failure == 'collect':
                raise RuntimeError('expected collect failure')

        def items(self):
            raise RuntimeError('expected rebuild failure')

    def fail_cat(*args, **kwargs):
        raise RuntimeError('expected transfer failure')

    if failure == 'transfer':
        monkeypatch.setattr(torch, 'cat', fail_cat)
    enabled = gc.isenabled()
    gc.disable()
    try:
        source = torch.ones(16)
        ref = weakref.ref(source)
        tree = Broken(a=source)
        # 不使用pytest.raises保存异常traceback，避免测试本身成为输入所有者。
        try:
            copy(tree)
        except RuntimeError as error:
            assert 'expected' in str(error)
        else:
            pytest.fail('必须传播原始异常')
        del source, tree
        assert ref() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要服务器1 CUDA')
def test_cuda_allocated_flat_without_cyclic_gc(copier):
    copy, _ = copier
    device = torch.device('cuda:0')
    torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(4):
            source = torch.ones(16 * 1024**2, device=device)
            ref = weakref.ref(source)
            result = copy(dict(a=source, again=source))
            assert result['a'] is result['again']
            del source
            assert ref() is None
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == baseline
            del result
    finally:
        if enabled:
            gc.enable()
