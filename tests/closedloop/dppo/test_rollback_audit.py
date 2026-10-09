"""异常回滚的完整字节摘要测试，防止只比较打印值或只核验模型而漏掉 Adam/RNG。

构造大于 NumPy 默认摘要展示长度的 RNG 数组，改变中间元素；同时覆盖张量位差、
dtype、容器结构和 float 符号零。仅使用小型内存状态，不读取正式训练数据。
"""
import copy
import numpy as np
import torch
from gem.closedloop.dppo.rollback_audit import exact_state_digest, compare_recovered_state


def test_middle_rng_byte_and_optimizer_value_are_not_elided():
    expected = dict(rng=np.arange(2048, dtype=np.uint32), adam=torch.ones(4))
    actual = copy.deepcopy(expected)
    assert compare_recovered_state(expected, actual)['identical']
    actual['rng'][1024] += 1
    actual['adam'][2] = torch.nextafter(actual['adam'][2], torch.tensor(2.))
    report = compare_recovered_state(expected, actual)
    assert not report['identical']
    assert all(not field['identical'] for field in report['fields'].values())


def test_digest_binds_dtype_shape_and_scalar_bytes():
    values = [torch.ones(2), torch.ones(2, dtype=torch.float64), torch.ones(1, 2),
              [1, 2], (1, 2), 0., -0., np.ones(2)]
    assert len({exact_state_digest(value) for value in values}) == len(values)
    assert exact_state_digest({'a': 1, 'b': 2}) == exact_state_digest({'b': 2, 'a': 1})
