"""列式完整控制轨迹的还原、静态内容身份和奖励等价性测试。

复用实际奖励输入夹具，比较转换前后每个控制步及四个子步的字段、dtype、shape；
测试故障异构行和静态SHA篡改。编码不触发物理推进，不能据此宣称PhysX已验收。
"""
import copy
from collections.abc import Mapping
import numpy as np
import pytest
from gem.runtime.trajectory_blocks import pack_trace, unpack_trace, expand_feedback
from gem.closedloop.dppo.journal_codec import encode_binary, decode_payload
from tools.profile_closedloop_rpc import compare
from tests.closedloop.dppo.test_data_learning import actual_step


def test_full_control_and_substep_fields_survive_both_codecs():
    rows = [actual_step(i, speed=1.) for i in range(25)]
    block = pack_trace(rows)
    decoded = decode_payload(encode_binary(dict(trace_block=block)))
    result = expand_feedback(decoded)
    compare(rows, result['trace'])
    assert len(result['trace']) == 25
    assert all(len(row['physics_substeps']) == 4 for row in result['trace'])


def test_heterogeneous_fault_rows_and_empty():
    rows = [dict(tick=12, action=np.zeros(21), valid=True), dict(tick=24, valid=False, error='fault')]
    compare(rows, unpack_trace(pack_trace(rows)))
    assert unpack_trace(pack_trace([])) == []


def test_constant_sha_tamper_and_column_count():
    block = pack_trace([dict(source='asset-hash', state=np.arange(4)) for _ in range(2)])
    broken = copy.deepcopy(block)
    broken['columns']['fields']['source']['value'] = 'different-asset'
    with pytest.raises(ValueError, match='SHA'):
        unpack_trace(broken)
    block['count'] = 3
    with pytest.raises(ValueError, match='count'):
        unpack_trace(block)


def test_constant_cache_distinguishes_scalar_types_signed_zero_and_tampering():
    import hashlib
    import json
    from gem.runtime.trajectory_blocks import _scalar_sha
    _scalar_sha.cache_clear()
    for value in (False, 0, 0.0, -0.0, True, 1, 1.0, None, '脚'):
        block = pack_trace([dict(x=value) for _ in range(3)])
        node = block['columns']['fields']['x']
        assert node['sha256'] == hashlib.sha256(json.dumps(value, ensure_ascii=False,
            allow_nan=False, separators=(',', ':')).encode()).hexdigest()
        assert all(type(row['x']) is type(value) for row in unpack_trace(block))
        corrupted = copy.deepcopy(block); corrupted['columns']['fields']['x']['sha256'] = '0'*64
        with pytest.raises(ValueError, match='SHA'): unpack_trace(corrupted)
    assert _scalar_sha.cache_info().maxsize == 8192


def test_lazy_rows_validate_before_access_and_preserve_all_consumed_evidence():
    import pickle
    import json
    rows=[actual_step(i,speed=1.) for i in range(25)]
    block=pack_trace(rows)
    lazy=unpack_trace(block,readonly_views=True,lazy=True)
    assert isinstance(lazy[0],Mapping) and not lazy[0]._cache
    compare(rows,lazy)
    compare(rows,copy.deepcopy(lazy))
    fresh=unpack_trace(block,readonly_views=True,lazy=True)
    compare(rows,pickle.loads(pickle.dumps(fresh)))
    with pytest.raises(TypeError,match='read-only'):lazy[0]['tick']=99
    constant=pack_trace([dict(x=1,nested=dict(y='value'))]*2)
    view=unpack_trace(constant,readonly_views=True,lazy=True)
    with pytest.raises(TypeError,match='JSON serializable'):
        json.dumps(view)
    assert json.loads(json.dumps(copy.deepcopy(view)))==[dict(x=1,nested=dict(y='value'))]*2
    compare(view, decode_payload(encode_binary(view)))
    constant['columns']['fields']['nested']['fields']['y']['sha256']='wrong'
    with pytest.raises(ValueError,match='SHA'):
        unpack_trace(constant,readonly_views=True,lazy=True)
