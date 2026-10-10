"""二进制完整执行证据的编码、持久化与 ACK 故障注入验收。

使用带数组和故障非有限值的真实协议结构，验证 dtype/shape/字段、规范 SHA、
别名与深拷贝一致性、重复身份冲突、旧格式读取和 FULL 事务。故障注入覆盖截断、
篡改、磁盘满、数据库只读、ACK 丢失与断连同序号重试；不启动物理或服务器任务。
"""
import copy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from gem.closedloop.dppo.buffer import StepJournal
from gem.closedloop.dppo.journal_codec import FORMAT, encode_binary, decode_payload
from gem.closedloop.dppo.rpc import AcknowledgedBackend
from tests.closedloop.dppo.test_budget_rpc import Client
from tools.profile_closedloop_rpc import compare


def reply():
    a = np.arange(48, dtype='>f8').reshape(4,12)[:,::2]
    return dict(backend_session_id='session', mutation_seq=1, episode_id='e',
        value=a, same=a, bools=np.array([True,False]), names=np.array(['脚','foot']),
                scalar=np.array(3.,dtype=np.float32), failure=[float('nan'),float('inf'),-float('inf')])


def test_readonly_control_views_preserve_all_values_and_reject_mutation():
    from gem.runtime.trajectory_blocks import pack_trace, unpack_trace
    rows = [dict(tick=i*12, qpos=np.arange(28, dtype=np.float32)+i,
                 substeps=[dict(force=np.arange(6, dtype=np.float64)+j) for j in range(4)],
                 names=['left','right']) for i in range(25)]
    block = pack_trace(rows)
    original = encode_binary(block)
    copied, views = unpack_trace(block), unpack_trace(block, readonly_views=True)
    compare(copied, views)
    for row in views:
        assert np.shares_memory(row['qpos'], block['columns']['fields']['qpos']['values'])
        with pytest.raises(ValueError): row['qpos'][0] = 999.
    views[0]['names'][0] = 'changed'
    assert encode_binary(block) == original


def test_binary_canonical_raw_fields_and_old_read(tmp_path):
    value = reply()
    encoded = encode_binary(value)
    copied = {k:copy.deepcopy(value[k]) for k in reversed(value)}
    assert encode_binary(copied) == encoded
    actual = decode_payload(encoded, sha256=hashlib.sha256(encoded).hexdigest())
    compare(value, actual)
    assert actual['same'] is not actual['value']
    legacy = StepJournal.encode_result(value)
    compare(value, decode_payload(legacy.payload, sha256=legacy.sha256))
    assert hashlib.sha256(encoded).hexdigest() != legacy.sha256
    path = tmp_path/'journal.sqlite'
    with StepJournal(path, format=FORMAT) as journal:
        assert journal.connection.execute('PRAGMA synchronous').fetchone()[0] == 2
        assert journal.append_result(value)
        assert not journal.append_result(copied)
        changed = copy.deepcopy(value); changed['episode_id'] = 'another'
        with pytest.raises(ValueError, match='different payload'):
            journal.append_result(changed)
        payload, digest = journal.connection.execute('SELECT payload,sha256 FROM replies').fetchone()
        assert isinstance(payload, bytes)
        compare(value, decode_payload(payload, sha256=digest))
    with pytest.raises(ValueError, match='format'):
        StepJournal(path)
    with StepJournal(path, format=FORMAT) as journal:
        assert len(journal) == 1


@pytest.mark.parametrize('cut', [1, 8, 23, 40, -1])
def test_truncated_binary_is_rejected(cut):
    encoded = encode_binary(reply())
    with pytest.raises((ValueError, UnicodeDecodeError)):
        decode_payload(encoded[:cut])


def test_sha_tamper_and_trailing_bytes():
    value = encode_binary(reply())
    corrupted = value[:-1]+bytes([value[-1]^1])
    with pytest.raises(ValueError, match='SHA256'):
        decode_payload(corrupted, sha256=hashlib.sha256(value).hexdigest())
    with pytest.raises(ValueError, match='lengths'):
        decode_payload(value+b'X')


@pytest.mark.parametrize('failure', ['readonly','disk_full'])
def test_write_failure_never_ack(tmp_path, failure):
    with StepJournal(tmp_path/'journal.sqlite', format=FORMAT) as journal:
        client = Client()
        backend = AcknowledgedBackend(client, journal)
        if failure == 'readonly':
            journal.connection.execute('PRAGMA query_only=1')
        else:
            def fail(*args):
                raise OSError(28, 'No space left on device')
            journal.append_encoded = fail
        with pytest.raises((OSError,sqlite3.OperationalError)):
            backend.call('reset_episode', seed=42)
        assert not any(m == 'ack' for m,_ in client.calls)
        assert len(journal) == 0


@pytest.mark.parametrize('drop_method', ['execute','ack'])
def test_reconnect_does_not_change_mutation_or_duplicate_record(tmp_path, monkeypatch, drop_method):
    from gem.closedloop.dppo import rpc
    client = Client()
    original = client.call
    failed = [False]
    def call(method, **payload):
        result = original(method, **payload)
        if method == drop_method and not failed[0]:
            failed[0] = True
            raise EOFError('lost reply')
        return result
    client.call = call
    client.close = lambda: None
    monkeypatch.setattr(rpc, 'RpcClient', lambda *a, **kw: client)
    with StepJournal(tmp_path/'journal.sqlite', format=FORMAT) as journal:
        backend = AcknowledgedBackend(client, journal, socket_path='test-only')
        backend.call('reset_episode', seed=42)
        assert len(journal) == 1 and backend.sequence == 1
        assert {p['mutation_seq'] for m,p in client.calls if m=='execute'} == {1}


def test_independent_audit_reads_binary_and_checks_physics(tmp_path):
    from tests.closedloop.dppo.test_audit import envelope
    from tools.eval.audit_closedloop_dppo import check_journal
    path = tmp_path/'journal.sqlite'
    with StepJournal(path, format=FORMAT) as journal:
        journal.append_result(envelope('session',1,'reset_episode',dict(episode_id='e')))
    assert check_journal(path)['counts']['mutations'] == 1


@pytest.mark.parametrize('value', [{1:'invalid'}, {'__journal_array_v2__':{}}, {'__nonfinite__':'nan'}])
def test_scalar_fast_path_preserves_mapping_rejections(value):
    with pytest.raises((TypeError, ValueError)):
        encode_binary(value)


def test_scalar_fast_path_preserves_types_and_fault_markers():
    value = dict(items=[None, True, 1, -0.0, '汉字', float('nan'), float('inf'), -float('inf')])
    decoded = decode_payload(encode_binary(value))
    compare(value, decoded)
    assert np.signbit(decoded['items'][3])


def test_columnar_pickle_snapshot_keeps_bytes_and_source_isolation():
    import io
    import torch
    from gem.runtime.trajectory_blocks import pack_trace,ColumnarTrace
    from gem.closedloop.dppo.buffer import cpu_snapshot
    rows=[dict(v=np.array([i,-0.],dtype='>f8'),zero=(-0. if i%2 else 0.)) for i in range(25)]
    block=pack_trace(rows);trace=ColumnarTrace([block])
    cloned=cpu_snapshot(list(trace))
    output=io.BytesIO();torch.save(cloned,output);output.seek(0)
    restored=torch.load(output,weights_only=False)
    for i,row in enumerate(restored):
        assert row['v'].dtype==rows[i]['v'].dtype and row['v'].tobytes()==rows[i]['v'].tobytes()
        assert np.signbit(row['zero'])==np.signbit(rows[i]['zero'])
    block['columns']['fields']['v']['values'][:]=99
    assert cloned[0]['v'][0]==0 and restored[0]['v'][0]==0
    # 所有行引用同一个独立列快照，没有变成25份大数组。
    assert all(row._fields is cloned[0]._fields for row in cloned)


def test_columnar_join_preserves_non_native_array_bytes_and_signed_zero():
    from gem.runtime.trajectory_blocks import pack_trace, ColumnarTrace, concatenate_trace_blocks, unpack_trace
    values=[np.asarray([0.,-0.,i],dtype='>f8') for i in range(4)]
    blocks=[pack_trace([{'state':v} for v in values[:2]]),pack_trace([{'state':v} for v in values[2:]])]
    column=ColumnarTrace(blocks).column('state')
    assert column.dtype==values[0].dtype
    assert column.tobytes()==b''.join(v.tobytes() for v in values)
    joined=unpack_trace(concatenate_trace_blocks(blocks),readonly_views=True)
    for row,original in zip(joined,values,strict=True):
        assert row['state'].dtype==original.dtype
        assert row['state'].tobytes()==original.tobytes()


@pytest.mark.parametrize('fault',[False,True])
def test_fast_c_json_encoder_matches_reference_complete_bytes(fault):
    from pathlib import Path
    from gem.closedloop.dppo.journal_codec import encode_binary_reference
    from gem.runtime.trajectory_blocks import pack_trace
    value=reply()
    if not fault:value.pop('failure')
    value['path']=Path('有空 格/证据')
    value['zeros']=[0.,-0.]
    value['trace']=pack_trace([dict(tick=i*12,position=np.arange(21,dtype='>f4')+i,
        physics=[{'force':np.linspace(-1,1,6,dtype=np.float64)} for _ in range(4)]) for i in range(25)])
    assert encode_binary(value)==encode_binary_reference(value)
    for broken in ({1:'bad'},{'__journal_array_v2__':{}},{'__nonfinite__':'nan'}):
        with pytest.raises((TypeError,ValueError)):encode_binary(broken)
