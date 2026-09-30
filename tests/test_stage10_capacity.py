"""正式训练容量估计的CPU边界测试。

使用临时小文件代表完整checkpoint和执行证据，验证最大实测增量、重试余量、已有
运行字节、文件系统保留空间与run配额分别约束；拒绝没有完成的测量与越界路径。
测试不加载模型、不连接服务器、不启动GPU，不能替代正式服务器的实际空闲检查。
"""
from __future__ import annotations

import json
import hashlib

import pytest

from tools.eval.plan_closedloop_stage10_capacity import plan_capacity


def reference(tmp_path):
    root = tmp_path/'reference'
    for index in (1, 2):
        directory = root/'sessions'/'one'/'iterations'/f'{index:06d}'
        directory.mkdir(parents=True)
        checkpoint = root/'checkpoints'/f'{index}.pt'
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(b'c'*(1000+index))
        (directory/'trace.pt').write_bytes(b't'*(100+index))
        (directory/'summary.json').write_text(json.dumps(dict(status='accepted',iteration=index,
            checkpoint=str(checkpoint.relative_to(root)))))
        descriptor = dict(schema='genmo.closedloop.stage10.checkpoint_publication.v1',iteration=index,
            path=str(checkpoint.relative_to(root)),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            size_bytes=checkpoint.stat().st_size,session_id='one')
        publication = root/'checkpoints'/'publications'/f'{index}.json'
        publication.parent.mkdir(exist_ok=True)
        publication.write_text(json.dumps(descriptor))
        (root/'latest.json').write_text(json.dumps(dict(descriptor,publication=str(publication.relative_to(root)))))
    return root


def configuration():
    return dict(stage10=dict(limits=dict(accepted_iterations=40),
        storage=dict(checkpoint_reserve_bytes=500, min_free_bytes=1000, max_run_bytes=100000)))


def test_capacity_uses_measured_maxima_and_separate_reserves(tmp_path):
    root = reference(tmp_path)
    result = plan_capacity(root,tmp_path/'new',configuration(),free_bytes=1000000)
    assert result['status'] == 'passed'
    assert result['remaining_iterations'] == 40
    assert result['projected_total_bytes'] == result['projected_additional_bytes']
    assert result['max_iteration_bytes'] == 1002+max(row['execution_evidence_bytes'] for row in result['measurements'])
    continued = plan_capacity(root,root,configuration(),free_bytes=1000000)
    assert continued['remaining_iterations'] == 38
    assert continued['projected_total_bytes'] == continued['target_existing_bytes']+continued['projected_additional_bytes']
    assert continued['target_existing_bytes'] == result['observed_run_bytes']


def test_capacity_fails_filesystem_and_run_quota_independently(tmp_path):
    root = reference(tmp_path)
    config = configuration()
    result = plan_capacity(root,tmp_path/'new',config,free_bytes=1000)
    assert result['checks'] == dict(within_run_quota=True,filesystem_keeps_free_reserve=False)
    config['stage10']['storage']['max_run_bytes']=1
    result = plan_capacity(root,tmp_path/'new',config,free_bytes=1000000)
    assert result['checks'] == dict(within_run_quota=False,filesystem_keeps_free_reserve=True)


def test_capacity_rejects_incomplete_or_escaping_evidence(tmp_path):
    root = reference(tmp_path)
    path = root/'sessions'/'one'/'iterations'/'000002'/'summary.json'
    saved = json.loads(path.read_text())
    path.write_text(json.dumps(dict(saved,status='failed')))
    with pytest.raises(ValueError,match='at least two'):
        plan_capacity(root,tmp_path/'new',configuration(),free_bytes=1000000)
    outside = tmp_path/'outside.pt'
    outside.write_bytes(b'x')
    path.write_text(json.dumps(dict(saved,checkpoint='../outside.pt')))
    with pytest.raises(ValueError,match='inside the reference'):
        plan_capacity(root,tmp_path/'new',configuration(),free_bytes=1000000)


def test_unpublished_accepted_summary_does_not_reduce_remaining_iterations(tmp_path):
    root = reference(tmp_path)
    directory = root/'sessions'/'one'/'iterations'/'000003'
    directory.mkdir()
    checkpoint = root/'checkpoints'/'3.pt'
    checkpoint.write_bytes(b'c'*1003)
    (directory/'summary.json').write_text(json.dumps(dict(status='accepted',iteration=3,checkpoint='checkpoints/3.pt')))
    result = plan_capacity(root,root,configuration(),free_bytes=1000000)
    assert result['existing_accepted_iterations'] == 2
    assert result['remaining_iterations'] == 38
    assert result['target_existing_bytes'] >= checkpoint.stat().st_size
    (root/'latest.json').unlink()
    result = plan_capacity(root,root,configuration(),free_bytes=1000000)
    assert result['existing_accepted_iterations'] == 0 and result['remaining_iterations'] == 40


def test_capacity_does_not_trust_changed_latest_checkpoint(tmp_path):
    root = reference(tmp_path)
    (root/'checkpoints'/'2.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='size or SHA differs'):
        plan_capacity(root,root,configuration(),free_bytes=1000000)


@pytest.mark.parametrize('factor,retries',[(.9,2),(float('nan'),2),(1.25,-1),(1.25,True)])
def test_capacity_rejects_invalid_margin_parameters(tmp_path,factor,retries):
    with pytest.raises(ValueError):
        plan_capacity(reference(tmp_path),tmp_path/'new',configuration(),
                      free_bytes=1000000,safety_factor=factor,retry_iterations=retries)
