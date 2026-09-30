"""长期训练执行归档与有意回收 checkpoint 的只读审计验收。

复用已有真实发布／恢复夹具，先生成四轮不同策略版本的完整记录，再在临时目录
将旧轮次原始执行文件打包为无损 tar.gz，并按 latest 两份保护规则回收第一轮权重。
正例同时覆盖旧计数契约和新 decision/attempt 契约；审计必须检查归档内每个文件
的原始 SHA、固定 GAE、跨恢复连续性，并明确旧权重只能复核退休前留存的元数据。
负例注入缺少退休记录、非法最新保护范围、里程碑误删、归档 SHA 和成员 SHA 篡改。
测试不能修改生产训练目录，所有临时字节均由 pytest basetemp 管理并统一回收。
"""
from __future__ import annotations

import copy
import tarfile

import pytest
import torch

from gem.closedloop.dppo.long_run import LongRunMaintenance
from tests.closedloop.dppo.test_stage10_audit_recovery import lifecycle as lifecycle, write
from tools.eval.audit_closedloop_stage10 import audit_run, read_json, sha256


def _archive(directory):
    files = sorted(path for path in directory.rglob('*') if path.is_file()
                   and path.name not in ('summary.json', 'lr_progress.json'))
    records = [dict(path=str(path.relative_to(directory)), size_bytes=path.stat().st_size, sha256=sha256(path))
               for path in files]
    archive = directory/'execution_evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for path in files:
            stream.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    manifest = dict(schema='genmo.closedloop.stage10.execution_archive.v1', archive=archive.name,
        archive_sha256=sha256(archive), archive_size_bytes=archive.stat().st_size,
        original_size_bytes=sum(row['size_bytes'] for row in records), members=records)
    write(directory/'archive_manifest.json', manifest)
    for path in files:
        path.unlink()


def _retire(root, checkpoint):
    publication_path = next(path for path in (root/'checkpoints/publications').glob('*.json')
                            if read_json(path)['path']==str(checkpoint.relative_to(root)))
    publication = read_json(publication_path)
    latest = read_json(root/'latest.json')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    audit_state = {key: saved[key] for key in ('version', 'identity', 'state', 'restore_environment')}
    audit_state['samplers'] = {'music': {key: saved['samplers']['music'][key] for key in ('split', 'catalog_identity')}}
    audit_state['actor_optimizer'] = {'param_groups': [{'lr': group['lr']}
                                                     for group in saved['actor_optimizer']['param_groups']]}
    retirement = dict(schema='genmo.closedloop.stage10.checkpoint_retirement.v1',
        **{key: publication[key] for key in ('iteration', 'path', 'sha256', 'size_bytes')},
        publication=str(publication_path.relative_to(root)),
        replacement={key: latest[key] for key in ('iteration', 'path', 'sha256', 'publication')},
        policy={'keep_last': 2, 'keep_every': 100}, purpose='bounded_checkpoint_retention', audit_state=audit_state)
    path = root/'checkpoints/retired'/f'{checkpoint.stem}.json'
    write(path, retirement)
    checkpoint.unlink()
    return path


def _prepared(life, *, counters=False):
    if counters:
        life.with_counter_contract()
    life.start()
    first = life.publish()
    life.finish()
    life.start(first)
    for _ in range(3):
        life.publish()
    life.finish()
    for directory in sorted((life.root/'sessions').glob('*/iterations/*')):
        _archive(directory)
    return _retire(life.root, first)


@pytest.mark.parametrize('counters', [False, True])
def test_archived_execution_and_retired_resume_checkpoint_are_audited_read_only(lifecycle, counters):
    _prepared(lifecycle, counters=counters)
    before = {str(path.relative_to(lifecycle.root)): sha256(path)
              for path in lifecycle.root.rglob('*') if path.is_file()}
    result = audit_run(lifecycle.root)
    assert result['status']=='passed', result['checks']
    assert result['checkpoint_storage']['intentionally_retired_checkpoints']==1
    assert result['checkpoint_storage']['all_historical_weight_bytes_revalidated'] is False
    iterations = [row for row in result['checks'] if row['name'].startswith('iteration:')]
    assert len(iterations)==4
    assert all(row['details']['rollout']['execution_archive']['lossless_original_bytes_verified'] for row in iterations)
    assert iterations[0]['details']['checkpoint']['storage']=='retired_metadata_only'
    after = {str(path.relative_to(lifecycle.root)): sha256(path)
             for path in lifecycle.root.rglob('*') if path.is_file()}
    assert before==after


@pytest.mark.parametrize('fault', ['missing_retirement', 'protected', 'milestone', 'state', 'archive_sha', 'member_sha'])
def test_retirement_or_archive_tampering_fails(lifecycle, fault):
    path = _prepared(lifecycle, counters=True)
    if fault=='missing_retirement':
        path.unlink()
    elif fault in ('protected', 'milestone', 'state'):
        value = read_json(path)
        if fault=='protected':
            value['policy']['keep_last'] = 4
        elif fault=='milestone':
            value['policy']['keep_every'] = 1
        else:
            value['audit_state']['state']['decision'] += 1
        write(path, value)
    else:
        manifest = next(lifecycle.root.glob('sessions/*/iterations/*/archive_manifest.json'))
        value = copy.deepcopy(read_json(manifest))
        if fault=='archive_sha':
            value['archive_sha256'] = '0'*64
        else:
            value['members'][0]['sha256'] = '0'*64
        write(manifest, value)
    result = audit_run(lifecycle.root)
    assert result['status']=='failed'
    assert result['failed_checks'] > 0


def test_production_archive_and_retirement_feed_read_only_auditor(lifecycle):
    """真实 LongRunMaintenance 生产者与审计器端到端对接，禁止只测自制格式。"""
    life = lifecycle.with_counter_contract().start()
    stage = dict(run_control={'max_walltime_seconds': 604800}, storage={
        'checkpoint_keep_last': 2, 'checkpoint_keep_every': 100, 'archive_completed_iterations': True})
    maintenance = LongRunMaintenance(life.manager, stage)
    first = life.publish()
    maintenance.archive_iteration(life.directory/'iterations'/'000001')
    life.finish()
    life.start(first)
    maintenance = LongRunMaintenance(life.manager, stage)
    removed = []
    for _ in range(3):
        life.publish()
        maintenance.archive_iteration(life.directory/'iterations'/f'{life.state["iteration"]:06d}')
        removed.extend(maintenance.prune_checkpoints())
    assert len(removed)==2
    assert life.manager.latest_checkpoint().is_file()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result['checks']
    assert result['checkpoint_storage']['complete_checkpoints']==2
    assert result['checkpoint_storage']['intentionally_retired_checkpoints']==2
