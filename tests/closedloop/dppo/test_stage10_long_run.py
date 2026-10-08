"""七天训练截止、执行证据归档和检查点保留策略的真实CPU文件生命周期测试。

测试使用真实RunManager排他锁、JSON原子发布、tar.gz归档和最小torch checkpoint，
不连接GPU或物理服务，不复制生产算法构造预期值。固定验证首次run创建时间决定
七天绝对截止，恢复不会延长或悄悄改变保留策略；完整二进制执行数据可以从归档
逐字节取回、文件与归档SHA一致，summary和学习率诊断仍可直接读取。

检查点测试通过真实发布接口形成latest及不可变发布凭证，核验initial、最近两份
和每100轮里程碑保留，普通旧文件回收前留下退休证明。未发布文件、内容被修改的
旧checkpoint、损坏的latest和归档校验失败均不能触发相应原件的静默删除。
全部文件使用pytest临时目录，调用方使用TemporaryDirectory作为basetemp并禁用缓存。
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
import tarfile

import pytest
import torch

from gem.closedloop.dppo.checkpoint import VERSION
from gem.closedloop.dppo import long_run
from gem.closedloop.dppo.long_run import LongRunMaintenance
from gem.closedloop.dppo.run_management import RunManager, file_sha256


def stage_settings():
    return dict(run_control=dict(max_walltime_seconds=604800), storage=dict(
        checkpoint_keep_last=2, checkpoint_keep_every=100,
        archive_completed_iterations=True))


def write_checkpoint(manager, iteration, *, publish=True, filename=None):
    """最小真实torch文件满足发布契约；此测试不声称验证网络或优化器恢复。"""
    path = manager.run_dir / 'checkpoints' / (filename or f'stage10_{iteration:06d}.pt')
    path.parent.mkdir(exist_ok=True)
    payload = {key: {} for key in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer',
        'rng', 'samplers', 'identity', 'config', 'optimizer_layout', 'restore_environment')}
    payload.update(version=VERSION, state=dict(iteration=iteration, buffer_size=0,
                                               pending_plan=False))
    payload['actor'] = dict(weight=torch.tensor([float(iteration)]))
    payload['samplers'] = dict(music=dict(split='train', catalog_identity='synthetic-complete-catalog'))
    payload['actor_optimizer'] = dict(param_groups=[dict(lr=1e-9)])
    torch.save(payload, path)
    if publish:
        manager.publish_checkpoint(iteration, path)
    return path


def write_execution(manager, iteration=1, *, status='accepted'):
    directory = manager.iteration_dir(iteration)
    evidence = {
        'rollout/chunk_000001.pt': b'rollout-with-denoising-chain\x00\xff',
        'raw_samples/sample.bin': bytes(range(256)) * 8,
        'execution_journal.sqlite': b'SQLite format 3\x00actual-execution-records',
        'fixed_targets.pt': b'fixed-values-and-advantages',
    }
    for relative, data in evidence.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (directory / 'summary.json').write_text(json.dumps(dict(status=status, iteration=iteration)))
    (directory / 'lr_progress.json').write_text('{"accepted_lr": 2e-9}')
    return directory, evidence


def test_seven_day_deadline_is_created_at_and_resume_does_not_extend(tmp_path):
    output = tmp_path / 'run'
    with RunManager(output) as manager:
        maintenance = LongRunMaintenance(manager, stage_settings())
        created = datetime.fromisoformat(json.loads((output / 'run_manifest.json').read_text())['created_at'])
        deadline = created + timedelta(days=7)
        assert maintenance.deadline == deadline
        assert not maintenance.expired(deadline - timedelta(microseconds=1))
        assert maintenance.expired(deadline)
        assert maintenance.expired(deadline + timedelta(days=1))
        original_policy = (output / 'long_run_policy.json').read_bytes()
    with RunManager(output, resume=True) as manager:
        resumed = LongRunMaintenance(manager, stage_settings())
        assert resumed.deadline == deadline
        assert resumed.expired(deadline)
        assert (output / 'long_run_policy.json').read_bytes() == original_policy


@pytest.mark.parametrize('section,key,value', [
    ('run_control', 'max_walltime_seconds', 604801),
    ('storage', 'checkpoint_keep_last', 3),
    ('storage', 'checkpoint_keep_every', 200),
    ('storage', 'archive_completed_iterations', False),
])
def test_resume_rejects_changed_deadline_or_retention_without_overwriting_policy(
        tmp_path, section, key, value):
    output = tmp_path / 'run'
    with RunManager(output) as manager:
        LongRunMaintenance(manager, stage_settings())
    original_policy = (output / 'long_run_policy.json').read_bytes()
    changed = copy.deepcopy(stage_settings())
    changed[section][key] = value
    with RunManager(output, resume=True) as manager:
        with pytest.raises(ValueError, match='cannot change'):
            LongRunMaintenance(manager, changed)
    assert (output / 'long_run_policy.json').read_bytes() == original_policy


def test_archive_preserves_all_binary_evidence_and_readable_summaries(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage_settings())
        write_checkpoint(manager, 1)
        directory, evidence = write_execution(manager)
        summaries = {name: (directory / name).read_bytes()
                     for name in ('summary.json', 'lr_progress.json')}
        result = maintenance.archive_iteration(directory)
        archive = directory / result['archive']
        manifest = json.loads((directory / 'archive_manifest.json').read_text())
        assert manifest == result
        assert result['archive_sha256'] == file_sha256(archive)
        assert result['original_size_bytes'] == sum(map(len, evidence.values()))
        assert {record['path'] for record in result['members']} == set(evidence)
        with tarfile.open(archive, 'r:gz') as stream:
            restored = {member.name: stream.extractfile(member).read() for member in stream}
        assert restored == evidence
        for record in result['members']:
            assert record['sha256'] == hashlib.sha256(evidence[record['path']]).hexdigest()
            assert record['size_bytes'] == len(evidence[record['path']])
            assert not (directory / record['path']).exists()
        assert {name: (directory / name).read_bytes() for name in summaries} == summaries
        assert not list(directory.glob('.execution.*.tmp'))
        assert {path.name for path in directory.iterdir()} == {
            'summary.json', 'lr_progress.json', 'archive_manifest.json', 'execution_evidence.tar.gz'}
        with pytest.raises(FileExistsError, match='immutable'):
            maintenance.archive_iteration(directory)


@pytest.mark.parametrize('status', ['running', 'failed', 'rejected'])
def test_archive_rejects_unaccepted_iteration_and_keeps_originals(tmp_path, status):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage_settings())
        write_checkpoint(manager, 1)
        directory, evidence = write_execution(manager, status=status)
        with pytest.raises(ValueError, match='Only accepted'):
            maintenance.archive_iteration(directory)
        assert all((directory / name).read_bytes() == data for name, data in evidence.items())
        assert not (directory / 'archive_manifest.json').exists()


def test_archive_rejects_iteration_without_matching_checkpoint_publication(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage_settings())
        write_checkpoint(manager, 1)
        directory, evidence = write_execution(manager, iteration=2)
        with pytest.raises(ValueError, match='exact accepted iteration'):
            maintenance.archive_iteration(directory)
        assert all((directory / name).read_bytes() == data for name, data in evidence.items())
        assert not (directory / 'execution_evidence.tar.gz').exists()


def test_archive_detects_changed_source_and_never_deletes_unverified_originals(tmp_path, monkeypatch):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage_settings())
        write_checkpoint(manager, 1)
        directory, evidence = write_execution(manager)
        source = directory / 'fixed_targets.pt'
        original_sha = long_run.file_sha256
        changed = False

        def mutate_after_hash(path):
            nonlocal changed
            digest = original_sha(path)
            if path == source and not changed:
                source.write_bytes(b'X' * len(evidence['fixed_targets.pt']))
                changed = True
            return digest

        monkeypatch.setattr(long_run, 'file_sha256', mutate_after_hash)
        with pytest.raises(ValueError, match='Archive SHA differs'):
            maintenance.archive_iteration(directory)
        assert changed and source.read_bytes() == b'X' * len(evidence['fixed_targets.pt'])
        for name, data in evidence.items():
            if name != 'fixed_targets.pt':
                assert (directory / name).read_bytes() == data
        assert not (directory / 'execution_evidence.tar.gz').exists()
        assert not (directory / 'archive_manifest.json').exists()
        assert not list(directory.glob('.execution.*.tmp'))


def test_checkpoint_retention_keeps_initial_latest_two_milestones_and_unpublished(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage_settings())
        initial = write_checkpoint(manager, 0, publish=False, filename='initial.pt')
        paths = {number: write_checkpoint(manager, number) for number in (1, 2, 99, 100, 101, 102)}
        unpublished = write_checkpoint(manager, 777, publish=False)
        unpublished_bytes = unpublished.read_bytes()
        removed = maintenance.prune_checkpoints()
        assert set(removed) == {str(paths[number].relative_to(manager.run_dir)) for number in (1, 2, 99)}
        assert initial.is_file() and unpublished.read_bytes() == unpublished_bytes
        assert all(paths[number].is_file() for number in (100, 101, 102))
        assert manager.latest_checkpoint() == paths[102]
        for number in (1, 2, 99):
            assert not paths[number].exists()
            retirement = manager.run_dir / 'checkpoints/retired' / f'{paths[number].stem}.json'
            record = json.loads(retirement.read_text())
            assert record['iteration'] == number and record['replacement']['iteration'] == 102
            assert record['policy'] == dict(keep_last=2, keep_every=100)
            assert record['audit_state']['state']['iteration'] == number
            assert record['audit_state']['samplers']['music']['split'] == 'train'
            assert (manager.run_dir / record['publication']).is_file()
        assert maintenance.prune_checkpoints() == []


@pytest.mark.parametrize('modified_iteration', [1, 3])
def test_changed_old_or_latest_checkpoint_prevents_deletion(tmp_path, modified_iteration):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage_settings())
        paths = {number: write_checkpoint(manager, number) for number in (1, 2, 3)}
        paths[modified_iteration].write_bytes(paths[modified_iteration].read_bytes() + b'tampered')
        before = {number: path.read_bytes() for number, path in paths.items()}
        with pytest.raises(ValueError, match='changed contents|different SHA'):
            maintenance.prune_checkpoints()
        assert {number: path.read_bytes() for number, path in paths.items()} == before
        assert not (manager.run_dir / 'checkpoints/retired').exists()


def test_retention_keeps_best_saved_extra_checkpoint_across_controlled_end_resumes(tmp_path):
    settings = stage_settings()
    settings['storage']['checkpoint_keep_every'] = 300
    output = tmp_path / 'run'
    with RunManager(output) as manager:
        maintenance = LongRunMaintenance(manager, settings)
        paths = {number: write_checkpoint(manager, number) for number in (300, 317, 325, 340)}
        best = dict(saved=True, iteration=317, checkpoint=dict(iteration=317,
            path=str(paths[317].relative_to(output)), sha256=file_sha256(paths[317])))
        (output / 'best_saved.json').write_text(json.dumps(best))
        assert maintenance.prune_checkpoints() == []
    with RunManager(output, resume=True) as manager:
        maintenance = LongRunMaintenance(manager, settings)
        paths[360] = write_checkpoint(manager, 360)
        assert maintenance.prune_checkpoints() == [str(paths[325].relative_to(output))]
        assert all(paths[number].is_file() for number in (300, 317, 340, 360))
        assert file_sha256(paths[317]) == best['checkpoint']['sha256']


@pytest.mark.parametrize('fault', ['outside_run', 'wrong_sha', 'unpublished'])
def test_invalid_best_saved_aborts_retention_before_any_deletion(tmp_path, fault):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage_settings())
        paths = {number: write_checkpoint(manager, number) for number in (1, 2, 3, 4)}
        descriptor = dict(iteration=1, path=str(paths[1].relative_to(manager.run_dir)), sha256=file_sha256(paths[1]))
        if fault == 'outside_run':
            outside = tmp_path / 'foreign.pt'
            outside.write_bytes(paths[1].read_bytes())
            descriptor['path'] = '../foreign.pt'
        elif fault == 'wrong_sha':
            descriptor['sha256'] = '0' * 64
        else:
            unknown = write_checkpoint(manager, 1, publish=False, filename='unpublished_copy.pt')
            descriptor.update(path=str(unknown.relative_to(manager.run_dir)), sha256=file_sha256(unknown))
        (manager.run_dir / 'best_saved.json').write_text(json.dumps(dict(saved=True, iteration=1, checkpoint=descriptor)))
        with pytest.raises(ValueError):
            maintenance.prune_checkpoints()
        assert all(path.exists() for path in paths.values())
        assert not (manager.run_dir / 'checkpoints/retired').exists()
