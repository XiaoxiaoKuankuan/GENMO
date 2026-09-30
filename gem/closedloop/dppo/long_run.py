"""第十步多日训练的截止时间、检查点保留和执行证据无损归档。

本模块只管理同一 RunManager 排他锁覆盖的运行目录，不参与采样、奖励或梯度计算。
七天时限从 run_manifest 的首次创建时间计算，断点恢复不重置时钟；入口仅在完整
更新及 checkpoint 发布后的边界停机。保留策略必须在新运行中显式配置并持久化，
恢复时不得悄悄改变。默认未配置时保持既有行为，不回收旧 checkpoint。

执行记录按轮以 gzip 压缩的 tar 文件保存，逐文件 SHA 校验归档内容后，原子发布
归档清单，再回收本轮已归档原件；summary 和学习率诊断仍可直接读取。所有原始
transition、去噪链和 SQLite 执行记录完整保留于归档，不抽样或丢弃历史执行数据。
checkpoint 只回收已发布且不属于最近若干轮/定期里程碑的文件，先校验最新恢复点
和旧文件，写入不可覆盖的退休证明后才删除旧文件；initial.pt 始终保留。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import os
import tarfile
import tempfile

import torch

from .run_management import _atomic_json, _read_json, _sync_dir, file_sha256


def validate_long_run_settings(stage):
    control = stage.get('run_control', {})
    if set(control) - {'max_walltime_seconds'}:
        raise ValueError('Unknown long-run control setting')
    seconds = control.get('max_walltime_seconds')
    if seconds is not None and (type(seconds) is not int or seconds <= 0):
        raise ValueError('max_walltime_seconds must be a positive integer')
    storage = stage['storage']
    last, every = storage.get('checkpoint_keep_last'), storage.get('checkpoint_keep_every')
    if (last is None) != (every is None):
        raise ValueError('Checkpoint retention requires both keep_last and keep_every')
    if last is not None and (type(last) is not int or last < 2 or type(every) is not int or every < 1):
        raise ValueError('Checkpoint retention requires keep_last >= 2 and keep_every >= 1')
    if type(storage.get('archive_completed_iterations', False)) is not bool:
        raise ValueError('archive_completed_iterations must be boolean')


class LongRunMaintenance:
    """调用方须持有 RunManager 锁，且仅在完成当前轮、关闭 SQLite 后调用归档。"""
    def __init__(self, manager, stage):
        validate_long_run_settings(stage)
        self.manager, self.guard = manager, manager.disk_guard
        self.storage = stage['storage']
        manifest = _read_json(manager.run_dir / 'run_manifest.json')
        self.created_at = datetime.fromisoformat(manifest['created_at'])
        seconds = stage.get('run_control', {}).get('max_walltime_seconds')
        self.deadline = None if seconds is None else self.created_at + timedelta(seconds=seconds)
        policy = dict(schema='genmo.closedloop.stage10.long_run_policy.v1',
                      created_at=manifest['created_at'], max_walltime_seconds=seconds,
                      deadline_utc=None if self.deadline is None else self.deadline.isoformat(),
                      checkpoint_keep_last=self.storage.get('checkpoint_keep_last'),
                      checkpoint_keep_every=self.storage.get('checkpoint_keep_every'),
                      archive_completed_iterations=self.storage.get('archive_completed_iterations', False))
        path = manager.run_dir / 'long_run_policy.json'
        if path.exists():
            if _read_json(path) != policy:
                raise ValueError('Resume cannot change the original long-run deadline or retention policy')
        else:
            _atomic_json(path, policy, disk_guard=self.guard)
        self.policy = policy

    def expired(self, now=None):
        return self.deadline is not None and (now or datetime.now(timezone.utc)) >= self.deadline

    def archive_iteration(self, directory):
        if not self.storage.get('archive_completed_iterations', False):
            return None
        directory = self.guard._path(directory)
        summary = _read_json(directory / 'summary.json')
        if summary.get('status') != 'accepted':
            raise ValueError('Only accepted, completely published iterations can be archived')
        latest = _read_json(self.manager.run_dir / 'latest.json')
        if latest['iteration'] != summary['iteration']:
            raise ValueError('Archive must follow publication of this exact accepted iteration')
        manifest_path = directory / 'archive_manifest.json'
        if manifest_path.exists():
            raise FileExistsError('Execution archive is immutable and already exists')
        files = []
        for child in directory.iterdir():
            if child.name in {'summary.json', 'lr_progress.json'}:
                continue
            self.guard._path(child)
            for path in ([child] if child.is_file() else sorted(child.rglob('*'))):
                self.guard._path(path)
                if path.is_file():
                    files.append(path)
        files.sort()
        if not files:
            raise ValueError('Cannot archive an empty execution record')
        members = [dict(path=str(path.relative_to(directory)), sha256=file_sha256(path),
                        size_bytes=path.stat().st_size) for path in files]
        original_size = sum(item['size_bytes'] for item in members)
        self.guard.check(original_size + 1048576)
        archive = directory / 'execution_evidence.tar.gz'
        descriptor, temporary_name = tempfile.mkstemp(prefix='.execution.', suffix='.tmp', dir=directory)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with tarfile.open(temporary, 'w:gz', compresslevel=6) as stream:
                for path, record in zip(files, members):
                    stream.add(path, arcname=record['path'], recursive=False)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            with tarfile.open(temporary, 'r:gz') as stream:
                for member, record in zip(stream, members, strict=True):
                    if not member.isfile() or member.name != record['path'] or member.size != record['size_bytes']:
                        raise ValueError('Archive member identity differs from original execution record')
                    digest = hashlib.sha256()
                    with stream.extractfile(member) as data:
                        for block in iter(lambda: data.read(1024 * 1024), b''):
                            digest.update(block)
                    if digest.hexdigest() != record['sha256']:
                        raise ValueError('Archive SHA differs from original execution record')
            # link 不覆盖任何已有归档；只有校验成功的完整包才能被公开引用。
            os.link(temporary, archive)
            temporary.unlink()
            _sync_dir(directory)
            self.guard.account_file(archive)
            result = dict(schema='genmo.closedloop.stage10.execution_archive.v1',
                          archive=archive.name, archive_sha256=file_sha256(archive),
                          original_size_bytes=original_size, archive_size_bytes=archive.stat().st_size,
                          members=members)
            _atomic_json(manifest_path, result, disk_guard=self.guard)
            for path in files:
                path.unlink()
                self.guard.account_file(path)
            for folder in sorted((p for p in directory.rglob('*') if p.is_dir()),
                                 key=lambda p: len(p.parts), reverse=True):
                folder.rmdir()
            _sync_dir(directory)
            self.manager.append_metrics(dict(event='execution_archived', iteration=summary['iteration'],
                manifest=str(manifest_path.relative_to(self.manager.run_dir)),
                original_size_bytes=original_size, archive_size_bytes=result['archive_size_bytes']))
            return result
        finally:
            temporary.unlink(missing_ok=True)
            self.guard.account_file(temporary)

    def prune_checkpoints(self):
        keep_last = self.storage.get('checkpoint_keep_last')
        if keep_last is None:
            return []
        # 在任何删除前重新验证最新恢复点，旧 checkpoint 不影响这个恢复点的完整性。
        self.manager.latest_checkpoint()
        latest = _read_json(self.manager.run_dir / 'latest.json')
        publications = sorted((self.manager.run_dir / 'checkpoints/publications').glob('*.json'))
        removed = []
        for publication in publications:
            record = _read_json(publication)
            if (record['iteration'] > latest['iteration'] - keep_last
                    or record['iteration'] % self.storage['checkpoint_keep_every'] == 0):
                continue
            path = self.guard._path(self.manager.run_dir / record['path'])
            if path.name == 'initial.pt' or record['iteration'] >= latest['iteration']:
                raise ValueError('Refusing to retire the initial or latest checkpoint')
            retirement = self.manager.run_dir / 'checkpoints/retired' / f'{path.stem}.json'
            tombstone = dict(schema='genmo.closedloop.stage10.checkpoint_retirement.v1',
                **{key: record[key] for key in ('iteration', 'path', 'sha256', 'size_bytes')},
                publication=str(publication.relative_to(self.manager.run_dir)),
                replacement={key: latest[key] for key in ('iteration', 'path', 'sha256', 'publication')},
                policy=dict(keep_last=keep_last, keep_every=self.storage['checkpoint_keep_every']),
                purpose='bounded_checkpoint_retention')
            if not path.exists():
                existing = _read_json(retirement)
                if any(existing[key] != tombstone[key] for key in ('iteration', 'path', 'sha256', 'publication')):
                    raise ValueError('Missing checkpoint has no matching retirement evidence')
                continue
            if path.stat().st_size != record['size_bytes'] or file_sha256(path) != record['sha256']:
                raise ValueError('Refusing to retire a checkpoint with changed contents')
            if not retirement.exists():
                saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                tombstone['audit_state'] = dict(
                    **{key: saved[key] for key in ('version', 'identity', 'state', 'restore_environment')},
                    samplers=dict(music={key: saved['samplers']['music'][key]
                                         for key in ('split', 'catalog_identity')}),
                    actor_optimizer=dict(param_groups=[dict(lr=group['lr'])
                        for group in saved['actor_optimizer']['param_groups']]))
                del saved
                _atomic_json(retirement, tombstone, disk_guard=self.guard)
            else:
                existing = _read_json(retirement)
                if any(existing[key] != tombstone[key] for key in ('iteration', 'path', 'sha256', 'publication')):
                    raise ValueError('Existing retirement evidence does not match checkpoint')
            path.unlink()
            self.guard.account_file(path)
            _sync_dir(path.parent)
            self.manager.append_metrics(dict(event='checkpoint_retention_completed',
                iteration=record['iteration'], path=record['path'], sha256=record['sha256'],
                retirement=str(retirement.relative_to(self.manager.run_dir))))
            removed.append(record['path'])
        return removed
