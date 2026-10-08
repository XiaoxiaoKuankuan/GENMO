"""第十步多日训练的截止时间、检查点保留和执行证据无损归档。

本模块只管理同一 RunManager 排他锁覆盖的运行目录，不参与采样、奖励或梯度计算。
七天时限从 run_manifest 的首次创建时间计算，断点恢复不重置时钟；入口在完整
接受更新、执行证据封存的边界停机，最终完整 checkpoint 仍同步保存。保留策略
必须在新运行中显式配置并持久化，
恢复时不得悄悄改变。默认未配置时保持既有行为，不回收旧 checkpoint。

执行记录按轮以 gzip 压缩的 tar 文件保存，逐文件 SHA 校验归档内容后，原子发布
归档清单，再回收本轮已归档原件；summary 和学习率诊断仍可直接读取。所有原始
transition、去噪链和 SQLite 执行记录完整保留于归档，不抽样或丢弃历史执行数据。
checkpoint 只回收已发布且不属于最近若干轮/定期里程碑的文件，先校验最新恢复点
和旧文件，写入不可覆盖的退休证明后才删除旧文件；initial.pt 始终保留。

新 v2 路径以不可变 seal_manifest 证明本轮接受且全部 journal 已关闭，不再要求
本轮恰好有 latest checkpoint。单工作线程最多四个在途轮；满队列产生背压，异常
回传训练主线程，停止前 drain。包、清单和原件回收之间发生崩溃时，恢复重新校验
已发布归档并幂等补齐，完整模型保存与 latest 发布始终留在同步主线程。
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
from .archives import BoundedArchiveWorker
from .checkpoint import VERSION_V2


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
        self._archive_worker = None
        self._closed = False
        manager._maintenance.append(self)

    def expired(self, now=None):
        return self.deadline is not None and (now or datetime.now(timezone.utc)) >= self.deadline

    def archive_iteration(self, directory):
        if not self.storage.get('archive_completed_iterations', False):
            return None
        directory = self.guard._path(directory)
        if (directory / 'seal_manifest.json').exists():
            return self._archive_sealed(directory)
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

    def _seal(self, directory):
        seal = _read_json(directory / 'seal_manifest.json')
        summary = _read_json(directory / 'summary.json')
        if (seal.get('schema') != 'genmo.closedloop.stage10.iteration_seal.v2'
                or seal.get('journals_closed') is not True
                or summary.get('status') != 'accepted'
                or summary.get('iteration') != seal.get('iteration')
                or file_sha256(directory / 'summary.json') != seal.get('summary_sha256')):
            raise ValueError('Archive requires an immutable accepted seal with closed journals')
        names = set()
        for record in seal.get('members', []):
            relative = Path(record['path'])
            if relative.is_absolute() or '..' in relative.parts or str(relative) in names:
                raise ValueError('Invalid or duplicate sealed archive member path')
            names.add(str(relative))
            self.guard._path(directory / relative)
            if (type(record.get('size_bytes')) is not int or record['size_bytes'] < 0
                    or not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64):
                raise ValueError('Invalid sealed member identity')
        if not names:
            raise ValueError('Cannot archive an empty seal')
        return seal

    def _verify_archive(self, archive, members):
        with tarfile.open(archive, 'r:gz') as stream:
            for member, record in zip(stream, members, strict=True):
                if not member.isfile() or member.name != record['path'] or member.size != record['size_bytes']:
                    raise ValueError('Archive member identity differs from immutable seal')
                digest = hashlib.sha256()
                with stream.extractfile(member) as data:
                    for block in iter(lambda: data.read(1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != record['sha256']:
                    raise ValueError('Archive SHA differs from immutable seal')

    def _verify_originals(self, directory, members, *, allow_missing=False):
        for record in members:
            path = self.guard._path(directory / record['path'])
            if allow_missing and not path.exists():
                continue
            if (not path.is_file() or path.stat().st_size != record['size_bytes']
                    or file_sha256(path) != record['sha256']):
                raise ValueError('Original execution evidence differs from immutable seal')

    def _archive_sealed(self, directory):
        seal = self._seal(directory)
        members = seal['members']
        manifest_path = directory / 'archive_manifest.json'
        archive = directory / 'execution_evidence.tar.gz'
        original_size = sum(record['size_bytes'] for record in members)
        seal_sha = file_sha256(directory / 'seal_manifest.json')
        # 清单已发布但原件回收中断时，校验完整包后幂等继续回收，不重新压缩。
        if manifest_path.exists():
            result = _read_json(manifest_path)
            if (result.get('schema') != 'genmo.closedloop.stage10.execution_archive.v2'
                    or result.get('seal_sha256') != seal_sha or result.get('members') != members
                    or result.get('archive') != archive.name
                    or not archive.is_file() or archive.stat().st_size != result.get('archive_size_bytes')
                    or file_sha256(archive) != result.get('archive_sha256')):
                raise ValueError('Published execution archive differs from its immutable seal')
            self._verify_archive(archive, members)
            self._verify_originals(directory, members, allow_missing=True)
        else:
            self._verify_originals(directory, members)
            if archive.exists():
                # 包完成并原子公开、清单尚未公开就崩溃的窗口。
                self._verify_archive(archive, members)
            else:
                with self.guard.reservation(original_size + 1048576):
                    descriptor, temporary_name = tempfile.mkstemp(prefix='.execution.', suffix='.tmp', dir=directory)
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    try:
                        with tarfile.open(temporary, 'w:gz', compresslevel=6) as stream:
                            for record in members:
                                stream.add(directory / record['path'], arcname=record['path'], recursive=False)
                        with temporary.open('rb') as stream:
                            os.fsync(stream.fileno())
                        self._verify_archive(temporary, members)
                        self._verify_originals(directory, members)
                        os.link(temporary, archive)
                        temporary.unlink()
                        _sync_dir(directory)
                        # 在归还临时空间预留前计入完整包，前台不能看见一瞬间的虚假余量。
                        self.guard.account_file(archive)
                    finally:
                        temporary.unlink(missing_ok=True)
                        self.guard.account_file(temporary)
            self.guard.account_file(archive)
            result = dict(schema='genmo.closedloop.stage10.execution_archive.v2',
                          archive=archive.name, archive_sha256=file_sha256(archive),
                          original_size_bytes=original_size, archive_size_bytes=archive.stat().st_size,
                          seal_sha256=seal_sha, iteration=seal['iteration'], members=members)
            _atomic_json(manifest_path, result, disk_guard=self.guard)
        # 删除原件前再次核验；失败时保留剩余原件和已验证包作为可恢复证据。
        for record in members:
            path = self.guard._path(directory / record['path'])
            if path.exists():
                self._verify_originals(directory, [record])
                path.unlink()
                self.guard.account_file(path)
        for temporary in directory.glob('.execution.*.tmp'):
            self.guard._path(temporary)
            temporary.unlink()
            self.guard.account_file(temporary)
        for folder in sorted((path for path in directory.rglob('*') if path.is_dir()),
                             key=lambda path: len(path.parts), reverse=True):
            if not any(folder.iterdir()):
                folder.rmdir()
        _sync_dir(directory)
        self.manager.append_metrics(dict(event='execution_archived', iteration=seal['iteration'],
            manifest=str(manifest_path.relative_to(self.manager.run_dir)),
            original_size_bytes=original_size, archive_size_bytes=result['archive_size_bytes']))
        return result

    def enqueue_archive(self, directory):
        if self._closed:
            raise RuntimeError('Long-run maintenance is closed')
        if not self.storage.get('archive_completed_iterations', False):
            return False
        directory = self.guard._path(directory)
        self._seal(directory)  # 未封存或 journal 未关闭的目录绝不进入后台队列。
        if self._archive_worker is None:
            self._archive_worker = BoundedArchiveWorker(self.archive_iteration, max_pending=4)
        return self._archive_worker.submit(directory)

    def recover_archives(self):
        """重启后重试未归档/未回收完的封存目录；不读取失败或仍开放的 journal。"""
        queued = []
        for path in sorted(self.manager.run_dir.rglob('seal_manifest.json')):
            self.guard._path(path)
            directory = path.parent
            seal = self._seal(directory)
            manifest = directory / 'archive_manifest.json'
            if not manifest.exists() or any((directory / item['path']).exists() for item in seal['members']):
                if self.enqueue_archive(directory):
                    queued.append(str(directory.relative_to(self.manager.run_dir)))
        return queued

    def drain(self):
        if self._archive_worker is not None:
            self._archive_worker.drain()

    def close(self):
        if not self._closed:
            self._closed = True
            if self._archive_worker is not None:
                self._archive_worker.close()

    def prune_checkpoints(self):
        keep_last = self.storage.get('checkpoint_keep_last')
        if keep_last is None:
            return []
        # 在任何删除前重新验证最新恢复点，旧 checkpoint 不影响这个恢复点的完整性。
        self.manager.latest_checkpoint()
        latest = _read_json(self.manager.run_dir / 'latest.json')
        publications = sorted((self.manager.run_dir / 'checkpoints/publications').glob('*.json'))
        records = [_read_json(path) for path in publications]
        # 稀疏保存时 keep_last 表示最近 N 个完整 checkpoint，不是最近 N 个外层轮。
        durable_records = [record for record in records if record['iteration'] <= latest['iteration']]
        recent = {record['iteration'] for record in sorted(durable_records, key=lambda item: item['iteration'])[-keep_last:]}
        removed = []
        for publication in publications:
            record = _read_json(publication)
            if (record['iteration'] > latest['iteration'] or record['iteration'] in recent
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
                sampler_states = saved.get('samplers', {})
                if saved.get('version') == VERSION_V2:
                    sampler_states = saved['rank_states'][0]['samplers']
                music = sampler_states.get('music', {})
                tombstone['audit_state'] = dict(
                    **{key: saved[key] for key in ('version', 'identity', 'state', 'restore_environment')},
                    samplers=dict(music={key: music[key] for key in ('split', 'catalog_identity') if key in music}),
                    actor_optimizer=dict(param_groups=[dict(lr=group['lr'])
                        for group in saved['actor_optimizer']['param_groups']]))
                if saved.get('version') == VERSION_V2:
                    tombstone['audit_state']['world_size'] = saved['world_size']
                    tombstone['audit_state']['rank_states'] = [dict(
                        rank=item['rank'], state=item['state'],
                        samplers={name: {key: value[key] for key in (
                            'version', 'split', 'catalog_identity', 'draw_count',
                            'batch_size', 'bc_update_steps') if key in value}
                            for name, value in item['samplers'].items()},
                        rng={key: item['rng'][key] for key in (
                            'cuda_scope', 'cuda_device', 'generator_devices')})
                        for item in saved['rank_states']]
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
