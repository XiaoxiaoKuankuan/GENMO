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
best_saved 指向的本运行已发布完整模型也始终保留，指针路径、轮次或 SHA 不可信时
先拒绝本次回收，不能删掉模型后才发现最佳指针已经失效。

新 v2 路径以不可变 seal_manifest 证明本轮接受且全部 journal 已关闭，不再要求
本轮恰好有 latest checkpoint。单调度线程最多四个在途轮，正式 v2 压缩/校验/发布/
回收由独立单 CPU 子进程执行，不 fork 已有 CUDA；满队列产生背压，异常
回传训练主线程，停止前 drain。包、清单和原件回收之间发生崩溃时，恢复重新校验
已发布归档并幂等补齐，完整模型保存与 latest 发布始终留在同步主线程。

归档计时独立于不可变证据清单：工作线程只向带锁的完成记录写入真实wall与阶段
耗时，主线程通过drain_archive_timings取快照后再写TensorBoard/JSONL。入队预检、
队列submit/backpressure和后台压缩/校验/fsync/发布/回收分别统计；未完成任务与
失败重试明确标记，不修改原始归档返回值或把后台耗时计作同步checkpoint保存。
"""
from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from .archives import ArchiveProcessClient, BoundedArchiveWorker
from .archive_store import archive_guard, archive_path, prepare_secondary_store, validate_secondary_store
from .checkpoint import VERSION_V2
from .run_management import _atomic_json, _read_json, _sync_dir, file_sha256


def validate_long_run_settings(stage):
    control = stage.get('run_control', {})
    if set(control) - {'max_walltime_seconds'}:
        raise ValueError('Unknown long-run control setting')
    seconds = control.get('max_walltime_seconds')
    if seconds is not None and (type(seconds) is not int or seconds <= 0):
        raise ValueError('max_walltime_seconds must be a positive integer')
    storage = stage['storage']
    level=storage.get('archive_compression_level',6)
    if type(level) is not int or not 1<=level<=9:
        raise ValueError('archive_compression_level must be an integer from 1 to 9')
    from .gzip_archive import validate_compression
    validate_compression(storage.get('archive_compression_backend','python'),
                         storage.get('archive_compression_threads',1))
    validate_secondary_store(storage)
    if storage.get('archive_secondary') and stage.get('version') not in (None, 'genmo.closedloop.stage10.v2'):
        raise ValueError('Secondary archive storage requires the sealed v2 training format')
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
        self.compression_level=self.storage.get('archive_compression_level',6)
        self.compression_backend=self.storage.get('archive_compression_backend','python')
        self.compression_threads=self.storage.get('archive_compression_threads',1)
        if self.compression_backend=='pigz' and shutil.which('pigz') is None:
            raise FileNotFoundError('Explicit pigz archive backend is unavailable')
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
        if 'archive_compression_level' in self.storage:
            policy['archive_compression_level']=self.compression_level
        for name in ('archive_compression_backend','archive_compression_threads'):
            if name in self.storage:policy[name]=self.storage[name]
        if self.storage.get('archive_secondary') is not None:
            policy['archive_secondary'] = self.storage['archive_secondary']
            prepare_secondary_store(manager.run_dir, policy['archive_secondary'], resume=path.exists())
        if path.exists():
            if _read_json(path) != policy:
                raise ValueError('Resume cannot change the original long-run deadline or retention policy')
        else:
            _atomic_json(path, policy, disk_guard=self.guard)
        self.policy = policy
        self._archive_worker = None
        self._archive_process = None
        self._closed = False
        self._timing_mutex = threading.RLock()
        self._archive_timings, self._enqueue_timings = [], []
        manager._maintenance.append(self)

    @staticmethod
    @contextmanager
    def _archive_stage(timings, key):
        """阶段在本次归档的局部dict累计；异常仍记录真实wall，不共享线程局部状态。"""
        started = time.perf_counter()
        try:
            yield
        finally:
            timings[key] = timings.get(key, 0.)+time.perf_counter()-started

    def drain_archive_timings(self):
        """主线程取走已完成归档/入队统计；后台不调用TensorBoard或训练对象。

        返回独立快照，正在运行的任务不伪造完成耗时；尚无完成记录时total_seconds
        为None。records含失败/恢复重试，successful_seconds只累加已完成成功任务。
        入队计时拆分seal预检与submit wall，submit包含有界队列背压，不冒充压缩耗时。
        """
        with self._timing_mutex:
            archives, enqueues = self._archive_timings, self._enqueue_timings
            self._archive_timings, self._enqueue_timings = [], []
        successful = [r for r in archives if r['status'] == 'passed']
        return dict(schema='genmo.closedloop.stage10.archive_timing.v1', records=archives, enqueues=enqueues,
            completed_count=len(archives), successful_count=len(successful),
            total_seconds=sum(r['total_seconds'] for r in archives) if archives else None,
            successful_seconds=sum(r['total_seconds'] for r in successful) if successful else None,
            pending_count=0 if self._archive_worker is None else self._archive_worker.pending_count)

    def expired(self, now=None):
        return self.deadline is not None and (now or datetime.now(timezone.utc)) >= self.deadline

    def archive_iteration(self, directory):
        if not self.storage.get('archive_completed_iterations', False):
            return None
        started, stages, result, failure = time.perf_counter(), {}, None, None
        try:
            result = self._archive_iteration(directory, stages)
            return result
        except BaseException as error:
            failure = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            name = Path(directory).name
            record = dict(directory=str(directory), iteration=(result.get('iteration') if result is not None else None),
                status='passed' if failure is None else 'failed', total_seconds=time.perf_counter()-started,
                stage_seconds=stages, thread=threading.current_thread().name, error=failure)
            if record['iteration'] is None and name.isdigit():
                record['iteration'] = int(name)
            with self._timing_mutex:
                self._archive_timings.append(record)

    def _archive_iteration(self, directory, timings):
        directory = self.guard._path(directory)
        if (directory / 'seal_manifest.json').exists():
            return self._archive_sealed(directory, timings=timings)
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

    def _archive_sealed(self, directory, *, timings=None):
        timings = {} if timings is None else timings
        with self._archive_stage(timings, 'seal_check_seconds'):
            seal = self._seal(directory)
        members = seal['members']
        manifest_path = directory / 'archive_manifest.json'
        archive, destination_guard = archive_guard(self.manager.run_dir, directory, self.guard)
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
            with self._archive_stage(timings, 'verification_seconds'):
                self._verify_archive(archive, members)
                self._verify_originals(directory, members, allow_missing=True)
        else:
            with self._archive_stage(timings, 'verification_seconds'):
                self._verify_originals(directory, members)
            if archive.exists():
                # 包完成并原子公开、清单尚未公开就崩溃的窗口。
                with self._archive_stage(timings, 'verification_seconds'):
                    self._verify_archive(archive, members)
            else:
                with destination_guard.reservation(original_size + 1048576):
                    descriptor, temporary_name = tempfile.mkstemp(prefix='.execution.', suffix='.tmp', dir=archive.parent)
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    try:
                        with self._archive_stage(timings, 'compression_seconds'):
                            from .gzip_archive import compressed_tar
                            with compressed_tar(temporary,level=getattr(self,'compression_level',6),
                                    backend=getattr(self,'compression_backend','python'),
                                    threads=getattr(self,'compression_threads',1)) as stream:
                                for record in members:
                                    stream.add(directory / record['path'], arcname=record['path'], recursive=False)
                        with self._archive_stage(timings, 'archive_fsync_seconds'):
                            with temporary.open('rb') as stream:
                                os.fsync(stream.fileno())
                        with self._archive_stage(timings, 'verification_seconds'):
                            self._verify_archive(temporary, members)
                            self._verify_originals(directory, members)
                        with self._archive_stage(timings, 'publish_seconds'):
                            os.link(temporary, archive)
                            temporary.unlink()
                            _sync_dir(archive.parent)
                        # 在归还临时空间预留前计入完整包，前台不能看见一瞬间的虚假余量。
                        destination_guard.account_file(archive)
                    finally:
                        temporary.unlink(missing_ok=True)
                        destination_guard.account_file(temporary)
            destination_guard.account_file(archive)
            result = dict(schema='genmo.closedloop.stage10.execution_archive.v2',
                          archive=archive.name, archive_sha256=file_sha256(archive),
                          original_size_bytes=original_size, archive_size_bytes=archive.stat().st_size,
                          seal_sha256=seal_sha, iteration=seal['iteration'], members=members)
            with self._archive_stage(timings, 'publish_seconds'):
                _atomic_json(manifest_path, result, disk_guard=self.guard)
        destination_guard.account_file(archive)
        # 删除原件前再次核验；失败时保留剩余原件和已验证包作为可恢复证据。
        for record in members:
            path = self.guard._path(directory / record['path'])
            if path.exists():
                with self._archive_stage(timings, 'verification_seconds'):
                    self._verify_originals(directory, [record])
                with self._archive_stage(timings, 'reclaim_seconds'):
                    path.unlink()
                    self.guard.account_file(path)
        with self._archive_stage(timings, 'reclaim_seconds'):
            for temporary in archive.parent.glob('.execution.*.tmp'):
                destination_guard._path(temporary)
                temporary.unlink()
                destination_guard.account_file(temporary)
            for folder in sorted((path for path in directory.rglob('*') if path.is_dir()),
                                 key=lambda path: len(path.parts), reverse=True):
                if not any(folder.iterdir()):
                    folder.rmdir()
            _sync_dir(directory)
        self.manager.append_metrics(dict(event='execution_archived', iteration=seal['iteration'],
            manifest=str(manifest_path.relative_to(self.manager.run_dir)),
            original_size_bytes=original_size, archive_size_bytes=result['archive_size_bytes']))
        return result

    def _account_process_archive(self, directory, seal, touched):
        """只更新本轮文件账本；不扫描其它轮历史，异常进程的临时文件也计入。"""
        paths = {directory/'execution_evidence.tar.gz', directory/'archive_manifest.json'}
        paths.update(directory/record['path'] for record in seal['members'])
        paths.update(Path(path) for path in touched)
        paths.update(directory.glob('.*.tmp'))
        # 前台刷新可能在压缩期间记录过临时文件，IPC中断也须移除已消失的账本项。
        with self.guard._mutex:
            paths.update(path for path in self.guard._sizes
                         if path.parent == directory and path.name.startswith('.') and path.name.endswith('.tmp'))
        for path in paths:
            if not self.guard._path(path).is_relative_to(directory):
                raise ValueError('Archive process accounting escapes the sealed iteration')
            self.guard.account_file(path)

    def _archive_via_process(self, directory):
        """正式v2异步归档回调：父进程预占额度，子进程处理字节，父进程接收证据。"""
        started, stages, result, response, failure = time.perf_counter(), {}, None, None, None
        seal = self._seal(directory)
        pid = None
        try:
            # reservation只在修改计数时持锁，等待子进程期间前台仍可检查/写文件。
            with self.guard.reservation(sum(row['size_bytes'] for row in seal['members'])+1048576):
                try:
                    if self._archive_process is None:
                        self._archive_process = ArchiveProcessClient()
                    pid = self._archive_process.pid
                    response = self._archive_process.archive(directory, run_dir=self.manager.run_dir,
                        min_free_bytes=self.guard.min_free_bytes,
                        **({'compression_level':self.compression_level} if 'archive_compression_level' in self.storage else {}),
                        **({'compression_backend':self.compression_backend,'compression_threads':self.compression_threads}
                           if 'archive_compression_backend' in self.storage or 'archive_compression_threads' in self.storage else {}))
                    result = response['result']
                    stages.update(response['stage_seconds'])
                finally:
                    # 在归还预留空间前计入已公开文件；进程死亡时也保留临时文件占用。
                    self._account_process_archive(directory, seal,
                        [] if response is None else response.get('accounted_paths', []))
            for record in response['metrics']:
                self.manager.append_metrics(record)
            return result
        except BaseException as error:
            failure = dict(type=type(error).__name__, message=str(error))
            if hasattr(error, 'response'):
                stages.update(error.response.get('stage_seconds', {}))
            raise
        finally:
            record = dict(directory=str(directory), iteration=seal['iteration'],
                status='passed' if failure is None else 'failed', total_seconds=time.perf_counter()-started,
                stage_seconds=stages, thread=threading.current_thread().name, error=failure,
                execution_backend='independent_cpu_process', archive_process_pid=pid)
            with self._timing_mutex:
                self._archive_timings.append(record)

    def enqueue_archive(self, directory):
        if self._closed:
            raise RuntimeError('Long-run maintenance is closed')
        if not self.storage.get('archive_completed_iterations', False):
            return False
        started, stages, queued, failure = time.perf_counter(), {}, False, None
        try:
            with self._archive_stage(stages, 'precheck_seconds'):
                directory = self.guard._path(directory)
                self._seal(directory)  # 未封存或 journal 未关闭的目录绝不进入后台队列。
            if self._archive_worker is None:
                self._archive_worker = BoundedArchiveWorker(self._archive_via_process, max_pending=4,
                                                            on_close=self._close_archive_process)
            with self._archive_stage(stages, 'submit_wall_seconds'):
                queued = self._archive_worker.submit(directory)
            return queued
        except BaseException as error:
            failure = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            record = dict(directory=str(directory), queued=queued, status='passed' if failure is None else 'failed',
                          total_seconds=time.perf_counter()-started, stage_seconds=stages, error=failure,
                          pending_count=0 if self._archive_worker is None else self._archive_worker.pending_count,
                          peak_pending_count=0 if self._archive_worker is None else self._archive_worker.peak_pending_count,
                          queue_capacity=4)
            with self._timing_mutex:
                self._enqueue_timings.append(record)

    def recover_archives(self):
        """重启后重试未归档/未回收完的封存目录；不读取失败或仍开放的 journal。"""
        queued = []
        for path in sorted(self.manager.run_dir.rglob('seal_manifest.json')):
            self.guard._path(path)
            directory = path.parent
            seal = self._seal(directory)
            manifest = directory / 'archive_manifest.json'
            if manifest.exists():
                location = archive_path(self.manager.run_dir, directory)
                if not location.is_file() or location.stat().st_size != _read_json(manifest)['archive_size_bytes']:
                    raise ValueError('Published archive is missing or has changed size during recovery')
            if not manifest.exists() or any((directory / item['path']).exists() for item in seal['members']):
                if self.enqueue_archive(directory):
                    queued.append(str(directory.relative_to(self.manager.run_dir)))
        return queued

    def drain(self):
        if self._archive_worker is not None:
            self._archive_worker.drain()

    def _close_archive_process(self):
        """在创建子进程的调度线程退出前完成EOF/wait，避免PDEATHSIG误杀正常关闭。"""
        if self._archive_process is not None:
            self._archive_process.close()

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                if self._archive_worker is not None:
                    self._archive_worker.close()
            finally:
                if self._archive_process is not None:
                    self._archive_process.close()

    def prune_checkpoints(self):
        keep_last = self.storage.get('checkpoint_keep_last')
        if keep_last is None:
            return []
        # 在任何删除前重新验证最新恢复点，旧 checkpoint 不影响这个恢复点的完整性。
        self.manager.latest_checkpoint()
        latest = _read_json(self.manager.run_dir / 'latest.json')
        publications = sorted((self.manager.run_dir / 'checkpoints/publications').glob('*.json'))
        records = [_read_json(path) for path in publications]
        protected_best = None
        best_pointer = self.manager.run_dir / 'best_saved.json'
        if best_pointer.exists() or best_pointer.is_symlink():
            best = _read_json(self.guard._path(best_pointer))
            if best is not None:
                descriptor = best.get('checkpoint') if isinstance(best, dict) else None
                if (not isinstance(best, dict) or best.get('saved') is not True or not isinstance(descriptor, dict)
                        or type(descriptor.get('iteration')) is not int
                        or best.get('iteration') != descriptor['iteration']
                        or not 0 <= descriptor['iteration'] <= latest['iteration']
                        or not isinstance(descriptor.get('path'), str)
                        or Path(descriptor['path']).is_absolute()):
                    raise ValueError('Invalid best_saved checkpoint descriptor; refusing checkpoint retirement')
                best_path = self.guard._path(self.manager.run_dir / descriptor['path'])
                protected_best = str(best_path.relative_to(self.manager.run_dir))
                if protected_best != descriptor['path'] or not best_path.is_file():
                    raise ValueError('best_saved checkpoint must be an existing canonical path inside this run')
                if descriptor['iteration'] == 0:
                    if protected_best != 'checkpoints/initial.pt':
                        raise ValueError('Initial best_saved must reference this run initial checkpoint')
                else:
                    matching = [record for record in records if all(record.get(key) == descriptor.get(key)
                                for key in ('iteration', 'path', 'sha256'))]
                    if len(matching) != 1 or best_path.stat().st_size != matching[0]['size_bytes']:
                        raise ValueError('best_saved has no matching publication in this run')
                if file_sha256(best_path) != descriptor.get('sha256'):
                    raise ValueError('best_saved checkpoint SHA changed; refusing checkpoint retirement')
        # 稀疏保存时 keep_last 表示最近 N 个完整 checkpoint，不是最近 N 个外层轮。
        durable_records = [record for record in records if record['iteration'] <= latest['iteration']]
        recent = {record['iteration'] for record in sorted(durable_records, key=lambda item: item['iteration'])[-keep_last:]}
        removed = []
        for publication in publications:
            record = _read_json(publication)
            if (record['iteration'] > latest['iteration'] or record['iteration'] in recent
                    or record['iteration'] % self.storage['checkpoint_keep_every'] == 0
                    or record['path'] == protected_best):
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
