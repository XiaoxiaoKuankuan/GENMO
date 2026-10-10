"""稀疏模型保存下的执行证据封存、异步背压和崩溃恢复测试。

本文件使用临时目录、微型 SQLite journal 与事件同步，不启动实际训练。覆盖普通
接受轮无需 checkpoint 即可归档、开放 journal 拒绝入队、归档损坏不得删原件、
原子包/清单两个崩溃窗口的幂等恢复，以及单调度线程最多四个在途任务和关闭 drain。
正式v2还检查实际独立进程PID、无CUDA/torchrun环境和写锁FD、进程死亡/超时先回收
再归还磁盘预留，以及Linux父进程死亡后子进程跟随退出并由测试精确wait回收。
另检查恢复时接受水位可标记作废、资源消耗保留，以及精确租约结算不能二次退款。
"""
import tarfile
import threading
import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gem.closedloop.dppo import long_run
from gem.closedloop.dppo.archives import ArchiveProcessClient, ArchiveWorkerError, BoundedArchiveWorker
from gem.closedloop.dppo.long_run import LongRunMaintenance
from gem.closedloop.dppo.run_management import RunManager, _atomic_json, _read_json, file_sha256


def stage():
    return dict(run_control={}, storage=dict(archive_completed_iterations=True))


def iteration(manager, number=1, *, close=True):
    directory = manager.iteration_dir(number)
    journal = manager.journal(number)
    journal.append_result(dict(backend_session_id='session', mutation_seq=number, result=dict(answer=number)))
    if close:
        journal.close()
    (directory / 'fixed_targets.pt').write_bytes(b'fixed-on-policy-targets')
    _atomic_json(directory / 'summary.json', dict(iteration=number, status='accepted'))
    return directory, journal


def test_sealed_noncheckpoint_iteration_archives_and_close_drains(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        directory, journal = iteration(manager)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        assert not (manager.run_dir / 'latest.json').exists()
        assert maintenance.enqueue_archive(directory)
    manifest = _read_json(directory / 'archive_manifest.json')
    assert manifest['schema'].endswith('.v2') and manifest['iteration'] == 1
    assert file_sha256(directory / manifest['archive']) == manifest['archive_sha256']
    assert not (directory / 'execution_journal.sqlite').exists()
    assert not (directory / 'fixed_targets.pt').exists()


def test_open_journal_or_unsealed_directory_never_enters_queue(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        directory, journal = iteration(manager, close=False)
        with pytest.raises(ValueError, match='open execution journal'):
            manager.seal_iteration(directory, 1, closed_journals=[journal])
        with pytest.raises(FileNotFoundError):
            maintenance.enqueue_archive(directory)
        assert maintenance._archive_worker is None


def test_changed_original_stays_after_seal_failure(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        directory, journal = iteration(manager)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        (directory / 'fixed_targets.pt').write_bytes(b'changed')
        with pytest.raises(ValueError, match='differs from immutable seal'):
            maintenance.archive_iteration(directory)
        assert (directory / 'execution_journal.sqlite').exists()
        assert not (directory / 'archive_manifest.json').exists()


@pytest.mark.parametrize('backend',['python','pigz'])
def test_archive_published_before_manifest_recovers_without_recompression(tmp_path, monkeypatch, backend):
    with RunManager(tmp_path / 'run') as manager:
        configured=stage();configured['storage'].update(archive_compression_backend=backend,
            archive_compression_threads=4 if backend=='pigz' else 1)
        maintenance = LongRunMaintenance(manager, configured)
        directory, journal = iteration(manager)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        atomic = long_run._atomic_json
        def fail_manifest(path, *args, **kwargs):
            if Path(path).name == 'archive_manifest.json':
                raise OSError('injected publication interruption')
            return atomic(path, *args, **kwargs)
        monkeypatch.setattr(long_run, '_atomic_json', fail_manifest)
        with pytest.raises(OSError, match='interruption'):
            maintenance.archive_iteration(directory)
        archive = directory / 'execution_evidence.tar.gz'
        sha = file_sha256(archive)
        assert (directory / 'execution_journal.sqlite').exists()
        monkeypatch.setattr(long_run, '_atomic_json', atomic)
        result = maintenance.archive_iteration(directory)
        assert result['archive_sha256'] == sha
        assert maintenance.archive_iteration(directory) == result


def test_partial_original_removal_recovers_from_verified_manifest(tmp_path, monkeypatch):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        directory, journal = iteration(manager)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        unlink = Path.unlink
        def fail_second(path, *args, **kwargs):
            if path == directory / 'fixed_targets.pt':
                raise OSError('injected removal interruption')
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'unlink', fail_second)
        with pytest.raises(OSError, match='removal interruption'):
            maintenance.archive_iteration(directory)
        assert (directory / 'archive_manifest.json').exists()
        assert not (directory / 'execution_journal.sqlite').exists()
        assert (directory / 'fixed_targets.pt').exists()
        monkeypatch.setattr(Path, 'unlink', unlink)
        maintenance.enqueue_archive(directory)
        maintenance.drain()
        assert not (directory / 'fixed_targets.pt').exists()


def test_restart_recovers_only_sealed_completed_directories(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        directory, journal = iteration(manager, 1)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        unsealed, _ = iteration(manager, 2)
    with RunManager(tmp_path / 'run', resume=True) as manager:
        maintenance = LongRunMaintenance(manager, stage())
        queued = maintenance.recover_archives()
        assert len(queued) == 1 and Path(queued[0]).name == directory.name
        maintenance.drain()
        assert maintenance.recover_archives() == []
        assert not (directory / 'fixed_targets.pt').exists()
        assert (unsealed / 'fixed_targets.pt').exists()


def test_corrupt_archive_never_removes_remaining_originals(tmp_path, monkeypatch):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        directory, journal = iteration(manager)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        atomic = long_run._atomic_json
        monkeypatch.setattr(long_run, '_atomic_json', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('stop')))
        with pytest.raises(OSError):
            maintenance.archive_iteration(directory)
        monkeypatch.setattr(long_run, '_atomic_json', atomic)
        (directory / 'execution_evidence.tar.gz').write_bytes(b'corrupt')
        with pytest.raises(tarfile.ReadError):
            maintenance.archive_iteration(directory)
        assert (directory / 'execution_journal.sqlite').exists()
        assert (directory / 'fixed_targets.pt').exists()


def test_worker_bounds_inflight_and_backpressures_then_drains(tmp_path):
    started, release, submitted = threading.Event(), threading.Event(), threading.Event()
    completed = []
    def archive(path):
        started.set()
        assert release.wait(5)
        completed.append(path)
    worker = BoundedArchiveWorker(archive, max_pending=4)
    producer = None
    try:
        for number in range(4):
            assert worker.submit(tmp_path / str(number))
        assert started.wait(1) and worker.pending_count == 4
        assert worker.peak_pending_count == 4
        assert not worker.submit(tmp_path / '0')
        producer = threading.Thread(target=lambda: (worker.submit(tmp_path / '4'), submitted.set()))
        producer.start()
        assert not submitted.wait(.1)
        release.set()
        producer.join(2)
        assert submitted.is_set()
        worker.drain()
        assert len(completed) == 5 and worker.pending_count == 0
        assert worker.peak_pending_count == 4
    finally:
        release.set()
        if producer is not None:
            producer.join(2)
        worker.close()


def test_worker_failure_surfaces_and_retains_queued_tasks(tmp_path):
    release = threading.Event()
    calls = []
    def archive(path):
        calls.append(path)
        assert release.wait(3)
        raise OSError('injected background failure')
    worker = BoundedArchiveWorker(archive)
    worker.submit(tmp_path / 'first')
    worker.submit(tmp_path / 'second')
    release.set()
    with pytest.raises(ArchiveWorkerError):
        worker.drain()
    assert len(calls) == 1
    with pytest.raises(ArchiveWorkerError):
        worker.submit(tmp_path / 'third')
    with pytest.raises(ArchiveWorkerError):
        worker.close()


def eventually(predicate, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Expected archive process state was not reached')


def test_real_process_is_single_cpu_worker_with_backpressure_and_no_run_lock(tmp_path, monkeypatch):
    monkeypatch.setenv('RANK', '5')
    monkeypatch.setenv('TORCHELASTIC_RUN_ID', 'inherited-training')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        directories = []
        producer, pid = None, None
        try:
            for number in range(1, 6):
                directory, journal = iteration(manager, number)
                manager.seal_iteration(directory, number, closed_journals=[journal])
                directories.append(directory)
            for directory in directories[:4]:
                maintenance.enqueue_archive(directory)
            eventually(lambda: maintenance._archive_process is not None)
            pid = maintenance._archive_process.pid
            assert pid != os.getpid()
            os.kill(pid, signal.SIGSTOP)
            submitted = threading.Event()
            producer = threading.Thread(target=lambda: (maintenance.enqueue_archive(directories[4]), submitted.set()))
            producer.start()
            assert not submitted.wait(.1) and maintenance._archive_worker.pending_count == 4
            environment = dict(value.split(b'=', 1) for value in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0') if value)
            assert environment[b'CUDA_VISIBLE_DEVICES'] == b''
            assert b'RANK' not in environment and b'TORCHELASTIC_RUN_ID' not in environment
            descriptors = [str(path.resolve()) for path in (Path('/proc')/str(pid)/'fd').iterdir()]
            assert str(manager.run_dir/'.run.lock') not in descriptors
            os.kill(pid, signal.SIGCONT)
            producer.join(5)
            assert submitted.is_set()
            maintenance.drain()
            records = maintenance.drain_archive_timings()['records']
            assert len(records) == 5
            assert {row['archive_process_pid'] for row in records} == {pid}
            assert all(row['execution_backend']=='independent_cpu_process' for row in records)
            assert manager.disk_guard._reserved_bytes == 0
            for directory in directories:
                manifest = _read_json(directory/'archive_manifest.json')
                assert file_sha256(directory/manifest['archive']) == manifest['archive_sha256']
                assert not (directory/'fixed_targets.pt').exists()
        finally:
            if pid is not None and maintenance._archive_process._process.poll() is None:
                os.kill(pid, signal.SIGCONT)
            if producer is not None:
                producer.join(5)
            maintenance.close()
        assert maintenance._archive_process._process.returncode == 0


@pytest.mark.parametrize('failure', ['killed', 'timeout'])
def test_process_failure_reaped_before_reservation_release_and_keeps_originals(tmp_path, monkeypatch, failure):
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        directory, journal = iteration(manager)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        if failure == 'timeout':
            monkeypatch.setattr(long_run, 'ArchiveProcessClient', lambda: ArchiveProcessClient(response_timeout_s=.02))
        maintenance.enqueue_archive(directory)
        eventually(lambda: maintenance._archive_process is not None)
        client = maintenance._archive_process
        if failure == 'killed':
            os.kill(client.pid, signal.SIGKILL)
        with pytest.raises(ArchiveWorkerError):
            maintenance.drain()
        assert client._process.returncode is not None
        assert manager.disk_guard._reserved_bytes == 0
        assert (directory/'fixed_targets.pt').exists() and (directory/'execution_journal.sqlite').exists()
        assert not (directory/'archive_manifest.json').exists()
        with pytest.raises(ArchiveWorkerError):
            maintenance.close()


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux PDEATHSIG/subreaper contract')
def test_parent_sigkill_stops_and_reaps_real_archive_process(tmp_path):
    # 临时成为subreaper，精确wait已归属的孙进程，避免容器PID1不回收测试僵尸。
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0  # PR_GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    code = ('from gem.closedloop.dppo.archives import ArchiveProcessClient; import time; '
            'client=ArchiveProcessClient(); print(client.pid,flush=True); time.sleep(120)')
    parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    pid, reaped = None, False
    try:
        pid = int(parent.stdout.readline())
        assert (Path('/proc')/str(pid)).exists()
        parent.kill()
        parent.wait(timeout=5)

        def child_exited():
            nonlocal reaped
            found, status = os.waitpid(pid, os.WNOHANG)
            reaped = found == pid
            if reaped:
                assert os.WIFEXITED(status) or os.WIFSIGNALED(status)
            return reaped

        eventually(child_exited)
        assert not (Path('/proc')/str(pid)).exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if pid is not None and not reaped:
            try:
                found, _ = os.waitpid(pid, os.WNOHANG)
                if found == 0:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        parent.stdout.close()
        parent.stderr.close()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def limits():
    return dict(accepted_iterations=20, optimizer_attempts=200, generations=200,
                control_steps=200, physics_steps=800)


def test_lease_settlement_is_atomic_idempotent_and_survives_resume(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        budget = manager.budget(limits())
        reserved = dict(generations=10, control_steps=20, physics_steps=80)
        used = dict(generations=7, control_steps=13, physics_steps=52)
        budget.reserve('collect-rank0-iteration1', **reserved)
        record = budget.settle_lease('collect-rank0-iteration1', reserved, used, lease_id='lease-1')
        assert budget.state_dict()['used']['control_steps'] == 13
        assert budget.settle_lease('collect-rank0-iteration1', reserved, used, lease_id='lease-1') == record
        with pytest.raises(ValueError, match='only once'):
            budget.settle_lease('collect-rank0-iteration1', used, used, lease_id='lease-2')
    with RunManager(tmp_path / 'run', resume=True) as manager:
        budget = manager.budget(limits())
        assert budget.settle_lease('collect-rank0-iteration1', reserved, used, lease_id='lease-1') == record
        with pytest.raises(ValueError, match='four times'):
            budget.settle_lease('other', dict(control_steps=2, physics_steps=8),
                                dict(control_steps=1, physics_steps=8), lease_id='bad')


def test_reconcile_marks_unsaved_tail_and_preserves_spent_budget(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        budget = manager.budget(limits())
        budget.reserve('updates', accepted_iterations=2, optimizer_attempts=7)
        for number in (1, 2):
            directory, journal = iteration(manager, number)
            manager.seal_iteration(directory, number, closed_journals=[journal])
        before = budget.state_dict()
    with RunManager(tmp_path / 'run', resume=True) as manager:
        budget = manager.budget(limits())
        event = manager.reconcile_accepted(1)
        assert _read_json(event)['previous_accepted']['iteration'] == 2
        assert _read_json(manager.run_dir / 'accepted.json')['iteration'] == 1
        assert budget.state_dict() == before
        directory, journal = iteration(manager, 2)
        manager.seal_iteration(directory, 2, closed_journals=[journal])
        assert _read_json(manager.run_dir / 'accepted.json')['iteration'] == 2


@pytest.mark.parametrize('level',[1,6])
@pytest.mark.parametrize('backend',['python','pigz'])
def test_lossless_compression_candidate_keeps_member_sha_and_recovery(tmp_path,level,backend):
    import tarfile
    import hashlib
    configured=stage();configured['storage']['archive_compression_level']=level
    configured['storage'].update(archive_compression_backend=backend,
        archive_compression_threads=4 if backend=='pigz' else 1)
    with RunManager(tmp_path/'run') as manager:
        maintenance=LongRunMaintenance(manager,configured)
        directory,journal=iteration(manager)
        manager.seal_iteration(directory,1,closed_journals=[journal])
        assert maintenance.enqueue_archive(directory)
        maintenance.close()
        manifest=_read_json(directory/'archive_manifest.json')
        assert maintenance.archive_iteration(directory)==manifest
    with tarfile.open(directory/manifest['archive'],'r:gz') as stream:
        for member,record in zip(stream,manifest['members'],strict=True):
            assert hashlib.sha256(stream.extractfile(member).read()).hexdigest()==record['sha256']


@pytest.mark.parametrize('backend,threads',[('unknown',1),('python',2),('pigz',0),('pigz',5),('pigz',True)])
def test_archive_compression_rejects_unbounded_or_ambiguous_configuration(backend,threads):
    from gem.closedloop.dppo.gzip_archive import validate_compression
    with pytest.raises(ValueError):validate_compression(backend,threads)


def test_failed_pigz_cannot_publish_or_remove_originals(tmp_path,monkeypatch):
    fake=tmp_path/'bin';fake.mkdir()
    executable=fake/'pigz'
    executable.write_text('#!/bin/sh\n# 测试压缩器明确失败，不能当作合法gzip发布。\necho injected-compressor-failure >&2\nexit 7\n')
    executable.chmod(0o755);monkeypatch.setenv('PATH',str(fake))
    with RunManager(tmp_path/'run') as manager:
        configured=stage();configured['storage'].update(archive_compression_backend='pigz',archive_compression_threads=4)
        maintenance=LongRunMaintenance(manager,configured)
        directory,journal=iteration(manager);manager.seal_iteration(directory,1,closed_journals=[journal])
        with pytest.raises((RuntimeError,BrokenPipeError)):
            maintenance.archive_iteration(directory)
        assert (directory/'execution_journal.sqlite').exists()
        assert (directory/'fixed_targets.pt').exists()
        assert not (directory/'archive_manifest.json').exists()
        assert not (directory/'execution_evidence.tar.gz').exists()
        assert not list(directory.glob('.execution.*.tmp'))


def test_pigz_follows_archive_parent_death(tmp_path):
    """归档worker突然退出时，压缩器也必须退出，不能留下持有临时包的孤儿。"""
    import json
    libc=ctypes.CDLL(None,use_errno=True);previous=ctypes.c_int()
    assert libc.prctl(37,ctypes.byref(previous),0,0,0)==0
    assert libc.prctl(36,1,0,0,0)==0
    code=('from gem.closedloop.dppo.gzip_archive import compressed_tar; import time\n'
          f'with compressed_tar({json.dumps(str(tmp_path/"unpublished.gz"))},backend="pigz",threads=4):\n'
          ' print("ready",flush=True)\n time.sleep(120)\n')
    parent=subprocess.Popen([sys.executable,'-B','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    pid,reaped=None,False
    try:
        assert parent.stdout.readline().strip()=='ready'
        children=Path('/proc')/str(parent.pid)/'task'/str(parent.pid)/'children'
        eventually(lambda:bool(children.read_text().strip()))
        ids=children.read_text().split();assert len(ids)==1;pid=int(ids[0])
        eventually(lambda:(Path('/proc')/str(pid)/'comm').read_text().strip()=='pigz')
        parent.kill();parent.wait(timeout=5)
        def exited():
            nonlocal reaped
            found,_=os.waitpid(pid,os.WNOHANG);reaped=found==pid
            return reaped
        eventually(exited)
        assert not (Path('/proc')/str(pid)).exists()
    finally:
        if parent.poll() is None:parent.kill();parent.wait(timeout=5)
        if pid is not None and not reaped:
            try:
                found,_=os.waitpid(pid,os.WNOHANG)
                if found==0:os.kill(pid,signal.SIGKILL);os.waitpid(pid,0)
            except ChildProcessError:pass
        parent.stdout.close();parent.stderr.close()
        assert libc.prctl(36,previous.value,0,0,0)==0
