"""一万轮双盘归档的完整性、故障恢复、审计读取与容量测试。

所有数据写入 pytest 的独立临时目录，使用真实 RunManager、SQLite journal、tar
压缩器及独立归档进程；不连接服务器、不启动 GPU，不操作历史训练。验证奇偶轮
确定性分盘、整包/成员校验、原件删除顺序、发布中断后的幂等恢复、容量不足保留
原件、损坏归档拒绝回收、绑定丢失和路径链接拒绝。副盘存储不会绕过独立审计器。
"""
from pathlib import Path

import pytest

from gem.closedloop.dppo import long_run
from gem.closedloop.dppo.archive_store import archive_path
from gem.closedloop.dppo.long_run import LongRunMaintenance
from gem.closedloop.dppo.run_management import DiskCapacityError, RunManager, _read_json, file_sha256
from tests.closedloop.dppo.test_stage10_async_archives import iteration
from tools.eval.audit_closedloop_stage10 import archived_execution, _physical


def settings(tmp_path, *, quota=10**7):
    return dict(run_control={}, storage=dict(archive_completed_iterations=True,
        archive_secondary=dict(root=str(tmp_path/'secondary'), max_bytes=quota, min_free_bytes=1)))


@pytest.mark.parametrize('asynchronous', [False, True])
def test_two_disks_preserve_evidence_and_read_only_audit(tmp_path, asynchronous):
    stage = settings(tmp_path)
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, stage)
        directories = []
        for number in (1, 2):
            directory, journal = iteration(manager, number)
            manager.seal_iteration(directory, number, closed_journals=[journal])
            if asynchronous:
                maintenance.enqueue_archive(directory)
            else:
                maintenance.archive_iteration(directory)
            directories.append(directory)
        maintenance.drain()
        for number, directory in enumerate(directories, 1):
            path = archive_path(manager.run_dir, directory)
            assert path.is_relative_to(tmp_path/('run' if number == 1 else 'secondary'))
            manifest = _read_json(directory/'archive_manifest.json')
            assert file_sha256(path) == manifest['archive_sha256']
            assert not (directory/'fixed_targets.pt').exists()
            summary = dict(targets_path=str((directory/'fixed_targets.pt').relative_to(manager.run_dir)))
            with archived_execution(manager.run_dir, summary) as report:
                assert report['lossless_original_bytes_verified']
                assert _physical(directory/'fixed_targets.pt').read_bytes() == b'fixed-on-policy-targets'
        assert maintenance.recover_archives() == []
    with RunManager(tmp_path/'run', resume=True) as manager:
        maintenance = LongRunMaintenance(manager, stage)
        assert maintenance.recover_archives() == []


def test_secondary_quota_failure_keeps_all_originals(tmp_path):
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, settings(tmp_path, quota=2048))
        directory, journal = iteration(manager, 2)
        manager.seal_iteration(directory, 2, closed_journals=[journal])
        with pytest.raises(DiskCapacityError, match='quota'):
            maintenance.archive_iteration(directory)
        assert (directory/'execution_journal.sqlite').is_file()
        assert (directory/'fixed_targets.pt').is_file()
        assert not (directory/'archive_manifest.json').exists()


@pytest.mark.parametrize('corrupt', [False, True])
def test_secondary_publish_interruption_is_recoverable_or_rejects_corruption(tmp_path, monkeypatch, corrupt):
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, settings(tmp_path))
        directory, journal = iteration(manager, 2)
        manager.seal_iteration(directory, 2, closed_journals=[journal])
        atomic = long_run._atomic_json
        def fail_manifest(path, *args, **kwargs):
            if Path(path).name == 'archive_manifest.json':
                raise OSError('injected manifest interruption')
            return atomic(path, *args, **kwargs)
        monkeypatch.setattr(long_run, '_atomic_json', fail_manifest)
        with pytest.raises(OSError, match='interruption'):
            maintenance.archive_iteration(directory)
        path = archive_path(manager.run_dir, directory)
        original_sha = file_sha256(path)
        assert (directory/'fixed_targets.pt').exists()
        monkeypatch.setattr(long_run, '_atomic_json', atomic)
        if corrupt:
            path.write_bytes(b'corrupt')
            import tarfile
            with pytest.raises(tarfile.ReadError):
                maintenance.archive_iteration(directory)
            assert (directory/'fixed_targets.pt').exists()
        else:
            result = maintenance.archive_iteration(directory)
            assert result['archive_sha256'] == original_sha
            assert maintenance.archive_iteration(directory) == result


@pytest.mark.parametrize('fault', ['binding_missing', 'archive_missing', 'directory_link'])
def test_recovery_rejects_missing_store_or_changed_path(tmp_path, fault):
    stage = settings(tmp_path)
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, stage)
        directory, journal = iteration(manager, 2)
        manager.seal_iteration(directory, 2, closed_journals=[journal])
        maintenance.archive_iteration(directory)
        path = archive_path(manager.run_dir, directory)
    if fault == 'binding_missing':
        next((tmp_path/'secondary').glob('*/binding.json')).unlink()
    elif fault == 'archive_missing':
        path.unlink()
    else:
        moved = path.parent.with_name('moved')
        path.parent.rename(moved)
        path.parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        with RunManager(tmp_path/'run', resume=True) as manager:
            maintenance = LongRunMaintenance(manager, stage)
            maintenance.recover_archives()


def test_no_secondary_preserves_existing_archive_path(tmp_path):
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, dict(run_control={}, storage=dict(archive_completed_iterations=True)))
        directory, journal = iteration(manager, 2)
        manager.seal_iteration(directory, 2, closed_journals=[journal])
        maintenance.archive_iteration(directory)
        assert archive_path(manager.run_dir, directory) == directory/'execution_evidence.tar.gz'


def test_secondary_free_space_failure_preserves_closed_journal(tmp_path, monkeypatch):
    import shutil
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, settings(tmp_path))
        directory, journal = iteration(manager, 2)
        manager.seal_iteration(directory, 2, closed_journals=[journal])
        usage = shutil.disk_usage(tmp_path)
        monkeypatch.setattr(shutil, 'disk_usage', lambda path: usage._replace(free=0))
        with pytest.raises(DiskCapacityError, match='free space'):
            maintenance.archive_iteration(directory)
        assert (directory/'execution_journal.sqlite').is_file()
        assert (directory/'fixed_targets.pt').is_file()


def test_real_parallel_session_layout_uses_outer_iteration(tmp_path):
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, settings(tmp_path))
        old, journal = iteration(manager, 2)
        directory = manager.run_dir/'sessions'/manager.session_id/'iterations'/'000002'
        directory.parent.mkdir(parents=True)
        old.rename(directory)
        manager.seal_iteration(directory, 2, closed_journals=[directory/'execution_journal.sqlite'])
        maintenance.archive_iteration(directory)
        path = archive_path(manager.run_dir, directory)
        assert path.is_relative_to(tmp_path/'secondary')
        assert path.is_file() and path.parent.name == '000002'
