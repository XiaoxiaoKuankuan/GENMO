"""第二阶段完整执行证据的双盘存储定位与独立容量保护。

一万轮训练约产生 2.8 TB 无损执行归档，另有周期验证和模型。配置可显式指定第二
存储根目录：偶数外层轮的 tar.gz 写入第二盘，奇数轮仍放在运行目录；原始证据、
seal、归档清单、模型和预算始终保留既有逻辑路径。这里不改采样、奖励或更新规则，
也不搬动历史训练。未配置第二盘的旧运行完全沿用本地归档路径。

第二盘以 run_manifest 的 SHA 隔离所有者，binding 同时绑定主目录、运行身份和
容量策略；拒绝符号链接、越界、复用其他运行的目录和恢复时丢失的存储。归档仍在
目标目录内暂存、逐成员校验、fsync、原子发布，成功后才删除主盘原件。读取者根据
同一不可变策略解析真实路径，再按既有清单验证大小、整包 SHA 和成员 SHA。

每块盘保留独立空闲下限和配额。后台单进程缓存第二盘 DiskGuard 的增量账本，
首次访问以及每一百轮重新扫描，避免每轮扫描整个归档历史；该缓存不接触训练 RNG、
模型或 CUDA。旧 checkpoint 的读取和既有单盘审计入口继续兼容。
"""
from __future__ import annotations

from pathlib import Path

from .run_management import DiskGuard, _atomic_json, _read_json, file_sha256


_GUARDS = {}


def validate_secondary_store(storage):
    value = storage.get('archive_secondary')
    if value is None:
        return
    if (not isinstance(value, dict) or set(value) != {'root', 'max_bytes', 'min_free_bytes'}
            or not isinstance(value['root'], str) or not Path(value['root']).is_absolute()
            or '..' in Path(value['root']).parts):
        raise ValueError('archive_secondary requires an absolute root and explicit byte limits')
    for key in ('max_bytes', 'min_free_bytes'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('archive_secondary byte limits must be positive integers')
    if not storage.get('archive_completed_iterations', False):
        raise ValueError('archive_secondary requires complete execution archives')


def _no_links(path):
    path = Path(path)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Archive storage paths cannot contain symlinks')
    return path.resolve()


def _binding(run_dir, specification):
    run_dir = _no_links(run_dir)
    digest = file_sha256(run_dir/'run_manifest.json')
    root = _no_links(specification['root'])
    if root.is_relative_to(run_dir) or run_dir.is_relative_to(root):
        raise ValueError('Secondary archive storage must be separate from the primary run')
    store = root/digest
    return store, dict(schema='genmo.closedloop.stage10.archive_store.v1',
        primary_run=str(run_dir), run_manifest_sha256=digest, specification=specification)


def prepare_secondary_store(run_dir, specification, *, resume):
    """在训练进程持有运行锁时建立独占绑定；恢复不得重新创建已经丢失的副盘。"""
    store, binding = _binding(run_dir, specification)
    path = store/'binding.json'
    _no_links(path)
    if path.exists():
        if _read_json(path) != binding:
            raise ValueError('Secondary archive store binding differs from this run')
    elif resume:
        raise FileNotFoundError('Secondary archive store binding is missing during resume')
    else:
        store.mkdir(parents=True, exist_ok=False)
        _atomic_json(path, binding)
    guard = DiskGuard(store, min_free_bytes=specification['min_free_bytes'],
                      max_run_bytes=specification['max_bytes'])
    guard.check()
    return store


def archive_path(run_dir, directory):
    """只定位，不创建文件；偶数轮走显式副盘，所有其他运行保持原有单盘路径。"""
    run_dir, directory = _no_links(run_dir), _no_links(directory)
    relative = directory.relative_to(run_dir)
    local = directory/'execution_evidence.tar.gz'
    policy_path = run_dir/'long_run_policy.json'
    specification = _read_json(policy_path).get('archive_secondary') if policy_path.exists() else None
    if specification is None:
        return local
    if _iteration(relative) % 2:
        return local
    store, binding = _binding(run_dir, specification)
    if _read_json(_no_links(store/'binding.json')) != binding:
        raise ValueError('Secondary archive store binding differs from this run')
    return _no_links(store/relative/local.name)


def _iteration(relative):
    parts = relative.parts
    if len(parts) == 4 and parts[0] == 'sessions' and parts[2] == 'iterations':
        number = parts[3]
    elif len(parts) == 3 and parts[0] == 'iterations':
        number = parts[1]  # RunManager通用封存入口保留attempt目录，仍按外层轮分盘。
    else:
        raise ValueError('Secondary archive directory must be an outer iteration')
    if not number.isdigit() or int(number) < 1:
        raise ValueError('Secondary archive requires a positive outer iteration')
    return int(number)


def archive_guard(run_dir, directory, primary_guard):
    """为真正写入的文件系统选择容量保护；压缩失败的临时占用也会入账。"""
    target = archive_path(run_dir, directory)
    if target.parent == Path(directory).resolve():
        return target, primary_guard
    specification = _read_json(Path(run_dir)/'long_run_policy.json')['archive_secondary']
    store, _ = _binding(run_dir, specification)
    key = (str(store), specification['min_free_bytes'], specification['max_bytes'])
    if key not in _GUARDS:
        _GUARDS[key] = DiskGuard(store, min_free_bytes=key[1], max_run_bytes=key[2])
    guard = _GUARDS[key]
    if _iteration(Path(directory).resolve().relative_to(Path(run_dir).resolve())) % 100 == 0:
        guard.refresh()
    guard.check()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, guard
