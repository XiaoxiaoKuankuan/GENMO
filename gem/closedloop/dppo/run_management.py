"""第十步连续训练的运行目录、执行证据和持久化资源管理。

本模块不决定 Actor/Critic 算法，只为单个训练写入者提供可恢复的外围设施。运行目录
持有进程级排他锁；预算分别累计已接受更新、优化器尝试、生成、控制和物理步，执行
之前预占并立即持久化，未知执行消耗不退还。已接受更新的预算消耗与最后成功发布的
checkpoint 水位分别记录：发布失败时保留消耗，恢复进度以 latest 指向的完整状态为准。

rollout 每条转移只写一次，按小块发布 SHA 清单，完整清单最后原子发布；不会随着
采集增长反复重写整个 Buffer。每个 iteration/attempt 使用独立 SQLite 执行日志，
复用原 StepJournal 的 FULL 事务，调用方仍须在 append_result 成功后才向后端 ACK。
磁盘检查同时考虑文件系统剩余空间和本 run 配额；不自动删除任何正式执行证据。

完整 checkpoint 继续由既有 checkpoint.save_checkpoint 写入。这里检查格式、计算
SHA 并原子更新 latest 指针，不复制或重新序列化大模型。信号处理只设置停止标志，
实际停止必须由训练入口在无 pending、已完成落盘的合法边界执行。本模块不冒充恢复
PhysX 内部状态，也不允许未完成 rollout 自动混入下一轮 on-policy 数据。

同一运行的预算只允许经显式扩展入口单调提高：旧消耗原样保留，扩展原因、配置和父
checkpoint 的 SHA 写入不可覆盖的事件，再原子更新账本引用。恢复验证这条证据链，
未引用的中断事件不授权扩限；不能手工修改上限或把准备运行的计数归零后冒充续训。
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import threading
import uuid

import torch

from .budget import BudgetExceeded
from .buffer import StepJournal, UpperTransition
from .checkpoint import VERSION as CHECKPOINT_VERSION, VERSION_V2, _validate_rank_states
from .performance import profiled


BUDGET_KEYS = ('accepted_iterations', 'optimizer_attempts', 'generations', 'control_steps', 'physics_steps')


def _budget_limits(limits):
    if set(limits) != set(BUDGET_KEYS):
        raise ValueError(f'budget limits must contain exactly {BUDGET_KEYS}')
    result = {key: _integer(limits[key], key, 1) for key in BUDGET_KEYS}
    if result['physics_steps'] != 4 * result['control_steps']:
        raise ValueError('physics budget must equal four times control budget')
    return result


def validate_budget_progress(saved, live):
    """恢复允许已有证据的单调扩限，但不接受消耗回退或替换原扩展链。"""
    saved_limits, live_limits = _budget_limits(saved['limits']), _budget_limits(live['limits'])
    before, after = saved.get('limit_extensions', []), live.get('limit_extensions', [])
    if after[:len(before)] != before or len(after) < len(before):
        raise ValueError('Resume cannot replace or roll back budget extension history')
    limits = saved_limits
    for entry in after[len(before):]:
        if entry.get('previous_limits') != limits:
            raise ValueError('Resume budget extension does not continue checkpoint limits')
        updated = _budget_limits(entry['new_limits'])
        if updated == limits or any(updated[key] < limits[key] for key in BUDGET_KEYS):
            raise ValueError('Resume budget extension must increase limits monotonically')
        limits = updated
    if limits != live_limits:
        raise ValueError('Resume cannot change limits without budget extension evidence')
    if set(saved.get('used', {})) != set(BUDGET_KEYS) or set(live.get('used', {})) != set(BUDGET_KEYS):
        raise ValueError('Resume requires all spent budget counters')
    if any(_integer(live['used'][key], key) < _integer(saved['used'][key], key) for key in BUDGET_KEYS):
        raise ValueError('Resume cannot roll back spent budget')
    if any(live.get('lease_settlements', {}).get(key) != value
           for key, value in saved.get('lease_settlements', {}).items()):
        raise ValueError('Resume cannot remove or replace a settled lease')


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {value}')))


def _sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _storage_estimate(value, seen=None):
    """序列化前保守预留张量、数组和Python元数据空间，写后仍核验实际占用。"""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes() + 512
    if isinstance(value, str):
        return len(value.encode()) + 64
    if isinstance(value, dict):
        return 128 + sum(_storage_estimate(key, seen) + _storage_estimate(item, seen) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return 128 + sum(_storage_estimate(item, seen) for item in value)
    if hasattr(value, 'nbytes'):
        return int(value.nbytes) + 128
    if isinstance(value, UpperTransition):
        return _storage_estimate(vars(value), seen)
    return 64


def _atomic_json(path, value, *, replace=False, disk_guard=None):
    path = Path(path)
    if disk_guard is not None:
        path = disk_guard._path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise FileExistsError(path)
    data = _json_bytes(value)
    if disk_guard is not None:
        disk_guard.check(len(data))
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # 排他创建最终路径，禁止覆盖既有正式证据。
            temporary.unlink()
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        if disk_guard is not None:
            disk_guard.account_file(path)


class DiskCapacityError(RuntimeError):
    pass


class DiskGuard:
    """启动/显式刷新时清点目录；日常写入按文件增量记账，避免每个控制步扫描历史。"""
    def __init__(self, run_dir, *, min_free_bytes=0, max_run_bytes=None):
        self.run_dir = Path(run_dir).resolve()
        self.min_free_bytes = _integer(min_free_bytes, 'min_free_bytes')
        self.max_run_bytes = None if max_run_bytes is None else _integer(max_run_bytes, 'max_run_bytes', 1)
        self._sizes = {}
        self._mutex = threading.RLock()
        self._reserved_bytes = 0
        self.refresh()

    def _path(self, path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError('managed evidence cannot be a symlink')
        path = path.resolve()
        if not path.is_relative_to(self.run_dir):
            raise ValueError('managed evidence must remain inside this run directory')
        return path

    @property
    def used_bytes(self):
        with self._mutex:
            return self._used_bytes

    def refresh(self):
        with self._mutex:
            return self._refresh_locked()

    def _refresh_locked(self):
        sizes = {}
        for directory, folders, files in os.walk(self.run_dir, followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in folders):
                raise ValueError('run evidence directories cannot be symlinks')
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    raise ValueError(f'run evidence cannot be a symlink: {path}')
                try:
                    sizes[path.resolve()] = path.stat().st_size
                except FileNotFoundError:
                    # 异步 worker 可能正回收已归档文件；消失文件不计入刷新快照。
                    continue
        self._sizes = sizes
        self._used_bytes = sum(sizes.values())
        return self.used_bytes

    def account_file(self, path):
        path = self._path(path)
        with self._mutex:
            previous = self._sizes.get(path, 0)
            if path.exists():
                self._sizes[path] = path.stat().st_size
            else:
                self._sizes.pop(path, None)
            self._used_bytes += self._sizes.get(path, 0) - previous

    def check(self, required_bytes=0):
        with self._mutex:
            return self._check_locked(required_bytes)

    def _check_locked(self, required_bytes=0):
        required = _integer(required_bytes, 'required_bytes')
        free = shutil.disk_usage(self.run_dir).free
        if free - required - self._reserved_bytes < self.min_free_bytes:
            raise DiskCapacityError('filesystem free space would fall below the configured reserve')
        if self.max_run_bytes is not None and self.used_bytes + required + self._reserved_bytes > self.max_run_bytes:
            raise DiskCapacityError('run evidence would exceed its configured byte quota')
        return dict(free_bytes=free, used_bytes=self.used_bytes, required_bytes=required,
                    min_free_bytes=self.min_free_bytes, max_run_bytes=self.max_run_bytes)

    @contextmanager
    def reservation(self, required_bytes):
        """后台归档的临时空间仍占用配额，防止前台同时写入时重复使用相同余量。"""
        required = _integer(required_bytes, 'reserved bytes')
        with self._mutex:
            self._check_locked(required)
            self._reserved_bytes += required
        try:
            yield
        finally:
            with self._mutex:
                self._reserved_bytes -= required


class TrainingBudget:
    """单写入者预算，接口兼容 UpperEnvironment；运行锁由 RunManager 持有。"""
    def __init__(self, path, limits, *, disk_guard=None):
        self.path, self.disk_guard = Path(path), disk_guard
        self.limits = _budget_limits(limits)
        if self.path.exists():
            self.state = _read_json(self.path)
            self._validate()
            if self.state['limits'] != self.limits:
                raise ValueError('resume cannot silently change an existing run budget')
        else:
            self.state = dict(schema='genmo.closedloop.stage10.budget.v1', limits=self.limits.copy(),
                              used={key: 0 for key in BUDGET_KEYS}, phases={})
            self._save(self.state)

    def _validate(self):
        if self.state.get('schema') != 'genmo.closedloop.stage10.budget.v1':
            raise ValueError('unsupported continuous-training budget schema')
        if set(self.state.get('used', {})) != set(BUDGET_KEYS) or set(self.state.get('limits', {})) != set(BUDGET_KEYS):
            raise ValueError('incomplete persistent budget')
        for key in BUDGET_KEYS:
            used = _integer(self.state['used'][key], key)
            limit = _integer(self.state['limits'][key], key, 1)
            if used > limit or sum(_integer(phase.get(key, 0), key) for phase in self.state['phases'].values()) != used:
                raise ValueError('persistent budget totals disagree or exceed limits')
        if any(set(phase) - set(BUDGET_KEYS) for phase in self.state['phases'].values()):
            raise ValueError('unknown persistent budget counter')
        settled_phases = set()
        for lease_id, record in self.state.get('lease_settlements', {}).items():
            if (not isinstance(lease_id, str) or not lease_id
                    or set(record) != {'phase', 'reserved', 'used'}
                    or not isinstance(record['phase'], str) or not record['phase']
                    or record['phase'] in settled_phases):
                raise ValueError('invalid or duplicate persistent lease settlement')
            settled_phases.add(record['phase'])
            reserved, used = record['reserved'], record['used']
            if not reserved or set(reserved) != set(used) or set(reserved) - set(BUDGET_KEYS):
                raise ValueError('invalid persistent lease counters')
            for key in reserved:
                if not _integer(used[key], key) <= _integer(reserved[key], key):
                    raise ValueError('persistent lease usage exceeds reservation')
                if used[key] > self.state['phases'].get(record['phase'], {}).get(key, 0):
                    raise ValueError('persistent lease usage exceeds recorded phase')
            for counters in (reserved, used):
                if ('control_steps' in counters or 'physics_steps' in counters) and (
                        set(counters) & {'control_steps', 'physics_steps'} != {'control_steps', 'physics_steps'}
                        or counters['physics_steps'] != 4 * counters['control_steps']):
                    raise ValueError('persistent lease physics counter differs from controls')
        self._validate_extensions()

    def _validate_extensions(self):
        entries = self.state.get('limit_extensions', [])
        if not isinstance(entries, list):
            raise ValueError('invalid budget extension history')
        previous_limits, previous_used, seen = None, {key: 0 for key in BUDGET_KEYS}, set()
        for index, entry in enumerate(entries):
            if set(entry) != {'path', 'sha256', 'previous_limits', 'new_limits'}:
                raise ValueError('invalid budget extension descriptor')
            relative = Path(entry['path'])
            if (relative.is_absolute() or relative.parent != Path('budget_extensions')
                    or not re.fullmatch(r'[a-f0-9-]{36}\.json', relative.name) or str(relative) in seen):
                raise ValueError('invalid or duplicate budget extension path')
            seen.add(str(relative))
            path = self.path.parent / relative
            if path.is_symlink() or path.parent.is_symlink() or file_sha256(path) != entry['sha256']:
                raise ValueError('budget extension SHA mismatch')
            event = _read_json(path)
            if (event.get('schema') != 'genmo.closedloop.stage10.budget_extension.v1'
                    or event.get('previous_extensions') != entries[:index]
                    or event.get('previous_limits') != entry['previous_limits']
                    or event.get('new_limits') != entry['new_limits']):
                raise ValueError('budget extension identity mismatch')
            old, new = _budget_limits(event['previous_limits']), _budget_limits(event['new_limits'])
            if previous_limits is not None and old != previous_limits:
                raise ValueError('budget extension limits are not contiguous')
            if old == new or any(new[key] < old[key] for key in BUDGET_KEYS):
                raise ValueError('budget extension must increase limits monotonically')
            if (not isinstance(event.get('reason'), str) or not event['reason'].strip()
                    or any(not re.fullmatch(r'[0-9a-f]{64}', event.get(key, ''))
                           for key in ('checkpoint_sha256', 'config_sha256'))):
                raise ValueError('budget extension requires reason and checkpoint/config SHA')
            used, phases = event.get('used', {}), event.get('phases', {})
            if set(used) != set(BUDGET_KEYS) or not isinstance(phases, dict):
                raise ValueError('budget extension lacks spent counters')
            for key in BUDGET_KEYS:
                count = _integer(used[key], key)
                if (not previous_used[key] <= count <= min(old[key], self.state['used'][key])
                        or sum(_integer(phase.get(key, 0), key) for phase in phases.values()) != count):
                    raise ValueError('budget extension spent counters disagree or roll back')
            if any(set(phase) - set(BUDGET_KEYS) for phase in phases.values()):
                raise ValueError('budget extension contains unknown phase counters')
            previous_limits, previous_used = new, used
        if entries and previous_limits != self.state['limits']:
            raise ValueError('persistent limits do not match the budget extension chain')

    def extend_limits(self, limits, *, reason, checkpoint_sha256, config_sha256):
        """调用者持有RunManager锁并核验最新checkpoint后，显式扩展同一运行预算。"""
        updated_limits = _budget_limits(limits)
        if updated_limits == self.limits or any(updated_limits[key] < self.limits[key] for key in BUDGET_KEYS):
            raise ValueError('explicit budget extension must increase at least one limit and decrease none')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('explicit budget extension requires a nonempty reason')
        if any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
               for value in (checkpoint_sha256, config_sha256)):
            raise ValueError('explicit budget extension requires checkpoint and config SHA256')
        self._validate()
        relative = Path('budget_extensions') / f'{uuid.uuid4()}.json'
        event = dict(schema='genmo.closedloop.stage10.budget_extension.v1',
            created_at=datetime.now(timezone.utc).isoformat(), reason=reason.strip(),
            checkpoint_sha256=checkpoint_sha256, config_sha256=config_sha256,
            previous_limits=copy.deepcopy(self.limits), new_limits=updated_limits,
            used=copy.deepcopy(self.state['used']), phases=copy.deepcopy(self.state['phases']),
            previous_extensions=copy.deepcopy(self.state.get('limit_extensions', [])))
        path = self.path.parent / relative
        _atomic_json(path, event, disk_guard=self.disk_guard)
        descriptor = dict(path=str(relative), sha256=file_sha256(path),
                          previous_limits=copy.deepcopy(self.limits), new_limits=copy.deepcopy(updated_limits))
        updated = copy.deepcopy(self.state)
        updated.update(limits=updated_limits,
                       limit_extensions=[*updated.get('limit_extensions', []), descriptor])
        self._save(updated)
        self.state, self.limits = updated, copy.deepcopy(updated_limits)
        return copy.deepcopy(descriptor)

    def iteration_capacity(self, optimizer_attempts):
        """在采集前核对可精确预知的更新预算，避免采完一轮才发现候选次数不足。"""
        _integer(optimizer_attempts, 'optimizer attempts per iteration', 1)
        required = dict(accepted_iterations=1, optimizer_attempts=optimizer_attempts)
        exhausted = {key: dict(required=count, remaining=self.limits[key]-self.state['used'][key])
                     for key, count in required.items() if self.limits[key]-self.state['used'][key] < count}
        return dict(can_start=not exhausted, exhausted=exhausted)

    @profiled('storage.budget_write')
    def _save(self, value):
        _atomic_json(self.path, value, replace=True, disk_guard=self.disk_guard)

    @profiled('storage.budget_reserve')
    def reserve(self, phase, **amounts):
        if not isinstance(phase, str) or not phase or not amounts:
            raise ValueError('budget reservation requires a named phase and amounts')
        if any(record['phase'] == phase for record in self.state.get('lease_settlements', {}).values()):
            raise ValueError('A settled lease phase cannot be reserved again')
        updated = copy.deepcopy(self.state)
        for key, count in amounts.items():
            if key not in BUDGET_KEYS:
                raise ValueError(f'unknown continuous-training budget counter: {key}')
            _integer(count, key)
            if updated['used'][key] + count > self.limits[key]:
                raise BudgetExceeded(f'{key} budget exhausted in {phase}')
            updated['used'][key] += count
            entries = updated['phases'].setdefault(phase, {})
            entries[key] = entries.get(key, 0) + count
        if self.disk_guard is not None:
            self.disk_guard.check()
        self._save(updated)
        self.state = updated

    def accept_iteration(self, phase='update', *, identity=None):
        # 接受后发布失败也不回退资源消耗；latest checkpoint独立给出可恢复进度。
        self.reserve(phase, accepted_iterations=1)

    @profiled('storage.budget_settle_control')
    def settle_control(self, phase, requested, result):
        _integer(requested, 'requested controls')
        if not result.get('physics_count_exact', True):
            return
        actual = _integer(result['executed_control_steps'], 'executed controls')
        physics = _integer(result.get('executed_physics_steps'), 'executed physics')
        if not 0 <= actual <= requested or not 4 * actual <= physics <= 4 * requested:
            raise ValueError('cannot settle inconsistent physical execution')
        refunds = dict(control_steps=requested - actual, physics_steps=4 * requested - physics)
        phase_used = self.state['phases'].get(phase)
        if phase_used is None or phase_used.get('control_steps', 0) < requested or phase_used.get('physics_steps', 0) < 4*requested:
            raise ValueError('control settlement requires a sufficient reserved phase budget')
        for key, count in refunds.items():
            if count > self.state['used'][key] or count > self.state['phases'].get(phase, {}).get(key, 0):
                raise ValueError('control settlement exceeds its reserved phase budget')
        if not any(refunds.values()):
            return
        updated = copy.deepcopy(self.state)
        for key, count in refunds.items():
            updated['used'][key] -= count
            updated['phases'][phase][key] -= count
        self._save(updated)
        self.state = updated

    def settle_lease(self, phase, reserved, used, *, lease_id):
        """仅精确完成的多 rank 租约可退未用额度；持久唯一 ID 阻止重启后二次退款。"""
        if not isinstance(phase, str) or not phase or not isinstance(lease_id, str) or not lease_id:
            raise ValueError('Lease settlement requires a phase and a unique lease ID')
        if not isinstance(reserved, dict) or not isinstance(used, dict) or set(reserved) != set(used):
            raise ValueError('Lease settlement requires matching reserved and used counters')
        if not reserved or set(reserved) - set(BUDGET_KEYS):
            raise ValueError('Invalid lease counters')
        for key in reserved:
            _integer(reserved[key], key)
            _integer(used[key], key)
            if used[key] > reserved[key]:
                raise ValueError('Lease used counters exceed reservation')
        for counters in (reserved, used):
            if ('control_steps' in counters or 'physics_steps' in counters) and (
                    set(counters) & {'control_steps', 'physics_steps'} != {'control_steps', 'physics_steps'}
                    or counters['physics_steps'] != 4 * counters['control_steps']):
                raise ValueError('Lease physics counter must equal four times controls')
        record = dict(phase=phase, reserved=copy.deepcopy(reserved), used=copy.deepcopy(used))
        previous = self.state.get('lease_settlements', {}).get(lease_id)
        if previous is not None:
            if previous != record:
                raise ValueError('Lease ID was already settled with different counters')
            return copy.deepcopy(previous)
        if any(item['phase'] == phase for item in self.state.get('lease_settlements', {}).values()):
            raise ValueError('Each lease phase may be settled only once')
        updated = copy.deepcopy(self.state)
        for key, amount in reserved.items():
            if amount > updated['phases'].get(phase, {}).get(key, 0):
                raise ValueError('Lease reservation exceeds recorded phase usage')
            refund = amount - used[key]
            updated['used'][key] -= refund
            updated['phases'][phase][key] -= refund
        updated.setdefault('lease_settlements', {})[lease_id] = record
        self._save(updated)
        self.state = updated
        return copy.deepcopy(record)

    def summary(self):
        return self.state_dict()

    def state_dict(self):
        return copy.deepcopy(self.state)


class RolloutWriter:
    """每条转移排他写入一次；每个块和最终完整清单都是不可覆盖的正式证据。"""
    def __init__(self, directory, *, policy_version, chunk_size=16, disk_guard=None):
        self.directory = Path(directory)
        if disk_guard is not None:
            self.directory = disk_guard._path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.policy_version = _integer(policy_version, 'policy_version')
        self.chunk_size = _integer(chunk_size, 'chunk_size', 1)
        self.disk_guard = disk_guard
        self._identities, self._pending, self._chunks = set(), [], []
        self._count = self._controls = self._physics = 0
        self._finished = False

    def append(self, transition):
        if self._finished:
            raise RuntimeError('cannot append to a completed rollout')
        if not isinstance(transition, UpperTransition):
            raise TypeError('rollout records must be UpperTransition objects')
        transition.validate()
        if not transition.transition_valid or transition.identity['policy_version'] != self.policy_version:
            raise ValueError('rollout requires valid transitions from one current policy version')
        identity = {key: transition.identity[key] for key in
                    ('run_id', 'backend_session_id', 'episode_id', 'decision_id', 'policy_version')}
        key = _json_bytes(identity)
        if key in self._identities:
            raise ValueError('duplicate transition identity cannot be appended twice')
        block = self.directory / f'chunk_{len(self._chunks):06d}'
        block.mkdir(exist_ok=True)
        path = block / f'transition_{self._count:09d}.pt'
        if self.disk_guard is not None:
            self.disk_guard.check(2 * _storage_estimate(transition) + 65536)
        descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=block)
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                torch.save(transition, stream)
                stream.flush()
                os.fsync(stream.fileno())
            size = temporary.stat().st_size
            if self.disk_guard is not None:
                self.disk_guard.account_file(temporary)
                self.disk_guard.check()
            digest = file_sha256(temporary)
            os.link(temporary, path)
            temporary.unlink()
            _sync_dir(block)
        finally:
            temporary.unlink(missing_ok=True)
            if self.disk_guard is not None:
                self.disk_guard.account_file(temporary)
                self.disk_guard.account_file(path)
        record = dict(path=path.name, sha256=digest, size_bytes=size, identity=identity,
                      executed_control_steps=transition.executed_control_steps,
                      executed_physics_steps=transition.executed_physics_steps)
        self._identities.add(key)
        self._pending.append(record)
        self._count += 1
        self._controls += transition.executed_control_steps
        self._physics += transition.executed_physics_steps
        if len(self._pending) == self.chunk_size:
            self._flush_chunk()
        return path

    def _flush_chunk(self):
        if not self._pending:
            return
        path = self.directory / f'chunk_{len(self._chunks):06d}' / 'manifest.json'
        manifest = dict(schema='genmo.closedloop.stage10.rollout_chunk.v1', policy_version=self.policy_version,
                        records=self._pending, record_count=len(self._pending))
        _atomic_json(path, manifest, disk_guard=self.disk_guard)
        self._chunks.append(dict(path=str(path.relative_to(self.directory)), sha256=file_sha256(path),
                                 record_count=len(self._pending)))
        self._pending = []

    def finish(self):
        if self._finished:
            raise RuntimeError('rollout manifest has already been published')
        if not self._count:
            raise ValueError('cannot publish an empty rollout')
        self._flush_chunk()
        path = self.directory / 'manifest.json'
        _atomic_json(path, dict(schema='genmo.closedloop.stage10.rollout.v1', complete=True,
            policy_version=self.policy_version, transition_count=self._count,
            executed_control_steps=self._controls, executed_physics_steps=self._physics,
            chunks=self._chunks), disk_guard=self.disk_guard)
        self._finished = True
        return path


class GuardedStepJournal:
    """保留StepJournal的先事务落盘语义，另对本轮SQLite/WAL占用做增量检查。"""
    def __init__(self, path, disk_guard):
        self.path, self.disk_guard = disk_guard._path(path), disk_guard
        if self.path.exists():
            raise FileExistsError('each new attempt requires a new execution journal')
        disk_guard.check(65536)
        self._journal = StepJournal(self.path)
        self._closed = False
        self._account()

    def _account(self):
        for suffix in ('', '-wal', '-shm'):
            self.disk_guard.account_file(Path(str(self.path) + suffix))

    def append_result(self, result):
        encoded = self._journal.encode_result(result)
        self.disk_guard.check(2 * len(encoded.payload) + 65536)
        try:
            return self._journal.append_encoded(encoded)
        finally:
            self._account()

    def __len__(self):
        return len(self._journal)

    def close(self):
        if not self._closed:
            self._journal.close()
            self._closed = True
            self._account()

    @property
    def closed(self):
        return self._closed


class StopSignal:
    """信号处理器只置位，训练入口在合法保存边界读取stop_requested并结束。"""
    def __init__(self):
        self.stop_requested = False
        self.signal_number = None
        self._handlers = {}

    def request(self, signal_number=None, _frame=None):
        self.stop_requested = True
        self.signal_number = signal_number

    def install(self):
        if self._handlers:
            raise RuntimeError('stop signal handlers are already installed')
        try:
            for number in (signal.SIGINT, signal.SIGTERM):
                previous = signal.getsignal(number)
                signal.signal(number, self.request)
                self._handlers[number] = previous
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self):
        for number, handler in self._handlers.items():
            signal.signal(number, handler)
        self._handlers.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.restore()


class RunManager:
    """单进程运行锁、轮次attempt目录、预算及checkpoint发布；不清理正式产物。"""
    def __init__(self, run_dir, *, resume=False, min_free_bytes=0, max_run_bytes=None):
        self.run_dir = Path(run_dir).expanduser().resolve()
        if resume and not self.run_dir.is_dir():
            raise FileNotFoundError(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = (self.run_dir / '.run.lock').open('a+b')
        self._closed, self._journals, self._directories = False, [], {}
        self._budget = None
        self._metrics_mutex = threading.RLock()
        self._maintenance = []
        self.session_id = str(uuid.uuid4())
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError('another training writer already owns this run directory') from None
        try:
            self.disk_guard = DiskGuard(self.run_dir, min_free_bytes=min_free_bytes, max_run_bytes=max_run_bytes)
            path = self.run_dir / 'run_manifest.json'
            if resume:
                manifest = _read_json(path)
                if manifest.get('schema') != 'genmo.closedloop.stage10.run.v1':
                    raise ValueError('unsupported run manifest')
                self.run_id = manifest['run_id']
            else:
                if any(path.name != '.run.lock' for path in self.run_dir.iterdir()):
                    raise FileExistsError('new training runs require an empty output directory')
                self.run_id = self.run_dir.name
                _atomic_json(path, dict(schema='genmo.closedloop.stage10.run.v1', run_id=self.run_id,
                    created_at=datetime.now(timezone.utc).isoformat()), disk_guard=self.disk_guard)
            self.append_metrics(dict(event='run_resume' if resume else 'run_created', session_id=self.session_id))
        except BaseException:
            self.close()
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('run manager is closed; no writer lock is held')

    def budget(self, limits, *, for_extension=False, incremental=False):
        self._ensure_open()
        if for_extension:
            # 扩展前先按旧账本构造；入口完成checkpoint和配置身份校验后才提交扩展事件。
            limits = _read_json(self.run_dir / 'budget.json')['limits']
        if self._budget is None:
            path = self.run_dir / 'budget.json'
            if incremental or (path.exists() and _read_json(path).get('schema') == 'genmo.closedloop.stage10.budget_ledger.v2'):
                from .budget_ledger import IncrementalBudget
                self._budget = IncrementalBudget(path, limits, disk_guard=self.disk_guard)
            else:
                self._budget = TrainingBudget(path, limits, disk_guard=self.disk_guard)
        elif self._budget.limits != limits:
            raise ValueError('cannot change active run budget limits')
        return self._budget

    def iteration_dir(self, index, attempt=None):
        self._ensure_open()
        index = _integer(index, 'iteration index')
        attempt = self.session_id if attempt is None else attempt
        if not isinstance(attempt, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', attempt):
            raise ValueError('attempt must be a bounded plain identifier')
        key = (index, attempt)
        if key not in self._directories:
            path = self.run_dir / 'iterations' / f'{index:06d}' / attempt
            path.mkdir(parents=True, exist_ok=False)
            self._directories[key] = path
        return self._directories[key]

    def rollout_writer(self, index, *, policy_version, chunk_size=16, attempt=None):
        return RolloutWriter(self.iteration_dir(index, attempt) / 'rollout', policy_version=policy_version,
                             chunk_size=chunk_size, disk_guard=self.disk_guard)

    def journal(self, index, attempt=None):
        journal = GuardedStepJournal(self.iteration_dir(index, attempt) / 'execution_journal.sqlite', self.disk_guard)
        self._journals.append(journal)
        return journal

    @profiled('storage.disk_check')
    def check_disk(self, required_bytes=0, *, refresh=False):
        self._ensure_open()
        if refresh:
            self.disk_guard.refresh()
        return self.disk_guard.check(required_bytes)

    def append_metrics(self, record):
        with self._metrics_mutex:
            return self._append_metrics_locked(record)

    def _append_metrics_locked(self, record):
        self._ensure_open()
        if not isinstance(record, dict):
            raise TypeError('metrics require a dictionary')
        # 每次进程启动使用独立追加文件，崩溃留下的末尾半行证据保留且不会污染新session。
        path = self.run_dir / 'metrics' / f'{self.session_id}.jsonl'
        path.parent.mkdir(exist_ok=True)
        payload = dict(record)
        payload.setdefault('session_id', self.session_id)
        payload.setdefault('time_utc', datetime.now(timezone.utc).isoformat())
        data = _json_bytes(payload)
        self.disk_guard.check(len(data))
        with path.open('ab') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.disk_guard.account_file(path)
        return path

    @profiled('storage.seal')
    def seal_iteration(self, directory, iteration, *, metadata=None, closed_journals=()):
        """发布独立接受水位及不可变文件清单；普通轮不要求写完整模型 checkpoint。"""
        self._ensure_open()
        iteration = _integer(iteration, 'accepted iteration', 1)
        directory = self.disk_guard._path(directory)
        summary_path = directory / 'summary.json'
        summary = _read_json(summary_path)
        if summary.get('status') != 'accepted' or summary.get('iteration') != iteration:
            raise ValueError('Only the matching accepted iteration can be sealed')
        known_closed = set()
        for journal in [*self._journals, *closed_journals]:
            if isinstance(journal, (str, Path)):
                # 路径表示调用方已在全 rank barrier 后确认关闭的远端 journal。
                path = self.disk_guard._path(journal)
            else:
                path = self.disk_guard._path(journal.path)
                if not getattr(journal, 'closed', getattr(journal, '_closed', False)):
                    if path.is_relative_to(directory):
                        raise ValueError('Cannot seal an open execution journal')
                    continue
            known_closed.add(path)
        preserved = {'summary.json', 'lr_progress.json', 'seal_manifest.json',
                     'archive_manifest.json', 'execution_evidence.tar.gz', 'superseded.json'}
        members = []
        for path in sorted(directory.rglob('*')):
            self.disk_guard._path(path)
            if not path.is_file() or path.parent == directory and path.name in preserved:
                continue
            if path.name.endswith(('-wal', '-shm')):
                raise ValueError('SQLite WAL/SHM must be closed and removed before sealing')
            if path.suffix == '.sqlite' and path not in known_closed:
                raise ValueError('Every execution journal requires an explicit closed declaration')
            if path.name.endswith('.tmp'):
                raise ValueError('Incomplete temporary evidence cannot be sealed')
            self.disk_guard.account_file(path)
            members.append(dict(path=str(path.relative_to(directory)), size_bytes=path.stat().st_size,
                                sha256=file_sha256(path)))
        if not members:
            raise ValueError('Cannot seal an empty execution record')
        seal_path = directory / 'seal_manifest.json'
        seal = dict(schema='genmo.closedloop.stage10.iteration_seal.v2', iteration=iteration,
                    session_id=self.session_id, summary_sha256=file_sha256(summary_path),
                    journals_closed=True, members=members, metadata=copy.deepcopy(metadata or {}))
        if seal_path.exists():
            if _read_json(seal_path) != seal:
                raise ValueError('Iteration seal differs from its immutable evidence')
        else:
            _atomic_json(seal_path, seal, disk_guard=self.disk_guard)
        accepted_path = self.run_dir / 'accepted.json'
        descriptor = dict(schema='genmo.closedloop.stage10.accepted_iteration.v2', iteration=iteration,
                          seal=str(seal_path.relative_to(self.run_dir)), seal_sha256=file_sha256(seal_path),
                          session_id=self.session_id)
        if accepted_path.exists():
            previous = _read_json(accepted_path)
            if previous['iteration'] > iteration or previous['iteration'] == iteration and previous != descriptor:
                raise ValueError('Accepted watermark must advance; reconcile stale tail before resume')
        _atomic_json(accepted_path, descriptor, replace=True, disk_guard=self.disk_guard)
        return seal_path

    def reconcile_accepted(self, durable_iteration):
        """恢复较早完整 checkpoint 时标记未持久训练尾部作废；已花资源不回退。"""
        self._ensure_open()
        durable_iteration = _integer(durable_iteration, 'durable iteration')
        accepted_path = self.run_dir / 'accepted.json'
        if not accepted_path.exists():
            return None
        previous = _read_json(accepted_path)
        if previous['iteration'] <= durable_iteration:
            return None
        event = dict(schema='genmo.closedloop.stage10.superseded_tail.v2',
                     session_id=self.session_id, durable_iteration=durable_iteration,
                     previous_accepted=previous, spent_budget_preserved=True)
        path = self.run_dir / 'superseded_tails' / f'{self.session_id}.json'
        _atomic_json(path, event, disk_guard=self.disk_guard)
        _atomic_json(accepted_path, dict(schema='genmo.closedloop.stage10.accepted_iteration.v2',
                     iteration=durable_iteration, session_id=self.session_id,
                     reconciliation=str(path.relative_to(self.run_dir))), replace=True,
                     disk_guard=self.disk_guard)
        self.append_metrics(dict(event='unsaved_training_tail_superseded',
                            durable_iteration=durable_iteration, accepted_iteration=previous['iteration']))
        return path

    def publish_checkpoint(self, iteration, checkpoint_path, metadata=None):
        self._ensure_open()
        iteration = _integer(iteration, 'checkpoint iteration', 1)
        path = self.disk_guard._path(checkpoint_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError('complete checkpoint must be a regular file inside this run')
        latest_path = self.run_dir / 'latest.json'
        if latest_path.exists() and _read_json(latest_path)['iteration'] >= iteration:
            raise ValueError('checkpoint publication must advance the accepted iteration watermark')
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        required = {'actor', 'critic', 'actor_optimizer', 'critic_optimizer', 'identity',
                    'config', 'state', 'optimizer_layout', 'restore_environment'}
        version = payload.get('version')
        required.update({'rank_states', 'world_size'} if version == VERSION_V2 else {'rng', 'samplers'})
        if version not in (CHECKPOINT_VERSION, VERSION_V2) or not required.issubset(payload):
            raise ValueError('latest requires a complete Actor/Critic training checkpoint')
        if version == VERSION_V2 and payload['world_size'] != _validate_rank_states(payload['rank_states']):
            raise ValueError('Checkpoint rank count differs from its world size')
        state = payload['state']
        if state.get('buffer_size') != 0 or state.get('pending_plan') is not False or state.get('iteration') != iteration:
            raise ValueError('checkpoint must match the iteration at an empty-buffer/no-pending boundary')
        del payload
        self.disk_guard.account_file(path)
        self.disk_guard.check()
        descriptor = dict(schema='genmo.closedloop.stage10.checkpoint_publication.v1', iteration=iteration,
            path=str(path.relative_to(self.run_dir)), sha256=file_sha256(path), size_bytes=path.stat().st_size,
            session_id=self.session_id, metadata=copy.deepcopy(metadata or {}),
            budget=None if self._budget is None else self._budget.state_dict())
        publication = self.run_dir / 'checkpoints' / 'publications' / f'{iteration:09d}-{self.session_id}.json'
        _atomic_json(publication, descriptor, disk_guard=self.disk_guard)
        _atomic_json(latest_path, dict(descriptor, publication=str(publication.relative_to(self.run_dir))),
                     replace=True, disk_guard=self.disk_guard)
        return latest_path

    def latest_checkpoint(self):
        self._ensure_open()
        latest = _read_json(self.run_dir / 'latest.json')
        if latest.get('schema') != 'genmo.closedloop.stage10.checkpoint_publication.v1':
            raise ValueError('unsupported checkpoint publication')
        publication = self.disk_guard._path(self.run_dir / latest['publication'])
        if _read_json(publication) != {key: value for key, value in latest.items() if key != 'publication'}:
            raise ValueError('latest pointer differs from its immutable checkpoint publication')
        path = self.disk_guard._path(self.run_dir / latest['path'])
        if not path.is_file() or path.stat().st_size != latest['size_bytes'] or file_sha256(path) != latest['sha256']:
            raise ValueError('latest checkpoint is missing, incomplete or has a different SHA')
        return path

    def close(self):
        if not self._closed:
            failure = None
            try:
                for maintenance in self._maintenance:
                    try:
                        maintenance.close()
                    except Exception as error:
                        failure = failure or error
                for journal in self._journals:
                    try:
                        journal.close()
                    except Exception as error:
                        failure = failure or error
            finally:
                if self._budget is not None and hasattr(self._budget, "close"):
                    self._budget.close()
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
                self._closed = True
            if failure is not None:
                raise failure

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
