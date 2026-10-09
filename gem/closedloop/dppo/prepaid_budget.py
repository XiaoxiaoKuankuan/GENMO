"""世界采样使用的有界、已预付资源子账本。

根进程必须先将整个rank最坏资源额度持久化预占，才能建立本子账本。单环境每次
advance/generation仍立即检查自己的上限并生成原格式SHA链事件，但最多64个事件
合并为一次SQLite FULL事务；不会为每个小RPC重复fsync。达到界限同步写入形成
真实背压，完整state/summary、阶段结算和关闭前强制刷写，不能无限后台积压。

如果进程在刷写前退出，根预算中整份预占仍算已耗，未确认余额不能退款。子账本
不具有增加授权或独立恢复物理的能力。异常刷写后禁止继续；审计仍使用既有v2
事件重放，不另造预算算法。记录刷写次数、等待时间和队列峰值，供端到端测速。
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

from .budget_ledger import IncrementalBudget, _validate_operations, _apply, _digest
from .run_management import _atomic_json, _json_bytes


class PrepaidBudget(IncrementalBudget):
    def __init__(self, path, limits, *, parent_lease, maximum_pending=64, disk_guard=None):
        parent_lease = Path(parent_lease)
        raw = parent_lease.read_bytes()
        prepaid = json.loads(raw)
        if (prepaid.get('schema') != 'genmo.closedloop.stage10.budget.v1' or
                any(limits[key] > prepaid['limits'][key] for key in limits)):
            raise ValueError('World sub-budget requires an already allocated parent lease')
        if type(maximum_pending) is not int or not 1 <= maximum_pending <= 256:
            raise ValueError('Unbounded budget event queue')
        super().__init__(path, limits, disk_guard=disk_guard)
        self.maximum_pending, self.pending = maximum_pending, []
        self.flush_count = self.peak_pending = 0
        self.flush_seconds = 0.
        self.closed = False
        _atomic_json(self.path.with_name('budget_prepaid.json'), dict(
            schema='genmo.world_prepaid_budget.v1', parent_lease=parent_lease.name,
            parent_sha256=hashlib.sha256(raw).hexdigest(), maximum_pending=maximum_pending,
            crash_rule='unsettled_parent_credit_remains_spent'), disk_guard=disk_guard)

    def _change(self, operations, *, identity):
        if self.poisoned or self.closed: raise RuntimeError('Prepaid budget requires recovery')
        changes, settlements, accepted = _validate_operations(
            self.state, operations, self._settled_phases, self._accepted)
        payload = _json_bytes(dict(operations=operations))
        head = _digest(self.head, payload)
        self.pending.append((self.sequence+1, identity, self.head, payload, head))
        _apply(self.state, changes, settlements, accepted)
        self._settled_phases.update(record['phase'] for record in settlements.values())
        self._accepted.update(accepted)
        self.sequence += 1; self.head = head
        self.peak_pending = max(self.peak_pending, len(self.pending))
        if len(self.pending) >= self.maximum_pending: self.flush()

    def flush(self):
        if self.poisoned: raise RuntimeError('Prepaid budget write failed; no settlement permitted')
        if not self.pending: return
        started = time.perf_counter()
        try:
            if self.disk_guard is not None:
                self.disk_guard.check(sum(len(row[3]) for row in self.pending)*2+65536)
            with self.connection:
                self.connection.executemany('INSERT INTO events VALUES (?,?,?,?,?)', self.pending)
            self.pending.clear(); self.flush_count += 1
            if self.disk_guard is not None: self.disk_guard.account_file(self.database)
        except BaseException:
            self.poisoned = True
            raise
        finally: self.flush_seconds += time.perf_counter()-started

    def summary(self):
        self.flush()
        return super().summary()

    def state_dict(self):
        self.flush()
        return super().state_dict()

    def timing(self):
        return dict(flush_count=self.flush_count, flush_seconds=self.flush_seconds,
            pending_events=len(self.pending), peak_pending_events=self.peak_pending,
            maximum_pending_events=self.maximum_pending)

    def close(self):
        if not self.closed:
            try: self.flush()
            finally:
                super().close(); self.closed = True
