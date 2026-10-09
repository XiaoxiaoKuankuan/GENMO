"""第二阶段根进程增量预算账本及旧格式兼容读取。

SQLite以FULL事务追加带前序SHA的事件，八卡预占/结算分别只提交一个事务；内存
仅在事务持久化后更新计数，失败时禁止继续使用旧余额。唯一lease和接受轮次身份
阻止重复退款或计费，未知执行仍保留原预占。初始完整v1快照作为不可变起点，新
budget.json仅发布数据库身份，不随历史增长重写；完整checkpoint仍保存可独立
检查的v1状态，普通轮保存带链头SHA的常量大小引用。恢复/审计重放原始事务，核对
每一事件的余额、唯一身份和SHA链。旧JSON可显式迁移，原文件保留为数据库内基线；
新数值执行身份仍要求新run，账本迁移本身不授权旧模型跨执行契约续训或扩限。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from .budget import BudgetExceeded
from .performance import profiled
from .run_management import BUDGET_KEYS, TrainingBudget, _atomic_json, _budget_limits, _integer, _json_bytes

SCHEMA = 'genmo.closedloop.stage10.budget_ledger.v2'
REFERENCE = 'genmo.closedloop.stage10.budget_reference.v2'


def _digest(previous, payload):
    return hashlib.sha256(previous.encode()+b'\n'+payload).hexdigest()


class IncrementalBudget(TrainingBudget):
    def __init__(self, path, limits, *, disk_guard=None):
        self.path, self.disk_guard = Path(path), disk_guard
        self.limits = _budget_limits(limits)
        self.database = self.path.with_suffix('.sqlite')
        descriptor = json.loads(self.path.read_text()) if self.path.exists() else None
        if descriptor is None or descriptor.get('schema') != SCHEMA:
            initial = (TrainingBudget(self.path, limits, disk_guard=disk_guard).state_dict()
                       if descriptor is not None else dict(schema='genmo.closedloop.stage10.budget.v1',
                           limits=self.limits.copy(), used={key: 0 for key in BUDGET_KEYS}, phases={}))
            if self.database.exists():
                # 未发布数据库不含执行权限；禁止覆盖可能属于另一次中断的事务。
                raise FileExistsError('Unpublished budget database requires explicit recovery inspection')
            connection = sqlite3.connect(self.database)
            try:
                connection.execute('PRAGMA synchronous=FULL')
                with connection:
                    connection.execute('CREATE TABLE base (payload BLOB NOT NULL, sha256 TEXT NOT NULL)')
                    connection.execute('CREATE TABLE events (seq INTEGER PRIMARY KEY, identity TEXT UNIQUE NOT NULL, '
                                       'previous TEXT NOT NULL, payload BLOB NOT NULL, sha256 TEXT NOT NULL)')
                    payload = _json_bytes(initial)
                    base_sha = hashlib.sha256(payload).hexdigest()
                    connection.execute('INSERT INTO base VALUES (?,?)', (payload, base_sha))
                _atomic_json(self.path, dict(schema=SCHEMA, database=self.database.name,
                    base_sha256=base_sha), replace=descriptor is not None, disk_guard=disk_guard)
            finally:
                connection.close()
        self.connection = sqlite3.connect(self.database)
        self.connection.execute('PRAGMA synchronous=FULL')
        self.state, self.sequence, self.head = replay_budget(self.path)
        self._validate()
        if self.state['limits'] != self.limits:
            raise ValueError('resume cannot silently change an existing run budget')
        self._settled_phases = {r['phase'] for r in self.state.get('lease_settlements', {}).values()}
        self._accepted = set(self.state.get('accepted_identities', []))
        self.poisoned = False

    def _save(self, value):
        raise RuntimeError('Incremental budget requires an explicit transaction; full-state overwrite forbidden')

    def _change(self, operations, *, identity):
        if self.poisoned:
            raise RuntimeError('Budget transaction outcome requires recovery before continuing')
        # 验证只拷贝本次触及的phase和固定大小余额，历史字典不随事务复制。
        changes, settlements, accepted = _validate_operations(self.state, operations, self._settled_phases, self._accepted)
        payload = _json_bytes(dict(operations=operations))
        head = _digest(self.head, payload)
        if self.disk_guard is not None:
            self.disk_guard.check(2*len(payload)+65536)
        try:
            with self.connection:
                self.connection.execute('INSERT INTO events VALUES (?,?,?,?,?)',
                    (self.sequence+1, identity, self.head, payload, head))
        except BaseException:
            self.poisoned = True
            raise
        _apply(self.state, changes, settlements, accepted)
        self._settled_phases.update(record['phase'] for record in settlements.values())
        self._accepted.update(accepted)
        self.sequence += 1
        self.head = head
        if self.disk_guard is not None:
            self.disk_guard.account_file(self.database)

    @profiled('storage.budget_transaction')
    def reserve_many(self, entries):
        self._change([dict(kind='reserve', phase=phase, amounts=dict(amounts)) for phase, amounts in entries],
                     identity='reserve:'+str(uuid.uuid4()))

    def reserve(self, phase, **amounts):
        self.reserve_many([(phase, amounts)])

    @profiled('storage.budget_transaction')
    def settle_many(self, entries):
        operations = []
        for phase, reserved, used, lease_id in entries:
            record = dict(phase=phase, reserved=reserved, used=used)
            previous = self.state.get('lease_settlements', {}).get(lease_id)
            if previous is not None:
                if previous != record:
                    raise ValueError('Lease ID already settled with different counters')
                continue
            operations.append(dict(kind='settle', identity=lease_id, **record))
        if operations:
            self._change(operations, identity='settle:'+str(uuid.uuid4()))

    def settle_lease(self, phase, reserved, used, *, lease_id):
        self.settle_many([(phase, reserved, used, lease_id)])
        return copy.deepcopy(self.state['lease_settlements'][lease_id])

    def settle_control(self,phase,requested,result):
        """仅退回已确认未执行的子步；终止短advance以独立事务幂等结算。"""
        if not result.get('physics_count_exact',True):return
        requested=_integer(requested,'requested controls')
        actual=_integer(result['executed_control_steps'],'executed controls')
        physics=_integer(result['executed_physics_steps'],'executed physics')
        if not 0<=actual<=requested or not 4*actual<=physics<=4*requested:raise ValueError('Inconsistent physical settlement')
        if actual==requested and physics==4*requested:return
        session=result.get('backend_session_id');sequence=result.get('mutation_seq')
        if not isinstance(session,str) or not session or type(sequence)is not int or sequence<1:
            raise ValueError('Partial control settlement requires acknowledged execution identity')
        identity=f'{session}:{sequence}'
        record=dict(phase=phase,requested=requested,actual=actual,physics=physics)
        previous=self.state.get('control_settlements',{}).get(identity)
        if previous is not None:
            if previous!=record:raise ValueError('Control settlement identity payload changed')
            return
        self._change([dict(kind='control_settle',identity=identity,**record)],identity='control:'+identity)

    def accept_iteration(self, phase='update', *, identity=None):
        if not isinstance(identity, str) or not identity:
            raise ValueError('Incremental acceptance requires a unique session/iteration identity')
        self._change([dict(kind='accept', phase=phase, identity=identity)], identity='accept:'+identity)

    def summary(self):
        return dict(schema=REFERENCE, limits=self.limits.copy(), used=self.state['used'].copy(),
                    ledger=dict(sequence=self.sequence, sha256=self.head))

    def state_dict(self):
        value = copy.deepcopy(self.state)
        value['ledger'] = dict(sequence=self.sequence, sha256=self.head)
        return value

    def close(self):
        self.connection.close()


def _validate_operations(state, operations, settled_phases, accepted_ids):
    used = state['used'].copy()
    phases, settlements, accepted, controls = {}, {}, [], {}
    if not isinstance(operations, list) or not operations:
        raise ValueError('Budget transaction must contain operations')
    for operation in operations:
        phase, kind = operation.get('phase'), operation.get('kind')
        if not isinstance(phase, str) or not phase or phase in settled_phases:
            raise ValueError('Invalid or already settled budget phase')
        values = phases.setdefault(phase, state['phases'].get(phase, {}).copy())
        if kind in ('reserve', 'accept'):
            amounts = operation.get('amounts') if kind == 'reserve' else dict(accepted_iterations=1)
            if kind == 'accept':
                identity = operation.get('identity')
                if not isinstance(identity, str) or not identity or identity in accepted_ids or identity in accepted:
                    raise ValueError('Duplicate accepted iteration identity')
                accepted.append(identity)
            if not amounts or set(amounts)-set(BUDGET_KEYS):
                raise ValueError('Unknown or empty budget reservation')
            for key, count in amounts.items():
                count = _integer(count, key)
                if used[key]+count > state['limits'][key]:
                    raise BudgetExceeded(f'{key} budget exhausted in {phase}')
                used[key] += count
                values[key] = values.get(key, 0)+count
        elif kind == 'control_settle':
            identity=operation.get('identity')
            if not isinstance(identity,str) or not identity or identity in controls or identity in state.get('control_settlements',{}):
                raise ValueError('Duplicate control settlement identity')
            requested=_integer(operation['requested'],'requested controls')
            actual=_integer(operation['actual'],'executed controls');physics=_integer(operation['physics'],'executed physics')
            if not 0<=actual<=requested or not 4*actual<=physics<=4*requested:
                raise ValueError('Inconsistent physical settlement')
            for key,reserved,spent in (('control_steps',requested,actual),('physics_steps',4*requested,physics)):
                if reserved>values.get(key,0) or reserved>used[key]:raise ValueError('Control settlement exceeds reservation')
                used[key]-=reserved-spent;values[key]-=reserved-spent
            controls[identity]=dict(phase=phase,requested=requested,actual=actual,physics=physics)
        elif kind == 'settle':
            identity = operation.get('identity')
            reserved, spent = operation.get('reserved'), operation.get('used')
            if (not isinstance(identity, str) or not identity or identity in settlements
                    or identity in state.get('lease_settlements', {}) or
                    any(item['phase'] == phase for item in settlements.values())):
                raise ValueError('Duplicate or invalid settlement identity')
            if not isinstance(reserved, dict) or not reserved or set(reserved) != set(spent) or set(reserved)-set(BUDGET_KEYS):
                raise ValueError('Invalid settlement counters')
            for counts in (reserved, spent):
                for key, count in counts.items():
                    _integer(count, key)
                if ('control_steps' in counts or 'physics_steps' in counts) and counts.get('physics_steps') != 4*counts.get('control_steps', -1):
                    raise ValueError('Physics settlement must contain four substeps per control')
            for key, count in reserved.items():
                if spent[key] > count or count > values.get(key, 0):
                    raise ValueError('Settlement exceeds reserved phase')
                refund = count-spent[key]
                used[key] -= refund
                values[key] -= refund
            settlements[identity] = dict(phase=phase, reserved=reserved.copy(), used=spent.copy())
        else:
            raise ValueError('Unknown budget operation')
    return dict(used=used, phases=phases, control_settlements=controls), settlements, accepted


def _apply(state, changes, settlements, accepted):
    state['used'] = changes['used']
    state['phases'].update(changes['phases'])
    if changes.get('control_settlements'):
        state.setdefault('control_settlements',{}).update(changes['control_settlements'])
    if settlements:
        state.setdefault('lease_settlements', {}).update(settlements)
    if accepted:
        state.setdefault('accepted_identities', []).extend(accepted)


def replay_budget(path, *, sequence=None):
    path = Path(path)
    descriptor = json.loads(path.read_text())
    if descriptor.get('schema') != SCHEMA or descriptor.get('database') != path.with_suffix('.sqlite').name:
        raise ValueError('Invalid incremental budget descriptor')
    database = path.parent/descriptor['database']
    if database.is_symlink() or not database.is_file():
        raise ValueError('Missing or aliased budget database')
    connection = sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True)
    try:
        if connection.execute('PRAGMA quick_check').fetchone() != ('ok',):
            raise ValueError('Damaged budget database')
        bases = connection.execute('SELECT payload,sha256 FROM base').fetchall()
        if len(bases) != 1:
            raise ValueError('Budget requires one immutable base')
        payload, head = bases[0]
        if hashlib.sha256(payload).hexdigest() != head or head != descriptor['base_sha256']:
            raise ValueError('Budget base SHA mismatch')
        state = json.loads(payload)
        settled = {r['phase'] for r in state.get('lease_settlements', {}).values()}
        accepted = set(state.get('accepted_identities', []))
        current = 0
        for number, previous, payload, digest in connection.execute('SELECT seq,previous,payload,sha256 FROM events ORDER BY seq'):
            if sequence is not None and number > sequence:
                break
            if number != current+1 or previous != head or _digest(head, payload) != digest:
                raise ValueError('Budget transaction SHA/sequence mismatch')
            changes, records, identities = _validate_operations(state, json.loads(payload)['operations'], settled, accepted)
            _apply(state, changes, records, identities)
            settled.update(r['phase'] for r in records.values()); accepted.update(identities)
            current, head = number, digest
        if sequence is not None and current != sequence:
            raise ValueError('Budget reference points beyond durable transactions')
        return state, current, head
    finally:
        connection.close()


def read_budget_state(path, reference=None):
    descriptor = json.loads(Path(path).read_text())
    if descriptor.get('schema') != SCHEMA:
        return descriptor
    anchor = (reference or {}).get('ledger')
    state, sequence, head = replay_budget(path, sequence=None if anchor is None else anchor['sequence'])
    state['ledger'] = dict(sequence=sequence, sha256=head)
    if reference is not None:
        if (state['ledger'] != anchor or state['used'] != reference['used'] or state['limits'] != reference['limits']):
            raise ValueError('Budget reference differs from durable transaction history')
    return state


def expand_budget_reference(root, value):
    if value is None or value.get('schema') != REFERENCE:
        return value
    return read_budget_state(Path(root)/'budget.json', value)
