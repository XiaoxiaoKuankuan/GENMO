"""第二阶段真实块级rollout持久化与原始链单份引用。

每个块最多chunk_size条已冻结的UpperTransition，块文件一次序列化、fsync、原子
发布后再发布SHA清单。完整随机链、旧概率、均值和生成结果只保存在原raw_samples
证据中，块内保存限定rank目录内的路径、大小、SHA及剩余转移字段；读取时先验证
原证据再重建原UpperTransition，算法和离线审计继续使用同一数据结构。未满块只
驻内存，不声称已发布；物理执行仍由独立FULL journal保障，崩溃尾部不能进入学习。
v1逐文件格式继续可读，新格式显式标记v2并绑定运行配置，不覆盖旧清单或原始证据。
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import tempfile

import torch

from .buffer import UpperTransition
from .performance import measure, profiled
from .run_management import RolloutWriter, _atomic_json, _json_bytes, _sync_dir, file_sha256


def _write_block(path, rows, guard):
    with measure('storage.rollout_serialize'):
        buffer = io.BytesIO()
        torch.save(rows, buffer)
    content = buffer.getbuffer()
    try:
        if guard is not None:
            guard.check(len(content)+65536)
        digest = hashlib.sha256(content).hexdigest()
        fd, temporary = tempfile.mkstemp(prefix='.'+path.name, suffix='.tmp', dir=path.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, 'wb') as stream:
                with measure('storage.rollout_write'):
                    stream.write(content); stream.flush()
                with measure('storage.rollout_fsync'):
                    os.fsync(stream.fileno())
            os.link(temporary, path)
            temporary.unlink()
            _sync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
            if guard is not None:
                guard.account_file(path)
        return digest, len(content)
    finally:
        content.release(); buffer.close()


class BlockRolloutWriter(RolloutWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._objects = []

    def append(self, transition):
        if self._finished:
            raise RuntimeError('Cannot append to completed rollout')
        transition.validate()
        if not transition.transition_valid or transition.identity['policy_version'] != self.policy_version:
            raise ValueError('Rollout requires valid current-policy transitions')
        identity = {key: transition.identity[key] for key in
                    ('run_id', 'backend_session_id', 'episode_id', 'decision_id', 'policy_version')}
        key = _json_bytes(identity)
        if key in self._identities:
            raise ValueError('Duplicate transition identity')
        raw = Path(transition.metadata['raw_sample_path']).resolve(strict=True)
        rank_root = self.directory.parent.resolve()
        if not raw.is_relative_to(rank_root) or not _valid_raw_relative(raw.relative_to(rank_root), transition.metadata) or raw.is_symlink():
            raise ValueError('Raw trace must remain in its rank raw_samples directory')
        evidence = transition.metadata.get('raw_evidence_identity')
        if not isinstance(evidence, dict) or evidence.get('size_bytes') != raw.stat().st_size:
            raise ValueError('Raw trace requires its immutable publication size and SHA')
        fields = vars(transition).copy()
        fields['metadata'] = transition.metadata.copy()
        for name in ('chain', 'old_log_prob', 'free_mask'):
            fields.pop(name)
        for name in ('sampler_trace', 'generated'):
            fields['metadata'].pop(name)
        self._objects.append(dict(schema='genmo.rollout.raw_reference.v2', fields=fields,
            raw=dict(path=str(raw.relative_to(rank_root)), **evidence)))
        self._pending.append(dict(identity=identity, index=len(self._objects)-1,
            storage='block_raw_reference.v2', executed_control_steps=transition.executed_control_steps,
            executed_physics_steps=transition.executed_physics_steps))
        self._identities.add(key)
        self._count += 1
        self._controls += transition.executed_control_steps
        self._physics += transition.executed_physics_steps
        if len(self._objects) == self.chunk_size:
            self._flush_chunk()

    @profiled('storage.rollout_block')
    def _flush_chunk(self):
        if not self._pending:
            return
        directory = self.directory/f'chunk_{len(self._chunks):06d}'
        directory.mkdir(exist_ok=False)
        block = directory/'transitions.pt'
        sha, size = _write_block(block, self._objects, self.disk_guard)
        records = [dict(row, path=block.name, sha256=sha, size_bytes=size) for row in self._pending]
        path = directory/'manifest.json'
        _atomic_json(path, dict(schema='genmo.closedloop.stage10.rollout_chunk.v2', policy_version=self.policy_version,
                               records=records, record_count=len(records)), disk_guard=self.disk_guard)
        self._chunks.append(dict(path=str(path.relative_to(self.directory)), sha256=file_sha256(path), record_count=len(records)))
        self._pending, self._objects = [], []

    def finish(self):
        if self._finished or not self._count:
            raise ValueError('Rollout must be nonempty and not already published')
        self._flush_chunk()
        path = self.directory/'manifest.json'
        _atomic_json(path, dict(schema='genmo.closedloop.stage10.rollout.v2', complete=True,
            policy_version=self.policy_version, transition_count=self._count,
            executed_control_steps=self._controls, executed_physics_steps=self._physics,
            chunks=self._chunks), disk_guard=self.disk_guard)
        self._finished = True
        return path


def _valid_raw_relative(relative, metadata):
    """GPU路径额外绑定环境槽位，保持旧单环境路径且拒绝越界/交叉环境引用。"""
    if relative.is_absolute() or '..' in relative.parts:return False
    if relative.parent == Path('raw_samples'):return True
    slot = metadata.get('collector_env_slot')
    return type(slot) is int and slot >= 0 and relative.parent == Path(f'env{slot:03d}/raw_samples')


def load_rollout_record(path, record, *, rank_directory, physical=lambda path: path, cache=None):
    """校验块与原证据后重建；physical仅供已验证归档的路径解析器使用。"""
    path = Path(path)
    cache = {} if cache is None else cache
    key = (str(path), record['sha256'], record['size_bytes'])
    if key not in cache:
        actual = physical(path)
        if actual.stat().st_size != record['size_bytes'] or file_sha256(actual) != record['sha256']:
            raise ValueError('Rollout block size/SHA mismatch')
        cache.clear()  # 最多保留一个块，离线全量审计不无限增长
        cache[key] = torch.load(actual, map_location='cpu', weights_only=False, mmap=True)
    payload = cache[key]
    if record.get('storage') != 'block_raw_reference.v2':
        return payload
    index = record.get('index')
    if type(index) is not int or not 0 <= index < len(payload):
        raise ValueError('Invalid rollout block index')
    compact = payload[index]
    if compact.get('schema') != 'genmo.rollout.raw_reference.v2':
        raise ValueError('Invalid raw trace reference schema')
    reference = compact['raw']
    relative = Path(reference['path'])
    if not _valid_raw_relative(relative, compact['fields']['metadata']):
        raise ValueError('Raw trace reference escapes rank directory')
    raw_path = physical(Path(rank_directory)/relative)
    if raw_path.stat().st_size != reference['size_bytes'] or file_sha256(raw_path) != reference['sha256']:
        raise ValueError('Raw trace size/SHA mismatch')
    raw = torch.load(raw_path, map_location='cpu', weights_only=False, mmap=True)
    trace = raw['trace']
    fields = compact['fields'].copy()
    if raw['policy_version'] != fields['identity']['policy_version']:
        raise ValueError('Raw trace policy identity mismatch')
    fields['metadata'] = dict(fields['metadata'], sampler_trace=trace, generated=raw['generated'])
    result = UpperTransition(**fields, chain=trace['chain'][0], old_log_prob=trace['old_log_probs'][0],
                             free_mask=trace['free_mask'][0])
    result.validate()
    return result
