"""训练侧带确认水位的冻结 GMT 客户端。

每个有副作用的操作都带 worker session 和单调 mutation_seq；回复先交给持久化 journal，
再确认回收。断链只重发同一个已编号请求，绝不换编号重做物理动作。业务错误同样保存
并确认，执行失败的完整 trace 留给环境适配器判断有效性。Stage9 必须协商 ack.v2，
不允许静默退回第八步只按 advance_id 缓存的接口。
"""
from __future__ import annotations

import time

from gem.runtime.closedloop_protocol import RemoteError, RpcClient
from .performance import measure, profiled, record_cpu


class AcknowledgedBackend:
    MUTATIONS = {'reset_episode', 'reserve_prefix', 'prepare_plan', 'commit_plan',
                 'discard_plan', 'advance'}

    def __init__(self, client, journal, *, socket_path=None, timeout_s=120):
        self.client, self.journal = client, journal
        self.socket_path, self.timeout_s = socket_path, timeout_s
        hello = client.call('hello')
        if 'ack.v2' not in hello.get('execution_protocols', []):
            raise RuntimeError('Stage9 requires acknowledged execution protocol ack.v2')
        self.session_id = hello['backend_session_id']
        self.batched_lane_ack = hello.get('batched_lane_ack')
        if hello['executed_seq'] != hello['acked_seq']:
            raise RuntimeError('Cannot adopt a worker with an unknown outstanding mutation')
        self.sequence = int(hello['executed_seq'])
        self.episode_id = None
        self.last_envelope = None
        self.last_call_timing = None

    @profiled('rpc.transport_including_remote_execution')
    def _transport(self, method, **payload):
        try:
            return self._client_call(method, **payload)
        except (OSError, EOFError):
            if self.socket_path is None:
                raise
            self.client.close()
            self.client = RpcClient(self.socket_path, timeout_s=self.timeout_s)
            hello = self.client.call('hello')
            if hello.get('backend_session_id') != self.session_id:
                raise RuntimeError('Worker session changed during uncertain execution') from None
            # 传输层sequence重新开始，业务mutation_seq保持不变。
            return self._client_call(method, **payload)

    def _client_call(self, method, **payload):
        try:
            return self.client.call(method, **payload)
        finally:
            for name, seconds in getattr(self.client, 'last_call_timing', {}).items():
                if name.endswith('_seconds'):
                    record_cpu('rpc.protocol.'+name, seconds)

    def call(self, method, **payload):
        if method not in self.MUTATIONS:
            started = time.perf_counter()
            try:
                return self._transport(method, **payload)
            finally:
                elapsed = time.perf_counter()-started
                self.last_call_timing = dict(method=method, total_seconds=elapsed,
                    transport_seconds=elapsed, journal_seconds=0., ack_seconds=0., critical_seconds=elapsed)
        started = time.perf_counter()
        journal_seconds = ack_seconds = transport_seconds = 0.
        try:
            return self._mutation(method, payload, started)
        finally:
            # _mutation逐段更新；失败也保留实际耗时，但从不提前ACK未持久化的结果。
            previous = self.last_call_timing
            if previous is not None and previous.get('_started') == started:
                journal_seconds = previous['journal_seconds']
                ack_seconds = previous['ack_seconds']
                transport_seconds = previous['transport_seconds']
            elapsed = time.perf_counter()-started
            self.last_call_timing = dict(method=method, total_seconds=elapsed,
                transport_seconds=transport_seconds, journal_seconds=journal_seconds,
                ack_seconds=ack_seconds, critical_seconds=max(0., elapsed-journal_seconds),
                journal_interval=(previous.get('journal_interval') if previous is not None
                                  and previous.get('_started')==started else None),
                journal_detail=dict(getattr(self.journal,'last_timing',{})))

    def _mutation(self, method, payload, started):
        """只分离部署RPC与训练journal计时，持久化/ACK顺序和重发身份保持原样。"""
        timing = dict(_started=started, transport_seconds=0., journal_seconds=0., ack_seconds=0.)
        self.last_call_timing = timing
        seq = self.sequence + 1
        expected = payload.get('expected_episode_id', self.episode_id)
        beginning = time.perf_counter()
        try:
            envelope = self._transport('execute', backend_session_id=self.session_id,
                mutation_seq=seq, expected_episode_id=expected, operation=method, payload=payload)
        finally:
            timing['transport_seconds'] = time.perf_counter()-beginning
        if (envelope['backend_session_id'] != self.session_id or envelope['mutation_seq'] != seq
                or envelope['operation'] != method):
            raise RuntimeError('Mutation response identity mismatch')
        self.last_envelope = envelope
        beginning = time.perf_counter()
        try:
            with measure('rpc.journal'):
                self.journal.append_result(envelope)
        finally:
            timing['journal_seconds'] = time.perf_counter()-beginning
            timing['journal_interval'] = (beginning,beginning+timing['journal_seconds'])
        beginning = time.perf_counter()
        try:
            with measure('rpc.ack'):
                self._transport('ack', backend_session_id=self.session_id, through_seq=seq)
        finally:
            timing['ack_seconds'] = time.perf_counter()-beginning
        self.sequence = seq
        if not envelope['ok']:
            raise RemoteError(envelope['error'])
        result = envelope['result']
        if method == 'reset_episode':
            self.episode_id = result['episode_id']
        if method == 'advance':
            from gem.runtime.trajectory_blocks import expand_feedback
            with measure('rpc.trace_block_decode'):
                result = expand_feedback(result)
            result = {**result, 'backend_session_id': self.session_id, 'mutation_seq': seq}
        return result
