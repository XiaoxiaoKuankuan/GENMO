"""训练侧带确认水位的冻结 GMT 客户端。

每个有副作用的操作都带 worker session 和单调 mutation_seq；回复先交给持久化 journal，
再确认回收。断链只重发同一个已编号请求，绝不换编号重做物理动作。业务错误同样保存
并确认，执行失败的完整 trace 留给环境适配器判断有效性。Stage9 必须协商 ack.v2，
不允许静默退回第八步只按 advance_id 缓存的接口。
"""
from __future__ import annotations

from gem.runtime.closedloop_protocol import RemoteError, RpcClient


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
        if hello['executed_seq'] != hello['acked_seq']:
            raise RuntimeError('Cannot adopt a worker with an unknown outstanding mutation')
        self.sequence = int(hello['executed_seq'])
        self.episode_id = None
        self.last_envelope = None

    def _transport(self, method, **payload):
        try:
            return self.client.call(method, **payload)
        except (OSError, EOFError):
            if self.socket_path is None:
                raise
            self.client.close()
            self.client = RpcClient(self.socket_path, timeout_s=self.timeout_s)
            hello = self.client.call('hello')
            if hello.get('backend_session_id') != self.session_id:
                raise RuntimeError('Worker session changed during uncertain execution')
            # 传输层sequence重新开始，业务mutation_seq保持不变。
            return self.client.call(method, **payload)

    def call(self, method, **payload):
        if method not in self.MUTATIONS:
            return self._transport(method, **payload)
        seq = self.sequence + 1
        expected = payload.get('expected_episode_id', self.episode_id)
        envelope = self._transport('execute', backend_session_id=self.session_id,
            mutation_seq=seq, expected_episode_id=expected, operation=method, payload=payload)
        if (envelope['backend_session_id'] != self.session_id or envelope['mutation_seq'] != seq
                or envelope['operation'] != method):
            raise RuntimeError('Mutation response identity mismatch')
        self.last_envelope = envelope
        self.journal.append_result(envelope)
        self._transport('ack', backend_session_id=self.session_id, through_seq=seq)
        self.sequence = seq
        if not envelope['ok']:
            raise RemoteError(envelope['error'])
        result = envelope['result']
        if method == 'reset_episode':
            self.episode_id = result['episode_id']
        if method == 'advance':
            result = {**result, 'backend_session_id': self.session_id, 'mutation_seq': seq}
        return result
