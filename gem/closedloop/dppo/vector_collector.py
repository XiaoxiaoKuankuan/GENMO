"""同进程统一PhysX世界的已确认客户端与冻结关闭校验。

世界/环境请求仍核对序号、episode、实际执行及ACK；关闭核验GMT参数身份和所有
执行回执。单线程续体调度只在world_collector实现，本模块不再创建每环境线程
或独立RPC泵，也不保留旧配额采集器及其checkpoint适配。
"""
from __future__ import annotations

from .rpc import AcknowledgedBackend


class VectorWorldClient(AcknowledgedBackend):
    MUTATIONS={'exchange','begin_rollout','end_rollout'}


def validate_vector_close(result):
    """正常退出必须保留冻结参数和世界/全部lane已持久化确认的真实回执。"""
    from gem.closedloop.dppo.execution_state import _assert_frozen
    _assert_frozen(result)
    if (result.get('closed') is not True or result.get('gmt_parameters_frozen') is not True
            or result.get('actual_module_sha256')!=result.get('initial_module_sha256')
            or not result.get('actual_module_sha256') or result.get('pending_env_ids')!=[]
            or result.get('deferred_reset_env_ids')!=[] or result.get('restart_required') is not False
            or not result.get('lane_journals')):
        raise RuntimeError('Vector worker closed with incomplete execution or changed parameters')
    for journal in [result.get('execution_journal',{}),*result.get('lane_journals',[])]:
        if (not journal.get('backend_session_id') or journal.get('executed_seq')!=journal.get('acked_seq')
                or journal.get('outstanding_seq') is not None):
            raise RuntimeError('Vector worker closed before all execution was acknowledged')


class VectorLaneBackend(AcknowledgedBackend):
    MUTATIONS=AcknowledgedBackend.MUTATIONS|{'retire','activate','join_fragment'}

    def call(self,method,**payload):
        self.nested_journal_seconds=0.
        try:result=super().call(method,**payload)
        finally:
            timing=self.last_call_timing
            if timing is not None and timing.get('method')==method:
                timing['nested_world_journal_seconds']=self.nested_journal_seconds
                timing['journal_seconds']+=self.nested_journal_seconds
                timing['critical_seconds']=max(0.,timing['total_seconds']-timing['journal_seconds'])
        return result

    def _client_call(self,method,**payload):
        try:return super()._client_call(method,**payload)
        finally:
            self.nested_journal_seconds+=getattr(self.client,'last_call_timing',{}).get('nested_journal_seconds',0.)
