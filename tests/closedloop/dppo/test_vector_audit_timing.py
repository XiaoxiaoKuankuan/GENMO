"""验证世界和环境两层journal耗时不进入部署关键路径，ACK顺序保持不变。

使用有明确墙钟延时的RPC/journal替身，模拟世界execute和ACK各自必须完成的
审计持久化。改变审计延时只能增加总墙钟，不得等量增加critical_seconds；
环境journal必须仍在环境ACK之前。测试在服务器1执行，不启动本地环境。
"""
import time
from gem.closedloop.dppo.vector_collector import VectorLaneBackend


class Client:
    def __init__(self,delay,events):self.delay=delay;self.events=events;self.last_call_timing={}
    def call(self,method,**payload):
        if method=='hello':return dict(execution_protocols=['ack.v2'],backend_session_id='world-lane',executed_seq=0,acked_seq=0)
        self.events.append(method)
        start=time.perf_counter();time.sleep(self.delay)
        self.last_call_timing=dict(nested_journal_seconds=time.perf_counter()-start)
        if method=='ack':return {}
        return dict(backend_session_id='world-lane',mutation_seq=payload['mutation_seq'],operation=payload['operation'],
            ok=True,result=dict(reserved=True),error=None)


class Journal:
    def __init__(self,events):self.events=events
    def append_result(self,result):self.events.append('durable');time.sleep(.005)


def test_nested_journal_delay_excluded_without_early_ack():
    records=[]
    for delay in (.001,.04):
        events=[];backend=VectorLaneBackend(Client(delay,events),Journal(events))
        assert backend.call('reserve_prefix',request={})==dict(reserved=True)
        assert events==['execute','durable','ack']
        records.append(backend.last_call_timing)
    assert records[1]['total_seconds']-records[0]['total_seconds']>.07
    assert abs(records[1]['critical_seconds']-records[0]['critical_seconds'])<.02
    for r in records:
        assert r['nested_world_journal_seconds']>0
        assert r['journal_seconds']>r['nested_world_journal_seconds']
        assert r['critical_seconds']==max(0.,r['total_seconds']-r['journal_seconds'])
