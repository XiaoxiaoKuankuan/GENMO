"""第九步预算与确认客户端的CPU故障路径测试。

不启动Isaac或GPU；验证副作用预占不会因重启清零、部分推进不按完整步退款，
以及持久化失败时不ACK、业务异常先记录再确认、session能力不可降级的关键约束。
"""
import pytest

from gem.closedloop.dppo.budget import RunBudget, BudgetExceeded
from gem.closedloop.dppo.rpc import AcknowledgedBackend
from gem.runtime.closedloop_protocol import RemoteError


def test_budget_restart_and_unknown_physics(tmp_path):
    path=tmp_path/'budget.json'
    b=RunBudget(path,generations=2,control_steps=10,iterations=1)
    b.reserve('main',generations=1,control_steps=5,physics_steps=20)
    b.settle_control('main',5,dict(physics_count_exact=False,executed_control_steps=2,executed_physics_steps=None))
    b=RunBudget(path,generations=2,control_steps=10,iterations=1)
    assert b.state['used']['control_steps']==5
    b.reserve('main',generations=1)
    with pytest.raises(BudgetExceeded):
        b.reserve('main',generations=1)
    b.settle_control('main',5,dict(physics_count_exact=True,executed_control_steps=2,executed_physics_steps=10))
    assert b.state['used']['control_steps']==2
    assert b.state['used']['physics_steps']==10


class Client:
    def __init__(self):
        self.calls=[]
        self.ok=True
    def call(self,method,**payload):
        self.calls.append((method,payload))
        if method=='hello':
            return dict(execution_protocols=['ack.v2'],backend_session_id='s',executed_seq=0,acked_seq=0)
        if method=='execute':
            return dict(backend_session_id='s',mutation_seq=payload['mutation_seq'],operation=payload['operation'],
                ok=self.ok,result={'episode_id':'e'},error={'code':'invalid_qpos','message':'bad reference'})
        return {}


class Journal:
    def __init__(self,fail=False):
        self.rows=[];self.fail=fail
    def append_result(self,value):
        if self.fail:
            raise OSError('disk full')
        self.rows.append(value)


def test_no_ack_when_record_not_durable():
    client=Client();backend=AcknowledgedBackend(client,Journal(fail=True))
    with pytest.raises(OSError):
        backend.call('reset_episode',seed=42)
    assert [m for m,_ in client.calls]==['hello','execute']


def test_business_error_recorded_and_acknowledged_once():
    client=Client();journal=Journal();backend=AcknowledgedBackend(client,journal)
    client.ok=False
    with pytest.raises(RemoteError):
        backend.call('prepare_plan',generated_plan={})
    assert len(journal.rows)==1
    assert client.calls[-1]==('ack',{'backend_session_id':'s','through_seq':1})
    assert backend.sequence==1


def test_no_silent_legacy_fallback():
    class Legacy:
        def call(self,*args,**kwargs):
            return {}
    with pytest.raises(RuntimeError,match='ack.v2'):
        AcknowledgedBackend(Legacy(),Journal())
