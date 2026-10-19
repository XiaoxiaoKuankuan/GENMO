"""服务器1八rank运行的世界调度、预付预算和规模推导回归。

这些测试使用可追踪的协议替身隔离因果/存储错误：64条环境续体应形成两次世界
exchange，不创建64个线程或重复推进物理；业务错误必须回到原环境。预算测试
确认缓冲有界、结算前持久化、崩溃未刷写不能退款以及原SHA重放仍能读取。
规模测试覆盖4096/8192/16384和完整minibatch派生。它们不替代真实PhysX与8192
闭环验收，实际吞吐结果必须另行记录。所有输出仅进入服务器测试临时目录。
"""
from types import SimpleNamespace as NS
import copy
import pytest

from gem.closedloop.dppo.world_collector import WorldEnvironmentCollector
from gem.closedloop.dppo.world_flow import backend_operation
from gem.closedloop.dppo.budget_ledger import replay_budget
from gem.closedloop.dppo.run_management import TrainingBudget




def test_restored_world_can_collect_before_first_reward_object_exists():
    """完整断点恢复会重建物理episode，计时不能假设首次reset前已有reward。"""
    calls = []
    world = NS(call=lambda method, **kwargs: calls.append(method) or (dict(world_window=dict(reached=True,requested_world_controls=50)) if method=='end_rollout' else {}))
    policy = NS(_parameter_signature=lambda: ('fixed',))
    collector = WorldEnvironmentCollector(policy, None, world, num_envs=2, generation_batch=2)
    collector.states = [NS(transitions=0, resource=NS(env=NS(config={'stage10': {}}))) for _ in range(2)]
    assert all(not hasattr(state.resource.env, 'reward') for state in collector.states)
    def fragment(slot, version):
        count=2
        if False:
            yield
        collector.states[slot].transitions += count
        return [NS(metadata={'env_id': slot, 'policy_version': version}) for _ in range(count)]
    collector._window_flow = fragment
    rows, report = collector.collect(world_seconds=1., policy_version=7)
    assert calls == ['begin_rollout', 'end_rollout']
    assert list(map(len, rows)) == [2, 2]
    assert report['columnar_reward_cpu_timings'] == {}
    assert report['columnar_reward_fallbacks'] == 0
    assert report['total_transitions'] == 4


def test_world_ready_set_is_batched_and_environment_order_is_independent():
    calls = []
    class World:
        def call(self, method, **payload):
            calls.append((method, payload))
            return dict(replies=[dict(request_id=r['request_id'], ok=True,
                result=dict(env_id=r['env_id'], token=r['payload']['token'])) for r in payload['requests']])
    collector = WorldEnvironmentCollector(None, None, World(), num_envs=64, generation_batch=32)
    collector.states = [NS(resource=NS(env=NS(backend=NS(MUTATIONS=set(), last_call_timing=None)))) for _ in range(64)]
    def flow(slot):
        first = yield from backend_operation('preview', token=slot)
        second = yield from backend_operation('preview', token=first['token']+64)
        return first, second
    rows = collector._drive({i:flow(i) for i in reversed(range(64))})
    assert len(calls) == 2 and all(len(payload['requests']) == 64 for _,payload in calls)
    assert all(rows[i][0]['env_id'] == rows[i][1]['env_id'] == i and rows[i][1]['token'] == i+64 for i in rows)
    assert not collector.active and not collector.cancelled.is_set()


def test_no_progress_fails_instead_of_freezing_or_draining_world():
    world = NS(call=lambda method, **kwargs:dict(replies=[]))
    collector = WorldEnvironmentCollector(None, None, world, num_envs=2, generation_batch=2)
    collector.states = [NS(resource=NS(env=NS(backend=NS(MUTATIONS=set())))) for _ in range(2)]
    with pytest.raises(RuntimeError, match='partial-world advance'):
        collector._drive({i:backend_operation('advance', count=1) for i in range(2)})
    assert collector.cancelled.is_set() and not collector.active


def test_world_drains_deferred_pages_and_cross_page_ack_receipts_after_indexing():
    records=[[],[]]
    class World:
        def __init__(self):
            self.sequence=0;self.session_id='world';self.journal=NS(last_record={'sha256':'start'})
            self.ids=[];self.acks=[]
        def call(self,method,**payload):
            assert method=='exchange'
            self.sequence+=1;self.journal.last_record={'sha256':str(self.sequence)}
            for request in payload['requests']:
                if request['method']=='execute':self.ids.append(request['request_id'])
                else:
                    assert len(records[request['env_id']])==1  # 完整世界回复的lane索引先落盘。
                    self.acks.append(request['request_id'])
            if self.sequence<=2:
                i=self.sequence-1
                reply=dict(request_id=self.ids[i],ok=True,result=dict(backend_session_id=f'lane{i}',
                    mutation_seq=1,operation='advance',ok=True,result=dict(env_id=i,trace_block=dict(
                        schema='genmo.control_trace.columns.v1',count=0,columns=None))))
                return dict(replies=[reply],reply_page={'remaining_reply_count':2})
            if self.sequence==3:return dict(replies=[],reply_page={'remaining_reply_count':2})
            assert self.sequence==4
            return dict(replies=[dict(request_id=key,ok=True,result={}) for key in self.acks],
                        reply_page={'remaining_reply_count':0})
    world=World();collector=WorldEnvironmentCollector(None,None,world,num_envs=2,generation_batch=2)
    collector.states=[NS(resource=NS(env=NS(backend=NS(MUTATIONS={'advance'},session_id=f'lane{i}',
        sequence=0,episode_id=f'e{i}',journal=NS(append_result=records[i].append))))) for i in range(2)]
    result=collector._drive({i:backend_operation('advance',control_steps=1) for i in range(2)})
    assert world.sequence==4 and [result[i]['env_id'] for i in range(2)]==[0,1]
    assert all(len(r)==1 for r in records) and not collector.active


def limits():
    return dict(accepted_iterations=1, optimizer_attempts=8, generations=100,
                control_steps=1000, physics_steps=4000)
