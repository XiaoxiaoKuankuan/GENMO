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
from gem.closedloop.dppo.prepaid_budget import PrepaidBudget
from gem.closedloop.dppo.budget_ledger import replay_budget
from gem.closedloop.dppo.run_management import TrainingBudget
from gem.closedloop.dppo.training_scale import derive_training_scale, budget_requirements


@pytest.mark.parametrize('collector_kind', ['world', 'threaded'])
def test_boundary_wait_extends_control_coverage_without_inventing_decision(tmp_path, monkeypatch, collector_kind):
    """真实采样器统计应是1条决策、30个控制步；边界等待5步不能记成第2条决策。"""
    import threading
    from gem.closedloop.dppo.full_dataset import FullMusicCatalog, FullMusicSampler
    from gem.closedloop.dppo.vector_collector import VectorEnvironmentCollector
    from gem.closedloop.dppo import vector_boundary
    from tests.closedloop.dppo.test_full_dataset import make_catalog_data

    sampler = FullMusicSampler(FullMusicCatalog(make_catalog_data(tmp_path)), random_start=False)
    task = sampler.next_task()
    row = NS(transition_valid=True, identity=dict(policy_version=7), executed_control_steps=25,
             metadata={}, terminated=False, truncated=False, reason=None)
    env = NS(backend=NS(call=lambda *a, **kw: None), step=lambda: row)
    def step_flow():
        if False: yield
        return row
    env.step_flow = step_flow
    def finish(env, last, **kwargs):
        last.executed_control_steps += 5
        last.metadata['fragment_tail'] = {}
        return 5, False
    def finish_flow(*args, **kwargs):
        if False: yield
        return finish(*args, **kwargs)
    monkeypatch.setattr(vector_boundary, 'finish_vector_fragment', finish)
    monkeypatch.setattr(vector_boundary, 'finish_vector_fragment_flow', finish_flow)
    cls = WorldEnvironmentCollector if collector_kind == 'world' else VectorEnvironmentCollector
    collector = object.__new__(cls)
    collector.states = [NS(resource=NS(env=env, sampler=sampler), task=task, transitions=0)]
    collector.cancelled = threading.Event()
    collector.boundary_wait_contract = 'bounded_reference_wait.v1'
    collector.boundary_wait_max_controls = 100
    if collector_kind == 'world':
        flow = collector._fragment_flow(0, 1, 7)
        next(flow)  # 真实join_fragment协议边界；此测试不执行物理。
        with pytest.raises(StopIteration) as end: flow.send({})
        rows = end.value.value
    else:
        rows = collector._fragment_profiled(0, 1, 7)
    coverage = sampler.coverage()['per_source'][task['sample']['dataset']]
    assert coverage['upper_transitions'] == len(rows) == collector.states[0].transitions == 1
    assert coverage['control_steps'] == rows[0].executed_control_steps == 30
    assert sampler.state_dict()['active']['control_steps'] == 30


def test_restored_world_can_collect_before_first_reward_object_exists():
    """完整断点恢复会重建物理episode，计时不能假设首次reset前已有reward。"""
    calls = []
    world = NS(call=lambda method, **kwargs: calls.append(method) or {})
    policy = NS(_parameter_signature=lambda: ('fixed',))
    collector = WorldEnvironmentCollector(policy, None, world, num_envs=2, generation_batch=2)
    collector.states = [NS(transitions=0, resource=NS(env=NS(config={'stage10': {}}))) for _ in range(2)]
    assert all(not hasattr(state.resource.env, 'reward') for state in collector.states)
    def fragment(slot, count, version):
        if False:
            yield
        collector.states[slot].transitions += count
        return [NS(metadata={'env_id': slot, 'policy_version': version}) for _ in range(count)]
    collector._fragment_flow = fragment
    rows, report = collector.collect(count_per_rank=4, policy_version=7)
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
    with pytest.raises(RuntimeError, match='no hidden drain'):
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


def test_prepaid_events_have_bounded_writeback_and_exact_legacy_replay(tmp_path):
    root = TrainingBudget(tmp_path/'resource_lease.json', limits())
    child = PrepaidBudget(tmp_path/'budget.json', limits(), parent_lease=root.path, maximum_pending=3)
    for _ in range(7):child.reserve('collect', generations=1)
    assert child.flush_count == 2 and len(child.pending) == 1 and child.peak_pending == 3
    state = child.state_dict()
    assert not child.pending and child.flush_count == 3
    actual, sequence, head = replay_budget(child.path)
    assert actual['used'] == state['used'] and sequence == 7 and head == state['ledger']['sha256']
    child.close(); child.close()


def test_unflushed_crash_does_not_refund_parent_reservation(tmp_path):
    root = TrainingBudget(tmp_path/'root.json', limits()); root.reserve('rank0', generations=30)
    lease = TrainingBudget(tmp_path/'resource_lease.json', dict(limits(), generations=30))
    child = PrepaidBudget(tmp_path/'budget.json', dict(limits(), generations=30), parent_lease=lease.path)
    child.reserve('collect', generations=1)
    child.connection.close()  # 模拟未刷写进程退出，故意不调用close/结算。
    assert replay_budget(child.path)[0]['used']['generations'] == 0
    assert TrainingBudget(root.path, limits()).state_dict()['used']['generations'] == 30


@pytest.mark.parametrize('decisions,steps', [(2,2),(4,2),(8,4),(16,8),(32,16)])
def test_production_scale_derived_without_old_160_and_four_step_limits(decisions, steps):
    config = dict(runtime=dict(num_envs=64), stage10=dict(distributed=dict(world_size=8),
        scale=dict(decisions_per_environment=decisions, actor_minibatch_chains=2048, critic_epochs=4),
        training=dict(denoising_steps=20, ppo_epochs=2, critic_batch=1024, bc_batch=128)))
    derive_training_scale(config); train = config['stage10']['training']
    assert train['rollout_upper_steps'] == 512*decisions
    assert train['max_actor_optimizer_steps'] == steps
    assert train['critic_steps'] == 4*512*decisions//1024
    broken = copy.deepcopy(config); broken['stage10']['training']['rollout_upper_steps'] = 160
    with pytest.raises(ValueError, match='Conflicting'):derive_training_scale(broken)
