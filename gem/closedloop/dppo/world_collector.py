"""单线程世界级批量闭环调度器，替代每环境线程和独立RPC等待循环。

所有活动环境共享一个因果状态机和权威world journal。各环境续体只保存自己的
音乐、参考、episode、历史及奖励状态；调度器集中提交reserve/prepare/commit与
advance，收到完整世界回复并可靠保存后批量ACK，才恢复对应续体。GENMO请求按
独立配置合批，批内全部来自不同真实环境，20步链及每环境RNG身份不变。

正常rollout结束只封存本轮证据，不reset/retire健康机器人，不推进额外drain。
本模块不伪造物理恢复：断点恢复仍创建新PhysX世界和新episode，保留预算消耗与
采样游标。若严格每环境配额造成未完成的异步物理请求，明确报错并保留证据，绝不
静默冻结部分机器人、扔掉实际执行或把额外长尾算作吞吐提升。
"""
from __future__ import annotations
import copy
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

from .world_flow import BackendOperation, GenerationOperation, LocalOperation, backend_operation
from .vector_collector import VectorEnvironmentCollector, validate_vector_close
from .dual_collector import _QueuedPolicy
from .vector_generation import generate_batch
from gem.runtime.closedloop_protocol import RemoteError

WORLD_FLOW_CONTRACT = 'genmo.world_batched_flow.v1'


class _ImmediateExecutor:
    """兼容既有边界状态捕获接口；所有资源实际都由当前唯一调度线程持有。"""
    def submit(self, function, *args, **kwargs):
        future = Future()
        try: future.set_result(function(*args, **kwargs))
        except BaseException as error: future.set_exception(error)
        return future
    def shutdown(self, **kwargs): pass


class _DirectLane:
    def __init__(self, owner, slot): self.owner,self.slot=owner,slot;self.last_call_timing={}
    def call(self, method, **payload):
        if self.owner.active: raise RuntimeError('Active world operations must use batched continuations')
        identifier = self.owner._request_id()
        result = self.owner.world_call('exchange', requests=[dict(env_id=self.slot,method=method,payload=payload,request_id=identifier)])
        if len(result['replies']) != 1: raise RuntimeError('Boundary lane call cannot create deferred physics')
        reply = result['replies'][0]
        if not reply['ok']: raise RemoteError(reply['error'])
        return reply['result']
    def close(self): pass


class WorldEnvironmentCollector(VectorEnvironmentCollector):
    def __init__(self, policy, factory, world_client, *, num_envs=64, generation_batch=64, timeout_seconds=600., **kwargs):
        if generation_batch < 1 or generation_batch > num_envs: raise ValueError('Invalid independent sampling batch')
        self.policy,self.factory=policy,factory
        self.world=world_client() if callable(world_client) else world_client
        self.num_envs,self.generation_batch,self.timeout_seconds=num_envs,generation_batch,timeout_seconds
        self.states=[None]*num_envs;self.executors=[_ImmediateExecutor() for _ in range(num_envs)]
        self.active=False;self.cancelled=threading.Event();self.rpc_pending={};self.rpc_sequence=0
        self.batch_reports=[];self.world_timing={};self.restore_records=None;self.shutdown_result=None
        self.rpc_thread=None;self.rpc_error=None
        self.audit_seconds=0.

    def _request_id(self):
        self.rpc_sequence+=1
        return f'worldflow:{self.rpc_sequence}'

    def lane_client(self, slot): return _DirectLane(self,slot)
    def world_call(self, method, **payload):
        started=time.perf_counter()
        result=self.world.call(method,**payload)
        self.audit_seconds += (getattr(self.world, 'last_call_timing', None) or {}).get('journal_seconds', 0.)
        self.world_timing['world_rpc_seconds']=self.world_timing.get('world_rpc_seconds',0.)+time.perf_counter()-started
        self.world_timing['world_rpc_calls']=self.world_timing.get('world_rpc_calls',0)+1
        for name,value in result.get('timing',{}).items():self.world_timing[name]=self.world_timing.get(name,0.)+value
        if result.get('fatal'): raise RuntimeError(f'World execution failed after durable evidence: {result["fatal"]}')
        return result
    def world_idle(self,function):
        if self.active or self.rpc_pending: raise RuntimeError('World boundary is not idle')
        return function(self.world)

    def _resource(self, slot):
        if self.states[slot] is None:
            resource=self.factory(slot,_QueuedPolicy(self,slot))
            resource.env.collector_env_slot=slot
            self.states[slot]=SimpleNamespace(resource=resource,task=None,transitions=0)
        return self.states[slot]

    def _drive(self, flows):
        """集中推进所有续体；一个exchange包含整个就绪集合，物理永不逐环境调用。"""
        if self.active: raise RuntimeError('Nested world flow')
        self.active=True
        ready={slot:(None,None) for slot in flows};results={};waiting={};started=time.perf_counter()
        try:
            while len(results)<len(flows):
                commands=[];generations=[]
                for slot,(value,error) in ready.items():
                    continuation_started=time.perf_counter()
                    try: operation=flows[slot].throw(error) if error is not None else flows[slot].send(value)
                    except StopIteration as finished:
                        results[slot]=finished.value;continue
                    finally:
                        self.world_timing['continuation_seconds']=self.world_timing.get('continuation_seconds',0.)+time.perf_counter()-continuation_started
                    if isinstance(operation,GenerationOperation):generations.append((slot,operation.packet))
                    elif isinstance(operation,BackendOperation):commands.append((slot,operation))
                    elif isinstance(operation,LocalOperation):
                        raise RuntimeError('World training cannot hide synchronous per-environment generation')
                    else:raise TypeError('Unknown world continuation message')
                ready={}
                if time.perf_counter()-started>self.timeout_seconds:raise TimeoutError('Bounded world collection deadline')
                # 生成与控制事件由同一调度器组织，不启动N个actor/物理事务线程。
                for begin in range(0,len(generations),self.generation_batch):
                    group=generations[begin:begin+self.generation_batch]
                    generation_started=time.perf_counter()
                    generated,timing=generate_batch(self.policy,[packet for _,packet in group],shared_storage=True)
                    owner=self.states[group[0][0]].resource.env
                    block=generated[0]['_shared_raw_block'][0]
                    block['policy_version']=owner.policy_version
                    raw_path=owner.output.parent/'raw_samples'/f'world_batch_{len(self.batch_reports):06d}.pt'
                    raw_started=time.perf_counter()
                    identity=owner._save_evidence(block,raw_path)
                    raw_seconds=time.perf_counter()-raw_started
                    self.world_timing['raw_block_seconds']=self.world_timing.get('raw_block_seconds',0.)+raw_seconds
                    self.audit_seconds += raw_seconds+timing['bulk_trace_transfer_seconds']
                    for item in generated:
                        _,index=item.pop('_shared_raw_block')
                        item['_raw_evidence']=dict(path=str(raw_path),identity=dict(identity,
                            storage='genmo.world_batch_raw.v3',index=index,count=len(group)))
                    self.batch_reports.append(dict(environment_slots=[slot for slot,_ in group],
                        effective_rows=len(group),padding_rows=0,pipeline_timing=timing,
                        generation_seconds=timing['condition_batch_seconds']+timing['generate_and_world_seconds'],
                        wall_and_audit_seconds=time.perf_counter()-generation_started,
                        components=dict(self.policy.last_sample_timing)))
                    for (slot,_),value in zip(group,generated):ready[slot]=((value,timing),None)
                requests=[]
                for slot,operation in commands:
                    backend=self.states[slot].resource.env.backend
                    identity=self._request_id();mutation=operation.method in backend.MUTATIONS
                    payload=(dict(backend_session_id=backend.session_id,mutation_seq=backend.sequence+1,
                        expected_episode_id=operation.payload.get('expected_episode_id',backend.episode_id),
                        operation=operation.method,payload=operation.payload) if mutation else operation.payload)
                    waiting[identity]=(slot,operation,mutation,time.perf_counter())
                    requests.append(dict(env_id=slot,method='execute' if mutation else operation.method,
                                         payload=payload,request_id=identity))
                if requests or waiting:
                    # 空exchange只在确有未完成物理请求时驱动，不做空轮询写盘。
                    replies=self.world_call('exchange',requests=requests)['replies']
                    while replies:
                        acknowledgements=[]
                        for reply in replies:
                            slot,operation,mutation,begin=waiting.pop(reply['request_id'])
                            backend=self.states[slot].resource.env.backend
                            if not reply['ok']:
                                ready[slot]=(None,RemoteError(reply['error']));continue
                            envelope=reply['result'];value=envelope;error=None
                            if mutation:
                                if (envelope['backend_session_id']!=backend.session_id or
                                        envelope['mutation_seq']!=backend.sequence+1 or envelope['operation']!=operation.method):
                                    raise RuntimeError('World/lane reply identity mismatch')
                                # 第一版索引仍同步落盘。完整数组仅存在权威世界journal一次。
                                reference=dict(backend_session_id=backend.session_id,mutation_seq=envelope['mutation_seq'],
                                    operation=operation.method,ok=envelope['ok'],world_reply_reference=dict(
                                        schema='genmo.world_reply_reference.v1',journal='../world_journal.sqlite',
                                        sha256=self.world.journal.last_record['sha256'],
                                        backend_session_id=self.world.session_id,mutation_seq=self.world.sequence,
                                        request_id=reply['request_id']))
                                indexed=time.perf_counter()
                                backend.journal.append_result(reference)
                                index_seconds=time.perf_counter()-indexed
                                self.world_timing['lane_index_seconds']=self.world_timing.get('lane_index_seconds',0.)+index_seconds
                                self.audit_seconds += index_seconds
                                backend.last_envelope=envelope;backend.sequence=envelope['mutation_seq']
                                if not envelope['ok']:error=RemoteError(envelope['error']);value=None
                                else:value=envelope['result']
                                acknowledgements.append(dict(env_id=slot,method='ack',payload=dict(
                                    backend_session_id=backend.session_id,through_seq=backend.sequence),request_id=self._request_id()))
                                if operation.method=='reset_episode' and error is None:backend.episode_id=value['episode_id']
                                if operation.method=='advance' and error is None:
                                    from gem.runtime.trajectory_blocks import expand_feedback
                                    value={**expand_feedback(value, readonly_views=True), 'backend_session_id':backend.session_id,'mutation_seq':backend.sequence}
                            backend.last_call_timing=dict(method=operation.method,total_seconds=time.perf_counter()-begin,
                                journal_seconds=0.,critical_seconds=time.perf_counter()-begin,
                                audit_scope='authoritative_world_journal_plus_reference_index')
                            ready[slot]=(value,error)
                        if acknowledgements:
                            received=self.world_call('exchange',requests=acknowledgements)['replies']
                            ack_ids={entry['request_id'] for entry in acknowledgements}
                            for reply in received:
                                if reply['request_id'] in ack_ids and not reply['ok']:raise RemoteError(reply['error'])
                            replies=[entry for entry in received if entry['request_id'] not in ack_ids]
                        else:replies=[]
                if not ready and waiting:
                    raise RuntimeError('Exact environment quotas reached an asynchronous physical boundary; no hidden drain or partial-world freeze is allowed')
                if not ready and not waiting and len(results)<len(flows):raise RuntimeError('World state machine made no progress')
            if waiting:raise RuntimeError('World finished with pending physical replies')
            return results
        except BaseException:
            self.cancelled.set()
            raise
        finally:self.active=False

    def _fragment_flow(self,slot,count,policy_version):
        state=self.states[slot];env,sampler=state.resource.env,state.resource.sampler
        env.policy_version=policy_version
        yield from backend_operation('join_fragment')
        rows=[]
        for _ in range(count):
            if state.task is None:
                task=sampler.next_task()
                if task['sample']['row']['split']!='train':raise ValueError('Validation sample in training world')
                yield from env.reset_task_flow(task['sample'],task['music'],seed=env.config['stage9']['seed']+env.episode_count,
                    phase='train',music_start_frame=task['music_start_frame'])
                state.task=task
            row=yield from env.step_flow()
            if not row.transition_valid or row.identity['policy_version']!=policy_version:raise ValueError('Invalid/mixed behavior policy')
            task=state.task
            row.metadata.update(collector_env_slot=slot,training_task=dict(dataset=task['sample']['dataset'],
                sample_id=task['sample']['row']['sample_id'],split='train',music_start_frame=task['music_start_frame'],
                manifest_sha256=task['sample']['manifest_sha256']))
            sampler.record_execution(task,row.executed_control_steps);rows.append(row);state.transitions+=1
            if row.terminated or row.truncated:state.task=None
        if not rows[-1].terminated:
            rows[-1].truncated=True;rows[-1].reason=rows[-1].reason or 'collector_fragment_boundary'
        return rows

    def collect(self, *,count_per_rank,policy_version):
        if count_per_rank%self.num_envs:raise ValueError('World quota must assign equal complete decision counts')
        for slot in range(self.num_envs):self._resource(slot)
        self.world_call('begin_rollout',env_ids=list(range(self.num_envs)))
        signature=self.policy._parameter_signature();self.batch_reports=[];self.world_timing={};started=time.perf_counter()
        before = [state.transitions for state in self.states]
        try:
            fragments=self._drive({slot:self._fragment_flow(slot,count_per_rank//self.num_envs,policy_version) for slot in range(self.num_envs)})
        except BaseException as error:
            from .budget import atomic_json
            atomic_json(self.states[0].resource.env.output.parent/'failed_world_progress.json', dict(
                status='failed', error_type=type(error).__name__, error=str(error), policy_version=policy_version,
                requested_transitions=count_per_rank, completed_per_environment=[
                    state.transitions-value for state,value in zip(self.states,before)],
                seconds=time.perf_counter()-started, batches=self.batch_reports, world_timing=self.world_timing,
                audit_seconds=self.audit_seconds, partial_execution_is_not_accepted_rollout=True))
            raise
        self.world_call('end_rollout',env_ids=list(range(self.num_envs)))
        if signature!=self.policy._parameter_signature():raise RuntimeError('Policy changed during collection')
        rows=[fragments[i] for i in range(self.num_envs)]
        return rows,dict(schema=WORLD_FLOW_CONTRACT,fragment_contract=WORLD_FLOW_CONTRACT,
            total_transitions=sum(map(len,rows)),fragment_lengths=list(map(len,rows)),allocated_envs=self.num_envs,
            active_envs=self.num_envs,seconds=time.perf_counter()-started,batches=self.batch_reports,
            world_timing=self.world_timing,normal_boundary_resets=0,administrative_drain_controls=0,
            training_reward_seconds=sum(row.metadata.get('training_reward_seconds',0.) for fragment in rows for row in fragment),
            real_environment_batch=True,gae_contract='independent_per_env_episode_contiguous_fragment')

    def calibrate(self,*,count_per_rank,warmup=1,samples=2):
        for slot in range(self.num_envs):self._resource(slot)
        def initialize(slot):
            state=self.states[slot];env=state.resource.env;sampler=state.resource.sampler
            saved=copy.deepcopy(sampler.state_dict());task=sampler.next_task();sampler.load_state_dict(saved)
            yield from env.reset_task_flow(task['sample'],task['music'],seed=env.config['stage9']['seed'],
                phase='calibration',music_start_frame=task['music_start_frame'])
        self._drive({i:initialize(i) for i in range(self.num_envs)})
        durations=[[] for _ in self.states]
        def generate(slot):
            env=self.states[slot].resource.env
            sample=yield from env.generate_flow()
            if sample['prepared'] is None or sample['rejection']:raise RuntimeError('World calibration reference rejected')
            yield from backend_operation('discard_plan',prepared_plan_id=sample['prepared']['prepared_plan_id'])
            env.decision+=1
            return sample['critical_ready_seconds']
        for index in range(warmup+samples):
            records=self._drive({i:generate(i) for i in range(self.num_envs)})
            if index>=warmup:
                for slot,value in records.items():durations[slot].append(value)
        self._drive({i:backend_operation('retire') for i in range(self.num_envs)})
        clock=self.states[0].resource.env.deployment_clock
        return dict(durations=durations,latency_budget_s=clock.budget_seconds,batches=self.batch_reports,
            deployment_profile_sha256=clock.sha256,scope='wallclock_only_fixed_modeled_deployment_v3')

    def close(self):
        self.cancelled.set()
        for state in self.states:
            if state is not None:state.resource.close()
        try:
            self.shutdown_result=self.world_call('close');validate_vector_close(self.shutdown_result)
        finally:self.world.journal.close();self.world.client.close()
        return self.shutdown_result

    def state_dict(self):
        result=super().state_dict()
        result.update(schema='genmo.gpu_vector_collector.boundary.v2',fragment_contract=WORLD_FLOW_CONTRACT)
        return result

    def load_state_dict(self,saved):
        if saved.get('schema')!='genmo.gpu_vector_collector.boundary.v2' or saved.get('fragment_contract')!=WORLD_FLOW_CONTRACT:
            raise ValueError('World collector cannot transparently resume the old threaded collector')
        # 共用已验收的逻辑RNG/预算恢复，输入版本先严格核验；物理仍明确重建。
        adapted=copy.deepcopy(saved)
        from .vector_boundary import FRAGMENT_CONTRACT
        adapted.update(schema='genmo.gpu_vector_collector.boundary.v1',fragment_contract=FRAGMENT_CONTRACT)
        return super().load_state_dict(adapted)
