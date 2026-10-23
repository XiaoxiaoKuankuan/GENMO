"""单线程世界级批量闭环调度器，替代每环境线程和独立RPC等待循环。

所有活动环境共享一个因果状态机和权威world journal。各环境续体只保存自己的
音乐、参考、episode、历史及奖励状态；调度器集中提交reserve/prepare/commit与
advance，收到完整世界回复并可靠保存后批量ACK，才恢复对应续体。GENMO请求按
独立配置合批，批内全部来自不同真实环境，20步链及每环境RNG身份不变。

rollout 以共享世界的模拟时间结束，各环境有效转移数可以不同；不设每环境决策
配额，不再执行配额尾部等待。所有实际控制步由后端计数和预算账本核验，普通
窗口边界保留健康机器人、音乐和参考。恢复仍重建 PhysX，不恢复中途物理状态。
"""
from __future__ import annotations
import copy
import threading
import time
import traceback
from concurrent.futures import Future
from types import SimpleNamespace

from .world_flow import BackendOperation, GenerationOperation, LocalOperation, backend_operation
from .vector_collector import validate_vector_close
from .vector_generation import generate_batch
from .env_adapter import WarmupEpisodeEnded
from gem.runtime.closedloop_protocol import RemoteError

WORLD_FLOW_CONTRACT = 'genmo.world_batched_flow.v1'




class _WorldPolicyView:
    """环境只读取当前策略身份，采样请求必须交给同线程世界批处理。"""
    def __init__(self,owner):
        self.owner=owner;self.actor=owner.policy.actor;self.steps=owner.policy.steps
        self.guidance_scale=owner.policy.guidance_scale;self.last_sample_timing={}
    @property
    def kernel_config(self):return self.owner.policy.kernel_config
    def sample_rollout(self,*args,**kwargs):
        raise RuntimeError('World generation must use the batched continuation scheduler')


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


class WorldEnvironmentCollector:
    def __init__(self, policy, factory, world_client, *, num_envs=64, generation_batch=64, timeout_seconds=600.,
                 training_storage='durable_cpu.v1'):
        if generation_batch < 1 or generation_batch > num_envs: raise ValueError('Invalid independent sampling batch')
        self.policy,self.factory=policy,factory
        if training_storage not in ('durable_cpu.v1','ephemeral_device'):
            raise ValueError('Unknown world training-data lifecycle')
        self.training_storage=training_storage
        self.world=world_client() if callable(world_client) else world_client
        self.num_envs,self.generation_batch,self.timeout_seconds=num_envs,generation_batch,timeout_seconds
        self.states=[None]*num_envs
        self.active=False;self.cancelled=threading.Event();self.rpc_pending={};self.rpc_sequence=0
        self.batch_reports=[];self.world_timing={};self.restore_records=None;self.shutdown_result=None
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
        key='world_'+method+'_calls'
        self.world_timing[key]=self.world_timing.get(key,0)+1
        for name in ('journal_seconds','transport_seconds','ack_seconds'):
            key='world_'+name
            self.world_timing[key]=self.world_timing.get(key,0.)+(getattr(self.world,'last_call_timing',{}) or {}).get(name,0.)
        for name,seconds in (getattr(self.world,'last_call_timing',{}) or {}).get('journal_detail',{}).items():
            key='world_journal_'+name
            self.world_timing[key]=self.world_timing.get(key,0.)+seconds
        for name,value in result.get('timing',{}).items():self.world_timing[name]=self.world_timing.get(name,0.)+value
        if result.get('fatal'): raise RuntimeError(f'World execution failed after durable evidence: {result["fatal"]}')
        window=result.get('world_window')
        if window is not None:
            self.window_record=window
            if window['reached']:
                for state in self.states:
                    if state is not None:state.resource.env.rollout_window_reached=True
        return result
    def world_idle(self,function):
        if self.active or self.rpc_pending: raise RuntimeError('World boundary is not idle')
        return function(self.world)

    def _resource(self, slot):
        if self.states[slot] is None:
            resource=self.factory(slot,_WorldPolicyView(self))
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
                    generated,timing=generate_batch(self.policy,[packet for _,packet in group],shared_storage=True,
                        device_trace=self.training_storage=='ephemeral_device')
                    owner=self.states[group[0][0]].resource.env
                    block=generated[0]['_shared_raw_block'][0]
                    block['policy_version']=owner.policy_version
                    raw_path=owner.output.parent/'raw_samples'/f'world_batch_{len(self.batch_reports):06d}.pt'
                    raw_started=time.perf_counter()
                    if self.training_storage=='ephemeral_device':
                        identity=dict(storage='ephemeral_device',sha256=None,size_bytes=None,
                            memory_identity=f'{owner.config["stage9"]["run_id"]}:{owner.iteration}:{owner.attempt}:{len(self.batch_reports)}',
                            recovery='discard_incomplete_rollout_and_resample_from_complete_checkpoint')
                    else:identity=owner._save_evidence(block,raw_path)
                    raw_seconds=time.perf_counter()-raw_started
                    self.world_timing['raw_block_seconds']=self.world_timing.get('raw_block_seconds',0.)+raw_seconds
                    self.audit_seconds += raw_seconds+timing['bulk_trace_transfer_seconds']
                    for item in generated:
                        _,index=item.pop('_shared_raw_block')
                        item['_raw_evidence']=dict(path=str(raw_path) if self.training_storage=='durable_cpu.v1' else None,
                            identity=dict(identity,storage=identity.get('storage','genmo.stage2.asymmetric_world_batch.v1'),index=index,count=len(group)))
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
                    response=self.world_call('exchange',requests=requests)
                    replies=response['replies'];ack_pending=set()
                    pages_left=response.get('reply_page',{}).get('remaining_reply_count',0)
                    while replies or pages_left or ack_pending:
                        if time.perf_counter()-started>self.timeout_seconds:
                            raise TimeoutError('Bounded world reply page deadline')
                        acknowledgements=[]
                        for reply in replies:
                            if reply['request_id'] in ack_pending:
                                if not reply['ok']:raise RemoteError(reply['error'])
                                ack_pending.remove(reply['request_id']);continue
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
                                    value={**expand_feedback(value, readonly_views=True, lazy=True, columnar=True), 'backend_session_id':backend.session_id,'mutation_seq':backend.sequence}
                            backend.last_call_timing=dict(method=operation.method,total_seconds=time.perf_counter()-begin,
                                journal_seconds=0.,critical_seconds=time.perf_counter()-begin,
                                audit_scope='authoritative_world_journal_plus_reference_index')
                            ready[slot]=(value,error)
                        if acknowledgements or pages_left or ack_pending:
                            ack_pending.update(entry['request_id'] for entry in acknowledgements)
                            if (acknowledgements and not pages_left and
                                    getattr(self.world,'batched_lane_ack',None)=='durable_world_and_lane_indices.v1'):
                                response=self.world_call('ack_lanes',backend_session_id=self.world.session_id,
                                                         requests=acknowledgements)
                            else:
                                response=self.world_call('exchange',requests=acknowledgements)
                            replies=response['replies']
                            pages_left=response.get('reply_page',{}).get('remaining_reply_count',0)
                            if not replies and not pages_left and ack_pending:
                                raise RuntimeError('World reply pages lost lane ACK receipts')
                        else:replies=[]
                if not ready and waiting:
                    raise RuntimeError('World window lost a pending physical reply; partial-world advance is forbidden')
                if not ready and not waiting and len(results)<len(flows):raise RuntimeError('World state machine made no progress')
            if waiting:raise RuntimeError('World finished with pending physical replies')
            return results
        except BaseException:
            self.cancelled.set()
            raise
        finally:self.active=False


    def _window_flow(self,slot,policy_version):
        """每个环境连续产生有效决策，统一世界边界到达时一起停止。"""
        state=self.states[slot];env,sampler=state.resource.env,state.resource.sampler
        env.policy_version=policy_version
        yield from backend_operation('join_fragment')
        rows=[]
        while not env.rollout_window_reached:
            try:
                if state.task is None:
                    task=sampler.next_task()
                    if task['sample']['row']['split']!='train':raise ValueError('Validation sample in training world')
                    yield from env.reset_task_flow(task['sample'],task['music'],seed=env.config['stage9']['seed']+env.episode_count,
                        phase='train',music_start_frame=task['music_start_frame'],
                        **({'initialization': task['sampling']['initialization']} if 'sampling' in task else {}))
                    state.task=task
                    self.world_timing['completed_environment_resets']=self.world_timing.get('completed_environment_resets',0)+1
                if env.rollout_window_reached:break
                if env.snapshot['tick']<600:
                    yield from env.continue_warmup_flow()
            except WarmupEpisodeEnded:
                # 只接收已结算的正常物理终止，完整性错误仍交给上层停止运行。
                self.world_timing['warmup_physical_failures']=self.world_timing.get('warmup_physical_failures',0)+1
                state.task=None
                continue
            if env.rollout_window_reached:break
            row=yield from env.step_flow()
            if not row.transition_valid or row.identity['policy_version']!=policy_version:raise ValueError('Invalid/mixed behavior policy')
            task=state.task
            row.metadata.update(collector_env_slot=slot,training_task=dict(dataset=task['sample']['dataset'],
                sample_id=task['sample']['row']['sample_id'],split='train',music_start_frame=task['music_start_frame'],
                manifest_sha256=task['sample']['manifest_sha256'],sampling=copy.deepcopy(task.get('sampling'))))
            sampler.record_execution(task,row.executed_control_steps)
            from .rolling import rolling_enabled, store_rollout_on_host
            if rolling_enabled(env.config):store_rollout_on_host(row)
            rows.append(row);state.transitions+=1
            if row.terminated or row.truncated:state.task=None
        if rows and not rows[-1].terminated:
            # 这里只切当前Actor版本的GAE；正常物理episode、参考和音乐继续存在。
            rows[-1].truncated=True
            rows[-1].reason=rows[-1].reason or 'ppo_world_window_boundary'
        return rows

    def collect(self, *,world_seconds,policy_version):
        if world_seconds<=0 or world_seconds%0.5:raise ValueError('World duration must lie on the decision grid')
        for slot in range(self.num_envs):self._resource(slot)
        for state in self.states:state.resource.env.rollout_window_reached=False
        self.window_record=None
        self.world_call('begin_rollout',env_ids=list(range(self.num_envs)))
        signature=self.policy._parameter_signature();self.batch_reports=[];self.world_timing={};started=time.perf_counter()
        protocol_before=(self.world_call('transport_statistics') if
            getattr(self.world,'transport_statistics_contract',None)=='rpc.transport_cumulative.v1' else None)
        profile = None
        if self.states[0].resource.env.config['stage10'].get('performance',{}).get('python_world_profile',False):
            import cProfile
            self.world_timing['python_diagnostic_profiler_enabled']=True
            profile=cProfile.Profile();profile.enable()
        before = [state.transitions for state in self.states]
        from .vector_reward_adapter import capture_columnar_timings
        reward_timings={}
        try:
            with capture_columnar_timings() as reward_timings:
                fragments=self._drive({slot:self._window_flow(slot,policy_version) for slot in range(self.num_envs)})
                self.world_call('end_rollout',env_ids=list(range(self.num_envs)))
        except BaseException as error:
            from .budget import atomic_json
            atomic_json(self.states[0].resource.env.output.parent/'failed_world_progress.json', dict(
                status='failed', error_type=type(error).__name__, error=str(error), policy_version=policy_version,
                traceback=traceback.format_exc(),
                requested_world_seconds=world_seconds, completed_per_environment=[
                    state.transitions-value for state,value in zip(self.states,before)],
                seconds=time.perf_counter()-started, batches=self.batch_reports, world_timing=self.world_timing,
                audit_seconds=self.audit_seconds, columnar_reward_cpu_timings=reward_timings,
                partial_execution_is_not_accepted_rollout=True))
            raise
        finally:
            if profile is not None:
                profile.disable()
                profile.dump_stats(str(self.states[0].resource.env.output.parent/'world_python_profile.pstats'))
        if signature!=self.policy._parameter_signature():raise RuntimeError('Policy changed during collection')
        rows=[fragments[i] for i in range(self.num_envs)]
        from .rolling import generation_capacity
        capacity=generation_capacity(self.states[0].resource.env.config, world_seconds)
        if any(len(fragment)>capacity for fragment in rows):
            raise RuntimeError('Shared decision-grid capacity exceeded; cannot perform multiple on-policy PG steps')
        protocol_after=self.world_call('transport_statistics') if protocol_before is not None else {}
        protocol_delta={key:value-protocol_before.get(key,0.) for key,value in protocol_after.items()}
        if self.window_record is None or not self.window_record['reached']:
            raise RuntimeError('Shared world did not attest its real simulation window')
        if self.window_record['requested_world_controls']!=round(world_seconds*50):
            raise RuntimeError('Collector and PhysX world configured different rollout durations')
        return rows,dict(schema=WORLD_FLOW_CONTRACT,fragment_contract=WORLD_FLOW_CONTRACT,
            collection_contract='shared_world_sim_time.v1',world_window=self.window_record,
            quota_alignment_removed=True,physical_episodes_preserved=True,
            decision_capacity_per_environment=capacity,
            total_transitions=sum(map(len,rows)),fragment_lengths=list(map(len,rows)),allocated_envs=self.num_envs,
            active_envs=self.num_envs,seconds=time.perf_counter()-started,batches=self.batch_reports,
            world_timing=self.world_timing,columnar_reward_cpu_timings=reward_timings,
            columnar_reward_timing_scope='all_consumed_controls_in_this_world_window',
            columnar_reward_timing_contract='per_consume_call_across_episode_replacements.v2',
            columnar_reward_fallbacks=reward_timings.fallback_calls,
            columnar_reward_fallback_scope='actual_consume_batch_fallbacks_this_rollout',
            server_rpc_transport_timing=protocol_delta,normal_boundary_resets=0,
            training_reward_seconds=sum(row.metadata.get('training_reward_seconds',0.) for fragment in rows for row in fragment),
            training_reward_timing_scope='accepted_transition_metadata_excludes_separate_warmup',
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
        # 记录Python循环垃圾回收真实停顿，不删慢样本，不从部署延迟扣除。
        import gc
        gc_events=[];gc_started={}
        def observe_gc(phase,info):
            generation=info['generation']
            if phase=='start':gc_started[generation]=time.perf_counter()
            elif generation in gc_started:
                gc_events.append(dict(sample_index=index,generation=generation,
                    seconds=time.perf_counter()-gc_started.pop(generation),
                    collected=info['collected'],uncollectable=info['uncollectable']))
        gc.callbacks.append(observe_gc)
        try:
            for index in range(warmup+samples):
                records=self._drive({i:generate(i) for i in range(self.num_envs)})
                if index>=warmup:
                    for slot,value in records.items():durations[slot].append(value)
        finally:
            gc.callbacks.remove(observe_gc)
        self._drive({i:backend_operation('retire') for i in range(self.num_envs)})
        clock=self.states[0].resource.env.deployment_clock
        return dict(durations=durations,calibration_reservation_seconds=self.states[0].resource.env.latency_budget_s,
            batches=self.batch_reports,gc_events=gc_events,
            deployment_profile_sha256=None if clock is None else clock.sha256,
            scope='measured_generation_prepare_wallclock_without_execution_no_prior_latency_profile')

    def close(self):
        self.cancelled.set()
        for state in self.states:
            if state is not None:state.resource.close()
        try:
            self.shutdown_result=self.world_call('close');validate_vector_close(self.shutdown_result)
        finally:self.world.journal.close();self.world.client.close()
        return self.shutdown_result

    def state_dict(self):
        if self.active or self.cancelled.is_set() or self.rpc_pending:
            raise RuntimeError('World checkpoint requires an idle decision boundary')
        from .execution_state import capture_execution_state
        saved=[]
        for state in self.states:
            if state is None:
                saved.append(None);continue
            env=state.resource.env
            saved.append(dict(execution=dict(capture_execution_state(env),policy_version=env.policy_version,
                iteration=env.iteration),sampler=state.resource.sampler.state_dict(),transitions=state.transitions,
                seed=env.config['stage9']['seed'],budget=env.budget.state_dict(),
                ended_physical_episode=None if state.task is None else
                    dict(episode_id=env.snapshot['episode_id'],tick=env.snapshot['tick'])))
        return dict(schema='genmo.gpu_vector_collector.boundary.v2',num_envs=self.num_envs,
            fragment_contract=WORLD_FLOW_CONTRACT,numerical_layout=self.policy.numerical_layout,states=saved,
            restore_environment='fresh_PhysX_episodes_preserve_rng_cursors_and_spent_budget')

    def load_state_dict(self,saved):
        if self.active or any(state is not None for state in self.states):
            raise RuntimeError('Restore requires a new world collector')
        if (saved.get('schema')!='genmo.gpu_vector_collector.boundary.v2'
                or saved.get('fragment_contract')!=WORLD_FLOW_CONTRACT
                or saved.get('num_envs')!=self.num_envs
                or saved.get('numerical_layout')!=self.policy.numerical_layout
                or len(saved['states'])!=self.num_envs):
            raise ValueError('World topology or current logical checkpoint contract differs')
        from .execution_state import restore_execution_state
        for slot,record in enumerate(saved['states']):
            if record is None:continue
            resource=self.factory(slot,_WorldPolicyView(self))
            if resource.env.config['stage9']['seed']!=record['seed']:
                raise ValueError('Per-environment seed differs')
            resource.sampler.load_state_dict(record['sampler'])
            restore_execution_state(resource.env,record['execution'])
            resource.env.collector_env_slot=slot
            self.states[slot]=SimpleNamespace(resource=resource,task=None,transitions=record['transitions'])
        self.restore_records=None
