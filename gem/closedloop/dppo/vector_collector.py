"""一张GPU一个共享PhysX世界的真实多环境GENMO采样协调器。

每个环境线程只负责自己的音乐、历史、前缀、奖励和journal；它没有独立Isaac实例。
线程通过有界请求队列使用世界级RPC，所有物理推进由GMT向量服务统一调度。GENMO
请求收集为来自不同真实环境的batch，每条链使用独立Generator，一次执行20步扩散。
共享Actor在完整rollout期间保持冻结，采完每rank20条后才允许DPPO更新。

外层exchange回复先在世界journal持久化并确认，再分发给逐环境ack.v2客户端；环境
journal仍先落盘后ACK。结束片段显式retire，下一轮继续时activate，不把活动但等待
参考的机器人冻结。断点只保存逻辑游标/RNG并要求重建物理世界，不伪造PhysX恢复。
本模块复用已有单环境上层执行语义、奖励和完整采集链，不扩大全局训练batch。
"""
from __future__ import annotations
import queue
import copy
import threading
import time
from concurrent.futures import Future,ThreadPoolExecutor,TimeoutError as FutureTimeout
from types import SimpleNamespace
from .dual_collector import DualEnvironmentCollector,split_trace
from .rpc import AcknowledgedBackend
from gem.runtime.closedloop_protocol import RemoteError
import torch


class VectorWorldClient(AcknowledgedBackend):
    MUTATIONS={'exchange','begin_rollout'}


class VectorLaneBackend(AcknowledgedBackend):
    MUTATIONS=AcknowledgedBackend.MUTATIONS|{'retire','activate','join_fragment','drain_fragment'}

    def call(self,method,**payload):
        result=super().call(method,**payload)
        if method=='drain_fragment':
            from gem.runtime.trajectory_blocks import expand_feedback
            result=expand_feedback(result)
        return result


class QueuedLaneClient:
    def __init__(self,owner,env_id):
        self.owner,self.env_id=owner,env_id
        self.closed=False
        self.last_call_timing={}

    def call(self,method,**payload):
        if self.closed or self.owner.cancelled.is_set():raise RuntimeError('Vector lane stopped')
        future=Future()
        self.owner.rpc_requests.put((self.env_id,method,payload,future,time.perf_counter()))
        deadline=time.perf_counter()+self.owner.timeout_seconds
        while True:
            try:return future.result(timeout=.1)
            except FutureTimeout:
                if future.done():raise
                if self.owner.cancelled.is_set():raise RuntimeError('Vector request cancelled')
                if time.perf_counter()>deadline:raise TimeoutError('Vector lane request deadline')

    def close(self):self.closed=True


class VectorEnvironmentCollector(DualEnvironmentCollector):
    def __init__(self,policy,factory,world_client,*,num_envs=8,batch_wait_seconds=.1,timeout_seconds=600.):
        if type(num_envs)is not int or num_envs<1:raise ValueError('Positive environment count required')
        if policy.numerical_layout!='sample_matrix_bmm_fp32.v1':
            raise ValueError('Vector sampling requires validated row-independent policy')
        self.policy,self.factory=policy,factory
        self.world_factory=world_client if callable(world_client) else None
        self.world=None if self.world_factory else world_client
        self.num_envs=num_envs
        self.batch_wait_seconds,self.timeout_seconds=batch_wait_seconds,timeout_seconds
        self.requests,self.rpc_requests=queue.Queue(),queue.Queue()
        self.cancelled=threading.Event()
        self.executors=[ThreadPoolExecutor(max_workers=1,thread_name_prefix=f'vector-env-{i}') for i in range(num_envs)]
        self.states=[None]*num_envs
        self.active=False
        self.batch_reports=[]
        self.rpc_pending={}
        self.rpc_sequence=0
        self.world_timing={}
        self.restore_records=None
        self.world_commands=queue.Queue()
        self.rpc_stop=threading.Event()
        self.rpc_error=None
        self.rpc_thread=None
        self.remaining_generations=None
        if self.world_factory is not None:
            ready=Future()
            self.rpc_thread=threading.Thread(target=self._rpc_loop,args=(ready,),name='gpu-world-rpc',daemon=True)
            self.rpc_thread.start();ready.result(timeout=timeout_seconds)

    def _rpc_loop(self,ready):
        """RPC/socket/SQLite均由本线程创建及使用，生成线程不会挡住参考准备。"""
        try:
            self.world=self.world_factory();ready.set_result(True)
            while not self.rpc_stop.is_set():
                try:method,payload,future=self.world_commands.get_nowait()
                except queue.Empty:pass
                else:
                    try:future.set_result(self.world.call(method,**payload))
                    except BaseException as error:future.set_exception(error);raise
                    if method=='close':break
                if not self._pump_rpc():time.sleep(.0005)
        except BaseException as error:
            self.rpc_error=error;self.cancelled.set()
            if not ready.done():ready.set_exception(error)
            for future,_ in self.rpc_pending.values():
                if not future.done():future.set_exception(error)
        finally:
            if self.world is not None:
                self.world.journal.close();self.world.client.close()

    def world_call(self,method,**payload):
        if self.rpc_thread is None:return self.world.call(method,**payload)
        if self.rpc_error is not None:raise RuntimeError('Vector RPC dispatcher failed') from self.rpc_error
        future=Future();self.world_commands.put((method,payload,future))
        return future.result(timeout=self.timeout_seconds)

    def _service(self):
        if self.rpc_error is not None:raise RuntimeError('Vector RPC dispatcher failed') from self.rpc_error
        return False if self.rpc_thread is not None else self._pump_rpc()

    def lane_client(self,env_id):return QueuedLaneClient(self,env_id)

    def _run_boundary_jobs(self,futures,*,allow_generation=False):
        """只在静止世界边界服务初始化RPC，不容许意外GENMO请求。"""
        deadline=time.perf_counter()+self.timeout_seconds
        pending=[]
        while not all(f.done() for f in futures):
            for f in futures:
                if f.done() and f.exception() is not None:raise f.exception()
            if time.perf_counter()>deadline:raise TimeoutError('Vector boundary operation deadline')
            if not allow_generation and not self.requests.empty():raise RuntimeError('Unexpected generation at checkpoint boundary')
            worked=self._service()
            if allow_generation:
                pending,generated=self._generate_ready(pending,len(futures));worked=worked or generated
            if not worked:time.sleep(.0005)
        return [f.result() for f in futures]

    def _generate_ready(self,pending,enabled):
        if self.remaining_generations is not None:
            enabled=sum(count>0 for count in self.remaining_generations)
        while len(pending)<enabled:
            try:pending.append(self.requests.get_nowait())
            except queue.Empty:break
        if not pending or (len(pending)<enabled and time.perf_counter()-pending[0][4]<self.batch_wait_seconds):
            return pending,False
        slots=[p[0] for p in pending]
        if len(set(slots))!=len(slots):raise RuntimeError('Duplicate causal environment generation request')
        conditions={k:torch.cat([p[1][k] for p in pending]) for k in pending[0][1]}
        begin=time.perf_counter()
        trace=self.policy.sample_rollout(conditions,generator=[p[2] for p in pending])
        end=time.perf_counter()
        self.batch_reports.append(dict(environment_slots=slots,effective_rows=len(pending),padding_rows=0,
            generation_seconds=end-begin,queue_wait_seconds=[begin-p[4] for p in pending]))
        for i,p in enumerate(pending):
            if self.remaining_generations is not None:
                self.remaining_generations[p[0]]-=1
            p[3].set_result((split_trace(trace,i,len(pending)),dict(self.policy.last_sample_timing)))
        return [],True

    def calibrate(self,*,count_per_rank=20,warmup=1,samples=2):
        """用真实多环境请求校准新的批量部署延迟；校准不进入训练rollout。"""
        from .dual_collector import _QueuedPolicy
        enabled=min(self.num_envs,count_per_rank)
        def initialize(slot):
            if self.states[slot] is None:
                resource=self.factory(slot,_QueuedPolicy(self,slot))
                self.states[slot]=SimpleNamespace(resource=resource,task=None,transitions=0)
            state=self.states[slot];env=state.resource.env;env.collector_env_slot=slot
            sampler=state.resource.sampler;saved=copy.deepcopy(sampler.state_dict());task=sampler.next_task();sampler.load_state_dict(saved)
            env.reset_task(task['sample'],task['music'],seed=env.config['stage9']['seed'],phase='calibration',music_start_frame=task['music_start_frame'])
        self._run_boundary_jobs([self.executors[i].submit(initialize,i) for i in range(enabled)])
        durations=[[] for _ in range(enabled)]
        for iteration in range(warmup+samples):
            def generate(slot):
                env=self.states[slot].resource.env
                generated=env.generate()
                if generated['rejection'] or generated['prepared'] is None:raise RuntimeError('Calibration reference rejected')
                env.backend.call('discard_plan',prepared_plan_id=generated['prepared']['prepared_plan_id'])
                env.decision+=1
                return generated['critical_ready_seconds']
            values=self._run_boundary_jobs([self.executors[i].submit(generate,i) for i in range(enabled)],allow_generation=True)
            prime=getattr(self.policy.actor.denoiser.forward,'prime_sample_batches',None)
            if prime is not None and iteration==0:
                prime(enabled)
            if iteration>=warmup:
                for i,v in enumerate(values):durations[i].append(v)
        import math
        latency=math.ceil((max(map(max,durations))*1.25+.04)*50)/50
        def finish(slot):
            env=self.states[slot].resource.env;env.latency_budget_s=latency
            env.backend.call('retire')
        self._run_boundary_jobs([self.executors[i].submit(finish,i) for i in range(enabled)])
        return dict(durations=durations,latency_budget_s=latency,batches=list(self.batch_reports),
            scope='real_vector_deployment_calibration_no_training_transitions')

    def state_dict(self):
        if self.active or self.cancelled.is_set() or self.rpc_pending:raise RuntimeError('Vector checkpoint requires consistent idle boundary')
        def capture(state):
            from tools.train_closedloop_stage10 import capture_execution_state
            env=state.resource.env
            return dict(execution=dict(capture_execution_state(env),policy_version=env.policy_version,iteration=env.iteration),sampler=state.resource.sampler.state_dict(),
                transitions=state.transitions,seed=env.config['stage9']['seed'],
                budget=env.budget.state_dict(),
                ended_physical_episode=None if state.task is None else dict(episode_id=env.snapshot['episode_id'],tick=env.snapshot['tick']))
        saved=[None]*self.num_envs
        for i,(executor,state) in enumerate(zip(self.executors,self.states)):
            if state is not None:saved[i]=executor.submit(capture,state).result(timeout=self.timeout_seconds)
        return dict(schema='genmo.gpu_vector_collector.boundary.v1',num_envs=self.num_envs,
            numerical_layout=self.policy.numerical_layout,states=saved,
            restore_environment='fresh_PhysX_episodes_preserve_rng_cursors_and_spent_budget')

    def load_state_dict(self,saved):
        if self.active or any(s is not None for s in self.states):raise RuntimeError('Restore requires new vector collector')
        if (saved.get('schema')!='genmo.gpu_vector_collector.boundary.v1' or saved.get('num_envs')!=self.num_envs
                or saved.get('numerical_layout')!=self.policy.numerical_layout or len(saved['states'])!=self.num_envs):
            raise ValueError('Vector execution topology/contract differs from checkpoint')
        from .dual_collector import _QueuedPolicy
        self.restore_records=saved['states']
        def restore(slot,record):
            from tools.train_closedloop_stage10 import restore_execution_state
            resource=self.factory(slot,_QueuedPolicy(self,slot))
            if resource.env.config['stage9']['seed']!=record['seed']:raise ValueError('Per-environment seed differs')
            resource.sampler.load_state_dict(record['sampler'])
            restore_execution_state(resource.env,record['execution'])
            resource.env.collector_env_slot=slot
            # 旧物理episode明确结束，新world不会伪称中途PhysX恢复。
            self.states[slot]=SimpleNamespace(resource=resource,task=None,transitions=record['transitions'])
        self._run_boundary_jobs([self.executors[i].submit(restore,i,r) for i,r in enumerate(saved['states']) if r is not None])
        self.restore_records=None

    def _fragment(self,slot,count,policy_version):
        from .performance import PhaseProfiler,activate,deactivate
        profiler=PhaseProfiler('cpu',slot,detailed=True);token=activate(profiler)
        try:return self._fragment_profiled(slot,count,policy_version)
        finally:
            deactivate(token)
            if self.states[slot] is not None:self.states[slot].profile=profiler.report()

    def _fragment_profiled(self,slot,count,policy_version):
        if self.states[slot] is None:
            from .dual_collector import _QueuedPolicy
            resource=self.factory(slot,_QueuedPolicy(self,slot))
            self.states[slot]=SimpleNamespace(resource=resource,task=None,transitions=0)
        self.states[slot].resource.env.backend.call('join_fragment')
        rows=super()._fragment(slot,count,policy_version)
        state=self.states[slot];env=state.resource.env
        from .vector_boundary import finish_vector_fragment
        extra,ended=finish_vector_fragment(env,rows[-1],continue_episode=state.task is not None)
        if extra and state.task is not None:state.resource.sampler.record_execution(state.task,extra)
        if ended:state.task=None
        return rows

    def _pump_rpc(self):
        requests=[]
        while len(requests)<self.num_envs*2:
            try:env_id,method,payload,future,started=self.rpc_requests.get_nowait()
            except queue.Empty:break
            self.rpc_sequence+=1
            key=str(self.rpc_sequence)
            self.rpc_pending[key]=(future,started)
            requests.append(dict(env_id=env_id,method=method,payload=payload,request_id=key))
        if not requests:return False
        result=self.world.call('exchange',requests=requests)
        for name,value in result.get('timing',{}).items():self.world_timing[name]=self.world_timing.get(name,0.)+value
        if result.get('fatal'):
            raise RuntimeError(f'GPU world failed; full evidence persisted before ACK: {result["fatal"].get("message")}')
        for reply in result['replies']:
            future,started=self.rpc_pending.pop(reply['request_id'])
            if reply['ok']:future.set_result(reply['result'])
            else:future.set_exception(RemoteError(reply['error']))
        return True

    def collect(self,*,count_per_rank=20,policy_version):
        if self.active or self.cancelled.is_set() or count_per_rank<1:raise ValueError('Invalid collector state/count')
        self.active=True
        # 多于20个已分配环境时不伪造额外训练转移，只有预算内环境进入活动集。
        enabled=min(count_per_rank,self.num_envs)
        counts=[count_per_rank//enabled+(i<count_per_rank%enabled) for i in range(enabled)]
        self.remaining_generations=list(counts)
        self.world_call('begin_rollout',env_ids=list(range(enabled)))
        signature=self.policy._parameter_signature()
        futures=[self.executors[i].submit(self._fragment,i,count,policy_version) for i,count in enumerate(counts)]
        started=time.perf_counter();self.batch_reports=[];self.world_timing={}
        pending=[]
        try:
            while not all(f.done() for f in futures):
                for f in futures:
                    if f.done() and f.exception() is not None:raise f.exception()
                if time.perf_counter()-started>self.timeout_seconds:raise TimeoutError('Vector collection deadline')
                worked=self._service()
                pending,generated=self._generate_ready(pending,enabled)
                worked=worked or generated
                if not worked:time.sleep(.0005)
            fragments=[f.result() for f in futures]
            if self.rpc_pending or not self.rpc_requests.empty():raise RuntimeError('Unfinished physical requests at rollout boundary')
            if self.policy._parameter_signature()!=signature:raise RuntimeError('Actor changed during vector collection')
            return fragments,dict(schema='genmo.gpu_vector_collector.v1',total_transitions=sum(map(len,fragments)),
                allocated_envs=self.num_envs,active_envs=enabled,fragment_lengths=list(map(len,fragments)),
                seconds=time.perf_counter()-started,batches=self.batch_reports,world_timing=self.world_timing,
                environment_thread_profiles=[state.profile for state in self.states[:enabled]],
                resource_scope='sum_thread_work_can_overlap_not_walltime',
                gae_contract='independent_per_env_episode_contiguous_fragment',real_environment_batch=True)
        except BaseException as error:
            self.cancelled.set()
            while True:
                try:pending.append(self.requests.get_nowait())
                except queue.Empty:break
            for p in pending:
                if not p[3].done():p[3].set_exception(error)
            for f,_ in self.rpc_pending.values():
                if not f.done():f.set_exception(error)
            while True:
                try:p=self.rpc_requests.get_nowait()
                except queue.Empty:break
                if not p[3].done():p[3].set_exception(error)
            raise
        finally:self.active=False;self.remaining_generations=None

    def close(self):
        self.cancelled.set()
        # resource.close只关闭所属线程journal、预算及队列句柄，不再向已停调度器发RPC。
        futures=[e.submit(s.resource.close) for e,s in zip(self.executors,self.states) if s is not None]
        errors=[]
        for f in futures:
            try:f.result(timeout=self.timeout_seconds)
            except BaseException as e:errors.append(e)
        for e in self.executors:e.shutdown(wait=True,cancel_futures=True)
        if self.rpc_thread is not None:
            if self.rpc_error is None:
                try:self.world_call('close')
                except BaseException as error:errors.append(error)
            self.rpc_stop.set();self.rpc_thread.join(timeout=self.timeout_seconds)
            if self.rpc_thread.is_alive():errors.append(TimeoutError('Vector RPC shutdown deadline'))
        if errors:raise RuntimeError('Vector resource cleanup failed') from errors[0]
