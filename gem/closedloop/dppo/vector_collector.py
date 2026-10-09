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
import threading
import time
from concurrent.futures import Future,ThreadPoolExecutor
from .dual_collector import DualEnvironmentCollector,split_trace
from .rpc import AcknowledgedBackend
from gem.runtime.closedloop_protocol import RemoteError
import torch


class VectorWorldClient(AcknowledgedBackend):
    MUTATIONS={'exchange'}


class VectorLaneBackend(AcknowledgedBackend):
    MUTATIONS=AcknowledgedBackend.MUTATIONS|{'retire','activate'}


class QueuedLaneClient:
    def __init__(self,owner,env_id):
        self.owner,self.env_id=owner,env_id
        self.closed=False
        self.last_call_timing={}

    def call(self,method,**payload):
        if self.closed or self.owner.cancelled.is_set():raise RuntimeError('Vector lane stopped')
        future=Future()
        self.owner.rpc_requests.put((self.env_id,method,payload,future,time.perf_counter()))
        return future.result(timeout=self.owner.timeout_seconds)

    def close(self):self.closed=True


class VectorEnvironmentCollector(DualEnvironmentCollector):
    def __init__(self,policy,factory,world_client,*,num_envs=8,batch_wait_seconds=.003,timeout_seconds=600.):
        if type(num_envs)is not int or num_envs<1:raise ValueError('Positive environment count required')
        if policy.numerical_layout!='sample_matrix_bmm_fp32.v1':
            raise ValueError('Vector sampling requires validated row-independent policy')
        self.policy,self.factory,self.world=policy,factory,world_client
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

    def lane_client(self,env_id):return QueuedLaneClient(self,env_id)

    def _fragment(self,slot,count,policy_version):
        if self.states[slot] is not None and self.states[slot].task is not None:
            self.states[slot].resource.env.backend.call('activate')
        rows=super()._fragment(slot,count,policy_version)
        env=self.states[slot].resource.env
        env.backend.call('retire')
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
        signature=self.policy._parameter_signature()
        futures=[self.executors[i].submit(self._fragment,i,count,policy_version) for i,count in enumerate(counts)]
        started=time.perf_counter();self.batch_reports=[];self.world_timing={}
        pending=[]
        try:
            while not all(f.done() for f in futures):
                for f in futures:
                    if f.done() and f.exception() is not None:raise f.exception()
                if time.perf_counter()-started>self.timeout_seconds:raise TimeoutError('Vector collection deadline')
                worked=self._pump_rpc()
                while len(pending)<enabled:
                    try:pending.append(self.requests.get_nowait())
                    except queue.Empty:break
                if pending and (len(pending)==enabled or time.perf_counter()-pending[0][4]>=self.batch_wait_seconds):
                    slots=[p[0] for p in pending]
                    if len(set(slots))!=len(slots):raise RuntimeError('Duplicate causal environment generation request')
                    conditions={k:torch.cat([p[1][k] for p in pending]) for k in pending[0][1]}
                    begin=time.perf_counter()
                    trace=self.policy.sample_rollout(conditions,generator=[p[2] for p in pending])
                    end=time.perf_counter()
                    self.batch_reports.append(dict(environment_slots=slots,effective_rows=len(pending),padding_rows=0,
                        generation_seconds=end-begin,queue_wait_seconds=[begin-p[4] for p in pending]))
                    for i,p in enumerate(pending):
                        p[3].set_result((split_trace(trace,i,len(pending)),dict(self.policy.last_sample_timing)))
                    pending=[];worked=True
                if not worked:time.sleep(.0005)
            fragments=[f.result() for f in futures]
            if self.rpc_pending or not self.rpc_requests.empty():raise RuntimeError('Unfinished physical requests at rollout boundary')
            if self.policy._parameter_signature()!=signature:raise RuntimeError('Actor changed during vector collection')
            return fragments,dict(schema='genmo.gpu_vector_collector.v1',total_transitions=sum(map(len,fragments)),
                allocated_envs=self.num_envs,active_envs=enabled,fragment_lengths=list(map(len,fragments)),
                seconds=time.perf_counter()-started,batches=self.batch_reports,world_timing=self.world_timing,
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
        finally:self.active=False

    def close(self):
        self.cancelled.set()
        # resource.close只关闭所属线程journal、预算及队列句柄，不再向已停调度器发RPC。
        futures=[e.submit(s.resource.close) for e,s in zip(self.executors,self.states) if s is not None]
        errors=[]
        for f in futures:
            try:f.result(timeout=self.timeout_seconds)
            except BaseException as e:errors.append(e)
        for e in self.executors:e.shutdown(wait=True,cancel_futures=True)
        if errors:raise RuntimeError('Vector resource cleanup failed') from errors[0]
