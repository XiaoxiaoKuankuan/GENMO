"""每 rank 两个独立环境的 ready 队列采样原型，默认训练入口不自动启用。

每个工作线程独占环境、后端连接、SQLite journal、预算、音乐采样器和连续任务状态。
线程只提交已取得真实反馈的当前生成请求；协调线程收集最多两个请求，在同一冻结
Actor 上合并执行20步随机扩散，再将各自的链和计时交还调用线程。每个环境仍严格
因果推进，网络 batch 的两行对应两个真实环境，并各用自己的 Generator。

factory必须在对应线程中构建资源，返回具有 env、sampler、writer、close 属性的对象。
它不得直接创建未经容量检查的Isaac进程；调用者先证明资源可用。此模块不含正式作业
切换，不伪称旧单环境checkpoint可恢复两环境。失败会使所有等待采样请求失败，外层
仍需执行已有多rank故障协调及预算不退款。环境/音乐状态跨普通采集边界保留，GAE
交付按环境分开的连续片段；重建后端时必须从显式恢复边界开始新episode。
"""
from __future__ import annotations

import copy
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from types import SimpleNamespace
import torch


def fixed_fragment_targets(fragments, critic, device, *, critic_version, gamma_upper=.99, lambda_upper=.95):
    """分别按环境/episode/连续区间计算GAE；返回按环境顺序的行，留待全局归一化一次。"""
    from .trainer import fixed_targets
    targets = [fixed_targets(rows,critic,device, gamma_upper=gamma_upper,lambda_upper=lambda_upper,
                             reuse_values=True,critic_version=critic_version,normalize=False)
               for rows in fragments]
    result = {}
    for key in targets[0]:
        values = [item[key] for item in targets]
        if torch.is_tensor(values[0]):
            result[key] = torch.cat(values)
        elif not all(value==values[0] for value in values):
            raise ValueError('Independent fragment return contracts differ')
        else:
            result[key] = values[0]
    return [row for rows in fragments for row in rows],result


def split_trace(trace, index, batch_size):
    result = {}
    for key,value in trace.items():
        if key == 'conditions':
            result[key] = {name:tensor[index:index+1] for name,tensor in value.items()}
        elif torch.is_tensor(value) and key != 'timestep_map':
            if value.shape[0] != batch_size:
                raise ValueError(f'Unexpected batched trace field {key}')
            result[key] = value[index:index+1]
        else:
            result[key] = copy.deepcopy(value)
    return result


class _QueuedPolicy:
    def __init__(self, owner, slot):
        self.owner, self.slot = owner, slot
        self.actor = owner.policy.actor
        self.steps = owner.policy.steps
        self.guidance_scale = owner.policy.guidance_scale
        self.last_sample_timing = {}

    @property
    def kernel_config(self):
        return self.owner.policy.kernel_config

    def sample_rollout(self, conditions, *, generator, **kwargs):
        if kwargs:
            raise ValueError('Dual collector only accepts actual stochastic requests')
        if self.owner.cancelled.is_set():
            raise RuntimeError('Dual collection aborted')
        future = Future()
        self.owner.requests.put((self.slot, conditions, generator, future, time.perf_counter()))
        deadline = time.perf_counter()+self.owner.timeout_seconds
        while True:
            try:
                trace, timing = future.result(timeout=.02)
                break
            except FutureTimeoutError:
                if future.done():
                    raise
                if self.owner.cancelled.is_set():
                    raise RuntimeError('Dual collection aborted while waiting for generation')
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Timed out waiting for batched generation')
        self.last_sample_timing = timing
        return trace


class DualEnvironmentCollector:
    def __init__(self, policy, factory, *, batch_wait_seconds=.003, timeout_seconds=120.):
        if getattr(policy,'numerical_layout',None) != 'sample_matrix_bmm_fp32.v1':
            raise ValueError('Two environments require the validated row-independent numeric contract')
        if batch_wait_seconds < 0 or timeout_seconds <= 0:
            raise ValueError('Invalid bounded ready queue timing')
        self.policy, self.factory = policy, factory
        self.batch_wait_seconds, self.timeout_seconds = batch_wait_seconds, timeout_seconds
        self.requests, self.cancelled = queue.Queue(), threading.Event()
        # 一个单线程executor对应一个环境，保证SQLite连接一直由创建线程使用。
        self.executors = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'gmt-env-{i}') for i in range(2)]
        self.states, self.active = [None,None], False
        self.batch_reports = []

    def _fragment(self, slot, count, policy_version):
        state = self.states[slot]
        if state is None:
            resource = self.factory(slot, _QueuedPolicy(self,slot))
            state = self.states[slot] = SimpleNamespace(resource=resource, task=None, transitions=0)
        env, sampler = state.resource.env, state.resource.sampler
        env.collector_env_slot = slot
        env.policy_version = policy_version
        rows = []
        for _ in range(count):
            if self.cancelled.is_set():
                raise RuntimeError('Peer environment failed')
            if state.task is None:
                task = sampler.next_task()
                if task['sample']['row']['split'] != 'train':
                    raise ValueError('Validation task cannot enter training collector')
                env.reset_task(task['sample'], task['music'], seed=env.config['stage9']['seed']+env.episode_count,
                               phase='train', music_start_frame=task['music_start_frame'])
                state.task = task
            row = env.step()
            if not row.transition_valid or row.identity['policy_version'] != policy_version:
                raise ValueError('Invalid or mixed-version environment transition')
            task = state.task
            row.metadata['collector_env_slot'] = slot
            row.metadata['training_task'] = dict(dataset=task['sample']['dataset'],
                sample_id=task['sample']['row']['sample_id'], split='train', music_start_frame=task['music_start_frame'],
                manifest_sha256=task['sample']['manifest_sha256'])
            sampler.record_execution(task, row.executed_control_steps)
            rows.append(row)
            state.transitions += 1
            if row.terminated or row.truncated:
                state.task = None
        # 只切断本次GAE片段；不reset尚未结束的真实环境，也不丢弃其音乐游标。
        if rows and not rows[-1].terminated:
            rows[-1].truncated = True
            rows[-1].reason = rows[-1].reason or 'collector_fragment_boundary'
        return rows

    def collect(self, *, count_per_rank=20, policy_version):
        if self.active or self.cancelled.is_set() or count_per_rank < 2 or count_per_rank%2:
            raise ValueError('Dual collector requires an idle healthy collector and an even total count')
        self.active = True
        signature = self.policy._parameter_signature()
        futures = [ex.submit(self._fragment,i,count_per_rank//2,policy_version) for i,ex in enumerate(self.executors)]
        started = time.perf_counter()
        self.batch_reports = []
        pending = []
        validated_resources = False
        try:
            while not all(f.done() for f in futures):
                for future in futures:
                    if future.done() and future.exception() is not None:
                        raise future.exception()
                if time.perf_counter()-started > self.timeout_seconds:
                    raise TimeoutError('Dual collection exceeded finite deadline')
                try:
                    pending = [self.requests.get(timeout=.01)]
                except queue.Empty:
                    continue
                deadline = time.perf_counter()+self.batch_wait_seconds
                while len(pending)<2:
                    try:
                        pending.append(self.requests.get(timeout=max(0.,deadline-time.perf_counter())))
                    except queue.Empty:
                        break
                slots = [item[0] for item in pending]
                if len(set(slots)) != len(slots):
                    raise ValueError('Duplicate causal request from the same environment')
                if not validated_resources and all(state is not None for state in self.states):
                    left,right = [state.resource for state in self.states]
                    if left.env is right.env or left.sampler is right.sampler or left.env.backend is right.env.backend:
                        raise ValueError('Environments, samplers and backend connections must be independent')
                    if left.env.backend.session_id == right.env.backend.session_id:
                        raise ValueError('Two workers cannot share an execution session')
                    validated_resources = True
                context = {key:torch.cat([item[1][key] for item in pending]) for key in pending[0][1]}
                generating = time.perf_counter()
                trace = self.policy.sample_rollout(context, generator=[item[2] for item in pending])
                finished = time.perf_counter()
                timing = dict(self.policy.last_sample_timing)
                self.batch_reports.append(dict(environment_slots=slots, effective_rows=len(pending), padding_rows=0,
                    cfg_rows=len(pending)*(2 if self.policy.cfg_batch else 1),
                    ready_queue_wait_seconds=[generating-item[4] for item in pending],
                    batched_generation_seconds=finished-generating,
                    request_until_ready_seconds=[finished-item[4] for item in pending]))
                for index,item in enumerate(pending):
                    item[3].set_result((split_trace(trace,index,len(pending)),dict(timing)))
                pending = []
            fragments = [f.result() for f in futures]
            if self.policy._parameter_signature() != signature:
                raise RuntimeError('Actor changed during collection')
            return fragments, dict(schema='genmo.dual_collector.prototype.v1',
                total_transitions=sum(map(len,fragments)), fragment_lengths=list(map(len,fragments)),
                seconds=time.perf_counter()-started, batches=self.batch_reports,
                gae_contract='independent_per_environment_episode_contiguous_fragment',
                persistent_episode=[state.task is not None for state in self.states])
        except BaseException as error:
            self.cancelled.set()
            while True:
                try:
                    pending.append(self.requests.get_nowait())
                except queue.Empty:
                    break
            for item in pending:
                if not item[3].done():
                    item[3].set_exception(error)
            raise
        finally:
            self.active = False

    def close(self):
        self.cancelled.set()
        futures = [executor.submit(state.resource.close) for executor,state in zip(self.executors,self.states) if state is not None]
        errors = []
        for future in futures:
            try:
                future.result(timeout=self.timeout_seconds)
            except Exception as error:
                errors.append(error)
        for executor in self.executors:
            executor.shutdown(wait=True,cancel_futures=True)
        if errors:
            raise RuntimeError('Dual environment close failed after all resources were closed') from errors[0]

    def state_dict(self):
        """一致采集边界的双游标快照；不声称保存PhysX内部状态或退还资源预算。"""
        if self.active or self.cancelled.is_set() or any(state is None for state in self.states):
            raise RuntimeError('Checkpoint requires an initialized healthy collection boundary')
        def capture(state):
            from tools.train_closedloop_stage10 import capture_execution_state
            env = state.resource.env
            task = state.task
            return dict(execution=dict(capture_execution_state(env), policy_version=env.policy_version,iteration=env.iteration),
                sampler=copy.deepcopy(state.resource.sampler.state_dict()), transitions=state.transitions,
                budget=(copy.deepcopy(env.budget.state_dict()) if getattr(env,'budget',None) is not None else None),
                active_episode=None if task is None else dict(sample_id=task['sample']['row']['sample_id'],
                    dataset=task['sample']['dataset'], music_start_frame=task['music_start_frame'],
                    task_index=task.get('task_index'), boundary_tick=int(env.snapshot['tick'])))
        states=[ex.submit(capture,state).result(timeout=self.timeout_seconds) for ex,state in zip(self.executors,self.states)]
        return dict(schema='genmo.dual_collector.boundary.v1', numerical_layout=self.policy.numerical_layout,
            environment_count=2, states=states, restore_environment='rebuild_at_new_episode_boundary_no_budget_refund')

    def load_state_dict(self, saved):
        """仅新建collector恢复；重新建立后端，明确记录故障回退后结束的旧物理episode。"""
        if self.active or any(state is not None for state in self.states):
            raise RuntimeError('Restore requires a fresh collector')
        if (saved.get('schema')!='genmo.dual_collector.boundary.v1' or saved.get('environment_count')!=2
                or saved.get('numerical_layout')!=self.policy.numerical_layout or len(saved.get('states',[]))!=2):
            raise ValueError('Incompatible dual collector checkpoint')
        def restore(slot, record):
            from tools.train_closedloop_stage10 import restore_execution_state
            resource=self.factory(slot,_QueuedPolicy(self,slot))
            resource.env.collector_env_slot=slot
            spent_generations = None
            if record.get('budget') is not None:
                live = resource.env.budget.state_dict()
                if (live['limits'] != record['budget']['limits'] or any(
                        live['used'][key] < value for key,value in record['budget']['used'].items())):
                    resource.close()
                    raise ValueError('Dual collector recovery cannot refund or reset execution budget')
                spent_generations = live['used']['generations']
            resource.sampler.load_state_dict(record['sampler'])
            restore_execution_state(resource.env,record['execution'],spent_generations=spent_generations)
            self.states[slot]=SimpleNamespace(resource=resource,task=None,transitions=record['transitions'])
            return dict(slot=slot, rebuilt_backend_session=resource.env.backend.session_id,
                        closed_prior_episode=record['active_episode'], budget_refunded=False)
        return [ex.submit(restore,i,record).result(timeout=self.timeout_seconds)
                for i,(ex,record) in enumerate(zip(self.executors,saved['states']))]
