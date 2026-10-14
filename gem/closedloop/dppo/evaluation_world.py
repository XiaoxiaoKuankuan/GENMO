"""固定评估/完整MC任务在同一个共享PhysX世界中的多环境驱动。

复用训练WorldEnvironmentCollector的reserve/prepare/commit/advance因果续体，
每个slot维护独立机器人、音乐游标、噪声索引和奖励窗口；同一GPU仅一个评估世界。
按剩余歌曲长度分配slot，完成所有任务的slot显式retire，不能让无参考机器人继续
推进。每个任务仍由evaluation_episode_flow计算、按原计划顺序发布相同格式报告。
根预算已预付，全体slot在单调度线程共享该账本，所有真实控制步和四子步继续计量。
权威世界journal持久化，lane索引可在内存确认；原始随机链仍按批保存供严格评估。
本模块不训练网络、不重置训练世界、不创建每环境线程、不改变部署模拟时钟。
"""
from __future__ import annotations
import copy
from types import SimpleNamespace
import torch
from .world_collector import WorldEnvironmentCollector
from .world_flow import backend_operation
from .vector_collector import VectorLaneBackend
from .vector_environment import VectorUpperEnvironment
from .memory_execution import MemoryReceiveJournal


def assign_tasks(tasks, slots):
    """长歌优先分配，保留原任务索引、任务本身和所有种子的固定噪声身份。"""
    if slots<1:raise ValueError('Positive evaluation slot count required')
    queues=[[] for _ in range(slots)];loads=[0.]*slots
    for index,task in sorted(enumerate(tasks),key=lambda item:(-item[1]['remaining_music_seconds'],item[0])):
        slot=min(range(slots),key=lambda i:(loads[i],i))
        queues[slot].append((index,task));loads[slot]+=task['remaining_music_seconds']+1.
    return queues


def balanced_rank_tasks(plan, world):
    """以样本为单位长歌优先分rank，同一样本全部seed始终在同rank。"""
    groups={}
    for task in plan['tasks']:
        groups.setdefault((task['dataset'],task['sample_id']),[]).append(task)
    queues=[[] for _ in range(world)];loads=[0.]*world
    for key,group in sorted(groups.items(),key=lambda item:(-sum(t['remaining_music_seconds'] for t in item[1]),item[0])):
        rank=min(range(world),key=lambda r:(loads[r],r))
        queues[rank].extend(group)
        loads[rank]+=sum(t['remaining_music_seconds']+1. for t in group)
    return queues


class EvaluationWorldDriver:
    def __init__(self,c,config,budget,path):
        self.c,self.config,self.budget,self.path=c,config,budget,path
        handle=c.active_evaluation_world
        self.num_envs=handle.num_envs
        self.collector=WorldEnvironmentCollector(c.policy,self._factory,handle.world,
            num_envs=self.num_envs,generation_batch=self.num_envs,timeout_seconds=3600.,
            training_storage='durable_cpu.v1')
        for slot in range(self.num_envs):self.collector._resource(slot)
        self.env=self.collector.states[0].resource.env
        self.results=[]

    def _factory(self,slot,proxy):
        config=copy.deepcopy(self.config)
        config['runtime']['evidence_level']='full'
        journal=MemoryReceiveJournal(self.path/f'env{slot:03d}'/'lane_index.sqlite')
        backend=VectorLaneBackend(self.collector.lane_client(slot),journal)
        # 常驻世界在上一批retire后仍保留episode身份；新客户端只能接管已确认的同一水位。
        previous=getattr(self.c.active_evaluation_world,'lane_boundaries',{}).get(slot)
        if previous is not None:
            if (backend.session_id,backend.sequence)!=(previous['session_id'],previous['sequence']):
                raise RuntimeError('Persistent evaluation lane changed session or acknowledged sequence')
            backend.episode_id=previous['episode_id']
        elif backend.sequence:
            raise RuntimeError('Cannot adopt a used evaluation lane without its confirmed episode identity')
        env=VectorUpperEnvironment(config,backend,self.c.builder,proxy,self.budget,self.path/f'env{slot:03d}')
        env.disk_guard=self.c.guard;env.latency_budget_s=self.c.env.latency_budget_s
        env.authoritative_journal_path=str(self.c.active_evaluation_world.journal.path)
        env.policy_version=self.c.state['policy_version'];env.iteration=self.c.state['iteration']
        return SimpleNamespace(env=env,journal=journal,close=journal.close)

    def run(self,tasks,flow):
        queues=assign_tasks(tasks,self.num_envs)
        used_before=self.budget.state_dict()['used']['control_steps']
        def slot_flow(slot):
            env=self.collector.states[slot].resource.env
            for index,task in queues[slot]:
                episode,error=yield from flow(env,index,task)
                self.results.append((index,episode,error))
                if error is not None:break
            yield from backend_operation('retire')
        with torch.no_grad():self.collector._drive({i:slot_flow(i) for i in range(self.num_envs)})
        allocated=int(self.collector.world_timing.get('allocated_physics_steps',0))
        active=self.budget.state_dict()['used']['control_steps']-used_before
        if allocated%4 or allocated<4*active:
            raise RuntimeError('Evaluation physical workload accounting differs from actual control records')
        extra=allocated//4-active
        if extra:self.budget.reserve('inactive_evaluation_PhysX_slots',control_steps=extra,physics_steps=4*extra)
        self.execution_summary=dict(allocated_physics_steps=allocated,actual_GMT_controls=active,
            inactive_physics_steps=4*extra,timing=self.collector.world_timing)
        return self.results

    def close(self):
        # 常驻世界归会话所有；这里只释放本次评估的lane索引。
        self.c.active_evaluation_world.lane_boundaries={
            slot:dict(session_id=state.resource.env.backend.session_id,
                sequence=state.resource.env.backend.sequence,episode_id=state.resource.env.backend.episode_id)
            for slot,state in enumerate(self.collector.states)}
        for state in self.collector.states:state.resource.close()
