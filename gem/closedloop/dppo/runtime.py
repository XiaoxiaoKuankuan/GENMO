"""Stage10 每rank一个同进程GPU世界与独立环境状态的内存采集运行时。

调用方先启动Isaac App并初始化唯一世界，再将原生冻结GMT服务交给本运行时。
世界级续体和参考因果逻辑继续复用已验收实现，不创建每环境线程、不走Unix RPC。
普通训练采用有界内存接收和GPU随机链；严格/周期诊断切换完整持久化，物理控制、
四子步、奖励和真实终止不变。modeled_deployment.v3继续决定模拟到达时刻。

每轮根预算先持久化预占，各环境内存子账本逐操作检查；成功才结算实际消耗。
GAE按各环境/episode/连续执行片段独立计算，全局标准化一次。Critic只在采集
完成且尚未优化时固定old/next值；训练期间世界不推进。恢复只恢复逻辑计数、
音乐游标和RNG身份，明确关闭旧episode并创建新PhysX状态，不恢复部分rollout。
完整随机链在当前优化和最终20步KL验收完成前由DeviceRolloutBuffer持有。
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from .budget import atomic_json
from .rollout_helpers import fixed_fragment_targets
from .full_dataset import FullMusicSampler
from .time_sampling import StratifiedTimePool, execution_coverage
from .memory_execution import MemoryReceiveJournal,PrepaidMemoryBudget,DeviceRolloutBuffer
from .parallel_support import begin_lease,finish_lease,collection_credit,local_call,root_call
from .returns import normalize_advantages_global
from .run_management import GuardedStepJournal
from .trainer import populate_values
from .vector_collector import VectorWorldClient,VectorLaneBackend
from .vector_environment import VectorUpperEnvironment
from .world_collector import WorldEnvironmentCollector
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics


def freeze_buffer_targets(buffer, lengths, critic, device, *, critic_version, batch_size,
                          gamma_upper, lambda_upper):
    """在Buffer拥有的独立快照上固定价值；不能回到append之前的原可变对象计算GAE。"""
    rows=buffer.transitions
    if sum(lengths)!=len(rows) or any(type(n) is not int or n<0 for n in lengths):
        raise ValueError('Vector fragment lengths must cover the complete buffer')
    from .reference_credit import validate_reward_intervals
    validate_reward_intervals(rows)
    populate_values(rows,critic,device,critic_version=critic_version,batch_size=batch_size)
    fragments=[];start=0
    for n in lengths:
        fragments.append(rows[start:start+n]);start+=n
    return fixed_fragment_targets(fragments,critic,device,critic_version=critic_version,
        gamma_upper=gamma_upper,lambda_upper=lambda_upper)[1]


def _source_summary(rows):
    """按训练任务数据来源汇总本rank真实转移的奖励、执行时长与物理失败，供分来源学习曲线。"""
    summary={}
    for row in rows:
        source=row.metadata.get('training_task',{}).get('dataset','unknown')
        item=summary.setdefault(source,dict(transitions=0,reward_sum=0.,executed_seconds=0.,physical_failures=0))
        item['transitions']+=1;item['reward_sum']+=float(row.rewards.sum())
        item['executed_seconds']+=row.executed_control_steps*.02
        item['physical_failures']+=int(bool(row.metadata.get('terminal_snapshot',{}).get('terminated',False)))
    return summary


def episode_boundary_summary(rows):
    """由真实转移核对20秒边界；PPO窗口截断不算物理重置，失败优先于超时。"""
    timeouts=[r for r in rows if r.metadata.get('physical_time_limit_reached',False)]
    elapsed=[r.metadata['physical_episode_elapsed_seconds'] for r in timeouts]
    return dict(time_limit_episodes=len(timeouts),
        bootstrap_time_limit_episodes=sum(bool(r.metadata.get('time_limit_bootstrap')) for r in timeouts),
        elapsed_seconds_min=min(elapsed,default=None),elapsed_seconds_max=max(elapsed,default=None),
        physical_failure_episodes=sum(bool(r.metadata.get('terminal_snapshot',{}).get('terminated',False)) for r in rows),
        music_end_episodes=sum(r.reason=='music_end' for r in rows),
        examples=[dict(episode_id=r.identity['episode_id'],env_id=r.identity.get('env_id'),
            control_tick_end=r.control_tick_end,elapsed_seconds=r.metadata['physical_episode_elapsed_seconds'],
            music_start_frame=r.metadata.get('music_start_frame'),
            dataset=r.metadata.get('training_task',{}).get('dataset'),
            bootstrapped=bool(r.metadata.get('time_limit_bootstrap'))) for r in timeouts[:8]])


class _DirectTransport:
    def __init__(self,service):self.service=service;self.last_call_timing={}
    def call(self,method,**payload):return self.service.call(method,**payload)
    def close(self):pass


class InProcessTrainingRuntime:
    def __init__(self,c,service):
        self.c,self.service=c,service
        self.num_envs=c.config['runtime']['num_envs'];self.phase=None;self.strict=False
        dataset = c.stage['dataset']
        self.time_pool = StratifiedTimePool(c.catalog, seed=c.stage['seed'], rank=c.distributed.rank,
            world_size=c.stage['distributed']['world_size'],
            minimum_remaining_seconds=c.stage['ppo_rollout']['world_seconds'],
            bin_seconds=dataset['time_bin_seconds'], standing_start_probability=dataset['standing_start_probability'])
        world_path=c.rank_dir/'bootstrap_world.sqlite'
        world=VectorWorldClient(_DirectTransport(service),MemoryReceiveJournal(world_path))
        self.collector=WorldEnvironmentCollector(c.policy,self._factory,world,num_envs=self.num_envs,
            generation_batch=c.config['runtime']['generation_batch'],
            training_storage='ephemeral_device')
        self.iteration,self.policy_version=c.state['iteration'],c.state['policy_version']

    @property
    def latency_budget_s(self):
        if self.c.config['runtime'].get('phase_contract')=='single_request_calibration.v1':
            return self.c.config['timing']['calibration_reservation_seconds']
        return self.c.config['timing']['deployment_profile']['prefix_budget_seconds']

    def _journal(self,path):
        return GuardedStepJournal(path,self.c.guard,format='genmo.execution_journal.ndarray.v2') if self.strict else MemoryReceiveJournal(path)

    def _factory(self,slot,proxy):
        c=self.c
        config=copy.deepcopy(c.config)
        config['stage9']['seed']=(c.config['stage10']['seed']+1000003*c.distributed.rank+100003*slot)%2**32
        backend=VectorLaneBackend(self.collector.lane_client(slot),None)
        builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])),motion_frames=config['stage10']['generation_horizon']['frames'])
        env=VectorUpperEnvironment(config,backend,builder,proxy,None,c.rank_dir/'ephemeral'/f'env{slot:03d}')
        env.disk_guard=c.guard
        resource=SimpleNamespace(env=env,journal=None,sampler=FullMusicSampler(c.catalog,split='train',
            seed=config['stage9']['seed'],minimum_remaining_seconds=c.stage['ppo_rollout']['world_seconds'],
            random_start=c.stage['dataset']['random_start'],source_probabilities=c.stage['dataset']['source_probabilities'],
            time_pool=self.time_pool))
        self._bind(slot,resource)
        def close():resource.journal.close();resource.env.budget.close();backend.client.close()
        resource.close=close
        return resource

    def _bind(self,slot,resource):
        if self.phase is None:raise RuntimeError('Physical sampling needs a prepaid phase')
        if resource.journal is not None:resource.journal.close()
        if resource.env.budget is not None:resource.env.budget.close()
        directory=self.phase.path/f'env{slot:03d}' if self.strict else self.c.rank_dir/'ephemeral'/f'env{slot:03d}'
        directory.mkdir(parents=True,exist_ok=True)
        resource.journal=self._journal(directory/'execution_journal.sqlite')
        resource.env.backend.journal=resource.journal
        resource.env.budget=PrepaidMemoryBudget(dict(accepted_iterations=1,optimizer_attempts=1,**self.phase.credits[slot]))
        resource.env.output=directory
        resource.env.iteration=self.c.state['rollouts_completed']
        resource.env.policy_version=self.c.state['policy_version']
        level='full' if self.strict else self.c.config['runtime'].get('evidence_level','full')
        resource.env.config['runtime']['evidence_level']=level
        if getattr(resource.env,'reward',None) is not None:resource.env.reward.evidence_level=level

    def _open(self,label,counts,strict,*,world_seconds=None):
        c=self.c
        if self.phase is not None:raise RuntimeError('Cannot overlap physical rollout phases')
        self.strict=bool(strict)
        self.service.set_evidence_level('full' if strict else c.config['runtime'].get('evidence_level','full'))
        path=c.session/'collections'/label/f'rank{c.distributed.rank:02d}'
        local_call(c.distributed,lambda:path.mkdir(parents=True,exist_ok=False))
        credits=[collection_credit(n+3,c.stage['ppo_rollout']['world_seconds'],self.latency_budget_s) for n in counts]
        if world_seconds is not None:
            # 全世界时间含reset预热，额外最多一格用于合法边界；失败不凭配额增加执行。
            from .rolling_schedule import decision_ticks
            grid_controls=decision_ticks(c.config)//12
            controls=round(world_seconds*50)+grid_controls
            credits=[dict(control_steps=controls,physics_steps=4*controls,
                generations=(controls+grid_controls-1)//grid_controls+2) for _ in counts]
        total={key:sum(v[key] for v in credits) for key in credits[0]}
        per_rank=c.distributed.all_gather_object(total);lease=f'{c.session.name}/{label}'
        parent=local_call(c.distributed,lambda:begin_lease(c.distributed,c.manager,c.budget,path,lease,per_rank,c.guard))
        self.phase=SimpleNamespace(path=path,credits=credits,per_rank=per_rank,parent=parent,lease=lease,world_seconds=world_seconds)
        self.collector.world.journal.close();self.collector.world.journal=self._journal(path/'world_journal.sqlite')
        self.collector.training_storage='durable_cpu.v1' if self.strict else 'ephemeral_device'
        for slot,state in enumerate(self.collector.states):
            if state is not None:self._bind(slot,state.resource)

    def _finish(self):
        c,phase=self.c,self.phase
        used={key:0 for key in phase.per_rank[c.distributed.rank]}
        for state in self.collector.states:
            if state is not None:
                for key in used:used[key]+=state.resource.env.budget.used[key]
                state.resource.env.budget.close();state.resource.journal.close()
        self.collector.world.journal.close()
        billed=dict(used)
        if phase.world_seconds is not None:
            allocated_controls=self.num_envs*self.collector.window_record['actual_world_controls']
            if used['control_steps']>allocated_controls:raise RuntimeError('Lane controls exceed the real allocated world interval')
            billed.update(control_steps=allocated_controls,physics_steps=4*allocated_controls)
        self.last_physics_accounting=dict(contract='all_allocated_PhysX_slots.v1',
            budgeted_control_slots=billed['control_steps'],allocated_physics_steps=billed['physics_steps'],
            inactive_physics_steps=billed['physics_steps']-used['physics_steps'],
            actual_GMT_controls=used['control_steps'],actual_GMT_physics_steps=used['physics_steps'])
        phase.parent.reserve('confirmed_allocated_physical_execution',**billed)
        state=finish_lease(c.distributed,c.budget,phase.parent,phase.lease,phase.per_rank)
        self.phase=None
        return used,state

    def collect(self,label,*,strict=False):
        c=self.c;seconds=c.stage['ppo_rollout']['world_seconds']
        self._open(label,[0]*self.num_envs,strict,world_seconds=seconds)
        self.apply_restore()
        start=time.perf_counter()
        fragments,report=local_call(c.distributed,lambda:self.collector.collect(world_seconds=seconds,policy_version=c.state['policy_version']))
        def prepare_learning():
            buffer=DeviceRolloutBuffer(fragments,c.policy,c.distributed.device)
            rows=buffer.transitions
            targets=freeze_buffer_targets(buffer,buffer.fragments,c.critic,c.distributed.device,
                critic_version=c.state['critic_updates'],batch_size=256,gamma_upper=c.settings['gamma_upper'],
                lambda_upper=c.settings['lambda_upper'])
            return buffer,targets
        buffer,targets=local_call(c.distributed,prepare_learning)
        rows=buffer.transitions
        from .joint_limits import summarize_boundaries
        report['joint_limits']=summarize_boundaries(
            [d['joint_limit_diagnostics'] for r in rows for d in r.metadata['reward_details']],
            generated=[r.metadata['generated']['joint_limit_diagnostics'] for r in rows])
        counts=torch.stack([row.free_mask.sum() for row in rows]).cpu().tolist() if rows else []
        local=[dict(owner_rank=c.distributed.rank,local_index=i,valid=True,has_free=n>0) for i,n in enumerate(counts)]
        manifest=[item for shard in c.distributed.all_gather_object(local) for item in shard]
        if manifest:targets=normalize_advantages_global(targets,distributed=c.distributed)
        else:targets.update(advantage_global_count=0,advantage_global_mean=0.,advantage_global_std=0.)
        used,budget=self._finish()
        contiguous=[]
        for fragment in fragments:
            length=0;previous=None
            for row in fragment:
                continues=previous is not None and not previous.terminated and not previous.truncated and (
                    previous.identity['episode_id']==row.identity['episode_id'] and
                    previous.control_tick_end==row.control_tick_begin)
                if length and not continues:contiguous.append(length);length=0
                length+=1;previous=row
            if length:contiguous.append(length)
        samplers = [state.resource.sampler for state in self.collector.states if state is not None]
        report.update(time_sampling=self.time_pool.summary(),
            episode_boundaries=episode_boundary_summary(rows),
            sampling_coverage=execution_coverage(c.catalog, samplers),
            through_targets_and_budget_seconds=time.perf_counter()-start,
            physics_accounting=self.last_physics_accounting,
            actual_world_allocated_physics_steps=self.last_physics_accounting['allocated_physics_steps'],
            control_steps=sum(r.executed_control_steps for r in rows),physics_steps=sum(r.executed_physics_steps for r in rows),
            actual_confirmed_control_steps=used['control_steps'],actual_confirmed_physics_steps=used['physics_steps'],
            non_rollout_control_steps=used['control_steps']-sum(r.executed_control_steps for r in rows),
            reward_sum=sum(float(r.rewards.sum()) for r in rows),
            executed_seconds=sum(r.executed_control_steps for r in rows)*.02,
            physical_failures=sum(bool(r.metadata.get('terminal_snapshot',{}).get('terminated',False)) for r in rows),
            behavior_version=c.state['policy_version'],strict_evidence=bool(strict),resource_used=used,
            tensor_storage='GPU_until_optimizer_and_final_KL_complete',
            gae_variance=targets['advantage_global_std']**2,gae_contiguous_lengths=contiguous,
            advantage_control=targets.get('advantage_control','none'),by_source=_source_summary(rows))
        if strict:
            from .rollout_storage import BlockRolloutWriter
            writer=BlockRolloutWriter(self.c.session/'collections'/label/f'rank{c.distributed.rank:02d}'/'rollout',
                policy_version=c.state['policy_version'],chunk_size=64,disk_guard=c.guard)
            for row in rows:writer.append(row)
            writer.finish()
        if c.stage['ppo_rollout'].get('credit_contract'):
            path=c.session/'collections'/label/f'rank{c.distributed.rank:02d}'/'reference_credit.json'
            updates=[dict(env_slot=i,**item) for i,state in enumerate(self.collector.states)
                for item in state.resource.env.reference_credit.drain_updates()]
            atomic_json(path,dict(contract=c.stage['ppo_rollout']['credit_contract'],
                behavior_version=c.state['policy_version'],tick_hz=600,reward_reassignment=False,plans=updates))
            c.guard.account_file(path)
            report['reference_credit']=dict(path=str(path),plan_updates=len(updates),
                gamma_upper=c.settings['gamma_upper'],lambda_upper=c.settings['lambda_upper'])
        atomic_json(c.session/'collections'/label/f'rank{c.distributed.rank:02d}'/'collection.json',report)
        self.iteration,self.policy_version=c.state['iteration'],c.state['policy_version']
        return buffer,targets,manifest,report

    def calibrate(self,*,warmup=2,samples=8):
        if self.num_envs!=1:raise ValueError('Deployment calibration requires one real robot request per rank')
        self._open('deployment_calibration',[warmup+samples],False)
        saved_pool = self.time_pool.state_dict()
        try:
            result=local_call(self.c.distributed,lambda:self.collector.calibrate(
                count_per_rank=warmup+samples,warmup=warmup,samples=samples))
        finally:
            self.time_pool.load_state_dict(saved_pool)
        used,_=self._finish()
        result.update(actual_resource_used=used,target='single_robot_single_request',
            actual_environment_count=1,numerical_execution=self.c.policy.numerical_execution,
            kernel=self.c.policy.kernel_config,assets=self.c.identity['assets'],
            measurements_are_training_wall_not_used_as_current_simulated_clock=True)
        return result

    def checkpoint_state(self):
        if self.collector.active:raise RuntimeError('Checkpoint requires an idle world')
        from gem.closedloop.dppo.execution_state import capture_execution_state
        records=[]
        for state in self.collector.states:
            if state is None:records.append(None);continue
            env=state.resource.env
            records.append(dict(execution=dict(capture_execution_state(env),policy_version=env.policy_version,iteration=env.iteration),
                sampler=state.resource.sampler.state_dict(),transitions=state.transitions,
                ended_on_restore_episode=None if state.task is None else state.task['sample']['row']['sample_id']))
        return dict(schema='genmo.stage10.inprocess_collector.time_sampling.v1',states=records,
            time_pool=self.time_pool.state_dict(),
            generation_horizon_frames=self.c.stage['generation_horizon']['frames'],
            ppo_rollout=copy.deepcopy(self.c.stage['ppo_rollout']),
            environment_count=self.num_envs,restore='fresh_PhysX_episodes_no_partial_rollout_reuse')

    def restore_logical(self,saved):
        if saved.get('schema')!='genmo.stage10.inprocess_collector.time_sampling.v1' or saved['environment_count']!=self.num_envs:
            raise ValueError('Old vector collector is not a 当前 complete restore')
        if saved.get('time_pool', {}).get('identity') != self.time_pool.identity:
            raise ValueError('Shared time sampling changed; create a new training identity')
        self.saved_logical=saved

    def apply_restore(self):
        if not hasattr(self,'saved_logical'):return
        from gem.closedloop.dppo.execution_state import restore_execution_state
        self.time_pool.load_state_dict(self.saved_logical['time_pool'])
        for slot,record in enumerate(self.saved_logical['states']):
            if record is None:continue
            resource=self.collector._resource(slot).resource
            resource.sampler.load_state_dict(record['sampler'])
            restore_execution_state(resource.env,record['execution'])
            self.collector.states[slot].transitions=record['transitions']
            self.collector.states[slot].task=None
        del self.saved_logical

    def verify_frozen(self):return self.collector.world_call('verify_frozen')

    def close(self):
        # 正常关闭无需写新的普通journal；尚未确认的数据仍由服务的关闭验收发现。
        self.collector.world.journal=MemoryReceiveJournal(self.c.rank_dir/'shutdown_memory')
        for state in self.collector.states:
            if state is not None:state.resource.journal=MemoryReceiveJournal(self.c.rank_dir/'shutdown_lane')
        return self.collector.close()
