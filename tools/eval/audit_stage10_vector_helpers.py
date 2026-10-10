"""GPU多环境训练产物的独立身份、计数和完整采样池审计。

旧八卡审计每rank只有一套decision及backend_session；GPU共享世界必须逐环境绑定
lane journal、真实生成批次和checkpoint中的采样器。此模块只读取已校验SHA的产物，
不修改历史、不创建训练状态、不运行模型或物理。不同环境的相同decision编号合法，
同一环境的编号重复或跨lane引用必须失败。恢复状态必须包括所有环境的音乐全池
排列、游标、执行计数和显式参考边界合同，不能拿占位的rank级环境状态替代。
这些检查补充原全局概率、GAE、KL、Adam、预算和归档审计，不降低原数值门槛。
"""
from collections import Counter
from tools.eval.audit_closedloop_stage10 import require,integer,number,close


def audit_boundary_wait_metadata(item, identity):
    """独立验证授权尾段的精确控制/物理和逐步奖励，不以汇总标签替代原始转移。"""
    import torch
    runtime = identity.get('execution_contract', {}).get('runtime', {})
    tail = item.metadata.get('fragment_tail')
    if runtime.get('vector_boundary_wait_contract') is None or tail is None:
        return None
    maximum = runtime['vector_boundary_wait_max_control_steps']
    require(runtime['vector_boundary_wait_contract']=='bounded_reference_wait.v1'
            and tail.get('boundary_wait_contract')==runtime['vector_boundary_wait_contract']
            and tail.get('wait_limit_controls')==maximum, 'Boundary wait contract or cap differs')
    count = integer(tail['executed_control_steps'], 'Boundary wait controls')
    reserved = integer(tail['budget_reserved_controls'], 'Boundary wait reservation')
    require(count<=reserved<=maximum and reserved<=tail['requested_control_steps']
            and tail['executed_physics_steps']==4*count and count<=item.executed_control_steps,
            'Boundary wait control/physics/budget count differs')
    require(tail['begin_tick']>=item.control_tick_begin and tail['end_tick']==item.control_tick_end
            and tail['end_tick']-tail['begin_tick']==12*count,
            'Boundary wait physical interval differs')
    supported = tail.get('maximum_supported_decision_tick')
    require(supported is None or tail['end_tick']<=supported, 'Boundary wait exceeded valid reference')
    integer(tail['execution_sequence'], 'Boundary wait acknowledged sequence', 1)
    require(number(tail['wait_wall_seconds'], 'Boundary wait walltime')>=0,
            'Boundary wait walltime invalid')
    require(type(tail['wait_limit_reached']) is bool and
            (not tail['wait_limit_reached'] or count==reserved==maximum), 'Boundary wait cap flag differs')
    values = []
    if count:
        details = item.metadata['reward_details'][-count:]
        require(len(details)==count and all(d.get('transition_valid') is True for d in details),
                'Boundary wait step reward evidence missing')
        values = [number(d['reward'], 'Boundary wait step reward') for d in details]
        # 正控制步尾段从未终止状态开始；若末状态物理终止，原奖励在最后控制步加一次失败惩罚。
        if item.metadata['terminal_snapshot'].get('terminated'):
            values[-1] -= identity['reward']['failure_penalty']
        require(torch.equal(torch.tensor(values, dtype=item.rewards.dtype), item.rewards[-count:]),
                'Boundary wait rewards differ from stored actual control rewards')
    close(tail['reward_sum'], sum(values), 'Boundary wait exact step reward sum')
    return dict(tail)


def is_vector(identity):
    return identity.get('execution_contract',{}).get('runtime',{}).get('backend')=='gpu_vectorized.v1'


def audit_modeled_clock(metadata, task, begin, identity):
    """从原配置/任务独立重建部署延迟，拒绝墙钟、拓扑或持久化污染模拟时间。"""
    from gem.closedloop.dppo.deployment_clock import MODELED_CLOCK,DeploymentClock
    contract=identity['execution_contract']
    if contract['runtime'].get('timing_contract')!=MODELED_CLOCK:return
    clock=DeploymentClock(contract['timing']['deployment_profile'])
    expected=clock.sample(seed=identity['base_seed'],sample_id=task['sample_id'],
        music_start_frame=task['music_start_frame'],decision_tick=begin)
    require(metadata.get('timing_contract')==MODELED_CLOCK and
            metadata.get('timing',{}).get('deployment_clock')==expected,
            'Modeled deployment clock differs from immutable task/profile identity')
    generated=metadata['generated']
    require(generated['deadline_tick']<=begin+clock.budget_ticks,
            'Training wallclock increased modeled prefix budget')


def lane_sessions(frozen, count):
    lanes=frozen.get('lane_journals',[])
    require(len(lanes)==count and frozen.get('pending_env_ids')==[], 'Vector lanes missing or still executing')
    require(frozen.get('gmt_parameters_frozen') is True and
            frozen.get('actual_module_sha256')==frozen.get('initial_module_sha256') and
            bool(frozen.get('actual_module_sha256')), 'Vector frozen module differs')
    ids=[frozen['execution_journal']['backend_session_id']]
    for lane in lanes:
        require(lane['executed_seq']==lane['acked_seq'] and lane.get('outstanding_seq') is None,
                'Vector lane has unacknowledged execution')
        ids.append(lane['backend_session_id'])
    require(len(set(ids))==count+1, 'Vector lane/world sessions are not independent')
    return ids[1:]


def audit_vector_rows(rows, collection, frozen, identity):
    runtime=identity['execution_contract']['runtime'];count=runtime['num_envs']
    total=identity['training_contract']['rollout_upper_steps_per_rank']
    quotas=[total//count+(slot<total%count) for slot in range(count)]
    world = runtime.get('vector_collection_contract') == 'genmo.world_batched_flow.v1'
    require(collection.get('schema')==('genmo.world_batched_flow.v1' if world else 'genmo.gpu_vector_collector.v1') and
            collection.get('real_environment_batch') is True and collection.get('allocated_envs')==count and
            collection.get('active_envs')==count and collection.get('total_transitions')==len(rows)==total and
            collection.get('fragment_lengths')==quotas and
            collection.get('fragment_contract')==(runtime['vector_collection_contract'] if world else runtime['vector_fragment_contract']), 'Vector rollout topology differs')
    bounded = world and runtime.get('vector_boundary_wait_contract') is not None
    if world:
        require(collection.get('normal_boundary_resets')==0, 'World rollout reset healthy robots at boundary')
        if bounded:
            require(collection.get('boundary_wait_contract')==runtime['vector_boundary_wait_contract']
                    and collection.get('boundary_wait_max_controls')==runtime['vector_boundary_wait_max_control_steps'],
                    'World bounded wait identity differs')
        else:
            require(collection.get('administrative_drain_controls')==0,
                    'World legacy boundary contract does not authorize drain')
    sessions=lane_sessions(frozen,count)
    begin=0
    tails=[]
    for slot,size in enumerate(quotas):
        fragment=rows[begin:begin+size];begin+=size
        require(all(r['identity'].get('env_id')==slot and
                    r['identity']['backend_session_id']==sessions[slot] for r in fragment),
                'Vector rollout references another environment or lane session')
        decisions=[integer(r['identity']['decision_id'],'Vector decision') for r in fragment]
        require(all(b==a+1 for a,b in zip(decisions,decisions[1:])), 'Vector per-environment decisions are not continuous')
        require(fragment[-1]['terminated'] or fragment[-1]['truncated'], 'Vector GAE fragment is not closed')
        require(all(r['count']>0 for r in fragment), 'Vector rollout counts a zero-control action')
        if bounded:
            require(fragment[-1].get('audited_fragment_tail') is not None and
                    all(r.get('audited_fragment_tail') is None for r in fragment[:-1]),
                    'World bounded wait must be verified exactly once at each environment tail')
            tails.append(fragment[-1]['audited_fragment_tail'])
    if bounded:
        require(collection['administrative_drain_controls']==sum(t['executed_control_steps'] for t in tails)
                and collection['administrative_drain_physics_steps']==sum(t['executed_physics_steps'] for t in tails)
                and collection['administrative_wait_limit_reached']==sum(t['wait_limit_reached'] for t in tails),
                'World bounded wait aggregate hides control/physics/cap work')
        close(collection['administrative_drain_reward'],sum(t['reward_sum'] for t in tails), 'World bounded wait reward aggregate')
        close(collection['administrative_wait_max_wall_seconds'],max(t['wait_wall_seconds'] for t in tails),
              'World bounded wait walltime aggregate')
    generated=Counter()
    for batch in collection['batches']:
        slots=batch['environment_slots']
        require(len(slots)==len(set(slots))==batch['effective_rows'] and batch['padding_rows']==0 and
                all(type(slot)is int and 0<=slot<count for slot in slots), 'Vector generation batch is padded or mixes identities')
        generated.update(slots)
    require(generated==Counter(dict(enumerate(quotas))), 'Vector real requests do not match saved chains')


def audit_vector_checkpoint(local, rows, identity, rank):
    runtime=identity['execution_contract']['runtime'];count=runtime['num_envs']
    saved=local.get('vector_collector',{})
    world = runtime.get('vector_collection_contract') == 'genmo.world_batched_flow.v1'
    require(saved.get('schema')==('genmo.gpu_vector_collector.boundary.v2' if world else 'genmo.gpu_vector_collector.boundary.v1') and saved.get('num_envs')==count and
            saved.get('fragment_contract')==(runtime['vector_collection_contract'] if world else runtime['vector_fragment_contract']) and
            saved.get('numerical_layout')==identity['performance_contract']['numerical_layout'] and
            saved.get('restore_environment')=='fresh_PhysX_episodes_preserve_rng_cursors_and_spent_budget' and
            len(saved.get('states',[]))==count, 'Vector checkpoint topology or reference contract differs')
    if world and runtime.get('vector_boundary_wait_contract') is not None:
        require(saved.get('boundary_wait_contract')==runtime['vector_boundary_wait_contract']
                and saved.get('boundary_wait_max_controls')==runtime['vector_boundary_wait_max_control_steps'],
                'Vector checkpoint bounded wait contract differs')
    counters=[]
    for slot,record in enumerate(saved['states']):
        require(isinstance(record,dict), 'Vector checkpoint missing an environment')
        sampler=record['sampler'];execution=record['execution']
        require(sampler['split']=='train' and sampler['catalog_identity']==identity['dataset'] and
                sampler['source_probabilities']==identity['sampling']['source_probabilities'] and
                sampler['random_start']==identity['sampling']['random_start'], 'Vector sampler differs from full training pool')
        counts=identity['dataset']['sample_counts']['train']
        require(set(sampler['orders'])==set(counts), 'Vector sampler missing source')
        for source,n in counts.items():
            require(sorted(sampler['orders'][source])==list(range(n)) and
                    0<=sampler['cursors'][source]<=n, 'Vector sampler has an incomplete full-pool permutation')
        own=[r for r in rows if r['identity']['env_id']==slot]
        decision=integer(execution['decision'],'Vector saved decision')
        attempt=integer(execution['attempt'],'Vector saved attempt')
        require(own and decision==own[-1]['identity']['decision_id']+1 and decision<=attempt and
                record['transitions']==local['iteration']*len(own), 'Vector saved counter differs from accepted per-environment rows')
        require(execution['policy_version']==own[-1]['identity']['policy_version'] and
                execution['iteration']==local['iteration']-1, 'Vector saved execution belongs to another rollout version')
        require(record['seed']==(identity['base_seed']+1000003*rank+100003*slot)%2**32,
                'Vector saved environment seed differs')
        require(number(execution['latency_budget_s'],'Vector latency')>0, 'Vector invalid latency budget')
        if runtime.get('timing_contract')=='modeled_deployment.v3':
            from gem.closedloop.dppo.deployment_clock import DeploymentClock
            clock=DeploymentClock(identity['execution_contract']['timing']['deployment_profile'])
            require(execution.get('deployment_profile_sha256')==clock.sha256 and
                    execution['latency_budget_s']==clock.budget_seconds,
                    'Vector checkpoint deployment clock/profile differs')
        counters.append(dict(env_id=slot,decision=decision,attempt=attempt,
            episode_count=integer(execution['episode_count'],'Vector episode count')))
    return dict(rank=rank,environments=counters,full_pool_permutations_verified=True)
