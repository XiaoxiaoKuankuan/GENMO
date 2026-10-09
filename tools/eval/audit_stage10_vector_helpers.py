"""GPU多环境训练产物的独立身份、计数和完整采样池审计。

旧八卡审计每rank只有一套decision及backend_session；GPU共享世界必须逐环境绑定
lane journal、真实生成批次和checkpoint中的采样器。此模块只读取已校验SHA的产物，
不修改历史、不创建训练状态、不运行模型或物理。不同环境的相同decision编号合法，
同一环境的编号重复或跨lane引用必须失败。恢复状态必须包括所有环境的音乐全池
排列、游标、执行计数和显式参考边界合同，不能拿占位的rank级环境状态替代。
这些检查补充原全局概率、GAE、KL、Adam、预算和归档审计，不降低原数值门槛。
"""
from collections import Counter
from tools.eval.audit_closedloop_stage10 import require,integer,number


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
    require(collection.get('schema')=='genmo.gpu_vector_collector.v1' and
            collection.get('real_environment_batch') is True and collection.get('allocated_envs')==count and
            collection.get('active_envs')==count and collection.get('total_transitions')==len(rows)==total and
            collection.get('fragment_lengths')==quotas and
            collection.get('fragment_contract')==runtime['vector_fragment_contract'], 'Vector rollout topology differs')
    sessions=lane_sessions(frozen,count)
    begin=0
    for slot,size in enumerate(quotas):
        fragment=rows[begin:begin+size];begin+=size
        require(all(r['identity'].get('env_id')==slot and
                    r['identity']['backend_session_id']==sessions[slot] for r in fragment),
                'Vector rollout references another environment or lane session')
        decisions=[integer(r['identity']['decision_id'],'Vector decision') for r in fragment]
        require(all(b==a+1 for a,b in zip(decisions,decisions[1:])), 'Vector per-environment decisions are not continuous')
        require(fragment[-1]['terminated'] or fragment[-1]['truncated'], 'Vector GAE fragment is not closed')
        require(all(r['count']>0 for r in fragment), 'Vector rollout counts a zero-control action')
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
    require(saved.get('schema')=='genmo.gpu_vector_collector.boundary.v1' and saved.get('num_envs')==count and
            saved.get('fragment_contract')==runtime['vector_fragment_contract'] and
            saved.get('numerical_layout')==identity['performance_contract']['numerical_layout'] and
            saved.get('restore_environment')=='fresh_PhysX_episodes_preserve_rng_cursors_and_spent_budget' and
            len(saved.get('states',[]))==count, 'Vector checkpoint topology or reference contract differs')
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
