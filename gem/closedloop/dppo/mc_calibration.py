"""独立完整音乐物理轨迹的 Critic 预热、MC/TD 对照与周期价值校准。

训练世界暂停并保留原状态，每卡临时使用一个独立 GMT/Isaac 世界。固定当前 Actor
收集 train 和 val 的完整 episode，保存因果条件与全部积分奖励，真实终止 MC 尾值
为零。训练 bank 用于价值头预热，验证 bank 只评估独立 MC 误差；比例候选从同一
价值权重、优化器和随机状态起点开始，根据独立 MSE 选择，实际选中权重保存。
Actor 正式更新后不复用旧 MC 数据。周期校准默认只测当前 Actor 的新完整轨迹偏差，
不改 Actor，也不解冻 Critic 预训练编码器。
"""
from __future__ import annotations
import copy
import math
import time
import torch
from .budget import atomic_json
from .complete_episode import CompleteEpisodeBank
from .evaluation import build_evaluation_tasks, evaluate_policy, _digest, _model_fingerprint
from .env_adapter import UpperEnvironment
from .parallel_support import capture_local_rng, restore_local_rng, local_call, root_call, begin_lease, finish_lease
from .run_management import GuardedStepJournal
from .memory_execution import MemoryReceiveJournal
from .training_checkpoint import immutable_cpu_snapshot
from .updater import critic_update
from .trainer import populate_values, fixed_targets
from .periodic_monitor import summarize_critic_predictions


class ValueConditionCache:
    def __init__(self, rows, device):self.rows=rows;self.device=torch.device(device)
    def context(self, rows):
        return {k:torch.cat([r.context[k] for r in rows]).to(self.device) for k in rows[0].context}
    def critic_context(self,rows):
        return {key:torch.cat([r.metadata['critic_context'][key] for r in rows]).to(self.device)
                for key in rows[0].metadata['critic_context']}

    def get(self, key, rows):
        if key!='remaining_music_seconds':raise KeyError(key)
        return torch.tensor([r.metadata[key] for r in rows],device=self.device,dtype=torch.float32)


def mc_execution_counts(used,world_execution=None):
    """学习时钟只累计真实GMT控制；退役slot占用的PhysX工作量仍完整进入资源账本。"""
    active=used['control_steps'] if world_execution is None else world_execution['actual_GMT_controls']
    if (used['physics_steps']!=4*used['control_steps'] or not 0<=active<=used['control_steps']
            or world_execution is not None and world_execution['allocated_physics_steps']!=used['physics_steps']):
        raise ValueError('Complete MC active/allocated physical counts disagree')
    return [active,4*active,used['physics_steps']]


def complete_mc_plan(catalog, split, *, samples_per_source, seeds, selection_seed):
    """两个bank均按独立音乐选最长可用配对样本；验证规模允许显式分来源配额。"""
    if split!='train':
        from .periodic_monitor import build_balanced_plan
        return build_balanced_plan(catalog,samples_per_source=samples_per_source,seeds=seeds,
            selection_seed=selection_seed,seconds=None,unique_music=True,split=split)
    selected=catalog
    if split=='train':
        from gem.closedloop.evaluation_music import select_train_music
        selected=copy.copy(catalog)
        selected.samples={'train':{source:[] for source in catalog.samples['train']}}
        for sample in select_train_music(catalog.data_root,sorted(catalog.samples['train']),
                samples_per_source,seed=selection_seed):
            source=sample['dataset'];identifier=sample['row']['sample_id']
            audited=next((s for s in catalog.samples['train'][source] if s['row']['sample_id']==identifier),None)
            if audited is None or audited['row']!=sample['row'] or audited['group_id']!=sample['group_id']:
                raise ValueError('MC unique music selection differs from audited training catalog')
            selected.samples['train'][source].append(audited)
    plan=build_evaluation_tasks(selected,split=split,eval_count=samples_per_source*len(selected.samples[split]),
        balanced_samples_per_source=samples_per_source,seeds=seeds,selection_seed=selection_seed,
        episode_seconds=None,critic_mc=split=='train')
    if split=='train':
        plan['selection_unit']='unique_global_music_group_and_audio_longest_paired_sample'
        plan['selection_rule']='seeded_group_longest_unique_global_group_and_audio_sha256'
        plan['plan_sha256']=_digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    return plan



def matched_reset_mc_plan(plan, stage):
    """匹配训练的站立/配对reset比例与合法时间bin，固定独立音乐及seed至真实终点。

    这是选定起点后的完整episode MC，不宣称从歌曲开头完整播放；整曲质量评估仍
    保持原固定32任务开头启动。MC任务独立采样，绝不污染训练无放回采样池。
    """
    from .time_sampling import MOTION_RESET_CONTRACT
    from .evaluation import stable_noise_seed
    import numpy as np
    plan=copy.deepcopy(plan);dataset=stage['dataset']
    plan['mc_start_distribution']='match_training_reset.v1'
    plan['start_mode']='training_reset_mixture'
    for task in plan['tasks']:
        rng=np.random.default_rng(stable_noise_seed(stage['seed']+937,task['task_id']))
        frames=int(task['full_music_frames'])
        maximum=max(0,frames-max(2,math.ceil(stage['ppo_rollout']['world_seconds']*30)))
        if rng.random()<dataset['standing_start_probability']:
            start=0;initialization='standing'
        else:
            width=round(dataset['time_bin_seconds']*30)
            begin=int(rng.integers(maximum//width+1))*width
            start=int(rng.integers(begin,min(begin+width,maximum+1)))
            initialization=MOTION_RESET_CONTRACT
        task.update(music_start_frame=start,remaining_music_seconds=(frames-start)/30.,initialization=initialization)
        identity={k:task[k] for k in ('dataset','sample_id','split','seed','music_start_frame','initialization')}
        task['task_id']=_digest(identity)[:24]
        task['noise_index_start']=stable_noise_seed(task['seed'],_digest(identity))
    plan['plan_sha256']=_digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    return plan


def collect_complete_bank(c, label, split, *, samples_per_source, seeds):
    from .vector_evaluation import isolated_vector_evaluation,isolated_env_count
    actor_before=_model_fingerprint(c.actor);signature=c.policy._parameter_signature()
    bank=CompleteEpisodeBank(policy_version=c.state['policy_version'],kernel_identity=c.policy.kernel_config,
        gamma_upper=c.settings['gamma_upper'])
    plan=root_call(c.distributed,lambda:complete_mc_plan(c.catalog,split,
        samples_per_source=samples_per_source,seeds=seeds,selection_seed=c.stage['seed']+419))
    if c.stage['critic']['complete_mc'].get('start_distribution')=='match_training_reset.v1':
        plan=matched_reset_mc_plan(plan,c.stage)
    nseed=len(seeds)
    from .evaluation_world import balanced_rank_tasks
    task_shards=balanced_rank_tasks(plan,c.distributed.world_size)
    shards=[]
    for rank in range(c.distributed.world_size):
        child=copy.deepcopy(plan)
        child['tasks']=task_shards[rank]
        child['task_count']=len(child['tasks']);child['selected_sample_count']=len(child['tasks'])//nseed
        child['plan_sha256']=_digest({k:v for k,v in child.items() if k!='plan_sha256'})
        shards.append(child)
    credits=[]
    for child in shards:
        # 每首独立reset预热50控制步；真实轨迹期间每个动作至少一个合法决策间隔。
        controls=sum(math.ceil(t['remaining_music_seconds']*50)+75 for t in child['tasks'])
        generations=sum(math.ceil(t['remaining_music_seconds']/.5)+5 for t in child['tasks'])
        credits.append(dict(control_steps=max(1,controls),physics_steps=4*max(1,controls),generations=max(1,generations)))
        for key in ('control_steps','physics_steps'):credits[-1][key]*=isolated_env_count(c,'mc_'+label)
    path=c.session/'mc_calibration'/label/f'rank{c.distributed.rank:02d}'
    path.mkdir(parents=True,exist_ok=False);lease=f'{c.session.name}/mc/{label}'
    prepaid=begin_lease(c.distributed,c.manager,c.budget,path,lease,credits,c.guard)
    saved_rng=capture_local_rng(c.generators)
    try:
        with isolated_vector_evaluation(c,'mc_'+label):
            journal=MemoryReceiveJournal(path/'execution.sqlite')
            c.backend.journal=journal
            cfg=copy.deepcopy(c.config);cfg['stage10']['physical_episode']['safety_limit_seconds']=None
            cfg['runtime']['evidence_level']='full'
            cfg['runtime']['evaluation_metrics_only']=True
            from .vector_environment import VectorUpperEnvironment
            from gem.closedloop.online_conditions import OnlineConditionBuilder
            driver=None
            if getattr(getattr(c,'active_evaluation_world',None),'num_envs',1)>1:
                from .evaluation_world import EvaluationWorldDriver
                driver=EvaluationWorldDriver(c,cfg,prepaid,path);env=driver.env
            else:env=VectorUpperEnvironment(cfg,c.backend,c.builder,c.policy,prepaid,path)
            env.disk_guard=c.guard;env.latency_budget_s=c.vector.latency_budget_s
            env.policy_version=c.state['policy_version'];env.iteration=c.state['iteration']
            def capture(task,row):
                if signature!=c.policy._parameter_signature():raise RuntimeError('Actor changed during complete MC sampling')
                bank.append(task,row)
            try:
                def run():
                    if not shards[c.distributed.rank]['tasks']:return None
                    return evaluate_policy(env,c.policy,shards[c.distributed.rank],path/'report',catalog=c.catalog,
                        actor_identity=dict(model_fingerprint=actor_before,policy_version=c.state['policy_version']),
                        frozen_modules={'critic':c.critic},local_cuda_only=True,transition_sink=capture,world_driver=driver)
                evaluation_report=local_call(c.distributed,run)
                if evaluation_report is not None:
                    import json
                    for entry in evaluation_report['episode_manifests']:
                        episode=json.loads((path/'report'/entry['path']).read_text())
                        if episode['end_reason']=='physical_warmup_failure':
                            bank.excluded.append(dict(task_id=episode['task_id'],reason='physical_warmup_failure',
                                states=0,warmup_control_steps=episode['warmup_control_steps']))
            finally:
                try:
                    if driver is not None:driver.close()
                finally:journal.close()
        if _model_fingerprint(c.actor)!=actor_before:raise RuntimeError('MC evaluation changed Actor weights')
        local_used=prepaid.state_dict()['used']
        c.state['budget']=finish_lease(c.distributed,c.budget,prepaid,lease,credits)
        execution_counts=mc_execution_counts(local_used,
            None if driver is None or not shards[c.distributed.rank]['tasks'] else driver.execution_summary)
        consumed=c.distributed.sum_tensor(torch.tensor(execution_counts,
            device=c.distributed.device,dtype=torch.int64)).cpu().tolist()
        field='mc_training' if split=='train' else 'mc_validation'
        c.state[field+'_control_steps']=c.state.get(field+'_control_steps',0)+consumed[0]
        c.state[field+'_physics_steps']=c.state.get(field+'_physics_steps',0)+consumed[1]
        c.state[field+'_allocated_physics_steps']=c.state.get(field+'_allocated_physics_steps',0)+consumed[2]
        c.state[field+'_inactive_physics_steps']=c.state.get(field+'_inactive_physics_steps',0)+consumed[2]-consumed[1]
        if split=='train':
            c.state['training_control_steps']+=consumed[0]
            c.state['training_physics_steps']+=consumed[1]
        bank.training_data(policy_version=c.state['policy_version'],kernel=c.policy.kernel_config)
        if c.stage['critic']['complete_mc'].get('retain_dataset',False):
            bank_record=dict(schema='genmo.stage2.causal_tracking.complete_episode.value_dataset.v1',report=bank.report(),
                execution_work=dict(actual_GMT_controls=execution_counts[0],active_physics_steps=execution_counts[1],
                    allocated_physics_steps=execution_counts[2],inactive_physics_steps=execution_counts[2]-execution_counts[1]),
                actor_fingerprint=actor_before,policy_version=c.state['policy_version'],kernel=c.policy.kernel_config,
                reward_configuration=c.stage['reward'],plan=plan,
                episodes=[dict(task=e['task'],end_reason=e['end_reason'],control_steps=e['control_steps'],
                    states=len(e['rows'])) for e in bank.episodes],
                rows=[dict(context=r.context,metadata=r.metadata,identity=r.identity,rewards=r.rewards,
                    executed_control_steps=r.executed_control_steps,begin=r.control_tick_begin,end=r.control_tick_end,
                    terminated=r.terminated,truncated=r.truncated)
                    for e in bank.episodes for r in e['rows']],
                returns=torch.cat([e['returns'] for e in bank.episodes]) if bank.episodes else torch.empty(0))
            torch.save(bank_record,path/'complete_mc.pt')
            c.guard.account_file(path/'complete_mc.pt')
        return bank
    finally:restore_local_rng(saved_rng,c.generators)


@torch.no_grad()
def independent_mc_metrics(c, bank):
    rows,targets=bank.training_data(policy_version=c.state['policy_version'],kernel=c.policy.kernel_config)
    predictions=[];c.critic.eval()
    for start in range(0,len(rows),128):
        part=rows[start:start+128];cache=ValueConditionCache(part,c.distributed.device)
        predictions.extend(c.critic(cache.critic_context(part),cache.get('remaining_music_seconds',part)).cpu().tolist())
    modes=[e['task'].get('initialization','standing') for e in bank.episodes for _ in e['rows']]
    from .value_credit_diagnostics import episode_value_records, summarize_value_credit
    local_records=[];offset=0
    for episode in bank.episodes:
        count=len(episode['rows'])
        local_records.extend(episode_value_records(episode,predictions[offset:offset+count],
            gamma_upper=c.settings['gamma_upper'],lambda_upper=c.settings['lambda_upper'],
            world_seconds=c.stage['ppo_rollout']['world_seconds']))
        offset+=count
    shards=c.distributed.all_gather_object(dict(predictions=predictions,targets=targets.tolist(),bank=bank.report(),modes=modes,
        value_credit_records=local_records))
    p=[v for r in shards for v in r['predictions']];t=[v for r in shards for v in r['targets']]
    if not t:raise ValueError('No complete nonadministrative MC validation episode')
    modes=[v for r in shards for v in r['modes']]
    by_initialization={mode:summarize_critic_predictions([v for v,m in zip(p,modes) if m==mode],
        [v for v,m in zip(t,modes) if m==mode]) for mode in sorted(set(modes))}
    return dict(**summarize_critic_predictions(p,t),by_initialization=by_initialization,
        value_credit=summarize_value_credit([v for r in shards for v in r['value_credit_records']]),
        episodes=sum(r['bank']['complete_episodes'] for r in shards),
        physical_control_steps=sum(r['bank']['real_control_steps'] for r in shards),
        scope='independent_complete_MC_current_fixed_behavior_no_bootstrap')


def calibrate_critic(c, label, *, fit, fit_if_ev_below=None):
    """完整MC价值校准；fit_if_ev_below只用于周期调用：独立验证EV低于阈值才拟合。

    周期重拟合在独立训练种子bank上训练候选，用验证bank选最优；最优候选验证MSE不优于
    拟合前时恢复原Critic/Adam，不计入critic_updates。预热(fit=True)语义不变。
    """
    started=time.perf_counter();cfg=c.stage['critic']['complete_mc']
    validation=collect_complete_bank(c,label+'_val','val',samples_per_source=cfg['samples_per_source'],seeds=cfg['validation_seeds'])
    before=independent_mc_metrics(c,validation)
    periodic_refit=False
    if not fit and fit_if_ev_below is not None:
        periodic_refit=fit=before['explained_variance'] is None or before['explained_variance']<fit_if_ev_below
    report=dict(before=before,after=before,candidates=[],behavior_version=c.state['policy_version'],fit=fit,
        refit_threshold=fit_if_ev_below,periodic_refit=periodic_refit)
    if fit:
        training=collect_complete_bank(c,label+'_train','train',
            samples_per_source=cfg.get('training_samples_per_source',cfg['samples_per_source']),seeds=cfg['training_seeds'])
        rows,mc=training.training_data(policy_version=c.state['policy_version'],kernel=c.policy.kernel_config)
        # TD/GAE以同一初始Critic为基准；真实终止没有自举，候选共享固定标签与样本。
        td=[]
        for episode in training.episodes:
            part=episode['rows'];populate_values(part,c.critic,c.distributed.device,critic_version=c.state['critic_updates'],batch_size=128)
            begin=0
            for end in range(1,len(part)+1):
                if end==len(part) or part[end-1].control_tick_end-part[begin].control_tick_begin>=c.stage['ppo_rollout']['world_seconds']*600:
                    td.append(fixed_targets(part[begin:end],c.critic,c.distributed.device,reuse_values=True,critic_version=c.state['critic_updates'],
                        gamma_upper=c.settings['gamma_upper'],lambda_upper=c.settings['lambda_upper'],normalize=False)['returns'])
                    begin=end
        td=torch.cat(td) if td else torch.empty(0,dtype=torch.float64)
        manifest=[item for shard in c.distributed.all_gather_object([
            dict(owner_rank=c.distributed.rank,local_index=i,valid=True,has_free=True) for i in range(len(rows))]) for item in shard]
        initial=immutable_cpu_snapshot(c.critic.state_dict());optimizer=immutable_cpu_snapshot(c.critic_optimizer.state_dict())
        rng=capture_local_rng(c.generators);best=None;best_state=None;best_optimizer=None
        for ratio in cfg['mix_candidates']:
            candidate_start=time.perf_counter()
            c.critic.load_state_dict(initial);c.critic_optimizer.load_state_dict(optimizer);restore_local_rng(rng,c.generators)
            result=critic_update(c.critic,c.critic_optimizer,rows,dict(returns=(1-ratio)*td+ratio*mc),
                ValueConditionCache(rows,c.distributed.device),manifest,distributed=c.distributed,
                generator=c.generators['critic'],epochs=cfg['epochs'],batch=c.stage['critic']['global_batch'])
            metrics=independent_mc_metrics(c,validation)
            candidate=dict(mc_ratio=ratio,validation=metrics,training=result,
                wall_seconds=max(c.distributed.all_gather_object(time.perf_counter()-candidate_start)))
            report['candidates'].append(candidate)
            if best is None or metrics['mse']<best['validation']['mse']:
                best=candidate;best_state=immutable_cpu_snapshot(c.critic.state_dict());best_optimizer=immutable_cpu_snapshot(c.critic_optimizer.state_dict())
        accepted=not periodic_refit or best['validation']['mse']<before['mse']
        if accepted:
            c.critic.load_state_dict(best_state);c.critic_optimizer.load_state_dict(best_optimizer)
            c.state['critic_updates']+=best['training']['optimizer_steps']
        else:
            c.critic.load_state_dict(initial);c.critic_optimizer.load_state_dict(optimizer)
        report['refit_accepted']=accepted
        report.update(selected_mc_ratio=best['mc_ratio'] if accepted else None,
            candidate_after=best['validation'],after=best['validation'] if accepted else before,
            attempted_critic_steps=sum(r['training']['optimizer_steps'] for r in report['candidates']))
        report['relative_validation_mse_reduction']=1-report['after']['mse']/max(before['mse'],1e-12)
        gate=cfg.get('quality_gate',{})
        report['quality_gate_passed']=(report['relative_validation_mse_reduction']>=gate.get('minimum_relative_mse_reduction',0.)
            and report['after']['explained_variance'] is not None
            and report['after']['explained_variance']>=gate.get('minimum_explained_variance',0.))
        training.episodes.clear();validation.episodes.clear()
    report['wall_seconds']=max(c.distributed.all_gather_object(time.perf_counter()-started))
    def publish():
        atomic_json(c.session/'mc_calibration'/f'{label}.json',report)
        if getattr(c,'writer',None) is not None:
            from .tensorboard_metrics import mc_scalars
            c.writer.write('critic_mc',c.state['iteration'],mc_scalars(report))
            c.writer.flush()
    root_call(c.distributed,publish)
    return report
