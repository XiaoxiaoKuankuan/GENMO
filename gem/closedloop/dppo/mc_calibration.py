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
from .training_checkpoint import immutable_cpu_snapshot
from .updater import critic_update
from .trainer import populate_values, fixed_targets
from .periodic_monitor import summarize_critic_predictions


class ValueConditionCache:
    def __init__(self, rows, device):self.rows=rows;self.device=torch.device(device)
    def context(self, rows):
        return {k:torch.cat([r.context[k] for r in rows]).to(self.device) for k in rows[0].context}
    def get(self, key, rows):
        if key!='remaining_music_seconds':raise KeyError(key)
        return torch.tensor([r.metadata[key] for r in rows],device=self.device,dtype=torch.float32)


def collect_complete_bank(c, label, split, *, samples_per_source, seeds):
    from .vector_evaluation import isolated_vector_evaluation
    actor_before=_model_fingerprint(c.actor);signature=c.policy._parameter_signature()
    bank=CompleteEpisodeBank(policy_version=c.state['policy_version'],kernel_identity=c.policy.kernel_config,
        gamma_upper=c.settings['gamma_upper'])
    plan=root_call(c.distributed,lambda:build_evaluation_tasks(c.catalog,split=split,eval_count=samples_per_source*4,
        balanced_samples_per_source=samples_per_source,seeds=seeds,selection_seed=c.stage['seed']+419,
        episode_seconds=None,critic_mc=split=='train'))
    nseed=len(seeds)
    shards=[]
    for rank in range(c.distributed.world_size):
        child=copy.deepcopy(plan)
        child['tasks']=[t for i,t in enumerate(plan['tasks']) if (i//nseed)%c.distributed.world_size==rank]
        child['task_count']=len(child['tasks']);child['selected_sample_count']=len(child['tasks'])//nseed
        child['plan_sha256']=_digest({k:v for k,v in child.items() if k!='plan_sha256'})
        shards.append(child)
    credits=[]
    for child in shards:
        # 每首独立reset预热50控制步；真实轨迹期间每个动作至少一个合法决策间隔。
        controls=sum(math.ceil(t['remaining_music_seconds']*50)+75 for t in child['tasks'])
        generations=sum(math.ceil(t['remaining_music_seconds']/.5)+5 for t in child['tasks'])
        credits.append(dict(control_steps=max(1,controls),physics_steps=4*max(1,controls),generations=max(1,generations)))
    path=c.session/'mc_calibration'/label/f'rank{c.distributed.rank:02d}'
    path.mkdir(parents=True,exist_ok=False);lease=f'{c.session.name}/mc/{label}'
    prepaid=begin_lease(c.distributed,c.manager,c.budget,path,lease,credits,c.guard)
    saved_rng=capture_local_rng(c.generators)
    try:
        with isolated_vector_evaluation(c,'mc_'+label):
            journal=GuardedStepJournal(path/'execution.sqlite',c.guard,format='genmo.execution_journal.ndarray.v2')
            c.backend.journal=journal
            cfg=copy.deepcopy(c.config);cfg['stage10']['physical_episode']['safety_limit_seconds']=None
            from .vector_environment import VectorUpperEnvironment
            from gem.closedloop.online_conditions import OnlineConditionBuilder
            env=VectorUpperEnvironment(cfg,c.backend,c.builder,c.policy,prepaid,path)
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
                        frozen_modules={'critic':c.critic},local_cuda_only=True,transition_sink=capture)
                local_call(c.distributed,run)
            finally:journal.close()
        if _model_fingerprint(c.actor)!=actor_before:raise RuntimeError('MC evaluation changed Actor weights')
        finish_lease(c.distributed,c.budget,prepaid,lease,credits)
        bank.training_data(policy_version=c.state['policy_version'],kernel=c.policy.kernel_config)
        bank_record=dict(report=bank.report(),actor_fingerprint=actor_before,plan=plan,
            rows=[dict(context=r.context,metadata=r.metadata,identity=r.identity,rewards=r.rewards,
                executed_control_steps=r.executed_control_steps,begin=r.control_tick_begin,end=r.control_tick_end)
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
        predictions.extend(c.critic(cache.context(part),cache.get('remaining_music_seconds',part)).cpu().tolist())
    shards=c.distributed.all_gather_object(dict(predictions=predictions,targets=targets.tolist(),bank=bank.report()))
    p=[v for r in shards for v in r['predictions']];t=[v for r in shards for v in r['targets']]
    if not t:raise ValueError('No complete nonadministrative MC validation episode')
    return dict(**summarize_critic_predictions(p,t),episodes=sum(r['bank']['complete_episodes'] for r in shards),
        physical_control_steps=sum(r['bank']['real_control_steps'] for r in shards),
        scope='independent_complete_MC_current_fixed_behavior_no_bootstrap')


def calibrate_critic(c, label, *, fit):
    started=time.perf_counter();cfg=c.stage['critic']['complete_mc']
    validation=collect_complete_bank(c,label+'_val','val',samples_per_source=cfg['samples_per_source'],seeds=cfg['validation_seeds'])
    before=independent_mc_metrics(c,validation)
    report=dict(before=before,candidates=[],behavior_version=c.state['policy_version'],fit=fit)
    if fit:
        training=collect_complete_bank(c,label+'_train','train',samples_per_source=cfg['samples_per_source'],seeds=cfg['training_seeds'])
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
            c.critic.load_state_dict(initial);c.critic_optimizer.load_state_dict(optimizer);restore_local_rng(rng,c.generators)
            result=critic_update(c.critic,c.critic_optimizer,rows,dict(returns=(1-ratio)*td+ratio*mc),
                ValueConditionCache(rows,c.distributed.device),manifest,distributed=c.distributed,
                generator=c.generators['critic'],epochs=cfg['epochs'],batch=c.stage['critic']['global_batch'])
            metrics=independent_mc_metrics(c,validation)
            candidate=dict(mc_ratio=ratio,validation=metrics,training=result)
            report['candidates'].append(candidate)
            if best is None or metrics['mse']<best['validation']['mse']:
                best=candidate;best_state=immutable_cpu_snapshot(c.critic.state_dict());best_optimizer=immutable_cpu_snapshot(c.critic_optimizer.state_dict())
        c.critic.load_state_dict(best_state);c.critic_optimizer.load_state_dict(best_optimizer)
        c.state['critic_updates']+=best['training']['optimizer_steps']
        report.update(selected_mc_ratio=best['mc_ratio'],after=best['validation'],
            attempted_critic_steps=sum(r['training']['optimizer_steps'] for r in report['candidates']))
        training.episodes.clear();validation.episodes.clear()
    report['wall_seconds']=max(c.distributed.all_gather_object(time.perf_counter()-started))
    root_call(c.distributed,lambda:atomic_json(c.session/'mc_calibration'/f'{label}.json',report))
    return report
