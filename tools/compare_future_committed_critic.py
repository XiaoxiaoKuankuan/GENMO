"""在同一批完整物理MC轨迹上公平比较历史Critic与未来已提交参考Critic。

读取P0新世界基线的SHA核验episode和逐决策因果快照，行政截断不生成零尾MC。
固定seed42轨迹训练、seed1729轨迹测试，音乐相同但物理/生成噪声不同；明确不冒充
未见音乐泛化。旧新网络从同一旧权重出发，使用相同训练采样索引/固定更新数/优化配置；
困难状态提高采样比例，两者完全一致。测试集不参与训练、选epoch或早停。
冻结Stage1 Actor及其条件编码，只拟合Critic；复算整体/切分/大根误差/失败前/100秒后
Value误差、EV和8秒GAE对完整MC优势一致性。只有困难状态与整体均改善才发布有效标志。
保存独立新schema的权重和分析，不冒充Actor训练checkpoint或自动部署的预热成果。
可只读加载已拟合权重，分别遮蔽参考几何与时序信息，检查改善来自哪里；不再优化或选epoch。
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic,initialize_future_critic_shared
from gem.closedloop.dppo.complete_episode import complete_mc_returns
from gem.closedloop.dppo.returns import compute_gae,validate_value_state_boundaries
from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.budget import atomic_json
from tools.summarize_decisive_physics import load_report
from tools.analyze_decisive_credit import discounted_complete


def stats(pred,target,ids):
    p=np.asarray(pred)[ids];y=np.asarray(target)[ids];e=p-y
    return dict(states=len(e),rmse=float(np.sqrt(np.mean(e*e))) if len(e) else None,
        bias=float(e.mean()) if len(e) else None,
        explained_variance=float(1-np.var(e)/np.var(y)) if len(e)>1 and np.var(y)>1e-12 else None)


def gae_advantages(episodes,values,gamma,lam):
    output=np.zeros(len(values));boundary=np.zeros(len(values),bool)
    for episode in episodes:
        start,end=episode['indices'];rows=episode['rows'];begin=0
        for finish in range(1,len(rows)+1):
            if finish<len(rows) and rows[finish-1].control_tick_end-rows[begin].control_tick_begin<4800:continue
            part=rows[begin:finish];indices=list(range(start+begin,start+finish))
            result=compute_gae(rewards=[r.rewards for r in part],values=values[indices],
                next_values=[values[i+1] if i+1<end else 0. for i in indices],
                executed_steps=[r.executed_control_steps for r in part],
                bootstrap_mask=[not r.terminated for r in part],continuation_mask=[i+1<len(part) for i in range(len(part))],
                event_rewards=[r.metadata.get('event_reward',0.) for r in part],gamma_upper=gamma,lambda_upper=lam)
            output[indices]=result['advantages_raw'].numpy()
            if finish<len(rows):
                cut=rows[finish-1].control_tick_end
                for i in range(begin,finish):boundary[start+i]=0<cut-rows[i].control_tick_begin<=600
            begin=finish
    return output,boundary


def evaluate(pred,targets,episodes,groups,gamma,lam):
    adv,_=gae_advantages(episodes,pred,gamma,lam);mc=np.asarray(targets)-np.asarray(pred);out={}
    for name,ids in groups.items():
        row=stats(pred,targets,ids);a=adv[ids];b=mc[ids]
        row.update(gae_mc_correlation=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-12 and np.std(b)>1e-12 else None,
            gae_mc_sign_agreement=float(np.mean(np.sign(a)==np.sign(b))) if len(a) else None)
        out[name]=row
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--updates',type=int,default=384);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--fitted-source',type=Path,help='只读已有比较权重做消融，新增Critic更新为零')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.manual_seed(20261024);np.random.seed(20261024);torch.backends.cuda.matmul.allow_tf32=False
    started=time.time();session=next(a.run.glob('sessions/*/resolved_config.yaml')).parent
    config=yaml.safe_load((session/'resolved_config.yaml').read_text());config['runtime']['genmo_device']=a.device
    phase=session/'phases/decisive_fresh_p0_baseline/rank00'
    report,plan,physical=load_report(phase/'report/report.json')
    saved=torch.load(phase/'causal_references.pt',map_location='cpu',weights_only=False)
    bytask={}
    for r in saved['conditions']:bytask.setdefault(r.task_id,[]).append(r)
    rows=[];targets=[];episodes=[];metadata=[];excluded=[]
    gamma=float(config['stage9']['gamma_upper']);lam=float(config['stage9']['lambda_upper'])
    for e in physical:
        rr=bytask.get(e['task_id'],[])
        if not rr or not rr[-1].terminated or rr[-1].truncated:
            excluded.append(dict(task=e['task_id'],reason=e['end_reason']));continue
        if len(rr)!=len(e['decisions']):raise ValueError('MC rows and physical decisions differ')
        target=complete_mc_returns(rr,gamma_upper=gamma).numpy()
        np.testing.assert_allclose(target,discounted_complete(e['decisions'],gamma),atol=1e-10,rtol=0)
        begin=len(rows);final=rr[-1].control_tick_end
        for r,d in zip(rr,e['decisions']):
            validate_value_state_boundaries(r)
            if r.identity!=d['identity']:raise ValueError('MC identity mismatch')
            if 'committed_reference' not in r.metadata['critic_context']:raise ValueError('Missing decision-time committed cache')
            errors=d['controls'][0]['raw_errors'] if d['controls'] else {}
            metadata.append(dict(task_id=e['task_id'],sample_id=e['sample_id'],dataset=e['dataset'],seed=e['seed'],
                time=(r.control_tick_begin-600)/600.,root_pos=float(np.sqrt(errors.get('anchor_pos_squared_error',0.))),
                root_ori=float(np.sqrt(errors.get('anchor_ori_squared_error',0.))),
                failure_before5=bool(e['physical_failure'] and final-r.control_tick_begin<=3000)))
        rows.extend(rr);targets.extend(target.tolist());episodes.append(dict(indices=(begin,len(rows)),rows=rr))
    y=np.asarray(targets);train=np.array([r['seed']==42 for r in metadata]);test=np.array([r['seed']==1729 for r in metadata])
    if not train.any() or not test.any() or not np.all(train|test):raise ValueError('Explicit train/test trajectory seeds required')
    actor,_,_=load_actor(config);actor.requires_grad_(False);fingerprint=_model_fingerprint(actor)
    cc=config['stage10']['critic'];kw={k:cc[k] for k in ('dimension','layers','heads','history_scales')}
    old=PretrainedConditionCritic(actor,**kw).to(a.device)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);old.load_state_dict(payload['critic'],strict=True)
    new=PretrainedConditionCritic(actor,**kw,future_reference=True).to(a.device);migration=initialize_future_critic_shared(new,old.state_dict())
    features=[];remaining=[]
    for start in range(0,len(rows),a.batch):
        part=rows[start:start+a.batch];ctx={k:torch.cat([r.metadata['critic_context'][k] for r in part]).to(a.device) for k in part[0].metadata['critic_context']}
        features.append(new.encode_frozen(ctx));remaining.extend(r.metadata['remaining_music_seconds'] for r in part)
    f={k:torch.cat([v[k] for v in features]) for k in features[0]};remaining=torch.tensor(remaining,device=a.device)
    target=torch.tensor(y,device=a.device,dtype=torch.float32)
    def predict(model,zero_future=False,ablation=None):
        result=[];model.eval()
        with torch.no_grad():
            for start in range(0,len(rows),a.batch):
                ff={k:v[start:start+a.batch] for k,v in f.items()}
                if zero_future:
                    ff={k:(torch.zeros_like(v) if k.startswith('committed_') else v) for k,v in ff.items()}
                if ablation:
                    ff['committed_reference']=ff['committed_reference'].clone()
                    if ablation=='geometry':ff['committed_reference'][...,:36]=0.
                    elif ablation=='timing':
                        ff['committed_reference'][...,36:]=0.
                        for key in ('committed_metadata','committed_metadata_valid'):ff[key]=torch.zeros_like(ff[key])
                    else:raise ValueError('Unknown future cache ablation')
                result.extend(model.forward_encoded(ff,remaining[start:start+a.batch]).cpu().tolist())
        return np.asarray(result)
    initial=predict(old);np.testing.assert_array_equal(initial,predict(new))
    _,cuts=gae_advantages(episodes,initial,gamma,lam)
    pos=np.array([r['root_pos'] for r in metadata]);ori=np.array([r['root_ori'] for r in metadata])
    late=np.array([r['time']>=100 for r in metadata]);failure=np.array([r['failure_before5'] for r in metadata])
    pos_high=pos>=np.quantile(pos[train],.9);ori_high=ori>=np.quantile(ori[train],.9)
    group_masks=dict(all=np.ones(len(rows),bool),rollout_cut_before1s=cuts,large_root_position=pos_high,
        large_root_orientation=ori_high,failure_before5s=failure,after100s=late)
    groups={k:np.flatnonzero(v&test) for k,v in group_masks.items()}
    hard=train&(pos_high|ori_high|late|failure|cuts);normal_ids=np.flatnonzero(train);hard_ids=np.flatnonzero(hard)
    rng=np.random.default_rng(20261024);schedule=[]
    for _ in range(a.updates):
        indices=np.concatenate((rng.choice(normal_ids,a.batch//2),rng.choice(hard_ids,a.batch-a.batch//2)))
        schedule.append(torch.as_tensor(indices,device=a.device))
    report_out=dict(schema='genmo.future_committed_critic.matched_mc.v1',scope='held_out_seed_same_music_not_unseen_music',
        source_run=str(a.run),checkpoint=str(a.checkpoint),actor_fingerprint=fingerprint,migration=migration,
        source_reports_sha256=hashlib.sha256((phase/'report/report.json').read_bytes()).hexdigest(),
        train_states=int(train.sum()),test_states=int(test.sum()),train_seed=42,test_seed=1729,
        complete_tasks=len(episodes),excluded=excluded,updates=a.updates,batch=a.batch,hard_sampling_fraction=.5,
        initial=evaluate(initial,y,episodes,groups,gamma,lam),models={},training_updates_actor=0)
    report_out['new_critic_updates_per_model']=0 if a.fitted_source else a.updates
    if a.fitted_source:
        prior=json.loads((a.fitted_source/'summary.json').read_text())
        for key in ('actor_fingerprint','source_reports_sha256','updates','train_seed','test_seed'):
            if prior[key]!=report_out[key]:raise ValueError('Fitted source behavior/data/schedule differs: '+key)
        report_out['fitted_source']=str(a.fitted_source)
    for name,model in (('old_refit',old),('future_refit',new)):
        if a.fitted_source:
            fitted=torch.load(a.fitted_source/(name+'.pt'),map_location='cpu',weights_only=False)
            if (fitted['schema']!='genmo.stage2.future_committed_critic_mc_comparison.v1'
                    or fitted['context_schema']!=model.context_schema or fitted['actor_fingerprint']!=fingerprint):
                raise ValueError('Wrong fixed fitted Critic identity')
            model.load_state_dict(fitted['critic'],strict=True)
        history_ids={id(v) for v in model.history_encoder.parameters()}
        optimizer=torch.optim.AdamW([dict(params=list(model.history_encoder.parameters()),lr=cc['history_learning_rate']),
            dict(params=[v for v in model.parameters() if v.requires_grad and id(v) not in history_ids],lr=cc['learning_rate'])],weight_decay=0.)
        for step,indices in enumerate([] if a.fitted_source else schedule):
            model.train();optimizer.zero_grad(set_to_none=True)
            value=model.forward_encoded({k:v[indices] for k,v in f.items()},remaining[indices]);loss=(value-target[indices]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MC regression')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step%64==0:print(name,step,float(loss),flush=True)
        pred=predict(model);report_out['models'][name]=evaluate(pred,y,episodes,groups,gamma,lam)
        np.savez(a.output_dir/(name+'_predictions.npz'),prediction=pred,target=y,train=train,test=test)
        if not a.fitted_source:
            torch.save(dict(schema='genmo.stage2.future_committed_critic_mc_comparison.v1',context_schema=model.context_schema,
                critic={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                actor_fingerprint=fingerprint,updates=a.updates,not_full_training_checkpoint=True),a.output_dir/(name+'.pt'))
    report_out['models']['future_without_cache']=evaluate(predict(new,True),y,episodes,groups,gamma,lam)
    for ablation in ('geometry','timing'):
        report_out['models']['future_without_'+ablation]=evaluate(predict(new,ablation=ablation),y,episodes,groups,gamma,lam)
    if a.fitted_source:
        for name in ('old_refit','future_refit','future_without_cache'):
            for group in groups:
                np.testing.assert_allclose(report_out['models'][name][group]['rmse'],prior['models'][name][group]['rmse'],atol=1e-6,rtol=0)
    hard_groups=[k for k,ids in groups.items() if k!='all' and len(ids)]
    report_out['quality_gate_passed']=bool(hard_groups and all(
        report_out['models']['future_refit'][k]['rmse']<report_out['models']['old_refit'][k]['rmse'] for k in hard_groups)
        and report_out['models']['future_refit']['all']['rmse']<=report_out['models']['old_refit']['all']['rmse'])
    report_out['quality_gate_scope']='all_nonempty_hard_groups_and_overall_rmse_improve_against_equal_budget_old_refit'
    report_out['actor_unchanged']=_model_fingerprint(actor)==fingerprint
    if not report_out['actor_unchanged']:raise AssertionError('Actor changed during Critic-only comparison')
    report_out['wall_seconds']=time.time()-started
    atomic_json(a.output_dir/'summary.json',report_out)
    with (a.output_dir/'states.jsonl').open('w') as file:
        for i,r in enumerate(metadata):file.write(json.dumps(dict(r,target=y[i],training=bool(train[i])),ensure_ascii=False)+'\n')
    print(json.dumps(report_out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
