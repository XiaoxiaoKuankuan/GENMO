"""独立模型副本上的梯度消融、AdamW候选及Newton真实方向复检。

读取指定完整检查点的Actor和AdamW状态、同版本真实episode与接管请求。每个anchor
只计算一次同条件的dance/BC/local/global/track/joint梯度，按四组目标和两种可训练
模块组成实际AdamW候选。每个候选后恢复全部参数、优化器矩和RNG，逐张量核验。
Newton从episode初态重放原控制前史恢复接触，再执行同一个候选参考；额外对全量
组合候选从原请求时刻运行两次真实在线生成，检查下一轮历史反馈而非固定续接。
仅诊断，不更改正式训练；支持域超限、物理恢复失败与不可分辨方向都保留为失败项。
"""
from collections import defaultdict
from dataclasses import replace
import copy
import gc
import json
from pathlib import Path
import time
import torch

from .checkpoint import unpack_value,capture_rng,restore_rng
from .dance_losses import (reference_objectives,consumed_music_support,consumed_music_ticks,
    AnchorAlignment,execution_loss)
from .drift_diagnostics import write_json
from .music_repair_audit import anchors_from_episode,restore_physical_prefix
from .output_trust_region import check_outputs,motion_distance
from .teacher_training import parameter_module,_norm,_cosine
from .validation import ModelSupport,UnsupportedModelState


def assert_equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    else:assert a==b


def set_episode_context(controls,generations):
    """只从episode初态构造变换并传播到历史快照，不读取未来状态作为输入。"""
    from .episode_objective import with_episode_alignment
    initial=controls[0].start
    alignment=AnchorAlignment.at_anchor(initial.state,initial.teacher_reference.read_current_future(initial.tick,0)[0])
    controls=tuple(replace(s,start=with_episode_alignment(s.start,alignment),end=with_episode_alignment(s.end,alignment)) for s in controls)
    generations=tuple(replace(g,request=replace(g.request,snapshot=with_episode_alignment(g.request.snapshot,alignment))) for g in generations)
    return controls,generations


def candidate_objectives(trainer,sample,local):
    engine=trainer.imagination
    interval,plan,noise=engine.rollout_anchor(sample)
    lp,gp,_=reference_objectives(plan,local,sample.state.teacher_reference,engine.adapter,
        anchor_state=sample.state.state,anchor_tick=sample.state.tick,scales=engine.rewards.config.get('dance_scales'),
        global_support_ticks=consumed_music_support(sample,engine.horizon_steps),
        global_control_ticks=consumed_music_ticks(sample,engine.horizon_steps),
        global_reference_records=sample.transitions,global_reference_version=sample.generation.reference_version)
    t=interval.event['objective_terms']
    from .episode_objective import execution_mass
    mass=execution_mass(engine,trainer.config)
    values={'dance':t['dance_loss'].mean()/mass,'track':t['current_reference_loss'].mean()/mass,
        'local':lp,'global':gp,'joint_limit':t['joint_limit_cost'].mean()/mass}
    if engine.rewards.config.get('episode_dance_weight',0.):values['episode']=t['episode_dance_loss'].mean()/mass
    return values,interval,plan,noise


def rolling_two_requests(collector,controls,initial,sample):
    """从原请求物理前史继续真实生成两次；新历史来自本次实际Newton执行。"""
    request=sample.generation.request
    error=restore_physical_prefix(collector,controls,initial,request.ticket.request_tick)
    collector.state=request.snapshot.detached();collector.terminal=False
    collector.controls=[];collector.events=[];collector.metrics=[];collector.generations={}
    collector.trajectory=[];collector.cancelled_requests=[];collector.anchor_horizon=0
    collector.record_controls=True;collector.record_contacts=True;collector.timing_mode='fixed_delay_replay'
    collector.delay_events=collector.state.teacher_reference.identity['delivery_events'];collector.wall_started=time.perf_counter()
    collector.noise_seed=42;collector.collect(2)
    q=torch.cat([s.end.state for s in collector.controls])
    return {'request_tick':request.ticket.request_tick,'end_tick':collector.state.tick,'steps':len(q),
        'restore_error':error,'dance_world_mean':sum(m['dance_loss_unaligned'] for m in collector.metrics)/len(collector.metrics),
        'joint_speed_rms':float(q[:,34:55].square().mean().sqrt()),
        'joint_range_mean':float((q[:,13:34].amax(0)-q[:,13:34].amin(0)).mean()),
        'actual_requests':[e['request_tick'] for e in collector.events],'reason':collector.terminal,
        'state':q.cpu(),'events':collector.events}


def run_candidates(collector,actor,critic,engine,original,cfg,tasks,output,device):
    from .application import SupervisionStream,build_stage1_losses,task_inputs
    from .trainer import Stage2Trainer
    from .short_audit import phase_labels
    settings=cfg['diagnostic'];run=Path(settings['source_run']);version=settings['version']
    checkpoint=Path(settings['checkpoint']);ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    actor.load_state_dict(ck['actor'],strict=True);actor.eval();collector.actor_version=version
    trainer=Stage2Trainer(actor,critic,engine,build_stage1_losses(actor.stage1,original).to(device),cfg['training'])
    if 'genmo_optimizer' in ck:trainer.actor_optimizer.load_state_dict(ck['genmo_optimizer'])
    engine.support_guard=ModelSupport(json.loads((run/'state_support_contract.json').read_text()),device=device)
    del ck
    supervision=SupervisionStream(cfg['stage1']['data_config'],1,42,device)
    named=[(n,p) for n,p in actor.named_parameters() if p.requires_grad];params=[p for _,p in named]
    base_actor=copy.deepcopy(actor.state_dict());base_opt=copy.deepcopy(trainer.actor_optimizer.state_dict())
    weights={'dance':1.,'track':cfg['rewards']['track_weight'],'local':cfg['training']['local_prior_weight'],
        'global':cfg['training']['global_prior_weight'],'bc':cfg['training']['bc_weight'],
        'joint_limit':cfg['rewards']['joint_limit_weight']}
    if cfg['rewards'].get('episode_dance_weight',0.):weights['episode']=cfg['rewards']['episode_dance_weight']
    variants={'dance':['dance'],'dance_priors':['dance','local','global'],
        'dance_bc':['dance','bc'],'full':list(weights)}
    if 'episode' in weights:
        for k in variants:
            if 'episode' not in variants[k]:variants[k].append('episode')
    if settings.get('variants'):variants={k:v for k,v in variants.items() if k in settings['variants']}
    from .episode_objective import execution_mass
    mass=execution_mass(engine,cfg['training'])
    def primary(interval):
        t=interval.event['objective_terms']
        return float((t['dance_loss']+cfg['rewards'].get('episode_dance_weight',0.)*t.get('episode_dance_loss',0.)).mean())/mass
    reports=[];failures=[]
    root=run/'evaluations/fixed_delay_replay'/('initial_genmo_baseline' if version==0 else f'actor_{version:06}')
    for task in tasks:
        source=next(root.glob('shard_*/'+task['id']+'_seed42_real_controls.pt'))
        payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True);memo={}
        controls=unpack_value(payload['controls'],engine.adapter,device,memo)
        generations=unpack_value(payload['generations'],engine.adapter,device,memo)
        if cfg['rewards'].get('episode_dance_weight',0.):controls,generations=set_episode_context(controls,generations)
        _,initial=task_inputs(task,engine.adapter,device)
        samples=anchors_from_episode(controls,generations,engine.horizon_steps)
        seconds=settings.get('anchor_seconds',[2.0])
        selected=[]
        for seconds_at in seconds:
            candidates=sorted(samples,key=lambda s:abs(s.state.tick-seconds_at*600))
            selected.append(candidates[0])
        for sample in selected:
            actor.load_state_dict(base_actor);trainer.actor_optimizer.load_state_dict(copy.deepcopy(base_opt))
            batch=supervision.next();rng=capture_rng();local=engine.teacher.local(sample.generation.request)
            try:
                values,interval,plan,noise=candidate_objectives(trainer,sample,local)
                execution_grad=torch.autograd.grad(interval.end.state.square().mean(),noise,retain_graph=True)[0]
                if not torch.isfinite(execution_grad).all() or not float(execution_grad.norm()):raise FloatingPointError('执行导数无效')
                gradients={k:[None if g is None else g.detach().clone() for g in torch.autograd.grad(v,params,retain_graph=True,allow_unused=True)] for k,v in values.items()}
                bc_rng=capture_rng();prediction=actor.original_supervision(batch)
                bc,_=trainer.stage1_losses(batch,prediction['pred_x_start'],prediction['static_conf_logits'],global_step=595000)
                gradients['bc']=[None if g is None else g.detach().clone() for g in torch.autograd.grad(bc,params,allow_unused=True)]
                before_losses={k:float(v.detach()) for k,v in values.items()};before_losses['bc']=float(bc.detach())
                before=replace(plan,world_qpos=plan.world_qpos.detach(),contact=plan.contact.detach())
                model_before=primary(interval)
                module_metrics={m:{k:_norm([g for (n,_),g in zip(named,gs) if parameter_module(n)==m]) for k,gs in gradients.items()}
                    for m in ['stage1_backbone','history_prefix','future_encoder','diffusion_output']}
                del values,interval,plan,noise,prediction,bc,execution_grad
                def physical(plan):
                    restore_error=restore_physical_prefix(collector,controls,initial,sample.state.tick)
                    states=[];contacts=[]
                    with torch.no_grad():
                        inter,_,_=engine.rollout_anchor(sample,plan_override=plan,transition_fn=collector.env.step_physics,
                            on_step=lambda z,_:(states.append(z.state.cpu()),contacts.append(collector.env.contact_events())))
                    return primary(inter),torch.cat(states),contacts,restore_error
                real_before,states_before,contacts_before,restore_error=physical(before)
                real_repeat,_,_,_=physical(before);noise_floor=max(1e-8,2*abs(real_repeat-real_before))
                rolling_before=rolling_two_requests(collector,controls,initial,sample)
                for label,terms in variants.items():
                    for scope in settings.get('scopes',['all','future_only']):
                        actor.load_state_dict(base_actor);trainer.actor_optimizer.load_state_dict(copy.deepcopy(base_opt));restore_rng(rng)
                        trainer.actor_optimizer.zero_grad(set_to_none=True)
                        composed={k:[None if g is None else weights[k]*g for g in gradients[k]] for k in terms}
                        primary_grad=[sum([composed[k][i] for k in ('dance','episode') if k in composed and composed[k][i] is not None])
                            if any(k in composed and composed[k][i] is not None for k in ('dance','episode')) else None for i in range(len(params))]
                        primary_norm=_norm(primary_grad)
                        for k,ratio in cfg['training'].get('soft_auxiliary_gradient_budget',{}).items():
                            if k in composed:
                                scale=min(1.,ratio*primary_norm/max(_norm(composed[k]),1e-20))
                                composed[k]=[None if g is None else scale*g for g in composed[k]]
                        if cfg['training'].get('auxiliary_gradient_projection',False) and primary_norm:
                            for k in ('bc','local','global','track'):
                                if k not in composed:continue
                                dot=sum(float((g*p).sum()) for g,p in zip(composed[k],primary_grad) if g is not None and p is not None)
                                if dot<0:composed[k]=[(g if g is not None else torch.zeros_like(p))-dot/primary_norm**2*p if p is not None else g for g,p in zip(composed[k],primary_grad)]
                        for i,(name,p) in enumerate(named):
                            if scope=='future_only' and not name.startswith('future_encoder.'):continue
                            gs=[composed[k][i] for k in terms if composed[k][i] is not None]
                            p.grad=sum(gs[1:],gs[0].clone()) if gs else None
                        del composed,primary_grad
                        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));trainer.actor_optimizer.step()
                        delta=[p.detach()-base_actor[n] for n,p in named]
                        effects={m:{k:sum(float((g*d).sum()) for (n,_),g,d in zip(named,gs,delta) if g is not None and (m=='all' or parameter_module(n)==m)) for k,gs in gradients.items()}
                            for m in ['all','stage1_backbone','history_prefix','future_encoder','diffusion_output']}
                        try:
                            with torch.no_grad():
                                after_values,after_interval,after,_=candidate_objectives(trainer,sample,local)
                                model_after=primary(after_interval)
                                after_losses={k:float(v) for k,v in after_values.items()}
                                restore_rng(bc_rng);prediction=actor.original_supervision(batch)
                                bc,_=trainer.stage1_losses(batch,prediction['pred_x_start'],prediction['static_conf_logits'],global_step=595000);after_losses['bc']=float(bc)
                            p=sample.generation.request.ticket.prefix_frames;clock=sample.generation.request.ticket.request_tick+torch.arange(60,device=device)*20
                            mask=(torch.arange(60,device=device)>=p)&torch.isin(clock,consumed_music_support(sample,16))
                            fixed=sample.state.teacher_reference
                            absolute=fixed.source_at(clock.clamp(fixed.lower,fixed.upper))
                            align=AnchorAlignment.at_reference_anchor(sample.state.reference.read_current_future(sample.state.tick,0)[0],fixed.read_current_future(sample.state.tick,0)[0])
                            reasons,output_metrics=check_outputs(before.world_qpos,after.world_qpos,local.world_qpos,align.poses(absolute),mask,
                                prefix=p,config=cfg['training']['output_trust_region'],unaligned_global_target=absolute)
                            real_after,states_after,contacts_after,error=physical(after)
                            md=model_after-model_before;rd=real_after-real_before
                            rolling_after=rolling_two_requests(collector,controls,initial,sample) if label=='full' and scope=='all' else None
                            row={'version':version,'task':task['id'],'anchor_tick':sample.state.tick,'request_tick':sample.generation.request.ticket.request_tick,
                                'phases':phase_labels(sample,sample.transitions[0].event['contacts'],None,engine.adapter,16),
                                'variant':label,'trainable':scope,'comparison_objective':'local_plus_episode' if 'episode' in weights else 'local_H16',
                                'loss_before':before_losses,'loss_after':after_losses,
                                'module_gradient_norms':module_metrics,'raw_gradient_dot_actual_parameter_delta':effects,
                                'gradient_norm_before_clip':norm,'model_dance_before':model_before,'model_dance_after':model_after,
                                'physical_dance_before':real_before,'physical_dance_after':real_after,'model_delta':md,'physical_delta':rd,
                                'physical_repeat_noise_floor':noise_floor,'direction_resolvable':abs(rd)>noise_floor,
                                'same_descent_sign':(md<0)==(rd<0) if abs(rd)>noise_floor and abs(md)>1e-8 else None,
                                'physical_restore_error':max(error,restore_error),'output_trust_reasons':reasons,'output_metrics':output_metrics,
                                'reference_regeneration_max_abs':float((before.world_qpos-sample.generation.plan.world_qpos).abs().max()),
                                'physical_velocity_rms_before':float(states_before[:,34:55].square().mean().sqrt()),
                                'physical_velocity_rms_after':float(states_after[:,34:55].square().mean().sqrt()),
                                'contacts_changed':contacts_before!=contacts_after}
                            if rolling_after:
                                row['next_two_real_generations']={k:{a:b for a,b in v.items() if a not in ['state','events']} for k,v in [('before',rolling_before),('after',rolling_after)]}
                            artifact=output/f'candidate_{len(reports):04}.pt'
                            torch.save({'report':row,'reference_before':before.world_qpos.cpu(),'reference_after':after.world_qpos.detach().cpu(),
                                'physical_before':states_before,'physical_after':states_after,'contacts_before':contacts_before,'contacts_after':contacts_after,
                                'rolling_before':rolling_before if rolling_after else None,'rolling_after':rolling_after},artifact)
                            row['artifact']=str(artifact);reports.append(row)
                            with (output/'candidates.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                            print(json.dumps({k:row[k] for k in ['version','task','anchor_tick','variant','trainable','model_delta','physical_delta','same_descent_sign']},ensure_ascii=False),flush=True)
                        except (UnsupportedModelState,RuntimeError,ValueError) as error:
                            failures.append({'version':version,'task':task['id'],'anchor':sample.state.tick,'variant':label,'scope':scope,'error':str(error),'support_details':getattr(error,'details',None)})
                        finally:
                            actor.load_state_dict(base_actor);trainer.actor_optimizer.load_state_dict(copy.deepcopy(base_opt));restore_rng(rng)
                            assert_equal(actor.state_dict(),base_actor);assert_equal(trainer.actor_optimizer.state_dict(),base_opt)
                            trainer.actor_optimizer.zero_grad(set_to_none=True)
                del gradients
            except (UnsupportedModelState,RuntimeError,ValueError) as error:
                failures.append({'version':version,'task':task['id'],'anchor':sample.state.tick,'error':str(error),'support_details':getattr(error,'details',None)})
            write_json(output/'summary.json',{'cases':reports,'failures':failures,'complete_rollback_verified':True,'newton_real_execution':True,'source_checkpoint':str(checkpoint)})
        del controls,generations,payload,memo,samples,selected;gc.collect()
    actor.load_state_dict(base_actor);engine.teacher.assert_frozen();engine.adapter.bundle.assert_unchanged(trainer.frozen_identity)


def run_drift(collector,actor,critic,engine,original,cfg,tasks,output,device):
    if cfg['diagnostic']['phase']=='candidates':
        return run_candidates(collector,actor,critic,engine,original,cfg,tasks,output,device)
    from .drift_experiment import run_experiment
    return run_experiment(collector,actor,critic,engine,original,cfg,tasks,output,device)
