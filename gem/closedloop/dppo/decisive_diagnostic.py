"""RoboFFT代理与真实扩散链的有限方向诊断，以及冻结Stage1参考干预。

数据流：先用现有连续滚动物理采集器取得真实行为及生产GAE；仅本入口临时保留
同一次采样的完整20步链，并把它放在独立diagnostic字段中，正常RoboFFT紧凑模式
完全不变。固定MC噪声后分别计算真实优势、置换优势、无优势score与真实10步
Gaussian联合转移score。方向按模块匹配参数扰动范数，每卡独立评估一个正/负
方向，全部卡使用同一完整音乐任务与初态种子，不调用Adam或修改Critic/GMT。

参考干预只作用在尚未承诺的自由坐标，并用平滑起落窗连接已知前缀；保留原提交、
插值、限位检查与物理奖励合同。所有结果写入独立诊断目录，结束恢复Actor字节。
小队列置信区间用于探索，不能替代全验证集或真机验收。真实链梯度是增广随机链
的REINFORCE方向，不宣称已积分消去隐变量得到最终动作的显式概率密度。
"""
from __future__ import annotations

import copy
import math
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

from .actor_optimizer import actor_parameter_group
from .budget import atomic_json
from .execution_checks import policy_phase
from .parallel_support import local_call, root_call, finish_lease
from .tensor_cache import RolloutTensorCache


def clip_masks(ratio, advantage, clip=.01):
    """区分Ratio越界、符号相关真正裁剪和仍保留梯度的越界样本。"""
    outside=(ratio<1-clip)|(ratio>1+clip)
    active=((advantage>0)&(ratio>1+clip))|((advantage<0)&(ratio<1-clip))
    return dict(outside=outside,active=active,outside_active_gradient=outside&~active,
                zero_advantage=advantage==0)


@contextmanager
def capture_true_chain(policy):
    original=policy.sample_rollout
    def sample(conditions,**kwargs):
        kwargs['store_chain']=True
        trace=original(conditions,**kwargs)
        for key in ('chain','old_log_probs','old_means','old_stds','sampling_noise'):
            trace['diagnostic_'+key]=trace.pop(key)
        return trace
    policy.sample_rollout=sample
    try:yield
    finally:policy.sample_rollout=original


def module_stats(first,second):
    values={}
    for name in first:
        group=actor_parameter_group(name)
        a,b=first[name].double(),second[name].double()
        item=values.setdefault(group,[0.,0.,0.])
        item[0]+=float((a*b).sum());item[1]+=float(a.square().sum());item[2]+=float(b.square().sum())
    total=np.sum(list(values.values()),axis=0).tolist();values['all']=total
    return {key:dict(cosine=v[0]/max(math.sqrt(v[1]*v[2]),1e-300),
                    first_norm=math.sqrt(v[1]),second_norm=math.sqrt(v[2])) for key,v in values.items()}


def matched_direction(raw,reference_norms):
    norms={}
    for name,value in raw.items():
        key=actor_parameter_group(name);norms[key]=norms.get(key,0.)+float(value.double().square().sum())
    return {name:value*(reference_norms[actor_parameter_group(name)]/max(math.sqrt(norms[actor_parameter_group(name)]),1e-30))
            for name,value in raw.items()}


def _gradient(c,rows,cache,coefficients,kind,*,weighted=True):
    """最终动作代理或原始真实Gaussian联合转移；不采用坐标均值代理DPPO。"""
    policy,actor=c.policy,c.actor
    actor.zero_grad(set_to_none=True)
    count=sum(c.distributed.all_gather_object(len(rows)))
    from .optimization_support import _local_phase
    with _local_phase(c.distributed,'decisive_gradient_'+kind),policy_phase(policy):
        for start in range(0,len(rows),4):
            selected=rows[start:start+4];condition=cache.context(selected)
            coefficient=coefficients[start:start+len(selected)].to(c.distributed.device)
            if kind=='gaussian':
                from .gaussian_probability import gaussian_statistics
                for step in policy.trainable_steps:
                    trace=[r.metadata['sampler_trace'] for r in selected]
                    state=torch.cat([t['diagnostic_chain'][:,step] for t in trace]).to(c.distributed.device)
                    following=torch.cat([t['diagnostic_chain'][:,step+1] for t in trace]).to(c.distributed.device)
                    old_mean=torch.cat([t['diagnostic_old_means'][:,step] for t in trace]).to(c.distributed.device)
                    old_std=torch.cat([t['diagnostic_old_stds'][:,step] for t in trace]).to(c.distributed.device)
                    parameters=policy.transition_parameters(condition,state,step)
                    statistic=gaussian_statistics(following,old_mean,old_std,parameters['mean'],parameters['std'],parameters['free_mask'])
                    # 全部10步、自由坐标联合求和，保持真实Gaussian score；只除样本数。
                    (statistic['joint_log_ratio']*coefficient/count).sum().backward()
            else:
                records=[r.mc_record for r in selected]
                for k in range(0,16,4):
                    loss=policy.losses(condition,cache.get('action_x0',selected),
                        torch.stack([r.eps[k:k+4] for r in records]),
                        torch.stack([r.sampling_indices[k:k+4] for r in records]),
                        torch.stack([r.original_timesteps[k:k+4] for r in records]),
                        time_weights=torch.stack([r.time_weights for r in records]) if weighted else None)
                    scale=policy.robofft['logratio_alpha']*policy.robofft['logprob_normalizer']
                    (-loss.mean(1)*coefficient*scale/4/count).sum().backward()
    c.distributed.sum_gradients(actor)
    result={name:p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu')
            for name,p in actor.named_parameters() if p.requires_grad}
    actor.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def proxy_losses(c,rows,cache):
    parts=[]
    with policy_phase(c.policy):
        for begin in range(0,len(rows),4):
            selected=rows[begin:begin+4];records=[r.mc_record for r in selected];losses=[]
            for k in range(0,16,4):
                losses.append(c.policy.losses(cache.context(selected),cache.get('action_x0',selected),
                    torch.stack([r.eps[k:k+4] for r in records]),torch.stack([r.sampling_indices[k:k+4] for r in records]),
                    torch.stack([r.original_timesteps[k:k+4] for r in records]),time_weights=torch.stack([r.time_weights for r in records])).cpu())
            parts.append(torch.cat(losses,1))
    return torch.cat(parts)


def historical_clip_audit(c,rows,cache,base,output):
    """新鲜同一旧行为样本与固定MC上的历史模型对照，明确不伪称原轮原MC重放。"""
    from ..robofft.surrogate import action_objective
    root=c.output.parent/'stage2_robofft_execweighted_20261024_3round'/'checkpoints'
    reports=[];old=torch.stack([r.mc_record.old_loss for r in rows]);adv=cache.get('advantages',rows).cpu()
    try:
        for iteration in (1,2,3):
            candidates=list(root.glob(f'stage10_r{iteration:06d}_*.pt'))
            if len(candidates)!=1:continue
            payload=torch.load(candidates[0],map_location='cpu',mmap=True,weights_only=False)
            c.actor.load_state_dict(payload['actor']);del payload
            new=proxy_losses(c,rows,cache).requires_grad_()
            options={k:c.policy.robofft[k] for k in ('beta_neg','logprob_normalizer','logratio_alpha')}
            metric=action_objective(new,old,adv,clip=c.settings['ppo_clip'],**options)
            coefficient,=torch.autograd.grad(metric['loss'].sum(),new)
            masks=clip_masks(metric['ratio'].detach(),adv,c.settings['ppo_clip'])
            local=[dict(ratio=float(metric['ratio'][i]),advantage=float(adv[i]),
                outside=bool(masks['outside'][i]),active_clip=bool(masks['active'][i]),
                derivative_zero=bool(coefficient[i].abs().sum()==0),mc_clamp_fraction=float(metric['logratio_clamp_fraction'][i])) for i in range(len(rows))]
            allrows=[x for part in c.distributed.all_gather_object(local) for x in part]
            reports.append(dict(iteration=iteration,checkpoint=str(candidates[0]),actions=len(allrows),
                ratio_outside_fraction=np.mean([x['outside'] for x in allrows]).item(),
                active_clip_fraction=np.mean([x['active_clip'] for x in allrows]).item(),
                actual_zero_derivative_fraction=np.mean([x['derivative_zero'] for x in allrows]).item(),rows=allrows))
    finally:c.actor.load_state_dict(base)
    root_call(c.distributed,lambda:atomic_json(output/'signed_clip_audit.json',dict(
        scope='fresh_parent28_behavior_fixed_new_MC_against_historical_models_not_original_round_clip_reconstruction',reports=reports)))


def heldout_score_audit(c,rows,cache,base,direction,output):
    """用前半环境拟合score均值方向，后半环境检验该方向的非零均值。

    对固定MC作对称有限差分，按物理episode聚类，不以同一批拟合范数宣称显著。
    同音乐跨环境仍可能相关，最终报告明确为探索性环境聚类，而非独立音乐确证。
    """
    half=len(rows)//2;parts=[]
    for sign in (1,-1):
        apply_direction(c.actor,base,direction,sign*.1)
        parts.append(-proxy_losses(c,rows,cache).mean(1)*c.policy.robofft['logratio_alpha']*c.policy.robofft['logprob_normalizer'])
    c.actor.load_state_dict(base)
    derivative=(parts[0]-parts[1])/.2
    clusters={}
    for i in range(half,len(rows)):
        identity=rows[i].identity;key=str((c.distributed.rank,identity.get('backend_session_id'),identity.get('env_id'),identity.get('episode_id')))
        clusters.setdefault(key,[]).append(float(derivative[i]))
    gathered=c.distributed.all_gather_object(clusters);allclusters={k:v for part in gathered for k,v in part.items()}
    averages=np.array([np.mean(v) for v in allclusters.values()]);rng=np.random.default_rng(20261024)
    boot=np.mean(rng.choice(averages,(10000,len(averages)),replace=True),axis=1)
    result=dict(scope='heldout_environment_half_fixed_MC_symmetric_score_derivative',
        clusters=len(averages),states=sum(map(len,allclusters.values())),mean=float(averages.mean()),
        interval95=np.quantile(boot,[.025,.975]).tolist(),clusters_values=allclusters,
        limitation='shared_music_and_single_behavior_window_limit_independence; not proof_of_exact_population_score')
    root_call(c.distributed,lambda:atomic_json(output/'heldout_score_mean.json',result))
    return result


@torch.no_grad()
def apply_direction(actor,base,direction,scale):
    actor.load_state_dict(base)
    for name,p in actor.named_parameters():
        if name in direction:p.add_(direction[name].to(p),alpha=scale)


@torch.no_grad()
def replay_outputs(policy,rows):
    """同一真实链的全部创新噪声重放，用于最终参考变化检查。"""
    condition={k:torch.cat([r.context[k] for r in rows]).to(next(policy.actor.parameters()).device) for k in rows[0].context}
    device=condition['future_valid'].device
    with policy_phase(policy):
        prepared=policy.prepare_conditions(condition)
        state=torch.cat([r.metadata['sampler_trace']['diagnostic_chain'][:,0] for r in rows]).to(device)
        for step in range(20):
            p=policy.transition_parameters(condition,state,step,prepared=prepared)
            if step==19:state=p['mean']
            else:
                noise=torch.cat([r.metadata['sampler_trace']['diagnostic_sampling_noise'][:,step] for r in rows]).to(device)
                state=policy.actor._constrain(p['mean']+p['std']*noise,prepared['adapted'])
        physical=policy.actor.endecoder.denormalize(state)
        physical=torch.where(prepared['adapted']['known_mask'],prepared['adapted']['known_physical'],physical)
        return physical.detach().cpu()


def _plan(c):
    from .periodic_monitor import build_balanced_plan
    from .evaluation import _digest
    plan=build_balanced_plan(c.catalog,samples_per_source=c.stage['evaluation']['samples_per_source'],
        seeds=[42,1729],selection_seed=42,seconds=None,start_mode='beginning',unique_music=True,
        split='train',training_progress=True,original_split='train')
    chosen={}
    for source in sorted({t['dataset'] for t in plan['tasks']}):
        group=[t for t in plan['tasks'] if t['dataset']==source and t['seed']==42]
        group=sorted(group,key=lambda t:(0 if source=='FineDance' and t['sample_id']=='115' else 1,t['remaining_music_seconds']))
        chosen[source]=group[:2]
    keys={(t['dataset'],t['sample_id']) for group in chosen.values() for t in group}
    plan['tasks']=[t for t in plan['tasks'] if (t['dataset'],t['sample_id']) in keys]
    plan['selected_sample_count']=len(keys);plan['task_count']=len(plan['tasks'])
    plan['requested_eval_count']=len(keys)
    plan['selected_unique_group_count']=len(keys);plan['selected_unique_audio_count']=len(keys)
    plan['periodic_subset']['source_sample_counts']={s:len(t) for s,t in chosen.items()}
    plan.pop('plan_sha256');plan['plan_sha256']=_digest(plan)
    return plan


def evaluate_local_candidates(c,plan,label):
    """每rank完整相同任务，但Actor/干预不同；不跨rank混为一个策略评估。"""
    from .vector_evaluation import isolated_vector_evaluation
    from .evaluation_runtime import _new_phase
    from .evaluation_world import EvaluationWorldDriver
    from .evaluation import evaluate_policy, _model_fingerprint
    from .periodic_monitor import snapshot_critic_transition
    from .rolling_schedule import generation_credit
    control=sum(math.ceil(t['remaining_music_seconds']*50)+100 for t in plan['tasks'])*8
    generation=sum(generation_credit(c.config,t['remaining_music_seconds'],overhead=8) for t in plan['tasks'])
    credits=[dict(control_steps=control,physics_steps=control*4,generations=generation)]*c.distributed.world_size
    path,lease,budget=_new_phase(c,'decisive_'+label,credits)
    conditions=[];reference_sums=dict(count=0,root_position_squared=0.,root_rotation_squared=0.,leg_squared=0.)
    def sink(task,row):
        if label=='A_baseline' or label.startswith(('repeatability_','fresh_')):
            snapshot=snapshot_critic_transition(row);snapshot.task_id=task['task_id'];conditions.append(snapshot)
        # 逐状态生成输出单独保存，后续只按实际消费帧匹配，不使用执行后状态作目标。
        trace=row.metadata['sampler_trace']
        record=dict(task_id=task['task_id'],identity=row.identity,decision_index=row.identity.get('decision_index'),
            qpos30=trace['qpos30'].detach().cpu(),qpos=trace['qpos'].detach().cpu(),
            generated=row.metadata['generated'],
            provenance=row.metadata.get('compact_execution_provenance'))
        reference_records.append(record)
    reference_records=[]
    with isolated_vector_evaluation(c,'decisive_'+label):
        cfg=copy.deepcopy(c.config);cfg['runtime'].update(evidence_level='full',evaluation_metrics_only=True)
        driver=EvaluationWorldDriver(c,cfg,budget,path)
        try:
            result=local_call(c.distributed,lambda:evaluate_policy(driver.env,c.policy,plan,path/'report',catalog=c.catalog,
                actor_identity=dict(model_fingerprint=_model_fingerprint(c.actor),diagnostic_label=label,rank=c.distributed.rank),
                frozen_modules={'critic':c.critic},local_cuda_only=True,world_driver=driver,transition_sink=sink,
                progress=lambda event:print(f'[DECISIVE {label} rank={c.distributed.rank}] {event}',flush=True)))
        finally:driver.close()
    torch.save(dict(conditions=conditions,references=reference_records),path/'causal_references.pt')
    c.state['budget']=finish_lease(c.distributed,c.budget,budget,lease,credits)
    return str(path/'report/report.json')


def smooth_free_window(mask):
    """前缀之后六帧缓入，末尾六帧缓出；任何已知/无效坐标严格为零。"""
    free=mask.any(-1);frames=torch.arange(mask.shape[1],device=mask.device)[None,:]
    first=torch.where(free,frames,mask.shape[1]).amin(1,keepdim=True)
    last=torch.where(free,frames,-1).amax(1,keepdim=True)
    phase=torch.minimum((frames-first)/6,(last-frames)/6).clamp(0,1)
    return .5-.5*torch.cos(math.pi*phase)


@contextmanager
def perturb_reference(policy,variant,audit):
    original=policy.sample_rollout
    @torch.no_grad()
    def sample(conditions,**kwargs):
        trace=original(conditions,**kwargs)
        base=trace['qpos30'];free=trace['free_mask'];window=smooth_free_window(free)
        proposal=base.clone();codec=policy.actor.endecoder.codec;kin=policy.actor.endecoder.kinematics
        if variant.startswith('height'):
            proposal[:,:,2]+=window*(.01 if variant.endswith('plus') else -.01)
        elif variant.startswith('yaw'):
            quaternion=codec.rotation_features_to_quat(base[:,:,3:9]);angle=window*(.02 if variant.endswith('plus') else -.02)
            w,x,y,z=quaternion.unbind(-1);co,si=torch.cos(angle/2),torch.sin(angle/2)
            rotated=torch.stack((co*w-si*z,co*x-si*y,co*y+si*x,co*z+si*w),-1)
            proposal[:,:,3:9]=codec.rotation_quat_to_features(rotated)
        elif variant.startswith('leg'):
            indices=[9+i for i,n in enumerate(kin.joint_order) if any(s in n.lower() for s in ('hip','knee','ankle'))]
            standing=kin.make_standing_qpos().to(base)[7:]
            factor=.02 if variant.endswith('plus') else -.02
            for i in indices:proposal[:,:,i]+=factor*window*(base[:,:,i]-standing[i-9])
        else:
            strength={'smooth05':.05,'smooth10':.1,'smooth20':.2,'identity':0.}[variant]
            smooth=base.clone();smooth[:,1:-1,9:]=.25*base[:,:-2,9:]+.5*base[:,1:-1,9:]+.25*base[:,2:,9:]
            proposal[:,:,9:]+=strength*window[:,:,None]*(smooth[:,:,9:]-base[:,:,9:])
        proposal=torch.where(free,proposal,base)
        # 只接受相对Stage1未新增硬限位的扰动；原模型已有超限不在此被静默修正。
        limits=kin.joint_limits.to(base)
        guard_base=trace.get('diagnostic_bank_baseline',base)
        before=(guard_base[:,:,9:]<limits[:,0])|(guard_base[:,:,9:]>limits[:,1])
        after=(proposal[:,:,9:]<limits[:,0])|(proposal[:,:,9:]>limits[:,1])
        legal=~(after&~before).flatten(1).any(1)
        qbase=codec.decode_to_canonical_qpos(guard_base);qnew=codec.decode_to_canonical_qpos(proposal)
        oldbody=kin.forward_body_positions(qbase);newbody=kin.forward_body_positions(qnew)
        # 额外速度/加速度/FK连续性增量上限，小扰动不通过则整条计划回退原参考。
        delta=proposal-guard_base;velocity=torch.diff(delta[:,:,9:],dim=1)*30
        acceleration=torch.diff(velocity,dim=1)*30
        body_delta=newbody-oldbody
        body_velocity=torch.diff(body_delta,dim=1)*30
        legal &= velocity.abs().flatten(1).amax(1)<.6
        legal &= acceleration.abs().flatten(1).amax(1)<20.
        legal &= body_velocity.abs().flatten(1).amax(1)<.25
        legal &= torch.isfinite(newbody).flatten(1).all(1)
        proposal=torch.where(legal[:,None,None],proposal,base)
        if not torch.equal(proposal[~free],base[~free]):raise AssertionError('Known prefix changed')
        trace['qpos30']=proposal;trace['qpos']=codec.decode_to_canonical_qpos(proposal)
        normalized=(proposal-policy.actor.endecoder.mean.to(proposal))/policy.actor.endecoder.std.to(proposal)
        normalized=torch.where(conditions['future_valid'][...,None],normalized,0.)
        trace['normalized']=normalized;trace['action_x0']=normalized.clone()
        audit['plans']+=len(base);audit['accepted']+=int(legal.sum());audit['rejected']+=int((~legal).sum())
        for name,value in [('joint_velocity_delta_max',velocity),('joint_acceleration_delta_max',acceleration),('body_velocity_delta_max',body_velocity)]:
            audit[name]=max(audit.get(name,0.),float(value.abs().max()))
        return trace
    policy.sample_rollout=sample
    try:yield
    finally:policy.sample_rollout=original


@contextmanager
def frozen_reference_bank(policy,bank,*,variant=None,audit=None,bridge=False):
    """保存Stage1世界参考；干预重用同seed计划自由段，仅重新接入当前已承诺前缀。

    相同任务/决策噪声身份从基线银行取世界qpos和contact，不重新选择更有利的参考。
    若干预存活超过基线失败点而没有后续计划，则明确计数并恢复原Stage1闭环生成；
    不把缺失的反事实参考补造为已配对样本。prepare/commit及插值仍由生产入口执行。
    """
    from . import world_collector
    from .joint_limits import generated_boundaries_tensors,generated_boundary_reports
    original=world_collector.generate_batch
    def generate(current_policy,packets,**kwargs):
        outputs,timing=original(current_policy,packets,**kwargs)
        intervention_started=time.perf_counter()
        for index,(item,packet) in enumerate(zip(outputs,packets)):
            key=int(packet['seed']);trace=item['trace'];generated=item['generated']
            if variant is None:
                bank[key]=dict(qpos_world=np.array(generated['qpos_world'],copy=True),contact=trace['contact'].detach().cpu().clone())
                continue
            device=trace['qpos30'].device;codec=policy.actor.endecoder.codec;kin=policy.actor.endecoder.kinematics
            current=torch.as_tensor(generated['qpos_world'],device=device,dtype=torch.float32)[None]
            if key in bank:
                fixed=torch.as_tensor(bank[key]['qpos_world'],device=device,dtype=torch.float32)[None].clone()
                untouched=fixed.clone()
                prefix=int(generated['prefix_frames']);fixed[:,:prefix]=current[:,:prefix]
                anchor=codec.build_canonical_anchor(current)
                components,_,_=codec.canonicalize(fixed,kin.forward_body_positions(fixed),anchor=anchor)
                physical=codec.assemble_features(components)
                if bridge:
                    original_components,_,_=codec.canonicalize(untouched,kin.forward_body_positions(untouched),anchor=anchor)
                    original_physical=codec.assemble_features(original_components)
                    # 已承诺偏移及其末端斜率在六帧内Hermite衰减，防止每次重规划强行归零。
                    difference=trace['qpos30']-original_physical
                    value=difference[:,prefix-1:prefix];slope=difference[:,prefix-1:prefix]-difference[:,prefix-2:prefix-1]
                    u=((torch.arange(60,device=device)-prefix+1)/6).clamp(0,1)[None,:,None]
                    carry=(2*u**3-3*u**2+1)*value+(u**3-2*u**2+u)*6*slope
                    physical=physical+carry
                    trace['diagnostic_bank_baseline']=original_physical
                trace['qpos30']=torch.where(trace['free_mask'],physical,trace['qpos30'])
                trace['qpos']=codec.decode_to_canonical_qpos(trace['qpos30'])
                trace['contact']=bank[key]['contact'].to(device)
                audit['reused_baseline_plans']=audit.get('reused_baseline_plans',0)+1
            else:
                audit['missing_after_baseline_end']=audit.get('missing_after_baseline_end',0)+1
            saved=policy.sample_rollout
            policy.sample_rollout=lambda conditions,**kw:trace
            try:
                with perturb_reference(policy,variant,audit):
                    trace=policy.sample_rollout(trace['conditions'])
            finally:policy.sample_rollout=saved
            anchors=torch.tensor([item['meta']['world_anchor']],device=device,dtype=torch.float32)
            world=codec.apply_world_anchor(trace['qpos'],anchors)
            boundary=generated_boundaries_tensors(world,trace['free_mask'][:,:,9:],[packet['snapshot']],kin.joint_order)
            generated.update(qpos_world=world[0].cpu().numpy().copy(),qpos30=trace['qpos30'][0].cpu().numpy().copy(),
                contact=trace['contact'][0].cpu().numpy().copy(),joint_limit_diagnostics=generated_boundary_reports({k:v.cpu() for k,v in boundary.items()})[0])
            item['trace']=trace
            block,position=item['_shared_raw_block']
            for field in ('qpos','qpos30','normalized','action_x0','contact'):
                block['trace'][field][position:position+1]=trace[field]
        timing['diagnostic_reference_intervention_seconds']=time.perf_counter()-intervention_started
        return outputs,timing
    world_collector.generate_batch=generate
    try:yield
    finally:world_collector.generate_batch=original


def run_diagnostic(c):
    if c.distributed.world_size!=8 or not c.robofft:raise ValueError('Explicit eight-GPU RoboFFT diagnosis required')
    started=time.time();rank=c.distributed.rank;output=c.output/'decisive'
    output.mkdir(exist_ok=True)
    base={n:t.detach().cpu().clone() for n,t in c.actor.state_dict().items()}
    try:
        print(f'[DECISIVE rank={rank}] collecting true behavior chain',flush=True)
        with capture_true_chain(c.policy):buffer,targets,manifest,collection=c.vector.collect('decisive_behavior',strict=True)
        allrows=buffer.transitions
        from ..robofft.execution_weights import weights_for_rows
        temporal,audits=weights_for_rows(allrows,c.policy.robofft['execution_weighting'])
        eligible=[i for i,r in enumerate(allrows) if bool(targets['valid'][i]) and bool(r.free_mask.any())]
        choose=torch.linspace(0,len(eligible)-1,min(64,len(eligible))).long().tolist()
        positions=[eligible[i] for i in choose];rows=[allrows[i] for i in positions]
        subtargets={k:v[positions] if torch.is_tensor(v) and v.shape==(len(allrows),) else v for k,v in targets.items()}
        cache=RolloutTensorCache(rows,subtargets,c.distributed.device,max_device_bytes=2*1024**3)
        from ..robofft.records import MCRecord
        rng=torch.Generator().manual_seed(20261024+rank)
        with torch.no_grad():
            for i,row in zip(positions,rows):
                mc=c.policy.draw_mc(row.action_x0[None],rng)
                losses=c.policy.losses(cache.context([row]),cache.get('action_x0',[row]),mc['eps'],mc['sampling_indices'],mc['original_timesteps'],time_weights=temporal[i][None])
                row.mc_record=MCRecord(mc['eps'][0],mc['sampling_indices'][0],mc['original_timesteps'][0],losses[0].cpu(),torch.arange(16),time_weights=temporal[i])
        torch.save(dict(rows=rows,targets=subtargets,collection=collection,weight_audits=[audits[i] for i in positions]),output/f'behavior_rank{rank:02d}.pt')
        adv=subtargets['advantages'].double().cpu()
        gathered=c.distributed.all_gather_object(adv.tolist());pool=torch.tensor([v for part in gathered for v in part],dtype=torch.float64)
        shuffled=pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(78443))]
        offset=sum(len(x) for x in gathered[:rank]);permuted=shuffled[offset:offset+len(rows)]
        beta=float(c.policy.robofft['beta_neg'])
        weights=lambda a:torch.where(a<0,beta*a,a)
        mean_weight=float(weights(pool).mean())
        historical_clip_audit(c,rows,cache,base,output)
        directions={};report=dict(actions=len(pool),mean_advantage=float(pool.mean()),mean_effective_advantage=mean_weight,
            beta_neg=beta,score='negative_forward_denoising_loss_scaled_0.1',true_chain='joint_sum_steps9_to18_no_coordinate_mean',
            full_chain_diagnostic_only=True,optimizer_updates=0)
        fit=torch.zeros_like(adv);fit[:len(adv)//2]=2.
        specifications=[('A',weights(adv),'proxy',True),('B',weights(permuted),'proxy',True),
            ('S',torch.ones_like(adv),'proxy',True),('D',adv,'gaussian',True),
            ('beta1',adv,'proxy',True),('uniform',weights(adv),'proxy',False),('Sfit',fit,'proxy',True)]
        for label,coefficient,kind,weighted in specifications:
            print(f'[DECISIVE rank={rank}] gradient {label}',flush=True)
            directions[label]=_gradient(c,rows,cache,coefficient,kind,weighted=weighted)
        generator=torch.Generator().manual_seed(8675309)
        directions['C']={n:torch.randn(v.shape,generator=generator,dtype=v.dtype) for n,v in directions['A'].items()}
        report['cosines']={pair:module_stats(directions[pair[0]],directions[pair[1]]) for pair in ('AB','AD','AS','BD','AC')}
        report['beta1_vs_A']=module_stats(directions['beta1'],directions['A'])
        report['uniform_vs_A']=module_stats(directions['uniform'],directions['A'])
        centered={n:directions['A'][n]-mean_weight*directions['S'][n] for n in directions['A']}
        report['centered_A_vs_A']=module_stats(centered,directions['A'])
        report['score_drift_to_A_norm']=abs(mean_weight)*report['cosines']['AS']['all']['second_norm']/max(report['cosines']['AS']['all']['first_norm'],1e-30)
        reference={}
        for group in c.actor_optimizer.param_groups:
            reference[group['name']]=float(group['lr'])*math.sqrt(sum(p.numel() for p in group['params']))*.5
        report['heldout_score_mean']=heldout_score_audit(c,rows,cache,base,matched_direction(directions['Sfit'],reference),output)
        for key in ('A','B','C','D'):directions[key]=matched_direction(directions[key],reference)
        report['matched_module_update_norms']=reference
        # 每卡负责一个正/反方向，所有方向全局数据与模块范数相同。
        label=('A','B','C','D')[rank//2];sign=1 if rank%2==0 else -1;direction=directions[label]
        probe=rows[:4];initial=replay_outputs(c.policy,probe)
        apply_direction(c.actor,base,direction,sign)
        changed=replay_outputs(c.policy,probe);mask=torch.stack([r.free_mask.cpu() for r in probe])
        delta=(changed-initial).double();probe_report=dict(label=label,sign=sign,
            free_qpos30_rms=float(delta[mask].square().mean().sqrt()),root_xyz_rms=float(delta[:,:,:3].square().mean().sqrt()),
            rotation6d_rms=float(delta[:,:,3:9].square().mean().sqrt()),joint_rms=float(delta[:,:,9:].square().mean().sqrt()))
        c.actor.load_state_dict(base)
        report['probes']=c.distributed.all_gather_object(probe_report)
        # 在不看物理奖励的前提下预先校准一次共同扰动半径，避免数值上不可见的实验。
        a_rms=max(x['joint_rms'] for x in report['probes'] if x['label']=='A')
        largest=max(x['joint_rms'] for x in report['probes'])
        multiplier=min(10.,max(.1,.002/max(a_rms,1e-12)))
        multiplier=min(multiplier,.01/max(largest,1e-12))
        direction={n:v*multiplier for n,v in direction.items()}
        report['common_radius_multiplier']=multiplier
        report['matched_module_update_norms']={k:v*multiplier for k,v in reference.items()}
        root_call(c.distributed,lambda:atomic_json(output/'gradient_directions.json',report))
        torch.save(direction,output/f'direction_rank{rank:02d}.pt')
        del directions,centered
        cache.close();buffer.clear()
        plan=root_call(c.distributed,lambda:_plan(c));root_call(c.distributed,lambda:atomic_json(output/'plan.json',plan))
        c.stage['evaluation']['envs_per_rank']=8;c.stage['evaluation']['persistent_world']=True
        results={}
        results['A_baseline']=evaluate_local_candidates(c,plan,'A_baseline')
        apply_direction(c.actor,base,direction,sign)
        results['A_candidate']=evaluate_local_candidates(c,plan,f'{label}_{"plus" if sign>0 else "minus"}')
        c.actor.load_state_dict(base);del direction
        # 冻结Stage1副本原本用于前9/末1，加载到可训练位置使20步全为相同Stage1。
        stage1={n:t.detach().cpu() for n,t in c.policy.frozen_policy.actor.state_dict().items()}
        c.actor.load_state_dict(stage1)
        bank={}
        with frozen_reference_bank(c.policy,bank):results['C_baseline']=evaluate_local_candidates(c,plan,'C_Stage1')
        variant=('height_plus','height_minus','yaw_plus','yaw_minus','leg_plus','leg_minus','smooth10','smooth20')[rank]
        audit=dict(variant=variant,plans=0,accepted=0,rejected=0)
        with frozen_reference_bank(c.policy,bank,variant=variant,audit=audit):
            results['C_candidate']=evaluate_local_candidates(c,plan,'C_'+variant)
        atomic_json(output/f'results_rank{rank:02d}.json',dict(rank=rank,direction=label,sign=sign,reference_audit=audit,reports=results))
        c.actor.load_state_dict(base)
        all_results=c.distributed.all_gather_object(dict(rank=rank,direction=label,sign=sign,reference_audit=audit,reports=results))
        result=dict(status='passed',results=all_results,wall_seconds=time.time()-started,actor_restored=True,training_updates=0)
        root_call(c.distributed,lambda:atomic_json(output/'summary.json',result))
        return result
    finally:
        c.actor.load_state_dict(base);c.actor.zero_grad(set_to_none=True)


def run_repeatability(c):
    """有限零干预复测：每个Actor重复两次20秒队列，保存初始因果状态供独立比对。

    前四rank为父Actor，后四rank为Stage1；同组权重、音乐、物理reset合同与全部
    采样噪声相同，第二次复用已reset的评估世界。没有训练或方向扰动，专门量化
    持久PhysX世界/数值执行在跨phase重复时的误差；20秒上限不冒充完整音乐验证。
    """
    if c.distributed.world_size!=8 or not c.robofft:raise ValueError('Explicit eight-GPU finite diagnosis required')
    from .evaluation import _digest
    started=time.time();rank=c.distributed.rank;base={n:t.detach().cpu().clone() for n,t in c.actor.state_dict().items()}
    output=c.output/'repeatability';output.mkdir(exist_ok=True)
    try:
        if rank>=4:c.actor.load_state_dict(c.policy.frozen_policy.actor.state_dict())
        plan=root_call(c.distributed,lambda:_plan(c));plan['episode_seconds']=20.
        plan['evaluation_purpose']='zero_update_repeatability_20s_not_complete_music';plan['complete_music_scope']=False
        plan.pop('plan_sha256');plan['plan_sha256']=_digest(plan)
        root_call(c.distributed,lambda:atomic_json(output/'plan.json',plan))
        c.stage['evaluation']['envs_per_rank']=8;c.stage['evaluation']['persistent_world']=True
        first=evaluate_local_candidates(c,plan,'repeatability_first')
        second=evaluate_local_candidates(c,plan,'repeatability_second')
        rows=c.distributed.all_gather_object(dict(rank=rank,actor='parent28' if rank<4 else 'Stage1',first=first,second=second))
        result=dict(status='passed',reports=rows,wall_seconds=time.time()-started,training_updates=0)
        root_call(c.distributed,lambda:atomic_json(output/'summary.json',result));return result
    finally:
        c.actor.load_state_dict(base);c.actor.zero_grad(set_to_none=True)


def run_fresh_physics(c,source):
    """排除持久世界跨phase复位残留：每个候选使用独立新建并关闭的PhysX世界。

    A复用同一真实行为计算且已存档的八个方向，和原首次新建世界基线比较完整音乐；
    C重新生成相同Stage1参考银行，在新世界进行20秒有界配对，以满足总诊断时限。
    不把20秒C结果冒充长音乐或晚期失败修复。保存全部首状态、消费参考及拒绝账本，
    供离线核对新参考生效前的状态和奖励是否逐元素一致。
    """
    import json
    from .evaluation import _digest
    if c.distributed.world_size!=8 or not c.robofft:raise ValueError('Explicit eight-GPU finite diagnosis required')
    started=time.time();rank=c.distributed.rank;base={n:t.detach().cpu().clone() for n,t in c.actor.state_dict().items()}
    output=c.output/'fresh_physics';output.mkdir(exist_ok=True);source=Path(source)
    try:
        plan=json.loads((source/'decisive/plan.json').read_text())
        c.stage['evaluation']['envs_per_rank']=8;c.stage['evaluation']['persistent_world']=False
        label=('A','B','C','D')[rank//2];sign=1 if rank%2==0 else -1
        direction=torch.load(source/'decisive'/f'direction_rank{rank:02d}.pt',map_location='cpu',weights_only=False)
        apply_direction(c.actor,base,direction,sign);del direction
        a=evaluate_local_candidates(c,plan,'fresh_'+label+('_plus' if sign>0 else '_minus'))
        c.actor.load_state_dict(c.policy.frozen_policy.actor.state_dict())
        c.stage['evaluation']['envs_per_rank']=16
        bounded=copy.deepcopy(plan);bounded['episode_seconds']=20.;bounded['complete_music_scope']=False
        bounded['evaluation_purpose']='fresh_world_reference_intervention_20s_not_complete_music'
        bounded.pop('plan_sha256');bounded['plan_sha256']=_digest(bounded)
        root_call(c.distributed,lambda:atomic_json(output/'reference_plan.json',bounded))
        bank={}
        with frozen_reference_bank(c.policy,bank):b=evaluate_local_candidates(c,bounded,'fresh_C_Stage1')
        variant=('height_plus','height_minus','yaw_plus','yaw_minus','leg_plus','leg_minus','smooth10','smooth20')[rank]
        audit=dict(variant=variant,plans=0,accepted=0,rejected=0)
        with frozen_reference_bank(c.policy,bank,variant=variant,audit=audit):
            candidate=evaluate_local_candidates(c,bounded,'fresh_C_'+variant)
        local=dict(rank=rank,direction=label,sign=sign,reference_audit=audit,reports=dict(A_candidate=a,C_baseline=b,C_candidate=candidate))
        atomic_json(output/f'results_rank{rank:02d}.json',local)
        results=c.distributed.all_gather_object(local)
        result=dict(status='passed',results=results,source=str(source),wall_seconds=time.time()-started,training_updates=0,world_reuse=False)
        root_call(c.distributed,lambda:atomic_json(output/'summary.json',result));return result
    finally:
        c.actor.load_state_dict(base);c.actor.zero_grad(set_to_none=True)


def run_reference_only(c,source):
    """单种子八音乐的20秒合法参考诊断，独立新世界、连续前缀桥接及零扰动对照。"""
    import json
    from .evaluation import _digest
    started=time.time();rank=c.distributed.rank;base={n:t.detach().cpu().clone() for n,t in c.actor.state_dict().items()}
    output=c.output/'fresh_physics';output.mkdir(exist_ok=True)
    try:
        plan=json.loads((Path(source)/'decisive/plan.json').read_text());plan['tasks']=[t for t in plan['tasks'] if t['seed']==42]
        plan.update(episode_seconds=20.,task_count=8,seeds=[42],complete_music_scope=False,evaluation_purpose='training_progress',diagnostic_scope='continuous_bank_bridge_20s_one_seed_zero_intervention_control')
        plan.pop('plan_sha256');plan['plan_sha256']=_digest(plan)
        root_call(c.distributed,lambda:atomic_json(output/'reference_plan.json',plan))
        c.stage['evaluation']['envs_per_rank']=8;c.stage['evaluation']['persistent_world']=False
        c.actor.load_state_dict(c.policy.frozen_policy.actor.state_dict());bank={}
        with frozen_reference_bank(c.policy,bank):before=evaluate_local_candidates(c,plan,'fresh_C_Stage1')
        variant=('height_plus','height_minus','yaw_plus','yaw_minus','leg_minus','smooth05','smooth10','identity')[rank]
        audit=dict(variant=variant,plans=0,accepted=0,rejected=0,bridge=True)
        with frozen_reference_bank(c.policy,bank,variant=variant,audit=audit,bridge=True):
            after=evaluate_local_candidates(c,plan,'fresh_C_'+variant)
        row=dict(rank=rank,reference_audit=audit,reports=dict(C_baseline=before,C_candidate=after))
        atomic_json(output/f'results_rank{rank:02d}.json',row)
        result=dict(status='passed',results=c.distributed.all_gather_object(row),wall_seconds=time.time()-started,training_updates=0,world_reuse=False)
        root_call(c.distributed,lambda:atomic_json(output/'summary.json',result));return result
    finally:c.actor.load_state_dict(base);c.actor.zero_grad(set_to_none=True)


def compare_zero_reports(before,after,*,prefix_only=False):
    """严格核验完整零扰动：初态/控制器/缓存、所有奖励误差及生成噪声时钟。"""
    from tools.summarize_decisive_physics import load_report
    _,plan,left=load_report(before);_,other,right=load_report(after)
    if (plan!=other if not prefix_only else plan['tasks']!=other['tasks']):raise ValueError('Zero intervention task plans differ')
    result=[]
    for a,b in zip(left,right):
        if a['task_id']!=b['task_id']:raise ValueError('Zero task order differs')
        checks={k:a.get(k)==b.get(k) for k in ('initial_pairing_state','pairing_clock','pairing_music_sha256',
            'noise_index_start','seed','music_start_frame','physical_failure','end_reason')}
        if prefix_only:
            checks.pop('physical_failure');checks.pop('end_reason')
        checks['initial_present']=a.get('initial_pairing_state') is not None
        ca=[r for d in a['decisions'] for r in d['controls']]
        cb=[r for d in b['decisions'] for r in d['controls']]
        if prefix_only:ca=ca[:len(cb)]
        checks['control_count']=len(ca)==len(cb)
        differences={k:max((abs(x[k]-y[k]) for x,y in zip(ca,cb)),default=0.) for k in ('reward','tick')}
        error=max((abs(x['raw_errors'][k]-y['raw_errors'][k]) for x,y in zip(ca,cb) for k in x['raw_errors']),default=0.)
        checks['all_rewards_exact']=differences['reward']==0.;checks['all_ticks_exact']=differences['tick']==0.
        checks['all_eight_errors_exact']=error==0.
        checks['deployment_clock']=(len(a['decisions'])>=len(b['decisions']) if prefix_only else len(a['decisions'])==len(b['decisions'])) and all(
            x['latency_seconds']==y['latency_seconds'] and x['prefix_frames']==y['prefix_frames']
            for x,y in zip(a['decisions'],b['decisions']))
        result.append(dict(task_id=a['task_id'],passed=all(checks.values()),checks=checks,
            reward_max_difference=differences['reward'],tracking_max_difference=error,controls=len(ca)))
    return dict(passed=all(r['passed'] for r in result),tasks=result,scope='complete_music_exact_zero_intervention')


def run_p0(c,source):
    """每槽单任务、每阶段新世界的Stage1全音乐零扰动门槛，同时采集未来缓存MC。"""
    import json
    started=time.time();output=c.output/'p0';output.mkdir(exist_ok=True)
    base={n:t.detach().cpu().clone() for n,t in c.actor.state_dict().items()}
    try:
        plan=json.loads((Path(source)/'decisive/plan.json').read_text())
        c.stage['evaluation'].update(envs_per_rank=len(plan['tasks']),persistent_world=False)
        c.config['runtime'].update(paired_state_audit=True,future_reference_context=True)
        c.actor.load_state_dict(c.policy.frozen_policy.actor.state_dict())
        before=evaluate_local_candidates(c,plan,'fresh_p0_baseline')
        after=evaluate_local_candidates(c,plan,'fresh_p0_zero')
        result=local_call(c.distributed,lambda:compare_zero_reports(before,after))
        result.update(rank=c.distributed.rank,before=before,after=after)
        atomic_json(output/f'rank{c.distributed.rank:02d}.json',result)
        all_results=c.distributed.all_gather_object(result)
        summary=dict(status='passed' if all(r['passed'] for r in all_results) else 'failed',results=all_results,
            world_reuse=False,tasks_per_slot=1,training_updates=0,wall_seconds=time.time()-started)
        root_call(c.distributed,lambda:atomic_json(output/'summary.json',summary))
        if summary['status']!='passed':raise RuntimeError('P0 zero intervention failed; no physical superiority claim allowed')
        return summary
    finally:c.actor.load_state_dict(base);c.actor.zero_grad(set_to_none=True)


def run_reference_after_p0(c,source,*,local_source=None):
    """只有完整P0和本入口零参考回放通过后才运行合法C；局部先于完整困难音乐。"""
    import json
    from .evaluation import _digest
    from .reference_interventions import fixed_world_intervention,VARIANTS
    source=Path(source);gate=json.loads((source/'p0/summary.json').read_text())
    if gate['status']!='passed':raise ValueError('P0 zero gate has not passed')
    rank=c.distributed.rank;baseline=Path(gate['results'][rank]['before'])
    plan=json.loads((baseline.parent/'selection.json').read_text())
    full=local_source is not None
    if full:
        local=json.loads((Path(local_source)/'reference_c/summary.json').read_text())
        if local['status']!='passed' or not all(r['zero_gate']['passed'] and not r['audit'].get('invalid_plans') for r in local['results']):
            raise ValueError('Local legal-reference/zero gates must pass before full music')
    else:
        plan.update(episode_seconds=20.,complete_music_scope=False)
        plan.pop('plan_sha256');plan['plan_sha256']=_digest(plan)
    data=torch.load(baseline.parent.parent/'causal_references.pt',map_location='cpu',weights_only=False)
    bank={int(r['generated']['seed']):r['generated'] for r in data['references']};del data
    output=c.output/'reference_c';output.mkdir(exist_ok=True)
    c.stage['evaluation'].update(envs_per_rank=len(plan['tasks']),persistent_world=False)
    c.config['runtime'].update(paired_state_audit=True,future_reference_context=True)
    base={n:t.detach().cpu().clone() for n,t in c.actor.state_dict().items()}
    try:
        c.actor.load_state_dict(c.policy.frozen_policy.actor.state_dict())
        if not full:
            zero_audit=dict(plans=0)
            with fixed_world_intervention(c,bank,'identity',zero_audit):zero=evaluate_local_candidates(c,plan,'fresh_C_zero')
            zero_gate=local_call(c.distributed,lambda:compare_zero_reports(baseline,zero,prefix_only=True))
            if not all(c.distributed.all_gather_object(zero_gate['passed'])):raise ValueError('C zero replay failed; no candidate executed')
        else:
            zero_gate=local['results'][rank]['zero_gate'];zero=str(baseline)
        variant=VARIANTS[rank];audit=dict(variant=variant,plans=0,allow_unpaired_extension=full)
        try:
            with fixed_world_intervention(c,bank,variant,audit):
                candidate=evaluate_local_candidates(c,plan,'fresh_C_'+('full_' if full else '')+variant)
        finally:atomic_json(output/f'audit_rank{rank:02d}.json',audit)
        row=dict(rank=rank,variant=variant,zero_gate=zero_gate,audit=audit,baseline=zero,candidate=candidate,
            full_music=full,original_full_baseline=str(baseline))
        atomic_json(output/f'rank{rank:02d}.json',row)
        summary=dict(status='passed',results=c.distributed.all_gather_object(row),full_music=full,
            p0_source=str(source),local_source=None if local_source is None else str(local_source),training_updates=0)
        root_call(c.distributed,lambda:atomic_json(output/'summary.json',summary));return summary
    finally:c.actor.load_state_dict(base);c.actor.zero_grad(set_to_none=True)
