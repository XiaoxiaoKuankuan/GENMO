"""固定真实诊断链的梯度可靠性与同条件Stage1保真补充检查。

只读取A实验保存的完整真实链、固定MC、父Actor及匹配范数方向；两半环境分别
估计RoboFFT与真实Gaussian梯度，防止把两个高方差估计的近正交误判成算法定论。
另外用拟合半区的Gaussian score均值方向，在留出半区检验score均值；同一组公共
条件和全部创新噪声重放八个候选、父模型和Stage1，按实际消费源帧报告变化。
这里不创建PhysX、不更新Adam/Critic、不覆盖原实验；真实物理结论来自A/C报告。
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.decisive_diagnostic import _gradient,matched_direction,module_stats,apply_direction,replay_outputs,proxy_losses
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.parallel_support import root_call
from gem.closedloop.dppo.budget import atomic_json


@torch.no_grad()
def gaussian_ratios(c,rows,cache):
    from gem.closedloop.dppo.gaussian_probability import gaussian_statistics
    values=[];kls=[];maximum_mean=maximum_log=0.
    with policy_phase(c.policy):
        for start in range(0,len(rows),4):
            selected=rows[start:start+4];condition=cache.context(selected);total=torch.zeros(len(selected),device=c.distributed.device,dtype=torch.float64);kl=torch.zeros_like(total)
            for step in range(9,19):
                traces=[r.metadata['sampler_trace'] for r in selected]
                get=lambda field,k:torch.cat([t['diagnostic_'+field][:,k] for t in traces]).to(c.distributed.device)
                params=c.policy.transition_parameters(condition,get('chain',step),step)
                old=get('old_means',step);std=get('old_stds',step)
                result=gaussian_statistics(get('chain',step+1),old,std,params['mean'],params['std'],params['free_mask'])
                total+=result['joint_log_ratio']
                kl+=result['joint_kl']
                maximum_mean=max(maximum_mean,float((params['mean']-old).abs().max()))
            maximum_log=max(maximum_log,float(total.abs().max()));values.extend(total.cpu().tolist());kls.extend(kl.cpu().tolist())
    return torch.tensor(values,dtype=torch.float64),dict(maximum_mean_difference=maximum_mean,maximum_joint_logratio=maximum_log,
        mean_conditional_chain_KL=float(np.mean(kls)),per_action_conditional_chain_KL=kls)


def physical_differences(a,b,masks,codec):
    delta=(a-b).double();q1=codec.decode_to_canonical_qpos(a);q0=codec.decode_to_canonical_qpos(b)
    selected=masks.any(-1);qa=torch.nn.functional.normalize(q1[:,:,3:7].double(),dim=-1);qb=torch.nn.functional.normalize(q0[:,:,3:7].double(),dim=-1)
    dot=(qa*qb).sum(-1).abs().clamp(0,1)
    angle=2*dot.acos();position=(q1[:,:,:3]-q0[:,:,:3]).double().square().sum(-1)
    legs=[9+i for i,n in enumerate(codec.kinematics.joint_order) if any(k in n.lower() for k in ('hip','knee','ankle'))]
    return dict(free_qpos30_rms=float(delta[masks].square().mean().sqrt()),
        root_position_rms_m=float(position[selected].mean().sqrt()),root_orientation_rms_rad=float(angle[selected].square().mean().sqrt()),
        leg_rms_rad=float(delta[:,:,legs][selected].square().mean().sqrt()),
        joint_velocity_rms_ratio=float(torch.diff(a[:,:,9:],dim=1).square().mean().sqrt()/torch.diff(b[:,:,9:],dim=1).square().mean().sqrt()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8 or rank!=local:raise ValueError('Requires explicitly allocated eight GPUs')
    torch.set_num_threads(3);torch.cuda.set_device(local)
    dist.init_process_group('gloo',timeout=timedelta(minutes=20));group=dist.new_group(backend='nccl',timeout=timedelta(minutes=20))
    collective=DistributedCollectives(rank,world,tensor_group=group,device=f'cuda:{local}')
    root_call(collective,lambda:args.output_dir.mkdir(parents=True,exist_ok=False))
    config=configuration(args.config,genmo_repo=ROOT);config['runtime']['genmo_device']=str(collective.device)
    actor,_,_=load_actor(config);policy=build_policy(actor,config['stage9'],config['stage10']['precision'])
    payload=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    actor.load_state_dict(payload['actor']);base={n:t.detach().cpu().clone() for n,t in actor.state_dict().items()};del payload
    c=SimpleNamespace(actor=actor,policy=policy,distributed=collective)
    source=args.run/'decisive';data=torch.load(source/f'behavior_rank{rank:02d}.pt',map_location='cpu',weights_only=False)
    rows=data['rows'];targets=data['targets'];cache=RolloutTensorCache(rows,targets,collective.device,max_device_bytes=2*1024**3)
    zero,zero_metrics=gaussian_ratios(c,rows,cache)
    report=dict(behavior_actions=sum(collective.all_gather_object(len(rows))),gaussian_zero_replay=collective.all_gather_object(zero_metrics),split={})
    grads={}
    for half in range(2):
        begin=0 if half==0 else len(rows)//2;end=len(rows)//2 if half==0 else len(rows)
        chosen=rows[begin:end];sub={k:v[begin:end] if torch.is_tensor(v) and v.shape==(len(rows),) else v for k,v in targets.items()}
        part=RolloutTensorCache(chosen,sub,collective.device,max_device_bytes=2*1024**3);adv=sub['advantages'].double().cpu()
        for label,kind,coeff in [('A','proxy',torch.where(adv<0,.5*adv,adv)),('D','gaussian',adv)]:
            print(f'[RELIABILITY rank={rank}] {label} half={half}',flush=True)
            grads[label+str(half)]=_gradient(c,chosen,part,coeff,kind)
        if half==0:grads['S_D']=_gradient(c,chosen,part,torch.ones_like(adv),'gaussian')
        part.close()
    for label in ('A','D'):report['split'][label]=module_stats(grads[label+'0'],grads[label+'1'])
    norms=json.loads((source/'gradient_directions.json').read_text())['matched_module_update_norms']
    score_direction=matched_direction(grads['S_D'],norms);del grads
    projections=[]
    for sign in (1,-1):
        apply_direction(actor,base,score_direction,sign)
        projected,_=gaussian_ratios(c,rows,cache);projections.append(projected)
    actor.load_state_dict(base);projection=(projections[0]-projections[1])/2
    local_rows=[]
    for index in range(len(rows)//2,len(rows)):
        task=rows[index].metadata['training_task']
        local_rows.append(dict(source=task['dataset'],music=task['sample_id'],value=float(projection[index])))
    allrows=[v for part in collective.all_gather_object(local_rows) for v in part]
    music={}
    for row in allrows:music.setdefault((row['source'],row['music']),[]).append(row['value'])
    mean=np.array([np.mean(v) for v in music.values()]);rng=np.random.default_rng(20261024)
    boot=rng.choice(mean,(10000,len(mean)),replace=True).mean(1)
    report['true_gaussian_heldout_score']=dict(mean=float(mean.mean()),music_cluster_95=np.quantile(boot,[.025,.975]).tolist(),music_clusters=len(mean),states=len(allrows))
    common=torch.load(source/'behavior_rank00.pt',map_location='cpu',weights_only=False)
    probes=common['rows'][:8];masks=torch.stack([r.free_mask.cpu() for r in probes]);current=replay_outputs(policy,probes)
    original=torch.cat([r.metadata['sampler_trace']['qpos30'].cpu() for r in probes])
    actor.load_state_dict(policy.frozen_policy.actor.state_dict());teacher=replay_outputs(policy,probes);actor.load_state_dict(base)
    direction=torch.load(source/f'direction_rank{rank:02d}.pt',map_location='cpu',weights_only=False)
    sign=1 if rank%2==0 else -1;apply_direction(actor,base,direction,sign)
    candidate=replay_outputs(policy,probes)
    chain_ratio,chain_stat=gaussian_ratios(c,rows,cache)
    old_loss=torch.stack([r.mc_record.old_loss for r in rows]);new_loss=proxy_losses(c,rows,cache)
    raw=(old_loss-new_loss)*policy.robofft['logprob_normalizer']
    proxy_ratio=raw.clamp(-10,10).mean(1)*policy.robofft['logratio_alpha']
    actor.load_state_dict(base)
    consumed=masks.clone()
    for i,audit in enumerate(common['weight_audits'][:8]):
        frames=torch.zeros(60,dtype=torch.bool);frames[audit['current_generated_frames']]=True;consumed[i]&=frames[:,None]
    codec=policy.actor.endecoder.codec
    difference=dict(rank=rank,direction=('A','B','C','D')[rank//2],sign=sign,
        zero_update_replay_maximum=float((current-original).abs().max()),
        all_free_change=physical_differences(candidate,current,masks,codec),
        actually_consumed_change=physical_differences(candidate,current,consumed,codec),
        parent_stage1_fidelity=physical_differences(current,teacher,consumed,codec),
        candidate_stage1_fidelity=physical_differences(candidate,teacher,consumed,codec),
        conditional_chain_statistics=chain_stat,
        proxy_and_joint_logratio=dict(correlation=float(np.corrcoef(proxy_ratio.numpy(),chain_ratio.numpy())[0,1]),
            proxy_mean=float(proxy_ratio.mean()),joint_mean=float(chain_ratio.mean()),proxy_std=float(proxy_ratio.std()),joint_std=float(chain_ratio.std()),
            scope='same_old_chain_conditionals; joint_latent_ratio_is_not_explicit_final_action_density'),
        parameter_update_l2=math.sqrt(sum(float(v.double().square().sum()) for v in direction.values())))
    report['fixed_common_inputs']=collective.all_gather_object(difference)
    root_call(collective,lambda:atomic_json(args.output_dir/'summary.json',report))
    cache.close();dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
