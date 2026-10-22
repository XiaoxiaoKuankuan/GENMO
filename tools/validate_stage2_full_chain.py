"""服务器八卡真实旧rollout的逐步坐标平均代理PPO数值与学习率验证。

输入为已有不可变fixed_data_validation目录及当前显式配置。先核对原始文件SHA，
重建冻结Stage1先验并加载采样时Actor，使用原状态/创新噪声/均值/实际标准差/mask，
验证零更新概率、FP64独立Normal公式、完整Autograd与流式微批梯度、多卡SUM。
四步使用同一优势，各自joint_log_ratio/D、各自裁剪，再按4*N归一化；文件名保留
历史路径，不代表当前仍使用整链PPO。随后各学习率从相同非空分组Adam、Actor、
BC控制状态和原rank RNG起点执行有限更新，BC保持关闭，保存所有
接受、拒绝、裁剪、参数变化、输出变化和耗时；候选之间完整恢复并核验字节。

本入口不创建Isaac世界、不产生新物理交互、不改变旧行为数据，也不把数值通过
当作物理收益。旧rollout的Critic/GAE目标仅用于同数据数值对照，不作为新策略MC。
输出必须是新的独立目录。原rollout、模型及正式日志只读；复用同一个脚本研究不同
LR/Epoch，不复制实验脚本。生产仍通过真实闭环训练和完整音乐配对评估判断质量。
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
import torch.distributed as dist

from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,backward_chain_surrogate
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.optimization_support import _parameters,_old_kernel,probability_check_local
from gem.closedloop.dppo.parallel_support import capture_local_rng,restore_local_rng,local_call,root_call
from gem.closedloop.dppo.retention_anchor import Stage1RetentionAnchor
from gem.closedloop.dppo.rollback_audit import compare_recovered_state
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer,set_actor_base_lr,validate_actor_optimizer
from gem.closedloop.dppo.training_checkpoint import file_sha256,immutable_cpu_snapshot
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.updater import actor_update,final_kl


def gradient_validation(policy,rows,cache,distributed,*,clip=.01,microbatches=(1,3,8)):
    selected=[row for row in rows if row.free_mask.any()][:2]
    denominator=sum(distributed.all_gather_object(len(selected)))
    actor=policy.actor;device=distributed.device
    actor.zero_grad(set_to_none=True)
    with policy_phase(policy):
        terms=[];counts=[]
        for step in policy.trainable_steps:
            params,mask=_parameters(policy,selected,[step]*len(selected),device,cache)
            old_mean,old_std=_old_kernel(selected,[step]*len(selected),device,cache)
            value=cache.get('chain',selected,[step+1]*len(selected)).double()
            a=torch.distributions.Normal(params['mean'].double(),params['std'].double())
            b=torch.distributions.Normal(old_mean,old_std)
            terms.append((a.log_prob(value)-b.log_prob(value)).masked_fill(~mask,0.).sum((-2,-1)))
            counts.append(mask.sum((-2,-1)))
        count=torch.stack(counts,1)
        log=torch.stack(terms,1)/count.clamp_min(1);adv=cache.get('advantages',selected).double()[:,None]
        objective=torch.minimum(log.exp()*adv,log.exp().clamp(1-clip,1+clip)*adv)
        loss=-objective.masked_fill(count==0,0.).sum()/(4*denominator)
        loss.backward()
    distributed.sum_gradients(actor)
    reference={n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
    variants=[]
    for micro in microbatches:
        actor.zero_grad(set_to_none=True)
        with policy_phase(policy):
            records=evaluate_chain_records(policy,selected,cache,microbatch=micro)
            backward_chain_surrogate(policy,selected,cache,records,clip=clip,
                global_chain_count=denominator,microbatch=micro)
        distributed.sum_gradients(actor)
        delta=sum((p.grad.double()-reference[n].double()).square().sum()
            for n,p in actor.named_parameters() if n in reference)
        scale=sum(g.double().square().sum() for g in reference.values())
        relative=float((delta/scale).sqrt())
        maximum=max(float((p.grad-reference[n]).abs().max()) for n,p in actor.named_parameters() if n in reference)
        variants.append(dict(microbatch=micro,relative_gradient_l2=relative,maximum_coordinate_error=maximum))
        if relative>2e-5:raise AssertionError('Real GPU streamed gradient mismatch: '+repr(variants[-1]))
    actor.zero_grad(set_to_none=True)
    return dict(passed=True,global_chains=denominator,variants=variants,
        objective='stepwise_coordinate_mean_clipped_ppo',ratio_is_joint_probability=False,
        reference='independent_FP64_Normal_four_independent_clips_full_Autograd_global_SUM',
        scope='actual_stage1_network_and_real_behavior_rows_not_physics_quality')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/closedloop/stage2_stepwise_mean.yaml')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--learning-rates',type=float,nargs='+',default=[1e-6,3e-6,1e-5])
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--gradient-only',action='store_true',help='只核验真实旧链概率及独立完整Autograd/流式梯度，不执行任何Adam')
    args=parser.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8 or rank!=local:raise ValueError('Requires one explicitly allocated eight-GPU node')
    dist.init_process_group('gloo',timeout=timedelta(minutes=30));group=None;cache=None
    try:
        check=[None]
        if rank==0:
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            check[0]=dict(ok=not busy and not args.output_dir.exists(),busy=busy)
        dist.broadcast_object_list(check,src=0)
        if not check[0]['ok']:raise RuntimeError('Fresh output directory and free GPUs required: '+repr(check[0]))
        torch.cuda.set_device(local);torch.set_num_threads(4)
        group=dist.new_group(backend='nccl',timeout=timedelta(minutes=20))
        distributed=DistributedCollectives(rank,world,tensor_group=group,device=f'cuda:{local}')
        root_call(distributed,lambda:args.output_dir.mkdir(parents=True,exist_ok=False))
        config=configuration(args.config,genmo_repo=ROOT)
        from gem.closedloop.dppo.actor_optimization import require_current_actor_objective
        require_current_actor_objective(config['stage9'])
        config['runtime']['genmo_device']=str(distributed.device)
        config['stage9']['bc_batch']=config['stage9']['bc_batch']//world
        config['stage9']['seed']+=1000003*rank
        actor,train,loading=local_call(distributed,lambda:load_actor(config))
        policy=build_policy(actor,config['stage9'],config['stage10']['precision'])
        header=json.loads((args.case/'case.json').read_text())
        learning_path=args.case/'learning_state.pt'
        if file_sha256(learning_path)!=header['learning_state_sha256']:raise ValueError('Behavior Actor file SHA differs')
        learning=torch.load(learning_path,map_location='cpu',weights_only=False,mmap=True)
        actor.load_state_dict(learning['actor'])
        payload=local_call(distributed,lambda:load_rollout(args.case/f'rank{rank:02d}'))
        if payload['kernel']!=policy.kernel_config:
            raise ValueError('Actual current Gaussian behavior kernel differs from the saved real rollout')
        rows,targets,manifest=payload['transitions'],payload['targets'],payload['global_manifest']
        cache=RolloutTensorCache(rows,targets,distributed.device,max_device_bytes=4*1024**3)
        probability=probability_check_local(policy,rows,global_manifest=manifest,distributed=distributed,
            denoising_microbatch=128,tensor_cache=cache,include_frozen=True)
        gradients=gradient_validation(policy,rows,cache,distributed,clip=config['stage9']['ppo_clip'])
        root_call(distributed,lambda:atomic_json(args.output_dir/'probability_and_gradient.json',dict(probability=probability,gradients=gradients)))
        if args.gradient_only:
            root_call(distributed,lambda:atomic_json(args.output_dir/'summary.json',dict(status='passed',
                scope='fixed_real_data_stepwise_probability_and_gradient_no_Adam_no_physics',
                case=str(args.case),case_sha256=file_sha256(args.case/'case.json'),
                probability=probability,gradient_validation=gradients)))
            return
        optimizer=make_actor_optimizer(actor,config['stage10']['training'])
        optimizer.load_state_dict(learning['actor_optimizer'])
        validate_actor_optimizer(actor,optimizer)
        controller=BCGradientController(**config['stage10']['bc_gradient'])
        controller.load_state_dict(learning['controller'])
        anchor=Stage1RetentionAnchor(config,actor,train,source_global_step=loading['source_global_step'])
        rank_state=torch.load(args.case/f'rank{rank:02d}'/'rank_state.pt',map_location='cpu',weights_only=False)
        anchor.load_state_dict(rank_state['bc'])
        generators={'actor':torch.Generator()}
        restore_local_rng(rank_state['rng'],generators)
        def snapshot():return immutable_cpu_snapshot(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),
            controller=controller.state_dict(),bc=anchor.state_dict(),rng=capture_local_rng(generators)))
        initial=snapshot();trials=[]
        for lr in args.learning_rates:
            actor.load_state_dict(initial['actor']);optimizer.load_state_dict(copy.deepcopy(initial['optimizer']))
            controller.load_state_dict(initial['controller']);anchor.load_state_dict(initial['bc'])
            restore_local_rng(initial['rng'],generators)
            audit=compare_recovered_state(initial,snapshot())
            if not audit['identical']:raise AssertionError('Candidate start restore failed: '+repr(audit))
            set_actor_base_lr(optimizer,lr)
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats(distributed.device)
            try:
                trust=copy.deepcopy(config['stage9']['trust_region']);trust['maximum_lr']=lr
                report=actor_update(policy,optimizer,rows,targets,cache,manifest,controller,distributed=distributed,
                    anchor=anchor,generator=generators['actor'],ppo_epochs=args.epochs,
                    minibatch_chains=config['stage10']['scale']['actor_minibatch_chains'],
                    microbatch=128,clip=config['stage9']['ppo_clip'],soft_kl=config['stage9']['kl_soft_stop_mean_coordinate'],
                    bc_samples_per_rollout=2048*args.epochs,bc_min_global_batch=256,
                    max_active_clip=config['stage9']['max_active_clip_fraction'],
                    trust_region=trust,kl_limits=config['stage10']['kl'],bc_enabled=False,
                    audit_full_rollout_objective=True)
                kl=final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=128,clip=config['stage9']['ppo_clip'])
                from gem.closedloop.dppo.training import _final_actor_acceptance
                acceptance=_final_actor_acceptance(report,kl,config['stage10']['kl'])
                report['accepted_optimizer_steps']=report['optimizer_steps'] if acceptance['passed'] else 0
                trial=dict(learning_rate=lr,epochs=args.epochs,actor=report,kl=kl,acceptance=acceptance,
                    start_restore=audit,wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)),
                    peak_allocated_bytes=max(distributed.all_gather_object(torch.cuda.max_memory_allocated(distributed.device))))
            except (FloatingPointError,RuntimeError) as error:
                message=str(error)
                if not any(word in message.lower() for word in ('ppo','nonfinite','overflow','underflow','streaming')):raise
                trial=dict(learning_rate=lr,epochs=args.epochs,acceptance=dict(passed=False,
                    failures={'numerical':message}),start_restore=audit,
                    wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)))
            trials.append(trial)
            root_call(distributed,lambda:atomic_json(args.output_dir/f'candidate_lr{lr:.0e}_epoch{args.epochs}.json',trial))
            if rank==0:print(json.dumps(dict(lr=lr,accepted=trial['acceptance']['passed'],
                steps=trial.get('actor',{}).get('optimizer_steps'),
                joint_kl=trial.get('kl',{}).get('chain_joint_kl'),
                failures=trial['acceptance']['failures']),ensure_ascii=False),flush=True)
        actor.load_state_dict(initial['actor']);optimizer.load_state_dict(copy.deepcopy(initial['optimizer']))
        controller.load_state_dict(initial['controller']);anchor.load_state_dict(initial['bc']);restore_local_rng(initial['rng'],generators)
        restored=compare_recovered_state(initial,snapshot())
        if not restored['identical']:raise AssertionError(restored)
        root_call(distributed,lambda:atomic_json(args.output_dir/'summary.json',dict(status='passed',
            scope='fixed_real_data_math_and_KL_calibration_not_physical_acceptance',
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            case=str(args.case),case_sha256=file_sha256(args.case/'case.json'),
            probability=probability,gradient_validation=gradients,trials=trials,final_restore=restored)))
    finally:
        if cache is not None:cache.close()
        if group is not None:dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__=='__main__':main()
