"""服务器八卡真实旧rollout的完整链PPO概率、梯度与学习率标定入口。

输入为已有不可变fixed_data_validation目录及当前显式配置。先核对原始文件SHA，
重建冻结Stage1先验并加载采样时Actor，使用原状态/创新噪声/均值/实际标准差/mask，
验证零更新概率、FP64独立Normal公式、完整Autograd与流式微批梯度、多卡SUM。
随后各学习率从相同Actor/Adam/BC/控制器/RNG起点执行有限完整链更新，保存所有
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
import math
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
from gem.closedloop.dppo.trainer import load_actor,trainable_actor_parameters
from gem.closedloop.dppo.training_checkpoint import file_sha256,immutable_cpu_snapshot
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.updater import actor_update,final_kl
from gem.closedloop.dppo.actor_backtracking import check_candidate_kl


def candidate_update_checks(report,kl,config):
    """分开报告硬KL和正式目标感知条件；离线候选不会提交到正式训练计数。

    固定LR试探故意不回溯，可能KL硬门槛通过但PPO目标下降。按生产原有margin、
    soft KL、active clip和目标条件重新判读；这里只解释结果，不改变本次真实更新。
    """
    hard=check_candidate_kl(kl,config['stage10']['kl'])
    settings=config['stage9'];trust=settings['trust_region']
    limits={k:v*trust['margin'] for k,v in config['stage10']['kl'].items() if k.startswith('chain_joint_')}
    limits['chain_joint_mean']=min(limits['chain_joint_mean'],settings['kl_soft_stop_chain_joint']*trust['margin'])
    limits['maximum_active_clip']=settings['max_active_clip_fraction']
    eligible=check_candidate_kl(kl,limits)
    if report['optimizer_steps']<1:eligible['failures']['no_optimizer_step']=True
    if trust.get('objective_aware'):
        for step in report['steps']:
            before=step.get('full_rollout_before');after=step.get('full_rollout_after')
            if before is None or after is None:
                eligible['failures']['objective_evidence_missing']=True
            else:
                step_check=check_candidate_kl(after,limits)
                if step_check['failures']:eligible['failures'][f'step_{step["optimizer_step"]}']=step_check['failures']
                if after['objective_mean']<before['objective_mean']-trust.get('objective_tolerance',1e-10):
                    eligible['failures']['objective_gain']=dict(value=after['objective_mean']-before['objective_mean'],
                        optimizer_step=step['optimizer_step'])
    eligible['passed']=not eligible['failures']
    eligible['scope']='existing_production_conditions_diagnostic_only_no_formal_step_committed'
    return hard,eligible


@torch.no_grad()
def rejected_probability_diagnostics(policy,rows,cache,manifest,distributed,microbatch):
    """高LR指数溢出时只报告真实log-ratio/KL，不截断概率、不把拒绝更新当作成功。"""
    from gem.closedloop.dppo.gaussian_probability import distribution
    selected=[rows[m['local_index']] for m in manifest
        if m['owner_rank']==distributed.rank and m['valid'] and m['has_free']]
    with policy_phase(policy):
        records=evaluate_chain_records(policy,selected,cache,microbatch=microbatch)
    local=torch.stack((records[:,:,0].sum(1),records[:,:,2].sum(1)),1).cpu()
    full=torch.cat(distributed.all_gather_object(local),0)
    return dict(chain_joint_kl=distribution(full[:,0]),chain_joint_log_ratio=distribution(full[:,1]),
        gaussian_steps=list(policy.trainable_steps),chains=len(full),
        ratio_underflow_fraction=float((full[:,1]<math.log(float.fromhex('0x0.0000000000001p-1022'))).double().mean()),
        ratio_overflow_fraction=float((full[:,1]>math.log(torch.finfo(torch.float64).max)).double().mean()),
        accepted=False,scope='unclamped_Gaussian_statistics_after_rejected_Adam_attempt')


def gradient_validation(policy,rows,cache,distributed):
    selected=[row for row in rows if row.free_mask.any()][:2]
    denominator=sum(distributed.all_gather_object(len(selected)))
    actor=policy.actor;device=distributed.device
    actor.zero_grad(set_to_none=True)
    with policy_phase(policy):
        terms=[]
        for step in policy.trainable_steps:
            params,mask=_parameters(policy,selected,[step]*len(selected),device,cache)
            old_mean,old_std=_old_kernel(selected,[step]*len(selected),device,cache)
            value=cache.get('chain',selected,[step+1]*len(selected)).double()
            a=torch.distributions.Normal(params['mean'].double(),params['std'].double())
            b=torch.distributions.Normal(old_mean,old_std)
            terms.append((a.log_prob(value)-b.log_prob(value)).masked_fill(~mask,0.).sum((-2,-1)))
        log=torch.stack(terms,1).sum(1);adv=cache.get('advantages',selected).double()
        loss=-torch.minimum(log.exp()*adv,log.exp().clamp(.9,1.1)*adv).sum()/denominator
        loss.backward()
    distributed.sum_gradients(actor)
    reference={n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
    variants=[]
    for micro in (1,3,8):
        actor.zero_grad(set_to_none=True)
        with policy_phase(policy):
            records=evaluate_chain_records(policy,selected,cache,microbatch=micro)
            backward_chain_surrogate(policy,selected,cache,records,clip=.1,
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
        reference=f'independent_FP64_Normal_full_{len(policy.trainable_steps)}_step_Autograd_global_SUM',
        scope='actual_stage1_network_and_real_behavior_rows_not_physics_quality')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/closedloop/stage2_random20s_gamma095_server2.yaml')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--learning-rates',type=float,nargs='+',default=[1e-6,3e-6,1e-5])
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--max-optimizer-steps',type=int,help='有限固定数据比较可限制优化步数，保留实际完整链Mini-batch')
    parser.add_argument('--adaptive-kl',action='store_true',help='实际联合KL回溯；各LR为候选上限而非承诺接受步长')
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
        # 新加入的训练时刻沿用原真实噪声；记录实际sigma，不能把floor当成全部噪声。
        first=[row for row in rows if row.free_mask.any()][:1]
        local_std={str(step):float(cache.get('old_stds',first,[step]).double().mean())
            for step in policy.trainable_steps} if first else {}
        probability['actual_std_first_eligible_chain_by_rank']=distributed.all_gather_object(local_std)
        probability['actual_std_scope']='normalized_Gaussian_sigma_not_floor_not_trajectory_diversity'
        gradients=gradient_validation(policy,rows,cache,distributed)
        root_call(distributed,lambda:atomic_json(args.output_dir/'probability_and_gradient.json',dict(probability=probability,gradients=gradients)))
        optimizer=torch.optim.AdamW(trainable_actor_parameters(actor),lr=args.learning_rates[0],weight_decay=0.)
        optimizer.load_state_dict(learning['actor_optimizer'])
        controller=BCGradientController(**config['stage10']['bc_gradient'])
        from gem.closedloop.dppo.continuation import DisabledBC
        anchor=(Stage1RetentionAnchor(config,actor,train,source_global_step=loading['source_global_step'])
            if config['stage9'].get('bc_enabled',True) else DisabledBC())
        controller.load_state_dict(learning['controller'])
        rank_state=torch.load(args.case/f'rank{rank:02d}'/'rank_state.pt',map_location='cpu',weights_only=False)
        generators=dict(actor=torch.Generator(),critic=torch.Generator())
        anchor.load_state_dict(rank_state['bc'])
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
            optimizer.param_groups[0]['lr']=lr
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats(distributed.device)
            try:
                trust=copy.deepcopy(config['stage9']['trust_region']) if args.adaptive_kl else None
                if trust is not None:trust['maximum_lr']=lr
                report=actor_update(policy,optimizer,rows,targets,cache,manifest,controller,distributed=distributed,
                    anchor=anchor,generator=generators['actor'],ppo_epochs=args.epochs,minibatch_chains=config['stage10']['scale']['actor_minibatch_chains'],
                    sampled_steps=len(policy.trainable_steps),step_strategy=f'intermediate{len(policy.trainable_steps)}',
                    max_optimizer_steps=args.max_optimizer_steps,audit_full_rollout_objective=True,
                    detailed_diagnostics=True,
                    microbatch=config['stage9']['denoising_microbatch'],clip=config['stage9']['ppo_clip'],soft_kl=config['stage9']['kl_soft_stop_chain_joint'],
                    bc_samples_per_rollout=config['stage9']['bc_samples_per_rollout'],
                    bc_min_global_batch=config['stage9']['bc_min_global_batch'],
                    bc_enabled=config['stage9'].get('bc_enabled',True),
                    max_active_clip=config['stage9']['max_active_clip_fraction'],
                    trust_region=trust,kl_limits=config['stage10']['kl'])
                kl=final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=128,clip=.1)
                hard,acceptance=candidate_update_checks(report,kl,config)
                report['accepted_optimizer_steps']=0
                report['diagnostic_optimizer_steps']=report['optimizer_steps']
                trial=dict(learning_rate=lr,epochs=args.epochs,actor=report,kl=kl,acceptance=acceptance,
                    hard_kl_acceptance=hard,formal_optimizer_steps_committed=0,
                    start_restore=audit,wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)),
                    peak_allocated_bytes=max(distributed.all_gather_object(torch.cuda.max_memory_allocated(distributed.device))))
            except (FloatingPointError,RuntimeError) as error:
                message=str(error)
                if not any(word in message.lower() for word in ('ppo','nonfinite','overflow','underflow','streaming')):raise
                trial=dict(learning_rate=lr,epochs=args.epochs,acceptance=dict(passed=False,
                    failures={'numerical':message}),start_restore=audit,
                    hard_kl_acceptance=None,formal_optimizer_steps_committed=0,
                    rejected_probability=rejected_probability_diagnostics(policy,rows,cache,manifest,distributed,
                        config['stage9']['denoising_microbatch']),
                    wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)))
            trials.append(trial)
            root_call(distributed,lambda:atomic_json(args.output_dir/f'candidate_lr{lr:.8g}_epoch{args.epochs}.json',trial))
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
