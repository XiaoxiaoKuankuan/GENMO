"""服务器1八卡对同一2048条真实链比较完整Actor与KL计算工作量。

使用只读归档和实际行为模型，先验证全部20步零更新概率。独立执行一次固定PPO
更新获得非空Adam，再对所有候选恢复相同参数、Adam、BC和随机状态。Actor比较
128/256/512微批，每次都覆盖两epoch、两个完整全局2048链minibatch及相同BC。
数值轮捕获全部未裁剪梯度并比较参数/Adam；性能轮不拷贝梯度，预热后重复计时。
KL独立比较256/512/1024，对所有有效链全部20步验收，不混用参数版本缓存。

这是固定数据算子实验，没有物理采集或模型发布；为比较相同工作量，明确禁用
本离线副本的软停止，仍报告最终完整KL及是否满足硬门槛，不能当作正式训练接受。
不重新生成old概率、不降低学习率、不启用失败的TF32/BF16/SDPA候选。每个候选
串行占用同一八卡，结果取最慢rank墙钟。临时解包结束清理，报告按用户要求留存。
"""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.stage10_fixed_inputs import load_rank
from tools.train_closedloop_stage10 import configuration
from tools.verify_stage10_saved_learning import compare_named
from tools.verify_stage10_gradient_reductions import module_directions,terminal_outputs
from gem.closedloop.dppo.trainer import load_actor,trainable_actor_parameters,SupervisedAnchor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.batch_execution import configure_gradients
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.parallel_support import cpu_snapshot
from gem.closedloop.dppo.updater_v2 import actor_update_v2,analytic_kl_local,probability_check_local


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('iteration','weights','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8 or not 3<=a.repeats<=10:raise ValueError('Eight GPUs and 3..10 repeats required')
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('gloo',timeout=timedelta(minutes=40))
    group=dist.new_group(backend='nccl',timeout=timedelta(minutes=20))
    c=DistributedCollectives(rank,8,device=f'cuda:{rank}',tensor_group=group)
    directory=a.output/f'rank{rank:02d}';directory.mkdir(parents=True,exist_ok=False)
    rows,targets,source=load_rank(a.iteration,rank,directory/'unpacked')
    if len(rows)!=256:raise ValueError('Fixed-work acceptance requires 2048 real global chains')
    config=configuration(a.config);config['runtime']['genmo_device']=f'cuda:{rank}'
    settings=config['stage10']['training'];performance=config['stage10']['performance']
    actor,train_config,_=load_actor(config)
    payload=torch.load(a.weights,map_location='cpu',mmap=True,weights_only=False)
    actor.load_state_dict(payload['actor'])
    policy=DPPODiffusionPolicy(actor,steps=20,eta=settings['eta'],std_floor=settings['std_floor'],
        guidance_scale=settings['guidance_scale'],cfg_batch=True,numerical_layout=performance['numerical_layout'],
        defer_checks=True,precision_mode=performance['precision_mode'],numerical_variant=performance['numerical_variant'])
    configure_gradients(actor,weight_reduction=performance['weight_reduction'],
        accumulation=performance['gradient_accumulation'],weight_reduction_overrides=performance['weight_reduction_overrides'])
    if any(r.metadata['sampler_trace']['kernel_config']!=policy.kernel_config for r in rows):
        raise ValueError('Cannot change saved behavior numerical contract')
    manifest=[v for shard in c.all_gather_object([dict(owner_rank=rank,local_index=i,
        valid=r.transition_valid,has_free=bool(r.free_mask.any())) for i,r in enumerate(rows)]) for v in shard]
    selected=[i for i,m in enumerate(manifest) if m['valid'] and m['has_free']]
    cache=RolloutTensorCache(rows,targets,c.device,max_device_bytes=4*1024**3)
    probabilities={str(b):probability_check_local(policy,rows,global_manifest=manifest,distributed=c,
        denoising_microbatch=b,tensor_cache=cache) for b in (128,256,512,1024)}
    optimizer=torch.optim.AdamW(trainable_actor_parameters(actor),lr=5e-9,weight_decay=0.)
    optimizer.load_state_dict(copy.deepcopy(payload['actor_optimizer']))
    config['stage9']['bc_batch']=128//8;config['stage9']['seed']=42+1000003*rank
    bc=SupervisedAnchor(config,actor,train_config);bc.distributed_global_mean=True
    kwargs=dict(global_manifest=manifest,distributed=c,actor_minibatch_internal_transitions=2048*20,
        soft_kl_limit=None,gradient_diagnostics=True,tensor_cache=cache,kl_check_mode='pre_step_plus_final',
        bc=bc,bc_weight=.1,clip=.01,gamma_denoising=.99)
    # 明确是离线的固定Adam预热，不计为真实环境交互，也不覆盖源模型。
    actor_update_v2(policy,optimizer,rows,targets,ppo_epochs=1,epoch_orders=[selected],
        max_optimizer_steps=1,denoising_microbatch=128,**kwargs)
    base=cpu_snapshot(actor.state_dict());adam_base=cpu_snapshot(optimizer.state_dict());bc_base=copy.deepcopy(bc.state_dict())
    if not adam_base['state']:raise ValueError('Nonempty Adam required')
    with torch.no_grad():base_outputs=terminal_outputs(policy,rows,c.device)
    orders=[[selected[i] for i in torch.randperm(len(selected),generator=torch.Generator().manual_seed(445+e)).tolist()] for e in range(2)]
    report=dict(source=source,global_chains=len(manifest),probabilities=probabilities,nonempty_adam=True,
        numerical_contract=policy.kernel_config,actor=[],kl=[],scope='fixed_real_data_no_physics_no_publication',
        soft_stop_disabled_only_for_fixed_work_benchmark=True,actor_lr=5e-9)
    def save():
        if rank==0:(a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    reference=None
    for micro in (128,256,512):
        samples=[];diagnostic=None
        for repeat in range(a.repeats+2):
            actor.load_state_dict(base);optimizer.load_state_dict(copy.deepcopy(adam_base));bc.load_state_dict(copy.deepcopy(bc_base))
            actor.zero_grad(set_to_none=True);captured=[]
            def observer(model,step):
                if rank==0:captured.append({n:None if v.grad is None else v.grad.detach().cpu().clone() for n,v in model.named_parameters()})
            c.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            update=actor_update_v2(policy,optimizer,rows,targets,ppo_epochs=2,epoch_orders=orders,
                max_optimizer_steps=2,denoising_microbatch=micro,gradient_observer=observer if repeat==0 else None,**kwargs)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            shards=c.all_gather_object(dict(seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved()))
            if repeat>=2:samples.append(max(s['seconds'] for s in shards))
            if update['optimizer_steps']!=2 or update['applied_internal_sample_visits']!=81920:raise ValueError('Fixed work changed')
            if repeat==0:
                kl=analytic_kl_local(policy,rows,global_manifest=manifest,distributed=c,denoising_microbatch=256,tensor_cache=cache)
                with torch.no_grad():outputs=terminal_outputs(policy,rows,c.device)
                output_shards=c.all_gather_object(dict(rank=rank,
                    changed_elements=int((outputs!=base_outputs).sum()),
                    max_change=float((outputs-base_outputs).abs().max())))
                if rank==0:
                    weights=cpu_snapshot(actor.state_dict());adam={f'{i}/{k}':v.cpu().clone() for i,state in optimizer.state_dict()['state'].items() for k,v in state.items() if torch.is_tensor(v)}
                    if reference is None:reference=(captured,weights,adam,kl,outputs)
                    diagnostic=dict(gradients=[compare_named(x,y,atol=3e-5,rtol=2e-4) for x,y in zip(captured,reference[0])],
                        modules=[module_directions(x,y) for x,y in zip(captured,reference[0])],
                        parameters=compare_named(weights,reference[1],atol=1e-7,rtol=0.),
                        adam=compare_named(adam,reference[2],atol=1e-7,rtol=2e-4),full_kl=kl,
                        terminal_outputs=compare_named({'mean':outputs},{'mean':reference[4]},atol=1e-7,rtol=0.),
                        actual_policy_change_ranks=output_shards,
                        hard_kl_accepted=kl['mean_joint_kl']<=.03,
                        full_kl_abs_difference=abs(kl['mean_joint_kl']-reference[3]['mean_joint_kl']))
                del captured
        report['actor'].append(dict(microbatch=micro,p50=float(np.percentile(samples,50)),
            p95=float(np.percentile(samples,95)),samples=samples,diagnostic=diagnostic,
            optimizer_steps=2,internal_visits=81920,global_bc_samples=256,memory_ranks=shards))
        save()
    actor.load_state_dict(base)
    kl_reference=None
    for micro in (256,512,1024):
        samples=[]
        for repeat in range(a.repeats+2):
            c.barrier();torch.cuda.synchronize();start=time.perf_counter()
            kl=analytic_kl_local(policy,rows,global_manifest=manifest,distributed=c,denoising_microbatch=micro,tensor_cache=cache)
            torch.cuda.synchronize();seconds=max(c.all_gather_object(time.perf_counter()-start))
            if repeat>=2:samples.append(seconds)
            if kl['fresh_internal_forwards']!=40960:raise ValueError('Incomplete final KL coverage')
        if kl_reference is None:kl_reference=kl
        report['kl'].append(dict(microbatch=micro,p50=float(np.percentile(samples,50)),p95=float(np.percentile(samples,95)),
            samples=samples,full_kl=kl,mean_difference=abs(kl['mean_joint_kl']-kl_reference['mean_joint_kl'])))
        save()
    report['cache']=c.all_gather_object(cache.report());save();cache.close()
    shutil.rmtree(directory/'unpacked')
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
