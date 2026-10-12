"""服务器1八卡：同一真实Stage10 v3 rollout的完整梯度、Adam、KL及吞吐对照。

输入必须来自collect-only或明确capture的真实Isaac闭环，并核验学习状态、rank
状态、rollout的SHA和实际行为核。不会改写old statistics，不重新采样物理任务。
若原Adam为空，先在此离线副本执行一次真实梯度更新以建立非空动量，再把模型、
Adam、BC/RNG和权重控制器冻结为全部候选的共同起点；此步不是训练接受轮次。

数值轮保存全部未裁剪合成梯度，核对每个模块方向、参数、Adam、末端输出及全20步
KL。计时轮不复制完整梯度，预热后比较同一完整工作量而不是单microbatch延迟。
本离线对照明确关闭软停止以保持更新覆盖一致，最终仍报告原硬KL是否通过；不把
超限的离线副本发布为正式策略。不同精度的迁移KL与自身概率检查分开，BF16和
编译候选即使更快也必须另有真实闭环效果，不能由此脚本自动进入正式训练。
"""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from gem.closedloop.dppo.checkpoint_v3 import file_sha256,immutable_cpu_snapshot
from gem.closedloop.dppo.diagnostic_rollout_v3 import load_rollout
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.gradient_merge_v3 import BCGradientController
from gem.closedloop.dppo.parallel_support import capture_local_rng,restore_local_rng,local_call,root_call
from gem.closedloop.dppo.policy_v3 import DenseDiffusionPolicy
from gem.closedloop.dppo.retention_anchor_v3 import Stage1RetentionAnchor
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor,trainable_actor_parameters
from gem.closedloop.dppo.updater_v2 import probability_check_local
from gem.closedloop.dppo.updater_v3 import actor_update_v3,final_kl_v3,check_kl_v3
from tools.verify_stage10_gradient_reductions import module_directions,terminal_outputs
from tools.verify_stage10_saved_learning import compare_named


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--microbatches',type=int,nargs='+',default=[64,128,256])
    p.add_argument('--kl-microbatches',type=int,nargs='+',default=[256,512,1024])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--sampled-steps',type=int,choices=[8,20],default=8)
    p.add_argument('--variants',nargs='+',choices=['reference','bf16_cudnn','compiled_fp32','stable_bmm'],default=['reference'])
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8 or not 1<=a.repeats<=5:raise ValueError('Eight GPUs and 1..5 finite repeats required')
    for k,v in dict(NCCL_CUMEM_HOST_ENABLE='0',NCCL_IB_DISABLE='1',NCCL_SOCKET_IFNAME='lo',TORCH_NCCL_BLOCKING_WAIT='1').items():os.environ.setdefault(k,v)
    torch.cuda.set_device(rank);torch.set_num_threads(4)
    dist.init_process_group('gloo',timeout=timedelta(minutes=40))
    group=dist.new_group(backend='nccl',timeout=timedelta(minutes=20))
    d=DistributedCollectives(rank,8,tensor_group=group,device=f'cuda:{rank}')
    root_call(d,lambda:a.output.mkdir(parents=True,exist_ok=False))
    case=json.loads((a.case/'case.json').read_text())
    if case['schema']!='genmo.stage10.fixed_learning_case.v3' or len(case['ranks'])!=8:raise ValueError('Complete eight-rank v3 case required')
    if file_sha256(a.case/'learning_state.pt')!=case['learning_state_sha256']:raise ValueError('Learning state SHA changed')
    r=a.case/f'rank{rank:02d}';record=next(x for x in case['ranks'] if x['rank']==rank)
    if file_sha256(r/'rank_state.pt')!=record['rank_state_sha256'] or file_sha256(r/'manifest.json')!=record['rollout_manifest_sha256']:
        raise ValueError('Rank input SHA changed')
    inputs=local_call(d,lambda:load_rollout(r));rows=inputs['transitions'];targets=inputs['targets'];manifest=inputs['global_manifest']
    if len(manifest)!=case['real_global_chains'] or case['real_global_chains']<1024:raise ValueError('Small diagnostic data cannot replace large real acceptance')
    source=torch.load(a.case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
    rank_state=torch.load(r/'rank_state.pt',map_location='cpu',weights_only=False)
    config=copy.deepcopy(source['config']);settings=config['stage9'];stage=config['stage10']
    config['runtime']['genmo_device']=str(d.device)
    actor,train,loading=local_call(d,lambda:load_actor(config));actor.load_state_dict(source['actor'])
    policy=DenseDiffusionPolicy(actor,std_schedule=settings['std_schedule'],**dict(precision=stage['precision']['mode'],
        attention=stage['precision']['attention'],linear_execution=stage['precision']['linear_execution'],
        precise_output=stage['precision']['precise_output'],compile_denoiser=stage['precision']['compile']))
    if policy.kernel_config!=source['kernel'] or inputs['kernel']!=source['kernel']:raise ValueError('Baseline behavior numerical contract changed')
    optimizer=torch.optim.AdamW(trainable_actor_parameters(actor),lr=settings['actor_lr'],weight_decay=0.)
    optimizer.load_state_dict(copy.deepcopy(source['actor_optimizer']))
    bc_config=copy.deepcopy(config);bc_config['stage9'].update(bc_batch=settings['bc_batch']//8,seed=42+1000003*rank)
    anchor=Stage1RetentionAnchor(bc_config,actor,train,source_global_step=loading['source_global_step'])
    anchor.load_state_dict(rank_state['bc'])
    controller=BCGradientController(**stage['bc_gradient']);controller.load_state_dict(source['controller'])
    generators=dict(actor=torch.Generator(),critic=torch.Generator())
    restore_local_rng(rank_state['rng'],generators)
    cache=RolloutTensorCache(rows,targets,d.device,max_device_bytes=8*1024**3)
    probability=probability_check_local(policy,rows,global_manifest=manifest,distributed=d,
        denoising_microbatch=512,tensor_cache=cache)
    kwargs=dict(distributed=d,anchor=anchor,generator=generators['actor'],ppo_epochs=2,
        minibatch_chains=stage['scale']['actor_minibatch_chains'],sampled_steps=a.sampled_steps,
        clip=settings['ppo_clip'],gamma_denoising=settings['gamma_denoising'],soft_kl=None,
        kl_reduction=stage['kl']['reduction'],objective_reduction=settings['objective_logprob_reduction'],
        grad_clip_norm=settings['grad_clip_norm'])
    offline_warm=None
    if not optimizer.state:
        offline_warm=actor_update_v3(policy,optimizer,rows,targets,cache,manifest,controller,
            **dict(kwargs,ppo_epochs=1,minibatch_chains=len(manifest),microbatch=128))
    base=immutable_cpu_snapshot(actor.state_dict());adam=immutable_cpu_snapshot(optimizer.state_dict())
    bc_base=immutable_cpu_snapshot(anchor.state_dict());controller_base=copy.deepcopy(controller.state_dict())
    rng_base=capture_local_rng(generators)
    if not adam['state']:raise ValueError('Nonempty Adam state required')
    report=dict(schema='genmo.stage10.fixed_compute_comparison.v3',source_case=str(a.case),
        source_case_sha256=file_sha256(a.case/'case.json'),real_global_chains=len(manifest),probability=probability,
        offline_warmup=offline_warm,all_candidates_same_nonempty_adam=True,old_statistics_not_rewritten=True,
        soft_stop_disabled_only_for_fixed_work=True,sampled_steps=a.sampled_steps,actor=[],kl=[])
    def publish():
        root_call(d,lambda:(a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'))
    def restore():
        actor.load_state_dict(base);optimizer.load_state_dict(copy.deepcopy(adam));anchor.load_state_dict(bc_base)
        controller.load_state_dict(controller_base);restore_local_rng(rng_base,generators)
        actor.zero_grad(set_to_none=True)
    baseline_outputs=terminal_outputs(policy,rows,d.device);reference=None
    for variant in a.variants:
        if variant!='reference':
            from gem.closedloop.dppo.policy_v3 import configure_dense_actor
            options=dict(precision='bf16_backbone' if variant=='bf16_cudnn' else stage['precision']['mode'],
                attention='sdpa_cudnn' if variant=='bf16_cudnn' else stage['precision']['attention'],
                linear_execution='stable_bmm' if variant=='stable_bmm' else stage['precision']['linear_execution'])
            if variant=='compiled_fp32':
                from gem.closedloop.dppo.compiled_forward_v3 import install_shared_forward
                install_shared_forward(policy)
            else:
                if hasattr(actor.denoiser,'_stage10_eager_forward'):raise ValueError('Compiled variant must be tested last or in a separate process')
                policy.numerical_execution.update(configure_dense_actor(actor,**options));policy.precision=options['precision']
        for micro in a.microbatches:
            samples=[];diagnostic=None
            for repeat in range(a.repeats+2):
                restore();captured=[]
                def observer(model,step):
                    if rank==0:captured.append({n:None if p.grad is None else p.grad.detach().cpu().clone()
                        for n,p in model.named_parameters()})
                d.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                result=actor_update_v3(policy,optimizer,rows,targets,cache,manifest,controller,microbatch=micro,
                    gradient_observer=observer if repeat==0 else None,**kwargs)
                torch.cuda.synchronize();elapsed=max(d.all_gather_object(time.perf_counter()-start))
                planned=2*((sum(m['valid'] and m['has_free'] for m in manifest)+kwargs['minibatch_chains']-1)//kwargs['minibatch_chains'])
                if result['optimizer_steps']!=planned or result['internal_sample_visits']!=result['planned_internal_visits']:
                    raise ValueError('Candidate changed actual complete optimizer/data work')
                if repeat>=2:samples.append(elapsed)
                if repeat==0:
                    kl=final_kl_v3(policy,rows,cache,manifest,distributed=d,microbatch=512,
                        clip=settings['ppo_clip'],objective_reduction=settings['objective_logprob_reduction'])
                    outputs=terminal_outputs(policy,rows,d.device)
                    def compare():
                        nonlocal reference
                        weights=immutable_cpu_snapshot(actor.state_dict())
                        state={f'{i}/{k}':v.cpu().clone() for i,x in optimizer.state_dict()['state'].items()
                            for k,v in x.items() if torch.is_tensor(v)}
                        current=dict(gradients=captured,weights=weights,adam=state,outputs=outputs,kl=kl)
                        if reference is None:reference=current
                        return dict(gradients=[compare_named(x,y,atol=3e-5,rtol=2e-4) for x,y in zip(captured,reference['gradients'])],
                            module_directions=[module_directions(x,y) for x,y in zip(captured,reference['gradients'])],
                            parameters=compare_named(weights,reference['weights'],atol=1e-7,rtol=0.),
                            adam=compare_named(state,reference['adam'],atol=1e-7,rtol=2e-4),
                            terminal_outputs=compare_named({'mean':outputs},{'mean':reference['outputs']},atol=1e-7,rtol=0.),
                            terminal_max_change_from_base=float((outputs-baseline_outputs).abs().max()),
                            full20_kl=kl,hard_kl=check_kl_v3(kl,stage['kl']),
                            precision_contract_change=variant!='reference')
                    diagnostic=root_call(d,compare)
                memory=d.all_gather_object(dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.memory_reserved()))
            report['actor'].append(dict(variant=variant,microbatch=micro,samples=samples,p50=float(np.percentile(samples,50)),
                p95=float(np.percentile(samples,95)),diagnostic=diagnostic,actual_optimizer_steps=result['optimizer_steps'],
                actual_internal_visits=result['internal_sample_visits'],actual_BC_samples=result['actual_global_bc_samples'],memory=memory))
            publish()
    for micro in a.kl_microbatches:
        samples=[];restore()
        for repeat in range(a.repeats+2):
            d.barrier();torch.cuda.synchronize();start=time.perf_counter()
            kl=final_kl_v3(policy,rows,cache,manifest,distributed=d,microbatch=micro,
                clip=settings['ppo_clip'],objective_reduction=settings['objective_logprob_reduction'])
            seconds=max(d.all_gather_object(time.perf_counter()-start))
            if repeat>=2:samples.append(seconds)
        report['kl'].append(dict(microbatch=micro,samples=samples,p50=float(np.percentile(samples,50)),
            p95=float(np.percentile(samples,95)),report=kl,numerical_execution=policy.numerical_execution))
        publish()
    report['cache']=d.all_gather_object(cache.report());publish();cache.close()
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
