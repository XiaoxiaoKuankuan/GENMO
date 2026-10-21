"""同一真实行为批次的4096、8192、整批累积和目标感知回溯公平对照。

八卡只读加载fixed_data_validation中的Actor、分组Adam、BC控制器、监督采样器和
各rank RNG；每个候选均从同一快照恢复，不重新生成或修改行为概率。复用生产路径的
四步联合PPO、FP64概率、完整rollout目标审计及真实KL约束。BC按本次明确要求关闭，
但原监督资产和状态保留。仅优化可训练Actor，冻结Stage1先验在恢复Actor之前构造。

输入不包含完整物理执行时不会补造记录。输出目标、梯度方向、参数和固定噪声参考
变化，仅证明反事实优化行为，不等价于PhysX执行收益。每个候选独立目录保留Actor
证据；本工具不启动物理、不启动正式训练，不覆写原case或训练checkpoint。
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

from gem.closedloop.dppo.actor_backtracking import check_candidate_kl
from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer,validate_actor_optimizer
from gem.closedloop.dppo.actor_step_diagnostics import output_probe
from gem.closedloop.dppo.attribution_runtime import probe_difference,probe_summary
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.chain_ppo import from_records
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.optimization_support import probability_check_local
from gem.closedloop.dppo.parallel_support import capture_local_rng,restore_local_rng,local_call,root_call
from gem.closedloop.dppo.retention_anchor import Stage1RetentionAnchor
from gem.closedloop.dppo.rollback_audit import exact_state_digest
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_checkpoint import file_sha256,immutable_cpu_snapshot
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.updater import actor_update,final_kl
from tools.analyze_stage2_probability import export_actor


def restore_experiment(actor,optimizer,controller,anchor,snapshot,generator):
    """恢复非空Adam时也严格核对参数名映射，不做单组到多组的隐式迁移。"""
    actor.load_state_dict(snapshot['actor'])
    current=optimizer.state_dict()['param_groups'];saved=snapshot['optimizer']['param_groups']
    fields=('param_names','name','lr_multiplier','params')
    if len(current)!=len(saved) or any(any(a.get(k)!=b.get(k) for k in fields)
            for a,b in zip(current,saved)):
        raise ValueError('行为Adam参数组映射与候选不同，禁止迁移状态')
    optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    validate_actor_optimizer(actor,optimizer)
    controller.load_state_dict(copy.deepcopy(snapshot['controller']))
    anchor.load_state_dict(copy.deepcopy(snapshot['bc']))
    restore_local_rng(snapshot['rng'],{'actor':generator})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--branches',nargs='+',choices=('4096','8192','full','objective4096'),
                        default=['4096','8192','full','objective4096'])
    args=parser.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8 or rank!=local:raise ValueError('仅支持服务器单节点八卡')
    dist.init_process_group('gloo',timeout=timedelta(minutes=35))
    cache=None;group=None
    try:
        control=DistributedCollectives(rank,world)
        def reserve():
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy:raise RuntimeError('八卡仍被占用: '+busy)
            args.output_dir.mkdir(parents=True,exist_ok=False)
        root_call(control,reserve)
        torch.cuda.set_device(local);torch.set_num_threads(4)
        group=dist.new_group(backend='nccl',timeout=timedelta(minutes=30))
        distributed=DistributedCollectives(rank,world,tensor_group=group,device=f'cuda:{local}')
        config=configuration(args.config,genmo_repo=ROOT)
        config['runtime']['genmo_device']=str(distributed.device)
        config['stage9']['seed']+=1000003*rank
        actor,train,loading=local_call(distributed,lambda:load_actor(config))
        policy=build_policy(actor,config['stage9'],config['stage10']['precision'])
        header=json.loads((args.case/'case.json').read_text())
        if file_sha256(args.case/'learning_state.pt')!=header['learning_state_sha256']:
            raise ValueError('不可变学习状态SHA不一致')
        learning=torch.load(args.case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
        for name,expected in [('manifest.json','rollout_manifest_sha256'),('rank_state.pt','rank_state_sha256')]:
            if file_sha256(args.case/f'rank{rank:02d}'/name)!=header['ranks'][rank][expected]:
                raise ValueError('不可变rank证据SHA不一致')
        payload=local_call(distributed,lambda:load_rollout(args.case/f'rank{rank:02d}'))
        if payload['kernel']!=policy.kernel_config:raise ValueError('实际行为采样核不一致')
        actor.load_state_dict(learning['actor'])
        optimizer=make_actor_optimizer(actor,config['stage10']['training'])
        controller=BCGradientController(**config['stage10']['bc_gradient'])
        anchor_config=copy.deepcopy(config);anchor_config['stage9']['bc_batch']//=world
        anchor=Stage1RetentionAnchor(anchor_config,actor,train,source_global_step=loading['source_global_step'])
        rank_state=torch.load(args.case/f'rank{rank:02d}'/'rank_state.pt',map_location='cpu',weights_only=False)
        snapshot=dict(actor=learning['actor'],optimizer=learning['actor_optimizer'],controller=learning['controller'],
            bc=rank_state['bc'],rng=rank_state['rng'])
        generator=torch.Generator()
        restore_experiment(actor,optimizer,controller,anchor,snapshot,generator)
        initial_digest=exact_state_digest(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),
            controller=controller.state_dict(),bc=anchor.state_dict(),rng=capture_local_rng({'actor':generator})))
        rows,targets,manifest=payload['transitions'],payload['targets'],payload['global_manifest']
        cache=RolloutTensorCache(rows,targets,distributed.device,max_device_bytes=4*1024**3)
        zero=probability_check_local(policy,rows,global_manifest=manifest,distributed=distributed,
            denoising_microbatch=128,tensor_cache=cache,include_frozen=True)
        root_call(distributed,lambda:atomic_json(args.output_dir/'zero_update.json',zero))
        eligible=[r for i,r in enumerate(rows) if bool(targets['valid'][i]) and r.free_mask.any()]
        selected=eligible[:4]
        count=sum(bool(m['valid']) and bool(m['has_free']) for m in manifest)
        summaries=[]
        metadata=[item for shard in distributed.all_gather_object([dict(identity=r.identity,
            source=r.metadata.get('training_task',{}).get('dataset','unknown'),
            music_start_frame=r.metadata.get('music_start_frame')) for i,r in enumerate(rows)
            if bool(targets['valid'][i]) and r.free_mask.any()]) for item in shard]
        for name in args.branches:
            restore_experiment(actor,optimizer,controller,anchor,snapshot,generator)
            restored_digest=exact_state_digest(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),
                controller=controller.state_dict(),bc=anchor.state_dict(),rng=capture_local_rng({'actor':generator})))
            if restored_digest!=initial_digest:raise AssertionError('候选起点逐字节不一致')
            output=args.output_dir/name
            root_call(distributed,lambda:output.mkdir(exist_ok=False))
            trust=copy.deepcopy(config['stage9']['trust_region'])
            trust['objective_aware']=name=='objective4096'
            trust['objective_search_attempts']=3
            batch=count if name=='full' else (4096 if name=='objective4096' else int(name))
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats(distributed.device)
            last_statistics={}
            def retain_statistics(stage,records,advantages,sources):
                last_statistics.update(records=records,advantages=advantages,sources=sources)
            def observer(step):
                root_call(distributed,lambda:atomic_json(output/f"step_{step['optimizer_step']:03d}.json",step))
                if rank==0:print(json.dumps(dict(branch=name,step=step['optimizer_step'],
                    objective=step['full_rollout_after']['objective_mean'])),flush=True)
            report=actor_update(policy,optimizer,rows,targets,cache,manifest,controller,distributed=distributed,
                anchor=anchor,generator=generator,ppo_epochs=1,minibatch_chains=batch,microbatch=128,
                clip=.1,soft_kl=.1,bc_samples_per_rollout=2048,bc_min_global_batch=256,
                bc_enabled=False,audit_full_rollout_objective=True,trust_region=trust,
                kl_limits=config['stage10']['kl'],detailed_diagnostics=True,step_observer=observer,
                full_rollout_observer=retain_statistics)
            kl=final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=128,clip=.1)
            counts=distributed.all_gather_object(len(eligible));offset=sum(counts[:rank])
            stats=from_records(last_statistics['records'][offset:offset+len(eligible)])
            indices=sorted(set(list(range(min(4,len(eligible))))+
                torch.topk(stats['joint_kl'],min(4,len(eligible))).indices.tolist()+
                torch.topk(stats['joint_log_ratio'],min(4,len(eligible))).indices.tolist()))
            selected=[eligible[i] for i in indices]
            candidate_state=immutable_cpu_snapshot(actor.state_dict())
            after_probe=output_probe(policy,selected,cache,max_chains=len(selected))
            actor.load_state_dict(snapshot['actor'])
            before_probe=output_probe(policy,selected,cache,max_chains=len(selected))
            actor.load_state_dict(candidate_state)
            probes=[v for shard in distributed.all_gather_object(probe_difference(before_probe,after_probe)) for v in shard]
            local_call(distributed,lambda:torch.save(dict(before=immutable_cpu_snapshot(before_probe),
                after=immutable_cpu_snapshot(after_probe),selection='ordinary_and_actual_high_KL_high_ratio',
                generation_world_anchors=[r.metadata.get('generation_world_anchor') for r in selected]),
                output/f'probe_rank{rank:02d}.pt'))
            result=dict(branch=name,minibatch_chains=batch,effective_chains=count,actor=report,final_kl=kl,
                acceptance=check_candidate_kl(kl,config['stage10']['kl']),output=probe_summary(probes),
                wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)),
                peak_allocated_bytes=max(distributed.all_gather_object(torch.cuda.max_memory_allocated())),
                same_initial_state_digest=initial_digest,physical_quality='unverified_counterfactual_only')
            result['export']=root_call(distributed,lambda:export_actor(policy,config,learning,report,
                result['acceptance'],args.case,output))
            root_call(distributed,lambda:atomic_json(output/'result.json',result))
            root_call(distributed,lambda:torch.save(dict(**last_statistics,metadata=metadata,
                scope='reused_production_full_rollout_FP64_statistics_no_extra_forward'),output/'final_chain_statistics.pt'))
            root_call(distributed,lambda:torch.save(dict(schema='stage2.batch_comparison.actor.v1',
                actor=immutable_cpu_snapshot(actor.state_dict()),kernel=policy.kernel_config,
                source_case=str(args.case),source_case_sha256=file_sha256(args.case/'case.json'),
                source_identity=learning['identity'],evaluation_only=True,training_resume=False),output/'actor.pt'))
            summaries.append(result)
            root_call(distributed,lambda:atomic_json(args.output_dir/'comparison.json',dict(
                complete=len(summaries)==len(args.branches),results=summaries,bc_enabled=False,
                objectives_are_not_physical_benefit=True)))
            del result,report,probes,candidate_state,before_probe,after_probe
        restore_experiment(actor,optimizer,controller,anchor,snapshot,generator)
    finally:
        if cache is not None:cache.close()
        if group is not None:dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__=='__main__':main()
