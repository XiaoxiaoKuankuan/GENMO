"""把真实已评估checkpoint与不可变旧行为链作八卡概率/参考变化对照。

不执行PPO、BC或任何优化器，不创建物理世界。先加载保存行为Actor并核验四步旧
密度及冻结身份，再用同一条件、旧链、实际std/mask和innovation读取指定checkpoint。
复用独立坐标归因与固定噪声输出探针，保存完整四步KL、log-ratio、尾部真实样本和
源帧变化。输出的反事实参考可与同一case实际消费来源关联；checkpoint自己的真实
PhysX评估另由其报告证明，禁止把这里的反事实变化当作机器人实际改善。

checkpoint必须通过当前资产、奖励、时钟、物理配置和数据核验，源码仅记录；已有训练
优化器及进度不恢复、不修改。完整累计Actor变化可能超过单轮信任域，这只是跨版本
诊断，不据此撤销已通过各自新rollout验收的更新。所有结果写新独立目录。
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from gem.closedloop.dppo.actor_step_diagnostics import output_probe
from gem.closedloop.dppo.attribution_runtime import row_metadata,probe_difference,probe_summary
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.chain_ppo import from_records
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.optimization_support import probability_check_local
from gem.closedloop.dppo.parallel_support import root_call,local_call
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training import _sources,_physical_configuration_identity
from gem.closedloop.dppo.training_checkpoint import file_sha256,load_evaluation_actor
from gem.closedloop.dppo.training_config import configuration
from tools.analyze_stage2_probability import whole_chain_statistics,coordinate_analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8 or rank!=local:raise ValueError('Single-node eight-GPU allocation required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=45));group=None;cache=None
    try:
        def reserve():
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy:raise RuntimeError('Actor attribution requires free allocated GPUs')
            args.output_dir.mkdir(parents=True,exist_ok=False)
        root_call(DistributedCollectives(rank,world),reserve)
        torch.cuda.set_device(local);torch.set_num_threads(4)
        group=dist.new_group(backend='nccl',timeout=timedelta(minutes=30))
        distributed=DistributedCollectives(rank,world,tensor_group=group,device=f'cuda:{local}')
        config=configuration(args.config,genmo_repo=ROOT);config['runtime']['genmo_device']=str(distributed.device)
        actor,_,_=local_call(distributed,lambda:load_actor(config));policy=build_policy(actor,config['stage9'],config['stage10']['precision'])
        header=json.loads((args.case/'case.json').read_text())
        if file_sha256(args.case/'learning_state.pt')!=header['learning_state_sha256']:raise ValueError('Behavior learning SHA mismatch')
        learning=torch.load(args.case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
        actor.load_state_dict(learning['actor'])
        payload=local_call(distributed,lambda:load_rollout(args.case/f'rank{rank:02d}'))
        if payload['kernel']!=policy.kernel_config:raise ValueError('Saved actual sampling kernel differs')
        rows,targets,manifest=payload['transitions'],payload['targets'],payload['global_manifest']
        cache=RolloutTensorCache(rows,targets,distributed.device,max_device_bytes=4*1024**3)
        eligible=[row for i,row in enumerate(rows) if bool(targets['valid'][i]) and row.free_mask.any()]
        zero=probability_check_local(policy,rows,global_manifest=manifest,distributed=distributed,
            denoising_microbatch=128,tensor_cache=cache,include_frozen=False)
        binding=dict(reward=config['stage9']['reward'],timing=config['timing'],source_files=_sources(config),
            physical_configuration=_physical_configuration_identity(config),data_catalog=learning['identity']['data_catalog'],
            generation_horizon=config['stage10']['generation_horizon'])
        checkpoint=local_call(distributed,lambda:load_evaluation_actor(args.checkpoint,actor=actor,kernel=policy.kernel_config,
            assets=config['stage10']['assets'],execution_binding=binding))
        metadata=[m for shard in distributed.all_gather_object([row_metadata(r,cache.get('advantages',[r])[0],rank) for r in eligible]) for m in shard]
        importance,records=whole_chain_statistics(policy,eligible,cache,distributed,metadata)
        sizes=distributed.all_gather_object(len(eligible));local_records=records[sum(sizes[:rank]):sum(sizes[:rank+1])]
        stats=from_records(local_records)
        indices=sorted(set(list(range(min(8,len(eligible))))+torch.topk(stats['joint_kl'].sum(1),min(4,len(eligible))).indices.tolist()+
            torch.topk(stats['joint_log_ratio'].sum(1),min(4,len(eligible))).indices.tolist()))
        selected=[eligible[i] for i in indices]
        after=output_probe(policy,selected,cache,max_chains=len(selected))
        coordinates=coordinate_analysis(policy,eligible,cache,distributed,args.output_dir)
        actor.load_state_dict(learning['actor']);before=output_probe(policy,selected,cache,max_chains=len(selected))
        samples=[s for shard in distributed.all_gather_object(probe_difference(before,after)) for s in shard]
        local_call(distributed,lambda:torch.save(dict(before={k:v.cpu() if torch.is_tensor(v) else v for k,v in before.items()},
            after={k:v.cpu() if torch.is_tensor(v) else v for k,v in after.items()}),args.output_dir/f'probe_vectors_rank{rank:02d}.pt'))
        root_call(distributed,lambda:atomic_json(args.output_dir/'output_attribution.json',probe_summary(samples)))
        root_call(distributed,lambda:atomic_json(args.output_dir/'summary.json',dict(status='passed',
            case=str(args.case),case_sha256=file_sha256(args.case/'case.json'),checkpoint=checkpoint,
            zero_update_probability=zero,coordinates=coordinates,importance=importance,output=probe_summary(samples),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            scope='checkpoint_vs_old_behavior_counterfactual_reference_not_optimizer_or_physical_reexecution')))
    finally:
        if cache is not None:cache.close()
        if group is not None:dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__=='__main__':main()
