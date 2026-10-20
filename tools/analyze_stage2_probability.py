"""服务器1八卡真实固定rollout的概率归因、模块隔离与候选优化实验。

输入是带SHA的6144链等真实fixed_data_validation及显式完整配置。复用生产策略、
真实高斯核、流式全链PPO、Stage1RetentionAnchor和显式分组AdamW；只切换允许训练的
参数范围及固定学习率倍率。行为张量只读，新目录独立保存每次尝试、整轮KL/ESS、坐标与输出归因、
最高权重真实样本及可选评估Actor。ESS仅诊断，不恢复训练硬门限。

坐标中间文件只在本次独立目录的私有工作子目录中短暂存在，精确汇总后删除；保留
完整链级证据和精确分位数报告。模块隔离使用同一次真实Adam增量逐组重放，明确其
非线性交互不能相加。任何未在PhysX执行的候选只报告反事实参考变化。该入口不创建
Isaac世界，不更改奖励、Stage1架构、GMT、噪声调度和物理时钟，不启动正式训练。
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

from gem.closedloop.dppo.actor_scope import SCOPES
from gem.closedloop.dppo.actor_optimizer import (
    make_actor_optimizer,actor_base_lr,actor_learning_rates,actor_parameter_group)
from gem.closedloop.dppo.actor_backtracking import check_candidate_kl
from gem.closedloop.dppo.actor_step_diagnostics import output_probe
from gem.closedloop.dppo.attribution_runtime import collect_coordinates,row_metadata,probe_difference,probe_summary
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,from_records
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.gradient_merge import BCGradientController,_group
from gem.closedloop.dppo.optimization_support import probability_check_local,_local_phase
from gem.closedloop.dppo.parallel_support import capture_local_rng,restore_local_rng,local_call,root_call
from gem.closedloop.dppo.probability_attribution import coordinate_report,importance_report,quantiles
from gem.closedloop.dppo.retention_anchor import Stage1RetentionAnchor
from gem.closedloop.dppo.rollback_audit import exact_state_digest
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_checkpoint import file_sha256,immutable_cpu_snapshot
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.updater import actor_update,final_kl


@torch.no_grad()
def whole_chain_statistics(policy,rows,cache,distributed,metadata):
    with _local_phase(distributed,'attribution_all_real_chains'),policy_phase(policy):
        local=evaluate_chain_records(policy,rows,cache,microbatch=128).cpu()
    shards=distributed.all_gather_object(local)
    records=torch.cat(shards);stats=from_records(records)
    report=importance_report(stats['joint_log_ratio'],stats['joint_kl'],
        [m['advantage'] for m in metadata],metadata)
    from gem.closedloop.dppo.gaussian_probability import ppo_surrogate
    report['full_rollout_clipped_objective_mean']=float(ppo_surrogate(
        stats,torch.tensor([m['advantage'] for m in metadata],dtype=torch.float64),clip=.1)['objective'].mean())
    return report,records


def coordinate_analysis(policy,rows,cache,distributed,output):
    detail,metadata=local_call(distributed,lambda:collect_coordinates(policy,rows,cache,rank=distributed.rank))
    scratch=output/'coordinate_work'
    root_call(distributed,lambda:scratch.mkdir(exist_ok=False))
    path=scratch/f'rank{distributed.rank:02d}.pt'
    local_call(distributed,lambda:torch.save(dict(detail=detail,metadata=metadata),path))
    del detail
    def aggregate():
        shards=[torch.load(scratch/f'rank{i:02d}.pt',map_location='cpu',weights_only=False,mmap=True)
            for i in range(distributed.world_size)]
        combined={key:torch.cat([s['detail'][key] for s in shards]) for key in shards[0]['detail']}
        meta=[m for shard in shards for m in shard['metadata']]
        report=coordinate_report(combined,meta,policy.actor.endecoder.kinematics.joint_order)
        z=combined['residual'][combined['mask']]
        report['sampling_kernel_audit']=dict(residual_tail_fractions={str(a):float((z.abs()>a).double().mean()) for a in (3,4,5,6)},
            normal_two_sided_tail_probabilities={str(a):math.erfc(a/math.sqrt(2)) for a in (3,4,5,6)},
            stored_noise_reconstruction_abs=quantiles(combined['sampling_noise_error'][combined['mask']]),
            independent_old_log_probability_abs_error=quantiles([v for m in meta for v in m['old_log_probability_abs_error_by_step']]),
            scope='actual_FP32_sample_reconstruction_and_independent_FP64_Normal_density_not_normality_hypothesis_gate')
        log=combined['log_ratio'].sum((1,2,3));kl=combined['kl'].sum((1,2,3))
        report['importance']=importance_report(log,kl,[m['advantage'] for m in meta],meta)
        torch.save(dict(metadata=meta,joint_log_ratio=log,joint_kl=kl),output/'chain_attribution.pt')
        atomic_json(output/'coordinate_attribution.json',report)
        del combined,shards
        for i in range(distributed.world_size):(scratch/f'rank{i:02d}.pt').unlink()
        scratch.rmdir()
        return dict(total=report['total'],concentration=report['concentration'],sampling_kernel_audit=report['sampling_kernel_audit'])
    return root_call(distributed,aggregate)


def module_ablation(policy,rows,cache,initial,distributed,output,*,selected=None,layerwise=False):
    """同一实际Adam增量逐层隔离，普通和尾部样本固定；结果不是可相加的KL贡献。"""
    selected=rows[:16] if selected is None else selected
    after=immutable_cpu_snapshot(policy.actor.state_dict())
    policy.actor.load_state_dict(initial)
    before_probe=output_probe(policy,selected,cache,max_chains=len(selected))
    groups=(['transformer_backbone','qpos_output','other']+
        sorted({actor_parameter_group(n) for n,p in policy.actor.named_parameters()
            if p.requires_grad and n.startswith('denoiser.blocks.')})) if layerwise else (
        sorted({_group(n) for n,p in policy.actor.named_parameters() if p.requires_grad}))
    from tools.analyze_stage2_reference_consumption import token,joint_reference_changes
    from gem.closedloop.dppo.execution_provenance import consumption_report
    from collections import defaultdict
    episodes=defaultdict(list);rows_by_id={}
    for row in rows:
        identity=token(row.identity);rows_by_id[identity]=row
        if 'compact_execution_provenance' in row.metadata:
            episodes[identity[:3]].append(row.metadata['compact_execution_provenance'])
    consumed={key:consumption_report(records) for key,records in episodes.items()}
    before_cpu=immutable_cpu_snapshot(before_probe)
    def matches(name,group):
        if not layerwise:return _group(name)==group
        if group=='transformer_backbone':return name.startswith('denoiser.blocks.')
        return actor_parameter_group(name)==group
    result=[]
    try:
        for group in ['all',*groups]:
            state={name:(after[name] if group=='all' or matches(name,group) else tensor) for name,tensor in initial.items()}
            policy.actor.load_state_dict(state)
            probe=output_probe(policy,selected,cache,max_chains=len(selected))
            samples=[s for shard in distributed.all_gather_object(probe_difference(before_probe,probe)) for s in shard]
            metadata=[m for shard in distributed.all_gather_object([row_metadata(r,cache.get('advantages',[r])[0],distributed.rank) for r in selected]) for m in shard]
            ratio,_=whole_chain_statistics(policy,selected,cache,distributed,metadata)
            mask=before_probe['mask'][:,None].expand_as(before_probe['mean'])
            standardized=(probe['mean'].double()-before_probe['mean'].double())/before_probe['actual_std'].double()
            totals=distributed.all_gather_object(dict(
                standardized_mean_square_sum=float(standardized.square()[mask].sum()),free_coordinates=int(mask.sum())))
            after_cpu=immutable_cpu_snapshot(probe)
            consumption=joint_reference_changes(episodes,consumed,
                [dict(before=before_cpu,after=after_cpu)],rows_by_id) if episodes else {'status':'unverified'}
            consumption=distributed.all_gather_object(consumption)
            result.append(dict(module=group,probability=ratio,output=probe_summary(samples),
                standardized_mean_delta_rms=math.sqrt(sum(v['standardized_mean_square_sum'] for v in totals)/
                    sum(v['free_coordinates'] for v in totals)),
                actual_consumed_reference_by_rank=consumption))
            if group in ('all','transformer_backbone','qpos_output'):
                local_call(distributed,lambda:torch.save(dict(before=before_cpu,after=after_cpu),
                    output/f'{group}_probe_vectors_rank{distributed.rank:02d}.pt'))
            root_call(distributed,lambda:atomic_json(output/'module_ablation.json',dict(groups=result,
                complete=False,probe_selection='ordinary_plus_high_KL_and_log_ratio_tails',
                scope='isolated_actual_same_Adam_delta_on_same_real_rows_nonlinear_effects_not_additive')))
            if distributed.rank==0:print(json.dumps(dict(ablation=group,joint_kl=ratio['joint_kl_mean'])),flush=True)
        root_call(distributed,lambda:atomic_json(output/'module_ablation.json',dict(groups=result,
            complete=True,probe_selection='ordinary_plus_high_KL_and_log_ratio_tails',
            sensitive_layers=[int(r['module'].rsplit('_',1)[1]) for r in
                sorted((r for r in result if r['module'].startswith('transformer_') and
                    r['module']!='transformer_backbone'),key=lambda r:r['probability']['joint_kl_mean'],reverse=True)[:2]],
            scope='isolated_actual_same_Adam_delta_on_same_real_rows_nonlinear_effects_not_additive')))
    finally:policy.actor.load_state_dict(after)


def export_actor(policy,config,learning,report,acceptance,case,output,*,allow_measured_profile_transfer=False):
    from gem.closedloop.dppo.training import _sources,_physical_configuration_identity
    from gem.closedloop.dppo.training_checkpoint import probability_candidate_binding
    source=copy.deepcopy(learning['identity']);identity=copy.deepcopy(source)
    current=dict(reward=config['stage9']['reward'],timing=config['timing'],
        physical_configuration=_physical_configuration_identity(config),generation_horizon=config['stage10']['generation_horizon'])
    identity.update(current)
    identity['source_files']=_sources(config)
    complete=report['optimizer_steps']==report['planned_optimizer_steps']
    if not complete or not acceptance['passed']:return dict(exported=False,reason='incomplete_update_or_joint_KL_rejection')
    transfer=probability_candidate_binding(source,identity,allow_measured_profile_transfer=allow_measured_profile_transfer)
    path=output/'actor_evaluation.pt'
    torch.save(dict(schema='genmo.stage2.probability_candidate_evaluation.v1',evaluation_only=True,
        actor=immutable_cpu_snapshot(policy.actor.state_dict()),kernel=policy.kernel_config,source_identity=source,
        evaluation_identity=identity,evaluation_transfer=transfer,
        hard_kl=acceptance,complete_update_and_soft_kl_passed=True,update=dict(
            optimizer_steps=report['optimizer_steps'],planned_optimizer_steps=report['planned_optimizer_steps'],
            actor_scope=policy.actor_scope),source_case_sha256=file_sha256(case/'case.json')),path)
    return dict(exported=True,path=str(path),sha256=file_sha256(path),training_resume=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/closedloop/stage2_full_chain.yaml')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--scope',choices=SCOPES,default='all')
    parser.add_argument('--learning-rate',type=float,default=1e-6)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--max-updates',type=int)
    parser.add_argument('--no-backtracking',action='store_true')
    parser.add_argument('--coordinates',action='store_true')
    parser.add_argument('--module-ablation',action='store_true')
    parser.add_argument('--layer-ablation',action='store_true',
        help='同一次真实Adam增量隔离16层及主干/输出头，覆盖普通和尾部探针并关联实际中心参考')
    parser.add_argument('--export-actor',action='store_true')
    parser.add_argument('--allow-measured-profile-transfer',action='store_true',
        help='显式将旧采集延迟样本池迁移到当前物理评估池；保留来源，必须新鲜配对PhysX评估，不允许训练恢复')
    args=parser.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8 or local!=rank:raise ValueError('Explicit single-node eight-GPU allocation required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=45));group=None;cache=None
    try:
        def reserve_node():
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            if busy:raise RuntimeError('Probability experiment requires free allocated GPUs: '+busy)
            args.output_dir.mkdir(parents=True,exist_ok=False)
        root_call(DistributedCollectives(rank,world),reserve_node)
        torch.cuda.set_device(local);torch.set_num_threads(4)
        group=dist.new_group(backend='nccl',timeout=timedelta(minutes=30))
        distributed=DistributedCollectives(rank,world,tensor_group=group,device=f'cuda:{local}')
        config=configuration(args.config,genmo_repo=ROOT);config['runtime']['genmo_device']=str(distributed.device)
        config['stage9']['actor_trainable_scope']=args.scope
        config['stage10']['training']['actor_trainable_scope']=args.scope
        config['stage9']['seed']+=1000003*rank
        actor,train,loading=local_call(distributed,lambda:load_actor(config))
        policy=build_policy(actor,config['stage9'],config['stage10']['precision'])
        header=json.loads((args.case/'case.json').read_text())
        if file_sha256(args.case/'learning_state.pt')!=header['learning_state_sha256']:raise ValueError('Behavior learning state SHA mismatch')
        learning=torch.load(args.case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
        if any(not torch.equal(value.detach().cpu(),learning['actor'][name]) for name,value in actor.state_dict().items()):
            raise ValueError('Fixed-data candidate must start from the unchanged original Stage1 Actor')
        actor.load_state_dict(learning['actor'])
        if file_sha256(args.case/f'rank{rank:02d}'/'manifest.json')!=header['ranks'][rank]['rollout_manifest_sha256']:
            raise ValueError('Behavior rollout manifest SHA differs from the case')
        payload=local_call(distributed,lambda:load_rollout(args.case/f'rank{rank:02d}'))
        if payload['kernel']!=policy.kernel_config:raise ValueError('Actual sampling kernel differs from immutable real data')
        if learning['actor_optimizer']['state']:raise ValueError('This experiment requires the audited empty behavior Adam; no implicit scope-state migration')
        rows,targets,manifest=payload['transitions'],payload['targets'],payload['global_manifest']
        cache=RolloutTensorCache(rows,targets,distributed.device,max_device_bytes=4*1024**3)
        eligible=[row for i,row in enumerate(rows) if bool(targets['valid'][i]) and row.free_mask.any()]
        probability=probability_check_local(policy,rows,global_manifest=manifest,distributed=distributed,
            denoising_microbatch=128,tensor_cache=cache,include_frozen=args.scope=='all')
        root_call(distributed,lambda:atomic_json(args.output_dir/'zero_update_probability.json',dict(
            report=probability,frozen_identity=policy.assert_frozen_identity(),
            frozen_gaussian_replayed=args.scope=='all',case_sha256=file_sha256(args.case/'case.json'))))
        metadata=[m for shard in distributed.all_gather_object([row_metadata(r,cache.get('advantages',[r])[0],rank) for r in eligible]) for m in shard]
        initial=immutable_cpu_snapshot(actor.state_dict());initial_digest=exact_state_digest(initial)
        optimizer_settings=dict(config['stage10']['training'],actor_lr=args.learning_rate)
        optimizer=make_actor_optimizer(actor,optimizer_settings)
        controller=BCGradientController(**config['stage10']['bc_gradient'])
        # 旧行为采集使用旧BC梯度控制契约；本次候选统一从当前控制器冷启动，禁止伪恢复。
        controller_origin=dict(kind='fresh_current_full_chain_controller',
            legacy_behavior_controller=learning['controller'],current_controller=controller.state_dict())
        anchor_config=copy.deepcopy(config);anchor_config['stage9']['bc_batch']//=world
        anchor=Stage1RetentionAnchor(anchor_config,actor,train,source_global_step=loading['source_global_step'])
        generator=torch.Generator().manual_seed(3045)
        rank_state_path=args.case/f'rank{rank:02d}'/'rank_state.pt'
        if file_sha256(rank_state_path)!=header['ranks'][rank]['rank_state_sha256']:
            raise ValueError('Behavior BC/RNG state SHA mismatch')
        rank_state=torch.load(rank_state_path,map_location='cpu',weights_only=False)
        anchor.load_state_dict(rank_state['bc']);restore_local_rng(rank_state['rng'],{'actor':generator})
        rng=capture_local_rng({'actor':generator});anchor_initial=copy.deepcopy(anchor.state_dict())
        optimizer_initial=copy.deepcopy(optimizer.state_dict());controller_initial=copy.deepcopy(controller.state_dict())
        before=output_probe(policy,eligible[:16],cache,max_chains=16)
        history=[];attempt_history=[];last_evaluation={};evaluation_reuses=0
        def whole():
            nonlocal evaluation_reuses
            signature=policy._parameter_signature()
            if last_evaluation.get('signature')==signature:
                evaluation_reuses+=1
                return last_evaluation['report'],last_evaluation['records']
            measured,records=whole_chain_statistics(policy,eligible,cache,distributed,metadata)
            last_evaluation.update(signature=signature,report=measured,records=records)
            return measured,records
        def persist_statistics(name,records):
            root_call(distributed,lambda:torch.save(dict(records=records,metadata=metadata,
                source_case_sha256=file_sha256(args.case/'case.json'),
                columns=['joint_kl','mean_coordinate_kl','joint_log_ratio','free_count','mean_change_rms',
                    'root_feature_change','rotation6d_change','joint_change']),args.output_dir/(name+'.chains.pt')))
        def observe_attempt():
            measured,records=whole()
            change=probe_difference(before,output_probe(policy,eligible[:16],cache,max_chains=16))
            samples=[s for shard in distributed.all_gather_object(change) for s in shard]
            entry=dict(attempt=len(attempt_history)+1,learning_rate=actor_base_lr(optimizer),
                learning_rates=actor_learning_rates(optimizer),
                full_rollout=measured,output=probe_summary(samples))
            attempt_history.append(entry)
            root_call(distributed,lambda:atomic_json(args.output_dir/f'attempt_{len(attempt_history):03d}.json',entry))
            persist_statistics(f'attempt_{len(attempt_history):03d}',records)
            return dict(whole_chain_ESS=measured['ess_fraction'],whole_chain_KL=measured['joint_kl_mean'])
        def observe_step(step):
            measured,records=whole()
            entry=dict(minibatch_accepted_adam_step=step['optimizer_step'],final_acceptance_pending=True,
                learning_rate=step['learning_rate'],full_rollout=measured,
                learning_rates=step['learning_rates'],
                parameter_update=step['parameter_update'],gradient_contributions=step['gradient_contributions'],
                trust_region=step['trust_region'])
            history.append(entry)
            root_call(distributed,lambda:atomic_json(args.output_dir/f'accepted_step_{len(history):03d}.json',entry))
            persist_statistics(f'accepted_step_{len(history):03d}',records)
            if rank==0:print(json.dumps(dict(scope=args.scope,step=step['optimizer_step'],lr=step['learning_rate'],
                KL=measured['joint_kl_mean'],ESS=measured['ess_fraction'],active_clip=measured['active_clip_fraction'])),flush=True)
        trust=None if args.no_backtracking else dict(config['stage9']['trust_region'],maximum_lr=args.learning_rate)
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats(distributed.device)
        report=actor_update(policy,optimizer,rows,targets,cache,manifest,controller,distributed=distributed,
            anchor=anchor,generator=generator,ppo_epochs=args.epochs,minibatch_chains=2048,microbatch=128,
            clip=.1,soft_kl=.1,bc_samples_per_rollout=2048*args.epochs,bc_min_global_batch=256,
            trust_region=trust,kl_limits=config['stage10']['kl'],max_optimizer_steps=args.max_updates,
            step_observer=observe_step,candidate_observer=observe_attempt if trust is not None else None)
        kl=final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=128,clip=.1)
        acceptance=check_candidate_kl(kl,config['stage10']['kl'])
        if not report['optimizer_steps']:acceptance=dict(passed=False,failures={'optimizer':'no accepted step'})
        report['accepted_optimizer_steps']=report['optimizer_steps'] if acceptance['passed'] else 0
        final_importance,records=whole()
        if not math.isclose(final_importance['joint_kl_mean'],kl['chain_joint_kl']['mean'],rel_tol=1e-9,abs_tol=1e-12):
            raise AssertionError('Reused probability diagnostics differ from independent final whole-rollout KL')
        persist_statistics('final',records)
        # 普通样本加每个rank自身高KL/高ratio尾部，前后始终使用完全相同条件和噪声。
        local_records=records[sum(distributed.all_gather_object(len(eligible))[:rank]):][:len(eligible)]
        stats=from_records(local_records)
        ordinary=[]
        for source in sorted({r.metadata.get('training_task',{}).get('dataset','unknown') for r in eligible}):
            indices=[i for i,r in enumerate(eligible) if r.metadata.get('training_task',{}).get('dataset','unknown')==source]
            ordinary.extend(indices[j*len(indices)//3] for j in (1,2))
        selected_indices=sorted(set(ordinary+
            torch.topk(stats['joint_kl'],min(4,len(eligible))).indices.tolist()+
            torch.topk(stats['joint_log_ratio'],min(4,len(eligible))).indices.tolist()))
        selected=[eligible[i] for i in selected_indices]
        after_state=immutable_cpu_snapshot(actor.state_dict());after_probe=output_probe(policy,selected,cache,max_chains=len(selected))
        if args.export_actor:
            root_call(distributed,lambda:torch.save(dict(schema='genmo.stage2.probability_diagnostic_actor.v1',
                actor=after_state,source_case_sha256=file_sha256(args.case/'case.json'),source_identity=learning['identity'],
                kernel=policy.kernel_config,actor_scope=policy.actor_scope,update=report,hard_kl=acceptance,
                evaluation_only=False,training_resume=False),args.output_dir/'actor_diagnostic.pt'))
        local_call(distributed,lambda:torch.save(dict(before={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in before.items()},
            after={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in output_probe(policy,eligible[:16],cache,max_chains=16).items()}),
            args.output_dir/f'ordinary_probe_vectors_rank{rank:02d}.pt'))
        actor.load_state_dict(initial);before_probe=output_probe(policy,selected,cache,max_chains=len(selected));actor.load_state_dict(after_state)
        probes=[s for shard in distributed.all_gather_object(probe_difference(before_probe,after_probe)) for s in shard]
        root_call(distributed,lambda:atomic_json(args.output_dir/'output_attribution.json',probe_summary(probes)))
        coordinates=coordinate_analysis(policy,eligible,cache,distributed,args.output_dir) if args.coordinates else None
        if args.module_ablation or args.layer_ablation:
            module_ablation(policy,eligible,cache,initial,distributed,args.output_dir,
                selected=selected,layerwise=args.layer_ablation)
        export=root_call(distributed,lambda:export_actor(policy,config,learning,report,acceptance,args.case,args.output_dir,
            allow_measured_profile_transfer=args.allow_measured_profile_transfer)) if args.export_actor else None
        actor.load_state_dict(initial);optimizer.load_state_dict(optimizer_initial);controller.load_state_dict(controller_initial)
        anchor.load_state_dict(anchor_initial);restore_local_rng(rng,{'actor':generator})
        restored=exact_state_digest(actor.state_dict())==initial_digest
        if not restored:raise AssertionError('Candidate Actor byte restore failed')
        result=dict(status='passed',scope=args.scope,actor_scope=policy.actor_scope,learning_rate=args.learning_rate,
            actor_lr_multipliers=optimizer_settings.get('actor_lr_multipliers'),
            epochs=args.epochs,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            case=str(args.case),case_sha256=file_sha256(args.case/'case.json'),zero_update_probability=probability,
            actor=report,final_kl=kl,acceptance=acceptance,importance=final_importance,coordinates=coordinates,
            output=probe_summary(probes),history=history,attempts=len(attempt_history),actor_restored=restored,export=export,
            wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)),
            peak_allocated_bytes=max(distributed.all_gather_object(torch.cuda.max_memory_allocated(distributed.device))),
            physical_quality='unverified_counterfactual_generation_only',ESS_is_diagnostic_not_training_gate=True,
            fixed_parameter_diagnostic_reuses=evaluation_reuses)
        result['controller_initialization']=controller_origin
        root_call(distributed,lambda:atomic_json(args.output_dir/'summary.json',result))
    except BaseException as error:
        if rank==0 and args.output_dir.exists():
            import traceback
            atomic_json(args.output_dir/'failure.json',dict(type=type(error).__name__,message=str(error),
                traceback=traceback.format_exc(),scope=args.scope,epochs=args.epochs,physical_quality='unverified'))
        raise
    finally:
        if cache is not None:cache.close()
        if group is not None:dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__=='__main__':main()
