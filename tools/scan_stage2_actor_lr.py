"""服务器2八卡、同一真实rollout的统一Actor学习率与优化目标扫描。

仅读取SHA核验的完整15+4+1行为链、已验收MC Critic产生的GAE和未更新Stage1 Actor。
所有候选从同一空Adam初始化开始；每种目标只计算一次全局PPO/BC固定梯度，LR和
KL门槛候选复用这份梯度，BC/RNG状态相同。记录真实尝试数及共享梯度计算成本，
不把复用梯度误报为重复训练，也不从其他候选继承动量。每次候选仍独立回溯并执行
全链最终KL；失败证据保留。输出评估专用实际Actor权重，不是训练恢复checkpoint。

显著生成越界预筛以同噪声探针相对Stage1的硬越界增量为准：最大量增加超过0.05rad
或越界比例增加超过2个百分点则不进入物理评估。这是有限实验筛选规则，不宣称
机器人安全；所有剩余不同权重候选仍需完整歌曲物理评估，数值相同权重可复用一次。
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from gem.closedloop.dppo.actor_backtracking import backtracked_step,check_candidate_kl
from gem.closedloop.dppo.actor_step_diagnostics import output_probe,compare_probes,advantage_diagnostics
from gem.closedloop.dppo.actor_experiment import optimization_binding
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.full_batch_actor import prepare_full_batch_gradient
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.kl_calibration import digest
from gem.closedloop.dppo.optimization_support import probability_check_local
from gem.closedloop.dppo.parallel_support import capture_local_rng,restore_local_rng,local_call,root_call
from gem.closedloop.dppo.retention_anchor import Stage1RetentionAnchor
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.run_management import TrainingBudget
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor,trainable_actor_parameters
from gem.closedloop.dppo.training_checkpoint import file_sha256,immutable_cpu_snapshot
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.updater import final_kl


def prescreen(transaction):
    if not transaction['accepted']:return dict(passed=False,reasons=['all_KL_attempts_rejected'])
    diagnostic=transaction['attempts'][-1]['diagnostics'];p=diagnostic['parameters']['modules']['all']
    output=diagnostic['outputs'];reasons=[]
    if p['changed_elements']==0:reasons.append('no_effective_parameter_change')
    if output['final_joint_rms_mean'] is None or output['final_joint_rms_mean']<=1e-7:
        reasons.append('no_measurable_reference_change')
    samples=output['samples']
    if not samples or any('hard_max_excess_after' not in r for r in samples):
        reasons.append('missing_authoritative_joint_limit_probe')
    else:
        before=max(r['hard_max_excess_before'] for r in samples);after=max(r['hard_max_excess_after'] for r in samples)
        f0=sum(r['hard_violation_fraction_before'] for r in samples)/len(samples)
        f1=sum(r['hard_violation_fraction_after'] for r in samples)/len(samples)
        if after>before+.05:reasons.append('generated_max_hard_violation_increase_above_0.05rad')
        if f1>f0+.02:reasons.append('generated_hard_violation_fraction_increase_above_0.02')
    return dict(passed=not reasons,reasons=reasons,scope='finite_same_noise_probe_not_physics_acceptance')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('Eight Server2 GPU ranks required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=40))
    preflight=[None]
    if rank==0:
        try:
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            if busy:raise RuntimeError('Existing GPU job; refusing to compete: '+busy)
            args.output.mkdir(parents=True,exist_ok=False)
            preflight[0]={'ok':True}
        except BaseException as e:preflight[0]={'ok':False,'error':repr(e)}
    dist.broadcast_object_list(preflight)
    if not preflight[0]['ok']:raise RuntimeError(preflight[0]['error'])
    torch.cuda.set_device(rank);torch.set_num_threads(4)
    group=dist.new_group(backend='nccl',timeout=timedelta(minutes=30))
    d=DistributedCollectives(rank,8,tensor_group=group,device=f'cuda:{rank}')
    cache=None
    try:
        case=json.loads((args.case/'case.json').read_text())
        if case['schema']!='genmo.stage2.asymmetric_fixed_learning_case.v1' or 'joint_limits' not in case:
            raise ValueError('Fresh identified real rollout with authoritative runtime joint limits required')
        if file_sha256(args.case/'learning_state.pt')!=case['learning_state_sha256']:
            raise ValueError('Fixed learning state SHA changed')
        rankdir=args.case/f'rank{rank:02d}';record=next(x for x in case['ranks'] if x['rank']==rank)
        if file_sha256(rankdir/'rank_state.pt')!=record['rank_state_sha256'] or file_sha256(rankdir/'manifest.json')!=record['rollout_manifest_sha256']:
            raise ValueError('Fixed rank state changed')
        inputs=local_call(d,lambda:load_rollout(rankdir))
        rows,targets,manifest=(inputs[k] for k in ('transitions','targets','global_manifest'))
        source=torch.load(args.case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
        rank_state=torch.load(rankdir/'rank_state.pt',map_location='cpu',weights_only=False)
        if source['actor_optimizer']['state']:raise ValueError('This scan starts all candidates from the same empty Stage1 Adam')
        configs=json.loads(args.candidates.read_text())['candidates']
        config=copy.deepcopy(source['config']);settings=config['stage9'];stage=config['stage10']
        scan_stage=configuration(configs[0]['path'])['stage10']
        if any(configuration(v['path'])['stage10']['bc_gradient']!=scan_stage['bc_gradient'] for v in configs):
            raise ValueError('Every candidate must use the same BC gradient target/controller settings')
        config['runtime']['genmo_device']=str(d.device)
        actor,train,loading=local_call(d,lambda:load_actor(config))
        if any(not torch.equal(v.cpu(),source['actor'][k]) for k,v in actor.state_dict().items()):
            raise ValueError('Fixed rollout Actor is not unchanged Stage1 s595000')
        policy=build_policy(actor,settings,stage['precision'])
        if policy.kernel_config!=source['kernel'] or inputs['kernel']!=source['kernel']:
            raise ValueError('Behavior kernel differs from actual sampling')
        policy.experimental_joint_limits=case['joint_limits']
        optimizer=torch.optim.AdamW(trainable_actor_parameters(actor),lr=5e-8,weight_decay=0.)
        base_adam=copy.deepcopy(optimizer.state_dict())
        bc_config=copy.deepcopy(config);bc_config['stage9'].update(bc_batch=128,seed=42+1000003*rank)
        anchor=Stage1RetentionAnchor(bc_config,actor,train,source_global_step=loading['source_global_step'])
        cache=RolloutTensorCache(rows,targets,d.device,max_device_bytes=8*1024**3)
        probability=probability_check_local(policy,rows,global_manifest=manifest,distributed=d,
            denoising_microbatch=128,tensor_cache=cache)
        generators=dict(actor=torch.Generator(),critic=torch.Generator())
        bound_std=torch.tensor([.04,.06,.08,.1],dtype=torch.float32,device=d.device)
        selected=[r for r in rows if r.transition_valid and bool(r.free_mask.any())]
        std=cache.get('old_stds',selected)[:,15:19,0,0]
        if len(selected) and not torch.equal(std,bound_std.expand_as(std)):
            raise ValueError('Actual Gaussian std is not the declared four-step 0.04/0.06/0.08/0.10')
        limits=dict(generations=1,control_steps=1,physics_steps=4,accepted_iterations=30,optimizer_attempts=270)
        budget=TrainingBudget(args.output/'budget.json',limits) if rank==0 else None
        report=dict(schema='genmo.stage2.uniform_lr_fixed_scan.v1',source_case=str(args.case),
            source_case_sha256=file_sha256(args.case/'case.json'),kernel_sha256=digest(policy.kernel_config),
            reward_sha256=digest(settings['reward']),same_stage1_actor=True,same_empty_adam=True,
            critic_initialization=stage['critic']['preheat_artifact'],real_chains=len(manifest),
            actual_four_step_std=bound_std.cpu().tolist(),probability=probability,
            advantages=advantage_diagnostics(rows,targets,d),gradient_computations=[],candidates=[],
            clip_at_old_policy_does_not_limit_the_first_gradient=True)
        for objective in ('coordinate_dppo','joint_score_pg'):
            actor.load_state_dict(source['actor']);optimizer.load_state_dict(copy.deepcopy(base_adam))
            anchor.load_state_dict(rank_state['bc']);restore_local_rng(rank_state['rng'],generators)
            controller=BCGradientController(**scan_stage['bc_gradient'])
            torch.cuda.reset_peak_memory_stats();d.barrier();started=time.perf_counter()
            gradient,observation,before,local=prepare_full_batch_gradient(policy,optimizer,rows,targets,cache,manifest,controller,
                distributed=d,anchor=anchor,objective=objective,minibatch_chains=16384,microbatch=128,
                clip=.003,gamma_denoising=settings['gamma_denoising'],bc_samples_per_rollout=1024)
            gradients={n:None if p.grad is None else p.grad.clone() for n,p in actor.named_parameters()}
            control=copy.deepcopy(controller.state_dict());rng=capture_local_rng(generators)
            gradient['shared_across_all_LR_and_KL_candidates_of_this_objective']=True
            report['gradient_computations'].append(gradient)
            def save_components():
                path=args.output/(objective+'_gradient_components.pt')
                with path.open('xb') as f:torch.save(immutable_cpu_snapshot(observation.components),f)
                return dict(path=str(path),sha256=file_sha256(path))
            gradient['component_vectors']=root_call(d,save_components)
            for candidate in (v for v in configs if v['objective']==objective):
                candidate_config=configuration(candidate['path']);s=candidate_config['stage10']
                if file_sha256(candidate['path'])!=candidate['sha256']:raise ValueError('Candidate configuration changed')
                actor.load_state_dict(source['actor']);optimizer.load_state_dict(copy.deepcopy(base_adam))
                optimizer.param_groups[0]['lr']=candidate['actor_lr']
                controller.load_state_dict(control);restore_local_rng(rng,generators)
                for n,p in actor.named_parameters():p.grad=None if gradients[n] is None else gradients[n].clone()
                def evaluate():
                    return final_kl(policy,rows,cache,manifest,distributed=d,microbatch=128,clip=.003,
                        objective_reduction='free_coordinate_mean')
                def observe():
                    parameter=local_call(d,lambda:observation.finish(optimizer,controller.value,gradient['gradient_clip_factor']))
                    same=d.all_gather_object(parameter)
                    if any(v!=same[0] for v in same):raise RuntimeError('Rank parameter update diagnostics differ')
                    after=local_call(d,lambda:output_probe(policy,local,cache))
                    return dict(parameters=parameter,outputs=compare_probes(before,after,d))
                def reserve():root_call(d,lambda:budget.reserve(candidate['name'],optimizer_attempts=1))
                torch.cuda.reset_peak_memory_stats();d.barrier();started=time.perf_counter()
                transaction=backtracked_step(actor,optimizer,controller,evaluate=evaluate,limits=s['kl'],distributed=d,
                    generators=generators,reserve_attempt=reserve,observe=observe,**s['training']['optimizer_backtracking'])
                final=evaluate()  # 接受候选仍执行第二次全链审查；拒绝后则证明恢复为零更新。
                verdict=check_candidate_kl(final,s['kl'])
                item=dict(**candidate,gradient=gradient,backtracking=transaction,final_kl=final,final_acceptance=verdict,
                    screening=prescreen(transaction),wall_seconds=max(d.all_gather_object(time.perf_counter()-started)),
                    peak_allocated_bytes_by_rank=d.all_gather_object(torch.cuda.max_memory_allocated()))
                if transaction['accepted'] and verdict['passed']:
                    def save_actor():
                        path=args.output/(candidate['name']+'.pt')
                        fingerprint=_model_fingerprint(actor)
                        payload=dict(schema='genmo.stage2.asymmetric_actor_evaluation.v1',evaluation_only=True,
                            complete_update_and_soft_kl_passed=True,actor=immutable_cpu_snapshot(actor.state_dict()),
                            kernel=policy.kernel_config,source_identity=source['identity'],source_case=str(args.case),
                            source_case_sha256=report['source_case_sha256'],trainable4_kl=final,hard_kl=verdict,
                            actor_experiment=dict(optimization=optimization_binding(s),physics_quality_validated=False),
                            update=dict(optimizer_steps=1,planned_optimizer_steps=1,objective=objective,
                                actor_lr=candidate['actor_lr'],accepted_lr=transaction['accepted_lr'],
                                real_global_chains=gradient['global_chain_count'],kl_limits=s['kl'],
                                global_minibatch_chains=16384,ppo_epochs=1))
                        with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                        return dict(path=str(path),sha256=file_sha256(path),actor_fingerprint=fingerprint,evaluation_only=True)
                    item['evaluation_actor']=root_call(d,save_actor)
                report['candidates'].append(item)
                report['budget']=root_call(d,lambda:budget.state_dict())
                root_call(d,lambda:atomic_json(args.output/'report.json',report))
                if rank==0:print(json.dumps(dict(candidate=candidate['name'],accepted_lr=transaction['accepted_lr'],
                    attempts=transaction['attempt_count'],screening=item['screening']),ensure_ascii=False),flush=True)
            del gradients,observation
        report['status']='completed_fixed_data_scan_not_physical_acceptance'
        root_call(d,lambda:atomic_json(args.output/'report.json',report))
    finally:
        if cache is not None:cache.close()
        dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
