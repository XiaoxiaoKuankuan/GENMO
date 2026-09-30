"""第九步冻结 GMT 闭环的有限 DPPO 采集、训练与恢复入口。

本入口默认只采集，使用 train 音乐、单环境 latency 执行以及独立 Stage9 输出目录。
真实环境始终由另一个解释器运行既有 GMT worker；Actor、Critic 和优化器只在本进程。
preflight 不加载大模型或启动仿真；train 才按校准、零更新对照、64条采集、Critic、
一次DPPO及BC、更新后对照的顺序运行。resume-check 恢复完整状态后重建 worker 并
采集16条新转移，不重复使用旧 Buffer。所有模式共享不可重置的磁盘预算。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import yaml

from gem.closedloop.baseline_provenance import collect_source_provenance, verify_source_provenance
from gem.closedloop.dppo.budget import RunBudget, atomic_json
from gem.closedloop.dppo.buffer import RolloutBuffer, StepJournal
from gem.closedloop.dppo.checkpoint import load_checkpoint, save_checkpoint
from gem.closedloop.dppo.critic import UpperCritic
from gem.closedloop.dppo.env_adapter import UpperEnvironment
from gem.closedloop.dppo.music_tasks import TrainMusicSampler
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.rpc import AcknowledgedBackend
from gem.closedloop.dppo.rewards import DEFAULT_CONFIG
from gem.closedloop.dppo.trainer import (load_actor, fixed_targets, critic_update, actor_update,
    probability_check, analytic_kl, SupervisedAnchor)
from gem.closedloop.evaluation_music import load_music_features, sha256_file
from gem.closedloop.frozen_actor import _fingerprint
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from tools.eval.run_closedloop_baseline import Workers, preflight


def configuration(path):
    override=yaml.safe_load(Path(path).read_text())
    base=override.pop('base_config')
    config=yaml.safe_load((Path(path).parent/base).read_text())
    def merge(target,source):
        for key,value in source.items():
            if isinstance(value,dict) and isinstance(target.get(key),dict):
                merge(target[key],value)
            else:
                target[key]=value
    merge(config,override)
    import copy
    reward=copy.deepcopy(DEFAULT_CONFIG)
    merge(reward,config['stage9'].get('reward',{}))
    config['stage9']['reward']=reward
    s=config['stage9']
    required=dict(rollout_upper_steps=64,ppo_epochs=1,denoising_steps=20,actor_lr=1e-6,
                  critic_lr=1e-4,gamma_upper=.99,lambda_upper=.95,gamma_denoising=.99,
                  ppo_clip=.01,grad_clip_norm=1.,denoising_microbatch=1,bc_batch=2)
    for key,value in required.items():
        if s[key]!=value:
            raise ValueError(f'This bounded acceptance requires {key}={value}')
    if s['execution_mode'] not in ('latency','paused'):
        raise ValueError('Invalid execution mode')
    if any(s[key]>limit for key,limit in (('max_generations',256),('max_control_steps',10000),('max_iterations',3))):
        raise ValueError('Stage9 hard limit exceeded in config')
    return config


def calibrate(env,sampler,output):
    task=sampler.next_task()
    env.reset_task(task['sample'],task['music'],seed=42,phase='calibration')
    durations=[]
    parity=None
    for i in range(16):
        generated=env.generate()
        if i==0:
            trace=generated['trace']
            parity=probability_check(env.policy,[SimpleNamespace(
                transition_valid=True,context=generated['context'],chain=trace['chain'][0],
                old_log_prob=trace['old_log_probs'][0],free_mask=trace['free_mask'][0],
                metadata={'sampler_trace':trace})])
        if generated['rejection'] or generated['prepared'] is None:
            raise RuntimeError('Stochastic preflight rejected a reference before any update')
        started=time.perf_counter()
        env.backend.call('commit_plan',prepared_plan_id=generated['prepared']['prepared_plan_id'],
                         expected_control_tick=env.snapshot['tick'])
        elapsed=generated['elapsed']+time.perf_counter()-started
        env.snapshot=env.backend.call('snapshot')
        env.decision+=1
        if i>=4:
            durations.append(elapsed)
        print(f'[CALIBRATION] {i+1}/16 {elapsed:.4f}s',flush=True)
    peak=max(durations)
    env.latency_budget_s=math.ceil((peak+max(.04,.25*peak))*50)/50
    report=dict(scope='limited_preflight_4_plus_12',durations=durations,
                latency_budget_s=env.latency_budget_s,includes_prepare_and_commit=True,
                real_model_probability_check=parity)
    atomic_json(output/'calibration.json',report)
    return report


def comparison(env,sampler,kind,output):
    sample=sampler.selected[0]
    music=load_music_features(env.config['paths']['data_root'],sample)
    env.reset_task(sample,music,seed=1729,phase=f'comparison_{kind}')
    env.comparison_noise_index=0
    results=[]
    for _ in range(4):
        result=env.step(deterministic=kind=='A')
        if dataclasses.is_dataclass(result):
            summary=dict(executed_control_steps=result.executed_control_steps,
                terminated=result.terminated,truncated=result.truncated,reason=result.reason,
                rewards=result.rewards.tolist(),metadata=result.metadata)
        else:
            summary=result
        # 大型链保留原始pt；人读报告只引用其位置。
        meta=summary['metadata']
        results.append(dict(executed_control_steps=summary['executed_control_steps'],reason=summary['reason'],
            terminated=summary['terminated'],truncated=summary['truncated'],reward_sum=sum(summary['rewards']),
            rejection=meta['rejection'],latency_seconds=meta['latency_seconds'],
            raw_sample_path=meta['raw_sample_path'],reward_details=meta['reward_details']))
        if summary['terminated'] or summary['truncated']:
            break
    report=dict(group=kind,sample=sample,seed=1729,results=results)
    env.comparison_noise_index=None
    atomic_json(output/f'comparison_{kind}.json',report)
    if kind=='B' and (not results or all(r['rejection'] or r['executed_control_steps']<25 for r in results)):
        raise RuntimeError('Stochastic zero-update execution failed its reference/stability gate')
    return report


def collect(env,sampler,count,output,phase):
    buffer=RolloutBuffer(capacity=count)
    starts=0
    while len(buffer.transitions)<count:
        task=sampler.next_task()
        env.reset_task(task['sample'],task['music'],seed=42+starts,phase=phase)
        starts+=1
        while True:
            transition=env.step()
            # 首四个真实episode覆盖四来源，行政截断保留可信bootstrap。
            cutoff=(len(buffer.transitions)+1==count or starts<=3)
            if cutoff and not transition.terminated:
                transition.truncated=True
                transition.reason=transition.reason or 'collection_boundary'
            buffer.append(transition)
            # 每条完整转移立即保存；中途预算/基础设施故障仍保留可审阅的已完成部分。
            partial=output/'rollout.partial.pt.tmp'
            torch.save(buffer.transitions,partial)
            os.replace(partial,output/'rollout.partial.pt')
            print(f'[COLLECT] {len(buffer.transitions)}/{count} {task["sample"]["dataset"]} '
                  f'm={transition.executed_control_steps} reason={transition.reason}',flush=True)
            if len(buffer.transitions)==count or transition.terminated or transition.truncated:
                break
    torch.save(buffer.transitions,output/'rollout.pt')
    (output/'rollout.partial.pt').unlink(missing_ok=True)
    return buffer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/closedloop/stage9_dppo_smoke.yaml')
    parser.add_argument('--mode',choices=('preflight','collect','critic','train','resume-check','eval'),default='collect')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--budget-file',type=Path,help='跨失败诊断和恢复共享的全轮预算；缺省为输出目录/budget.json')
    args=parser.parse_args(argv)
    if (args.mode=='resume-check') != bool(args.resume):
        parser.error('--resume is required only for resume-check')
    output=args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and args.mode!='resume-check':
        parser.error('Output must be new; resume-check uses the original run directory')
    output.mkdir(parents=True,exist_ok=True)
    report_output=output/('resume_check' if args.mode=='resume-check' else 'acceptance')
    report_output.mkdir(exist_ok=False)
    config=configuration(args.config)
    config['stage9']['run_id']=output.name
    config['output_root']=str(output)
    s=config['stage9']
    config_path=report_output/'resolved_config.yaml'
    config_path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    random.seed(s['seed']);np.random.seed(s['seed']);torch.manual_seed(s['seed'])
    torch.set_num_threads(int(config['runtime']['torch_threads']))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    budget=RunBudget(args.budget_file or output/'budget.json',generations=s['max_generations'],control_steps=s['max_control_steps'],iterations=s['max_iterations'])
    sampler=TrainMusicSampler(config['paths']['data_root'],s['music_selection'],seed=s['seed'])
    workers=journal=backend=check=None
    report=dict(mode=args.mode,status='running',stage9_passed=False)
    code=0
    try:
        check=preflight(config,sampler.selected,check_gpu=args.mode!='preflight')
        atomic_json(report_output/'preflight.json',check)
        if not check['ready']:
            raise RuntimeError('Assets or train music did not pass preflight')
        stage9_sources={
            'genmo_repo':[*(f'gem/closedloop/dppo/{name}.py' for name in
                ('__init__','policy','buffer','rewards','critic','returns','music_tasks','env_adapter','rpc','budget','trainer','checkpoint')),
                'tools/train_closedloop_dppo.py','gem/closedloop/losses.py','gem/closedloop/stage1_dataset.py',
                'configs/closedloop/stage9_dppo_smoke.yaml','configs/closedloop/stage9_dppo_server1.yaml'],
            'gmt_repo':['source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop/execution_journal.py']}
        provenance=collect_source_provenance(config['paths'],repository_state=check['repositories'],additional_files=stage9_sources)
        atomic_json(report_output/'source_identity.json',provenance)
        if args.mode=='preflight':
            report.update(status='passed',preflight_only=True)
            return 0
        actor,train_config,loading=load_actor(config)
        atomic_json(report_output/'actor_loading.json',loading)
        critic=UpperCritic(qpos_mean=actor.endecoder.mean,qpos_std=actor.endecoder.std,
            proprio_scales=tuple(train_config.model.proprio_scales)).to(config['runtime']['genmo_device'])
        actor_optimizer=torch.optim.AdamW(actor.parameters(),lr=s['actor_lr'],weight_decay=0.)
        critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=s['critic_lr'],weight_decay=0.)
        policy=DPPODiffusionPolicy(actor,steps=s['denoising_steps'],eta=s['eta'],std_floor=s['std_floor'],guidance_scale=2.5)
        bc=SupervisedAnchor(config,actor,train_config) if args.mode in ('train','resume-check') else None
        generator=torch.Generator().manual_seed(s['seed']+2002)
        identity=dict(assets=check['asset_sha256'],actor_interface=dict(actor.interface_config),
                      sampler=dict(steps=s['denoising_steps'],eta=s['eta'],std_floor=s['std_floor'],guidance_scale=2.5),
                      environment=config['environment'],termination=config['termination'],
                      source_manifest_sha256=provenance['source_manifest_sha256'],
                      reward=s['reward'],training_contract={k:s[k] for k in
                          ('gamma_upper','lambda_upper','gamma_denoising','ppo_clip','bc_weight','bc_batch','actor_lr','critic_lr')},
                      music_selection_sha256=sampler.selection_sha256,
                      bc_manifests={name:sha256_file(Path(s['bc_data_root'])/name/'manifests/train.jsonl')
                          for name in ('AIST++','AIOZ-GDANCE','FineDance','Mine')})
        state=dict(iteration=0,policy_version=0,actor_updates=0,critic_updates=0,buffer_size=0,pending_plan=False)
        samplers=dict(music=sampler)
        if bc is not None:
            samplers['bc']=bc
        if args.resume:
            state=load_checkpoint(args.resume,actor=actor,critic=critic,actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,identity=identity,samplers=samplers,generators={'critic':generator})
            live_budget=budget.state_dict()
            if state['budget']['limits']!=live_budget['limits'] or any(
                    live_budget['used'][key]<value for key,value in state['budget']['used'].items()):
                raise RuntimeError('Resume budget cannot roll back checkpoint consumption')
        workers=Workers(config,report_output)
        socket_path=Path(workers.temp.name)/'gmt.sock'
        command=[config['paths']['isaac_python'],'-B',str(Path(config['paths']['gmt_repo'])/'scripts/rsl_rl/serve_frozen_gmt.py'),
                 '--config',str(config_path),'--socket',str(socket_path),'--headless']
        client=workers.start('gmt',command,config['paths']['gmt_repo'],socket_path)
        journal=StepJournal(report_output/'execution_journal.sqlite')
        backend=AcknowledgedBackend(client,journal,socket_path=socket_path,timeout_s=config['runtime']['rpc_timeout_s'])
        builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])))
        env=UpperEnvironment(config,backend,builder,policy,budget,report_output)
        env.policy_version=state['policy_version'];env.iteration=state['iteration']
        if args.resume:
            env.latency_budget_s=state['latency_budget_s']
            env.attempt=max(state['attempt'],budget.state_dict()['used']['generations'])
            env.episode_count=state['episode_count']
            report['resume']=dict(restored_full_state=True,new_backend_session_id=backend.session_id,
                restored_actor_updates=state['actor_updates'],old_buffer_discarded=True)
        else:
            report['calibration']=calibrate(env,sampler,report_output)
            # 校准和对照不得消费正式采集音乐采样顺序。
            sampler=TrainMusicSampler(config['paths']['data_root'],s['music_selection'],seed=s['seed'])
            samplers['music']=sampler
        if args.mode in ('train','eval'):
            report['comparison_A']=comparison(env,sampler,'A',report_output)
            report['comparison_B']=comparison(env,sampler,'B',report_output)
        if args.mode=='eval':
            report['source_unchanged']=verify_source_provenance(provenance)
            if not report['source_unchanged']['unchanged']:
                raise RuntimeError('Source changed during frozen comparison')
            report.update(status='passed',updates_performed=False)
        if args.mode!='eval':
            count=16 if args.resume else s['rollout_upper_steps']
            buffer=collect(env,sampler,count,report_output,'resume' if args.resume else 'main')
            report['probability_check']=probability_check(policy,buffer.transitions)
            report['collected_upper_transitions']=len(buffer.transitions)
            report['prefix_over_18_fraction']=sum(int(t.context['known_qpos30_mask'].any(-1).sum())>18 for t in buffer.transitions)/count
            if args.mode in ('critic','train'):
                targets=fixed_targets(buffer.transitions,critic,config['runtime']['genmo_device'])
                torch.save(targets,report_output/'fixed_targets.pt')
                initial_actor=_fingerprint(actor)
                report['critic']=critic_update(critic,critic_optimizer,buffer.transitions,targets,generator=generator)
                report['critic']['actor_unchanged']=initial_actor==_fingerprint(actor)
                if not report['critic']['actor_unchanged']:
                    raise RuntimeError('Value loss altered Actor parameters')
                state['critic_updates']+=20
            if args.mode=='train':
                budget.reserve('update',iterations=1)
                before_critic=_fingerprint(critic)
                report['actor']=actor_update(policy,actor_optimizer,buffer.transitions,targets,bc=bc,bc_weight=s['bc_weight'],
                    clip=s['ppo_clip'],gamma_denoising=s['gamma_denoising'],grad_clip_norm=s['grad_clip_norm'])
                report['actor']['critic_unchanged']=before_critic==_fingerprint(critic)
                report['kl']=analytic_kl(policy,buffer.transitions)
                if report['kl']['mean_joint_kl']>s['kl_stop_joint']:
                    torch.save(dict(actor=actor.state_dict(),report=report),report_output/'failed_update_candidate.pt')
                    raise RuntimeError('Joint analytic KL exceeded configured stop threshold')
                state['actor_updates']+=1;state['iteration']+=1;state['policy_version']+=1
                env.policy_version=state['policy_version'];env.iteration=state['iteration']
                report['comparison_C']=comparison(env,sampler,'C',report_output)
            buffer.clear()
            report['source_unchanged']=verify_source_provenance(provenance)
            if not report['source_unchanged']['unchanged']:
                raise RuntimeError('Source changed while acceptance was running')
            if args.mode=='train':
                workers.entries[-1]['client']=backend.client
                workers.close()
                report['worker_shutdown']=workers.shutdown
                gmt=workers.shutdown['gmt']
                workers=None
                if (gmt.get('close_error') or gmt.get('forced_shutdown') or gmt.get('process_exit_code')!=0
                        or gmt.get('policy_unchanged') is not True or gmt.get('runtime_parameters_unchanged') is not True):
                    raise RuntimeError('GMT frozen-state verification failed before checkpoint publication')
                if any(sha256_file(config['paths'][key])!=digest for key,digest in check['asset_sha256'].items()):
                    raise RuntimeError('Original model/stats/FK asset changed during acceptance')
                state.update(budget=budget.state_dict(),latency_budget_s=env.latency_budget_s,
                             attempt=env.attempt,episode_count=env.episode_count)
                path=save_checkpoint(output/'checkpoints'/'stage9_000001.pt',actor=actor,critic=critic,
                    actor_optimizer=actor_optimizer,critic_optimizer=critic_optimizer,state=state,identity=identity,
                    config=config,samplers=samplers,generators={'critic':generator})
                report['checkpoint']=str(path)
            report.update(status='passed',stage9_passed=False,
                          acceptance_scope='resume_validation' if args.resume else args.mode)
    except BaseException as exc:
        code=1
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        if workers is not None:
            if backend is not None:
                workers.entries[-1]['client']=backend.client
            workers.close()
            report['worker_shutdown']=workers.shutdown
            gmt=workers.shutdown.get('gmt',{})
            if (gmt.get('close_error') or gmt.get('forced_shutdown') or gmt.get('process_exit_code')!=0
                    or gmt.get('policy_unchanged') is not True or gmt.get('runtime_parameters_unchanged') is not True):
                report['status']='failed';code=1
        if journal is not None and hasattr(journal,'close'):
            journal.close()
        if check is not None and args.mode!='preflight':
            current_hashes={key:sha256_file(config['paths'][key]) for key in check['asset_sha256']}
            report['original_assets_unchanged']=current_hashes==check['asset_sha256']
            if not report['original_assets_unchanged']:
                report['status']='failed';code=1
        if code==0 and args.mode=='resume-check':
            original=json.loads((output/'acceptance'/'summary.json').read_text())
            report['stage9_passed']=bool(original.get('mode')=='train' and original.get('status')=='passed'
                and original.get('actor',{}).get('parameters_changed')
                and original.get('actor',{}).get('ppo_only_gradient_norm',0)>0
                and original.get('critic',{}).get('parameters_changed')
                and original.get('probability_check',{}).get('passed')
                and original.get('collected_upper_transitions')==64
                and report.get('collected_upper_transitions')==16
                and report.get('probability_check',{}).get('passed'))
            atomic_json(output/'stage9_acceptance.json',dict(stage9_passed=report['stage9_passed'],
                training_summary=str(output/'acceptance'/'summary.json'),
                resume_summary=str(report_output/'summary.json'),budget=budget.state_dict()))
        report['budget']=budget.state_dict()
        report['exit_code']=code
        atomic_json(report_output/'summary.json',report)
        print(json.dumps(dict(status=report['status'],output=str(report_output),exit_code=code),ensure_ascii=False),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
