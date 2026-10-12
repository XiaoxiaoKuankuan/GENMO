"""Stage10 v3 八卡同进程闭环训练协调与有限验收运行主循环。

本模块组装Stage1可训练Actor、独立预训练特征Critic、冻结原生GMT及同进程Isaac
世界。先固定Actor采集独立真实轨迹预热Critic，再执行固定行为rollout、GAE、
多minibatch DPPO+BC、最后全20步KL和发布策略。普通执行数据更新后释放，完整
证据只在显式严格/诊断/异常边界保存。部署profile与训练墙钟独立，优化时不调用
任何物理推进。CLI给定有限轮次，配置的500轮预算不自动触发长期训练。

整轮快照覆盖Actor/Critic/两个Adam/BC/控制器/训练RNG，KL拒绝恢复这些状态后
停止会话；物理状态和已耗预算不退款。恢复独立v3完整checkpoint后重建PhysX
episode，丢弃旧rollout。固定评估使用独立GPU世界、训练RNG隔离和原奖励协议，
按累计真实控制交互触发；best保存实际完整权重。异步恢复点最多单在途，失败
显式传播，结束排空后才释放run锁。异常/受控停止与正常接受必须分别记录。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .budget import atomic_json
from .checkpoint_v3 import SCHEMA,AsyncCheckpointWriter,immutable_cpu_snapshot,load_v3,file_sha256
from .gradient_merge_v3 import BCGradientController
from .full_dataset import FullMusicCatalog
from .parallel_support import local_call,root_call,capture_local_rng,restore_local_rng
from .policy_v3 import DenseDiffusionPolicy
from .pretrained_critic import PretrainedConditionCritic
from .retention_anchor_v3 import Stage1RetentionAnchor
from .runtime_v3 import InProcessTrainingRuntime
from .run_management import RunManager,DiskGuard,StopSignal
from .tensor_cache import RolloutTensorCache
from .trainer import load_actor,trainable_actor_parameters
from .updater_v2 import probability_check_local
from .updater_v3 import actor_update_v3,critic_update_v3,final_kl_v3,check_kl_v3
from .periodic_monitor import log_metrics
from .evaluation import _model_fingerprint


class _FiniteDiagnosticComplete(Exception):
    """仅退出有限诊断主体；统一finally仍核验物理关闭并决定最终退出码。"""


def _sources(config):
    records={}
    for label,root in [('genmo',Path(config['paths']['genmo_repo'])),('gmt',Path(config['paths']['gmt_repo']))]:
        if label=='genmo':
            files=list((root/'gem').rglob('*.py'))
            files += [root/'tools'/name for name in ('train_stage10_v3_8gpu.py',
                'run_stage10_v3_8gpu.sh','train_closedloop_stage10.py')]
        else:
            files=list((root/'source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop').glob('*.py'))
            files+=list((root/'scripts/rsl_rl').glob('bumi4340*.py'))
            files+=list((root/'rsl_rl/rsl_rl').rglob('*.py'))
        records[label]={str(p.relative_to(root)):file_sha256(p) for p in sorted(set(files))}
    return records


def _snapshot(c,reason,*,wait=False):
    rank_state=dict(rank=c.distributed.rank,rng=capture_local_rng(c.generators),bc=c.bc.state_dict(),
        collector=c.vector.checkpoint_state())
    rank_states=c.distributed.all_gather_object(rank_state)
    def save():
        c.async_writer.poll()
        c.manager.check_disk(c.stage['storage']['checkpoint_reserve_bytes'])
        before=time.perf_counter()
        payload=immutable_cpu_snapshot(dict(schema=SCHEMA,identity=c.identity,config=c.base_config,
            actor=c.actor.state_dict(),critic=c.critic.state_dict(),actor_optimizer=c.actor_optimizer.state_dict(),
            critic_optimizer=c.critic_optimizer.state_dict(),bc_controller=c.controller.state_dict(),scheduler=None,
            state=c.state,rank_states=rank_states,budget=c.budget.state_dict(),
            actor_fingerprint=_model_fingerprint(c.actor)))
        queued=c.async_writer.submit(payload,reason=reason)
        queued['cpu_snapshot_seconds']=time.perf_counter()-before
        if wait:queued['publication']=c.async_writer.poll(wait=True)
        return queued
    return root_call(c.distributed,save)


def _rollback_snapshot(c):
    return immutable_cpu_snapshot(dict(actor=c.actor.state_dict(),critic=c.critic.state_dict(),
        actor_optimizer=c.actor_optimizer.state_dict(),critic_optimizer=c.critic_optimizer.state_dict(),
        bc=c.bc.state_dict(),controller=c.controller.state_dict(),rng=capture_local_rng(c.generators),
        actor_updates=c.state['actor_updates'],critic_updates=c.state['critic_updates']))


def _restore_rollback(c,saved):
    c.actor.load_state_dict(saved['actor']);c.critic.load_state_dict(saved['critic'])
    c.actor_optimizer.load_state_dict(saved['actor_optimizer']);c.critic_optimizer.load_state_dict(saved['critic_optimizer'])
    c.bc.load_state_dict(saved['bc']);c.controller.load_state_dict(saved['controller']);restore_local_rng(saved['rng'],c.generators)
    c.state['actor_updates'],c.state['critic_updates']=saved['actor_updates'],saved['critic_updates']
    c.actor_optimizer.zero_grad(set_to_none=True);c.critic_optimizer.zero_grad(set_to_none=True)
    from .rollback_audit import compare_recovered_state
    audit=compare_recovered_state(saved,_rollback_snapshot(c))
    if not audit['identical']:
        raise RuntimeError('Whole training rollback byte audit failed: '+json.dumps(audit,ensure_ascii=False))
    return audit


def _preserve_rollout_anomaly(c,buffer,targets,manifest,label):
    """在释放已完成采集前保存完整证据；异常保存失败不能掩盖原始训练错误。"""
    from .diagnostic_rollout_v3 import persist_rollout
    path=c.session/'anomalies'/label/f'rank{c.distributed.rank:02d}'/'rollout'
    return persist_rollout(path,buffer.transitions,targets,manifest,kernel=c.policy.kernel_config,
        identity=c.identity,full_execution=True,guard=c.guard)


def _evaluate(c,report,label):
    from .parallel_training import _evaluate_and_record
    result=_evaluate_and_record(c,report,label)
    if hasattr(c.bc,'validation_report'):
        from .policy_v3 import dense_precision_scope
        def retention():
            with dense_precision_scope(c.policy.precision):
                return c.bc.validation_report(c.actor,c.config,rank=c.distributed.rank,world=c.distributed.world_size)
        shards=c.distributed.all_gather_object(local_call(c.distributed,retention))
        samples=[row for shard in shards for row in shard['samples']]
        sources=sorted({row['meta']['dataset_id'] for row in samples})
        by_source={source:dict(count=sum(r['meta']['dataset_id']==source for r in samples),
            mean_loss=sum(r['loss'] for r in samples if r['meta']['dataset_id']==source)/
                sum(r['meta']['dataset_id']==source for r in samples)) for source in sources}
        result['stage1_retention']=dict(by_source=by_source,samples=samples,
            source_balanced_loss=sum(x['mean_loss'] for x in by_source.values())/len(by_source),
            source_loss_maturity=c.bc.source_global_step,scope=shards[0]['scope'])
    publication=_snapshot(c,'evaluated_'+label,wait=True)['publication']
    from .periodic_monitor import mark_saved_evaluation
    baseline=root_call(c.distributed,lambda:json.loads((c.output/'evaluation_baseline.json').read_text())
        if (c.output/'evaluation_baseline.json').exists() else result)
    selection=mark_saved_evaluation(result,baseline,publication,actor_fingerprint=_model_fingerprint(c.actor),
        output_dir=c.output,best_observed=result['selection']['best_observed'],
        best_saved=result['selection']['best_saved'])
    result['selection']=selection
    c.state.update({key:selection[key] for key in ('best_observed','best_saved')})
    def publish_selection():
        for key in ('best_observed','best_saved'):
            value=selection[key]
            if value is not None:atomic_json(c.output/(key+'.json'),value)
        atomic_json(c.session/'evaluations'/f'{label}.json',result)
        artifact=report['evaluation_artifacts'][-1]
        artifact['sha256']=file_sha256(c.output/artifact['path'])
        choice=selection.get('best_saved')
        if choice and choice.get('checkpoint',{}).get('path')==publication['path']:
            atomic_json(c.output/'best_checkpoint.json',publication)
    root_call(c.distributed,publish_selection)
    return result


def run(cargs,config,distributed):
    c=SimpleNamespace(distributed=distributed,config=copy.deepcopy(config),base_config=copy.deepcopy(config),
        stage=config['stage10'],settings=copy.deepcopy(config['stage9']),manager=None,writer=None,
        async_writer=None,vector=None,workers=None,maintenance=None,budget=None,output=cargs.output_dir.resolve())
    report=dict(schema='genmo.stage10.session.v3',status='running',iterations=[],warmup=[],evaluations=[],evaluation_artifacts=[])
    signal=StopSignal();signal.install();cache=None;buffer=None;error=None;saved=None;profile_token=None
    targets=None;manifest=None;label=None;anomaly_saved=None
    try:
        def initialize():
            c.manager=RunManager(c.output,resume=bool(cargs.resume),min_free_bytes=c.stage['storage']['min_free_bytes'],max_run_bytes=c.stage['storage']['max_run_bytes'])
            c.budget=c.manager.budget(c.stage['limits'],incremental=True)
            c.async_writer=AsyncCheckpointWriter(c.output,keep_last=c.stage['storage']['checkpoint_keep_last'])
            return dict(run_id=c.manager.run_id,session_id=c.manager.session_id)
        identity=root_call(distributed,initialize)
        c.session=c.output/'sessions'/identity['session_id'];c.rank_dir=c.session/f'rank{distributed.rank:02d}'
        c.rank_dir.mkdir(parents=True,exist_ok=False)
        c.guard=DiskGuard(c.session,min_free_bytes=c.stage['storage']['min_free_bytes'],max_run_bytes=c.stage['storage']['max_run_bytes'])
        c.config['runtime'].update(rank=distributed.rank,genmo_device=str(distributed.device),physics_device=str(distributed.device),
            asset_conversion_dir=str(c.rank_dir/'usd_assets'))
        c.config['stage9']['run_id']=identity['run_id'];c.settings=c.config['stage9']
        random.seed(c.stage['seed']);np.random.seed(c.stage['seed']);torch.manual_seed(c.stage['seed'])
        c.catalog=FullMusicCatalog(config['paths']['data_root'])
        audit=root_call(distributed,lambda:c.catalog.audit_files(require_audio=c.stage['dataset']['require_audio']))
        local_call(distributed,lambda:c.catalog.apply_audit(audit))
        c.actor,train,loading=local_call(distributed,lambda:load_actor(c.config))
        critic_config=c.stage['critic']
        if cargs.warmup_rollouts is not None:
            if cargs.warmup_rollouts<0:raise ValueError('Critic warmup count cannot be negative')
            critic_config['warmup_rollouts']=cargs.warmup_rollouts
            c.base_config['stage10']['critic']['warmup_rollouts']=cargs.warmup_rollouts
        def create_critic():
            if critic_config['type']=='pretrained_temporal':
                return PretrainedConditionCritic(c.actor,dimension=critic_config['dimension'],layers=critic_config['layers'],
                    heads=critic_config['heads'],freeze_encoders=critic_config['freeze_encoders']).to(distributed.device)
            if critic_config['type']=='legacy_mlp_control':
                from .critic import UpperCritic
                return UpperCritic(qpos_mean=c.actor.endecoder.mean,qpos_std=c.actor.endecoder.std,
                    proprio_scales=tuple(train.model.proprio_scales)).to(distributed.device)
            raise ValueError('Unknown explicit Critic architecture')
        c.critic=local_call(distributed,create_critic)
        distributed.broadcast_module(c.actor);distributed.broadcast_module(c.critic)
        precision=c.stage['precision']
        c.policy=DenseDiffusionPolicy(c.actor,std_schedule=c.settings['std_schedule'],precision=precision['mode'],
            attention=precision['attention'],linear_execution=precision['linear_execution'],
            compile_denoiser=precision['compile'],precise_output=precision['precise_output'],
            weight_gradient=precision.get('weight_gradient','direct_joint_gemm'))
        actor_initialization=dict(kind='source_stage1',source_global_step=loading['source_global_step'])
        if cargs.evaluation_actor:
            from .checkpoint_v3 import load_evaluation_actor
            actor_initialization=local_call(distributed,lambda:load_evaluation_actor(cargs.evaluation_actor,
                actor=c.actor,kernel=c.policy.kernel_config,assets=c.stage['assets']))
        c.actor_optimizer=torch.optim.AdamW(trainable_actor_parameters(c.actor),lr=c.settings['actor_lr'],weight_decay=0.)
        c.critic_optimizer=torch.optim.AdamW([p for p in c.critic.parameters() if p.requires_grad],lr=critic_config['learning_rate'],weight_decay=0.)
        c.controller=BCGradientController(**c.stage['bc_gradient'])
        anchor_config=copy.deepcopy(c.config)
        anchor_config['stage9'].update(bc_batch=c.settings['bc_batch']//distributed.world_size,
            seed=(c.stage['seed']+1000003*distributed.rank)%2**32)
        c.bc=Stage1RetentionAnchor(anchor_config,c.actor,train,source_global_step=loading['source_global_step'])
        c.bc.distributed_global_mean=True
        c.generators=dict(actor=torch.Generator().manual_seed(c.stage['seed']+3003),critic=torch.Generator().manual_seed(c.stage['seed']+2002))
        c.state=dict(iteration=0,rollouts_completed=0,warmup_completed=0,actor_updates=0,critic_updates=0,
            policy_version=0,optimizer_attempts=0,training_control_steps=0,training_physics_steps=0,
            training_upper_transitions=0,selected_actor_lr=c.settings['actor_lr'],
            execution_profile=dict(numerical_execution=c.policy.numerical_execution,microbatch=c.settings['denoising_microbatch']))
        assets=root_call(distributed,lambda:{k:file_sha256(config['paths'][k]) for k in c.stage['assets']})
        if assets!=c.stage['assets']:raise ValueError('Initial Stage1/GMT assets differ from declared SHA')
        c.identity=dict(schema='genmo.stage10.training_identity.v3',assets=assets,kernel=c.policy.kernel_config,
            training=c.stage['training'],critic=c.stage['critic'],bc_gradient=c.stage['bc_gradient'],kl=c.stage['kl'],
            scale=c.stage['scale'],reward=c.settings['reward'],runtime=c.base_config['runtime'],timing=c.base_config['timing'],
            dataset=c.stage['dataset'],evaluation=c.stage['evaluation'],
            source_files=root_call(distributed,lambda:_sources(config)),
            data_catalog=c.catalog.identity,source_stage1_step=loading['source_global_step'],
            actor_initialization=actor_initialization,
            recovery='fresh_PhysX_episodes_no_partial_rollout_reuse',torch_version=torch.__version__)
        payload=None
        if cargs.resume:
            pointer=root_call(distributed,lambda:json.loads((c.output/'latest.json').read_text()))
            checkpoint=c.output/pointer['path']
            if file_sha256(checkpoint)!=pointer['sha256']:raise ValueError('Complete checkpoint SHA mismatch')
            payload=local_call(distributed,lambda:load_v3(checkpoint,identity=c.identity,actor=c.actor,critic=c.critic,
                actor_optimizer=c.actor_optimizer,critic_optimizer=c.critic_optimizer,controller=c.controller))
            from .run_management import validate_budget_progress
            root_call(distributed,lambda:validate_budget_progress(payload['budget'],c.budget.state_dict()))
            c.state=copy.deepcopy(payload['state'])
            c.bc.load_state_dict(payload['rank_states'][distributed.rank]['bc'])
            restore_local_rng(payload['rank_states'][distributed.rank]['rng'],c.generators)
        from NoetixRobot.tasks.mimic.mimic_noetix_bumi4340_mha_sonic.closedloop.vector_backend import VectorFrozenGmtBackend
        from NoetixRobot.tasks.mimic.mimic_noetix_bumi4340_mha_sonic.closedloop.inprocess_service import InProcessVectorService
        service=local_call(distributed,lambda:InProcessVectorService(VectorFrozenGmtBackend(c.config)))
        if (torch.device(service.world.env.device)!=distributed.device
                or service.world.policy.device!=distributed.device):
            raise RuntimeError('Actor, frozen GMT and PhysX world must use the same allocated GPU')
        c.vector=InProcessTrainingRuntime(c,service);c.env=c.vector
        if payload:c.vector.restore_logical(payload['rank_states'][distributed.rank]['collector'])
        if payload:restore_local_rng(payload['rank_states'][distributed.rank]['rng'],c.generators)
        c.backend=SimpleNamespace(journal=None)
        from gem.closedloop.online_conditions import OnlineConditionBuilder
        from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
        from gem.robots.bumi.kinematics import BumiKinematics
        c.builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])))
        if distributed.rank==0:
            from torch.utils.tensorboard import SummaryWriter
            c.writer=SummaryWriter(str(c.output/'tensorboard'),max_queue=128,flush_secs=30)
        report.update(run_id=identity['run_id'],session_id=identity['session_id'],initial_state=copy.deepcopy(c.state))
        root_call(distributed,lambda:atomic_json(c.output/'training_identity.json',c.identity))
        root_call(distributed,lambda:(c.session/'resolved_config.yaml').write_text(yaml.safe_dump(c.base_config,allow_unicode=True,sort_keys=False)))
        if cargs.calibrate_only:
            calibration=c.vector.calibrate()
            report.update(status='passed',scope='single_request_deployment_calibration_not_training',
                calibrations=distributed.all_gather_object(calibration))
            root_call(distributed,lambda:atomic_json(c.output/'deployment_calibration.json',report))
            raise _FiniteDiagnosticComplete
        if payload is None:
            _snapshot(c,'initial',wait=True)
            if cargs.collect_only:
                report['initial_evaluation_skipped_for_collect_only']=True
            else:
                baseline=_evaluate(c,report,'initial')
                root_call(distributed,lambda:atomic_json(c.output/'evaluation_baseline.json',baseline))
        if cargs.evaluate_only:
            report.update(status='passed',scope='fixed_real_physics_evaluation_without_training',final_state=c.state)
            raise _FiniteDiagnosticComplete
        started=time.perf_counter();last_save=started
        next_eval=c.state['training_control_steps']+c.stage['evaluation']['every_control_steps']
        next_diagnostic=c.state['training_control_steps']+c.stage['storage']['diagnostic_every_control_steps']
        warmups=critic_config['warmup_rollouts'] if cargs.warmup_rollouts is None else cargs.warmup_rollouts
        while c.state['iteration']<cargs.stop_after_iteration:
            if any(distributed.all_gather_object(signal.stop_requested)) or time.perf_counter()-started>=c.stage['run_control']['max_walltime_seconds']:break
            is_warmup=c.state['warmup_completed']<warmups
            label=('warmup' if is_warmup else 'train')+f'_{c.state["rollouts_completed"]+1:06d}'
            strict=cargs.strict_evidence or c.state['training_control_steps']>=next_diagnostic
            if strict:next_diagnostic=c.state['training_control_steps']+c.stage['storage']['diagnostic_every_control_steps']
            round_start=time.perf_counter()
            anomaly_saved=None
            from .performance import PhaseProfiler,activate,deactivate
            frequency=c.stage['performance']['profiler_every']
            detailed=strict or c.state['rollouts_completed']==0 or (not is_warmup and c.state['iteration']==0) or (
                frequency>0 and c.state['rollouts_completed']%frequency==0)
            profiler=PhaseProfiler(distributed.device,distributed.rank,detailed=detailed)
            profile_token=activate(profiler)
            buffer,targets,manifest,collection=c.vector.collect(label,strict=strict)
            rows=buffer.transitions
            used=distributed.sum_tensor(torch.tensor([collection['actual_confirmed_control_steps'],collection['actual_confirmed_physics_steps'],len(rows)],
                device=distributed.device,dtype=torch.int64)).cpu().tolist()
            # 实际交互已发生，KL拒绝只恢复学习状态，不能让交互计数回退。
            c.state['training_control_steps']+=used[0]
            c.state['training_physics_steps']+=used[1]
            c.state['training_upper_transitions']+=used[2]
            cache=RolloutTensorCache(rows,targets,distributed.device,max_device_bytes=c.stage['storage']['tensor_cache_max_bytes'],preserve_device=True)
            probability_frequency=c.stage['checks']['full_probability_every']
            full_probability=(c.state['rollouts_completed']==0 or strict or
                (probability_frequency>0 and c.state['rollouts_completed']%probability_frequency==0))
            initial_probability=probability_check_local(c.policy,rows,global_manifest=manifest,distributed=distributed,
                denoising_microbatch=c.settings['kl_microbatch'],tensor_cache=cache,
                global_indices=None if full_probability else
                    [next(i for i,m in enumerate(manifest) if m['owner_rank']==rank and m['has_free'])
                     for rank in range(8) if any(m['owner_rank']==rank and m['has_free'] for m in manifest)])
            saved=local_call(distributed,lambda:_rollback_snapshot(c))
            if (not is_warmup and cargs.capture_validation_iteration is not None
                    and c.state['iteration']+1==cargs.capture_validation_iteration):
                from .diagnostic_rollout_v3 import persist_rollout
                case=c.session/'fixed_data_validation'/f'iteration{cargs.capture_validation_iteration:06d}'
                local_call(distributed,lambda:persist_rollout(case/f'rank{distributed.rank:02d}',rows,targets,manifest,
                    kernel=c.policy.kernel_config,identity=c.identity,full_execution=False,guard=c.manager.disk_guard if distributed.rank==0 else c.guard))
                local_call(distributed,lambda:torch.save(dict(bc=saved['bc'],rng=saved['rng']),
                    case/f'rank{distributed.rank:02d}'/'rank_state.pt'))
                root_call(distributed,lambda:torch.save(dict(actor=saved['actor'],actor_optimizer=saved['actor_optimizer'],
                    controller=saved['controller'],config=c.base_config,kernel=c.policy.kernel_config,identity=c.identity),case/'learning_state.pt'))
                rank_files=distributed.all_gather_object(dict(rank=distributed.rank,
                    rollout_manifest_sha256=file_sha256(case/f'rank{distributed.rank:02d}'/'manifest.json'),
                    rank_state_sha256=file_sha256(case/f'rank{distributed.rank:02d}'/'rank_state.pt')))
                root_call(distributed,lambda:atomic_json(case/'case.json',dict(schema='genmo.stage10.fixed_learning_case.v3',
                    real_global_chains=len(manifest),learning_state_sha256=file_sha256(case/'learning_state.pt'),
                    nonempty_adam=bool(saved['actor_optimizer']['state']),ranks=rank_files,
                    evidence_scope='real_PhysX_training_inputs_not_full_physical_archive')))
                if cargs.collect_only:
                    report.update(status='passed',scope='full_real_rollout_collection_not_training_acceptance',
                        fixed_data_case=str(case),collection_by_rank=distributed.all_gather_object(collection),
                        full_zero_update_probability=initial_probability)
                    raise _FiniteDiagnosticComplete
            update_start=time.perf_counter()
            physical_ticks=tuple(service.world.control_ticks)
            physics_time=service.world.env.sim.current_time
            critic=critic_update_v3(c.critic,c.critic_optimizer,rows,targets,cache,manifest,distributed=distributed,
                generator=c.generators['critic'],epochs=critic_config['epochs'],batch=critic_config['global_batch'])
            critic_seconds=max(distributed.all_gather_object(time.perf_counter()-update_start))
            c.state['critic_updates']+=critic['optimizer_steps']
            actor=None;kl=None
            if not is_warmup:
                actor_start=time.perf_counter()
                def reserve():
                    root_call(distributed,lambda:c.budget.reserve('actor_attempts',optimizer_attempts=1))
                    c.state['optimizer_attempts']+=1
                actor=actor_update_v3(c.policy,c.actor_optimizer,rows,targets,cache,manifest,c.controller,distributed=distributed,
                    anchor=c.bc,generator=c.generators['actor'],ppo_epochs=c.settings['ppo_epochs'],
                    minibatch_chains=c.stage['scale']['actor_minibatch_chains'],microbatch=c.settings['denoising_microbatch'],
                    sampled_steps=c.settings['ppo_steps_per_chain'],clip=c.settings['ppo_clip'],
                    gamma_denoising=c.settings['gamma_denoising'],soft_kl=c.settings['kl_soft_stop_coordinate'],
                    kl_reduction=c.stage['kl']['reduction'],objective_reduction=c.settings['objective_logprob_reduction'],
                    grad_clip_norm=c.settings['grad_clip_norm'],reserve_attempt=reserve)
                actor_seconds=max(distributed.all_gather_object(time.perf_counter()-actor_start))
                kl=final_kl_v3(c.policy,rows,cache,manifest,distributed=distributed,microbatch=c.settings['kl_microbatch'],
                    clip=c.settings['ppo_clip'],objective_reduction=c.settings['objective_logprob_reduction'])
                accepted=check_kl_v3(kl,c.stage['kl'])
                if not accepted['passed']:
                    anomaly_saved=local_call(distributed,lambda:_preserve_rollout_anomaly(c,buffer,targets,manifest,label))
                    rollback=local_call(distributed,lambda:_restore_rollback(c,saved))
                    rollbacks=distributed.all_gather_object(rollback)
                    root_call(distributed,lambda:atomic_json(c.session/(label+'_rejected.json'),dict(kl=kl,acceptance=accepted,actor=actor,
                        physical_and_resource_consumption_retained=True,training_state_restored=True,rollback_audits=rollbacks)))
                    raise RuntimeError('Final Full20 KL rejected; whole training update rolled back')
                c.state['actor_updates']+=actor['optimizer_steps']
                c.state['policy_version']+=1;c.state['iteration']+=1
                root_call(distributed,lambda:c.budget.accept_iteration('update',identity=f'{identity["session_id"]}/{label}'))
            else:
                actor_seconds=0.;c.state['warmup_completed']+=1
            def assert_paused_physics():
                if tuple(service.world.control_ticks)!=physical_ticks or service.world.env.sim.current_time!=physics_time:
                    raise RuntimeError('Training optimization advanced the PhysX world')
            local_call(distributed,assert_paused_physics)
            c.state['rollouts_completed']+=1
            cache_report=cache.report();cache.close();cache=None;buffer.clear();buffer=None;saved=None;del rows
            wall=max(distributed.all_gather_object(time.perf_counter()-round_start))
            profile_report=profiler.report();deactivate(profile_token);profile_token=None
            record=dict(label=label,state=copy.deepcopy(c.state),collection_by_rank=distributed.all_gather_object(collection),
                probability=initial_probability,critic=critic,actor=actor,full20_kl=kl,tensor_cache=cache_report,
                actor_seconds=actor_seconds,critic_seconds=critic_seconds,wall_seconds=wall,
                performance_by_rank=distributed.all_gather_object(profile_report),physics_paused_during_optimization=True,
                effective_upper_transitions_per_second=used[2]/wall,mode='critic_warmup' if is_warmup else 'dppo_bc')
            root_call(distributed,lambda:(c.manager.append_metrics(record),log_metrics(c.writer,c.output/'curves.jsonl',c.state['rollouts_completed'],
                record,prefix='v3',durable=False)))
            report['warmup' if is_warmup else 'iterations'].append(dict(label=label,wall_seconds=wall,actor_steps=0 if actor is None else actor['optimizer_steps']))
            if distributed.rank==0:print(f'[V3 ACCEPTED] {label} chains={used[2]} actor_steps={0 if actor is None else actor["optimizer_steps"]} wall={wall:.3f}s',flush=True)
            if c.state['training_control_steps']>=next_eval:
                _evaluate(c,report,f'{c.state["iteration"]:06d}')
                next_eval=c.state['training_control_steps']+c.stage['evaluation']['every_control_steps']
            if time.perf_counter()-last_save>=c.stage['storage']['checkpoint_interval_seconds']:
                _snapshot(c,'periodic');last_save=time.perf_counter()
            root_call(distributed,lambda:c.async_writer.poll())
        _snapshot(c,'controlled_end',wait=True)
        _evaluate(c,report,'final_'+f'{c.state["iteration"]:06d}')
        report.update(status='passed',final_state=c.state)
    except _FiniteDiagnosticComplete:
        pass
    except BaseException as caught:
        import traceback
        error=caught;report.update(status='failed',error=dict(type=type(caught).__name__,message=str(caught),traceback=traceback.format_exc()))
        traceback.print_exc()
        from .rollback_audit import release_failed_computation
        release_failed_computation(caught)
        if buffer is not None and targets is not None and manifest is not None:
            try:
                if anomaly_saved is None:
                    anomaly_saved=local_call(distributed,lambda:_preserve_rollout_anomaly(c,buffer,targets,manifest,label))
                report['anomaly_rollouts']=distributed.all_gather_object(anomaly_saved)
            except BaseException as evidence_error:
                report['anomaly_save_error']=repr(evidence_error)
        if saved is not None:
            try:
                c.actor_optimizer.zero_grad(set_to_none=True);c.critic_optimizer.zero_grad(set_to_none=True)
                rollback=local_call(distributed,lambda:_restore_rollback(c,saved))
                report['rollback_audits']=distributed.all_gather_object(rollback)
                report['whole_training_state_restored']=True
            except BaseException as restore_error:
                report['whole_training_state_restored']=False;report['rollback_error']=repr(restore_error)
    finally:
        if profile_token is not None:
            from .performance import deactivate
            deactivate(profile_token)
        if cache is not None:cache.close()
        if buffer is not None:buffer.clear()
        if hasattr(c,'identity'):
            try:
                report['source_unchanged']=root_call(distributed,lambda:_sources(config)==c.identity['source_files'])
                if not report['source_unchanged']:raise RuntimeError('Bound runtime source changed during session')
            except BaseException as caught:error=error or caught
        if c.vector is not None:
            try:report['world_shutdown']=local_call(distributed,c.vector.close)
            except BaseException as caught:
                report['world_shutdown_error']=repr(caught);error=error or caught
        if distributed.rank==0:
            try:
                if c.async_writer is not None:report['checkpoint_shutdown']=c.async_writer.close()
            except BaseException as caught:error=error or caught
            finally:
                if c.writer is not None:c.writer.close()
                if c.manager is not None:c.manager.close()
        report['exit_code']=1 if error else 0
        if error:report['status']='failed'
        if hasattr(c,'session'):
            root_call(distributed,lambda:atomic_json(c.session/'summary.json',report))
        signal.restore()
    return 1 if error else 0
