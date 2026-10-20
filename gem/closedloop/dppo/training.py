"""Stage10 八卡同进程闭环训练协调与有限验收运行主循环。

本模块组装Stage1可训练Actor、独立预训练特征Critic、冻结原生GMT及同进程Isaac
世界。先固定Actor采集独立真实轨迹预热Critic，再执行固定行为rollout、GAE、
一次全局策略梯度/DPPO+BC更新、可训练四步完整KL和发布策略。普通执行数据更新后释放，完整
证据只在显式严格/诊断/异常边界保存。部署profile与训练墙钟独立，优化时不调用
任何物理推进。CLI给定累计停止轮次，实验和正式任务分别受各自显式预算约束。

每次Actor更新采用完整四步KL回溯，超限只缩小当前步；保留先前接受更新及独立
Critic，跨轮缓慢调整统一学习率。物理状态和已耗预算不退款，普通拒绝仅保存小型
诊断。只在显式固定数据导出时保留完整学习快照；非有限和资源故障仍安全退出。
恢复独立当前完整checkpoint后重建PhysX
episode，丢弃旧rollout。固定评估使用独立GPU世界、训练RNG隔离和原奖励协议，
按累计真实控制交互触发；best保存实际完整权重。异步恢复点最多单在途，失败
显式传播，结束排空后才释放run锁。异常/受控停止与正常接受必须分别记录。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .budget import atomic_json
from .training_checkpoint import SCHEMA,AsyncCheckpointWriter,immutable_cpu_snapshot,load_training_checkpoint,file_sha256
from .gradient_merge import BCGradientController
from .full_dataset import FullMusicCatalog
from .parallel_support import local_call,root_call,capture_local_rng,restore_local_rng
from .pretrained_critic import PretrainedConditionCritic
from .retention_anchor import Stage1RetentionAnchor
from .runtime import InProcessTrainingRuntime
from .run_management import RunManager,DiskGuard,StopSignal
from .tensor_cache import RolloutTensorCache
from .trainer import load_actor,trainable_actor_parameters
from .optimization_support import probability_check_local
from .updater import actor_update,critic_update,final_kl,check_kl
from .full_batch_actor import full_batch_actor_update
from .compact_evidence import summary_only
from .periodic_monitor import log_metrics
from .tensorboard_metrics import DashboardWriter, training_scalars
from .evaluation import _model_fingerprint


class _FiniteDiagnosticComplete(Exception):
    """仅退出有限诊断主体；统一finally仍核验物理关闭并决定最终退出码。"""


def _requires_initial_evaluation(args, state, baseline_exists):
    if args.collect_only or getattr(args,'critic_preheat_only',False) or baseline_exists:
        return False
    if state['actor_updates'] or state['policy_version']:
        raise ValueError('Updated Actor has no durable initial evaluation; cannot invent Stage1 baseline')
    return True


def _next_interaction_boundary(used, interval):
    """按累计真实控制步对齐周期；重启不把下一次评估延后一个完整周期。"""
    if type(used) is not int or used < 0 or type(interval) is not int or interval < 1:
        raise ValueError('Nonnegative actual interactions and positive integer interval required')
    return (used // interval + 1) * interval


def _validate_calibrated_execution(profile, numerical_execution, assets, kernel=None):
    """部署延迟必须对应已测资产和实际前向；说明性mode文字不参与算子身份。"""
    if profile.get('measured_assets') != assets:
        raise ValueError('Deployment calibration assets differ from this Actor/GMT experiment')
    if kernel is not None:
        digest=hashlib.sha256(json.dumps(kernel,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if profile.get('behavior_kernel_sha256') != digest:
            raise ValueError('Behavior kernel changed; explicit deployment calibration required')
    measured = profile.get('numerical_execution', {})
    keys = ('schema', 'precision', 'linear', 'attention', 'attention_fallback',
            'matmul_tf32', 'master_dtype', 'layer_residual_dtype', 'probability_dtype',
            'final_projection', 'inference_requires_grad', 'condition_encoding',
            'network_rows', 'torch_version', 'compiler')
    changed = [key for key in keys if key not in measured or key not in numerical_execution
               or measured[key] != numerical_execution[key]]
    if changed:
        raise ValueError('Deployment execution changed; rerun explicit single-request calibration: '
                         + ', '.join(changed))


def _sources(config):
    records={}
    for label,root in [('genmo',Path(config['paths']['genmo_repo'])),('gmt',Path(config['paths']['gmt_repo']))]:
        if label=='genmo':
            files=list((root/'gem').rglob('*.py'))
            files += [root/'tools/train_closedloop_stage10.py',
                      root/'scripts/train_stage10_8gpu_server1.sh']
        else:
            robot_root=root/'source/NoetixRobot/NoetixRobot'
            # 实际任务的env/mdp/PD/观测会改变物理转移，不能只绑定closedloop适配层。
            files=list((robot_root/'tasks/mimic/mimic_noetix_bumi4340_mha_sonic').rglob('*.py'))
            files+=list(robot_root.rglob('*.py'))
            for directory in ('actuators','managers','terrains','assets/robots/bumi3_4340_lowpd'):
                files+=list((robot_root/directory).rglob('*.py'))
            files+=list((root/'scripts/rsl_rl').glob('bumi4340*.py'))
            files+=list((root/'scripts/motion').rglob('*.py'))
            # 参考转换、关节映射与任务配置同样会改变真实转移和恢复身份。
            files+=list((robot_root/'tasks/mimic/mimic_noetix_bumi4340_mha_sonic').rglob('*.yaml'))
            files+=[p for p in (robot_root/'assets/robots/bumi3_4340_lowpd').rglob('*') if p.is_file()]
            files+=list((root/'rsl_rl/rsl_rl').rglob('*.py'))
            # Isaac Lab基类和管理器直接决定控制/物理生命周期；同样绑定实际源码。
            lab=Path(config.get('runtime',{}).get('isaaclab_root','/nonexistent_isaaclab'))
            files+=list((lab/'source/isaaclab/isaaclab').rglob('*.py'))
        records[label]={str(p.relative_to(root)) if p.is_relative_to(root) else 'external/'+str(p):file_sha256(p) for p in sorted(set(files))}
    return records


def _periodic_evaluation_label(state):
    """预热时Actor轮次恒为零；按已完成rollout及交互计数区分每次周期评估。"""
    return (f'periodic_r{state["rollouts_completed"]:06d}_'
            f'c{state["training_control_steps"]:012d}_a{state["iteration"]:06d}')


def _physical_configuration_identity(config):
    """恢复同时绑定显式物理/终止配置和实际侧车资产，拒绝权重相同而任务改变。"""
    return dict(environment=copy.deepcopy(config['environment']),
        termination=copy.deepcopy(config['termination']),
        gmt_agent_config_sha256=file_sha256(config['paths']['gmt_agent_config']),
        isaac_contract_sha256=file_sha256(config['paths']['isaac_contract']))


def _overhead(c,name,started):
    """评估、MC、快照等非迭代前台耗时，累计到下一条轮次记录后清零；不进入checkpoint或身份。"""
    overhead=getattr(c,'overhead',None)
    if overhead is not None:overhead[name]=overhead.get(name,0.)+time.perf_counter()-started


def _poll_checkpoint(c,*,wait=False):
    """仅root调用：权重完整发布之后，才绑定对应评估并一次性写最终JSON。"""
    publication=c.async_writer.poll(wait=wait)
    pending=getattr(c,'pending_evaluation',None)
    if publication is not None and pending is not None:
        result,label,report,baseline=pending
        if publication['iteration']!=result['iteration']:
            raise RuntimeError('Pending evaluation does not match published checkpoint iteration')
        from .periodic_monitor import mark_saved_evaluation
        selection=mark_saved_evaluation(result,baseline,publication,
            actor_fingerprint=result['actor_identity']['model_fingerprint'],output_dir=c.output,
            best_observed=result['selection']['best_observed'],best_saved=result['selection']['best_saved'])
        result['selection']=selection
        c.state.update({key:selection[key] for key in ('best_observed','best_saved')})
        started=time.perf_counter()
        for key in ('best_observed','best_saved'):
            if selection[key] is not None:atomic_json(c.output/(key+'.json'),selection[key])
        path=c.session/'evaluations'/f'{label}.json'
        atomic_json(path,result)
        report['evaluation_artifacts'].append(dict(label=label,iteration=result['iteration'],
            path=str(path.relative_to(c.output)),sha256=file_sha256(path)))
        report['evaluations'].append(label)
        choice=selection.get('best_saved')
        if choice and choice.get('checkpoint',{}).get('path')==publication['path']:
            atomic_json(c.output/'best_checkpoint.json',publication)
        c.pending_evaluation=None
        _overhead(c,'evaluation_selection_publish_seconds',started)
    return publication


def _snapshot(c,reason,*,wait=False):
    rank_state=dict(rank=c.distributed.rank,rng=capture_local_rng(c.generators),bc=c.bc.state_dict(),
        collector=c.vector.checkpoint_state())
    rank_states=c.distributed.all_gather_object(rank_state)
    def save():
        _poll_checkpoint(c,wait=True)
        c.manager.check_disk(c.stage['storage']['checkpoint_reserve_bytes'])
        before=time.perf_counter()
        payload=immutable_cpu_snapshot(dict(schema=SCHEMA,identity=c.identity,config=c.base_config,
            actor=c.actor.state_dict(),critic=c.critic.state_dict(),actor_optimizer=c.actor_optimizer.state_dict(),
            critic_optimizer=c.critic_optimizer.state_dict(),bc_controller=c.controller.state_dict(),scheduler=c.actor_lr.state_dict(),
            state=c.state,rank_states=rank_states,budget=c.budget.state_dict(),
            actor_fingerprint=_model_fingerprint(c.actor)))
        queued=c.async_writer.submit(payload,reason=reason)
        queued['cpu_snapshot_seconds']=time.perf_counter()-before
        if wait:queued['publication']=_poll_checkpoint(c,wait=True)
        return queued
    result=root_call(c.distributed,save)
    c.last_saved_learning_version=_learning_version(c)
    c.last_saved_accepted_version=_accepted_version(c)
    return result


def _diagnostic_learning_snapshot(c):
    return immutable_cpu_snapshot(dict(actor=c.actor.state_dict(),critic=c.critic.state_dict(),
        actor_optimizer=c.actor_optimizer.state_dict(),critic_optimizer=c.critic_optimizer.state_dict(),
        bc=c.bc.state_dict(),controller=c.controller.state_dict(),rng=capture_local_rng(c.generators),
        actor_updates=c.state['actor_updates'],critic_updates=c.state['critic_updates']))


def _preserve_rollout_anomaly(c,buffer,targets,manifest,label):
    """在释放已完成采集前保存完整证据；异常保存失败不能掩盖原始训练错误。"""
    from .diagnostic_rollout import persist_rollout
    path=c.session/'anomalies'/label/f'rank{c.distributed.rank:02d}'/'rollout'
    return persist_rollout(path,buffer.transitions,targets,manifest,kernel=c.policy.kernel_config,
        identity=c.identity,full_execution=True,guard=c.guard)


def _evaluate(c,report,label,*,save=True):
    from .evaluation_runtime import _evaluate as evaluate
    root_call(c.distributed,lambda:_poll_checkpoint(c,wait=True))
    started=time.perf_counter()
    result=evaluate(c,label,write_report=False)
    _overhead(c,'evaluation_run_seconds',started)
    alert=(result.get('paired_baseline') or {}).get('regression_alert') or {}
    from .periodic_monitor import advance_regression_alert
    result['consecutive_regression_alert']=advance_regression_alert(c.state,alert,c.state['iteration'])
    if result['consecutive_regression_alert']['triggered'] and c.distributed.rank==0:
        print(f'[STAGE10 ALERT] {label} paired reward delta upper90={alert["upper90"]:.4f}<0; training continues',flush=True)
    if hasattr(c.bc,'validation_report'):
        stage=time.perf_counter()
        from .dense_policy import dense_precision_scope
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
        root_call(c.distributed,lambda:c.writer.write('validation',c.state['iteration'],
            {'Loss/stage1_validation':result['stage1_retention']['source_balanced_loss']}) if c.writer else None)
        _overhead(c,'evaluation_stage1_retention_seconds',stage)
    stage=time.perf_counter()
    baseline=root_call(c.distributed,lambda:summary_only(json.loads((c.output/'evaluation_baseline.json').read_text()))
        if (c.output/'evaluation_baseline.json').exists() else result)
    if c.distributed.rank==0:c.pending_evaluation=(result,label,report,baseline)
    if save:
        _snapshot(c,'evaluated_'+label)
    _overhead(c,'evaluation_total_seconds',started)
    return result


def _learning_version(c):
    """退出维护凭据只在本会话有效，绑定完成计数和两个网络的真实Tensor版本。"""
    counters=tuple(c.state.get(key) for key in
        ('iteration','rollouts_completed','actor_updates','critic_updates','policy_version'))
    tensors=tuple((id(value),value._version) for module in (c.actor,c.critic)
        for value in list(module.parameters())+list(module.buffers()))
    budget=json.dumps(c.state.get('budget'),sort_keys=True,separators=(',',':'))
    return counters,tensors,budget


def _accepted_version(c):
    """只跟踪已接受学习水位，连续拒绝不因物理/RNG前进反复保存同一模型。"""
    return tuple(c.state.get(key,0) for key in ('iteration','actor_updates','policy_version'))


def _completed_training_resume(args,state,limit):
    """累计训练已达请求水位时直接结束；显式评估、MC等只读诊断仍允许执行。"""
    modes=('calibrate_only','evaluate_only','collect_only','critic_preheat_only','mc_validation_only')
    return bool(args.resume and state['iteration']>=limit
        and not any(getattr(args,name,False) for name in modes))


def _finish_controlled_run(c,report,*,stopped):
    """已在最终轮维护中保存的同一完整状态不重复评估/序列化；任意rank失配则重做。"""
    stamp=getattr(c,'final_maintenance_version',None)
    if stamp is None and stopped:stamp=getattr(c,'last_saved_learning_version',None)
    reusable=stamp is not None and stamp==_learning_version(c)
    if all(c.distributed.all_gather_object(reusable)):
        root_call(c.distributed,lambda:_poll_checkpoint(c,wait=True))
        report['final_maintenance_reused']=True
        return
    if not stopped:
        _evaluate(c,report,'final_'+f'{c.state["iteration"]:06d}',save=False)
    _snapshot(c,'controlled_end',wait=True)


def _maintain_after_iteration(c,report,*,is_warmup,rejected,last_save,finishing=False):
    """先评估与MC维护，再合并保存原因；同轮只发布一份最终恢复状态。"""
    distributed=c.distributed
    complete_mc=c.stage['critic'].get('complete_mc',{})
    save_reasons=[]
    iteration_eval=c.stage['evaluation'].get('every_actor_iterations')
    periodic_evaluation=(c.state['training_control_steps']>=c.state['next_evaluation_control_steps'] or
            iteration_eval and not is_warmup and not rejected and c.state['iteration']%iteration_eval==0)
    if periodic_evaluation or finishing:
        c.state['next_evaluation_control_steps']=_next_interaction_boundary(
            c.state['training_control_steps'],c.stage['evaluation']['every_control_steps'])
        evaluation_label=(_periodic_evaluation_label(c.state) if periodic_evaluation else
            'final_'+f'{c.state["iteration"]:06d}')
        _evaluate(c,report,evaluation_label,save=False)
        save_reasons.append('evaluated_'+evaluation_label)
    from .run_control import iteration_checkpoint_reason
    checkpoint_reason=iteration_checkpoint_reason(c.state['iteration'],c.stage['storage'],
        is_warmup=is_warmup,rejected=rejected)
    if checkpoint_reason or time.perf_counter()-last_save>=c.stage['storage']['checkpoint_interval_seconds']:
        stage=time.perf_counter()
        save_reasons.append(checkpoint_reason or 'periodic')
    iteration_mc=complete_mc.get('every_actor_iterations')
    if complete_mc.get('enabled') and (c.state['training_control_steps']>=c.state['next_mc_calibration_control_steps'] or
            iteration_mc and not is_warmup and not rejected and c.state['iteration']%iteration_mc==0):
        from .mc_calibration import calibrate_critic
        stage=time.perf_counter()
        calibrate_critic(c,'policy_'+str(c.state['policy_version']),fit=False,
            fit_if_ev_below=complete_mc.get('refit_if_ev_below'))
        _overhead(c,'mc_calibration_seconds',stage)
        c.state['next_mc_calibration_control_steps']=_next_interaction_boundary(c.state['training_control_steps'],complete_mc['every_control_steps'])
        save_reasons.append('mc_calibrated')
    if finishing:save_reasons.append('controlled_end')
    rejected_unsaved=rejected and getattr(c,'last_saved_accepted_version',None)!=_accepted_version(c)
    if rejected_unsaved:save_reasons.append('retained_actor_state')
    if save_reasons:
        stage=time.perf_counter()
        if finishing or rejected_unsaved:
            _snapshot(c,'+'.join(save_reasons),wait=True)
            if finishing:c.final_maintenance_version=_learning_version(c)
        else:_snapshot(c,'+'.join(save_reasons))
        last_save=time.perf_counter()
        _overhead(c,'maintenance_snapshot_seconds',stage)
    return last_save


def run(cargs,config,distributed):
    c=SimpleNamespace(distributed=distributed,config=copy.deepcopy(config),base_config=copy.deepcopy(config),
        stage=config['stage10'],settings=copy.deepcopy(config['stage9']),manager=None,writer=None,
        async_writer=None,vector=None,workers=None,maintenance=None,budget=None,output=cargs.output_dir.resolve(),
        overhead={})
    report=dict(schema='genmo.stage10.session',status='running',iterations=[],warmup=[],evaluations=[],evaluation_artifacts=[])
    from .run_control import build_policy,validate_stopping_iteration
    stop_after_iteration=validate_stopping_iteration(cargs.stop_after_iteration,c.stage)
    signal=StopSignal();signal.install();cache=None;buffer=None;error=None;saved=None;profile_token=None
    targets=None;manifest=None;label=None;anomaly_saved=None
    try:
        def initialize():
            c.manager=RunManager(c.output,resume=bool(cargs.resume),min_free_bytes=c.stage['storage']['min_free_bytes'],max_run_bytes=c.stage['storage']['max_run_bytes'])
            c.budget=c.manager.budget(c.stage['limits'])
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
            return PretrainedConditionCritic(c.actor,dimension=critic_config['dimension'],layers=critic_config['layers'],
                heads=critic_config['heads'],history_scales=critic_config['history_scales']).to(distributed.device)
        c.critic=local_call(distributed,create_critic)
        distributed.broadcast_module(c.actor);distributed.broadcast_module(c.critic)
        precision=c.stage['precision']
        c.policy=build_policy(c.actor,c.settings,precision)
        if not cargs.calibrate_only:
            _validate_calibrated_execution(c.config['timing']['deployment_profile'],
                c.policy.numerical_execution,c.stage['assets'],c.policy.kernel_config)
        actor_initialization=dict(kind='source_stage1',source_global_step=loading['source_global_step'])
        if cargs.evaluation_actor:
            from .training_checkpoint import load_evaluation_actor
            binding=root_call(distributed,lambda:dict(reward=c.settings['reward'],timing=c.base_config['timing'],
                physical_configuration=_physical_configuration_identity(c.base_config),source_files=_sources(config),
                generation_horizon=c.stage['generation_horizon'],data_catalog=c.catalog.identity))
            actor_initialization=local_call(distributed,lambda:load_evaluation_actor(cargs.evaluation_actor,
                actor=c.actor,kernel=c.policy.kernel_config,assets=c.stage['assets'],execution_binding=binding))
        c.actor_optimizer=torch.optim.AdamW(trainable_actor_parameters(c.actor),lr=c.settings['actor_lr'],weight_decay=0.)
        history_ids={id(p) for p in c.critic.history_encoder.parameters()}
        c.critic_optimizer=torch.optim.AdamW([
            dict(params=list(c.critic.history_encoder.parameters()),lr=critic_config['history_learning_rate'],name='tracking_history'),
            dict(params=[p for p in c.critic.parameters() if p.requires_grad and id(p) not in history_ids],
                 lr=critic_config['learning_rate'],name='temporal_value')],weight_decay=0.)
        c.controller=BCGradientController(**c.stage['bc_gradient'])
        from .actor_lr import ActorLRController
        c.actor_lr=ActorLRController(c.settings['actor_lr'],c.settings['actor_lr_adaptation'])
        anchor_config=copy.deepcopy(c.config)
        anchor_config['stage9'].update(bc_batch=c.settings['bc_batch']//distributed.world_size,
            seed=(c.stage['seed']+1000003*distributed.rank)%2**32)
        c.bc=Stage1RetentionAnchor(anchor_config,c.actor,train,source_global_step=loading['source_global_step'])
        c.generators=dict(actor=torch.Generator().manual_seed(c.stage['seed']+3003),critic=torch.Generator().manual_seed(c.stage['seed']+2002))
        c.state=dict(iteration=0,rollouts_completed=0,warmup_completed=0,actor_updates=0,critic_updates=0,
            policy_version=0,optimizer_attempts=0,training_control_steps=0,training_physics_steps=0,
            training_upper_transitions=0,selected_actor_lr=c.settings['actor_lr'],
            execution_profile=dict(numerical_execution=c.policy.numerical_execution,microbatch=c.settings['denoising_microbatch']))
        assets=root_call(distributed,lambda:{k:file_sha256(config['paths'][k]) for k in c.stage['assets']})
        if assets!=c.stage['assets']:raise ValueError('Initial Stage1/GMT assets differ from declared SHA')
        c.identity=dict(schema='genmo.stage10.training_identity',assets=assets,kernel=c.policy.kernel_config,
            training=c.stage['training'],critic=c.stage['critic'],bc_gradient=c.stage['bc_gradient'],kl=c.stage['kl'],
            scale=c.stage['scale'],reward=c.settings['reward'],runtime=c.base_config['runtime'],timing=c.base_config['timing'],
            dataset=c.stage['dataset'],evaluation=c.stage['evaluation'],
            run_control=c.stage['run_control'],resource_limits=c.stage['limits'],
            physical_episode=c.stage['physical_episode'],ppo_rollout=c.stage['ppo_rollout'],generation_horizon=c.stage['generation_horizon'],
            physical_configuration=root_call(distributed,lambda:_physical_configuration_identity(c.base_config)),
            source_files=root_call(distributed,lambda:_sources(config)),
            data_catalog=c.catalog.identity,source_stage1_step=loading['source_global_step'],
            actor_initialization=actor_initialization,
            recovery='fresh_PhysX_episodes_no_partial_rollout_reuse',torch_version=torch.__version__)
        payload=None
        if not cargs.resume and not cargs.evaluate_only and critic_config.get('preheat_artifact'):
            from .critic_preheat import initialize_critic
            imported=local_call(distributed,lambda:initialize_critic(critic_config['preheat_artifact'],
                identity=c.identity,actor=c.actor,critic=c.critic,optimizer=c.critic_optimizer,
                genmo_repo=c.config['paths']['genmo_repo'],quality_gate=critic_config['complete_mc']['quality_gate']))
            c.state.update(critic_initialization=imported,mc_preheat_complete=True,
                critic_updates=imported['selected_updates'],
                training_control_steps=imported['prior_mc_control_steps'],
                training_physics_steps=imported['prior_mc_physics_steps'],
                prior_mc_control_steps=imported['prior_mc_control_steps'],
                prior_mc_physics_steps=imported['prior_mc_physics_steps'])
            c.state['next_mc_calibration_control_steps']=_next_interaction_boundary(
                c.state['training_control_steps'],critic_config['complete_mc']['every_control_steps'])
        if cargs.resume:
            pointer=root_call(distributed,lambda:json.loads((c.output/'latest.json').read_text()))
            checkpoint=c.output/pointer['path']
            if file_sha256(checkpoint)!=pointer['sha256']:raise ValueError('Complete checkpoint SHA mismatch')
            payload=local_call(distributed,lambda:load_training_checkpoint(checkpoint,identity=c.identity,actor=c.actor,critic=c.critic,
                actor_optimizer=c.actor_optimizer,critic_optimizer=c.critic_optimizer,controller=c.controller,lr_controller=c.actor_lr))
            from .run_management import validate_budget_progress
            root_call(distributed,lambda:validate_budget_progress(payload['budget'],c.budget.state_dict()))
            c.state=copy.deepcopy(payload['state'])
            c.last_saved_accepted_version=_accepted_version(c)
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
        c.builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])),motion_frames=config['stage10']['generation_horizon']['frames'])
        if distributed.rank==0:
            c.writer=DashboardWriter(c.output/'tensorboard_compact')
        report.update(run_id=identity['run_id'],session_id=identity['session_id'],initial_state=copy.deepcopy(c.state))
        root_call(distributed,lambda:atomic_json(c.output/'training_identity.json',c.identity))
        root_call(distributed,lambda:(c.session/'resolved_config.yaml').write_text(yaml.safe_dump(c.base_config,allow_unicode=True,sort_keys=False)))
        if _completed_training_resume(cargs,c.state,stop_after_iteration):
            report.update(status='passed',scope='already_at_requested_cumulative_iteration_no_updates_or_extra_checkpoint',
                final_state=c.state)
            raise _FiniteDiagnosticComplete
        if cargs.calibrate_only:
            calibration=c.vector.calibrate(warmup=c.config['timing']['calibration_warmup'],
                samples=c.config['timing']['calibration_samples'])
            report.update(status='passed',scope='single_request_deployment_calibration_not_training',
                calibrations=distributed.all_gather_object(calibration))
            root_call(distributed,lambda:atomic_json(c.output/'deployment_calibration.json',report))
            raise _FiniteDiagnosticComplete
        if payload is None:
            _snapshot(c,'initial',wait=True)
        if not (cargs.calibrate_only or cargs.collect_only or cargs.evaluate_only or getattr(cargs,'critic_preheat_only',False)):
            from .actor_optimization import optimization_binding,validate_optimization_configuration
            validate_optimization_configuration(c.config,c.stage['derived']['maximum_actor_steps'])
            report['actor_optimization']=dict(optimization=optimization_binding(c.stage),
                physics_quality_validated=False)
        complete_mc=c.stage['critic'].get('complete_mc',{})
        if getattr(cargs,'mc_validation_only',False):
            from .mc_calibration import calibrate_critic
            report['independent_mc_validation']=calibrate_critic(c,'validation_'+str(c.state['policy_version']),fit=False)
            report.update(status='passed',scope='independent_complete_MC_no_network_updates',final_state=c.state)
            raise _FiniteDiagnosticComplete
        if complete_mc.get('enabled') and not cargs.collect_only and not cargs.evaluate_only and not c.state.get('mc_preheat_complete'):
            from .mc_calibration import calibrate_critic
            report['mc_preheat']=calibrate_critic(c,'preheat',fit=True)
            c.state['mc_preheat_complete']=report['mc_preheat']['quality_gate_passed']
            c.state['next_mc_calibration_control_steps']=_next_interaction_boundary(c.state['training_control_steps'],complete_mc['every_control_steps'])
            _snapshot(c,'complete_mc_preheat',wait=True)
            if not c.state['mc_preheat_complete'] and not getattr(cargs,'critic_preheat_only',False):
                raise RuntimeError('Independent complete MC validation is insufficient; Actor learning remains disabled')
        if getattr(cargs,'critic_preheat_only',False):
            report.update(status='passed',scope='complete_MC_critic_preheat_without_Actor_updates',final_state=c.state)
            raise _FiniteDiagnosticComplete
        # 正式训练的固定价值诊断必须绑定已预热Critic；纯Actor评估不隐式训练Critic。
        if cargs.collect_only:
            report['initial_evaluation_skipped_for_finite_data_diagnostic']=True
        elif _requires_initial_evaluation(cargs,c.state,root_call(distributed,lambda:(c.output/'evaluation_baseline.json').exists())):
            baseline=_evaluate(c,report,'initial')
            root_call(distributed,lambda:atomic_json(c.output/'evaluation_baseline.json',baseline))
        if cargs.evaluate_only:
            report.update(status='passed',scope='fixed_real_physics_evaluation_without_training',final_state=c.state)
            raise _FiniteDiagnosticComplete
        started=time.perf_counter();last_save=started
        c.state.setdefault('next_evaluation_control_steps',_next_interaction_boundary(
            c.state['training_control_steps'],c.stage['evaluation']['every_control_steps']))
        c.state.setdefault('next_diagnostic_control_steps',_next_interaction_boundary(
            c.state['training_control_steps'],c.stage['storage']['diagnostic_every_control_steps']))
        warmups=(0 if complete_mc.get('enabled') else critic_config['warmup_rollouts']) if cargs.warmup_rollouts is None else cargs.warmup_rollouts
        while c.state['iteration']<stop_after_iteration:
            if any(distributed.all_gather_object(signal.stop_requested)):break
            is_warmup=c.state['warmup_completed']<warmups
            label=('warmup' if is_warmup else 'train')+f'_{c.state["rollouts_completed"]+1:06d}'
            periodic_diagnostic=c.state['training_control_steps']>=c.state['next_diagnostic_control_steps']
            strict=cargs.strict_evidence
            if strict or periodic_diagnostic:c.state['next_diagnostic_control_steps']=_next_interaction_boundary(
                c.state['training_control_steps'],c.stage['storage']['diagnostic_every_control_steps'])
            round_start=time.perf_counter()
            torch.cuda.reset_peak_memory_stats(distributed.device)
            anomaly_saved=None;rejected=False
            from .performance import PhaseProfiler,activate,deactivate
            frequency=c.stage['performance']['profiler_every']
            detailed=strict or c.state['rollouts_completed']==0 or (not is_warmup and c.state['iteration']==0) or (
                frequency>0 and c.state['rollouts_completed']%frequency==0)
            profiler=PhaseProfiler(distributed.device,distributed.rank,detailed=detailed)
            profile_token=activate(profiler)
            buffer,targets,manifest,collection=c.vector.collect(label,strict=strict)
            rows=buffer.transitions
            if periodic_diagnostic and not strict:
                from .diagnostic_rollout import sampled_diagnostic
                collection['sampled_diagnostic']=sampled_diagnostic(c,rows,targets,label)
            used=distributed.sum_tensor(torch.tensor([collection['actual_confirmed_control_steps'],collection['actual_world_allocated_physics_steps'],len(rows)],
                device=distributed.device,dtype=torch.int64)).cpu().tolist()
            # 实际交互已发生，KL拒绝只恢复学习状态，不能让交互计数回退。
            c.state['training_control_steps']+=used[0]
            c.state['training_physics_steps']+=used[1]
            c.state['training_upper_transitions']+=used[2]
            if not manifest:
                buffer.clear();buffer=None
                c.state['rollouts_completed']+=1
                deactivate(profile_token);profile_token=None
                empty_collections=distributed.all_gather_object(collection)
                root_call(distributed,lambda:c.manager.append_metrics(dict(label=label,mode='empty_world_window',
                    collection_by_rank=empty_collections,state=copy.deepcopy(c.state),
                    actor_steps=0,critic_steps=0,policy_version_unchanged=True)))
                continue
            cache=RolloutTensorCache(rows,targets,distributed.device,max_device_bytes=c.stage['storage']['tensor_cache_max_bytes'],preserve_device=True)
            probability_frequency=c.stage['checks']['full_probability_every']
            full_probability=(c.state['rollouts_completed']==0 or strict or
                (probability_frequency>0 and c.state['rollouts_completed']%probability_frequency==0))
            initial_probability=probability_check_local(c.policy,rows,global_manifest=manifest,distributed=distributed,
                denoising_microbatch=c.settings['kl_microbatch'],tensor_cache=cache,include_frozen=full_probability,
                global_indices=None if full_probability else
                    [next(i for i,m in enumerate(manifest) if m['owner_rank']==rank and m['has_free'])
                     for rank in range(8) if any(m['owner_rank']==rank and m['has_free'] for m in manifest)])
            if (cargs.collect_only or not is_warmup and cargs.capture_validation_iteration is not None
                    and c.state['iteration']+1==cargs.capture_validation_iteration):
                saved=local_call(distributed,lambda:_diagnostic_learning_snapshot(c))
                from .diagnostic_rollout import persist_rollout
                case_index=c.state['rollouts_completed']+1 if cargs.collect_only else cargs.capture_validation_iteration
                case=c.session/'fixed_data_validation'/f'iteration{case_index:06d}'
                local_call(distributed,lambda:persist_rollout(case/f'rank{distributed.rank:02d}',rows,targets,manifest,
                    kernel=c.policy.kernel_config,identity=c.identity,full_execution=False,guard=c.manager.disk_guard if distributed.rank==0 else c.guard))
                local_call(distributed,lambda:torch.save(dict(bc=saved['bc'],rng=saved['rng']),
                    case/f'rank{distributed.rank:02d}'/'rank_state.pt'))
                root_call(distributed,lambda:torch.save(dict(actor=saved['actor'],actor_optimizer=saved['actor_optimizer'],
                    controller=saved['controller'],config=c.base_config,kernel=c.policy.kernel_config,identity=c.identity),case/'learning_state.pt'))
                rank_files=distributed.all_gather_object(dict(rank=distributed.rank,
                    rollout_manifest_sha256=file_sha256(case/f'rank{distributed.rank:02d}'/'manifest.json'),
                    rank_state_sha256=file_sha256(case/f'rank{distributed.rank:02d}'/'rank_state.pt')))
                root_call(distributed,lambda:atomic_json(case/'case.json',dict(schema='genmo.stage2.asymmetric_fixed_learning_case.v1',
                    real_global_chains=len(manifest),learning_state_sha256=file_sha256(case/'learning_state.pt'),
                    nonempty_adam=bool(saved['actor_optimizer']['state']),ranks=rank_files,
                    evidence_scope='real_PhysX_training_inputs_not_full_physical_archive',
                    joint_limits={key:(list(value) if key=='joint_names' else value.tolist())
                        for key,value in service.world.snapshots([0])[0].items() if key in
                        ('joint_names','joint_hard_limits_rad','joint_soft_limits_rad')})))
                if cargs.collect_only:
                    c.state['rollouts_completed']+=1
                    item=dict(fixed_data_case=str(case),collection_by_rank=distributed.all_gather_object(collection),
                        full_zero_update_probability=initial_probability,world_window_index=case_index)
                    report.setdefault('fixed_data_cases',[]).append(item)
                    cache.close();cache=None;buffer.clear();buffer=None;saved=None;del rows
                    deactivate(profile_token);profile_token=None
                    if c.state['rollouts_completed']>=getattr(cargs,'collect_rollouts',1):
                        report.update(status='passed',scope='full_real_rollout_collection_not_training_acceptance',
                            **item,final_state=c.state)
                        raise _FiniteDiagnosticComplete
                    continue
            update_start=time.perf_counter()
            physical_ticks=tuple(service.world.control_ticks)
            physics_time=service.world.env.sim.current_time
            critic=critic_update(c.critic,c.critic_optimizer,rows,targets,cache,manifest,distributed=distributed,
                generator=c.generators['critic'],epochs=critic_config['epochs'],batch=critic_config['global_batch'])
            critic_seconds=max(distributed.all_gather_object(time.perf_counter()-update_start))
            c.state['critic_updates']+=critic['optimizer_steps']
            actor=None;kl=None
            if not is_warmup:
                actor_start=time.perf_counter()
                c.actor_optimizer.param_groups[0]['lr']=c.actor_lr.base_lr
                def reserve():
                    root_call(distributed,lambda:c.budget.reserve('actor_attempts',optimizer_attempts=1))
                    c.state['optimizer_attempts']+=1
                if c.settings['actor_objective']=='joint_score_pg':
                    snapshot=service.world.snapshots([0])[0]
                    c.policy.experimental_joint_limits={k:snapshot[k] for k in
                        ('joint_names','joint_hard_limits_rad','joint_soft_limits_rad')}
                    actor=full_batch_actor_update(c.policy,c.actor_optimizer,rows,targets,cache,manifest,c.controller,
                        distributed=distributed,anchor=c.bc,generator=c.generators['actor'],
                        backtracking=c.settings['optimizer_backtracking'],kl_limits=c.stage['kl'],
                        kl_microbatch=c.settings['kl_microbatch'],proposed_lr=c.actor_lr.base_lr,minimum_lr=c.settings['actor_lr_adaptation']['minimum'],
                        detailed_diagnostics=(c.state['iteration']==0 or
                            (c.state['iteration']+1)%c.stage['checks'].get('actor_diagnostics_every',100)==0),
                        ppo_epochs=c.settings['ppo_epochs'],
                        minibatch_chains=c.stage['scale']['actor_minibatch_chains'],microbatch=c.settings['denoising_microbatch'],
                        clip=c.settings['ppo_clip'],
                        grad_clip_norm=c.settings['grad_clip_norm'],reserve_attempt=reserve,bc_samples_per_rollout=c.settings['bc_samples_per_rollout'],
                        snr=c.settings.get('gradient_snr_diagnostic',False))
                else:
                    actor=actor_update(c.policy,c.actor_optimizer,rows,targets,cache,manifest,c.controller,
                        distributed=distributed,anchor=c.bc,generator=c.generators['actor'],
                        ppo_epochs=c.settings['ppo_epochs'],minibatch_chains=c.stage['scale']['actor_minibatch_chains'],
                        microbatch=c.settings['denoising_microbatch'],sampled_steps=c.settings['ppo_steps_per_chain'],
                        clip=c.settings['ppo_clip'],gamma_denoising=c.settings['gamma_denoising'],
                        backtracking=c.settings['optimizer_backtracking'],kl_limits=c.stage['kl'],
                        kl_microbatch=c.settings['kl_microbatch'],minimum_lr=c.settings['actor_lr_adaptation']['minimum'],
                        kl_reduction=c.stage['kl']['reduction'],
                        objective_reduction=c.settings['objective_logprob_reduction'],
                        grad_clip_norm=c.settings['grad_clip_norm'],reserve_attempt=reserve,
                        bc_samples_per_rollout=c.settings['bc_samples_per_rollout'],
                        snr=c.settings.get('gradient_snr_diagnostic',False))
                actor_seconds=max(distributed.all_gather_object(time.perf_counter()-actor_start))
                from .kl_reuse import reusable_accepted_kl
                kl=reusable_accepted_kl(actor,c.policy,cache,manifest,distributed,
                    clip=c.settings['ppo_clip'],reduction=c.settings['objective_logprob_reduction'])
                if kl is None:
                    kl=final_kl(c.policy,rows,cache,manifest,distributed=distributed,microbatch=c.settings['kl_microbatch'],
                        clip=c.settings['ppo_clip'],objective_reduction=c.settings['objective_logprob_reduction'])
                elif getattr(cargs,'verify_kl_reuse',False) and not getattr(c,'kl_reuse_verified',False):
                    started=time.perf_counter()
                    independent=final_kl(c.policy,rows,cache,manifest,distributed=distributed,microbatch=c.settings['kl_microbatch'],
                        clip=c.settings['ppo_clip'],objective_reduction=c.settings['objective_logprob_reduction'])
                    fields=('mean_coordinate_kl','maximum_step_mean_coordinate_kl','coordinate_kl','chain_joint_kl',
                        'per_denoising_step','covered_chains','covered_internal_transitions','gaussian_kl_steps')
                    if any(kl[key]!=independent[key] for key in fields):
                        raise RuntimeError('Reused KL differs from independent complete four-step recalculation')
                    kl['reuse_verification']=dict(passed=True,all_valid_chains_and_four_steps=True,
                        additional_wall_seconds=time.perf_counter()-started)
                    c.kl_reuse_verified=True
                from .actor_backtracking import check_candidate_kl
                accepted=check_candidate_kl(kl,c.stage['kl'])
                decisions=distributed.all_gather_object(accepted['passed'])
                if any(value!=decisions[0] for value in decisions):
                    raise RuntimeError('Ranks disagree on final KL acceptance')
                if not accepted['passed']:
                    raise RuntimeError('Final policy differs from optimizer-step complete KL acceptance')
                # 所有候选仅算高斯KL；最终接受模型的参考生成诊断只执行一次。
                if kl.get('reference_change') is None:
                    from .updater import reference_change_diagnostics
                    started_reference=time.perf_counter()
                    kl['reference_change']=reference_change_diagnostics(c.policy,rows,cache,distributed)
                    kl['reference_diagnostic_seconds']=max(distributed.all_gather_object(time.perf_counter()-started_reference))
                actor['lr_control']=c.actor_lr.update(kl,c.stage['kl'],accepted=actor['optimizer_steps'],
                    planned=actor['planned_optimizer_steps'],attempts=actor['attempted_optimizer_steps'])
                lr_states=distributed.all_gather_object(c.actor_lr.state_dict())
                if any(v!=lr_states[0] for v in lr_states):raise RuntimeError('Ranks disagree on adaptive LR state')
                c.state['selected_actor_lr']=c.actor_lr.base_lr
                rejected=actor['optimizer_steps']==0
                if rejected:
                    c.state['zero_actor_update_rollouts']=c.state.get('zero_actor_update_rollouts',0)+1
                    root_call(distributed,lambda:atomic_json(c.session/(label+'_no_actor_update.json'),
                        dict(actor=actor,kl=kl,critic_retained=True,action='resample_and_continue')))
            else:
                actor_seconds=0.
            def assert_paused_physics():
                if tuple(service.world.control_ticks)!=physical_ticks or service.world.env.sim.current_time!=physics_time:
                    raise RuntimeError('Training optimization advanced the PhysX world')
            local_call(distributed,assert_paused_physics)
            fingerprint_frequency=c.stage['checks']['full_fingerprint_every']
            fingerprints=None
            if strict or (fingerprint_frequency>0 and (c.state['rollouts_completed']+1)%fingerprint_frequency==0):
                fingerprints=distributed.all_gather_object(local_call(distributed,lambda:dict(
                    actor=_model_fingerprint(c.actor),critic=_model_fingerprint(c.critic))))
                if any(value!=fingerprints[0] for value in fingerprints):
                    raise RuntimeError('Periodic complete Actor/Critic fingerprints differ across ranks')
            memory=distributed.all_gather_object(dict(pytorch_peak_allocated=torch.cuda.max_memory_allocated(distributed.device),
                pytorch_peak_reserved=torch.cuda.max_memory_reserved(distributed.device),
                scope='current_round_PyTorch_allocator_excludes_PhysX_external_allocations'))
            # 全部验收完成后再推进接受轮次，失败可回滚学习状态；已用物理/优化尝试不退款。
            if rejected:
                pass
            elif not is_warmup:
                root_call(distributed,lambda:c.budget.accept_iteration('update',identity=f'{identity["session_id"]}/{label}'))
                c.state['actor_updates']+=actor['optimizer_steps']
                c.state['policy_version']+=1;c.state['iteration']+=1
                c.state['last_accepted_actor_lr']=actor['steps'][-1]['learning_rate']
            else:
                c.state['warmup_completed']+=1
            c.state['rollouts_completed']+=1
            c.state['budget']=root_call(distributed,lambda:c.budget.summary())
            cache_report=cache.report();cache.close();cache=None;buffer.clear();buffer=None;saved=None;del rows
            wall=max(distributed.all_gather_object(time.perf_counter()-round_start))
            profile_report=profiler.report();deactivate(profile_token);profile_token=None
            record=dict(label=label,state=copy.deepcopy(c.state),collection_by_rank=distributed.all_gather_object(collection),
                probability=initial_probability,critic=critic,actor=actor,trainable4_kl=kl,tensor_cache=cache_report,
                actor_seconds=actor_seconds,critic_seconds=critic_seconds,wall_seconds=wall,
                performance_by_rank=distributed.all_gather_object(profile_report),physics_paused_during_optimization=True,
                memory_by_rank=memory,periodic_model_fingerprints=fingerprints,
                effective_upper_transitions_per_second=used[2]/wall,
                mode='critic_warmup' if is_warmup else 'no_actor_update' if rejected else 'dppo_bc',
                timing_scope='actor_seconds_includes_all_step_KL_and_retries;final_KL_receipt_is_not_extra_time',
                overhead_since_previous_record=dict(c.overhead))
            root_call(distributed,lambda:(c.manager.append_metrics(record),log_metrics(c.writer,c.output/'curves.jsonl',c.state['iteration'],
                record,prefix='training',durable=False,dashboard_metrics=training_scalars(record))))
            c.overhead.clear()
            accepted_steps=0 if actor is None or rejected else actor['optimizer_steps']
            attempted_steps=0 if actor is None else actor['attempted_optimizer_steps']
            report['warmup' if is_warmup else 'iterations'].append(dict(label=label,mode=record['mode'],
                wall_seconds=wall,actor_steps=accepted_steps,attempted_actor_steps=attempted_steps))
            if distributed.rank==0:
                status=('NO_ACTOR_UPDATE' if rejected else 'PARTIAL_ACCEPTED' if actor is not None and
                    actor['optimizer_steps']<actor['planned_optimizer_steps'] else 'ACCEPTED')
                print(f'[STAGE10 {status}] {label} accepted_iteration={c.state["iteration"]} chains={used[2]} '
                    f'actor_steps={accepted_steps} attempted_actor_steps={attempted_steps} wall={wall:.3f}s',flush=True)
            if is_warmup and c.state['warmup_completed']==warmups:
                _snapshot(c,'critic_warmup_complete');last_save=time.perf_counter()
            last_save=_maintain_after_iteration(c,report,is_warmup=is_warmup,rejected=rejected,last_save=last_save,
                finishing=not is_warmup and not rejected and c.state['iteration']>=stop_after_iteration)
            root_call(distributed,lambda:_poll_checkpoint(c))
        stopped=any(distributed.all_gather_object(signal.stop_requested))
        _finish_controlled_run(c,report,stopped=stopped)
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
        if getattr(c,'evaluation_world',None) is not None:
            try:
                from .vector_evaluation import close_isolated_world
                local_call(distributed,lambda:close_isolated_world(c.evaluation_world))
                report['evaluation_world_shutdown']=dict(clean=True,wall_seconds=c.evaluation_world.close_seconds)
            except BaseException as caught:
                report['evaluation_world_shutdown']=dict(clean=False,error=repr(caught));error=error or caught
        if c.vector is not None:
            try:report['world_shutdown']=local_call(distributed,c.vector.close)
            except BaseException as caught:
                report['world_shutdown_error']=repr(caught);error=error or caught
        if distributed.rank==0:
            try:
                if c.async_writer is not None:
                    report['checkpoint_shutdown']=_poll_checkpoint(c,wait=True)
                    c.async_writer.close()
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
