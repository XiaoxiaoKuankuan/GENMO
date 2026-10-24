"""Stage10 完整一致快照、单在途异步保存与严格新格式恢复。

调用方只在完成rollout验收后的边界冻结Actor、Critic、两个Adam、BC控制器、
调度状态、各rank RNG和采样游标为独立CPU快照。后台线程只处理这些不可变CPU
对象，不访问活动GPU模型、不修改训练RNG。最多一个在途保存，队列满显式等待
并报告背压；发布之前校验大小、SHA、fsync和原子文件，latest只指向完整文件。

当前 checkpoint具有独立schema与训练身份，旧Stage9/Stage10格式禁止透明恢复。
恢复重建PhysX并重新采样，不携带当前rollout或中途仿真状态。模型/优化器状态
在正式写入前固定，历史checkpoint不覆盖。best/重要评估点保存实际完整权重。
保存异常在下一个安全边界传播并受控停止，不能在后台失败后继续宣称已持久化。
显式Actor LR迁移仅适用于新实验；在全部验证成功后覆盖Actor各组lr，其他Adam
状态、两网络和BC保持加载结果。它不放宽普通resume合同，不重新预热Critic。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import time
import uuid
import torch

from .budget import atomic_json
from .execution_compatibility import same_execution, contains_execution

SCHEMA='genmo.closedloop.stage2.denoising_step.checkpoint.v1'
LEGACY_JOINT_SCHEMA='genmo.closedloop.stage2.full_chain_joint.checkpoint.v1'


def immutable_cpu_snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):return {k:immutable_cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [immutable_cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(immutable_cpu_snapshot(v) for v in value)
    import copy
    return copy.deepcopy(value)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):digest.update(b)
    return digest.hexdigest()


class AsyncCheckpointWriter:
    def __init__(self,root,*,keep_last=2):
        self.root=Path(root);self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='checkpoint')
        self.pending=None;self.telemetry=[];self.closed=False
        if type(keep_last)is not int or keep_last<1:raise ValueError('Positive complete periodic retention required')
        self.keep_last=keep_last
        (self.root/'checkpoints').mkdir(parents=True,exist_ok=True)

    def poll(self,*,wait=False):
        if self.pending is None:return None
        if not wait and not self.pending.done():return None
        start=time.perf_counter();result=self.pending.result();self.pending=None
        result['foreground_wait_seconds']=time.perf_counter()-start
        self.telemetry.append(result)
        return result

    def submit(self,payload,*,reason):
        if self.closed or payload.get('schema')!=SCHEMA:raise ValueError('Invalid async checkpoint state')
        wait=self.poll(wait=True)
        if any(v.is_cuda for v in payload['actor'].values()):raise ValueError('Async writer accepts only immutable CPU model snapshots')
        self.pending=self.pool.submit(self._write,payload,reason)
        return dict(previous_backpressure=wait,queued=True,maximum_inflight=1)

    def _write(self,payload,reason):
        start=time.perf_counter();state=payload['state']
        # 恢复后可以在同一完成边界再次受控退出；每次发布独立身份，绝不覆盖旧权重。
        publication_id=f'{time.time_ns():020d}_{uuid.uuid4().hex[:12]}'
        name=f'stage10_r{state["rollouts_completed"]:06d}_a{state["actor_updates"]:06d}_{publication_id}_{reason}.pt'
        path=self.root/'checkpoints'/name;tmp=path.with_suffix('.pt.partial')
        if path.exists() or tmp.exists():raise FileExistsError(path)
        try:
            with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            sha=file_sha256(tmp);size=tmp.stat().st_size
            os.link(tmp,path);tmp.unlink()
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            descriptor=dict(schema='genmo.stage10.checkpoint_publication',iteration=state['iteration'],
                rollouts_completed=state['rollouts_completed'],path=str(path.relative_to(self.root)),
                sha256=sha,size_bytes=size,publication_id=publication_id,metadata=dict(reason=reason,actor_updates=state['actor_updates'],
                critic_updates=state['critic_updates'],actor_model_fingerprint=payload['actor_fingerprint'],
                reasons=reason.split('+')))
            atomic_json(self.root/'latest.json',descriptor)
            if reason=='best':atomic_json(self.root/'best_checkpoint.json',descriptor)
            # 只清理当前运行产生的普通周期文件；初始、评估、best和终态不在此集合。
            candidates=sorted(p for p in (self.root/'checkpoints').glob('stage10_r*_a*_periodic.pt')
                              if '+' not in p.stem)
            import json
            protected={path}
            for pointer_name in ('latest.json','best_checkpoint.json','best_saved.json'):
                pointer_path=self.root/pointer_name
                if pointer_path.exists():
                    pointer=json.loads(pointer_path.read_text())
                    reference=pointer.get('checkpoint',pointer)
                    if reference.get('path'):protected.add(self.root/reference['path'])
            for expired in candidates[:-self.keep_last]:
                if expired not in protected:expired.unlink()
            return dict(**descriptor,background_write_seconds=time.perf_counter()-start)
        except BaseException:
            tmp.unlink(missing_ok=True);raise

    def close(self):
        try:return self.poll(wait=True)
        finally:self.pool.shutdown(wait=True);self.closed=True


def _initialization_execution(identity, *, fragment_transfer=False, actor_lr_transfer=False):
    """显式新实验可以继承学习状态；片段实验另外重新校准折扣价值、重建采样池。

    fragment_transfer只允许修改物理采样时限、站立混合比例、gamma和旧预热文件。
    actor_lr_transfer仅额外放行Actor基础LR及其最大试探LR；不放行组倍率、
    其他trust_region字段、Critic、奖励或采样核，不能与片段迁移合用。
    actor_initialization仅记录来源，新实验加载后会绑定所选checkpoint；普通resume
    不经过此规范化，仍严格校验该记录。模型、Adam与实际执行合同另行逐项校验。
    """
    import copy
    result=copy.deepcopy(identity)
    result.pop('actor_initialization',None)
    result.pop('evaluation',None)
    result.get('scale',{}).pop('actor_minibatch_chains',None)
    for key in ('actor_minibatch_internal_transitions','max_actor_optimizer_steps','bc_batch'):
        result.get('training',{}).pop(key,None)
    # 此摘要由批量导出；原始核、KL、奖励、BC及未显式迁移的学习率仍直接比较。
    result.get('kl',{}).get('calibration',{}).pop('optimization_sha256',None)
    if actor_lr_transfer:
        result.get('training',{}).pop('actor_lr',None)
        result.get('training',{}).get('trust_region',{}).pop('maximum_lr',None)
    if fragment_transfer:
        result.get('training',{}).pop('gamma_upper',None)
        result.get('dataset',{}).pop('standing_start_probability',None)
        result.get('critic',{}).pop('preheat_artifact',None)
        for key in ('safety_limit_seconds','termination','time_limit_bootstrap'):
            result.get('physical_episode',{}).pop(key,None)
    return result


def _actor_lr_transfer_audit(previous, current, actor_optimizer, saved_optimizer):
    """加载前核对两个显式LR合同，并记录只更换组LR的迁移来源。

    配置中的actor_lr必须等于maximum_lr；旧Adam可能保存一次回溯后的实际LR，
    因此单独记录旧配置基础LR、旧优化器基础LR和各组实际LR。参数组身份/映射
    已由通用checkpoint校验核对，此处仍核对固定倍率与正有限LR，不修改任何状态。
    """
    import math
    from types import SimpleNamespace
    from .actor_optimizer import actor_base_lr, actor_learning_rates
    values=[]
    for label, identity in (('previous',previous),('current',current)):
        training=identity.get('training',{})
        base=training.get('actor_lr')
        maximum=training.get('trust_region',{}).get('maximum_lr')
        if any(type(value) not in (int,float) or not math.isfinite(value) or value<=0
               for value in (base,maximum)) or base!=maximum:
            raise ValueError(label+' Actor LR transfer requires actor_lr == maximum_lr, finite and positive')
        values.append(base)
    saved=SimpleNamespace(param_groups=saved_optimizer['param_groups'])
    old_optimizer_base=actor_base_lr(saved)
    if not math.isclose(actor_base_lr(actor_optimizer),values[1],rel_tol=1e-12,abs_tol=0.):
        raise ValueError('Target Actor optimizer base LR differs from the target configuration')
    return dict(transfer='actor_learning_rate_only.v1',
        previous_actor_base_lr=values[0],actor_base_lr=values[1],
        previous_optimizer_base_lr=old_optimizer_base,
        previous_actor_learning_rates=actor_learning_rates(saved),
        actor_learning_rates={group.get('name','uniform'):values[1]*group.get('lr_multiplier',1.)
                              for group in saved.param_groups},
        actor_parameters_preserved=True,actor_adam_moments_and_steps_preserved=True,
        actor_optimizer_changed_fields=['param_groups.lr'],
        actor_optimizer_state_except_learning_rates_preserved=True)


def _legacy_fragment_execution(previous, current):
    """显式迁移本分支旧坐标代理实验，保留真实物理核及全部模型/Adam。

    仅供新目录的initialize-random-fragments使用：本次同时切换完整链PPO、8192
    批量、20秒采样与gamma=.95，旧GAE/预热结论和学习率控制器不得当成等价恢复。
    实际权重、奖励、GMT、观测、时钟和数据身份必须相同；参数/矩映射另行逐项校验。
    """
    keys=('assets','kernel','reward','physical_configuration','timing','data_catalog',
          'source_stage1_step','generation_horizon')
    if any(not same_execution(previous.get(k),current.get(k)) for k in keys):
        return False
    architecture=('type','dimension','layers','heads','history_scales','learning_rate','history_learning_rate')
    if any(previous.get('critic',{}).get(k)!=current.get('critic',{}).get(k) for k in architecture):
        return False
    old_dataset=dict(previous.get('dataset',{}));new_dataset=dict(current.get('dataset',{}))
    old_dataset.pop('standing_start_probability',None);new_dataset.pop('standing_start_probability',None)
    if old_dataset!=new_dataset:
        return False
    for key in ('num_envs','generation_batch','physics_device','genmo_device','generation_schedule'):
        if previous.get('runtime',{}).get(key)!=current.get('runtime',{}).get(key):
            return False
    return (previous.get('training',{}).get('bc_enabled') is False
            and current.get('training',{}).get('bc_enabled') is False
            and current.get('training',{}).get('actor_objective')=='full_chain_clipped_ppo'
            and current.get('training',{}).get('actor_trainable_scope','all')=='all')


def _step_dppo_initialization(identity):
    """仅显式新算法迁移移除获准变化；物理、噪声、奖励、模型和GAE仍逐项严格比较。"""
    result=_initialization_execution(identity)
    result.pop('actor_initialization',None)
    result.pop('kl',None)
    training=result.get('training',{})
    for key in ('actor_objective','objective_step_reduction','actor_lr','actor_lr_control',
                'ppo_clip','kl_soft_stop_chain_joint','kl_soft_stop_denoising_step',
                'max_active_clip_fraction','trust_region'):
        training.pop(key,None)
    for key in ('sampling_contract','adaptive'):
        result.get('dataset',{}).pop(key,None)
    return result


def load_training_checkpoint(path,*,identity,actor,critic,actor_optimizer,critic_optimizer,controller,
                             new_experiment=False,fragment_transfer=False,actor_lr_transfer=False,
                             step_dppo_transfer=False):
    if fragment_transfer and not new_experiment:
        raise ValueError('Fragment transfer requires an explicit new experiment, never ordinary resume')
    if actor_lr_transfer and (not new_experiment or fragment_transfer):
        raise ValueError('Actor LR transfer requires its own explicit new experiment, never resume or fragment transfer')
    if step_dppo_transfer and (not new_experiment or fragment_transfer or actor_lr_transfer):
        raise ValueError('Step DPPO transfer requires its own explicit new experiment, never ordinary resume')
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    # 固定评估是只读监视配置；更换清单不改变已学参数、Adam、奖励或采样策略。
    # 配对评估仍严格比较自己的完整任务与执行配置，并单独重建Stage1基线。
    previous=payload.get('identity') or {}
    old_training={k:v for k,v in previous.items() if k!='evaluation'}
    new_training={k:v for k,v in identity.items() if k!='evaluation'}
    if new_experiment:
        old_training=_initialization_execution(old_training,fragment_transfer=fragment_transfer,
            actor_lr_transfer=actor_lr_transfer)
        new_training=_initialization_execution(new_training,fragment_transfer=fragment_transfer,
            actor_lr_transfer=actor_lr_transfer)
    legacy_fragment=(new_experiment and fragment_transfer
        and payload.get('schema')=='genmo.closedloop.stage2.asymmetric_tracking.checkpoint.v3'
        and _legacy_fragment_execution(previous,identity))
    if step_dppo_transfer:
        if (payload.get('schema')!=LEGACY_JOINT_SCHEMA
                or previous.get('training',{}).get('actor_objective')!='full_chain_clipped_ppo'
                or identity.get('training',{}).get('actor_objective')!='denoising_step_clipped_ppo'
                or identity.get('training',{}).get('objective_step_reduction')!='mean'
                or not same_execution(_step_dppo_initialization(previous),_step_dppo_initialization(identity))):
            raise ValueError('Step DPPO transfer changed an unapproved physical, kernel, reward or learning contract')
    elif not legacy_fragment and (payload.get('schema')!=SCHEMA or not same_execution(old_training,new_training)):
        raise ValueError('Checkpoint schema or actual training configuration differs')
    from .checkpoint import _validate_model_state,_validate_optimizer_state
    _validate_model_state(actor,payload['actor'],'当前 Actor')
    _validate_model_state(critic,payload['critic'],'当前 Critic')
    _validate_optimizer_state(actor_optimizer,payload['actor_optimizer'])
    _validate_optimizer_state(critic_optimizer,payload['critic_optimizer'])
    migration=None
    if actor_lr_transfer:
        from .actor_optimizer import validate_actor_optimizer, set_actor_base_lr
        validate_actor_optimizer(actor,actor_optimizer)
        migration=_actor_lr_transfer_audit(previous,identity,actor_optimizer,payload['actor_optimizer'])
    if step_dppo_transfer:
        # 在修改任何模型前校验目标LR和参数组；新目标不沿用旧联合裁剪Adam矩。
        from .actor_optimizer import validate_actor_optimizer, actor_base_lr
        validate_actor_optimizer(actor,actor_optimizer)
        if not math_is_valid_step_lr(identity,actor_base_lr(actor_optimizer)):
            raise ValueError('Step DPPO target optimizer LR differs from its configuration')
    # 先验证控制器，再修改任何模型；拒绝损坏快照时保持调用方当前学习状态。
    import copy
    copy.deepcopy(controller).load_state_dict(payload['bc_controller'])
    actor.load_state_dict(payload['actor'],strict=True);critic.load_state_dict(payload['critic'],strict=True)
    if step_dppo_transfer:
        actor_optimizer.state.clear()
        payload['step_dppo_transfer']=dict(transfer='joint_to_denoising_step_mean.v1',
            previous_objective='full_chain_clipped_ppo',objective='denoising_step_clipped_ppo',
            actor_parameters_preserved=True,actor_adam_reset=True,critic_and_adam_preserved=True,
            actor_base_lr=identity['training']['actor_lr'],
            old_rollout_reused=False,sampling_pool='fresh_adaptive_pool',
            checkpoint_schema_changed=True,ordinary_resume_compatibility=False)
    else:
        actor_optimizer.load_state_dict(payload['actor_optimizer'])
    critic_optimizer.load_state_dict(payload['critic_optimizer'])
    controller.load_state_dict(payload['bc_controller'])
    if migration is not None:
        # 所有身份/模型/优化器/控制器验证完成后，唯独覆盖Actor组LR；两Adam矩和步数保留。
        set_actor_base_lr(actor_optimizer,migration['actor_base_lr'])
        payload['actor_lr_transfer']=migration
    if previous.get('evaluation')!=identity.get('evaluation') and 'state' in payload:
        payload['state']=dict(payload['state'])
        for key in ('best_observed','best_saved','regression_monitor'):
            payload['state'].pop(key,None)
    if any(p.dtype!=torch.float32 for p in actor.parameters()):raise ValueError('Restored Actor master dtype changed')
    return payload


def math_is_valid_step_lr(identity, optimizer_lr):
    import math
    value=identity.get('training',{}).get('actor_lr')
    maximum=identity.get('training',{}).get('trust_region',{}).get('maximum_lr')
    return (type(value) in (int,float) and math.isfinite(value) and value>0
        and value==maximum and math.isclose(value,optimizer_lr,rel_tol=1e-12,abs_tol=0.))


def initialize_expanded_policy(path, *, identity, actor, actor_optimizer):
    """新实验显式从四步迁移到十步：只继承Actor和Adam，重新估计新奖励下的Value。

    先逐项核对资产、冻结Stage1、噪声、物理/观测及未授权的训练参数，再加载权重。
    旧rollout、Critic、优势、运行预算和仿真游标均不返回给恢复流程；普通resume仍严格
    要求原合同。新的学习率覆盖Adam中的旧lr，动量和步数保留用于公平的倍率对照。
    """
    import copy
    from .checkpoint import _validate_model_state, _validate_optimizer_state
    from .rewards import ROOT_REWARD_STD_VERSION, REWARD_STD_VERSION, ExecutionReward
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    old=payload.get('identity',{})
    if payload.get('schema')!=SCHEMA:
        raise ValueError('Expanded policy initialization requires a complete current checkpoint')
    for key in ('assets','physical_configuration','data_catalog','source_stage1_step',
                'generation_horizon','dataset','physical_episode','ppo_rollout','bc_gradient'):
        if not same_execution(old.get(key),identity.get(key)):
            raise ValueError('Expanded policy changes an unauthorized identity: '+key)
    for section in ('kernel','training','runtime','timing','kl'):
        before=copy.deepcopy(old[section]);after=copy.deepcopy(identity[section])
        if section=='kernel':
            b=before['frozen_prefix'];a=after['frozen_prefix']
            if b['steps']!=15 or a['steps']!=9:
                raise ValueError('Explicit expansion must be frozen15/train4 to frozen9/train10')
            for key in ('schema','steps','gaussian_kl_steps'):
                b.pop(key);a.pop(key)
        elif section=='training':
            for key in ('actor_lr','frozen_prefix_steps','ppo_step_strategy','ppo_steps_per_chain',
                        'actor_minibatch_internal_transitions'):
                before.pop(key,None);after.pop(key,None)
            before['trust_region'].pop('maximum_lr');after['trust_region'].pop('maximum_lr')
        elif section=='runtime':
            # 单机器人测时只改变标定运行的world规模，不改变后续训练物理配置。
            for key in ('num_envs','generation_batch','phase_contract','timing_contract'):
                before.pop(key,None);after.pop(key,None)
        elif section=='timing':
            before.pop('deployment_profile');after.pop('deployment_profile')
        elif section=='kl':
            before.pop('calibration');after.pop('calibration')
        if not same_execution(before,after):
            raise ValueError('Expanded policy changes an unauthorized contract: '+section)
    if old['reward']['version']!=REWARD_STD_VERSION or identity['reward']['version']!=ROOT_REWARD_STD_VERSION:
        raise ValueError('Explicit root reward v3 to v4 migration required')
    b=copy.deepcopy(ExecutionReward(old['reward']).config);a=copy.deepcopy(ExecutionReward(identity['reward']).config)
    b['version']=a['version'];b['tracking']['reward_std'].update(anchor_pos=.3,anchor_ori=.4)
    if b!=a:raise ValueError('Only root position/orientation reward std may change')
    _validate_model_state(actor,payload['actor'],'十步新实验 Actor')
    _validate_optimizer_state(actor_optimizer,payload['actor_optimizer'])
    desired_lrs=[group['lr'] for group in actor_optimizer.param_groups]
    actor.load_state_dict(payload['actor'],strict=True)
    actor_optimizer.load_state_dict(payload['actor_optimizer'])
    for group,lr in zip(actor_optimizer.param_groups,desired_lrs):group['lr']=lr
    return dict(checkpoint=str(Path(path).resolve()),sha256=file_sha256(path),
        actor_fingerprint=payload['actor_fingerprint'],source_iteration=payload['state']['iteration'],
        source_actor_updates=payload['state']['actor_updates'],
        transfer='expanded_denoising_root_reward.v1',actor_and_adam_preserved=True,
        critic_and_adam_preserved=False,original_budget_preserved=False,old_rollout_reused=False,
        fresh_physics_rollout_required=True,critic_recalibration_required=True,
        previous_kernel=old['kernel'],kernel=identity['kernel'],
        previous_reward=old['reward'],reward=identity['reward'],actor_learning_rates=desired_lrs)


def probability_candidate_binding(source,target,*,allow_measured_profile_transfer=False):
    """只允许显式更换实测延迟样本池/来源，不改变时钟、前缀预算、奖励或物理合同。

    旧rollout身份和目标评估身份分别保留；这种Actor只可做新鲜物理评估，绝不可作为
    相同训练身份恢复。实际50/200Hz、重规划周期与部署核设置仍逐项相等。
    """
    for key in ('assets','kernel','reward','physical_configuration','data_catalog','generation_horizon'):
        if source.get(key)!=target.get(key):raise ValueError('Candidate evaluation behavior contract differs: '+key)
    old=dict(source['timing']);new=dict(target['timing'])
    old_profile=old.pop('deployment_profile');new_profile=new.pop('deployment_profile')
    if old!=new:raise ValueError('Candidate evaluation physical/replanning clock differs')
    transferred=old_profile!=new_profile
    if transferred:
        if not allow_measured_profile_transfer:raise ValueError('Explicit measured delay profile transfer required')
        for key in set(old_profile)|set(new_profile):
            if key not in ('samples_seconds','provenance') and old_profile.get(key)!=new_profile.get(key):
                raise ValueError('Measured delay transfer changes kernel or prefix budget: '+key)
    import hashlib,json
    sha=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(measured_delay_profile_transfer=transferred,source_profile_sha256=sha(old_profile),
        evaluation_profile_sha256=sha(new_profile),allowed_differences=['samples_seconds','provenance'],
        new_PhysX_paired_evaluation_required=True,training_resume=False)


def load_evaluation_actor(path,*,actor,kernel,assets,execution_binding=None):
    """只读指定已完成checkpoint或固定数据候选的Actor，不恢复优化器/训练进度。

    正式checkpoint核对奖励、时钟、数据和物理配置，源码版本只作记录；允许改变纯评估任务
    数/长度和训练学习率以进行同条件P1/P2对照；绝不作为继续训练的恢复入口。
    """
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('schema')=='genmo.stage2.probability_candidate_evaluation.v1':
        target=payload.get('evaluation_identity',{})
        update=payload.get('update',{})
        if (payload.get('evaluation_only') is not True or not execution_binding or
                target.get('assets')!=assets or target.get('kernel')!=kernel or
                not contains_execution(target,execution_binding) or
                payload.get('hard_kl',{}).get('passed') is not True or
                type(update.get('optimizer_steps'))is not int or update['optimizer_steps']<1 or
                update['optimizer_steps']!=update.get('planned_optimizer_steps') or
                len(payload.get('source_case_sha256',''))!=64):
            raise ValueError('Probability candidate evaluation identity or complete joint-KL update differs')
        transfer=probability_candidate_binding(payload['source_identity'],target,allow_measured_profile_transfer=True)
        if transfer!=payload.get('evaluation_transfer'):raise ValueError('Probability candidate profile transfer evidence differs')
        from .checkpoint import _validate_model_state
        _validate_model_state(actor,payload['actor'],'probability candidate evaluation Actor')
        actor.load_state_dict(payload['actor'],strict=True)
        return dict(kind='probability_candidate_fresh_evaluation_only',path=str(Path(path).resolve()),
            sha256=file_sha256(path),source_case_sha256=payload['source_case_sha256'],update=update,
            evaluation_transfer=transfer,training_resume=False)
    if payload.get('schema') in (SCHEMA,LEGACY_JOINT_SCHEMA):
        # 只读物理对照允许旧算法完整权重；不放宽任何训练resume合同。
        identity=payload.get('identity',{})
        if (not execution_binding or identity.get('assets')!=assets or identity.get('kernel')!=kernel or
                not contains_execution(identity,execution_binding)):
            raise ValueError('Complete evaluation checkpoint execution identity differs')
        from .checkpoint import _validate_model_state
        _validate_model_state(actor,payload['actor'],'complete evaluation Actor')
        # 使用原Actor指纹算法的只读临时副本，先核验再修改调用者。
        import copy
        from .evaluation import _model_fingerprint
        candidate=copy.deepcopy(actor).cpu();candidate.load_state_dict(payload['actor'],strict=True)
        if _model_fingerprint(candidate)!=payload.get('actor_fingerprint'):
            raise ValueError('Complete evaluation Actor fingerprint differs')
        actor.load_state_dict(payload['actor'],strict=True)
        return dict(kind='complete_checkpoint_evaluation_only',path=str(Path(path).resolve()),sha256=file_sha256(path),
            source_iteration=payload['state']['iteration'],source_policy_version=payload['state']['policy_version'],
            training_resume=False)
    if (payload.get('schema')!='genmo.stage2.asymmetric_actor_evaluation.v1' or
            payload.get('evaluation_only') is not True or payload.get('kernel')!=kernel or
            payload.get('source_identity',{}).get('assets')!=assets or
            not execution_binding or not contains_execution(payload.get('source_identity',{}),execution_binding) or
            payload.get('hard_kl',{}).get('passed') is not True or
            payload.get('complete_update_and_soft_kl_passed') is not True):
        raise ValueError('Evaluation Actor asset/kernel/reward/physics/complete-update differs')
    update=payload.get('update',{})
    if (type(update.get('optimizer_steps')) is not int or update['optimizer_steps']<1 or
            update['optimizer_steps']!=update.get('planned_optimizer_steps') or
            not isinstance(payload.get('source_case_sha256'),str) or len(payload['source_case_sha256'])!=64):
        raise ValueError('Evaluation Actor is not a complete identified fixed-data update')
    descriptor=dict(kind='fixed_real_data_update_evaluation_only',path=str(Path(path).resolve()),
        sha256=file_sha256(path),source_case_sha256=payload['source_case_sha256'],update=update)
    from .checkpoint import _validate_model_state
    _validate_model_state(actor,payload['actor'],'当前 evaluation Actor')
    actor.load_state_dict(payload['actor'],strict=True)
    return descriptor
