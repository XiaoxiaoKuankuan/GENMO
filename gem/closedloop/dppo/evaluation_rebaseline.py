"""更换固定音乐评估清单或物理slot数量时，保留学习状态并补建真实Stage1物理基线。

输入是已恢复完整checkpoint、Critic预热完成的训练上下文。只在用户显式配置的
training_progress评估使用新任务清单时执行：从冻结Stage1副本构造只读策略视图，
沿用原20步随机核、同一Critic、固定音乐/种子和真实GMT物理协议执行完整评估。
原可训练Actor、Adam、BC、音乐采样器、轮次及训练RNG不替换、不重置；评估交互
照常记入实际预算。新的Actor漂移参考来自Stage1，价值诊断参考来自当前Critic，
这一来源明确写入报告，不能把当前训练Actor的结果冒充Stage1基线。

新基线完成后才发布根目录指针，旧验证集基线、固定参考及最佳模型指针另存档，
原始episode与checkpoint不修改。清单变化后重新开始最佳模型比较，不能把两套
音乐的分数直接拼接；同清单、同协议的后续评估仍遵循既有严格配对检查。
"""
from __future__ import annotations

import copy
import json
import shutil

from .budget import atomic_json
from .parallel_support import root_call, capture_local_rng, restore_local_rng
from .periodic_monitor import build_balanced_plan


MONITOR_FILES=('evaluation_baseline.json','fixed_diagnostic_reference.json',
               'best_observed.json','best_saved.json','best_checkpoint.json')


def _stage1_context(c, workspace):
    """共享不可变Stage1权重及只读Critic，不拷贝或覆盖当前训练Actor的任何张量。"""
    baseline=copy.copy(c)
    baseline.output=workspace
    baseline.state=copy.deepcopy(c.state)
    baseline.state.update(iteration=0,actor_updates=0,policy_version=0)
    for key in ('best_observed','best_saved','regression_monitor'):
        baseline.state.pop(key,None)
    baseline.actor=c.policy.frozen_policy.actor
    baseline.policy=copy.copy(c.policy)
    baseline.policy.actor=baseline.actor
    baseline.policy._phase_signature=None
    baseline.policy.last_sample_timing={}
    baseline.writer=None
    if hasattr(baseline,'_periodic_fixed_reference'):
        del baseline._periodic_fixed_reference
    return baseline


def ensure_training_progress_baseline(c):
    """任务或评估物理slot变化时补建基线；首次训练沿用原初始化流程。"""
    options=c.stage['evaluation']
    if options.get('purpose')!='training_progress':
        return None
    plan=root_call(c.distributed,lambda:build_balanced_plan(c.catalog,
        samples_per_source=options['samples_per_source'],seeds=options['seeds'],
        seconds=options['episode_seconds'],selection_seed=options['selection_seed'],
        start_mode=options.get('start_mode','beginning'),unique_music=options.get('unique_music',False),
        split=options['split'],training_progress=True,original_split=options['original_split']))
    previous=root_call(c.distributed,lambda:json.loads((c.output/'evaluation_baseline.json').read_text())
        if (c.output/'evaluation_baseline.json').exists() else None)
    fragment_transfer=bool(c.state.get('requires_stage1_rebaseline',False))
    slots=options.get('envs_per_rank',1)
    old_slots=((previous or {}).get('evaluation_identity',{}).get('physics') or {}).get('evaluation_envs_per_rank',slots)
    physics_change=previous is not None and old_slots!=slots
    if not fragment_transfer and not physics_change and (previous is None or previous['plan_sha256']==plan['plan_sha256']):
        return None
    workspace=c.output/'evaluation_protocols'/plan['plan_sha256']/c.session.name
    root_call(c.distributed,lambda:workspace.mkdir(parents=True,exist_ok=False))
    baseline=_stage1_context(c,workspace)
    from .evaluation_runtime import _evaluate
    rng=capture_local_rng(c.generators)
    try:
        result=_evaluate(baseline,'initial',write_report=True)
    finally:
        restore_local_rng(rng,c.generators)
    result['baseline_provenance']=dict(actor='immutable_Stage1_frozen_copy',
        source_stage1_step=c.identity.get('source_stage1_step'),
        requested_at_training_iteration=c.state['iteration'],
        critic='current_restored_critic_read_only_for_value_diagnostics',
        training_state_reset=False)

    def publish():
        atomic_json(workspace/'evaluation_baseline.json',result)
        archive=None
        if previous is not None:
            suffix=f'_slots{old_slots}' if physics_change else ''
            archive=c.output/'evaluation_protocols'/('previous_'+previous['plan_sha256']+suffix)
            archive.mkdir(parents=True,exist_ok=True)
            for name in MONITOR_FILES:
                source=c.output/name
                if source.exists():
                    target=archive/name
                    if target.exists() and target.read_bytes()!=source.read_bytes():
                        raise ValueError('Existing archived evaluation evidence differs: '+name)
                    if not target.exists():shutil.copy2(source,target)
        reference=json.loads((workspace/'fixed_diagnostic_reference.json').read_text())
        atomic_json(c.output/'fixed_diagnostic_reference.json',reference)
        atomic_json(c.output/'evaluation_baseline.json',result)
        for name in MONITOR_FILES[2:]:
            (c.output/name).unlink(missing_ok=True)
        change=dict(previous_plan_sha256=None if previous is None else previous['plan_sha256'],plan_sha256=plan['plan_sha256'],
            split=plan['split'],original_split=plan['pool_original_split'],task_count=plan['task_count'],
            iteration=c.state['iteration'],archive=None if archive is None else str(archive),stage1_baseline=str(workspace),
            optimizer_reset=False,MC_preheat_repeated=fragment_transfer,
            previous_evaluation_slots=old_slots,evaluation_slots=slots,
            reason=('new_fragment_value_target' if fragment_transfer else
                    'evaluation_physics_slots_change' if physics_change else 'evaluation_task_change'))
        atomic_json(c.output/'evaluation_monitor_change.json',change)
        return change
    change=root_call(c.distributed,publish)
    for key in ('best_observed','best_saved','regression_monitor'):
        c.state.pop(key,None)
    if hasattr(baseline,'_periodic_fixed_reference'):
        c._periodic_fixed_reference=baseline._periodic_fixed_reference
    c.state['budget']=root_call(c.distributed,lambda:c.budget.state_dict())
    c.state.pop('requires_stage1_rebaseline',None)
    if physics_change:
        # 同一checkpoint先补做新物理布局的当前Actor评估；旧布局分数不混入新曲线。
        c.state['requires_evaluation_protocol_start']=True
    return change
