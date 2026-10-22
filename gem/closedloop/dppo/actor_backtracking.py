"""Actor固定模块倍率、逐步代理PPO目标和独立高斯KL的原子回溯事务。

调用前完成一次全批量PPO/BC合成与裁剪。本模块保留模型、AdamW、BC控制器、指定
生成器及Python/NumPy/Torch随机状态，每次从相同起点用固定梯度执行一次AdamW，
按1、1/2、1/4等有限次数尝试。生产检查累计已用完整链的全部四步；
所有rank共同接受或拒绝，冻结参数不允许进入优化器或参数快照。失败尝试的预算不退款。
此处不提高噪声、不截断概率，也不改写行为统计；全部失败恢复起点并返回拒绝证据，
数值失败交给训练器恢复整轮；全部为有限信任域拒绝时，可由上层结束本批剩余
更新，保留此前已接受状态。显式minimum_lr以下不再执行几乎无效的Adam尝试，
不会把小于下限的候选强制接受。单步通过后仍执行完整rollout最终KL审查。
梯度与BC只计算一次，候选从相同Adam时刻重试；不靠重复Adam累计动量缩小步长。
"""
from __future__ import annotations

import copy
import math
import time
import torch

from .parallel_support import capture_local_rng, restore_local_rng, local_call
from .updater import check_kl
from .actor_optimizer import validate_actor_optimizer, actor_base_lr, set_actor_base_lr, actor_learning_rates


def check_candidate_kl(report, limits):
    result = check_kl(report, limits)
    for name in ('mean', 'p95', 'p99', 'max'):
        bound = limits.get('chain_joint_'+name)
        if bound is not None:
            value = report['chain_joint_kl'][name]
            if not math.isfinite(value) or value > bound:
                result['failures']['chain_joint_'+name] = dict(value=value, limit=bound)
    bound=limits.get('maximum_active_clip')
    if bound is not None:
        value=report['active_clip_fraction']
        if not math.isfinite(value) or value>bound:
            result['failures']['active_clip_fraction']=dict(value=value,limit=bound)
    reference=limits.get('objective_reference')
    if reference is not None:
        value=report['objective_mean'];tolerance=limits.get('objective_tolerance',1e-10)
        if not math.isfinite(value) or value<reference-tolerance:
            result['failures']['objective_gain']=dict(value=value-reference,limit=-tolerance)
    result['passed'] = not result['failures']
    result['joint_gate_is_empirical_not_theoretical_safety'] = True
    result['joint_gate_enabled']=any(limits.get('chain_joint_'+name) is not None for name in ('mean','p95','p99','max'))
    result['objective_metric']=report.get('objective','external_test_objective')
    return result


def backtracked_step(actor, optimizer, controller, *, evaluate, limits, distributed, generators=None,
                     max_attempts=9, factor=.5, reserve_attempt=None, observe=None, minimum_lr=None,
                     select_best=False,search_attempts=3):
    validate_actor_optimizer(actor, optimizer)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10 or factor != .5:
        raise ValueError('Finite backtracking requires 1..10 attempts with factor 0.5')
    initial_lr = actor_base_lr(optimizer)
    if minimum_lr is not None and (not math.isfinite(minimum_lr) or not 0 < minimum_lr <= initial_lr):
        raise ValueError('Minimum trial LR must be positive and no larger than the initial trial')
    initial_learning_rates = actor_learning_rates(optimizer)
    model = {n:p.detach().clone() for n,p in actor.named_parameters() if p.requires_grad}
    adam = copy.deepcopy(optimizer.state_dict())
    control = copy.deepcopy(controller.state_dict())
    rng = capture_local_rng(generators)
    # Adam与no-grad KL前向都不修改梯度；保留所有权及版本，避免复制第三份大向量。
    grads = {n:p.grad for n,p in actor.named_parameters() if p.requires_grad}
    versions = {n:g._version for n,g in grads.items() if g is not None}
    def restore():
        with torch.no_grad():
            for n,p in actor.named_parameters():
                if n in model:p.copy_(model[n])
        optimizer.load_state_dict(copy.deepcopy(adam))
        controller.load_state_dict(control)
        restore_local_rng(rng, generators)
        for n,p in actor.named_parameters():
            if n in grads:
                if grads[n] is not None and grads[n]._version!=versions[n]:
                    raise RuntimeError('KL candidate evaluation modified the fixed gradient')
                p.grad=grads[n]
    if select_best and ('objective_reference' not in limits or type(search_attempts)is not int or search_attempts<1):
        raise ValueError('Best candidate search requires explicit objective reference and positive search count')
    trials = [];best=None
    try:
        for index in range(max_attempts):
            if index:
                restore()
            lr = initial_lr * factor**index
            if minimum_lr is not None and lr < minimum_lr:
                break
            set_actor_base_lr(optimizer, lr)
            if reserve_attempt is not None:
                reserve_attempt()
            started = time.perf_counter()
            local_call(distributed,optimizer.step)
            try:
                with torch.no_grad():
                    kl = evaluate()
                    acceptance = check_candidate_kl(kl, limits)
            except FloatingPointError as error:
                kl=None
                acceptance=dict(passed=False,failures={'numerical':str(error)})
            flags = distributed.all_gather_object(acceptance['passed'])
            if any(flag != flags[0] for flag in flags):
                raise RuntimeError('Ranks disagree on backtracking acceptance')
            diagnostics = observe() if observe is not None and kl is not None else None
            trials.append(dict(attempt=index+1, learning_rate=lr, learning_rates=actor_learning_rates(optimizer), factor=factor**index,
                adam_step_executed=True,initial_state_restored_before_attempt=index>0,
                kl=kl, acceptance=acceptance, diagnostics=diagnostics,
                wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started))))
            if acceptance['passed']:
                # 候选审查和诊断无权消耗学习RNG；下一轮从一次梯度计算后的随机状态继续。
                if select_best:
                    score=kl['objective_mean']
                    if best is None or score>best['score']:
                        best=dict(score=score,lr=lr,index=index,
                            model={n:p.detach().clone() for n,p in actor.named_parameters() if p.requires_grad},
                            adam=copy.deepcopy(optimizer.state_dict()))
                    if len(trials)<search_attempts:continue
                    break
                restore_local_rng(rng, generators)
                return dict(accepted=True, initial_lr=initial_lr, accepted_lr=lr,
                    initial_learning_rates=initial_learning_rates, accepted_learning_rates=actor_learning_rates(optimizer),
                    backtracks=index, attempts=trials, attempt_count=len(trials),
                    same_initial_adam_per_attempt=True, fixed_gradient_reused=True)
        if best is not None:
            with torch.no_grad():
                for name,p in actor.named_parameters():
                    if name in best['model']:p.copy_(best['model'][name])
            optimizer.load_state_dict(best['adam']);controller.load_state_dict(control)
            restore_local_rng(rng,generators)
            return dict(accepted=True,initial_lr=initial_lr,accepted_lr=best['lr'],
                initial_learning_rates=initial_learning_rates,accepted_learning_rates=actor_learning_rates(optimizer),
                backtracks=len(trials)-1,attempts=trials,attempt_count=len(trials),
                selected_attempt=best['index']+1,selected_objective=best['score'],
                same_initial_adam_per_attempt=True,fixed_gradient_reused=True,
                selection='best_configured_clipped_objective_among_KL_qualified_trials')
        restore()
        finite_rejections=bool(trials) and all(t['kl'] is not None and
            set(t['acceptance']['failures']) <= {'mean_coordinate','maximum_step_mean_coordinate','p95_coordinate','mean_joint',
                'chain_joint_mean','chain_joint_p95',
                'chain_joint_p99','chain_joint_max','active_clip_fraction','objective_gain'} and
            all(math.isfinite(f['value']) for f in t['acceptance']['failures'].values())
            for t in trials)
        return dict(accepted=False, initial_lr=initial_lr, accepted_lr=None, backtracks=max(0,len(trials)-1),
            attempts=trials, attempt_count=len(trials), same_initial_adam_per_attempt=True,
            fixed_gradient_reused=True, initial_state_restored=True,
            rejection_reason='minimum_trial_lr' if len(trials)<max_attempts else 'attempts_exhausted',
            finite_trust_region_exhausted=finite_rejections)
    except BaseException:
        restore()
        raise
