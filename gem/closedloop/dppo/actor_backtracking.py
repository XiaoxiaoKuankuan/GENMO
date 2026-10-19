"""统一Actor学习率的单步KL回溯事务，供完整链PPO和数学参考调用。

调用前完成一次全批量PPO/BC合成与裁剪。本模块保留模型、AdamW、BC控制器、指定
生成器及Python/NumPy/Torch随机状态，每次从相同起点用固定梯度执行一次AdamW，
按1、1/2、1/4等有限次数尝试。生产检查累计已用完整链的全部四步；
所有rank共同接受或拒绝，冻结参数不允许进入优化器或参数快照。失败尝试的预算不退款。
此处不提高噪声、不截断概率，也不改写行为统计；全部失败恢复起点并返回拒绝证据，
由训练器拒绝整轮。单步通过后训练器仍执行完整rollout最终KL审查。
梯度与BC只计算一次，候选从相同Adam时刻重试；不靠重复Adam累计动量缩小步长。
"""
from __future__ import annotations

import copy
import math
import time
import torch

from .parallel_support import capture_local_rng, restore_local_rng, local_call
from .updater import check_kl


def uniform_actor_optimizer(actor, optimizer):
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 1:
        raise ValueError('Actor must use one uniform AdamW parameter group')
    expected = {id(p) for p in actor.parameters() if p.requires_grad}
    actual = [id(p) for p in optimizer.param_groups[0]['params']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('AdamW must contain every trainable Actor parameter exactly once and no frozen parameters')
    if any(p.dtype != torch.float32 for p in optimizer.param_groups[0]['params']):
        raise ValueError('FP32 master parameters required')
    if getattr(actor,'_stage2_row_gradient_hooks',[]) and optimizer.param_groups[0]['weight_decay']!=0:
        raise ValueError('Row-frozen Actor parameters require AdamW weight_decay=0')


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
    result['passed'] = not result['failures']
    result['joint_gate_is_empirical_not_theoretical_safety'] = True
    return result


def backtracked_step(actor, optimizer, controller, *, evaluate, limits, distributed, generators=None,
                     max_attempts=9, factor=.5, reserve_attempt=None, observe=None):
    uniform_actor_optimizer(actor, optimizer)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10 or factor != .5:
        raise ValueError('Finite backtracking requires 1..10 attempts with factor 0.5')
    group = optimizer.param_groups[0]
    initial_lr = float(group['lr'])
    if not math.isfinite(initial_lr) or not 0 < initial_lr <= 1e-5:
        raise ValueError('Explicit experimental Actor LR must be in (0,1e-5]')
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
    trials = []
    try:
        for index in range(max_attempts):
            if index:
                restore()
            lr = initial_lr * factor**index
            optimizer.param_groups[0]['lr'] = lr
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
            trials.append(dict(attempt=index+1, learning_rate=lr, factor=factor**index,
                kl=kl, acceptance=acceptance, diagnostics=diagnostics,
                wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started))))
            if acceptance['passed']:
                # 候选审查和诊断无权消耗学习RNG；下一轮从一次梯度计算后的随机状态继续。
                restore_local_rng(rng, generators)
                return dict(accepted=True, initial_lr=initial_lr, accepted_lr=lr,
                    backtracks=index, attempts=trials, attempt_count=len(trials),
                    same_initial_adam_per_attempt=True, fixed_gradient_reused=True)
        restore()
        return dict(accepted=False, initial_lr=initial_lr, accepted_lr=None, backtracks=max_attempts-1,
            attempts=trials, attempt_count=len(trials), same_initial_adam_per_attempt=True,
            fixed_gradient_reused=True, initial_state_restored=True)
    except BaseException:
        restore()
        raise
