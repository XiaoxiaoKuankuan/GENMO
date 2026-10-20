"""统一Actor学习率的单步KL回溯事务，供Clipped DPPO和单步score目标共用。

调用前完成一次全批量PPO/BC合成与裁剪。本模块保留模型、AdamW、BC控制器、指定
生成器及Python/NumPy/Torch随机状态，每次从相同起点用固定梯度执行一次AdamW，
按1、1/2、1/4等有限次数尝试。KL必须在同一真实旧链上覆盖全部可训练四步；
所有rank共同接受或拒绝，冻结参数不允许进入优化器。失败尝试的预算不退款。
此处不提高噪声、不截断概率，也不改写行为统计；全部失败恢复起点并返回拒绝证据，
由调用者恢复本步BC采样游标并保留此前接受步骤。完整检查结果可绑定身份后复用，
不重复最终前向。可复用快照只覆盖Actor及Adam，不复制Critic、GMT或冻结先验。
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


def check_candidate_kl(report, limits):
    result = check_kl(report, limits)
    for name in ('mean', 'p95', 'max'):
        bound = limits.get('chain_joint_'+name)
        if bound is not None:
            value = report['chain_joint_kl'][name]
            if not math.isfinite(value) or value > bound:
                result['failures']['chain_joint_'+name] = dict(value=value, limit=bound)
    result['passed'] = not result['failures']
    result['joint_gate_is_empirical_not_theoretical_safety'] = True
    return result


def _copy_tree(source, destination=None):
    """复用同设备张量存储；首次Adam状态出现时才分配，不往返CPU。"""
    if isinstance(source, torch.Tensor):
        if (not isinstance(destination, torch.Tensor) or destination.shape != source.shape
                or destination.device != source.device or destination.dtype != source.dtype):
            return source.detach().clone()
        destination.copy_(source.detach())
        return destination
    if isinstance(source, dict):
        destination = destination if isinstance(destination, dict) else {}
        for key in list(destination):
            if key not in source: del destination[key]
        for key, value in source.items(): destination[key] = _copy_tree(value, destination.get(key))
        return destination
    return copy.deepcopy(source)


class StepSnapshot:
    """每轮复用一份Actor/Adam快照；restore不重新分配已有Adam大张量。"""
    def __init__(self):
        self.model = None
        self.states = {}
        self.groups = None

    @torch.no_grad()
    def capture(self, actor, optimizer):
        self.model = _copy_tree(actor.state_dict(), self.model)
        parameters = [p for g in optimizer.param_groups for p in g['params']]
        for p in parameters:
            self.states[p] = _copy_tree(optimizer.state.get(p, {}), self.states.get(p))
        self.groups = [{k:copy.deepcopy(v) for k,v in g.items() if k != 'params'}
                       for g in optimizer.param_groups]

    @torch.no_grad()
    def restore(self, actor, optimizer):
        actor.load_state_dict(self.model)
        for p, state in self.states.items():
            if state: optimizer.state[p] = _copy_tree(state, optimizer.state.get(p))
            else: optimizer.state.pop(p, None)
        for group, saved in zip(optimizer.param_groups, self.groups):
            for key in list(group):
                if key != 'params' and key not in saved: del group[key]
            group.update(copy.deepcopy(saved))


def backtracked_step(actor, optimizer, controller, *, evaluate, limits, distributed, generators=None,
                     max_attempts=9, factor=.5, reserve_attempt=None, observe=None,
                     minimum_lr=0., workspace=None):
    uniform_actor_optimizer(actor, optimizer)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10 or factor != .5:
        raise ValueError('Finite backtracking requires 1..10 attempts with factor 0.5')
    group = optimizer.param_groups[0]
    initial_lr = float(group['lr'])
    if not math.isfinite(initial_lr) or not 0 < initial_lr <= 1e-5:
        raise ValueError('Explicit experimental Actor LR must be in (0,1e-5]')
    if not math.isfinite(minimum_lr) or not 0 <= minimum_lr <= initial_lr:
        raise ValueError('Invalid minimum trial learning rate')
    workspace = workspace if workspace is not None else StepSnapshot()
    local_call(distributed, lambda:workspace.capture(actor, optimizer))
    control = copy.deepcopy(controller.state_dict())
    rng = capture_local_rng(generators)
    grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in actor.named_parameters()}
    def restore():
        workspace.restore(actor, optimizer)
        controller.load_state_dict(control)
        restore_local_rng(rng, generators)
        for n, p in actor.named_parameters():
            p.grad = None if grads[n] is None else grads[n].clone()
    trials = []
    try:
        for index in range(max_attempts):
            lr = initial_lr * factor**index
            if lr < minimum_lr: break
            if index:
                local_call(distributed, restore)
            optimizer.param_groups[0]['lr'] = lr
            if reserve_attempt is not None:
                reserve_attempt()
            started = time.perf_counter()
            local_call(distributed,optimizer.step)
            with torch.no_grad():
                kl = evaluate()
                acceptance = check_candidate_kl(kl, limits)
            if any(isinstance(f,dict) and not math.isfinite(f['value'])
                   for f in acceptance['failures'].values()):
                raise FloatingPointError('Nonfinite KL is a numerical failure, not a rejected step')
            flags = distributed.all_gather_object(acceptance['passed'])
            if any(flag != flags[0] for flag in flags):
                raise RuntimeError('Ranks disagree on backtracking acceptance')
            diagnostics = observe() if observe is not None else None
            trials.append(dict(attempt=index+1, learning_rate=lr, factor=factor**index,
                kl=kl, acceptance=acceptance, diagnostics=diagnostics,
                wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started))))
            if acceptance['passed']:
                # 候选审查和诊断无权消耗学习RNG；下一轮从一次梯度计算后的随机状态继续。
                restore_local_rng(rng, generators)
                return dict(accepted=True, initial_lr=initial_lr, accepted_lr=lr,
                    backtracks=index, attempts=trials, attempt_count=len(trials),
                    same_initial_adam_per_attempt=True, fixed_gradient_reused=True)
        local_call(distributed, restore)
        return dict(accepted=False, initial_lr=initial_lr, accepted_lr=None, backtracks=max(0,len(trials)-1),
            attempts=trials, attempt_count=len(trials), same_initial_adam_per_attempt=True,
            fixed_gradient_reused=True, initial_state_restored=True)
    except BaseException:
        local_call(distributed, restore)
        raise
