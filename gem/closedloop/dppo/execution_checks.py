"""计算阶段的统一验证边界与设备端异常汇总。

静态模型检查在阶段入口执行，阶段内复用参数版本签名；离开阶段核验版本未变。
动态断言仍覆盖每一个内部去噪步骤，但仅累积设备端布尔标志，在阶段结束时一次
取回结果。调用者必须将此边界放在跨卡错误传播范围内、optimizer.step 或 GMT
提交之前。默认未启用时保留立即检查；该上下文不吞掉 Python、OOM 或通信异常。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

_CHECKS = ContextVar('stage10_deferred_checks', default=None)


def require_tensor(condition, message, error=ValueError):
    checks = _CHECKS.get()
    if checks is None:
        if not bool(condition):
            raise error(message)
    else:
        checks.append((condition.detach().all(), message, error))


@contextmanager
def policy_phase(policy):
    from .numerical_execution import precision_scope
    with precision_scope(getattr(policy, 'numerical_execution', None)):
        with _validation_phase(policy):
            yield


@contextmanager
def _validation_phase(policy):
    if not getattr(policy, 'defer_checks', False) or getattr(policy, '_phase_signature', None) is not None:
        yield
        return
    policy._prepare_actor()
    signature = policy._parameter_signature()
    policy._phase_signature = signature
    checks = []
    token = _CHECKS.set(checks)
    try:
        yield
        if checks:
            good = torch.stack([flag for flag, _, _ in checks]).cpu().tolist()
            for passed, (_, message, error) in zip(good, checks):
                if not passed:
                    raise error(message)
    finally:
        policy._phase_signature = None
        _CHECKS.reset(token)
    if policy._parameter_signature() != signature:
        raise RuntimeError('Actor changed inside a fixed-parameter compute phase')


def checked_policy_phase(function):
    @wraps(function)
    def call(policy, *args, **kwargs):
        with policy_phase(policy):
            return function(policy, *args, **kwargs)
    return call
