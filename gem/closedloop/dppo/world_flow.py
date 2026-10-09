"""闭环环境的可恢复因果续体与世界级调度消息。

同一份环境控制逻辑可由旧同步客户端逐条执行，也可由新世界调度器集中收集所有
环境的reserve/prepare/commit/advance后批量提交。续体只在明确RPC或批量GENMO
边界挂起，状态、异常、奖励和episode仍归属于原环境，不创建每环境线程，也不复制
一套控制/奖励公式。服务器回复须先由世界权威journal持久化，再恢复相关续体。
本模块不定义物理时钟；modeled_deployment.v3仍唯一决定模拟到达和前缀P。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendOperation:
    method: str
    payload: dict


@dataclass(frozen=True)
class GenerationOperation:
    packet: dict


@dataclass(frozen=True)
class LocalOperation:
    function: object
    arguments: dict


def backend_operation(method, **payload):
    return (yield BackendOperation(method, payload))


def run_synchronous(env, flow):
    """旧单环境/线程路径的兼容驱动，异常在原yield点抛回，保留业务拒绝处理。"""
    result, error = None, None
    while True:
        try:
            operation = flow.throw(error) if error is not None else flow.send(result)
        except StopIteration as finished:
            return finished.value
        result, error = None, None
        try:
            if isinstance(operation, BackendOperation):
                result = env.backend.call(operation.method, **operation.payload)
            elif isinstance(operation, LocalOperation):
                result = operation.function(**operation.arguments)
            elif isinstance(operation, GenerationOperation):
                from concurrent.futures import Future
                from .vector_generation import await_generation
                future = Future()
                import time
                owner = env.policy.owner
                owner.requests.put((env.collector_env_slot, operation.packet, None, future, time.perf_counter()))
                result = await_generation(owner, future)
            else:
                raise TypeError('Unknown world flow operation')
        except Exception as failure:
            error = failure
