"""闭环执行边界的计数快照、恢复检查和冻结后端确认。

本模块保存生成尝试、已执行决策和 episode 的显式计数，并检查部署延迟合同。
恢复只重建新的物理世界与噪声编号，不保存或伪造 PhysX 中间状态；失败时已经
消耗的生成编号不能复用。训练、向量采集和诊断共同使用这些函数，无需导入 CLI。
"""
from __future__ import annotations
import math

def _positive(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    if integer and type(value) is not int:
        raise ValueError(f'{name} must be an integer')



def validate_execution_state(state):
    """旧状态缺失噪声计数时拒绝恢复；不能把显式采样种子的组成部分猜成零。"""
    for name in ('decision', 'attempt', 'episode_count'):
        if name not in state or type(state[name]) is not int or state[name] < 0:
            raise ValueError(f'Stage10 checkpoint requires explicit nonnegative integer execution state: {name}')
    if state['decision'] > state['attempt']:
        raise ValueError('Stage10 decision count cannot exceed generation attempts')
    _positive(state.get('latency_budget_s'), 'checkpoint latency_budget_s')



def capture_execution_state(env):
    """初始校准后及每轮发布前共用的执行计数快照，不保存PhysX内部状态。"""
    result = {name:getattr(env,name) for name in ('decision', 'attempt', 'episode_count', 'latency_budget_s')}
    clock = getattr(env, 'deployment_clock', None)
    if clock is not None: result['deployment_profile_sha256'] = clock.sha256
    validate_execution_state(result)
    return result



def restore_execution_state(env, state, *, spent_generations=None):
    """恢复显式种子计数；失败尝试已消耗的generation编号仍不复用。"""
    validate_execution_state(state)
    if spent_generations is not None and (type(spent_generations) is not int or spent_generations < 0):
        raise ValueError('Spent generation counter must be a nonnegative integer')
    env.policy_version, env.iteration = state['policy_version'], state['iteration']
    env.decision, env.episode_count = state['decision'], state['episode_count']
    env.attempt = max(state['attempt'], state['attempt'] if spent_generations is None else spent_generations)
    clock = getattr(env, 'deployment_clock', None)
    if state.get('deployment_profile_sha256') != (None if clock is None else clock.sha256):
        raise ValueError('Checkpoint deployment clock/profile differs from this run')
    if clock is not None and state['latency_budget_s'] != clock.budget_seconds:
        raise ValueError('Checkpoint changed modeled prefix budget')
    env.latency_budget_s = state['latency_budget_s']



def _assert_frozen(result):
    if result.get('policy_unchanged') is not True or result.get('runtime_parameters_unchanged') is not True:
        raise RuntimeError('Frozen GMT policy or runtime parameters changed')
    journal = result.get('execution_journal', {})
    if journal.get('executed_seq') != journal.get('acked_seq'):
        raise RuntimeError('Checkpoint boundary contains an unacknowledged physical mutation')
