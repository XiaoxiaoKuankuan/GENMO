"""当前部署模拟时钟证据的独立核对，不保留旧配额等待或旧断点审计。

核对实际控制栅格、请求/到达/提交顺序及延迟profile身份；纯数学读取不推进仿真。
"""
def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_vector(identity):
    return identity.get('execution_contract',{}).get('runtime',{}).get('backend')=='gpu_vectorized.v1'


def audit_modeled_clock(metadata, task, begin, identity):
    """从原配置/任务独立重建部署延迟，拒绝墙钟、拓扑或持久化污染模拟时间。"""
    from gem.closedloop.dppo.deployment_clock import MODELED_CLOCK,DeploymentClock
    contract=identity['execution_contract']
    if contract['runtime'].get('timing_contract')!=MODELED_CLOCK:return
    clock=DeploymentClock(contract['timing']['deployment_profile'])
    expected=clock.sample(seed=identity['base_seed'],sample_id=task['sample_id'],
        music_start_frame=task['music_start_frame'],decision_tick=begin)
    require(metadata.get('timing_contract')==MODELED_CLOCK and
            metadata.get('timing',{}).get('deployment_clock')==expected,
            'Modeled deployment clock differs from immutable task/profile identity')
    generated=metadata['generated']
    require(generated['deadline_tick']<=begin+clock.budget_ticks,
            'Training wallclock increased modeled prefix budget')
