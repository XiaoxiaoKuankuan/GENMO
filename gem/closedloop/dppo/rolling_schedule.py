"""连续滚动生成的时间合同，不改变Actor、概率核或物理时钟。

本模块从显式实验配置解析公共决策网格。生成请求冻结该时刻的真实历史，实测
单请求延迟由DeploymentClock映射到600Hz时钟；期间每个50Hz控制及四个PhysX
子步照常执行旧参考。提交后在下一个30/50Hz公共点采样新状态，没有固定执行
0.5秒的配额。这里的异步是因果离散事件模拟，不宣称训练GPU墙钟实时调度。

预计提交预算只决定已知前缀，真实到达超过预算仍拒绝旧票据；不得偷看下一次
采样延迟、扩大已训练前缀范围或丢弃真实控制奖励。训练仍以0.5秒为gamma/lambda
的物理标定单位，实际转移用executed_control_steps指数换算。
"""
from __future__ import annotations

ROLLING = 'continuous_rolling.v1'
REFERENCE = 'executed_only_future_replace.v1'


def rolling(config):
    mode = config.get('runtime', {}).get('generation_schedule', 'fixed_grid.v1')
    if mode not in ('fixed_grid.v1', ROLLING):
        raise ValueError('Unknown generation schedule')
    return mode == ROLLING


def decision_ticks(config):
    return 60 if rolling(config) else 300


def next_decision_tick(start, arrival, config):
    grid = decision_ticks(config)
    return max(int(start) + grid, -(-int(arrival) // grid) * grid)


def validate_schedule(config):
    if not rolling(config):
        return
    runtime, timing = config['runtime'], config['timing']
    if runtime.get('reference_update_contract') != REFERENCE:
        raise ValueError('Rolling generation requires mutable future reference with executed-history protection')
    if timing['decision_ticks'] != 60 or timing['min_prefix_frames'] != 6:
        raise ValueError('Rolling v1 uses the 0.1-second common grid and trained prefix minimum 6')
    if runtime['backend'] != 'gpu_vectorized.v1' or runtime['timing_contract'] != 'modeled_deployment.v3':
        raise ValueError('Rolling v1 requires measured-delay discrete-event PhysX execution')
    if config['stage10']['training']['execution_mode'] != 'latency':
        raise ValueError('Rolling physical time cannot pause for generation')
    if config['stage10']['generation_horizon'] != {'frames': 60, 'fps': 30}:
        raise ValueError('Rolling v1 keeps the original 60-frame, 30Hz generator')
    if config['timing']['deployment_profile']['prefix_budget_seconds'] > .94:
        raise ValueError('The safe prefix must fit within the Stage1-trained 30-frame maximum')


def capacity(config, seconds):
    """最多每公共格一条真实链；只是容量，绝不补齐或强制采足。"""
    ticks = seconds * 600
    grid = decision_ticks(config)
    if seconds <= 0 or round(ticks) != ticks or round(ticks) % grid:
        raise ValueError('Physical window must lie on the configured common decision grid')
    return round(ticks) // grid
