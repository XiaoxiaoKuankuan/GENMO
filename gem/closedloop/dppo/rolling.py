"""连续滚动生成的独立调度合同，不参与 DPPO 概率、优化或奖励定义。

旧配置缺省仍是固定 300 tick 决策。新配置显式声明 60 tick 公共网格和可替换
前瞻参考合同，部署延迟仍来自单机器人实测时钟。一次请求完成后向上取整至合法
网格，不能把训练墙钟或参考长度当成执行周期。容量只是按最短网格估算的资源
上界，真实链数由八秒物理窗口决定；所有配置字段进入已有 runtime checkpoint
身份，旧断点不能隐式 resume 到新模式。本模块仅使用标准库，可独立测试。
"""
from __future__ import annotations

SCHEMA = 'genmo.continuous_rolling.v1'
REFERENCE_CONTRACT = 'gmt.executed_prefix_replaceable_lookahead.v1'


def rolling_enabled(config):
    return config.get('runtime', {}).get('generation_schedule', {}).get('mode') == 'continuous_rolling'


def decision_grid(config):
    return 60 if rolling_enabled(config) else 300


def next_decision_tick(start, arrival, grid):
    if type(start) is not int or type(arrival) is not int or grid not in (60, 300):
        raise ValueError('Integer deployment ticks and an explicit legal grid required')
    if start % grid or arrival < start:
        raise ValueError('Invalid generation decision/arrival tick')
    return max(start + grid, -(-arrival // grid) * grid)


def validate_schedule(config):
    settings = config.get('runtime', {}).get('generation_schedule')
    if settings is None:
        return
    expected = dict(schema=SCHEMA, mode='continuous_rolling', decision_grid_ticks=60,
                    reference_contract=REFERENCE_CONTRACT, minimum_prefix_frames=6)
    if settings != expected:
        raise ValueError('Unknown continuous rolling schedule/reference contract')
    if config['runtime'].get('backend') != 'gpu_vectorized.v1':
        raise ValueError('Continuous rolling requires the shared vector world')
    if config['runtime'].get('timing_contract') != 'modeled_deployment.v3':
        raise ValueError('Rolling uses measured single-robot deployment latency only')


def generation_capacity(config, seconds):
    import math
    return math.ceil(float(seconds) * 600 / decision_grid(config))


def store_rollout_on_host(row):
    """仅搬运采集所需张量，不裁链、不丢条件；新模式使用既有有界设备块缓存。"""
    import torch
    memo = {}
    def move(value):
        if torch.is_tensor(value):
            if id(value) not in memo:
                memo[id(value)] = value.detach().to('cpu')
            return memo[id(value)]
        if isinstance(value, dict):return {k:move(v) for k,v in value.items()}
        if isinstance(value, list):return [move(v) for v in value]
        if isinstance(value, tuple):return tuple(move(v) for v in value)
        return value
    for key in ('context','next_context','chain','old_log_prob','free_mask','metadata'):
        setattr(row,key,move(getattr(row,key)))


def scheduling_report(rows):
    """仅汇总已保存的真实区间/提交事件，不把模拟延迟换成服务器墙钟。"""
    from collections import Counter
    durations = Counter(int(r.executed_control_steps) for r in rows)
    prefixes = Counter(int(r.metadata['generated'].get('prefix_frames', 0)) for r in rows)
    return dict(schema=SCHEMA, transitions=len(rows),
        control_steps=sum(r.executed_control_steps for r in rows),
        duration_control_histogram={str(k):v for k,v in sorted(durations.items())},
        prefix_frame_histogram={str(k):v for k,v in sorted(prefixes.items())},
        successful_commits=sum(r.metadata.get('published') is not None for r in rows),
        cancelled_at_boundary=sum(r.metadata.get('pending_cancelled_at_boundary',False) for r in rows),
        rejections=dict(Counter(r.metadata['rejection']['code'] for r in rows if r.metadata.get('rejection'))),
        storage='host_complete_chains_with_bounded_device_cache')


def world_window_credit(config, seconds):
    """预付覆盖最大25步在途RPC；真实窗口只执行400步，未执行部分必须退款。"""
    controls=round(seconds*50)+25
    grid_controls=decision_grid(config)//12
    return dict(control_steps=controls, physics_steps=4*controls,
        generations=(controls+grid_controls-1)//grid_controls+2)
