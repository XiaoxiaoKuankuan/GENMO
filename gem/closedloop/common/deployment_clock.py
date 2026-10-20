"""第二阶段独立部署时钟：把目标机器人的延迟与训练服务器墙钟彻底分开。

modeled_deployment.v3 使用带来源身份的单请求实测延迟样本，在600Hz整数时钟上
按任务/决策身份确定性抽样。环境数量、微批大小、线程排队和训练审计耗时不进入
抽样键或前缀保护预算；等待期间仍由原执行器真实推进旧参考，不跳过物理或奖励。
实测训练墙钟继续完整记录，用于吞吐与部署性能诊断，但不得修改模拟到达时刻。

配置整体内容SHA写入每条转移与恢复状态。无额外可变RNG，恢复时沿用任务、决策
计数即可；不同profile或新旧时钟不能冒充相同运行身份。旧实时时钟路径仍可显式
运行，用于部署回放。本模块只使用标准库，独立于Torch、Isaac和训练进程数量。
"""
from __future__ import annotations

import hashlib
import json
import math

MODELED_CLOCK = 'modeled_deployment.v3'


class DeploymentClock:
    def __init__(self, profile):
        if not isinstance(profile, dict) or profile.get('schema') != 'genmo.deployment_delay_profile.v1':
            raise ValueError('Explicit measured deployment delay profile required')
        if profile.get('target') != 'single_robot_single_request':
            raise ValueError('Deployment target must not depend on training environment count')
        samples = profile.get('samples_seconds')
        if (not isinstance(samples, list) or not samples or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples)):
            raise ValueError('Deployment samples must be positive finite measured seconds')
        budget = profile.get('prefix_budget_seconds')
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            raise ValueError('Explicit positive prefix budget required')
        provenance = profile.get('provenance')
        if not isinstance(provenance, list) or not provenance or any(
                not isinstance(p, dict) or not p.get('path') or len(p.get('sha256', '')) != 64 for p in provenance):
            raise ValueError('Measured profile must bind its source evidence SHA256')
        raw = json.dumps(profile, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.sample_ticks = tuple(math.ceil(v*50)*12 for v in samples)
        self.budget_ticks = math.ceil(budget*50)*12
        self.budget_seconds = self.budget_ticks / 600.

    def sample(self, *, seed, sample_id, music_start_frame, decision_tick):
        # 不包含env_id/rank/N/run UUID。相同任务、外部种子和决策时刻在不同拓扑下相同。
        key = json.dumps([int(seed), str(sample_id), int(music_start_frame), int(decision_tick)], separators=(',', ':'))
        index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % len(self.sample_ticks)
        ticks = self.sample_ticks[index]
        return dict(contract=MODELED_CLOCK, profile_sha256=self.sha256, sample_index=index,
                    delay_ticks=ticks, delay_seconds=ticks/600., prefix_budget_ticks=self.budget_ticks,
                    arrival_tick=int(decision_tick)+ticks,
                    scope='modeled_single_robot_delay_independent_of_training_wallclock')


def clock_for_config(config):
    if config.get('runtime', {}).get('timing_contract') != MODELED_CLOCK:
        return None
    return DeploymentClock(config.get('timing', {}).get('deployment_profile'))
