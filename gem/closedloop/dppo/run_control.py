"""跨 checkpoint 保留的运行墙钟期限与显式策略构造。

48 小时期限从新运行启动时确定，保存原始开始时间和预算；恢复沿用同一期限，
包括中断期间的时间，防止反复恢复重新获得预算。模型更新回滚不退还墙钟时间。
策略构造从实际训练配置显式传入扩散步数、eta、CFG 与精度参数，消除默认值漂移。
本模块不启动仿真、不分配数据预算，也不改变实际控制时钟。
"""
from __future__ import annotations
import math
import time


class RunWallBudget:
    def __init__(self, seconds, *, started=None, clock=time.time):
        self.clock = clock
        self.limit = float(seconds)
        self.started = float(clock() if started is None else started)
        if not math.isfinite(self.limit) or self.limit <= 0 or not math.isfinite(self.started):
            raise ValueError('Finite positive run deadline required')

    def state_dict(self):
        elapsed = float(self.clock()) - self.started
        if elapsed < 0:
            raise RuntimeError('Wall clock moved before the persisted run start')
        return dict(schema='genmo.run_wall_budget.v1', started_unix_seconds=self.started,
                    limit_seconds=self.limit, elapsed_seconds=elapsed,
                    deadline_unix_seconds=self.started+self.limit, downtime_counts=True)

    @property
    def expired(self):
        return self.state_dict()['elapsed_seconds'] >= self.limit

    def restore(self, state):
        if state.get('schema') != 'genmo.run_wall_budget.v1' or state['limit_seconds'] != self.limit:
            raise ValueError('Run wall budget identity differs')
        if state['deadline_unix_seconds'] != state['started_unix_seconds'] + self.limit:
            raise ValueError('Corrupted persisted run deadline')
        self.started = float(state['started_unix_seconds'])
        self.state_dict()


def build_policy(actor, settings, precision):
    from .dense_policy import DenseDiffusionPolicy
    return DenseDiffusionPolicy(actor, std_schedule=settings['std_schedule'], steps=settings['denoising_steps'],
        eta=settings['eta'], guidance_scale=settings['guidance_scale'], precision=precision['mode'],
        attention=precision['attention'], linear_execution=precision['linear_execution'],
        compile_denoiser=precision['compile'], precise_output=precision['precise_output'],
        weight_gradient=precision.get('weight_gradient', 'direct_joint_gemm'))
