"""统一Actor学习率的跨rollout缓慢调节与严格恢复。

输入本轮相对固定行为策略的完整四步KL及真实接受/回溯次数，用三项门槛中最大的
占用比例控制下一轮基础学习率。首次零KL不触发加速；连续低占用且全部计划步骤
接受后才小幅提高，回溯或接近门槛则降低。所有Actor参数仍共用一个AdamW参数组。
本模块不修改概率、优势、奖励或物理时钟；控制器配置和状态一起持久化，恢复时
拒绝不匹配或非有限状态。它不代表物理收益或理论安全保证。
"""
from __future__ import annotations

import copy
import math


def kl_utilization(report, limits):
    values = (report['mean_coordinate_kl']/limits['mean_coordinate'],
              report['maximum_step_mean_coordinate_kl']/limits['maximum_step_mean_coordinate'],
              report['coordinate_kl']['p95']/limits['p95_coordinate'])
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise FloatingPointError('Nonfinite or negative complete KL utilization')
    return max(values)


class ActorLRController:
    def __init__(self, initial_lr, config):
        self.config = copy.deepcopy(config)
        keys = {'minimum', 'maximum', 'decrease_factor', 'increase_factor',
                'low_utilization', 'high_utilization', 'low_rounds'}
        if (set(config) != keys or not all(math.isfinite(float(v)) for v in config.values())
                or not 0 < config['minimum'] <= initial_lr <= config['maximum'] <= 1e-5
                or not 0 < config['decrease_factor'] < 1 < config['increase_factor']
                or not 0 < config['low_utilization'] < config['high_utilization'] < 1
                or type(config['low_rounds']) is not int or config['low_rounds'] < 1):
            raise ValueError('Invalid uniform adaptive Actor LR configuration')
        self.base_lr = float(initial_lr)
        self.low_streak = 0
        self.rounds = 0

    def update(self, report, limits, *, accepted, planned, attempts):
        u = kl_utilization(report, limits)
        before = self.base_lr
        retried = attempts > accepted
        complete = accepted == planned and accepted > 0
        if retried or not complete or u > self.config['high_utilization']:
            self.base_lr = max(self.config['minimum'], before*self.config['decrease_factor'])
            self.low_streak = 0
            reason = 'backtracking_or_partial_or_high_kl'
        elif u < self.config['low_utilization']:
            self.low_streak += 1
            reason = 'low_kl_observation'
            if self.low_streak >= self.config['low_rounds']:
                self.base_lr = min(self.config['maximum'], before*self.config['increase_factor'])
                self.low_streak = 0
                reason = 'sustained_low_kl_full_updates'
        else:
            self.low_streak = 0
            reason = 'hold'
        self.rounds += 1
        return dict(previous_base_lr=before, next_base_lr=self.base_lr, utilization=u,
                    low_streak=self.low_streak, reason=reason, accepted_steps=accepted,
                    planned_steps=planned, optimizer_attempts=attempts)

    def state_dict(self):
        return dict(schema='genmo.actor_lr.step_kl.v1', config=copy.deepcopy(self.config),
                    base_lr=self.base_lr, low_streak=self.low_streak, rounds=self.rounds)

    def load_state_dict(self, state):
        if (set(state) != {'schema','config','base_lr','low_streak','rounds'}
                or state['schema'] != 'genmo.actor_lr.step_kl.v1' or state['config'] != self.config
                or not math.isfinite(state['base_lr'])
                or not self.config['minimum'] <= state['base_lr'] <= self.config['maximum']
                or type(state['low_streak']) is not int
                or not 0 <= state['low_streak'] < self.config['low_rounds']
                or type(state['rounds']) is not int or state['rounds'] < state['low_streak']):
            raise ValueError('Invalid/mismatched adaptive Actor LR checkpoint')
        self.base_lr = state['base_lr']
        self.low_streak = state['low_streak']
        self.rounds = state['rounds']
