"""GENMO第二阶段的有界困难片段采样统计，不参与奖励或Actor概率计算。

输入为正常训练已经执行的50Hz奖励证据及真实物理终止标记，按任务的音乐起点bin
累计后续跟踪难度；这估计“从该起点开始的一段执行有多难”，不是把失败归给某次
生成动作。8秒PPO切分不算失败，20秒到时单独记为删失观察，预热失败只作诊断。
每轮结束后跨rank归并新增统计，更新EMA；下一轮20%的任务按这些权重重新抽起点，
其余80%仍使用原无放回覆盖。所有新任务都用当前Actor重采，不回放旧动作或旧优势。

优先级严格在[1,3]内，因此归一化后的自适应概率不超过均匀bin概率的3倍，同一音乐
聚合概率也满足相同比率上限，避免归一化破坏截断上限。数据源比例仍由上层决定。
统计、待归并增量及EMA都持久化；不读取验证集，不改变GAE，不修改冻结GMT。
"""
from __future__ import annotations

import copy
import math
import numpy as np


DEFAULTS = dict(enabled=False, fraction=.2, ema_alpha=.1, max_priority=3.)
FIELDS = ('seconds', 'severity_seconds', 'physical_failures', 'music_ends', 'timeouts',
          'warmup_failures', 'transitions')


def adaptive_settings(settings=None):
    result = dict(DEFAULTS, **(settings or {}))
    if set(result) != set(DEFAULTS) or type(result['enabled']) is not bool:
        raise ValueError('Invalid adaptive time sampling settings')
    for key in ('fraction', 'ema_alpha', 'max_priority'):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]):
            raise ValueError('Adaptive sampling settings must be finite numbers')
    if not 0 < result['fraction'] < 1 or not 0 < result['ema_alpha'] <= 1 or not 1 <= result['max_priority'] <= 3:
        raise ValueError('Adaptive mixture/EMA must be probabilities and priority must be in [1,3]')
    return result


def physical_severity(detail):
    """只用消费参考的腿、根高度与非偏航姿态误差，输出[0,1]困难度。"""
    track = detail['components']['track']['raw']
    physical = detail['diagnostics']['physical_state']
    values = (float(track['leg_joint_pos_squared_error'])/.04,
              abs(float(physical['root_height_error_m']))/.1,
              abs(float(physical['non_yaw_orientation_error_rad']))/.3)
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('Invalid consumed tracking error in adaptive sampler')
    return sum(min(1., x) for x in values)/len(values)


class AdaptiveTimeWeights:
    def __init__(self, bins, settings):
        self.settings = adaptive_settings(settings)
        self.sizes = {s: len(v) for s, v in bins.items()}
        self.statistics, self.pending, self.levels = {}, {}, {}
        self.updates = 0
        self._probabilities = {}

    def _key(self, task):
        metadata = task.get('sampling', {}).get('time_bin')
        if metadata is None:
            return None
        source, index = metadata['source'], metadata['bin_id']
        if source not in self.sizes or type(index) is not int or not 0 <= index < self.sizes[source]:
            raise ValueError('Adaptive outcome references an unknown time bin')
        return f'{source}/{index}'

    def observe(self, task, row=None, *, warmup_failure=False):
        key = self._key(task)
        if key is None:
            return
        update = self.pending.setdefault(key, [0.]*len(FIELDS))
        if warmup_failure:
            update[5] += 1
            return
        details = row.metadata.get('reward_details', [])
        if len(details) != row.executed_control_steps:
            raise ValueError('Adaptive feedback must cover all executed controls')
        for detail in details:
            dt = float(detail['dt_s'])
            if not math.isclose(dt, .02, abs_tol=1e-12):
                raise ValueError('Adaptive feedback must use real 50Hz intervals')
            update[0] += dt
            update[1] += physical_severity(detail)*dt
        failed = bool(row.metadata.get('terminal_snapshot', {}).get('terminated', False))
        update[2] += int(failed)
        update[3] += int(not failed and row.reason == 'music_end')
        update[4] += int(not failed and row.metadata.get('physical_time_limit_reached', False))
        update[6] += 1

    def merge(self, shards):
        """所有rank在同一采集边界调用；仅归并新增量，不能重复累计历史总数。"""
        merged = {}
        for shard in shards:
            for key, values in shard.items():
                self._validate_record(key, values)
                merged[key] = np.asarray(merged.get(key, np.zeros(len(FIELDS)))) + values
        for key, values in merged.items():
            self.statistics[key] = (np.asarray(self.statistics.get(key, np.zeros(len(FIELDS)))) + values).tolist()
            seconds, mass, failures, ends, timeouts, _, transitions = values
            if not transitions:
                continue
            observed_ends = failures+ends+timeouts
            failure_rate = failures/observed_ends if observed_ends else 0.
            severity = .7*(mass/seconds if seconds else 0.) + .3*failure_rate
            old = self.levels.get(key, severity)
            alpha = self.settings['ema_alpha']
            self.levels[key] = min(1., max(0., (1-alpha)*old+alpha*severity))
        self.pending = {}
        self.updates += 1
        self._probabilities = {}

    def probabilities(self, source):
        if source not in self._probabilities:
            weights = np.ones(self.sizes[source], dtype=np.float64)
            for key, level in self.levels.items():
                name, index = key.rsplit('/', 1)
                if name == source:
                    weights[int(index)] += (self.settings['max_priority']-1)*level
            self._probabilities[source] = weights/weights.sum()
        return self._probabilities[source]

    def _validate_record(self, key, values):
        source, index = key.rsplit('/', 1)
        if source not in self.sizes or not 0 <= int(index) < self.sizes[source]:
            raise ValueError('Unknown adaptive bin in checkpoint/collective')
        if len(values) != len(FIELDS) or not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError('Invalid adaptive outcome statistics')
        if values[1] > values[0]+1e-8:
            raise ValueError('Adaptive severity exceeds observed time')

    def state_dict(self):
        return copy.deepcopy(dict(schema='genmo.adaptive_start_bin.v1', settings=self.settings,
            sizes=self.sizes, statistics=self.statistics, pending=self.pending, levels=self.levels, updates=self.updates))

    def load_state_dict(self, state):
        if (state.get('schema') != 'genmo.adaptive_start_bin.v1' or state['settings'] != self.settings
                or state['sizes'] != self.sizes or type(state['updates']) is not int or state['updates'] < 0):
            raise ValueError('Adaptive sampling checkpoint settings differ')
        for name in ('statistics', 'pending'):
            for key, values in state[name].items():
                self._validate_record(key, values)
        for key, value in state['levels'].items():
            self._validate_record(key, [0.]*len(FIELDS))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid adaptive EMA')
        for name in ('statistics', 'pending', 'levels', 'updates'):
            setattr(self, name, copy.deepcopy(state[name]))
        self._probabilities = {}

    def summary(self):
        totals = np.sum(list(self.statistics.values()), axis=0) if self.statistics else np.zeros(len(FIELDS))
        return dict(enabled=True, fraction=self.settings['fraction'], updates=self.updates,
            observed_start_bins=len(self.statistics), totals=dict(zip(FIELDS, totals.tolist())),
            adaptive_probability_ratio_bound=self.settings['max_priority'],
            scope='training_start_bin_following_execution_not_action_credit',
            timeout_semantics='censored_window_not_full_music_success',
            per_source={s: dict(max_bin_probability=float(self.probabilities(s).max()),
                max_probability_ratio=float(self.probabilities(s).max()*n)) for s,n in self.sizes.items()})
