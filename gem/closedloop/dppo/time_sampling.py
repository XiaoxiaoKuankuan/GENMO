"""普通Actor训练的共享时间分层采样与真实覆盖汇总。

每个来源的全部合法音乐起点按固定秒数划成bin，先用共同种子打乱，再给各rank
分配互不重叠的固定bin集合。同一rank的所有物理环境共用一个无放回游标，因此
不会因为每个slot独立打乱而反复选择同一小批起点。集合走完才在本rank重新打乱，
bin内仍随机选30Hz帧；困难动作之前失败不会阻止后段在后续任务中被直接抽中。
可选adaptive只将显式比例任务改为全局有放回困难采样，其余继续原覆盖游标；
跨rank同步只在每轮采集结束发生，不在物理控制循环中执行collective。

保留显式比例的站立开头任务，其他任务用配对动作预热初始化。来源概率仍按任务
计量，本模块不冒充控制时间均衡，也不把已抽到的bin当成实际物理覆盖。实际覆盖
单独合并各slot回填的600Hz区间。随机状态、分片、游标和轮数全部进入恢复身份；
分片数量改变或旧采样checkpoint不能静默恢复。只操作CPU数据，不推进仿真。
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .full_dataset import SOURCES, _merge_intervals


TIME_SAMPLING_CONTRACT = 'rank_shared_time_bins.v1'
MOTION_RESET_CONTRACT = 'paired_motion_warmup.v1'


class StratifiedTimePool:
    def __init__(self, catalog, *, seed, rank, world_size, minimum_remaining_seconds,
                 bin_seconds, standing_start_probability, adaptive=None):
        if (type(seed) is not int or seed < 0 or type(rank) is not int or
                type(world_size) is not int or not 0 <= rank < world_size):
            raise ValueError('Time sampling requires an explicit seed and valid rank partition')
        if (type(bin_seconds) not in (int, float) or not math.isfinite(bin_seconds) or bin_seconds <= 0 or
                not math.isclose(bin_seconds * 30, round(bin_seconds * 30), abs_tol=1e-9)):
            raise ValueError('Time bins must contain a positive whole number of 30Hz frames')
        if type(minimum_remaining_seconds) not in (int,float) or not math.isfinite(minimum_remaining_seconds) or minimum_remaining_seconds <= 0:
            raise ValueError('Minimum remaining duration must be positive')
        if (type(standing_start_probability) not in (int, float) or not math.isfinite(standing_start_probability)
                or not 0 <= standing_start_probability < 1):
            raise ValueError('Standing probability must lie in [0,1), retaining random motion tasks')
        self.catalog = catalog
        self.identity = dict(contract=TIME_SAMPLING_CONTRACT, catalog_identity=catalog.identity,
            seed=seed, rank=rank, world_size=world_size, bin_frames=round(bin_seconds * 30),
            minimum_remaining_seconds=float(minimum_remaining_seconds),
            standing_start_probability=float(standing_start_probability))
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, rank, 731]))
        self.bins, self.shards, self.orders = {}, {}, {}
        self.cursors, self.epochs = dict.fromkeys(SOURCES, 0), dict.fromkeys(SOURCES, 0)
        self.standing_draws = self.motion_draws = 0
        for index, source in enumerate(SOURCES):
            bins = []
            for sample_index, sample in enumerate(catalog.samples['train'][source]):
                maximum = max(0, sample['row']['num_frames'] - max(2, math.ceil(minimum_remaining_seconds * 30)))
                for begin in range(0, maximum + 1, self.identity['bin_frames']):
                    bins.append((sample_index, begin, min(begin + self.identity['bin_frames'], maximum + 1)))
            self.bins[source] = bins
            partition = np.random.default_rng(np.random.SeedSequence([seed, index, 947])).permutation(len(bins))
            self.shards[source] = partition[rank::world_size].tolist()
            if not self.shards[source]:
                raise ValueError(f'{source}: fewer legal time bins than ranks; cannot claim disjoint sampling')
            self.orders[source] = self.rng.permutation(self.shards[source]).tolist()

        from .adaptive_sampling import adaptive_settings, AdaptiveTimeWeights
        settings = adaptive_settings(adaptive)
        self.adaptive = AdaptiveTimeWeights(self.bins, settings) if settings['enabled'] else None
        self.adaptive_draws = 0
        if self.adaptive is not None:
            self.identity['adaptive'] = settings

    def draw(self, source):
        if source not in SOURCES:
            raise ValueError('Unknown time sampling source')
        if self.rng.random() < self.identity['standing_start_probability']:
            self.standing_draws += 1
            sample_index = int(self.rng.integers(len(self.catalog.samples['train'][source])))
            return sample_index, 0, dict(initialization='standing', time_bin=None)
        prioritized = self.adaptive is not None and self.rng.random() < self.adaptive.settings['fraction']
        if prioritized:
            # 自适应部分从全局bin池有放回抽样，各rank共享归并后的概率。
            bin_id = int(self.rng.choice(len(self.bins[source]), p=self.adaptive.probabilities(source)))
            self.adaptive_draws += 1
        else:
            if self.cursors[source] == len(self.orders[source]):
                self.orders[source] = self.rng.permutation(self.shards[source]).tolist()
                self.cursors[source] = 0
                self.epochs[source] += 1
            bin_id = self.orders[source][self.cursors[source]]
            self.cursors[source] += 1
        sample_index, begin, end = self.bins[source][bin_id]
        self.motion_draws += 1
        start = int(self.rng.integers(begin, end))
        return sample_index, start, dict(initialization=MOTION_RESET_CONTRACT,
            sampling_component='adaptive_with_replacement' if prioritized else 'coverage_without_replacement',
            time_bin=dict(source=source, bin_id=bin_id, begin_frame=begin, end_frame_exclusive=end,
                          rank=self.identity['rank'], epoch=self.epochs[source]))

    def state_dict(self):
        state = dict(identity=self.identity, rng=self.rng.bit_generator.state,
            orders=self.orders, cursors=self.cursors, epochs=self.epochs,
            standing_draws=self.standing_draws, motion_draws=self.motion_draws)
        if self.adaptive is not None:
            state.update(adaptive=self.adaptive.state_dict(), adaptive_draws=self.adaptive_draws)
        return copy.deepcopy(state)

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        if state.get('identity') != self.identity:
            raise ValueError('Time sampler identity changed; start a new training run')
        for source in SOURCES:
            order = state['orders'][source]
            if (any(type(v) is not int for v in order) or sorted(order) != sorted(self.shards[source])
                    or type(state['cursors'][source]) is not int
                    or not 0 <= state['cursors'][source] <= len(order)
                    or type(state['epochs'][source]) is not int or state['epochs'][source] < 0):
                raise ValueError('Invalid shared time-bin checkpoint permutation/cursor/epoch')
        for key in ('standing_draws', 'motion_draws'):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError('Invalid time sampling draw count')
        expected = sum(state['epochs'][s] * len(self.shards[s]) + state['cursors'][s] for s in SOURCES)
        if self.adaptive is not None:
            if type(state.get('adaptive_draws')) is not int or state['adaptive_draws'] < 0:
                raise ValueError('Invalid adaptive draw count')
            expected += state['adaptive_draws']
        if state['motion_draws'] != expected:
            raise ValueError('Time sampling draw count differs from bin cursors')
        rng = np.random.default_rng()
        rng.bit_generator.state = state['rng']
        self.rng = rng
        for key in ('orders', 'cursors', 'epochs', 'standing_draws', 'motion_draws'):
            setattr(self, key, state[key])
        if self.adaptive is not None:
            self.adaptive.load_state_dict(state['adaptive'])
            self.adaptive_draws = state['adaptive_draws']

    def observe(self, task, row=None, *, warmup_failure=False):
        if self.adaptive is not None:
            self.adaptive.observe(task, row, warmup_failure=warmup_failure)

    def sync_feedback(self, distributed):
        if self.adaptive is not None:
            self.adaptive.merge(distributed.all_gather_object(self.adaptive.pending))

    def summary(self):
        return dict(contract=TIME_SAMPLING_CONTRACT, rank=self.identity['rank'],
            adaptive=None if self.adaptive is None else self.adaptive.summary(), adaptive_tasks_drawn=self.adaptive_draws,
            standing_tasks_drawn=self.standing_draws, motion_tasks_drawn=self.motion_draws,
            scope='issued_start_bins_not_executed_coverage',
            per_source={source: dict(global_start_bins=len(self.bins[source]),
                assigned_start_bins=len(self.shards[source]), completed_sweeps=self.epochs[source],
                current_sweep_draws=self.cursors[source]) for source in SOURCES})


def execution_coverage(catalog, samplers):
    """按真实执行区间求本rank并集；跨rank不可相加冒充全局唯一覆盖。"""
    intervals, counts = {}, {s: dict(control_steps=0, upper_transitions=0, tasks_drawn=0) for s in SOURCES}
    for sampler in samplers:
        for source in SOURCES:
            for key in counts[source]:
                counts[source][key] += sampler._counts[source][key]
        for key, spans in sampler._intervals.items():
            merged = intervals.get(key, [])
            for begin, end in spans:
                merged = _merge_intervals(merged, begin, end)
            intervals[key] = merged
    result = {}
    total_transitions = sum(item['upper_transitions'] for item in counts.values())
    for source in SOURCES:
        total = covered = complete = seen = 0
        for sample in catalog.samples['train'][source]:
            ticks = sample['row']['num_frames'] * 20
            spans = intervals.get(f"{source}/{sample['row']['sample_id']}", [])
            length = sum(end - begin for begin, end in spans)
            total += ticks; covered += length; seen += bool(spans)
            complete += len(spans)==1 and spans[0][0]==0 and spans[0][1]>=ticks-11
        result[source] = dict(**counts[source], unique_executed_samples=seen,
            fully_covered_samples=complete, catalog_samples=len(catalog.samples['train'][source]),
            covered_seconds=covered / 600, source_seconds=total / 600,
            unique_time_coverage=covered / total,
            transition_share=counts[source]['upper_transitions'] / max(1, total_transitions))
    return dict(scope='rank_local_union_actual_actor_controls_excludes_warmup_MC_BC',
        source_duration='num_frames/30', full_sample_tail_tolerance_ticks=11, per_source=result)
