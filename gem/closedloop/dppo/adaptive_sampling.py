"""来源内困难片段采样、跨卡反馈及有证据的短片段隔离。

沿用审计过的1秒起点bin和固定rank分片。每轮只同步新增的真实bin访问记录，
以物理失败率和跟踪分数的EMA构造难度；均匀分量、权重上限和邻域平滑避免
少数失败垄断训练。来源概率保持由FullMusicSampler决定，不改GAE或概率核。
一个episode在同一bin的连续控制只算一次访问，跨PPO窗口未完成的访问暂存，
窗口边界不算失败。warmup失败另记为真实失败，不能冒充训练后的跟踪结果。

只有跨多个采集轮、充分访问后持续失败，且姿态联合风险或长期无改善证据满足
阈值，才隔离对应短bin。新任务起点须保证20秒物理窗口不穿过隔离段；不删除
数据或固定评估歌曲。隔离是训练预算决策，不是动作绝对不可行的证明。
状态、计数、EMA、隔离理由、待结算访问和RNG均进入checkpoint，恢复身份严格。
只使用CPU数组和采集后的对象通信，不推进仿真，也不在某个rank私自发起collective。
"""
from __future__ import annotations

import copy
import math
import numpy as np

from .full_dataset import SOURCES
from .time_sampling import StratifiedTimePool, MOTION_RESET_CONTRACT

CONTRACT = 'rank_shared_adaptive_time_bins.v1'


def validate_adaptive(config):
    required = {'ema_alpha', 'uniform_fraction', 'maximum_weight', 'failure_weight',
        'neighbor_radius', 'minimum_visits', 'minimum_rounds', 'failure_rate_threshold',
        'severe_fraction', 'unlearnable_visits', 'unlearnable_rounds', 'minimum_improvement',
        'episode_seconds', 'excluded_segments'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Explicit adaptive sampling and quarantine settings required')
    for name in ('ema_alpha', 'uniform_fraction', 'failure_weight', 'failure_rate_threshold', 'severe_fraction'):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]) or not 0 < config[name] < 1:
            raise ValueError('Adaptive probability parameters must be finite and in (0,1)')
    for name in ('neighbor_radius', 'minimum_visits', 'minimum_rounds', 'unlearnable_visits', 'unlearnable_rounds'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError('Adaptive count thresholds must be positive integers')
    if (not 1 <= config['maximum_weight'] <= 100 or config['episode_seconds'] <= 0
            or not 0 <= config['minimum_improvement'] < 1
            or config['unlearnable_visits'] < config['minimum_visits']
            or config['unlearnable_rounds'] < config['minimum_rounds']):
        raise ValueError('Invalid adaptive cap, episode length or quarantine evidence thresholds')
    if not isinstance(config['excluded_segments'], list):
        raise ValueError('Excluded segments must be an explicit auditable list')
    for item in config['excluded_segments']:
        if (set(item) != {'source', 'sample_id', 'begin_seconds', 'end_seconds', 'reason', 'evidence'}
                or item['source'] not in SOURCES or not item['reason'] or not item['evidence']
                or not 0 <= item['begin_seconds'] < item['end_seconds']):
            raise ValueError('Exclusion requires exact sample, interval, reason and evidence')


class AdaptiveTimePool(StratifiedTimePool):
    def __init__(self, catalog, *, adaptive, **kwargs):
        validate_adaptive(adaptive)
        super().__init__(catalog, **kwargs)
        if self.identity['standing_start_probability'] != 0:
            raise ValueError('Adaptive motion pool currently requires no standing resets')
        self.config = copy.deepcopy(adaptive)
        self.identity.update(contract=CONTRACT, adaptive=self.config)
        self.statistics = {}
        self.pending = {}
        self.warmup_events = []
        self.round = 0
        self.quarantine = copy.deepcopy(adaptive['excluded_segments'])
        self._lookup = {}
        self._sample_bins = {}
        self._probabilities = {}
        # 观察bin覆盖整首，起点bin另受最少剩余时长限制。否则歌曲最后8秒的
        # 真实失败会被错误归到最后一个可抽起点，隔离区间也会偏移。
        self._maximum_start = {}
        for source_index, source in enumerate(SOURCES):
            self._lookup[source] = {s['row']['sample_id']: i for i, s in enumerate(catalog.samples['train'][source])}
            self._maximum_start[source] = {}
            bins = []
            for index, sample in enumerate(catalog.samples['train'][source]):
                frames = sample['row']['num_frames']
                self._maximum_start[source][index] = max(0, frames-max(2, math.ceil(self.identity['minimum_remaining_seconds']*30)))
                bins.extend((index, begin, min(begin+self.identity['bin_frames'], frames))
                    for begin in range(0, frames, self.identity['bin_frames']))
            self.bins[source] = bins
            partition = np.random.default_rng(np.random.SeedSequence([self.identity['seed'], source_index, 947])).permutation(len(bins))
            self.shards[source] = [int(i) for i in partition[self.identity['rank']::self.identity['world_size']]
                if bins[i][1] <= self._maximum_start[source][bins[i][0]]]
            by_sample = {}
            for bin_id, (index, begin, end) in enumerate(self.bins[source]):
                by_sample.setdefault(index, []).append(bin_id)
            self._sample_bins[source] = by_sample
        for item in self.quarantine:
            if item['sample_id'] not in self._lookup[item['source']]:
                raise ValueError('Configured excluded segment is not in the audited train catalog')
        self._recompute()

    def _key(self, source, bin_id):
        return source + '/' + str(bin_id)

    def _bin(self, source, sample_id, frame):
        ids = self._sample_bins[source][self._lookup[source][sample_id]]
        return ids[min(max(0, int(frame)//self.identity['bin_frames']), len(ids)-1)]

    def _blocked(self, source, bin_id):
        sample_index, begin, end = self.bins[source][bin_id]
        sample_id = self.catalog.samples['train'][source][sample_index]['row']['sample_id']
        episode_frames = math.ceil(self.config['episode_seconds']*30)
        return any(item['source'] == source and item['sample_id'] == sample_id
            and begin < item['end_seconds']*30 and end+episode_frames > item['begin_seconds']*30
            for item in self.quarantine)

    def _recompute(self):
        for source in SOURCES:
            # 平滑局限在同一首音乐内部，绝不跨歌曲索引边界传播失败。
            difficulty = np.zeros(len(self.bins[source]), dtype=np.float64)
            for bin_id in range(len(difficulty)):
                stat = self.statistics.get(self._key(source, bin_id))
                if stat:
                    a = self.config['failure_weight']
                    difficulty[bin_id] = a*stat['failure_ema']+(1-a)*(1-stat['track_ema'])
            smooth = difficulty.copy()
            for ids in self._sample_bins[source].values():
                values = difficulty[ids]
                for distance in range(1, self.config['neighbor_radius']+1):
                    if len(ids) <= distance:
                        continue
                    # 失败前的准备区更常被抽到，不把平滑当成额外失败次数。
                    smooth[np.asarray(ids[:-distance])] = np.maximum(
                        smooth[np.asarray(ids[:-distance])], values[distance:]*(.5**distance))
            eligible = [i for i in self.shards[source] if not self._blocked(source, i)]
            if not eligible:
                raise ValueError(f'{source}: quarantine exhausted this rank; explicit dataset review required')
            weight = 1+(self.config['maximum_weight']-1)*smooth[eligible].clip(0, 1)
            uniform = self.config['uniform_fraction']
            probability = uniform/len(eligible)+(1-uniform)*weight/weight.sum()
            self._probabilities[source] = (np.asarray(eligible), probability)

    def draw(self, source):
        eligible, probabilities = self._probabilities[source]
        bin_id = int(self.rng.choice(eligible, p=probabilities))
        sample_index, begin, end = self.bins[source][bin_id]
        self.motion_draws += 1
        end = min(end, self._maximum_start[source][sample_index]+1)
        return sample_index, int(self.rng.integers(begin, end)), dict(initialization=MOTION_RESET_CONTRACT,
            time_bin=dict(source=source, bin_id=bin_id, begin_frame=begin, end_frame_exclusive=end,
                rank=self.identity['rank'], adaptive_round=self.round),
            sampling_contract=CONTRACT, exclusion_count=len(self.quarantine))

    def feedback(self, rows):
        """提取本rank新增bin访问；PPO边界的未完访问留到下一轮，不当成失败。"""
        events = self.warmup_events
        self.warmup_events = []
        for row in rows:
            task = row.metadata['training_task']
            source, sample_id = task['dataset'], task['sample_id']
            slot = str(row.metadata['collector_env_slot'])
            episode = str(row.identity['episode_id'])
            details = row.metadata.get('reward_details', [])
            for detail in details:
                # 训练明细只保留真实GPU跟踪分数；审计步仍有完整components。
                # 不从综合奖励反推分数，也不把字段缺失默认为零。
                score = float(detail['tracking_score'] if 'tracking_score' in detail
                    else detail['components']['track']['score'])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('adaptive feedback requires a finite tracking score in [0,1]')
                frame = task['music_start_frame']+max(0, int(detail['tick'])-600)//20
                bin_id = self._bin(source, sample_id, frame)
                current = self.pending.get(slot)
                if current and (current['episode'] != episode or current['source'] != source or current['bin_id'] != bin_id):
                    events.append(self.pending.pop(slot))
                    current = None
                if current is None:
                    current = dict(source=source, bin_id=bin_id, episode=episode, controls=0,
                        track_sum=0., failure=False, severe=False)
                    self.pending[slot] = current
                current['controls'] += 1
                current['track_sum'] += score
            failed = bool(row.metadata.get('terminal_snapshot', {}).get('terminated', False))
            physical_end = failed or row.metadata.get('physical_time_limit_reached', False) or row.reason == 'music_end'
            if physical_end and slot in self.pending:
                current = self.pending.pop(slot)
                current['failure'] = failed
                current['severe'] = failed and bool(row.metadata.get('feasibility', {}).get(
                    'terminal', {}).get('combined_feasibility_risk', False))
                events.append(current)
        return events

    def record_warmup_failure(self, task, feasibility):
        """预热失败按实际起点另记，score不冒充有效策略跟踪分数。"""
        source = task['sample']['dataset']
        sample_id = task['sample']['row']['sample_id']
        self.warmup_events.append(dict(source=source,
            bin_id=self._bin(source, sample_id, task['music_start_frame']),
            episode='warmup', controls=1, track_sum=0., failure=True, warmup=True,
            severe=bool(feasibility.get('terminal', {}).get('combined_feasibility_risk', False))))

    def update(self, events):
        """所有rank传入同一份本轮全局新增事件；只EMA更新观察到的bin。"""
        self.round += 1
        grouped = {}
        for event in events:
            if event['controls'] <= 0:
                continue
            key = self._key(event['source'], event['bin_id'])
            grouped.setdefault(key, []).append(event)
        newly = []
        for key, visits in grouped.items():
            stat = self.statistics.setdefault(key, dict(visits=0, failures=0, severe_failures=0,
                observed_rounds=0, tracking_observed_rounds=0, warmup_failures=0, failure_ema=0., track_ema=1., recent_track=[], first_track=[]))
            failures = sum(v['failure'] for v in visits)
            tracking_visits=[v for v in visits if not v.get('warmup')]
            track = (sum(v['track_sum']/v['controls'] for v in tracking_visits)/len(tracking_visits)
                if tracking_visits else stat['track_ema'])
            a = self.config['ema_alpha'] if stat['visits'] else 1.
            stat['failure_ema'] = (1-a)*stat['failure_ema']+a*failures/len(visits)
            stat['track_ema'] = (1-a)*stat['track_ema']+a*track
            stat['visits'] += len(visits);stat['failures'] += failures
            stat['severe_failures'] += sum(v['severe'] for v in visits)
            stat['observed_rounds'] += 1
            stat['warmup_failures'] += sum(bool(v.get('warmup')) for v in visits)
            if tracking_visits:
                stat['tracking_observed_rounds'] += 1
                if len(stat['first_track']) < 4:stat['first_track'].append(track)
                stat['recent_track'] = (stat['recent_track']+[track])[-8:]
            rate = stat['failures']/stat['visits']
            enough = (stat['visits'] >= self.config['minimum_visits']
                and stat['observed_rounds'] >= self.config['minimum_rounds']
                and rate >= self.config['failure_rate_threshold'])
            severe = stat['severe_failures']/max(1, stat['failures']) >= self.config['severe_fraction']
            improvement = (float(np.mean(stat['recent_track'])-np.mean(stat['first_track']))
                if stat['recent_track'] else None)
            unlearnable = (stat['visits'] >= self.config['unlearnable_visits']
                and stat['tracking_observed_rounds'] >= self.config['unlearnable_rounds']
                and improvement is not None and improvement <= self.config['minimum_improvement'])
            warmup_unlearnable = (stat['warmup_failures'] >= self.config['unlearnable_visits']
                and stat['observed_rounds'] >= self.config['unlearnable_rounds']
                and stat['warmup_failures']/stat['visits'] >= self.config['failure_rate_threshold'])
            if enough and (severe or unlearnable or warmup_unlearnable):
                source = visits[0]['source'];bin_id = visits[0]['bin_id']
                sample_index, begin, end = self.bins[source][bin_id]
                sample_id = self.catalog.samples['train'][source][sample_index]['row']['sample_id']
                if not any(q['source'] == source and q['sample_id'] == sample_id and
                           q['begin_seconds'] <= begin/30 < q['end_seconds'] for q in self.quarantine):
                    item = dict(source=source, sample_id=sample_id, begin_seconds=begin/30, end_seconds=end/30,
                        reason=('repeated_physical_failure_with_pose_risk' if severe else
                            'persistent_warmup_failure' if warmup_unlearnable else 'persistent_failure_without_tracking_improvement'),
                        evidence=dict(round=self.round, **copy.deepcopy(stat), tracking_improvement=improvement))
                    self.quarantine.append(item);newly.append(item)
        self._recompute()
        return dict(global_new_bin_visits=sum(len(v) for v in grouped.values()),
            updated_bins=len(grouped), newly_quarantined=newly, round=self.round)

    def state_dict(self):
        return copy.deepcopy(dict(identity=self.identity, rng=self.rng.bit_generator.state,
            statistics=self.statistics, pending=self.pending, warmup_events=self.warmup_events, round=self.round, quarantine=self.quarantine,
            motion_draws=self.motion_draws))

    def load_state_dict(self, state):
        if state.get('identity') != self.identity:
            raise ValueError('Adaptive sampler identity changed; explicit new experiment required')
        state = copy.deepcopy(state)
        rng = np.random.default_rng();rng.bit_generator.state = state['rng']
        if type(state['round']) is not int or state['round'] < 0 or type(state['motion_draws']) is not int or state['motion_draws'] < 0:
            raise ValueError('Invalid adaptive checkpoint round or draws')
        for key, stat in state['statistics'].items():
            source, index = key.rsplit('/', 1)
            if source not in SOURCES or not 0 <= int(index) < len(self.bins[source]):
                raise ValueError('Adaptive checkpoint references unknown bin')
            if (not 0 <= stat['failures'] <= stat['visits'] or not 0 <= stat['severe_failures'] <= stat['failures']
                    or not 0 <= stat['failure_ema'] <= 1 or not 0 <= stat['track_ema'] <= 1):
                raise ValueError('Invalid adaptive failure/score statistics')
        candidate=copy.copy(self)
        candidate._probabilities={}
        candidate.rng = rng
        for name in ('statistics', 'pending', 'warmup_events', 'round', 'quarantine', 'motion_draws'):
            setattr(candidate, name, state[name])
        for item in candidate.quarantine:
            if item['source'] not in SOURCES or item['sample_id'] not in self._lookup[item['source']]:
                raise ValueError('Unknown quarantine sample in checkpoint')
            if not 0 <= item['begin_seconds'] < item['end_seconds'] or not item['reason'] or not item['evidence']:
                raise ValueError('Invalid quarantine evidence in checkpoint')
        candidate._recompute()
        self.__dict__.update(candidate.__dict__)

    def summary(self):
        return dict(contract=CONTRACT, rank=self.identity['rank'], round=self.round,
            motion_tasks_drawn=self.motion_draws, observed_bins=len(self.statistics),
            scope='source_conditional_adaptive_start_bins_not_physical_sample_multiplier',
            uniform_fraction=self.config['uniform_fraction'], maximum_weight=self.config['maximum_weight'],
            quarantine=copy.deepcopy(self.quarantine),
            per_source={s: dict(assigned_start_bins=len(self.shards[s]), eligible_start_bins=len(ids),
                excluded_start_bins=len(self.shards[s])-len(ids),
                maximum_probability=float(p.max()), effective_bins=float(1/np.square(p).sum()),
                maximum_probability_over_uniform=float(p.max()*len(ids)))
                for s, (ids, p) in self._probabilities.items()})
