"""连续滚动 RoboFFT 的执行相关时间权重及来源审计。

输入为本轮真实转移中的 compact_execution_provenance、每条计划的生成起点、
已知前缀和实际到达时间。先按物理环境/episode合并来源账本，沿父计划追踪被
复制的已知前缀，分别统计原生成帧被GMT当前中心和21点窗口使用的情况。实际
窗口包含前瞻和回看，均属于控制器输入；相邻源帧支持不冒充动力学因果导数。

已观察到执行窗口的自由帧权重为1，随后0.5秒为0.3，更远为0.1。跨rollout尾部
尚未观察完整寿命的计划，依据本条计划前缀、提交/到达时刻、实际决策间隔及
GMT前瞻预测近期范围，明确标记预测而不宣称已消费。已知/padding坐标永不训练。
时间权重按自由坐标总质量归一化，使常数误差下与原整窗损失尺度相同；权重在
采集后固定，新旧代理及全部epoch/MC共用。本模块不改采样核、参考、奖励或GAE。
"""
from __future__ import annotations

from collections import defaultdict
import math
import torch

from ..dppo.execution_provenance import SCHEMA, consumption_report

VERSION = 'genmo.robofft.execution_frame_weights.v1'


def validate_settings(settings):
    if settings is None:
        return
    if settings.get('version') != VERSION:
        raise ValueError('Unknown execution-weighted RoboFFT objective')
    expected = dict(near_weight=1., middle_weight=.3, far_weight=.1,
                    tail_seconds=.5, lookahead_ticks=120, frame_ticks=20)
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError('Execution weighting requires explicit 1/.3/.1 and actual 600Hz/30Hz/GMT clock')


def normalize_frame_weights(raw, free_mask):
    """按有效自由坐标质量归一；全一权重严格恢复原求和目标。"""
    raw = torch.as_tensor(raw, dtype=torch.float64, device=free_mask.device).detach()
    if raw.shape != free_mask.shape[:-1] or not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Finite nonnegative frame weights must align with free coordinates')
    mass = free_mask.double().sum(-1)
    total = mass.sum(-1, keepdim=True)
    weighted = (mass * raw).sum(-1, keepdim=True)
    if ((total > 0) & (weighted <= 0)).any():
        raise ValueError('Nonempty free action cannot have zero weight mass')
    return torch.where(mass > 0, raw * total / weighted.clamp_min(1e-30), 0.).detach()


def predicted_weights(free_mask, *, prefix_frames=None, decision_tick=0, arrival_tick=None,
                      interval_ticks=180, settings):
    """证据不足时的有标签预测；前缀由真实自由掩码决定，不固定取前15帧。"""
    validate_settings(settings)
    frames = free_mask.shape[-2]
    free_frames = torch.nonzero(free_mask.any(-1), as_tuple=False).flatten().tolist()
    raw = torch.full((frames,), settings['far_weight'], dtype=torch.float64)
    if not free_frames:
        return raw
    prefix = free_frames[0] if prefix_frames is None else int(prefix_frames)
    arrival = decision_tick + prefix * 20 if arrival_tick is None else int(arrival_tick)
    # GMT未来窗口与速度差分还会读取相邻源帧，预测范围保留一个源帧支持。
    high_end = math.ceil((arrival - decision_tick + max(int(interval_ticks), 12)
                         + settings['lookahead_ticks']) / 20) + 1
    high_end = min(frames - 1, max(prefix, high_end))
    raw[prefix:high_end + 1] = settings['near_weight']
    raw[high_end + 1:min(frames, high_end + 1 + round(settings['tail_seconds'] * 30))] = settings['middle_weight']
    return raw


def merged_consumption(rows):
    """合并真实来源版本；未提交到已提交可推进，固定来源字段不得冲突。"""
    groups = defaultdict(list)
    for row in rows:
        p = row.metadata.get('compact_execution_provenance')
        if p is not None:
            key = tuple(row.identity.get(k) for k in ('backend_session_id', 'env_id', 'episode_id'))
            groups[key].append(p)
    results = {}
    for key, records in groups.items():
        origins, controls = {}, []
        for p in records:
            if p.get('schema') != SCHEMA:
                raise ValueError('Execution weights require current compact provenance')
            for plan, item in p['plans'].items():
                previous = origins.setdefault(plan, {})
                for field, value in item.items():
                    if value is None:
                        continue
                    if field in previous and previous[field] != value:
                        raise ValueError('Conflicting immutable reference origin: ' + field)
                    previous[field] = value
            for step in p['steps']:
                step = dict(step)
                if 'consumed_plan_indices' in step:
                    step['consumed_plan_ids'] = [p['plan_dictionary'][i] for i in step.pop('consumed_plan_indices')]
                controls.append(step)
        # 单份规范账本避免每个转移复制所有计划；缺失历史由原审计报告明确保留。
        report = consumption_report([dict(schema=SCHEMA, plans=origins, steps=controls)])
        end = max((r['control_tick_end'] for r in controls), default=0)
        results[key] = (report, end)
    return results


def weights_for_rows(rows, settings):
    validate_settings(settings)
    merged = merged_consumption(rows)
    weights, audits = [], []
    for row in rows:
        free = row.free_mask.detach().cpu()
        key = tuple(row.identity.get(k) for k in ('backend_session_id', 'env_id', 'episode_id'))
        report, end = merged.get(key, ({'plans': {}, 'source_lineage_complete': False}, 0))
        plan = row.identity.get('plan_id')
        item = report['plans'].get(plan, {})
        begin = int(item.get('decision_tick', getattr(row, 'control_tick_begin', 0)))
        interval = int(getattr(row, 'control_tick_end', begin + 180)) - begin
        arrival = item.get('commit_tick')
        if arrival is None:
            arrival = item.get('generation_ready_tick')
        predicted = predicted_weights(free, prefix_frames=item.get('prefix_frames'), decision_tick=begin,
            arrival_tick=arrival, interval_ticks=interval, settings=settings)
        current = item.get('current_generated_frame_weight', [0.] * len(free))
        window = item.get('window_generated_frame_weight', [0.] * len(free))
        observed = torch.tensor(window, dtype=torch.float64) > 0
        # 至少再观察一个决策周期和前瞻，才能把本轮未见帧称为低观察帧。
        complete = bool(item.get('commit_tick') is not None and end >= int(item['commit_tick']) + max(interval, 12) + 140)
        raw = torch.full((len(free),), settings['far_weight'], dtype=torch.float64)
        if observed.any() and complete:
            last = int(torch.nonzero(observed).max())
            raw[last + 1:min(len(free), last + 16)] = settings['middle_weight']
            raw[observed] = settings['near_weight']
            scope = 'observed_GMT_window_in_retained_range'
        else:
            raw = predicted
            raw[observed] = settings['near_weight']
            scope = 'predicted_boundary_or_uncommitted_with_observed_union'
        normalized = normalize_frame_weights(raw, free)
        free_frames = free.any(-1)
        # 后继提交中的已知前缀仍沿父计划追溯；只有新生成区覆盖的自由帧列作覆盖。
        children = [child for child in report['plans'].values()
                    if child.get('parent_plan_id') == plan and child.get('commit_tick') is not None
                    and child.get('prefix_frames') is not None]
        replaced = [i for i in range(len(free)) if free_frames[i] and not observed[i]
                    and any(begin + i*20 >= max(int(child['commit_tick']),
                        int(child['decision_tick'])+int(child['prefix_frames'])*20) for child in children)]
        audit = dict(version=VERSION, scope=scope, plan_id=plan,
            source_lineage_complete=report.get('source_lineage_complete', False),
            current_generated_frames=[i for i, v in enumerate(current) if v > 0],
            window_generated_frames=[i for i, v in enumerate(window) if v > 0],
            window_only_generated_frames=[i for i, v in enumerate(window) if v > 0 and current[i] == 0],
            unobserved_free_frames_overwritten_by_committed_child=replaced,
            unobserved_free_frames=[i for i in range(len(free)) if free_frames[i] and not observed[i]],
            near_free_frames=int(((raw == 1.) & free_frames).sum()),
            middle_free_frames=int(((raw == .3) & free_frames).sum()),
            far_free_frames=int(((raw == .1) & free_frames).sum()),
            raw_weight_mass=float((raw * free.double().sum(-1)).sum()),
            normalized_weight_mass=float((normalized * free.double().sum(-1)).sum()),
            free_coordinate_count=int(free.sum()),
            unobserved_is_not_proof_of_never_consumed=True)
        weights.append(normalized)
        audits.append(audit)
    return weights, audits
