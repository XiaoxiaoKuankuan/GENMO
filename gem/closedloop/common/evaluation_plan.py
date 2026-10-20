"""公共固定音乐评价计划：来源均衡、group/audio去重与不可变任务身份。

从原评价器抽取相同选样算术，保留32任务协议和全量去重模式，不包含PPO漂移诊断、
优化器或训练状态。外部验证数据只用于评价，绝不提供AWR Replay样本。
"""
from __future__ import annotations
import copy
from collections.abc import Mapping
import numpy as np
from .evaluation import build_evaluation_tasks, _digest
from .full_dataset import SOURCES
from gem.closedloop.frozen_actor import stable_noise_seed
VERSION = 'genmo.closedloop.stage10.periodic_monitor.v1'
def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _plan_valid(plan):
    _require(plan.get('plan_sha256') == _digest({k: v for k, v in plan.items() if k != 'plan_sha256'}),
             'Periodic plan SHA differs')
    _require(plan.get('periodic_subset', {}).get('schema') == VERSION
             and plan.get('split') in ('val','test') and plan.get('training_buffer_allowed') is False,
             'Periodic evaluation requires an explicit held-out subset plan')
    tasks = plan['tasks']
    _require(len(tasks) == plan['task_count'] == plan['selected_sample_count'] * len(plan['seeds'])
             and len({t['task_id'] for t in tasks}) == len(tasks), 'Periodic tasks duplicate or omit seeds')
    for task in tasks:
        identity = {k: task[k] for k in ('dataset', 'sample_id', 'split', 'seed', 'music_start_frame')}
        _require(task['task_id'] == _digest(identity)[:24]
                 and task['noise_index_start'] == stable_noise_seed(int(task['seed']), _digest(identity)),
                 'Periodic task identity/noise differs')
    _require('periodic_partition' not in plan, 'Periodic helpers require the complete subset parent plan')
    nseed = len(plan['seeds'])
    _require(nseed > 0, 'Periodic plan has no seeds')
    for position in range(0, len(tasks), nseed):
        group = tasks[position:position+nseed]
        _require([t['seed'] for t in group] == plan['seeds']
                 and len({(t['dataset'], t['sample_id']) for t in group}) == 1,
                 'Periodic paired sample/seed ordering differs')
    counts=plan['periodic_subset']['source_sample_counts']
    _require(set(t['dataset'] for t in tasks) == set(SOURCES)
             and all(sum(t['dataset'] == source for t in tasks) == counts[source]*nseed for source in SOURCES),
             'Periodic parent does not contain the balanced four-source subset')

def build_balanced_plan(catalog, *, samples_per_source=4, seeds=(42, 1729), seconds=10., selection_seed=42,
                        start_mode='beginning',unique_music=False,split='val'):
    """在独立局部RNG中固定选样，复用全池任务的原task/noise身份。"""
    all_music=samples_per_source=='all_unique'
    per_source=isinstance(samples_per_source,Mapping)
    _require(all_music and unique_music or type(samples_per_source) is int and samples_per_source > 0
        or per_source and set(samples_per_source)==set(SOURCES)
        and all(type(n) is int and n>0 for n in samples_per_source.values()),
        'samples_per_source must be positive, an explicit four-source count map, or all_unique with unique_music')
    _require(split in ('val','test') and set(catalog.samples[split]) == set(SOURCES), 'Periodic plan requires four held-out sources')
    _require(start_mode in ('beginning','center','mixed'), 'Invalid periodic start_mode')
    _require(start_mode=='beginning' or seconds is not None, 'Complete song evaluation must begin at music start')
    full = build_evaluation_tasks(catalog, split=split, eval_count='all', seeds=seeds,
        selection_seed=selection_seed, start_mode='center' if start_mode=='center' else 'beginning', episode_seconds=seconds)
    rng = np.random.default_rng(selection_seed)
    chosen = set();source_counts={};seen_groups=set();seen_audio=set()
    for source in sorted(SOURCES):
        samples = sorted(catalog.samples[split][source], key=lambda s: str(s['row']['sample_id']))
        if unique_music:
            groups={}
            for sample in samples:groups.setdefault(sample['group_id'],[]).append(sample)
            samples=[min(group,key=lambda s:(-s['row']['num_frames'],s['row']['sample_id'])) for _,group in sorted(groups.items())]
            # 同一音频的不同动作片段不能充当新增的独立音乐。
            unique={}
            for sample in samples:unique.setdefault(sample['row']['source_audio_sha256'],sample)
            samples=[sample for sample in unique.values() if sample['group_id'] not in seen_groups
                     and sample['row']['source_audio_sha256'] not in seen_audio]
        count=len(samples) if all_music else samples_per_source[source] if per_source else samples_per_source
        _require(len(samples) >= count>0, f'Insufficient periodic held-out samples: {source}')
        source_counts[source]=count
        for index in sorted(rng.choice(len(samples), size=count, replace=False).tolist()):
            sample=samples[index]
            chosen.add((source, str(sample['row']['sample_id'])))
            seen_groups.add(sample['group_id']);seen_audio.add(sample['row']['source_audio_sha256'])
    plan = copy.deepcopy(full)
    tasks = [t for t in plan['tasks'] if (t['dataset'], t['sample_id']) in chosen]
    if start_mode=='mixed':
        center=build_evaluation_tasks(catalog,split=split,eval_count='all',seeds=seeds,
            selection_seed=selection_seed,start_mode='center',episode_seconds=seconds)
        centered={(t['dataset'],t['sample_id'],t['seed']):t for t in center['tasks']}
        counts={source:0 for source in SOURCES};mixed=[]
        for index in range(0,len(tasks),len(seeds)):
            group=tasks[index:index+len(seeds)];source=group[0]['dataset']
            mixed.extend(centered[(t['dataset'],t['sample_id'],t['seed'])] if counts[source]%2 else t for t in group)
            counts[source]+=1
        tasks=mixed
    samples = [t['sample'] for t in tasks[::len(full['seeds'])]]
    plan.update(tasks=tasks, selected_sample_count=len(chosen), task_count=len(tasks), requested_eval_count=len(chosen),
        selected_unique_group_count=len({s['group_id'] for s in samples}),
        selected_unique_audio_count=len({s['row']['source_audio_sha256'] for s in samples}),
        selection_rule='fixed_source_balanced_unique_music' if unique_music else 'fixed_source_balanced_without_replacement',
        start_mode=start_mode,
        periodic_subset=dict(schema=VERSION, samples_per_source=samples_per_source,
            source_sample_counts=source_counts,
            full_pool_parent_plan_sha256=full['plan_sha256'], full_heldout_acceptance=False))
    if unique_music:plan['selection_unit']='unique_music_group_and_audio_per_source'
    if seconds is None:plan['complete_music_scope']='full_available_paired_sample_not_uncropped_original_recording'
    plan['plan_sha256'] = _digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    _plan_valid(plan)
    return plan
