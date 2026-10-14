"""为三模型完整音乐对照补充每任务全新PhysX世界的初态控制实验。

原完整面板使用正式训练的逐episode重置协议；长期复用世界时，站立预热后的
首个实际状态存在毫弧度级差异。本工具从原冻结任务计划中，预先按每个split、
每个来源选最长音乐和第一个固定seed，共8组，不按模型得分挑选。每个rank仅
执行一首，三个模型分别由新的八卡进程启动，从而既没有上一首轨迹残留，也
没有上一模型在同一PhysX世界中的历史。8组复核不充当新增独立音乐数量。

仅显式替换评估入口的任务分片函数，所有模型、真实物理、奖励、20步随机行为
及原0.30秒部署合同沿用原核验入口。原完整评估源码不修改，分片扩展SHA与父
计划身份写入新实验spec。prepare只读制作计划；run每次只评估指定子目录的一
个模型，由调用者顺序启动，禁止与原GPU评估并行争卡。仍不启动训练或优化器。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.evaluation import _digest,_publish_json
from gem.closedloop.dppo.training_checkpoint import file_sha256
from tools.eval import run_stage10_group_comparison as runner


def single_task_partitions(plan,world_size):
    if world_size!=8:raise ValueError('Exactly eight fresh worlds required')
    mapping=plan['fresh_world_task_ranks']
    if set(mapping)!={t['task_id'] for t in plan['tasks']} or len(set(mapping.values()))!=len(mapping):
        raise ValueError('A fresh world may receive only one task')
    children=[]
    for rank in range(world_size):
        child=copy.deepcopy(plan);child.pop('plan_sha256')
        child['tasks']=[t for t in child['tasks'] if mapping[t['task_id']]==rank]
        child.update(task_count=len(child['tasks']),selected_sample_count=len(child['tasks']),
            selected_unique_group_count=len(child['tasks']),parent_plan_sha256=plan['plan_sha256'],
            group_partition=dict(rank=rank,world_size=world_size,world_lifecycle='fresh_process_one_task'))
        child['plan_sha256']=_digest(child);children.append(child)
    if sum(c['task_count'] for c in children)!=len(plan['tasks']):raise ValueError('Invalid assigned rank')
    return children


def prepare(parent,output):
    parent=parent.resolve(strict=True);spec=json.loads((parent/'comparison_spec.json').read_text())
    if spec['smoke']:raise ValueError('A complete-song parent is required')
    plans={};assignment=[];rank=0
    for split in ('test','val'):
        item=spec['plans'][split];path=parent/item['file']
        if file_sha256(path)!=item['sha256']:raise ValueError('Parent plan SHA mismatch')
        original=json.loads(path.read_text());child=copy.deepcopy(original);child.pop('plan_sha256')
        tasks=[];mapping={}
        for source in sorted({t['dataset'] for t in original['tasks']}):
            candidates=[t for t in original['tasks'] if t['dataset']==source and t['seed']==min(original['seeds'])]
            task=sorted(candidates,key=lambda t:(-t['full_music_seconds'],t['sample_id']))[0]
            tasks.append(task);mapping[task['task_id']]=rank
            assignment.append(dict(rank=rank,task=task));rank+=1
        child.update(tasks=tasks,seeds=[min(original['seeds'])],task_count=4,selected_sample_count=4,
            selected_unique_group_count=4,selected_unique_audio_count=4,selected_group_counts_by_source={s:1 for s in sorted({t['dataset'] for t in tasks})},
            parent_complete_plan_sha256=original['plan_sha256'],fresh_world_task_ranks=mapping,
            requested_eval_count='one_longest_music_per_source_split_first_seed',
            selection_rule='longest_music_per_source_split_before_control_results_tie_sample_id',
            world_lifecycle='new_process_and_PhysX_world_for_each_model_each_task')
        child['plan_sha256']=_digest(child);single_task_partitions(child,8);plans[split]=child
    if rank!=8:raise ValueError('Exactly eight music groups required')
    output.mkdir(parents=True,exist_ok=False)
    for model in spec['models']:
        directory=output/model['label'];directory.mkdir()
        _publish_json(directory/'catalog_audit.json',json.loads((parent/'catalog_audit.json').read_text()))
        for split,plan in plans.items():_publish_json(directory/(split+'_plan.json'),plan)
        child=copy.deepcopy(spec);child.pop('comparison_id');child.update(
            schema='genmo.stage10.fresh_world_control.v1',models=[model],
            source_comparison_id=spec['comparison_id'],source_comparison_spec_sha256=file_sha256(parent/'comparison_spec.json'),
            plans={s:dict(file=s+'_plan.json',sha256=file_sha256(directory/(s+'_plan.json')),groups=4,tasks=4) for s in plans},
            control_scope='eight_existing_music_groups_fresh_world_per_task_no_prior_episode_or_model',
            seeds=plans['test']['seeds'],rank_assignments=assignment)
        child['evaluation_extension'][str(Path(__file__).relative_to(ROOT))]=file_sha256(__file__)
        child['comparison_id']=_digest(child);_publish_json(directory/'comparison_spec.json',child)
    _publish_json(output/'selection.json',dict(source_comparison_id=spec['comparison_id'],assignments=assignment,
        expected_episodes=24,additional_independent_groups=0,reason='measure_and_remove_post_reset_world_history_confound'))
    print(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.parent is None:parser.error('--parent required')
        prepare(args.parent,args.output);return 0
    # 依赖注入只限本次独立评估进程；物理/概率/奖励/训练实现保持原样。
    runner.partition_by_music=single_task_partitions
    return runner.run_worker(args)


if __name__=='__main__':raise SystemExit(main())
