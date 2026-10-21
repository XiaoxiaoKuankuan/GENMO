"""汇总同起点完整链PPO批量实验及旧真实GMT消费支持上的反事实参考变化。

输入为compare_stage2_optimizer_batches输出和可选的实际中心消费来源gzip。读取
三方案真实整批目标、四步联合KL、Adam方向与同噪声探针；消费记录递归追溯后的
原计划/源帧/插值权重由现有执行来源分析器提供，本工具不假设前15帧被执行。
只有某个中心时刻的全部真实源计划均有探针时才计算该时刻的关节插值增量；缺少
来源只计缺失，不把未知增量补成零。共享普通探针单列，避免高KL尾部选择不同污染比较。

所有动作差异都固定旧条件与噪声，属于反事实参考，不能声称新机器人执行改善。
冻结末步前后比较采用normalized坐标；canonical root不直接解释为世界XYZ。
此工具不改变采样概率、Actor、优化器、奖励或物理状态，也不自动开始/停止训练。
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from gem.closedloop.dppo.budget import atomic_json


def batch_summary(result):
    report=result['actor'];history=report['full_rollout_objective_history']
    steps=report['steps'];first=history[0]['objective_mean'];last=history[-1]['objective_mean']
    return dict(branch=result['branch'],minibatch_chains=result['minibatch_chains'],
        valid_chains=result['effective_chains'],accepted_steps=report['optimizer_steps'],
        planned_steps=report['planned_optimizer_steps'],actual_adam_calls=report['attempted_optimizer_steps'],
        objective_aware=result['objective_aware'],initial_objective=first,final_objective=last,
        objective_gain=last-first,joint_kl=result['final_kl']['chain_joint_kl'],
        active_clip=result['final_kl']['active_clip_fraction'],
        learning_rates=[step['learning_rate'] for step in steps],
        gradient_direction=[dict(step=step['optimizer_step'],
            actual_gain=step['full_rollout_after']['objective_mean']-step['full_rollout_before']['objective_mean'],
            **step['full_gradient_direction']) for step in steps if step.get('full_gradient_direction')],
        probe=result['output']['distributions'],wall_seconds=result['wall_seconds'],
        peak_allocated_bytes=result['peak_allocated_bytes'])


def consume_reference_changes(offline,stream):
    candidates={}
    for directory in offline.iterdir():
        if not (directory/'result.json').is_file():continue
        probes={};ordinary=set()
        for path in sorted(directory.glob('probe_rank*.pt')):
            item=torch.load(path,map_location='cpu',weights_only=False)
            before,after=item['before'],item['after']
            for i,identity in enumerate(before['identities']):
                plan=identity['plan_id']
                probes[plan]=(after['qpos'][i,:,7:].double()-before['qpos'][i,:,7:].double())
                if i<4:ordinary.add(plan)
        candidates[directory.name]=dict(probes=probes,ordinary=ordinary,
            full_control_count=0,full_energy=0.,ordinary_control_count=0,ordinary_energy=0.,
            touched_controls_without_all_plan_probes=0)
    with gzip.open(stream,'rt') as handle:
        for line in handle:
            sources=json.loads(line)['pose_source']
            if not sources:continue
            for item in candidates.values():
                plans={source['plan_id'] for source in sources}
                available=plans & item['probes'].keys()
                if not available:continue
                if len(available)!=len(plans):
                    item['touched_controls_without_all_plan_probes']+=1
                    continue
                delta=sum(source['weight']*item['probes'][source['plan_id']][source['frame']] for source in sources)
                energy=float(delta.square().mean())
                item['full_control_count']+=1;item['full_energy']+=energy
                if plans<=item['ordinary']:
                    item['ordinary_control_count']+=1;item['ordinary_energy']+=energy
    return {name:dict(covered_center_controls=item['full_control_count'],
        joint_rms_rad=math.sqrt(item['full_energy']/item['full_control_count']) if item['full_control_count'] else None,
        common_ordinary_center_controls=item['ordinary_control_count'],
        common_ordinary_joint_rms_rad=math.sqrt(item['ordinary_energy']/item['ordinary_control_count']) if item['ordinary_control_count'] else None,
        incomplete_probed_support_controls=item['touched_controls_without_all_plan_probes'],
        scope='old_actual_center_pose_interpolation_support_same_old_conditions_counterfactual_joint_reference')
        for name,item in candidates.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--center-support',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=json.loads(args.comparison.read_text())
    results=[batch_summary(item) for item in source['results']]
    if not all(item['objective_aware'] for item in results):raise ValueError('Acceptance rules differ; this is not a batch-only comparison')
    digest={item['same_initial_state_digest'] for item in source['results']}
    if len(digest)!=1:raise ValueError('Initial Actor/Adam/BC/RNG differ')
    output=dict(complete=source['complete'],results=results,same_initial_state=True,
        physical_quality='requires_fresh_paired_PhysX_evaluation')
    if args.center_support:
        output['actual_consumer_support']=consume_reference_changes(args.comparison.parent,args.center_support)
    atomic_json(args.output,output)
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
