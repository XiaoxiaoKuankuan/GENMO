"""只读汇总原DPPO正式历史的裁剪、学习率回溯和KL利用率证据。

输入正式运行根目录下metrics/*.jsonl，逐行读取已经发布的训练记录，保留每轮
最终KL、Active Clip、实际接受LR和每次回溯原因，汇总时不把缺测值填成零。
输出写到显式指定的独立诊断文件，不修改正式checkpoint、latest、预算或日志。
历史的代理目标没有固定全rollout前后复算，因此这里不据此判断PPO更新方向；
方向判断由另一个固定真实数据八卡工具完成。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def describe(values):
    if not values:return None
    x=sorted(values)
    return dict(count=len(x),min=x[0],p50=statistics.median(x),max=x[-1],mean=statistics.mean(x))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve(strict=True);output=args.output.resolve()
    if output.is_relative_to(source):raise ValueError('诊断不得写入正式训练目录')
    rows=[];inputs=[];attempts=Counter();failures=Counter();reasons=Counter()
    lrs=[];retry_steps=0;total_steps=0
    for path in sorted((source/'metrics').glob('*.jsonl')):
        inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for line in path.open():
            record=json.loads(line)
            actor=record.get('actor');kl=record.get('trainable4_kl')
            if not actor or not kl:continue
            control=actor.get('lr_control',{});reasons[control.get('reason','missing')]+=1
            accepted=[];row_attempts=0
            for step in actor.get('steps',[]):
                total_steps+=1
                lr=step['learning_rate'];lrs.append(lr);accepted.append(lr)
                transaction=step['backtracking'];n=transaction['attempt_count']
                attempts[str(n)]+=1;row_attempts+=n;retry_steps+=int(n>1)
                for attempt in transaction['attempts']:
                    for name in attempt.get('acceptance',{}).get('failures',{}):failures[name]+=1
            rejected=actor.get('rejected_step')
            rows.append(dict(label=record['label'],iteration=record['state']['iteration'],
                optimizer_steps=actor['optimizer_steps'],planned=actor['planned_optimizer_steps'],
                attempted_optimizer_steps=actor['attempted_optimizer_steps'],accepted_learning_rates=accepted,
                active_clip_fraction=kl['objective_clip_fraction'],ordinary_clip_fraction=kl['clip_fraction'],
                mean_coordinate_kl=kl['mean_coordinate_kl'],maximum_step_mean_coordinate_kl=kl['maximum_step_mean_coordinate_kl'],
                p95_coordinate_kl=kl['coordinate_kl']['p95'],chain_joint_kl=kl['chain_joint_kl'],
                lr_control=control,early_stop_reason=actor.get('early_stop_reason'),
                rejected_step=rejected,probability_passed=record.get('probability',{}).get('passed')))
    if not rows:raise ValueError('没有完整正式训练记录')
    report=dict(source=str(source),inputs=inputs,rounds=len(rows),
        first_iteration=rows[0]['iteration'],last_iteration=rows[-1]['iteration'],
        accepted_adam_steps=total_steps,accepted_lr=describe(lrs),retry_steps=retry_steps,
        retry_step_fraction=retry_steps/total_steps,attempt_counts=dict(attempts),failure_counts=dict(failures),
        lr_control_reasons=dict(reasons),
        active_clip=describe([r['active_clip_fraction'] for r in rows]),
        rounds_with_nonzero_active_clip=sum(r['active_clip_fraction']>0 for r in rows),
        mean_chain_joint_kl=describe([r['chain_joint_kl']['mean'] for r in rows]),
        max_chain_joint_kl=describe([r['chain_joint_kl']['max'] for r in rows]),
        kl_utilization=describe([r['lr_control']['utilization'] for r in rows if 'utilization' in r['lr_control']]),
        incomplete_update_rounds=[r['label'] for r in rows if r['optimizer_steps']!=r['planned']],
        rows=rows,scope='published_original_DPPO_history_not_fixed_rollout_objective_validation')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    temporary.replace(output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','inputs')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
