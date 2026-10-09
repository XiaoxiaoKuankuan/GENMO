"""只读比较服务器1八卡部署时钟/外围采样验收产物，输出可追溯JSON报告。

基准与候选必须固定相同任务、模型和有限验收噪声种子；逐rank、逐轮核对160条
真实转移的任务、P、模拟到达、控制步、终止及奖励/扩散链差异。物理浮点差异如实
报告，不放宽既有零更新概率检查，也不把不同执行量的旧时钟时间称为纯性能收益。
计时取每轮最慢rank墙钟，分别报告批量生成、世界RPC、物理与证据处理，注明重叠。
本工具不运行训练、不修改checkpoint，只读取已有完整rollout和报告。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch


def summary(root):
    path=root/'report.json'
    data=json.loads(path.read_text())
    if data.get('status')!='passed' or len(data['ranks'])!=8:
        raise ValueError(f'Incomplete eight-rank report: {path}')
    rounds=[]
    for iteration in range(len(data['ranks'][0]['rounds'])):
        rows=[r['rounds'][iteration] for r in data['ranks']]
        rounds.append(dict(iteration=iteration+1,collection_seconds=max(r['seconds'] for r in rows),
            real_transitions=sum(r['total_transitions'] for r in rows),
            control_steps=sum(r['control_steps'] for r in rows),
            failures=sum(r['physical_failures'] for r in rows),
            prefix_frames=sorted(set(v for r in rows for v in r.get('prefix_frames',[]))),
            modeled_delay_ticks=sorted(set(v for r in rows for v in r.get('modeled_delay_ticks',[]))),
            generation_batch_seconds=max(r['generation_batch_wall_seconds'] for r in rows),
            phase_rank_max={key:max(r['world_timing'].get(key,0.) for r in rows) for key in rows[0]['world_timing']},
            timing_scope='rank_max_and_overlapping_components_not_additive'))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rounds=rounds)


def compare(reference,candidate,rounds):
    total=0;chain_error=reward_error=0.
    mismatches=[]
    for iteration in range(rounds):
        for rank in range(8):
            def load(root):
                return [r for rows in torch.load(root/f'rank{rank:02d}'/f'rollout_{iteration:06d}.pt',
                    map_location='cpu',weights_only=False) for r in rows]
            old,new=load(reference),load(candidate)
            if len(old)!=20 or len(new)!=20: raise ValueError('Local transition count differs')
            for index,(a,b) in enumerate(zip(old,new)):
                total+=1
                for key in ('training_task','collector_env_slot'):
                    if a.metadata[key]!=b.metadata[key]:mismatches.append([iteration,rank,index,key])
                for key in ('control_tick_begin','control_tick_end','executed_control_steps','terminated','truncated'):
                    if getattr(a,key)!=getattr(b,key):mismatches.append([iteration,rank,index,key])
                for key in ('prefix_frames','decision_tick','deadline_tick'):
                    if a.metadata['generated'][key]!=b.metadata['generated'][key]:mismatches.append([iteration,rank,index,key])
                if a.metadata['timing'].get('deployment_clock')!=b.metadata['timing'].get('deployment_clock'):
                    mismatches.append([iteration,rank,index,'deployment_clock'])
                chain_error=max(chain_error,float((a.chain-b.chain).abs().max()))
                if a.rewards.shape==b.rewards.shape:
                    reward_error=max(reward_error,float((a.rewards-b.rewards).abs().max()) if a.rewards.numel() else 0.)
    return dict(transitions_compared=total,causal_metadata_exact=not mismatches,mismatches=mismatches,
        max_chain_abs_difference=chain_error,max_step_reward_abs_difference=reward_error,
        scope='Exact_clock_task_counts_and_terminal_identity; numeric_differences_reported_without_changing_tolerances')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    left,right=summary(args.reference),summary(args.candidate)
    count=min(len(left['rounds']),len(right['rounds']))
    parity=compare(args.reference,args.candidate,count)
    result=dict(reference=left,candidate=right,comparison=parity,
        matched_workload=parity['causal_metadata_exact'],
        seconds_saved=[a['collection_seconds']-b['collection_seconds'] for a,b in zip(left['rounds'],right['rounds'])])
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if parity['causal_metadata_exact'] else 1


if __name__=='__main__':raise SystemExit(main())
