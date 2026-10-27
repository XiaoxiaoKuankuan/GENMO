"""漂移诊断证据汇总：真实轨迹、实际AdamW候选和采样漏斗。

只读已保存JSON/张量，不初始化模型、不运行训练、不修改旧正式目录。按歌曲等权
汇总完整真实误差，候选方向只统计超过重复物理噪声的案例；同时保留参数相同的
消融重复，单独报告独立物理方向，避免把重复候选当独立统计样本。记录真实转向、
关节速度相位与两次连续生成的变化；梯度点积不是物理改善证明。
"""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

from .drift_diagnostics import stats,sample_counts,write_json


def summarize(root,run):
    offline=root/('offline_v3' if (root/'offline_v3/summary.json').exists() else 'offline_v2')
    raw=json.loads((offline/'trajectory_objectives.json').read_text())
    log=json.loads((offline/'log_audit.json').read_text())
    rows=[];dedup={};failures=[];effects=[]
    for folder in sorted(root.glob('candidates_*')):
        if not folder.is_dir() or not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text());failures.extend(summary['failures'])
        for row in summary['cases']:
            rows.append(row)
            data=torch.load(row['artifact'],map_location='cpu',weights_only=False)
            before,after=data['physical_before'],data['physical_after']
            from supermimic.rotations import quat_to_mat,yaw_mat,so3_log
            rb,ra=quat_to_mat(before[:,3:7]),quat_to_mat(after[:,3:7])
            yaw=so3_log(yaw_mat(rb).transpose(-1,-2)@yaw_mat(ra))[:,2]
            phase=torch.nn.functional.cosine_similarity(before[:,34:55].flatten(),after[:,34:55].flatten(),dim=0)
            item={k:row[k] for k in ('version','task','anchor_tick','variant','trainable','phases','model_delta','physical_delta','direction_resolvable','same_descent_sign','output_trust_reasons')}
            item.update(root_yaw_candidate_change_rms_rad=float(yaw.square().mean().sqrt()),
                root_yaw_final_delta_rad=float(yaw[-1]),root_position_candidate_change_rms_m=float((after[:,:3]-before[:,:3]).square().mean().sqrt()),
                actual_velocity_phase_before_after=float(phase),actual_joint_velocity_rms_before=float(before[:,34:55].square().mean().sqrt()),
                actual_joint_velocity_rms_after=float(after[:,34:55].square().mean().sqrt()),
                actual_joint_range_mean_before=float((before[:,13:34].amax(0)-before[:,13:34].amin(0)).mean()),
                actual_joint_range_mean_after=float((after[:,13:34].amax(0)-after[:,13:34].amin(0)).mean()),
                contacts_changed=row['contacts_changed'],physical_restore_error=row['physical_restore_error'])
            if row.get('next_two_real_generations'):
                v=row['next_two_real_generations'];item['next_two_generation_world_dance_delta']=v['after']['dance_world_mean']-v['before']['dance_world_mean']
                item['next_two_generation_physical_seconds']=(v['before']['end_tick']-v['before']['request_tick'])/600
                item['next_two_generation_joint_range_delta']=v['after']['joint_range_mean']-v['before']['joint_range_mean']
                item['next_two_generation_joint_speed_delta']=v['after']['joint_speed_rms']-v['before']['joint_speed_rms']
            effects.append(item)
            # 同anchor/同scope且实际参考逐字节一致才去重，不能按相似损失粗略合并。
            import hashlib
            key=(row['version'],row['task'],row['anchor_tick'],hashlib.sha256(data['reference_after'].numpy().tobytes()).hexdigest())
            dedup.setdefault(key,item)
    write_json(root/'physical_candidate_details.json',effects)
    def physical_summary(xs):
        return {'cases':len(xs),'resolved':sum(r['same_descent_sign'] is not None for r in xs),
            'direction_matches':sum(r['same_descent_sign'] is True for r in xs),
            'direction_opposite':sum(r['same_descent_sign'] is False for r in xs),
            'physical_improved':sum(r['physical_delta']<0 for r in xs),
            'physical_delta':stats([r['physical_delta'] for r in xs]),
            'output_trust_valid':sum(not r['output_trust_reasons'] for r in xs)}
    physical={'all':physical_summary(effects),'unique_reference_candidates':physical_summary(list(dedup.values())),
        'by_version':{str(v):physical_summary([r for r in effects if r['version']==v]) for v in [0,100,300,700]},
        'variants':{f'{kind}/{scope}':physical_summary([r for r in effects if r['variant']==kind and r['trainable']==scope])
            for kind in ['dance','dance_priors','dance_bc','full'] for scope in ['all','future_only']},
        'direction_failures':[r for r in dedup.values() if r['same_descent_sign'] is False],
        'support_and_execution_failures':failures,'phase_counts_cases':dict(Counter(k for r in effects for k in r['phases']))}
    paired=defaultdict(dict)
    for row in rows:paired[(row['version'],row['task'],row['anchor_tick'],row['trainable'])][row['variant']]=row
    reversals=[]
    for key,items in paired.items():
        if all(k in items for k in ['dance','full']) and items['dance']['physical_delta']<0<items['full']['physical_delta']:
            reversals.append({'version':key[0],'task':key[1],'anchor_tick':key[2],'scope':key[3],
                'dance_only_physical_delta':items['dance']['physical_delta'],'full_physical_delta':items['full']['physical_delta'],
                'full_dance_gradient_dot_delta':items['full']['raw_gradient_dot_actual_parameter_delta']['all']['dance'],
                'full_rejection_reasons':items['full']['output_trust_reasons']})
    physical['dance_to_full_physical_worsening']=reversals
    write_json(root/'physical_direction_summary.json',physical)
    # 形成、合法和超域各组按真实请求去重，避免支持域重复检查放大困难动作频率。
    funnel=defaultdict(dict)
    def identity(r):return tuple(r.get(k) for k in ['collector','episode','task','request_tick'])
    with (run/'sampling_funnel.jsonl').open() as f:
        for line in f:
            d=json.loads(line)
            if d['actor_version']>=700:break
            for r in d.get('A_all_real_requests',[]):funnel['A_requests'][identity(r)]=r
            for r in d.get('B_anchor_outcomes',[]):
                if r.get('stage')=='B_formed_H_anchor':funnel['B_formed'][identity(r)]=r
            for r in d.get('BC_support_classification',[]):funnel['B_legal' if r['legal'] else 'C_outside'][identity(r)]=r
    with (run/'training.jsonl').open() as f:
        for line in f:
            d=json.loads(line)
            if d.get('accepted_actor_updates',0)>700:break
            for r in d.get('sample_distribution',[]):
                funnel['D_drawn'][identity(r)]=r
                if r.get('accepted'):funnel['E_updated'][identity(r)]=r
    distribution={k:sample_counts(list(v.values())) for k,v in funnel.items()}
    distribution['out_of_support_rate_by_speed']={k:sum(r.get('speed_bin')==k for r in funnel['C_outside'].values())/
        max(1,sum(r.get('speed_bin')==k for name in ['B_legal','C_outside'] for r in funnel[name].values())) for k in ['low','medium','high']}
    distribution['state_components_unique_failures']=dict(Counter(c['name'] for r in funnel['C_outside'].values() for c in (r.get('support_details') or {}).get('components',[])))
    distribution['first_failure_prediction_steps_unique']=dict(Counter((r.get('support_details') or {}).get('prediction_step') for r in funnel['C_outside'].values()))
    cfg=json.loads((run/'manifest.json').read_text())['config']
    train=json.loads(Path(cfg['tasks']).read_text());evaluation=json.loads(Path(cfg['formal']['evaluation_tasks']).read_text())
    distribution['training_music_count']=len(train);distribution['evaluation_music_count']=len(evaluation)
    distribution['training_evaluation_music_id_overlap']=sorted({t['id'] for t in train}&{t['id'] for t in evaluation})
    distribution['task_source_counts']={'training':dict(Counter(t['source'] for t in train)),'evaluation':dict(Counter(t['source'] for t in evaluation))}
    distribution['fixed_evaluation']=json.loads((offline/'evaluation_sample_distribution.json').read_text())
    write_json(root/'sampling_unique_summary.json',distribution)
    worse=json.loads((offline/'accepted_dance_worsening.json').read_text())
    trust={'accepted':log['accepted'],'all_candidates':log['candidate_count'],'accepted_dance_worse':len(worse),
        'actual_relative_worsening':stats([r['dance_delta']/max(1e-20,abs(r['loss_before']['dance'])) for r in worse]),
        'allowed_relative_worsening_in_138':stats([r['allowed_increase']/max(1e-20,abs(r['loss_before']['dance'])) for r in worse]),
        'absolute_plus_relative_tolerance':'1e-5 + 0.005 * abs(before); not a pure 0.5 percent cap',
        'outputs':log['accepted_output_statistics'],
        'next_two_generation':{'cases':sum('next_two_generation_world_dance_delta' in r for r in effects),
            'world_dance_delta':stats([r.get('next_two_generation_world_dance_delta') for r in effects]),
            'worse':sum(r.get('next_two_generation_world_dance_delta',0)>0 for r in effects)}}
    write_json(root/'trust_drift_summary.json',trust)
    objective={str(v):{'songs':len(part),
        'A_anchor_H16':float(np.mean([r['A_anchor_H16_mean']['mean'] for r in part])),
        'B_episode':float(np.mean([r['B_episode_once_loss'] for r in part])),
        'C_world':float(np.mean([r['C_world_loss'] for r in part])),
        'world_yaw_rms_rad':float(np.mean([r['C_world_yaw_rms_rad'] for r in part])),
        'components':{k:float(np.mean([r['B_components'][k] for r in part])) for k in part[0]['B_components']},
        'song_details':part} for v in [0,100,300,600,700] if (part:=[r for r in raw if r['version']==v])}
    write_json(root/'objective_summary.json',objective)
    print(json.dumps({'physical':physical['all'],'unique':physical['unique_reference_candidates'],'dance_to_full_reversals':len(reversals),
        'trust':{k:v for k,v in trust.items() if k not in ['outputs']},'support_rate':distribution['out_of_support_rate_by_speed']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();run=Path(args.run)
    cfg=json.loads((run/'manifest.json').read_text())['config'];sys.path.insert(0,cfg['supermimic']['repo'])
    torch.set_num_threads(2);summarize(Path(args.root),run)
