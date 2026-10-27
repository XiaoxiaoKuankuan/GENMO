"""把有界A/B训练和分离评价整理为曲线及机器可读汇总。

只读取指定实验目录中的metrics/evaluation，不把缺失的真实物理指标补成零，不把重建
当作纯噪声生成。曲线随checkpoint步数展示重建、多样性、音乐敏感性、动作幅度/速度
及整曲结束率；所有原始失败目录保留，图例显示实际评价模式。输出新目录可独立分享。
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--training-run',action='append',help='只绘制指定run，可重复；续训目录分别保留身份')
    args=parser.parse_args()
    root,output=Path(args.root),Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    training,evaluations={},[]
    for path in sorted(root.glob('*/metrics.jsonl')):
        if args.training_run and path.parent.name not in args.training_run:
            continue
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        training[path.parent.name]=rows
    keys=('unknown_reconstruction_rmse','within_model_seed_diversity_rms','music_condition_sensitivity_rms','root_xy_range_m',
          'root_yaw_circular_spread','joint_amplitude_mean_rad','generated_joint_speed_mean_rad_s','complete_song',
          'complete_song_with_hard_limits')
    for path in sorted(root.glob('*/evaluation.json')):
        value=json.loads(path.read_text())
        row={'name':path.parent.name,'step':value['checkpoint_step'],'mode':value['mode'],
             'training_run':value.get('training_run',Path(value['checkpoint']).parent.name),
             'evidence':value['evidence'],'timing_mode':value['evaluation_config'].get('timing_mode')}
        if args.training_run and row['training_run'] not in args.training_run:
            continue
        # 评价集合不同不能连成同一条曲线；选择身份不包含模型决定的成功/失败结果。
        requests=[{k:r.get(k) for k in ('source','frame','k','noise_seed')} for r in value['rows']]
        canonical=json.dumps(sorted(requests,key=lambda x:json.dumps(x,sort_keys=True)),sort_keys=True)
        row['evaluation_set_sha256']=hashlib.sha256(canonical.encode()).hexdigest()
        row['evaluation_sources']=sorted({r['source'] for r in value['rows']})
        for key in (*keys,'data_loss'):
            selected=[float(r[key]) for r in value['rows'] if key in r]
            if selected:row[key]=sum(selected)/len(selected)
        row['rows']=len(value['rows'])
        row['rejected_rows']=sum('rejection' in r for r in value['rows'])
        evaluations.append(row)
    run_names=sorted(set(training)|{r['training_run'] for r in evaluations})
    aliases={name:f'R{i+1}' for i,name in enumerate(run_names)}
    (output/'summary.json').write_text(json.dumps({'run_labels':{v:k for k,v in aliases.items()},
                                                   'training':{k:v[-1] for k,v in training.items() if v},
                                                   'evaluations':evaluations},ensure_ascii=False,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for ax,key in zip(axes.flat,keys):
        groups=defaultdict(list)
        for row in evaluations:
            if key in row:
                groups[(row['training_run'],row['mode'],row['evidence'],row['timing_mode'],row['evaluation_set_sha256'][:8])].append(row)
        for group,rows in groups.items():
            rows.sort(key=lambda r:r['step'])
            run,mode,evidence,timing,subset=group
            evidence='Newton' if evidence=='real_newton' else 'numeric'
            timing='fixed' if timing=='fixed_delay_replay' else 'wall'
            label=f'{aliases[run]} {mode[:3]} {evidence} {timing} {subset[:6]}'
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label=label)
        import textwrap
        ax.set_title(textwrap.fill(key.replace('_',' '),34),fontsize=10)
        ax.set_xlabel('supervised updates');ax.grid(alpha=.2)
        if key.startswith('complete_song'):ax.set_ylim(-.05,1.05)
        if groups:ax.legend(fontsize=7)
    fig.suptitle('Independent reconstruction / DDIM20 generation / Newton evaluation')
    fig.supxlabel('\n'.join(f'{alias}: {name}' for name,alias in aliases.items()),fontsize=8)
    fig.savefig(output/'evaluation_curves.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key in zip(axes.flat,('data_loss','execution_loss','physics_effective_updates','physics_update_fraction')):
        for name,rows in training.items():
            selected=[r for r in rows if key in r]
            if selected:ax.plot([r['step'] for r in selected],[r[key] for r in selected],label=aliases[name])
        ax.set_title(key);ax.set_xlabel('supervised updates');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.supxlabel('\n'.join(f'{alias}: {name}' for name,alias in aliases.items()),fontsize=8)
    fig.savefig(output/'training_curves.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
