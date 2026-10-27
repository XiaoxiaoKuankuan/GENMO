"""把有界A/B训练和分离评价整理为曲线及机器可读汇总。

只读取指定实验目录中的metrics/evaluation，不把缺失的真实物理指标补成零，不把重建
当作纯噪声生成。曲线随checkpoint步数展示重建、多样性、音乐敏感性、动作幅度/速度
及整曲结束率；所有原始失败目录保留，图例显示实际评价模式。输出新目录可独立分享。
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root,output=Path(args.root),Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    training,evaluations={},[]
    for path in sorted(root.glob('*/metrics.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        training[path.parent.name]=rows
    keys=('unknown_reconstruction_rmse','within_model_seed_diversity_rms','music_condition_sensitivity_rms','root_xy_range_m',
          'root_yaw_circular_spread','joint_amplitude_mean_rad','generated_joint_speed_mean_rad_s','complete_song')
    for path in sorted(root.glob('*/evaluation.json')):
        value=json.loads(path.read_text())
        row={'name':path.parent.name,'step':value['checkpoint_step'],'mode':value['mode'],
             'training_run':value.get('training_run',Path(value['checkpoint']).parent.name),
             'evidence':value['evidence'],'timing_mode':value['evaluation_config'].get('timing_mode')}
        for key in (*keys,'data_loss'):
            selected=[float(r[key]) for r in value['rows'] if key in r]
            if selected:row[key]=sum(selected)/len(selected)
        row['rows']=len(value['rows'])
        row['rejected_rows']=sum('rejection' in r for r in value['rows'])
        evaluations.append(row)
    (output/'summary.json').write_text(json.dumps({'training':{k:v[-1] for k,v in training.items() if v},
                                                   'evaluations':evaluations},ensure_ascii=False,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for ax,key in zip(axes.flat,keys):
        groups=defaultdict(list)
        for row in evaluations:
            if key in row:
                groups[(row['training_run'],row['mode'],row['evidence'],row['timing_mode'])].append(row)
        for group,rows in groups.items():
            rows.sort(key=lambda r:r['step'])
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label='/'.join(str(x) for x in group))
        ax.set_title(key);ax.set_xlabel('supervised updates');ax.grid(alpha=.2)
        if groups:ax.legend(fontsize=7)
    fig.suptitle('Independent reconstruction / DDIM20 generation / Newton evaluation')
    fig.savefig(output/'evaluation_curves.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key in zip(axes.flat,('data_loss','execution_loss','physics_effective_updates','physics_update_fraction')):
        for name,rows in training.items():
            selected=[r for r in rows if key in r]
            if selected:ax.plot([r['step'] for r in selected],[r[key] for r in selected],label=name)
        ax.set_title(key);ax.set_xlabel('supervised updates');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(output/'training_curves.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
