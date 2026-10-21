"""只读比较Stage2八项跟踪奖励尺度及大误差饱和，不改变训练或历史评分。

输入可选的真实评估report.json及独立输出目录。读取GMT原始平方误差，分别计算
A原std指数、B用户新std指数、C新指数与5%有界慢衰减项的离线分数；输入平方误差
只除sigma平方，绝不再平方。输出CSV/JSON数值表、四项位置姿态曲线和尾部斜率。
若提供真实报告，额外统计同一批实际控制下的分数与饱和比例。这些是反事实评分，
不表示机器人使用新奖励训练后的物理收益；C只存在于本工具，不进入正式奖励。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.rewards import TRACKING_STD,STAGE2_REWARD_STD


def scores(squared_error,name,alpha=.05):
    e=np.asarray(squared_error,dtype=np.float64)
    if not np.isfinite(e).all() or (e<0).any():raise ValueError('Expected finite nonnegative squared error')
    q=e/STAGE2_REWARD_STD[name]**2
    old=np.exp(-e/TRACKING_STD[name]**2);new=np.exp(-q)
    return old,new,(1-alpha)*new+alpha/(1+q)


def analyze(output,report=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');table=[]
    for ax,name in zip(axes.flat,('anchor_pos','anchor_ori','body_pos','body_ori')):
        angular=name.endswith('ori');x=np.linspace(0,3.14 if angular else 1.5,700)
        values=scores(x*x,name)
        for label,y in zip(('A: original exponential','B: new exponential','C: 5% slow tail (analysis only)'),values):ax.plot(x,y,label=label)
        ax.set(title=f'{name}: source std={TRACKING_STD[name]}, reward std={STAGE2_REWARD_STD[name]}',xlabel='sqrt(GMT squared error) [rad]' if angular else 'sqrt(GMT squared error) [m]',ylabel='Tracking score',ylim=(0,1.02))
        ax.grid(alpha=.25);ax.legend(fontsize=8)
        for error in (0,.05,.1,.2,.3,.5,.8,1.,1.5,2.,3.):
            old,new,slow=scores(error*error,name)
            sigma=STAGE2_REWARD_STD[name];q=(error/sigma)**2
            table.append(dict(term=name,error=error,A=float(old),B=float(new),C=float(slow),
                B_abs_slope=2*error/sigma**2*float(new),
                C_abs_slope=.95*2*error/sigma**2*float(new)+.05*2*error/sigma**2/(1+q)**2))
    fig.savefig(output/'reward_std_curves.png',dpi=180);fig.savefig(output/'reward_std_curves.pdf');plt.close(fig)
    tail,tail_axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,name in zip(tail_axes.flat,('anchor_pos','anchor_ori','body_pos','body_ori')):
        x=np.linspace(0,3.14 if name.endswith('ori') else 1.5,700)
        for label,y in zip(('A: original','B: requested','C: analysis only'),scores(x*x,name)):
            ax.semilogy(x,y,label=label)
        ax.set(title=name,xlabel='sqrt(GMT squared error) [rad]' if name.endswith('ori') else 'sqrt(GMT squared error) [m]',
            ylabel='Tracking score (log scale)',ylim=(1e-12,1.1))
        ax.axhline(1e-3,color='gray',linestyle='--',linewidth=.8);ax.grid(alpha=.25);ax.legend(fontsize=8)
    tail.savefig(output/'reward_std_tail_curves.png',dpi=180)
    tail.savefig(output/'reward_std_tail_curves.pdf');plt.close(tail)
    with (output/'reward_std_values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    result=dict(formula='A=exp(-squared_error/source_std^2); B=exp(-squared_error/reward_std^2); C=.95*B+.05/(1+squared_error/reward_std^2)',
        source_std=TRACKING_STD,reward_std=STAGE2_REWARD_STD,formal_candidate='B',C_training_enabled=False,
        saturation_thresholds={n:{str(p):STAGE2_REWARD_STD[n]*float(np.sqrt(-np.log(p))) for p in (1e-3,1e-6)} for n in TRACKING_STD},table=table)
    if report is not None:
        from gem.closedloop.dppo.compact_evidence import report_episodes
        report=Path(report);document=json.loads(report.read_text());samples={n:[] for n in TRACKING_STD}
        for episode in report_episodes(document):
            for decision in episode['decisions']:
                for control in decision['controls']:
                    for name in samples:samples[name].append(control['raw_errors'][name+'_squared_error'])
        result['actual_control_scoring']={}
        for name,values in samples.items():
            if not values:continue
            a,b,c=scores(values,name)
            result['actual_control_scoring'][name]=dict(count=len(values),A_mean=float(a.mean()),B_mean=float(b.mean()),C_mean=float(c.mean()),
                A_below_1e_3=float((a<1e-3).mean()),B_below_1e_3=float((b<1e-3).mean()),B_below_1e_6=float((b<1e-6).mean()))
        result['physical_claim']='same_recorded_physics_counterfactual_scoring_only'
    (output/'reward_std_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--report',type=Path)
    args=parser.parse_args();analyze(args.output,args.report)
