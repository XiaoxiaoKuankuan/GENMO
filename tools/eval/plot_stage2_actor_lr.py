"""将真实统一学习率扫描及完整物理分析画成独立中文PNG/SVG。

所有点均来自已保存的实际候选结果；横轴是提议学习率，纵轴分别表示回溯后真正
接受的学习率、固定条件关节参考变化以及真实Adam尝试次数。明确区分参数变化、
输出变化和物理完成率；不把训练loss或数值预筛当作机器人效果。可选物理图要求
Stage1与每个候选的独立analysis.json，展示完成率、总回报、时长及实际关节越界。
本工具只使用CPU绘图，不访问GPU、不载入模型、不修改实验结果；输出禁止覆盖。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def save(fig,path):
    for suffix in ('.png','.svg'):
        target=path.with_suffix(suffix)
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=170,bbox_inches='tight')
    plt.close(fig)


def fixed_scan(scan,output):
    rows=scan['rows'];colors={1:'#2864A0',3:'#D08A2E',10:'#A34B78'}
    fig,axes=plt.subplots(1,3,figsize=(16,4.9))
    labels={1:'原门槛',3:'3倍探索门槛',10:'10倍探索门槛'}
    proposals=sorted({r['proposed_lr'] for r in rows})
    for scale,color in colors.items():
        a=sorted([r for r in rows if r['objective']=='coordinate_dppo' and r['kl_multiplier']==scale],key=lambda r:r['proposed_lr'])
        b=sorted([r for r in rows if r['objective']=='joint_score_pg' and r['kl_multiplier']==scale],key=lambda r:r['proposed_lr'])
        for ax,key,factor in ((axes[0],'accepted_lr',1.),(axes[1],'final_joint_rms',180/np.pi),(axes[2],'attempts',1.)):
            ax.plot(proposals,[r[key]*factor for r in a],'-o',color=color,label=labels[scale])
            ax.plot(proposals,[r[key]*factor for r in b],'--',color=color,alpha=.7)
    axes[0].plot(proposals,proposals,':',color='#888888',label='不回溯：提议=接受')
    axes[0].set_yscale('log');axes[0].set_ylabel('真正接受的统一Actor学习率')
    axes[0].set_title('提议更大，不一定接受更多')
    axes[0].legend(fontsize=9,loc='upper left')
    axes[1].set_ylabel('最终关节参考变化RMS（度）');axes[1].set_title('固定音乐、观测和噪声：输出确实改变')
    axes[2].set_ylabel('每个候选的实际Adam尝试次数');axes[2].set_title('失败回溯也消耗计算')
    axes[2].set_yticks(range(1,8))
    for ax in axes:
        ax.set_xscale('log');ax.set_xticks(proposals,labels=['5e-8','3e-7','1e-6','3e-6','1e-5'])
        ax.set_xlabel('配置中提议的统一Actor学习率');ax.grid(alpha=.2)
    baseline=next(r for r in rows if r['name']=='coordinate_dppo_lr5e-08_kl1')
    fig.suptitle('6144条真实行为链 · 一次全批量更新 · 所有候选同一起点',fontsize=15,y=1.04)
    fig.text(.05,-.06,
        f'实线：坐标平均DPPO；虚线：joint-score。两目标的接受LR和回溯次数一致。'
        f'\n5e-8基线：{baseline["changed_fraction"]*100:.3f}%参数发生变化，'
        f'{baseline["fp32_zero"]*100:.3f}%非零Adam更新被FP32舍入为零。'
        '\n参考变化为128条链的同噪声抽查；此图不证明物理性能提升。3倍/10倍门槛仅属于独立探索实验。',fontsize=10)
    fig.tight_layout();save(fig,output/'fixed_scan')


def physics(paths,output):
    values=[json.loads(p.read_text()) for p in paths]
    names=['Stage1' if i==0 else (p.parent.parent.name if p.parent.name.startswith('attempt')
                                else p.parent.name) for i,p in enumerate(paths)]
    baseline=values[0]['aggregate'];x=np.arange(len(values))
    fig,axes=plt.subplots(2,2,figsize=(max(12,len(values)*.8),10))
    series=[('music_completion_count','完整音乐完成任务数（共32）',False),
            ('reward_sum','任务总回报变化（%）',True),
            ('executed_seconds','总执行时长变化（%）',True)]
    for ax,(key,title,relative) in zip(axes.flat,series):
        y=np.asarray([v['aggregate'][key] for v in values],dtype=float)
        if relative:y=(y/baseline[key]-1)*100
        ax.bar(x,y,color=['#777777']+['#2864A0']*(len(x)-1))
        ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
        if key=='music_completion_count':
            ax.set_ylim(0,33);ax.axhline(baseline[key],color='#888888',linestyle='--')
        else:ax.axhline(0,color='#888888',linestyle='--')
    ax=axes[1,1]
    for kind,offset,color in (('hard',-.18,'#9F3B47'),('soft',.18,'#D49B36')):
        y=[v['aggregate']['joint_limits']['actual'][kind]['max_excess_rad'] for v in values]
        ax.bar(x+offset,y,width=.36,color=color,label='硬限位' if kind=='hard' else '软限位')
    ax.set_ylabel('实际机器人最大关节越界（弧度）');ax.legend();ax.grid(axis='y',alpha=.2)
    for ax in axes.flat:ax.set_xticks(x,names,rotation=55,ha='right',fontsize=8)
    fig.suptitle('同条件完整音乐物理评估：完成率、总收益、执行时长必须一起看',fontsize=15)
    fig.tight_layout();save(fig,output/'physical_comparison')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,nargs='*',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    font_path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font_path)
    font_name=font_manager.FontProperties(fname=font_path).get_name()
    if 'CJK' not in font_name:raise RuntimeError('A real CJK font is required for the Chinese report')
    plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,
                         'svg.fonttype':'path','figure.facecolor':'white'})
    fixed_scan(json.loads(args.scan.read_text()),args.output)
    if args.analysis:physics(args.analysis,args.output)


if __name__=='__main__':main()
