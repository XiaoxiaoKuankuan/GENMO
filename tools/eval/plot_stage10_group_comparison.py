"""把严格核验后的完整音乐组三模型对照制作成中文静态分析图。

主图同时展示val/test的完成率、每秒回报、实际活动和关节边界，来源等权统计
避免AIOZ音乐较多支配结果。配对差值图以音乐组bootstrap区间显示不确定性；
奖励分解图只比较共同真实执行时段，不把提前失败后消失的难段当成提升。
另输出逐来源比较、关节边界热图与独立完整MC价值误差，供报告解释机制。
输入是analyze_stage10_group_comparison生成的SHA核验结果，不平滑、插补或
删除坏表现；图中明确实际动作、参考动作和行政/真实终止统计口径。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


MODELS=('stage1','iteration147','current')
LABELS=('Stage1','第147轮','第246轮')
COLORS=('#367bb5','#289c72','#e68a3b')
SOURCES=('AIOZ-GDANCE','AIST++','FineDance','Mine')


def source_mean(rows,key):
    groups=defaultdict(list)
    for row in rows:groups[row['dataset']].append(row[key])
    return float(np.mean([np.mean(v) for v in groups.values()]))


def save(fig,output,name):
    fig.savefig(output/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white')
    fig.savefig(output/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plot(data,output):
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False,
        'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig,axes=plt.subplots(2,4,figsize=(16.5,8))
    for row,split in enumerate(('val','test')):
        s=data['summaries'][split]
        for col,(key,title,scale) in enumerate((('completed','音乐完成率（来源等权，%）',100),
                ('reward_per_second','每秒回报（每任务→来源等权）',1),
                ('activity','实际活动度（rad/s）',1),('soft_joint_fraction','近/触软边界关节时间（%）',100))):
            ax=axes[row,col];v=[s[m]['source_balanced_means'][key]*scale for m in MODELS]
            ax.bar(LABELS,v,color=COLORS,width=.66)
            for j,value in enumerate(v):
                label=f'{value:.2f}'
                if key=='completed':label+=f"\n{s[MODELS[j]]['overall']['completed_count']}/{s[MODELS[j]]['overall']['episodes']}次"
                ax.text(j,value,label,ha='center',va='bottom',fontsize=10)
            ax.set_title(f'{split.upper()} · {title}',fontsize=11)
            ax.set_ylim(0,max(v)*1.22 if max(v)>0 else 1)
            if key=='completed':ax.set_ylim(0,120);ax.set_yticks([0,25,50,75,100])
    fig.suptitle('同条件完整音乐对照：Stage1 / 第147轮 / 第246轮',fontsize=18,y=1.01)
    fig.text(.01,-.025,'49个独立音乐组 × 2种子 × 3模型；val 27组/54次，test 22组/44次。柱高四来源等权；完成次数为实际总数。\n'
        '音乐奖励关闭，音乐条件保留。边界为实际奖励定义的近/触界；不等于硬越界。整首指数据集中可用配对音频全长。',fontsize=10)
    fig.tight_layout();save(fig,output,'complete_song_overview')

    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for row,split in enumerate(('val','test')):
        for col,(key,title,scale) in enumerate((('completed','音乐完成率 %',100),('reward_per_second','每秒回报',1),
                ('activity','实际活动度 rad/s',1),('soft_joint_fraction','近/触边界 %',100))):
            ax=axes[row,col];x=np.arange(4)
            for i,m in enumerate(MODELS):
                values=[data['summaries'][split][m]['by_source'][s]['means'][key]*scale for s in SOURCES]
                ax.bar(x+(i-1)*.24,values,.23,color=COLORS[i],label=LABELS[i])
            counts=[data['summaries'][split]['stage1']['by_source'][s]['music_groups'] for s in SOURCES]
            ax.set_xticks(x,[s.replace('-GDANCE','')+f'\n{n}组' for s,n in zip(SOURCES,counts)],fontsize=9)
            ax.set_title(split.upper()+' · '+title)
            if key=='completed':ax.set_ylim(0,110)
    axes[0,0].legend(fontsize=8,ncol=1,loc='lower right')
    fig.suptitle('按独立音乐来源拆开看，避免平均数掩盖退化',fontsize=18)
    fig.tight_layout();save(fig,output,'complete_song_by_source')

    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for row,split in enumerate(('val','test')):
        for col,(metric,label,scale) in enumerate((('completed','完成率变化（百分点）',100),
                ('reward_per_second','每秒回报变化',1),('activity','实际活动度变化（rad/s）',1))):
            ax=axes[row,col]
            for i,(name,text,color) in enumerate((('iteration147_minus_stage1','147 − Stage1',COLORS[1]),
                    ('current_minus_stage1','246 − Stage1',COLORS[2]),('current_minus_iteration147','246 − 147','#8563aa'))):
                result=data['paired'][split][name]['bootstrap'][metric]
                lo,hi=np.array(result['ci95'])*scale;value=result['mean']*scale
                ax.plot([lo,hi],[i,i],color=color,lw=3);ax.scatter([value],[i],c=color,s=55,zorder=3)
            ax.axvline(0,color='#444',lw=1);ax.set_yticks(range(3),['147 − Stage1','246 − Stage1','246 − 147'])
            ax.set_ylim(2.6,-.6);ax.set_xlabel(label);ax.set_title(split.upper()+' · 配对差与95%区间')
    fig.suptitle('同音乐、同种子的变化：置信区间按音乐组成组重抽样',fontsize=17)
    fig.text(.015,-.018,'来源内抽组、再来源等权；2个种子不计为2首独立音乐。小音乐池的区间仅反映本面板，不代表所有未见音乐。',fontsize=10)
    fig.tight_layout();save(fig,output,'paired_music_group_intervals')

    fig,axes=plt.subplots(2,2,figsize=(14,9))
    partnames=('tracking_score_contribution_per_second','tracking_gate_contribution_per_second','stable','joint_limit','other')
    partlabels=('姿态跟踪分','活动门控','稳定性','关节边界','其余及事件')
    palette=('#4486b5','#d96378','#48a985','#c99431','#96979d')
    for row,split in enumerate(('val','test')):
        for col,pair in enumerate(('iteration147_minus_stage1','current_minus_stage1')):
            ax=axes[row,col];rows=data['paired'][split][pair]['tasks'];x=np.arange(5)
            datasets=list(SOURCES)+['来源等权'];pos=np.zeros(5);neg=np.zeros(5)
            for part,label,color in zip(partnames,partlabels,palette):
                values=[]
                for source in datasets:
                    selected=rows if source=='来源等权' else [r for r in rows if r['dataset']==source]
                    calc=[]
                    for r in selected:
                        comp=r['common_components_delta_per_second']
                        value=r[part] if part in r else comp[part] if part in comp else sum(v for k,v in comp.items() if k not in ('track','stable','joint_limit'))
                        calc.append(dict(dataset=r['dataset'],value=value))
                    values.append(source_mean(calc,'value'))
                values=np.array(values);bottom=np.where(values>=0,pos,neg)
                ax.bar(x,values,bottom=bottom,color=color,label=label)
                pos+=np.maximum(values,0);neg+=np.minimum(values,0)
            ax.axhline(0,color='#555',lw=1);ax.set_xticks(x,['AIOZ','AIST++','FineDance','Mine','来源等权'])
            ax.set_title(split.upper()+' · '+('147 − Stage1' if col==0 else '246 − Stage1'))
            ax.set_ylabel('共同真实执行时段的回报变化 / 秒')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('收益或下降来自哪一项：逐控制步精确分解',fontsize=18)
    fig.text(.015,-.012,'tracking = 2.5 × 原始跟踪分 × 活动门控。分解逐步精确相加；较长模型额外执行的尾段单独保存在分析JSON中。',fontsize=10)
    fig.tight_layout();save(fig,output,'reward_change_attribution')

    fig,axes=plt.subplots(1,2,figsize=(15,12))
    names=sorted({r['joint'] for r in data['joint_rows']})
    for ax,split in zip(axes,('val','test')):
        matrix=[]
        for name in names:
            values=[]
            for model in MODELS:
                rows=[r for r in data['joint_rows'] if r['split']==split and r['model']==model and r['joint']==name]
                values.extend([source_mean(rows,k)*100 for k in ('actual_soft_fraction','reference_soft_fraction')])
            matrix.append(values)
        a=np.array(matrix);im=ax.imshow(a,aspect='auto',vmin=0,vmax=100,cmap='YlOrRd')
        for i in range(len(names)):
            for j in range(6):ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if a[i,j]>60 else '#222')
        ax.set_xticks(range(6),['Stage1\n实际','Stage1\n参考','147\n实际','147\n参考','246\n实际','246\n参考'])
        ax.set_yticks(range(len(names)),[n.replace('_joint','') for n in names],fontsize=9)
        ax.set_title(split.upper()+' · 各关节近/触界时间占比 %');ax.grid(False)
    fig.colorbar(im,ax=axes,shrink=.55,pad=.02,label='来源等权占比 %')
    fig.suptitle('实际机器人与实际消费参考的关节边界对照',fontsize=18)
    fig.subplots_adjust(wspace=.5,top=.92,bottom=.08,left=.16,right=.88)
    save(fig,output,'joint_boundary_heatmap')

    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for row,split in enumerate(('val','test')):
        for col,(key,title) in enumerate((('mse','独立完整MC：价值MSE'),('explained_variance','价值解释方差 EV'),('bias','价值预测偏差（预测−真实MC）'))):
            ax=axes[row,col]
            values=[data['critic'][split][m]['overall'][key] for m in MODELS]
            ax.bar(LABELS,values,color=COLORS)
            for i,v in enumerate(values):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=10)
            ax.set_title(split.upper()+' · '+title);ax.axhline(0,color='#666',lw=.7);ax.margins(y=.2)
    fig.suptitle('Critic是否能预测真实长回报：各模型自己的完整轨迹',fontsize=17)
    fig.text(.015,-.015,'MC按实际50Hz积分奖励与gamma=0.99/0.5秒反算，真实终止尾值0。按上层状态计数，长歌曲状态更多；阶段间状态分布也会变化。',fontsize=10)
    fig.tight_layout();save(fig,output,'complete_mc_critic_quality')


def plot_failure_case(data, root, output):
    """固定展示最长test歌曲201的第一个seed，绑定真实轨迹SHA，不挑最佳片段。"""
    candidates=[r for r in data['episodes'] if r['split']=='test' and r['dataset']=='FineDance'
        and r['sample_id']=='201' and r['model']=='stage1']
    if not candidates:return
    selected=min(candidates,key=lambda r:r['seed'])
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharey='row')
    for col,model in enumerate(MODELS):
        row=next(r for r in data['episodes'] if r['model']==model and r['task_id']==selected['task_id'])
        path=root/row['trajectory']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['trajectory_sha256']:
            raise ValueError('Failure-case trajectory differs from strict analysis')
        with np.load(path) as a:
            upper_time=(a['upper_tick']-a['tick'][0]+12)/600
            end=len(a['tick'])/50
            ax=axes[0,col]
            ax.plot(upper_time,a['value_prediction'],color='#b74b42',label='Critic预测V')
            ax.plot(upper_time,a['mc_return'],color='#256f9c',label='真实完整MC回报')
            ax.axvline(end,color='#444',ls='--',lw=1)
            ax.set_title(f'{LABELS[col]}：实际执行{end:.2f}秒')
            ax.set_ylabel('从该状态起的折扣回报');ax.set_xlabel('歌曲已执行时间 / 秒')
            ax.legend(fontsize=9)
            ax=axes[1,col];tail=upper_time>=max(0,end-20)
            ax.plot(upper_time[tail]-end,a['value_prediction'][tail],color='#b74b42',label='Critic预测V')
            ax.plot(upper_time[tail]-end,a['mc_return'][tail],color='#256f9c',label='真实完整MC回报')
            ax.axvline(0,color='#444',ls='--',lw=1);ax.axhline(0,color='#888',lw=.7)
            ax.set_title('结束前20秒：'+('朝向误差超限' if row['end_reason']=='yaw_anchor_ori' else row['end_reason']))
            ax.set_xlabel('相对真实结束时刻 / 秒');ax.set_ylabel('预测值与真实回报')
    fig.suptitle(f'长歌失败的价值盲区：FineDance 201，seed={selected["seed"]}',fontsize=18)
    fig.text(.02,-.02,'每个模型使用自己的完整真实轨迹和自身Critic。MC末端真实终止不自举；曲线不跨模型拼接、不裁去失败。\n'
        '该案例帮助解释机制，不代替49个独立音乐组的总体统计，也不能单独证明输入缺项是唯一原因。',fontsize=10)
    fig.tight_layout();save(fig,output,'failure_value_case_finedance201')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trajectory-root',type=Path,help='可选：核验本地真实轨迹，增加固定长歌失败案例图')
    args=parser.parse_args();data=json.loads(args.analysis.read_text())
    if data['status']!='passed':raise ValueError('Strict analysis must pass')
    args.output.mkdir(parents=True,exist_ok=False)
    plot(data,args.output)
    if args.trajectory_root:plot_failure_case(data,args.trajectory_root,args.output)
    print(args.output)


if __name__=='__main__':main()
