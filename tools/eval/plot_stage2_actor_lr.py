"""将真实统一学习率扫描及完整物理分析画成独立中文PNG/SVG。

所有点均来自已保存的实际候选结果；横轴是提议学习率，纵轴分别表示回溯后真正
接受的学习率、固定条件关节参考变化以及真实Adam尝试次数。明确区分参数变化、
输出变化和物理完成率；不把训练loss或数值预筛当作机器人效果。可选物理图要求
Stage1与每个候选的独立analysis.json，展示完成率、总回报、时长及实际关节越界。
可选--focus读取FineDance075/138已校验的真实50Hz轨迹，区分实际状态与消费参考，
展示yaw、关节跟踪及右臂yaw硬限位；不插补失败之后的轨迹，不用回放替代动力学。
本工具只使用CPU绘图，不访问GPU、不载入模型、不修改实验结果；输出禁止覆盖。
"""
from __future__ import annotations
import argparse
import hashlib
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


def short_name(path):
    name=path.parent.parent.name if path.parent.name.startswith('attempt') else path.parent.name
    if name=='stage1':return 'Stage1'
    for prefix,label in (('coordinate_dppo_lr','A'),('joint_score_pg_lr','B')):
        if name.startswith(prefix):
            rate,scale=name[len(prefix):].split('_kl')
            return f'{label} {rate} / KL×{scale}'
    return name


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
    names=['Stage1' if i==0 else short_name(p) for i,p in enumerate(paths)]
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


def candidate_grid(summary, output):
    """全部30配置的同条件物理结果；相同权重别名如实共用一次执行。"""
    cases={r['name']:r for r in summary['cases']}
    baseline=cases['stage1']['aggregate']
    proposals=sorted({r['proposed_lr'] for r in summary['candidates']})
    scales=[1,3,10]
    metrics=[('完成数相对Stage1的变化',lambda a:a['music_completion_count']-baseline['music_completion_count'],False),
             ('总回报相对Stage1的变化(%)',lambda a:100*(a['reward_sum']/baseline['reward_sum']-1),False),
             ('消费参考硬越界峰值增量(rad)',lambda a:a['joint_limits']['consumed_reference']['hard']['max_excess_rad']-
              baseline['joint_limits']['consumed_reference']['hard']['max_excess_rad'],True),
             ('实际机器人硬越界峰值增量(rad)',lambda a:a['joint_limits']['actual']['hard']['max_excess_rad']-
              baseline['joint_limits']['actual']['hard']['max_excess_rad'],True)]
    fig,axes=plt.subplots(2,4,figsize=(20,8))
    for column,(title,metric,reversed_color) in enumerate(metrics):
        matrices=[]
        for objective in ('coordinate_dppo','joint_score_pg'):
            values=np.full((3,5),np.nan)
            for c in summary['candidates']:
                if c['objective']==objective and c['physics_name'] in cases:
                    values[scales.index(c['kl_multiplier']),proposals.index(c['proposed_lr'])]=metric(cases[c['physics_name']]['aggregate'])
            matrices.append(values)
        bound=max(float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0. for v in matrices)
        bound=max(bound,1e-8)
        for row,values in enumerate(matrices):
            ax=axes[row,column]
            im=ax.imshow(values,cmap='RdBu_r' if reversed_color else 'RdBu',vmin=-bound,vmax=bound,aspect='auto')
            for y in range(3):
                for x in range(5):
                    v=values[y,x]
                    label='待完成' if not np.isfinite(v) else f'{v:+.0f}' if column==0 else f'{v:+.2f}' if column==1 else f'{v:+.4f}'
                    ax.text(x,y,label,ha='center',va='center',fontsize=10,
                            color='white' if np.isfinite(v) and abs(v)>.55*bound else '#222222')
            ax.set_xticks(range(5),['5e-8','3e-7','1e-6','3e-6','1e-5'])
            ax.set_yticks(range(3),['原KL','3倍KL','10倍KL'])
            ax.set_xlabel('提议统一Actor学习率（实际接受值见回溯图）',fontsize=9)
            ax.set_title(('A 坐标DPPO' if row==0 else 'B joint-score')+' · '+title,fontsize=11)
            fig.colorbar(im,ax=ax,fraction=.04,pad=.025)
    fig.suptitle(f"全部候选的真实配对物理结果：Stage1完成{baseline['music_completion_count']}/32，"
                 f"总回报{baseline['reward_sum']:.2f}；每组16首音乐×2seed",fontsize=15)
    fig.text(.04,.01,'蓝色表示该项相对改善，红色表示退化；各项必须同时判断。3倍/10倍KL只属探索，'
             '同权重别名共用实测结果。\n状态：'+summary['status']+'；选优使用同一音乐组，不构成独立确认或正式训练许可。',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95));save(fig,output/'physical_candidate_grid')


def focused(paths,case,output):
    if len(paths)!=2:raise ValueError('Focused trajectory plot compares Stage1 and exactly one candidate')
    limits=json.loads(case.read_text())['joint_limits']
    joint=limits['joint_names'].index('r_arm_yaw_joint')
    bounds=limits['joint_hard_limits_rad'][joint]
    keys=[('075',42),('075',1729),('138',42),('138',1729)]
    fig,axes=plt.subplots(4,4,figsize=(20,13),sharex='col')
    provenance=[]
    def yaw(quat):
        w,x,y,z=np.asarray(quat).T
        return np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    for path,color in zip(paths,('#777777','#2469A0')):
        data=json.loads(path.read_text());tasks={t['task_id']:t for t in data['tasks']}
        trajectories={(r['sample'],r['seed']):r for r in data['yaw']['focused_trajectories']}
        for column,key in enumerate(keys):
            info=trajectories[key];source=Path(info['path'])
            if not source.is_file():source=path.parent/source.name
            if hashlib.sha256(source.read_bytes()).hexdigest()!=info['sha256']:
                raise ValueError('Physical trajectory bytes changed')
            with np.load(source) as arrays:
                seconds=(arrays['tick']-arrays['tick'][0])/600+.02
                label=short_name(path)
                axes[0,column].plot(seconds,arrays['yaw'],color=color,label=label,linewidth=1.)
                if tasks[info['task_id']]['physical_failure']:
                    axes[0,column].plot(seconds[-1],arrays['yaw'][-1],'x',color=color,markersize=9,mew=2)
                actual=yaw(arrays['actual_quat']);reference=yaw(arrays['reference_quat'])
                axes[1,column].plot(seconds,actual-actual[0],color=color,linewidth=1.)
                axes[1,column].plot(seconds,reference-reference[0],color=color,linestyle='--',linewidth=.9)
                error=np.sqrt(np.mean((arrays['actual']-arrays['reference'])**2,axis=1))
                axes[2,column].plot(seconds,error,color=color,linewidth=1.)
                axes[3,column].plot(seconds,arrays['actual'][:,joint],color=color,linewidth=1.)
                axes[3,column].plot(seconds,arrays['reference'][:,joint],color=color,linestyle='--',linewidth=.9)
            provenance.append(dict(path=str(source),sha256=info['sha256']))
    for column,(sample,seed) in enumerate(keys):
        axes[0,column].set_title(f'FineDance {sample} · seed {seed}')
        axes[0,column].legend(fontsize=8)
        for row,label in enumerate(('实际yaw跟踪误差(rad)','相对初始朝向的转角(rad)',
                                    '21关节跟踪RMS(rad)','右臂yaw关节角(rad)')):
            axes[row,column].set_ylabel(label);axes[row,column].grid(alpha=.2)
        for bound in bounds:axes[3,column].axhline(bound,color='#A13737',linestyle=':',linewidth=1.)
        axes[3,column].set_xlabel('本次音乐实际执行时间(s)')
    fig.suptitle('真实Isaac闭环：FineDance075/138实际状态与GMT消费参考',fontsize=17)
    fig.text(.07,.003,'灰色：Stage1；蓝色：候选。第2/4行实线为实际状态、虚线为消费参考；红点线为运行时硬限位。'
             '\n×标记真实物理失败；终止后没有补零或延长。闭环状态随模型变化，这不是同观测开环差分。',fontsize=11)
    fig.tight_layout(rect=(0,.04,1,.96));save(fig,output/'finedance_trajectories')
    (output/'trajectory_sources.json').write_text(json.dumps(dict(sources=provenance,
        analyses=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        joint_limits_case=str(case),joint_limits_case_sha256=hashlib.sha256(case.read_bytes()).hexdigest()),indent=2))


def critic_failure(paths,output):
    """固定Critic的CPU预测与同一策略真实整曲MC对照，横轴不假设固定执行时长。"""
    values=json.loads(paths[0].read_text());physical=json.loads(paths[1].read_text())
    if (values['actor']!=physical['actor'] or values['source_report_sha256']!=physical['source']['sha256']
            or values['execution']['device']!='cpu'):
        raise ValueError('Critic/physics plot must identify the same Actor and physical trajectories')
    keys=[('075',42),('075',1729),('138',42),('138',1729)]
    fig,axes=plt.subplots(2,2,figsize=(12,8));summaries=[]
    for ax,(sample,seed) in zip(axes.flat,keys):
        task=next(t for t in physical['tasks'] if t['source']=='FineDance' and t['sample']==sample and t['seed']==seed)
        rows=sorted((r for r in values['rows'] if r['task_id']==task['task_id']),key=lambda r:r['decision'])
        x=np.asarray([r['decision'] for r in rows]);pred=np.asarray([r['prediction'] for r in rows]);target=np.asarray([r['target'] for r in rows])
        if not len(rows) or not np.array_equal(x,np.arange(len(rows))):raise ValueError('Missing complete MC decisions')
        error=pred-target
        ax.plot(x,pred,color='#2469A0',label='固定预热Critic预测（CPU）',linewidth=1.3)
        ax.plot(x,target,color='#C66E2B',label='该真实整曲的折扣MC回报',linewidth=1.3)
        if task['physical_failure']:
            ax.plot(x[-1],target[-1],'x',color='#B22F35',markersize=9,mew=2,label='真实物理失败')
            ax.annotate(f'终止前预测 {pred[-1]:.1f}\n实际MC {target[-1]:.1f}',
                (x[-1],pred[-1]),xytext=(-115,8),textcoords='offset points',fontsize=9,
                arrowprops=dict(arrowstyle='->',color='#777777'))
        ax.set_title(f'FineDance {sample} · seed {seed} · '+('yaw失败' if task['physical_failure'] else '音乐完成'))
        ax.set_xlabel('上层决策编号（从0开始）');ax.set_ylabel('折扣累计回报');ax.grid(alpha=.2);ax.legend(fontsize=8)
        summaries.append(dict(task_id=task['task_id'],sample=sample,seed=seed,end_reason=task['reason'],
            states=len(rows),mse=float(np.mean(error**2)),bias=float(np.mean(error)),
            final_prediction=float(pred[-1]),final_MC=float(target[-1]),
            last20_decisions_bias=float(error[-20:].mean())))
    fig.suptitle('初始Stage1闭环：预热Critic能否预见完整歌曲中的失败？',fontsize=15)
    fig.text(.06,.015,'MC使用实际50Hz积分奖励、原时间折扣及真实终止；不自举，不再次乘dt。'
        '\nCPU重算仅为补充诊断；跨设备1e-4精度检查未通过，不能冒充线上GPU原预测。单条MC不是无噪声价值真值。',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95));save(fig,output/'critic_failure_MC')
    (output/'critic_MC_sources.json').write_text(json.dumps(dict(tasks=summaries,
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]),ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,nargs='*',default=[])
    parser.add_argument('--summary',type=Path,help='严格汇总后的轻量物理JSON，绘制全部候选矩阵')
    parser.add_argument('--focus',type=Path,nargs=2,help='Stage1和一个候选的analysis.json；读取已核验实际轨迹')
    parser.add_argument('--critic',type=Path,nargs=2,help='CPU Critic诊断和同一物理analysis.json；绘制完整MC轨迹')
    parser.add_argument('--joint-limits-case',type=Path,help='真实采集的case.json，读取具名权威硬限位')
    parser.add_argument('--font',type=Path,help='无系统中文字体时明确指定Noto CJK字体文件')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    font_path=str(args.font) if args.font else subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font_path)
    font_name=font_manager.FontProperties(fname=font_path).get_name()
    if 'CJK' not in font_name:raise RuntimeError('A real CJK font is required for the Chinese report')
    plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,
                         'svg.fonttype':'path','figure.facecolor':'white'})
    fixed_scan(json.loads(args.scan.read_text()),args.output)
    if args.analysis:physics(args.analysis,args.output)
    if args.summary:candidate_grid(json.loads(args.summary.read_text()),args.output)
    if args.critic:critic_failure(args.critic,args.output)
    if args.focus:
        if not args.joint_limits_case:raise ValueError('Focused plot requires authoritative runtime joint limits')
        focused(args.focus,args.joint_limits_case,args.output)


if __name__=='__main__':main()
