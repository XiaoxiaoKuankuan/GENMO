"""将已核验的随机起点二十轮结果绘制为中文PNG/SVG。

输入analyze_stage2_random20.py的完整只读review，使用第0/4/8/12/16/20轮第一次
固定评估；第20轮结束重评仍保留在JSON中，不当作额外独立样本。分别绘制机器人
整曲表现、真实接受的Actor优化、当前策略MC价值和歌曲前中后段覆盖，防止用loss
代替物理收益或用随机抽样次数代替执行时间。图中不填补失败后数据或缺测指标。
本工具只使用CPU，不加载模型或启动训练；每个输出目录必须新建，中文字体必须
实际存在。所有图包含适用范围，单训练种子和16首音乐不能冒充正式长期验收。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.eval.plot_stage2_actor_lr import save


def plots(result, output):
    if result['status']!='completed' or result['accepted_optimizer_steps']!=20:
        raise ValueError('Plots require the completed, audited twenty-round experiment')
    output.mkdir(parents=True,exist_ok=False)
    first={}
    for row in result['evaluations']:first.setdefault(row['iteration'],row)
    values=[first[i] for i in sorted(first)];steps=np.array(sorted(first))
    assert steps.tolist()==[0,4,8,12,16,20]
    base=values[0]['aggregate']
    fig,axes=plt.subplots(2,3,figsize=(15,8.5));axes=axes.flatten()
    complete=[r['aggregate']['music_completion_count'] for r in values]
    axes[0].plot(steps,complete,'o-');axes[0].axhline(complete[0],color='gray',ls=':')
    axes[0].set_ylim(min(complete)-1,32.5);axes[0].set_ylabel('完成任务数 / 32')
    axes[0].set_title('完整歌曲完成率：先看是否新增失败')
    gain=[0.]+[r['paired']['reward']['paired_mean'] for r in values[1:]]
    lower=[0.]+[r['paired']['reward']['lower95'] for r in values[1:]]
    upper=[0.]+[r['paired']['reward']['upper95'] for r in values[1:]]
    axes[1].plot(steps,gain,'o-');axes[1].fill_between(steps,lower,upper,alpha=.2)
    axes[1].axhline(0,color='gray',ls=':');axes[1].set_ylabel('每任务总回报差（分）')
    axes[1].set_title('同曲配对收益：阴影为音乐组95%区间')
    axes[2].plot(steps,[r['aggregate']['reward_per_executed_second'] for r in values],'o-')
    axes[2].set_ylabel('总积分回报 / 实际执行秒数');axes[2].set_title('每秒分数要和完成率一起看')
    base_activity=base['activity']['actual_activity_rad_s']['mean']
    axes[3].plot(steps,[r['aggregate']['activity']['actual_activity_rad_s']['mean']/base_activity for r in values],'o-')
    axes[3].axhline(1,color='gray',ls=':');axes[3].set_ylabel('实际动作活动度 / 初始Stage1')
    axes[3].set_title('动作有没有缩小：1为初始水平')
    for domain,label in [('consumed_reference','GMT实际消费参考'),('generated','完整可修改生成参考')]:
        axes[4].plot(steps,[np.rad2deg(r['aggregate']['joint_limits'][domain]['hard']['max_excess_rad']) for r in values],'o-',label=label)
    axes[4].legend(fontsize=9);axes[4].set_ylabel('最大硬限位越界（度）');axes[4].set_title('参考越界：越小越好')
    axes[5].plot(steps,[np.rad2deg(r['aggregate']['joint_limits']['actual']['hard']['max_excess_rad']) for r in values],'o-')
    axes[5].set_ylabel('最大硬限位越界（度）');axes[5].set_title('真实机器人关节越界')
    for ax in axes:ax.set_xticks(steps);ax.set_xlabel('已接受Actor更新轮数');ax.grid(alpha=.2)
    fig.suptitle('随机起点 + 统一Actor LR 3e-7 + 三倍KL：完整歌曲物理对照',fontsize=16)
    fig.text(.5,.005,'同一新奖励、同一物理后端、16首独立音乐×2seed；单训练seed，未证明长期收敛。',ha='center')
    fig.tight_layout(rect=(0,.035,1,.95));save(fig,output/'physical_comparison')

    rows=[r for r in result['rows'] if r['mode']=='dppo_bc'];x=[r['iteration'] for r in rows]
    fig,axes=plt.subplots(2,3,figsize=(15,8.5));axes=axes.flatten()
    axes[0].plot(x,[r['backtracking']['accepted_lr'] for r in rows],'o-',ms=3)
    axes[0].axhline(3e-7,ls=':',color='gray',label='提议3e-7');axes[0].legend(fontsize=9)
    axes[0].set_ylabel('实际接受的统一学习率');axes[0].set_title('有没有反复退回极小学习率')
    for field,label,bound in [('mean_coordinate_kl','四步平均',.0009),('maximum_step_mean_coordinate_kl','最大单步均值',.0018)]:
        line=axes[1].plot(x,[r['final_kl'][field] for r in rows],label=label)[0]
        axes[1].axhline(bound,color=line.get_color(),ls=':',alpha=.7)
    line=axes[1].plot(x,[r['final_kl']['coordinate_kl']['p95'] for r in rows],label='四步P95')[0]
    axes[1].axhline(.0021,color=line.get_color(),ls=':',alpha=.7)
    axes[1].set_title('实际四步KL及对应硬上限');axes[1].legend(fontsize=9)
    axes[2].plot(x,[np.rad2deg(r['backtracking']['attempts'][-1]['diagnostics']['outputs']['final_joint_rms_mean']) for r in rows],'o-',ms=3)
    axes[2].set_ylabel('同观测/同噪声最终关节变化RMS（度）');axes[2].set_title('Actor输出是否真正变化')
    axes[3].plot(x,[100*r['backtracking']['attempts'][-1]['diagnostics']['parameters']['modules']['all']['changed_fraction'] for r in rows],'o-',ms=3)
    axes[3].set_ylabel('FP32参数实际改变比例（%）');axes[3].set_title('参数变化不等于物理收益')
    mc=result['independent_current_policy_MC']
    axes[4].plot([0]+[r['iteration'] for r in mc],[result['preheat']['after']['explained_variance']]+[r['metrics']['explained_variance'] for r in mc],'o-')
    axes[4].set_ylabel('完整MC Explained Variance');axes[4].set_title('当前策略的独立价值诊断')
    axes[5].plot(x,[r['wall_seconds'] for r in rows],'o-',ms=3,label='完整训练轮')
    axes[5].plot(x,[r['actor_seconds'] for r in rows],label='Actor（含回溯）')
    axes[5].plot(x,[r['critic_seconds'] for r in rows],label='Critic');axes[5].legend(fontsize=9)
    axes[5].set_ylabel('墙钟秒');axes[5].set_title('每轮计算时间（周期评估/MC另计）')
    for ax in axes:ax.set_xlabel('已接受Actor更新轮数');ax.grid(alpha=.2)
    fig.suptitle('数值优化、价值预测与墙钟成本',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,output/'optimization')

    coverage=result['sampling_coverage']['by_source'];sources=list(coverage)
    fig,ax=plt.subplots(figsize=(11,5));positions=np.arange(len(sources))
    for i,label in enumerate(['歌曲前1/3','歌曲中1/3','歌曲后1/3']):
        ax.bar(positions+(i-1)*.25,[coverage[s]['unique_coverage_by_third'][i]*100 for s in sources],width=.25,label=label)
    ax.set_xticks(positions,labels=sources);ax.set_ylabel('该时间段真实唯一覆盖（%）');ax.legend()
    ax.set_title('20轮实际Actor采集覆盖：跨rank去重，排除预热、MC与评估')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,output/'music_time_coverage')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font);name=font_manager.FontProperties(fname=font).get_name()
    if 'CJK' not in name:raise ValueError('A real Chinese font is required')
    plt.rcParams.update({'font.family':name,'axes.unicode_minus':False,'svg.fonttype':'path','figure.facecolor':'white'})
    plots(json.loads(args.review.read_text()),args.output)


if __name__=='__main__':main()
