#!/usr/bin/env python3
"""分析MuJoCo闭环中的GENMO参考抖动、真实执行与提前停止原因。

输入为本机sim2sim的episode目录，读取50Hz control_trace.npz、report.json，
以及可选的generated_plans.npz（逐次30Hz原始世界系GENMO输出）。分别统计参考
和实际关节的速度、加速度、最大单步变化，使用去趋势Welch功率谱量化6Hz以上
的速度变化，避免直接对带端点跳变的整曲FFT作判断。30Hz源计划仅分析固定前缀
之后的新生成段，并区分前缀衔接与段内变化；50Hz线性插值本身也可能带来加速度
折点，因此不同采样率的加速度不作为模型优劣的直接对照。

输出独立JSON和PNG分析图，不修改轨迹、权重、音乐、仿真状态或训练配置。
高频成分可能包含真实快动作，数值指标用于定位，不自动等同于不合理抖动。
脚本不调用任何模型，不执行动力学，也不把此前用户运行冒充本次新测试。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch


def metrics(q, fps):
    q=np.asarray(q,dtype=np.float64)
    velocity=np.diff(q,axis=0)*fps
    acceleration=np.diff(velocity,axis=0)*fps
    frequencies,power=welch(velocity,fs=fps,nperseg=min(256,len(velocity)),
                            axis=0,detrend='linear')
    result=dict(frames=len(q),fps=fps,
        velocity_rms_rad_s=float(np.sqrt(np.mean(velocity**2))),
        acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
        max_step_rad=float(np.max(np.abs(np.diff(q,axis=0)))),
        step_abs_p99_rad=float(np.quantile(np.abs(np.diff(q,axis=0)),.99)),
        velocity_power_above_6hz_fraction=float(power[frequencies>=6].sum()/max(power.sum(),1e-30)))
    return result,frequencies,power


def analyze(directory,output):
    directory=Path(directory);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((directory/'report.json').read_text())
    with np.load(directory/'control_trace.npz',allow_pickle=False) as trace:
        data={k:trace[k].copy() for k in trace.files}
    # 去掉站立预热和启动最初2秒；两个对象严格使用相同时间和关节集合。
    mask=data['music_seconds']>=2
    if mask.sum()<10:mask=data['music_seconds']>0
    actual=data['qpos'][mask,7:];reference=data['reference_qpos'][mask,7:]
    ref,freq,ref_power=metrics(reference,50);act,_,act_power=metrics(actual,50)
    largest=np.argsort(np.max(np.abs(np.diff(reference,axis=0)),axis=0))[-3:][::-1]
    result=dict(episode=str(directory.resolve()),reason=report['reason'],
        music=report['music']['audio'],simulated_seconds=report['simulated_seconds'],
        real_time_factor=report['real_time_factor'],warnings=report.get('warnings',{}),
        physical_failure_steps=report.get('physical_failure_steps'),
        fallback_steps=report.get('reference_fallback_steps'),
        reference_50hz=ref,actual_50hz=act,largest_reference_joint_indices=largest.tolist(),
        final_errors={key:float(data[key][-1]) for key in
            ('root_height_error_m','non_yaw_orientation_error_rad','yaw_error_rad',
             'end_effector_relative_height_error_m')},
        scope='derivatives/PSD are diagnostic; fast intended motion is not automatically noise')
    if (directory/'generated_plans.npz').exists():
        with np.load(directory/'generated_plans.npz',allow_pickle=False) as source:
            values=source['qpos_world'];prefixes=source['prefix_frames']
            speeds=[];accelerations=[];boundary=[];interior=[]
            for q,p in zip(values,prefixes):
                free=q[int(p):,7:].astype(np.float64)
                if len(free)>2:
                    speeds.extend((np.diff(free,axis=0)*30).ravel())
                    accelerations.extend((np.diff(free,n=2,axis=0)*900).ravel())
                    interior.extend(np.abs(np.diff(free,axis=0)).ravel())
                if p:boundary.extend(np.abs(q[int(p),7:]-q[int(p)-1,7:]))
            result['raw_generated_30hz']=dict(plans=len(values),committed=int(source['committed'].sum()),
                free_velocity_rms_rad_s=float(np.sqrt(np.mean(np.square(speeds)))),
                free_acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.square(accelerations)))),
                free_step_abs_p99_rad=float(np.quantile(interior,.99)),
                free_step_abs_max_rad=float(np.max(interior)),
                prefix_boundary_step_p99_rad=float(np.quantile(boundary,.99)),
                prefix_boundary_step_max_rad=float(np.max(boundary)))
    # 真实提交时刻已经被保护：记录提交处的单步变化，不能仅凭计划更新就断言发生跳变。
    steps=np.max(np.abs(np.diff(data['reference_qpos'][:,7:],axis=0)),axis=1)
    seam=[]
    for request in report['requests']:
        i=int(np.searchsorted(data['tick'],request['commit_tick']))
        if 0<i<len(steps):seam.append(float(steps[i-1]))
    result['commit_tick_step_max_rad']=max(seam,default=None)
    (output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,10),constrained_layout=True)
    time=data['music_seconds'][mask]
    for i in largest:
        axes[0].plot(time,reference[:,i],label=f'GENMO joint {i}',linewidth=.8)
        axes[0].plot(time,actual[:,i],linestyle='--',linewidth=.7,label=f'Actual joint {i}')
    axes[0].set(ylabel='Joint angle (rad)',title='Raw committed GENMO reference and actual dynamics')
    axes[0].legend(ncol=3,fontsize=8)
    axes[1].semilogy(freq,ref_power.mean(1)+1e-15,label='GENMO reference velocity')
    axes[1].semilogy(freq,act_power.mean(1)+1e-15,label='Actual velocity')
    axes[1].axvline(6,color='grey',linestyle=':');axes[1].legend()
    axes[1].set(xlabel='Frequency (Hz)',ylabel='Velocity PSD')
    for key in ('non_yaw_orientation_error_rad','end_effector_relative_height_error_m'):
        axes[2].plot(data['music_seconds'],data[key],label=key)
    axes[2].axhline(.6,color='grey',linestyle=':');axes[2].axhline(.15,color='grey',linestyle=':')
    axes[2].set(xlabel='Music time (s)',ylabel='Tracking error');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(output/'analysis.png',dpi=150);plt.close(fig)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(args.episode,args.output),ensure_ascii=False,indent=2))
