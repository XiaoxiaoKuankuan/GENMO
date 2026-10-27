"""只读汇总Stage2-A有界WM对照与真实方向证据，不训练网络、不重写实验。

输入为同一次真实数据准备及A01至D02独立进程的JSON产物；按0/25/.../150保留
各版本状态误差、六项奖励误差、完整目标有限响应与Newton方向门槛。奖励端点按
实际可辨识mask分别统计H8/H32/H64，不把无标签当零误差；原始JSON仍是逐样本证据。
另区分固定历史参数探针和当前WM新提出的完整H32 Adam候选，后者没有接受更新。
输出到全新目录：机器可读summary.json、可直接阅读的表格和独立PNG/PDF诊断图。
图只展示WM及真实配对证据，不替代整首歌曲质量评价，也不自动选择或启动Adapter。
"""

import argparse
import json
import math
from pathlib import Path

NAMES = ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori', 'full_objective')


def read(path):
    return json.loads(path.read_text())


def endpoint_metrics(endpoint):
    result = {}
    for i, name in enumerate(NAMES):
        pairs = [(a[i], p[i]) for ar, pr, mr in zip(endpoint['actual_delta'], endpoint['predicted_delta'],
                                                    endpoint['identifiable'])
                 for a, p, m in zip(ar, pr, mr) if m[i]]
        result[name] = dict(count=len(pairs),
            delta_rmse=math.sqrt(sum((a-p)**2 for a, p in pairs)/len(pairs)) if pairs else None,
            high_snr_opposites=sum(a*p < 0 and abs(p) > 1e-8 for a, p in pairs),
            mean_real_delta=sum(a for a, _ in pairs)/len(pairs) if pairs else None,
            mean_predicted_delta=sum(p for _, p in pairs)/len(pairs) if pairs else None)
    return result


def summarize(root):
    result = dict(data=read(root/'data01/data_summary.json'),
                  calibration=read(root/'calibration02/training_gradient_calibration.json'), runs={}, candidates={})
    for suffix in ('01', '02'):
        for group in 'ABCD':
            folder = root/(group+suffix)
            manifest, status = read(folder/'manifest.json'), read(folder/'status.json')
            versions = []
            for path in sorted(folder.glob('validation_*.json')):
                x = read(path)
                d = x['physical_parameter_direction']
                versions.append(dict(update=x['wm_updates'], wm_sha256=x['wm_sha256'],
                    one_step=x['true_state_one_step'], state=x['state_errors'],
                    reward_fields=x['reward_response']['reward_response_fields'],
                    reward_endpoints={h: endpoint_metrics(e) for h, e in
                        x['reward_response']['reward_response_endpoints'].items()},
                    physical_gates=d['gates'], physical_cases=d['cases']))
            result['runs'][group+suffix] = dict(manifest=manifest,status=status,versions=versions)
            assert [v['update'] for v in versions] == list(range(0, 151, 25)), folder
    for group in 'ABCD':
        folder=root/('candidate_'+group+'02')
        value=read(folder/'validation_150.json')
        result['candidates'][group] = dict(status=read(folder/'status.json'),
            proposal=read(folder/'candidate_report.json'),direction=value['physical_parameter_direction'])
    result['integration']=read(root/'integration64_02/integration.json')
    trace_transitions=0
    for name in ('trace_D02_new','trace_D02_known'):
        trace,manifest=read(root/name/'three_path_H1_H64.json'),read(root/name/'manifest.json')
        trace_transitions+=manifest['num_envs']*sum(prefix+len(t['steps'])
            for prefix,t in zip(trace['prefixes'],trace['traces']))
    result['initial_WM_identities']=sorted({r['manifest']['initial_WM'] for r in result['runs'].values()})
    result['fixed_label_identities']=sorted({r['manifest']['data_sha256'] for r in result['runs'].values()})
    result['total_real_transitions']=(result['data']['real_transitions']+
        sum(r['status']['new_audit_transitions'] for r in result['runs'].values())+
        sum(r['status']['new_audit_transitions'] for r in result['candidates'].values())+
        result['integration']['real_transitions']+trace_transitions)
    result['trace_real_transitions']=trace_transitions
    result['accepted_adapter_updates']=sum(r['status']['accepted_updates'] for r in result['runs'].values())
    assert result['accepted_adapter_updates']==0 and all(r['status']['accepted_updates']==0
        for r in result['candidates'].values()), '本报告不能把真实接受的更新漏报为零'
    return result


def tables(result):
    lines=['# 固定标签WM对照实际结果', '',
        '以下为真实标签上的预测误差与Newton方向；不是Adapter整歌收益。根位置/姿态为最大误差，q/qd/w/body_w为RMSE。', '',
        '|组/版本|一步归一化损失|H32 qd RMSE|H32 根w RMSE|H32 body_w RMSE|H32 完整奖励差RMSE|H32已知探针反向|通过|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, run in result['runs'].items():
        for v in run['versions']:
            s=v['state']['32'];g=v['physical_gates']['32']
            e=v['reward_endpoints']['32']['full_objective']
            lines.append(f"|{name}/{v['update']}|{v['one_step']['one_step_loss']:.7f}|"
                f"{s['joint_velocity_rad_s']:.6f}|{s['root_angular_velocity_rad_s']:.6f}|"
                f"{s['body_angular_velocity_rad_s']:.6f}|{e['delta_rmse']:.7f}|{g['high_snr_opposites']}/{g['cases']}|{g['passed']}|")
    lines += ['', '## 当前WM重新提出的H32完整Adam候选', '',
              '|组|FineDance189真实变化|WM预测变化|重复噪声|H32反向数|已接受更新|',
              '|---|---:|---:|---:|---:|---:|']
    for name, entry in result['candidates'].items():
        c=next(c for c in entry['direction']['cases'] if c['task']=='FineDance_189' and c['prefix']==50 and c['horizon']==32)
        lines.append(f"|{name}|{c['actual_full_objective_delta']:.9f}|{c['predicted_full_objective_delta']:.9f}|"
                     f"{c['repeat_noise']:.3g}|{entry['direction']['gates']['32']['high_snr_opposites']}/8|0|")
    return '\n'.join(lines)+'\n'


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for group in 'ABCD':
        rows=result['runs'][group+'02']['versions']; x=[r['update'] for r in rows]
        axes[0,0].plot(x,[r['state']['32']['joint_velocity_rad_s'] for r in rows],marker='o',label=group)
        axes[0,1].plot(x,[r['reward_endpoints']['32']['full_objective']['delta_rmse'] for r in rows],marker='o',label=group)
        c=[next(c for c in r['physical_cases'] if c['task']=='FineDance_189' and c['prefix']==50 and c['horizon']==32) for r in rows]
        axes[1,0].plot(x,[v['predicted_full_objective_delta'] for v in c],marker='o',label=group+' WM')
    axes[1,0].plot(x,[v['actual_full_objective_delta'] for v in c],color='black',linestyle='--',label='Newton')
    actual,predicted=[],[]
    for group in 'ABCD':
        c=next(c for c in result['candidates'][group]['direction']['cases']
               if c['task']=='FineDance_189' and c['prefix']==50 and c['horizon']==32)
        actual.append(c['actual_full_objective_delta']);predicted.append(c['predicted_full_objective_delta'])
    axes[1,1].bar([i-.18 for i in range(4)],actual,width=.36,label='Newton',color='#ba3541')
    axes[1,1].bar([i+.18 for i in range(4)],predicted,width=.36,label='WM',color='#2779ad')
    axes[1,1].set_xticks(range(4),list('ABCD'))
    titles=('Fixed validation: H32 joint velocity RMSE','Fixed validation: H32 reward-response RMSE',
            'Known probe: FineDance189, H32 objective change','New Adam candidate: FineDance189, H32 change')
    for i,(ax,title) in enumerate(zip(axes.flat,titles)):
        ax.set_title(title,fontsize=11);ax.grid(alpha=.2);ax.legend(fontsize=8)
        if i<3:ax.set_xlabel('Independent WM updates')
    axes[0,0].set_ylabel('rad/s');axes[0,1].set_ylabel('discounted reward difference')
    axes[1,0].axhline(0,color='gray',linewidth=.6);axes[1,1].axhline(0,color='gray',linewidth=.6)
    fig.suptitle('Stage2-A calibrated A/B/C/D: zero accepted Adapter updates',fontsize=14)
    fig.savefig(output/'wm_ablation.png',dpi=180);fig.savefig(output/'wm_ablation.pdf')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true');args=p.parse_args()
    result=summarize(args.root);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    (args.output/'tables.md').write_text(tables(result))
    if args.plot:plot(result,args.output)
    print(json.dumps(dict(output=str(args.output),real_transitions=result['total_real_transitions'],
                         accepted_adapter_updates=result['accepted_adapter_updates'])))


if __name__=='__main__':
    main()
