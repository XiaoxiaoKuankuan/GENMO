"""完整物理 episode 的局部价值误差与八秒截断只读对照。

输入是同一冻结行为策略已经完整执行的 MC episode，以及当前 Critic 在每个真实
生成前状态的预测。先复用生产 complete_mc_returns 核验标签，再按照实际控制时刻
切出八秒片段，复用生产 compute_gae 比较原 lambda 目标、lambda=1 自举目标与完整
MC。正常片段边界保留下一状态价值，真实终止尾值为零；不跨音乐、环境或策略版本。

输出区分失败前、正常片段边界前后、配对/站立起点与数据来源，保留可追溯的极端
高估状态。这里的八秒边界是在同一固定策略完整轨迹上重建的对照边界，不冒充实际
训练 rollout 边界。所有结果仅供价值校准诊断，不改奖励、Actor输入、GAE训练标签、
模型或优化器；小样本/单条轨迹的误差也不被解释成精确的期望价值偏差。
"""
from __future__ import annotations

import math
import torch

from .complete_episode import complete_mc_returns
from .returns import compute_gae


def episode_value_records(episode, predictions, *, gamma_upper=.99, lambda_upper=.98,
                          world_seconds=8., near_seconds=1.5):
    """用真实完整MC和同一组预测构造可核对的逐状态对照，不修改输入。"""
    rows=episode['rows'];n=len(rows)
    prediction=torch.as_tensor(predictions,dtype=torch.float64).detach().cpu()
    if prediction.shape!=(n,) or not torch.isfinite(prediction).all():
        raise ValueError('Finite aligned complete-episode value predictions required')
    if not math.isfinite(world_seconds) or world_seconds<=0 or near_seconds<=0:
        raise ValueError('Positive diagnostic physical intervals required')
    mc=complete_mc_returns(rows,gamma_upper=gamma_upper)
    stored=torch.as_tensor(episode['returns'],dtype=torch.float64).detach().cpu()
    if stored.shape!=mc.shape or not torch.equal(stored,mc):
        raise ValueError('Stored MC differs from production complete-episode returns')
    chunks=[];begin=0
    for end in range(1,n+1):
        if end==n or rows[end-1].control_tick_end-rows[begin].control_tick_begin>=world_seconds*600:
            chunks.append((begin,end));begin=end
    gae=torch.empty_like(mc);bootstrap_return=torch.empty_like(mc)
    for begin,end in chunks:
        part=rows[begin:end]
        arguments=dict(rewards=[r.rewards for r in part],values=prediction[begin:end],
            next_values=[float(prediction[i+1]) if i+1<n else 0. for i in range(begin,end)],
            executed_steps=[r.executed_control_steps for r in part],
            bootstrap_mask=[not r.terminated for r in part],
            continuation_mask=[i+1<len(part) for i in range(len(part))],
            event_rewards=[r.metadata.get('event_reward',0.) for r in part],
            gamma_upper=gamma_upper,normalize=False)
        gae[begin:end]=compute_gae(**arguments,lambda_upper=lambda_upper)['returns']
        bootstrap_return[begin:end]=compute_gae(**arguments,lambda_upper=1.)['returns']
    cuts=[rows[end-1].control_tick_end for _,end in chunks[:-1]]
    final_tick=rows[-1].control_tick_end
    failure=episode['end_reason'] not in ('music_end','task_complete')
    task=episode['task'];records=[]
    for i,row in enumerate(rows):
        # V预测对应生成前的状态；变长转移不能用结束时刻代替预测时刻。
        before_cut=any(0<cut-row.control_tick_begin<=near_seconds*600 for cut in cuts)
        after_cut=any(0<=row.control_tick_begin-cut<near_seconds*600 for cut in cuts)
        near_end=0<=final_tick-row.control_tick_begin<=near_seconds*600
        records.append(dict(task_id=task['task_id'],source=task['dataset'],
            initialization=task.get('initialization','standing'),
            identity=row.identity.copy(),control_tick_begin=row.control_tick_begin,
            control_tick_end=row.control_tick_end,terminated=bool(row.terminated),
            prediction=float(prediction[i]),mc=float(mc[i]),gae=float(gae[i]),
            lambda1_bootstrap=float(bootstrap_return[i]),
            value_error=float(prediction[i]-mc[i]),gae_error=float(gae[i]-mc[i]),
            bootstrap_error=float(bootstrap_return[i]-mc[i]),
            near_true_terminal=near_end,near_physical_failure=near_end and failure,
            before_reconstructed_cut=before_cut,after_reconstructed_cut=after_cut,
            end_reason=episode['end_reason']))
    return records


def summarize_value_credit(records):
    """统一按状态重算误差；无样本返回缺测，保留失败数而非过滤失败。"""
    def summarize(selected):
        result=dict(states=len(selected),tasks=len({r['task_id'] for r in selected}))
        for field in ('value_error','gae_error','bootstrap_error'):
            if not selected:
                result[field]=None;continue
            error=torch.tensor([r[field] for r in selected],dtype=torch.float64)
            if not torch.isfinite(error).all():raise FloatingPointError('Nonfinite local MC error')
            result[field]=dict(bias=float(error.mean()),rmse=float(error.square().mean().sqrt()),
                p95=float(torch.quantile(error,.95)),max=float(error.max()),min=float(error.min()))
        return result
    names=('near_physical_failure','near_true_terminal','before_reconstructed_cut','after_reconstructed_cut')
    groups={'all':summarize(records)}
    groups.update({name:summarize([r for r in records if r[name]]) for name in names})
    by_source={source:summarize([r for r in records if r['source']==source])
               for source in sorted({r['source'] for r in records})}
    by_initialization={mode:summarize([r for r in records if r['initialization']==mode])
                       for mode in sorted({r['initialization'] for r in records})}
    return dict(groups=groups,by_source=by_source,by_initialization=by_initialization,
        worst_overestimates=sorted(records,key=lambda r:r['value_error'],reverse=True)[:20],
        scope='current_fixed_policy_complete_MC_no_bootstrap_truth; reconstructed_8s_cuts_are_diagnostic_only',
        trajectory_noise='single_trajectory_MC_is_a_noisy_sample_of_expected_value',
        physical_failure_tasks=len({r['task_id'] for r in records if r['near_physical_failure']}))
