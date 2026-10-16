"""只读拆解固定真实rollout的GAE信号，核验全局分母与实际奖励时间语义。

每个rank先严格检查原始manifest/SHA及真实转移，依据episode、环境、Actor版本和
连续控制tick重建GAE连接，独立重算原始优势、价值目标及全局标准化。把TD误差
拆成已积分奖励与折扣价值差，再利用GAE的线性结构拆成奖励/价值两部分，记录
方差、协方差和按来源统计。大价值差不自动等于Critic错误，本工具不把短片段
bootstrap或保存的GAE标签称为完整MC，也不以相关性推断物理因果改善。

只使用CPU和既有已验收Critic的采样时预测，不重新运行网络、不改模型、不修改
rollout、奖励或old概率。读取完成后输出新不可覆盖JSON，保留来源SHA和分析源码
SHA；不与GPU作业竞争设备，不从同一数据额外做Actor更新。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.returns import compute_gae
from gem.closedloop.dppo.training_checkpoint import file_sha256


def decompose(rows,*,gamma_upper=.99,lambda_upper=.95):
    n=len(rows);continuation=[False]*n
    for i,(a,b) in enumerate(zip(rows,rows[1:])):
        continuation[i]=(a.transition_valid and b.transition_valid and not a.terminated and not a.truncated
            and all(a.identity[k]==b.identity[k] for k in ('backend_session_id','episode_id','policy_version','env_id'))
            and a.control_tick_end==b.control_tick_begin)
    common=dict(executed_steps=[r.executed_control_steps for r in rows],
        bootstrap_mask=[not r.terminated and r.next_context is not None for r in rows],
        continuation_mask=continuation,valid=[r.transition_valid for r in rows],
        gamma_upper=gamma_upper,lambda_upper=lambda_upper,normalize=False)
    rewards=[r.rewards for r in rows];values=[r.old_value for r in rows];next_values=[r.next_value for r in rows]
    events=[r.metadata.get('event_reward',0.) for r in rows]
    all_terms=compute_gae(rewards,values,next_values,event_rewards=events,**common)
    reward_terms=compute_gae(rewards,[0.]*n,[0.]*n,event_rewards=events,**common)
    value_terms=compute_gae([torch.zeros(r.executed_control_steps) for r in rows],values,next_values,**common)
    np.testing.assert_allclose((reward_terms['advantages_raw']+value_terms['advantages_raw']).numpy(),
                               all_terms['advantages_raw'].numpy(),rtol=0,atol=1e-10)
    boot=torch.as_tensor(common['bootstrap_mask'])
    discount=torch.tensor([gamma_upper**(r.executed_control_steps/25) for r in rows],dtype=torch.float64)
    shift=discount*torch.tensor(next_values,dtype=torch.float64)*boot-torch.tensor(values,dtype=torch.float64)
    output=dict(all_terms);output.update(td_value_difference=shift,
        td_error=all_terms['discounted_rewards']+shift,
        gae_reward_component=reward_terms['advantages_raw'],gae_value_component=value_terms['advantages_raw'])
    return output


def statistics(records):
    if not records:return dict(count=0)
    fields=('discounted_rewards','td_value_difference','td_error','advantages_raw',
            'gae_reward_component','gae_value_component','old_value','next_value','advantages')
    values={k:np.asarray([r[k] for r in records],dtype=float) for k in fields}
    def correlation(a,b):
        return None if np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])
    result={k:dict(mean=float(v.mean()),std=float(v.std()),minimum=float(v.min()),maximum=float(v.max()))
            for k,v in values.items()}
    result.update(count=len(records),real_control_steps=sum(r['controls'] for r in records),
        terminated_states=sum(r['terminated'] for r in records),
        administrative_states=sum(r['administrative'] for r in records),
        actual_reward_td_value_correlation=correlation(values['discounted_rewards'],values['td_value_difference']),
        advantage_immediate_reward_correlation=correlation(values['advantages_raw'],values['discounted_rewards']),
        gae_reward_value_correlation=correlation(values['gae_reward_component'],values['gae_value_component']),
        td_reward_value_covariance=float(np.mean((values['discounted_rewards']-values['discounted_rewards'].mean())*
                                                (values['td_value_difference']-values['td_value_difference'].mean()))))
    return result


def analyze(case,output):
    if output.exists():raise FileExistsError(output)
    info=json.loads((case/'case.json').read_text());records=[];provenance=[];errors=[]
    identities=[]
    for rank in info['ranks']:
        path=case/f'rank{rank["rank"]:02d}'
        if file_sha256(path/'manifest.json')!=rank['rollout_manifest_sha256']:
            raise ValueError('Fixed rank manifest changed')
        stored=load_rollout(path);rows=stored['transitions'];t=stored['targets']
        identities.append(stored['identity'])
        actual=decompose(rows,gamma_upper=stored['identity']['training']['gamma_upper'],
                         lambda_upper=stored['identity']['training']['lambda_upper'])
        for key in ('discounted_rewards','advantages_raw','returns'):
            delta=float((actual[key]-t[key]).abs().max()) if rows else 0.
            if delta>1e-9:raise ValueError('Stored GAE differs from independent episode/version/tick reconstruction')
            errors.append(delta)
        for i,row in enumerate(rows):
            if not row.transition_valid:continue
            r=dict(source=row.metadata['training_task']['dataset'],identity=row.identity,
                controls=row.executed_control_steps,terminated=row.terminated,
                administrative=row.metadata.get('physical_episode_administrative_truncation',False),
                old_value=row.old_value,next_value=row.next_value,advantages=float(t['advantages'][i]))
            r.update({key:float(actual[key][i]) for key in ('discounted_rewards','advantages_raw',
                'td_error','td_value_difference','gae_reward_component','gae_value_component')})
            records.append(r)
        provenance.append(dict(rank=rank['rank'],manifest_sha256=rank['rollout_manifest_sha256'],
                               rollout=json.loads((path/'manifest.json').read_text())))
        del actual,stored,rows,t
    if any(value!=identities[0] for value in identities):raise ValueError('Ranks do not share one behavior identity')
    raw=np.asarray([r['advantages_raw'] for r in records]);saved=np.asarray([r['advantages'] for r in records])
    normalization_error=float(np.max(np.abs(saved-(raw-raw.mean())/max(raw.std(),1e-8))))
    if normalization_error>1e-10:raise ValueError('Global valid advantage normalization differs')
    report=dict(schema='genmo.stage2.uniform_lr_advantage_audit.v1',
        source_case=str(case),source_case_sha256=file_sha256(case/'case.json'),
        implementation_sha256=file_sha256(__file__),ranks=provenance,
        gae_reconstruction_max_error=max(errors),normalization_max_error=normalization_error,
        overall=statistics(records),
        by_source={s:statistics([r for r in records if r['source']==s]) for s in sorted({r['source'] for r in records})},
        interpretation='value-difference dominance is not proof of value error; complete-current-policy MC is separate',
        scope='read_only_fixed_real_rollout_no_network_or_optimizer_execution')
    atomic_json(output,report);print(json.dumps(report['overall'],ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);analyze(args.case,args.output)
