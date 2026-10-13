"""固定行为策略完整物理 episode 的因果价值样本与真实 MC 回报。

只保存生成前的 Stage1 条件、真实 50Hz 积分奖励和明确的执行/终止身份，不保留
完整扩散链。普通 PPO 片段可以在同一行为版本下拼接成 MC episode；歌曲结束和
真实失败的尾部取零，不 bootstrap。行政截断整段排除，不能把已有片段冒充完整
回报。Actor 发布新版本后旧银行明确失效，不能继续当作新策略的无偏价值监督。
"""
from __future__ import annotations
import copy
from types import SimpleNamespace
import torch


def complete_mc_returns(rows, *, gamma_upper=.99):
    if not rows or not 0<float(gamma_upper)<=1:
        raise ValueError('Complete episode and valid time discount required')
    final=rows[-1]
    if not final.terminated or final.truncated:
        raise ValueError('Administrative or incomplete episode has no unbiased MC label')
    policy=rows[0].identity['policy_version'];episode=rows[0].identity['episode_id']
    gamma=float(gamma_upper)**(1/25)
    returns=torch.empty(len(rows),dtype=torch.float64);future=torch.zeros((),dtype=torch.float64)
    previous=None
    for row in rows:
        if row.identity['policy_version']!=policy or row.identity['episode_id']!=episode:
            raise ValueError('MC cannot cross episode or Actor behavior versions')
        if previous is not None and previous!=row.control_tick_begin:
            raise ValueError('MC physical controls are discontinuous')
        if row.control_tick_end-row.control_tick_begin!=12*row.executed_control_steps:
            raise ValueError('MC elapsed ticks disagree with real control count')
        if row.metadata.get('physical_episode_administrative_truncation'):
            raise ValueError('MC excludes physical administrative truncation')
        previous=row.control_tick_end
    for i in range(len(rows)-1,-1,-1):
        row=rows[i];r=torch.as_tensor(row.rewards,dtype=torch.float64).detach().cpu()
        n=row.executed_control_steps
        if r.shape!=(n,) or not torch.isfinite(r).all():raise ValueError('Invalid actual MC rewards')
        event=float(row.metadata.get('event_reward',0.))
        within=(r*gamma**torch.arange(n,dtype=torch.float64)).sum()+gamma**max(n-1,0)*event
        future=within+gamma**n*future
        returns[i]=future
    if not torch.isfinite(returns).all():raise FloatingPointError('Nonfinite complete MC return')
    return returns


class CompleteEpisodeBank:
    def __init__(self, *, policy_version, kernel_identity, gamma_upper=.99):
        self.policy_version=policy_version;self.kernel_identity=copy.deepcopy(kernel_identity)
        self.gamma_upper=gamma_upper;self.pending={};self.episodes=[];self.excluded=[]

    def require_behavior(self, version, kernel):
        if version!=self.policy_version or kernel!=self.kernel_identity:
            raise ValueError('Old MC labels belong to a different fixed Actor behavior')

    def append(self, task, row):
        self.require_behavior(row.identity['policy_version'],row.metadata['sampler_trace']['kernel_config'])
        key=(row.identity['backend_session_id'],row.identity['episode_id'])
        item=SimpleNamespace(context={k:v.detach().cpu().clone() for k,v in row.context.items()},
            identity=copy.deepcopy(row.identity),rewards=row.rewards.detach().cpu().clone(),
            executed_control_steps=row.executed_control_steps,control_tick_begin=row.control_tick_begin,
            control_tick_end=row.control_tick_end,terminated=bool(row.terminated),truncated=bool(row.truncated),
            next_context=None if row.next_context is None else {k:v.detach().cpu().clone() for k,v in row.next_context.items()},
            transition_valid=True,metadata={k:copy.deepcopy(row.metadata[k]) for k in
                ('remaining_music_seconds','next_remaining_music_seconds','event_reward','physical_episode_administrative_truncation') if k in row.metadata})
        if not row.transition_valid:raise ValueError('Invalid physics cannot enter MC bank')
        rows=self.pending.setdefault(key,[]);rows.append(item)
        physical_admin=row.metadata.get('physical_episode_administrative_truncation') or row.reason in {
            'physical_safety_limit','reference_horizon_truncated','bounded_boundary_wait_limit'}
        if physical_admin:
            self.excluded.append(dict(task_id=task['task_id'],reason=row.reason,states=len(rows)))
            self.pending.pop(key);return
        if row.terminated:
            # 片段truncated仅切GAE；完整episode真实终止时明确取消这一片段标志。
            rows[-1].truncated=False
            targets=complete_mc_returns(rows,gamma_upper=self.gamma_upper)
            self.episodes.append(dict(task=copy.deepcopy(task),rows=rows,returns=targets,
                end_reason=row.reason,control_steps=sum(r.executed_control_steps for r in rows)))
            self.pending.pop(key)

    def training_data(self, *, policy_version, kernel):
        self.require_behavior(policy_version,kernel)
        if self.pending:raise ValueError('Incomplete MC episodes remain; collect to true terminal or explicitly exclude')
        rows=[r for e in self.episodes for r in e['rows']]
        targets=torch.cat([e['returns'] for e in self.episodes]) if rows else torch.empty(0,dtype=torch.float64)
        return rows,targets

    def report(self):
        return dict(behavior_version=self.policy_version,complete_episodes=len(self.episodes),
            complete_states=sum(len(e['rows']) for e in self.episodes),excluded=self.excluded,
            pending_episodes=len(self.pending),real_control_steps=sum(e['control_steps'] for e in self.episodes),
            discount='gamma_upper^(actual_control_steps/25)',labels_are_integrated=True,terminal_bootstrap=0.)
