"""Stage10 的有界内存执行接收、资源子账本与GPU训练Buffer。

普通训练不把每个世界和lane回复写SQLite；完整执行回复被接收后按原序号ACK，
训练需要的随机链、旧统计量、条件、奖励和结束信息保留到本轮完成。此处明确采用
新的仿真恢复合同：未完成rollout可丢弃，完整checkpoint恢复时重建PhysX并重新
采样，不声称恢复进程崩溃时未知的物理状态。严格诊断使用原持久journal，不走
本模块。内存journal只保留最近接收的有界身份，不积累每轮全量物理证据。

资源子账本在根进程提前持久化预占额度后使用；每次实际reserve/settle仍检查上限，
同一rank阶段成功后汇总实际消耗，异常未结算额度保持已耗。没有无限后台写队列。
GPU Buffer仅接收已有只读采样张量视图，条件变为独立设备张量，参数版本与实际
控制/四子步计数严格核验。它不将已优化的数据留在下一个策略版本中。
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import torch


class MemoryReceiveJournal:
    def __init__(self,path,*,capacity=1):
        self.path=path;self.capacity=capacity;self.entries=OrderedDict();self.closed=False
        self.last_record={'sha256':None,'storage':'ephemeral_memory'}
        self.last_timing={}

    def append_result(self,result):
        if self.closed:raise RuntimeError('Memory receiver already closed')
        body=result.get('result',result)
        if not isinstance(body,dict):body=result
        session=result.get('backend_session_id',body.get('backend_session_id'))
        sequence=result.get('mutation_seq',body.get('mutation_seq'))
        if session is None or sequence is None:raise ValueError('Execution identity is required before ACK')
        identity=(session,sequence)
        if identity in self.entries:
            if self.entries[identity] is not result:
                from .buffer import StepJournal
                from .journal_codec import FORMAT
                old=StepJournal.encode_result(self.entries[identity],format=FORMAT)
                new=StepJournal.encode_result(result,format=FORMAT)
                if old.sha256!=new.sha256:raise ValueError('Repeated inprocess receipt changed its payload')
            return False
        self.entries[identity]=result
        while len(self.entries)>self.capacity:self.entries.popitem(last=False)
        self.last_record=dict(sha256=None,storage='ephemeral_memory',backend_session_id=session,mutation_seq=sequence)
        return True

    def close(self):self.entries.clear();self.closed=True


class PrepaidMemoryBudget:
    def __init__(self,limits):
        self.limits=dict(limits);self.used={k:0 for k in limits};self.closed=False
    def reserve(self,phase,**values):
        if self.closed:raise RuntimeError('Prepaid memory budget closed')
        for key,n in values.items():
            if type(n)is not int or n<0 or key not in self.limits or self.used[key]+n>self.limits[key]:
                raise RuntimeError(f'Prepaid resource limit exceeded: {key}={n}')
        for key,n in values.items():self.used[key]+=n
    def settle_control(self,phase,count,result):
        if not result.get('physics_count_exact',True):return
        actual=result['executed_control_steps'];physics=result['executed_physics_steps']
        if not 0<=actual<=count or physics!=4*actual:
            raise ValueError('Cannot settle an inconsistent physical execution')
        self.used['control_steps']-=count-actual
        self.used['physics_steps']-=4*count-physics
    def state_dict(self):return dict(limits=dict(self.limits),used=dict(self.used),recovery='parent_prepaid_credit_retained_on_failure')
    def close(self):self.closed=True


class DeviceRolloutBuffer:
    def __init__(self,fragments,policy,device):
        self._transitions=[];self.device=torch.device(device)
        signatures=set()
        flags=[]
        for fragment in fragments:
            for row in fragment:
                if not row.transition_valid or row.chain is None:raise ValueError('Invalid physical row in GPU rollout')
                row.context={k:v.detach().to(device,copy=True) for k,v in row.context.items()}
                row.next_context=(None if row.next_context is None else {k:v.detach().to(device,copy=True) for k,v in row.next_context.items()})
                row.chain=row.chain.detach().to(device);row.old_log_prob=row.old_log_prob.detach().to(device)
                row.free_mask=row.free_mask.detach().to(device)
                for key in ('critic_context','next_critic_context'):
                    if row.metadata[key] is not None:
                        row.metadata[key]={k:v.detach().to(device,copy=True) for k,v in row.metadata[key].items()}
                        actor_context=row.context if key=='critic_context' else row.next_context
                        if actor_context is None or any(k not in row.metadata[key] or
                                v.shape!=row.metadata[key][k].shape for k,v in actor_context.items()):
                            raise ValueError('Critic and Actor snapshots have different causal conditions')
                        flags.extend((v==row.metadata[key][k]).all() for k,v in actor_context.items())
                trace=row.metadata['sampler_trace']
                for key in ['chain','old_means','old_stds','old_log_probs','free_mask','sampling_noise']:
                    trace[key]=trace[key].detach().to(device)
                if trace['kernel_config']!=policy.kernel_config:raise ValueError('Behavior and learner kernel identities differ')
                if row.chain.shape!=(policy.steps+1,policy.generation_frames,30) or row.chain.dtype!=torch.float32 or row.old_log_prob.shape!=(len(policy.stochastic_steps),) or row.old_log_prob.dtype!=torch.float64:
                    raise ValueError('GPU buffer requires the full FP32 20-step chain and FP64 joint old probability')
                if row.executed_physics_steps!=4*row.executed_control_steps or row.control_tick_end-row.control_tick_begin!=12*row.executed_control_steps:
                    raise ValueError('Physical work or timestamp count changed')
                if len(row.rewards)!=row.executed_control_steps or not row.terminated and row.next_context is None:
                    raise ValueError('Missing actual reward or causal bootstrap state')
                expected=row.context['future_valid'][0,:,None]&~row.context['known_qpos30_mask'][0]
                flags.extend([torch.isfinite(row.chain).all(),torch.isfinite(row.old_log_prob).all(),(row.free_mask==expected).all()])
                signatures.add(row.identity['policy_version']);self._transitions.append(row)
        if self._transitions and (len(signatures)!=1 or not bool(torch.stack(flags).all())):
            raise ValueError('Mixed policy version, nonfinite rollout or changed free mask')
        from .tracking_context import validate_tracking_context
        # 逐帧因果内容仍全覆盖；每种状态只做一次批量验证，避免每条链逐字段同步。
        for name in ('critic_context','next_critic_context'):
            contexts=[row.metadata[name] for row in self._transitions if row.metadata[name] is not None]
            if contexts:
                validate_tracking_context({key:torch.cat([context[key] for context in contexts]) for key in contexts[0]})
        for row in self._transitions:
            for name,actor_context in (('critic_context',row.context),('next_critic_context',row.next_context)):
                if actor_context is not None:
                    row.metadata[name].update(actor_context)
        self.fragments=list(map(len,fragments))
    @property
    def transitions(self):return tuple(self._transitions)
    def __len__(self):return len(self._transitions)
    def clear(self):self._transitions.clear()
