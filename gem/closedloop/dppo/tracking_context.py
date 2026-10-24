"""非对称Critic的因果历史合同与Actor条件隔离。

Actor仍只接收Stage1十项条件。Critic额外接收后端逐50Hz记录的90维历史、有效mask、
实际采样时间和对应已承诺参考时间；所有时间必须逐帧一致且不晚于决策时刻。
历史的前48维必须与Actor实际历史一致，未知历史保留零及无效mask。这里不读取新动作、
去噪链、奖励或决策之后的实际结果，也不把当前快照复制成50帧历史。
该合同同时用于当前状态、bootstrap下一状态、完整MC银行及持久化恢复验证。
"""
from __future__ import annotations
import torch
from gem.closedloop.contracts import STAGE1_CONDITION_KEYS, validate_stage1_condition_batch

SCHEMA='genmo.critic.causal_tracking90.v2'
FUTURE_SCHEMA='genmo.critic.causal_tracking90_committed38.v3'
TRACKING_KEYS=('tracking_history','tracking_history_valid','tracking_history_times','tracking_reference_times')
COMMITTED_KEYS=('committed_reference','committed_valid','committed_times','committed_metadata',
    'committed_metadata_valid','committed_snapshot_tick','committed_policy_version')


def validate_tracking_context(context):
    legacy=set(STAGE1_CONDITION_KEYS)|set(TRACKING_KEYS)
    if set(context) not in (legacy,legacy|set(COMMITTED_KEYS)):
        raise ValueError('Critic requires exactly Actor conditions plus causal tracking history')
    actor={key:context[key] for key in STAGE1_CONDITION_KEYS}
    validate_stage1_condition_batch(actor,history_steps=50)
    values,valid,times,reference=(context[key] for key in TRACKING_KEYS)
    batch=len(context['decision_time'])
    if values.shape!=(batch,50,90) or valid.shape!=(batch,50) or valid.dtype!=torch.bool:
        raise ValueError('Tracking history must be [B,50,90] with a boolean mask')
    if times.shape!=valid.shape or reference.shape!=valid.shape or times.dtype!=torch.float64 or reference.dtype!=torch.float64:
        raise ValueError('Tracking timestamps must be FP64 [B,50]')
    checks=(torch.isfinite(values).all(),torch.isfinite(times).all(),torch.isfinite(reference).all(),
        (valid==context['proprio_history_valid']).all(),
        (times==context['proprio_history_times']).all(),(times==reference).all(),
        (times<=context['decision_time'][:,None]+1e-10).all(),
        torch.isclose(times[:,1:]-times[:,:-1],times.new_tensor(.02),atol=1e-10,rtol=0).all(),
        torch.isclose(times[:,-1],context['decision_time'],atol=1e-10,rtol=0).all(),
        torch.where(valid[...,None],values[...,:48]==context['proprio_history'],True).all(),
        torch.where(~valid[...,None],values==0.,True).all())
    passed=torch.stack(checks)
    if not bool(passed.all()):
        names=('finite_history','finite_times','finite_reference','actor_mask','actor_timestamps',
            'reference_timestamps','no_future','50Hz_grid','decision_endpoint','actor_proprio48','invalid_padding_zero')
        failed=[name for name,ok in zip(names,passed.cpu().tolist()) if not ok]
        raise ValueError('Noncausal or mismatched tracking history: '+','.join(failed)+
            f'; timestamp_delta={float((times-context["proprio_history_times"]).abs().max())}; '+
            f'proprio_delta={float(torch.where(valid[...,None],values[...,:48]-context["proprio_history"],0.).abs().max())}')
    if 'committed_reference' in context:
        f=context['committed_reference'];v=context['committed_valid'];t=context['committed_times']
        m=context['committed_metadata'];mv=context['committed_metadata_valid']
        tick=context['committed_snapshot_tick'];version=context['committed_policy_version']
        if (f.shape!=(batch,41,38) or v.shape!=(batch,41) or v.dtype!=torch.bool
                or t.shape!=v.shape or t.dtype!=torch.float64 or m.shape!=(batch,10)
                or mv.shape!=m.shape or mv.dtype!=torch.bool or tick.shape!=(batch,)
                or version.shape!=(batch,) or tick.dtype!=torch.int64 or version.dtype!=torch.int64):
            raise ValueError('Invalid future committed reference schema')
        expected=context['decision_time'][:,None]+torch.arange(10,51,device=t.device,dtype=torch.float64)[None]/50
        if not (torch.isfinite(f).all() and torch.isfinite(m).all()
                and torch.allclose(t,expected,atol=1e-10,rtol=0)
                and torch.allclose(tick.double()/600,context['decision_time'],atol=1e-10,rtol=0)
                and (version>=0).all() and torch.where(~v[...,None],f==0,True).all()
                and torch.where(~mv,m==0,True).all() and not mv[:,6:8].any()
                and (f[...,36:][v]>=0).all()):
            raise ValueError('Noncausal committed cache, unknown arrival exposed, or invalid padding')
    return context


def build_critic_context(actor_context,snapshot,*,policy_version=0):
    return build_critic_context_batch(actor_context,[snapshot],policy_version=policy_version)


def build_critic_context_batch(actor_context,snapshots,*,policy_version=0):
    """同一决策批次合并H2D和因果核验；按原整数tick在CPU做FP64秒转换。"""
    if len(snapshots)!=len(actor_context['decision_time']) or any(
            s.get('tracking_history_schema')!='gmt.causal_tracking_history90.v2' for s in snapshots):
        raise ValueError('Backend has no audited causal tracking history')
    device=actor_context['decision_time'].device
    def tensor(key,dtype,scale=1.):
        value=torch.stack([torch.as_tensor(s[key],dtype=dtype) for s in snapshots])
        if scale!=1.:value=value/scale
        return value.to(device).clone()
    result=dict(actor_context,tracking_history=tensor('tracking_history_values',torch.float32),
        tracking_history_valid=tensor('tracking_history_valid',torch.bool),
        tracking_history_times=tensor('tracking_history_ticks',torch.float64,600.),
        tracking_reference_times=tensor('tracking_reference_ticks',torch.float64,600.))
    present=['committed_reference' in s for s in snapshots]
    if any(present):
        if not all(present):raise ValueError('Mixed Critic state schemas in one batch')
        refs=[s['committed_reference'] for s in snapshots]
        if any(r['schema']!='gmt.causal_committed_reference38.v1' for r in refs):
            raise ValueError('Unknown backend committed cache schema')
        for target,key,dtype in (('committed_reference','values',torch.float32),('committed_valid','valid',torch.bool),
                ('committed_times','ticks',torch.float64),('committed_metadata','metadata',torch.float32),
                ('committed_metadata_valid','metadata_valid',torch.bool),('committed_snapshot_tick','snapshot_tick',torch.int64)):
            result[target]=torch.stack([torch.as_tensor(r[key],dtype=dtype) for r in refs]).to(device)
        result['committed_times']=result['committed_times']/600.
        versions=torch.as_tensor(policy_version,dtype=torch.int64,device=device)
        result['committed_policy_version']=versions.expand(len(refs)).clone()
    return validate_tracking_context(result)
