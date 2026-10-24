"""固定物理评估中的 RoboFFT 紧凑初始参考与代理漂移诊断。

初始评估保存实际最终 action_x0 的固定 MC/旧损失，并以独立种子保存生成参考探针。
后续评估复用同一 eps/t，报告 surrogate_kl 和同条件同噪声参考变化；不要求反向链，
不调用 Gaussian KL。真实跟踪误差、失败率、动作幅度仍由原物理评估器独立计算。
独立评估 RNG 不消耗训练 MC 或环境噪声；源码 SHA 不参与兼容门禁。
"""
from __future__ import annotations

from pathlib import Path
import torch

from . import ALGORITHM
from .records import MCRecord
from .surrogate import action_objective
from .updater import compare_outputs
from ..dppo.evaluation import _digest
from ..dppo.training_checkpoint import file_sha256, immutable_cpu_snapshot


def _loss(policy, trace, mc):
    device=next(policy.actor.parameters()).device
    context={k:v.to(device) for k,v in trace['conditions'].items()}
    parts=[]
    for k in range(len(mc['sampling_indices'])):
        parts.append(policy.losses(context,trace['action_x0'].to(device),mc['eps'][None,k:k+1],
            mc['sampling_indices'][None,k:k+1],mc['original_timesteps'][None,k:k+1],
            **({} if mc.get('time_weights') is None else dict(time_weights=mc['time_weights'][None]))))
    return torch.cat(parts,1)


@torch.no_grad()
def make_reference(policy, trace, *, seed, row=None):
    device=next(policy.actor.parameters()).device
    trace={k:v for k,v in trace.items() if k in ('conditions','action_x0','normalized','free_mask','kernel_config','algorithm')}
    action=trace['action_x0'].to(device)
    if action.shape[0]!=1 or trace.get('algorithm')!=ALGORITHM:
        raise ValueError('Fixed RoboFFT reference requires one actual compact action')
    draw=policy.draw_mc(action,torch.Generator().manual_seed(seed))
    mc={k:v[0] for k,v in draw.items()}
    if policy.robofft.get('execution_weighting') is not None:
        from .execution_weights import weights_for_rows, predicted_weights, normalize_frame_weights
        if row is not None:
            mc['time_weights'] = weights_for_rows([row], policy.robofft['execution_weighting'])[0][0]
        else:
            free=trace['free_mask'][0].cpu()
            mc['time_weights']=normalize_frame_weights(predicted_weights(free,
                settings=policy.robofft['execution_weighting']),free)
    old=_loss(policy,trace,mc)[0]
    record=MCRecord(mc['eps'],mc['sampling_indices'],mc['original_timesteps'],old,torch.arange(len(old)),
        time_weights=mc.get('time_weights'))
    probe=policy.sample_rollout({k:v.to(device) for k,v in trace['conditions'].items()},
        generator=torch.Generator(device=device).manual_seed(seed))
    snapshot={k:probe[k] for k in ('normalized','qpos','free_mask')}
    if mc.get('time_weights') is not None:snapshot['time_weights']=mc['time_weights'][None]
    return immutable_cpu_snapshot(dict(trace=trace,mc=record,probe_seed=seed,probe=snapshot))


def load_references(paths, *, max_chains=4):
    if isinstance(paths,(str,Path)):
        path=Path(paths);paths=sorted(path.glob('*.pt')) if path.is_dir() else [path]
    records=[]
    for path in list(paths)[:max_chains]:
        path=Path(path).resolve(strict=True)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['trace'].get('algorithm')!=ALGORITHM or 'chain' in payload['trace']:
            raise ValueError('Fixed RoboFFT reference must not contain Gaussian chains')
        payload['mc'].validate()
        records.append(dict(path=str(path),sha256=file_sha256(path),**payload))
    if not records:raise ValueError('No fixed compact Actor reference supplied')
    return dict(schema='genmo.robofft.fixed_reference.v1',records=records,
        reference_sha256=_digest([(r['path'],r['sha256']) for r in records]))


@torch.no_grad()
def fixed_action_drift(policy, reference, *, include_values=False):
    values=[];changes=[]
    device=next(policy.actor.parameters()).device
    for record in reference['records']:
        trace=record['trace'];mc=record['mc'];mc.validate()
        if trace['kernel_config']!=policy.kernel_config:
            raise ValueError('Fixed RoboFFT reference behavior kernel differs')
        current=_loss(policy,trace,{k:getattr(mc,k) for k in ('eps','sampling_indices','original_timesteps','time_weights')})
        metrics=action_objective(current,mc.old_loss[None].to(current),current.new_ones(1),
            **{k:policy.robofft[k] for k in ('beta_neg','logprob_normalizer','logratio_alpha')})
        values.extend(metrics['surrogate_kl'].cpu().tolist())
        now=policy.sample_rollout({k:v.to(device) for k,v in trace['conditions'].items()},
            generator=torch.Generator(device=device).manual_seed(record['probe_seed']))
        changes.append(compare_outputs({k:v.to(device) for k,v in record['probe'].items()},now))
    return dict(algorithm=ALGORITHM,surrogate_kl_values=values,reference_changes=changes,
        reference_sha256=reference['reference_sha256'],frozen_identity=policy.assert_frozen_identity(),
        diagnostic_scope='fixed_initial_action_surrogate_and_reference_change_not_physics')
