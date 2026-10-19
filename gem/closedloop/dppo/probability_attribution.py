"""完整扩散链的只读FP64坐标归因、采样核核验与重要性权重诊断。

输入必须来自不可变真实行为链及同一参数版本的当前高斯均值/实际标准差。坐标公式
复用生产gaussian_statistics，包含新旧方差差异；自由mask同时排除padding与已知前缀。
按四个随机步、生成帧、实际关节顺序、前缀、音乐来源和采样时间输出可加和统计。
root前三维仅称模型运动特征，不解释为世界XYZ。带符号log-ratio同时保留正负和绝对
贡献，不把相消后的带符号比例当作归因占比。所有分位数明确对应有效坐标或完整链。

ESS只在显式诊断中用logsumexp计算，不参与训练门限，不改PPO ratio、不裁权重，
不删除尾部链。采样尾部只报告样本量与正态理论尾率；不能因有限样本波动认定错误。
本模块不创建环境、不修改模型或行为数据；新候选的生成变化不构成物理收益证据。
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .gaussian_probability import gaussian_statistics

STEPS=(15,16,17,18)


def quantiles(value, *, absolute=False):
    value=torch.as_tensor(value,dtype=torch.float64).reshape(-1)
    if not value.numel():return None
    if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite attribution population')
    if absolute:value=value.abs()
    # 全部真实自由坐标可超过torch.quantile的2^24元素上限；NumPy使用完整总体。
    q=np.quantile(value.detach().cpu().numpy(),[0.,.5,.95,.99,1.]).tolist()
    return dict(zip(('min','p50','p95','p99','max'),q))


def feature_groups(joint_order):
    if len(joint_order)!=21 or len(set(joint_order))!=21:raise ValueError('Explicit unique 21-joint kinematics order required')
    legs=[9+i for i,name in enumerate(joint_order) if any(token in name for token in ('leg_','knee_','ankle_'))]
    if len(legs)!=12:raise ValueError('Expected exactly twelve named leg joints')
    return dict(root_motion_features=list(range(3)),rotation6d=list(range(3,9)),
        leg_joints=legs,other_joints=[i for i in range(9,30) if i not in legs])


def coordinate_statistics(value,old_mean,old_std,new_mean,new_std,mask):
    result=gaussian_statistics(value,old_mean,old_std,new_mean,new_std,mask,coordinate_details=True)
    detail=result['coordinates']
    # 不修改生产值；坐标加和只容忍FP64运算顺序与生产非负修正的舍入差。
    for key,field in (('kl','joint_kl'),('log_ratio','joint_log_ratio')):
        torch.testing.assert_close(detail[key].sum((-2,-1)),result[field],rtol=1e-10,atol=1e-10)
    return detail


def contribution_summary(detail,mask,total_kl):
    selected={key:value[mask].double() for key,value in detail.items() if key!='mask'}
    n=int(mask.sum())
    if not n:return dict(free_coordinates=0,total_kl=0.,kl_fraction=None,mean_coordinate_kl=None)
    kl=selected['kl'];ratio=selected['log_ratio'];delta=selected['mean_delta']
    positive=float(ratio.clamp_min(0).sum());negative=float(ratio.clamp_max(0).sum())
    absolute=float(ratio.abs().sum());signed=float(ratio.sum())
    return dict(free_coordinates=n,total_kl=float(kl.sum()),
        kl_fraction=float(kl.sum())/total_kl if total_kl>0 else None,
        mean_coordinate_kl=float(kl.mean()),kl_coordinate_quantiles=quantiles(kl),
        mean_delta_rms=float(delta.square().mean().sqrt()),mean_delta_abs_quantiles=quantiles(delta,absolute=True),
        standardized_mean_delta_rms=float(selected['standardized_mean_delta'].square().mean().sqrt()),
        standardized_mean_delta_abs_quantiles=quantiles(selected['standardized_mean_delta'],absolute=True),
        log_ratio_signed_sum=signed,log_ratio_positive_sum=positive,log_ratio_negative_sum=negative,
        log_ratio_absolute_sum=absolute,log_ratio_cancellation_fraction=1-abs(signed)/absolute if absolute else 0.,
        residual_mean=float(selected['residual'].mean()),residual_std=float(selected['residual'].std(unbiased=False)),
        residual_tail_fractions={str(a):float((selected['residual'].abs()>a).double().mean()) for a in (3,4,5,6)},
        residual_quantiles=quantiles(selected['residual']),actual_std_quantiles=quantiles(selected['actual_std']),
        std_log_change_abs_quantiles=quantiles(selected['std_log_change'],absolute=True),
        sampling_noise_reconstruction_abs=quantiles(selected['sampling_noise_error']) if 'sampling_noise_error' in selected else None)


def coordinate_report(detail,metadata,joint_order):
    """detail为CPU [N,4,T,30]；各分组独立划分，禁止跨分组重复相加。"""
    mask=detail['mask'];n,k,t,d=mask.shape
    if k!=4 or d!=30 or len(metadata)!=n:raise ValueError('Complete four-step chain attribution shape mismatch')
    total=float(detail['kl'][mask].sum());groups=feature_groups(joint_order)
    report=dict(total=contribution_summary(detail,mask,total),by_step={},by_frame={},by_feature={},by_joint={},
        by_prefix={},by_source={},by_music_time_bin={},root_feature_semantics='model_motion_features_not_world_XYZ')
    def selected(selector):return contribution_summary({key:value[selector] for key,value in detail.items()},mask[selector],total)
    for i,step in enumerate(STEPS):report['by_step'][str(step)]=selected((slice(None),i))
    for frame in range(t):report['by_frame'][str(frame)]=selected((slice(None),slice(None),frame))
    for name,indices in groups.items():report['by_feature'][name]=selected((slice(None),slice(None),slice(None),indices))
    for i,name in enumerate(joint_order):report['by_joint'][name]=selected((slice(None),slice(None),slice(None),9+i))
    for field,key in (('prefix_frames','by_prefix'),('source','by_source'),('music_time_bin','by_music_time_bin')):
        for value in sorted({str(row[field]) for row in metadata}):
            indices=torch.tensor([i for i,row in enumerate(metadata) if str(row[field])==value])
            report[key][value]=selected(indices)
    valid_kl=detail['kl'][mask].clamp_min(0)
    def top_share(values,fraction):
        count=max(1,math.ceil(values.numel()*fraction));den=float(values.sum())
        return float(torch.topk(values.reshape(-1),count).values.sum())/den if den>0 else None
    chain_kl=detail['kl'].sum((1,2,3))
    report['concentration']=dict(top_1pct_coordinate_kl_share=top_share(valid_kl,.01),
        top_5pct_coordinate_kl_share=top_share(valid_kl,.05),top_1pct_chain_kl_share=top_share(chain_kl,.01),
        largest_step_kl_share=max(v['kl_fraction'] or 0 for v in report['by_step'].values()),
        largest_feature_kl_share=max(v['kl_fraction'] or 0 for v in report['by_feature'].values()),
        rule='measured_shares_no_automatic_bug_or_physical_quality_claim')
    concentration=report['concentration'];denominator=report['total']['free_coordinates']
    enrichments={name:(item['kl_fraction'] or 0)/(item['free_coordinates']/denominator)
        for name,item in report['by_feature'].items() if item['free_coordinates']}
    report['attribution_flags']=dict(
        diffuse_small_coordinate_accumulation=bool(total>0 and
            (concentration['top_1pct_coordinate_kl_share'] or 0)<.15 and
            (concentration['top_1pct_chain_kl_share'] or 0)<.15 and
            report['total']['standardized_mean_delta_rms']<.1),
        concentrated_coordinates=bool((concentration['top_5pct_coordinate_kl_share'] or 0)>.5),
        concentrated_step=bool(concentration['largest_step_kl_share']>.6),
        concentrated_feature=bool(max(enrichments.values(),default=0)>3),
        concentrated_tail_chains=bool((concentration['top_1pct_chain_kl_share'] or 0)>.25),
        feature_kl_density_enrichment=enrichments,
        definitions='descriptive_explicit_thresholds_not_mutually_exclusive_not_training_gates')
    return report


def importance_report(log_ratio,joint_kl,advantages,metadata,*,clip=.1,top=12):
    log_ratio=torch.as_tensor(log_ratio,dtype=torch.float64).reshape(-1)
    joint_kl=torch.as_tensor(joint_kl,dtype=torch.float64).reshape(-1)
    advantages=torch.as_tensor(advantages,dtype=torch.float64).reshape(-1)
    if not len(log_ratio) or len(metadata)!=len(log_ratio) or joint_kl.shape!=log_ratio.shape or advantages.shape!=log_ratio.shape:
        raise ValueError('Importance diagnostics require one ratio/KL/advantage/identity per complete valid chain')
    if not torch.isfinite(torch.cat((log_ratio,joint_kl,advantages))).all():raise FloatingPointError('Nonfinite chain diagnostics')
    log_total=torch.logsumexp(log_ratio,0)
    ess=(2*log_total-torch.logsumexp(2*log_ratio,0)).exp()
    weights=(log_ratio-log_total).exp();order=torch.argsort(log_ratio,descending=True)
    low,high=math.log1p(-clip),math.log1p(clip)
    ordinary=(log_ratio<low)|(log_ratio>high)
    active=((advantages>0)&(log_ratio>high))|((advantages<0)&(log_ratio<low))
    limits=(math.log(float.fromhex('0x0.0000000000001p-1022')),math.log(torch.finfo(torch.float64).max))
    representable=(log_ratio>=limits[0])&(log_ratio<=limits[1])
    def entry(i):
        return {**metadata[i],'joint_log_ratio':float(log_ratio[i]),'joint_kl':float(joint_kl[i]),
            'normalized_weight':float(weights[i]),'advantage':float(advantages[i]),
            'ordinary_clip':bool(ordinary[i]),'active_clip':bool(active[i])}
    tails=[entry(i) for i in order[:top].tolist()]
    return dict(joint_log_ratio=quantiles(log_ratio),ratio=quantiles(log_ratio.exp()) if representable.all() else None,
        ratio_unrepresentable_count=int((~representable).sum()),ordinary_clip_fraction=float(ordinary.double().mean()),
        active_clip_fraction=float(active.double().mean()),joint_kl=quantiles(joint_kl),joint_kl_mean=float(joint_kl.mean()),
        effective_chains=float(ess),ess_fraction=float(ess/len(log_ratio)),
        top_weight_shares={str(fraction):float(weights[order[:max(1,math.ceil(len(order)*fraction))]].sum())
            for fraction in (.001,.01,.05)},highest_weight_chains=tails,
        highest_kl_chains=[entry(i) for i in torch.argsort(joint_kl,descending=True)[:top].tolist()],
        scope='offline_diagnostic_only_no_ESS_training_gate_no_weight_clamp')


def sampling_report(value,old_mean,old_std,mask,sampling_noise,old_log_prob):
    """独立Normal公式核验旧密度；真实FP32采样的减法舍入误差另行报告。"""
    x,mu,sigma=value.double(),old_mean.double(),old_std.double()
    z=(x-mu)/sigma;valid=z[mask]
    reference=torch.distributions.Normal(mu,sigma).log_prob(x).masked_fill(~mask,0.).sum((-2,-1))
    error=(reference-old_log_prob.double()).abs()
    residual_error=(z-sampling_noise.double())[mask].abs()
    # x_next是FP32 mean+std*noise后约束，按实际两个运算的舍入尺度说明重建误差。
    rounding_scale=(torch.finfo(value.dtype).eps*(mu.abs()+2*sigma.abs()*sampling_noise.double().abs())/sigma).expand_as(x)[mask]
    return dict(free_coordinates=int(mask.sum()),residual_mean=float(valid.mean()),
        residual_std=float(valid.std(unbiased=False)),residual_quantiles=quantiles(valid),
        residual_tail_fractions={str(a):float((valid.abs()>a).double().mean()) for a in (3,4,5,6)},
        gaussian_two_sided_tail_probabilities={str(a):math.erfc(a/math.sqrt(2)) for a in (3,4,5,6)},
        stored_noise_reconstruction_abs=quantiles(residual_error),fp32_rounding_scale=quantiles(rounding_scale),
        old_log_probability_abs_error=quantiles(error),
        statistical_scope='finite_sample_tail_counts_are_not_automatic_sampling_kernel_failures')
