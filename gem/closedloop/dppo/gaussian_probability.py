"""Stage2稳定高斯统计、逐步坐标平均代理PPO及独立真实联合KL。

行为策略仍保存 normalized qpos30 的完整 FP64 联合概率、旧均值和实际标准差。
训练目标对15～18四个随机转移分别取自由坐标平均log-ratio、指数和符号相关裁剪，
同一上层GAE在四步共享，目标按完整链数乘4归一化；代理比值不是真实联合概率比。
真实joint log-ratio与joint KL仍对全部自由坐标/四步求和。前缀和padding不参与概率；
冻结前15步通过身份校验抵消，确定性末步不具有高斯密度，不参与联合概率。

比值直接使用旧核的标准化残差和均值变化计算，不先求两个大联合 log-prob 再
相减；方差变化采用 expm1，固定方差时恰好退化为 z*d-d*d/2。KL 使用同一实际
高斯核计算 old||new，保持完整联合 KL 与坐标平均 KL 两套命名和量纲。这里不
做概率截断、不修改行为数据、不处理优化器，也不改变独立概率验收门槛。
"""
from __future__ import annotations

import math
import json
import torch

from .execution_checks import require_tensor


def gaussian_statistics(value, old_mean, old_std, new_mean, new_std, free_mask, *, coordinate_details=False):
    """返回 FP64 [B] 的直接联合 log-ratio、联合 KL 与自由坐标数。"""
    if value.shape != old_mean.shape or value.shape != new_mean.shape or value.shape != free_mask.shape:
        raise ValueError('Gaussian values, means and mask must have equal [B,T,D] shapes')
    if value.ndim != 3 or free_mask.dtype != torch.bool:
        raise ValueError('Expected [B,T,D] values and boolean free mask')
    for scale in (old_std, new_std):
        if scale.shape not in ((len(value), 1, 1), value.shape):
            raise ValueError('Actual Gaussian std must be [B,1,1] or [B,T,D]')
        require_tensor(torch.isfinite(scale).all() & (scale > 0).all(), 'Invalid actual Gaussian standard deviation')
    x = torch.where(free_mask, value.detach(), 0.).double()
    old = torch.where(free_mask, old_mean.detach(), 0.).double()
    new = torch.where(free_mask, new_mean, 0.).double()
    sigma0, sigma = old_std.detach().double(), new_std.double()
    # log1p在新旧标准差很接近时避免先做除法再log的额外消减。
    r = torch.log1p((sigma-sigma0)/sigma0)
    a = torch.exp(-r)
    z, d = (x-old)/sigma0, (new-old)/sigma
    ratio_terms = -r - .5*torch.expm1(-2*r)*z.square() + a*z*d - .5*d.square()
    kl_terms = r + .5*torch.expm1(-2*r) + .5*d.square()
    joint_ratio = ratio_terms.masked_fill(~free_mask, 0.).sum((-2,-1))
    joint_kl = kl_terms.masked_fill(~free_mask, 0.).sum((-2,-1))
    counts = free_mask.sum((-2,-1))
    require_tensor(torch.isfinite(joint_ratio).all() & torch.isfinite(joint_kl).all()
        & (joint_kl >= -1e-10).all(), 'Nonfinite Gaussian ratio or negative KL', FloatingPointError)
    result=dict(joint_log_ratio=joint_ratio, joint_kl=joint_kl.clamp_min(0.), free_count=counts,
        mean_coordinate_log_ratio=joint_ratio/counts.clamp_min(1),
        mean_coordinate_kl=joint_kl.clamp_min(0.)/counts.clamp_min(1))
    if coordinate_details:
        # 诊断复用同一完整异方差公式；普通训练不分配这些额外输出。
        result['coordinates']={key:torch.where(free_mask,tensor,0.) for key,tensor in dict(
            kl=kl_terms,log_ratio=ratio_terms,mean_delta=new-old,
            standardized_mean_delta=(new-old)/sigma0,residual=z,
            actual_std=sigma0.expand_as(x),std_log_change=r.expand_as(x)).items()}
        result['coordinates']['mask']=free_mask
    return result


def ppo_surrogate(statistics, advantages, *, clip, reduction='joint_sum'):
    """执行真正的符号相关 PPO clipping；零自由度转移不贡献目标。"""
    if reduction not in ('joint_sum','free_coordinate_mean') or not 0 < clip < 1:
        raise ValueError('Unknown proxy reduction or invalid PPO clip')
    log_ratio = statistics['joint_log_ratio' if reduction == 'joint_sum' else 'mean_coordinate_log_ratio']
    advantage = advantages.detach().to(log_ratio)
    valid = statistics['free_count'] > 0
    bad=~torch.isfinite(log_ratio) | ~torch.isfinite(advantage)
    overflow=log_ratio>math.log(torch.finfo(torch.float64).max)
    underflow=log_ratio<math.log(float.fromhex('0x0.0000000000001p-1022'))
    if bool((bad|overflow|underflow).any()):
        raise FloatingPointError('PPO candidate must be rolled back: '+json.dumps(dict(
            nonfinite_fraction=float(bad.double().mean()),ratio_overflow_fraction=float(overflow.double().mean()),
            ratio_underflow_fraction=float(underflow.double().mean()),
            action='reject_candidate_without_ratio_clamp_or_sample_filter')))
    lower, upper = math.log1p(-clip), math.log1p(clip)
    active_clip = ((advantage > 0)&(log_ratio > upper)) | ((advantage < 0)&(log_ratio < lower))
    # 先在log空间选择数学上正确的分支；被裁剪分支不对极端log-ratio取指数。
    objective_log = torch.where(active_clip, torch.where(advantage > 0, upper, lower), log_ratio)
    objective = objective_log.exp()*advantage
    ratio = log_ratio.exp()
    # 任何溢出/下溢均显式拒绝，而非改写ratio、删除样本或静默产生零梯度。
    require_tensor(torch.isfinite(log_ratio).all() & torch.isfinite(advantage).all()
        & torch.isfinite(ratio).all() & (ratio > 0).all() & torch.isfinite(objective).all(),
        'PPO joint ratio/objective nonfinite or underflow; candidate must be rolled back', FloatingPointError)
    return dict(objective=torch.where(valid,objective,0.), ratio=ratio, log_ratio=log_ratio,
        clipped=((log_ratio < lower)|(log_ratio > upper))&valid, objective_clipped=active_clip&valid,
        objective_log_gradient=torch.where(valid & ~active_clip, objective, 0.).detach(),
        reduction=reduction, ratio_is_joint_probability=reduction=='joint_sum')


def chain_statistics(joint_log_ratios, joint_kls, free_counts, *, steps=(15,16,17,18)):
    """四个旧路径条件高斯组成一个扩散链；不除坐标数、不做去噪折扣。"""
    if tuple(steps)!=(15,16,17,18) or joint_log_ratios.ndim!=2 or joint_log_ratios.shape[1]!=4:
        raise ValueError('Exactly four stochastic transitions 15..18 required per complete chain')
    if joint_kls.shape!=joint_log_ratios.shape or free_counts.shape!=joint_log_ratios.shape:
        raise ValueError('Chain log ratios, KL and free counts must have identical [chains,4] shape')
    counts=free_counts.sum(1)
    logratio=joint_log_ratios.double().sum(1)
    kl=joint_kls.double().sum(1)
    return dict(joint_log_ratio=logratio,joint_kl=kl,free_count=counts,
        mean_coordinate_log_ratio=logratio/counts.clamp_min(1),mean_coordinate_kl=kl/counts.clamp_min(1))


def distribution(values):
    """小型诊断向量转为JSON有限标量；拒绝非有限值，不滤掉异常样本。"""
    values=values.detach().double().flatten()
    if not len(values) or not torch.isfinite(values).all():
        raise FloatingPointError('Cannot summarize empty or nonfinite probability population')
    quantiles=torch.quantile(values,values.new_tensor([0.,.01,.05,.5,.95,.99,1.])).cpu().tolist()
    return dict(zip(('min','p01','p05','median','p95','p99','max'),quantiles),mean=float(values.mean()))


def chain_diagnostics(statistics, advantages, *, clip):
    """历史联合目标的只读数学诊断；当前训练/候选验收不得调用本函数。"""
    surrogate=ppo_surrogate(statistics,advantages,clip=clip,reduction='joint_sum')
    valid=statistics['free_count']>0
    logratio=surrogate['log_ratio'][valid].detach()
    return dict(chain_joint_log_ratio=distribution(logratio),
        objective_mean=float(surrogate['objective'][valid].mean()),
        behavior_objective_mean=float(advantages.detach().to(logratio)[valid].mean()),
        objective_gain_mean=float((surrogate['objective'][valid]-advantages.detach().to(logratio)[valid]).mean()),
        joint_score_gain_mean=float((advantages.detach().to(logratio)[valid]*logratio).mean()),
        chain_joint_ratio=distribution(surrogate['ratio'][valid]),chain_joint_kl=distribution(statistics['joint_kl'][valid]),
        clip_fraction=float(surrogate['clipped'][valid].double().mean()),
        active_clip_fraction=float(surrogate['objective_clipped'][valid].double().mean()),
        nonfinite_fraction=0.,ratio_underflow_fraction=0.,
        ratio_is_joint_probability=True,complete_chain_count=int(valid.sum()),
        denominator='one_complete_chain_one_upper_GAE_advantage')


def step_statistics(joint_log_ratios, joint_kls, free_counts):
    """保留[N,K]内部转移轴，K为显式4或10；不能在PPO裁剪之前合并步骤。"""
    if joint_log_ratios.ndim!=2 or joint_log_ratios.shape[1] not in (4,10):
        raise ValueError('Stepwise PPO requires four or ten stochastic transitions ending at step 18')
    if joint_kls.shape!=joint_log_ratios.shape or free_counts.shape!=joint_log_ratios.shape:
        raise ValueError('Step log ratios, KL and counts must have equal [N,K] shapes')
    require_tensor(torch.isfinite(joint_log_ratios).all() & torch.isfinite(joint_kls).all()
        & torch.isfinite(free_counts).all() & (joint_kls>=0).all() & (free_counts>=0).all()
        & (free_counts==free_counts.round()).all(), 'Invalid stepwise Gaussian statistics',FloatingPointError)
    # D=0是数学上的零目标项，不允许非零概率贡献通过clamp_min被隐藏。
    require_tensor(((free_counts>0)|((joint_log_ratios==0)&(joint_kls==0))).all(),
        'Empty Gaussian transition has a nonzero probability contribution',FloatingPointError)
    return dict(joint_log_ratio=joint_log_ratios.double(),joint_kl=joint_kls.double(),free_count=free_counts,
        mean_coordinate_log_ratio=joint_log_ratios.double()/free_counts.clamp_min(1),
        mean_coordinate_kl=joint_kls.double()/free_counts.clamp_min(1))


def stepwise_surrogate(statistics, advantages, *, clip):
    """返回[N,K]目标及对joint_log_ratio的导数，尚未除K*N_global。

    普通未裁剪项的导数为A*exp(joint_log_ratio/D)/D；真正符号相关平坦区和D=0
    项导数为零。这里的proxy保留完整自由参考坐标，但明确不是真实联合概率比。
    """
    log=statistics['joint_log_ratio']
    if log.ndim!=2 or log.shape[1] not in (4,10) or advantages.shape!=(len(log),):
        raise ValueError('Stepwise PPO needs four or ten transitions and one advantage per complete chain')
    result=ppo_surrogate(statistics,advantages[:,None],clip=clip,reduction='free_coordinate_mean')
    result['joint_log_gradient']=result['objective_log_gradient']/statistics['free_count'].clamp_min(1)
    result['chain_objective']=result['objective'].mean(1)
    return result


def stepwise_diagnostics(statistics, advantages, *, clip):
    """统一训练、候选验收和整批审计；目标分母为K*N，零自由度项仍占槽。

    ratio/clip分布的样本单位是D>0的单个内部转移，整链KL单位是完整行为链。
    coordinate_kl是每个内部转移的自由坐标平均KL分布，不是单个标量坐标KL分布。
    """
    out=stepwise_surrogate(statistics,advantages,clip=clip)
    valid=statistics['free_count']>0
    if not len(advantages) or not bool(valid.any()):
        raise ValueError('Stepwise diagnostics need at least one free valid behavior chain')
    joint=statistics['joint_kl'];coordinate=statistics['mean_coordinate_kl']
    ratio=distribution(out['ratio'][valid]);log=distribution(out['log_ratio'][valid])
    baseline=advantages.detach().to(joint)[:,None]*valid
    objective=float(out['objective'].mean());gain=float((out['objective']-baseline).mean())
    clipped=float(out['clipped'][valid].double().mean())
    active=float(out['objective_clipped'][valid].double().mean())
    per_step=[];step_count=joint.shape[1];first_step=19-step_count
    for k in range(step_count):
        mask=valid[:,k]
        per_step.append(dict(step=k+first_step,statistical_unit='single_stochastic_transition',
            valid_transitions=int(mask.sum()),joint_kl=distribution(joint[:,k]),
            free_coordinates=distribution(statistics['free_count'][:,k]),
            mean_coordinate_kl=distribution(coordinate[:,k]),joint_log_ratio=distribution(statistics['joint_log_ratio'][:,k]),
            mean_log_ratio=distribution(out['log_ratio'][mask,k]) if mask.any() else None,
            proxy_ratio=distribution(out['ratio'][mask,k]) if mask.any() else None,
            proxy_clip_fraction=float(out['clipped'][mask,k].double().mean()) if mask.any() else 0.,
            proxy_active_clip_fraction=float(out['objective_clipped'][mask,k].double().mean()) if mask.any() else 0.,
            stepwise_objective_mean=float(out['objective'][:,k].mean()),
            stepwise_objective_gain=float((out['objective']-baseline)[:,k].mean())))
    return dict(objective='stepwise_coordinate_mean_clipped_ppo',ratio_is_joint_probability=False,
        stepwise_objective_mean=objective,stepwise_objective_gain=gain,
        objective_mean=objective,objective_gain_mean=gain,behavior_objective_mean=float(baseline.mean()),
        mean_proxy_ratio=ratio['mean'],proxy_ratio=ratio,proxy_ratio_p95=ratio['p95'],proxy_ratio_p99=ratio['p99'],
        mean_log_ratio=log,proxy_clip_fraction=clipped,proxy_active_clip_fraction=active,
        clip_fraction=clipped,active_clip_fraction=active,
        chain_joint_kl=distribution(joint.sum(1)),chain_joint_log_ratio=distribution(statistics['joint_log_ratio'].sum(1)),
        mean_coordinate_kl=float(coordinate.mean()),maximum_step_mean_coordinate_kl=float(coordinate.mean(0).max()),
        coordinate_kl=distribution(coordinate),coordinate_KL_p95=distribution(coordinate)['p95'],
        mean_joint_kl=float(joint.mean()),joint_kl=distribution(joint),per_denoising_step=per_step,
        complete_chain_count=len(advantages),valid_internal_transitions=int(valid.sum()),
        free_coordinates_per_transition=distribution(statistics['free_count'][valid]),
        nonfinite_fraction=0.,ratio_underflow_fraction=0.,
        denominator=f'{step_count}_times_global_valid_behavior_chain_count',
        proxy_statistical_unit=f'one_valid_stochastic_transition_{first_step}_to_18',
        chain_KL_statistical_unit=f'one_complete_behavior_chain_sum_of_{step_count}_joint_KLs',
        coordinate_KL_p95_scope='per_transition_free_coordinate_mean_KL_not_scalar_coordinate_KL')
