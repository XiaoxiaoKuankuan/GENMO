"""逐去噪步坐标平均代理PPO的独立数学、归一化与配置验收。

输入是可解析FP64高斯分布及旧采样值，独立Normal公式验证不同方差、前缀/padding、
变长自由坐标、正负优势、逐步符号裁剪和两遍系数；全局链分母不依局部样本数改变。
同时验证诊断/来源审计使用同一目标、坐标KL与真实链KL分离，以及新正式配置不继承
旧联合门限。测试不启动物理或长训，不把合成概率通过称作真实机器人收益。
"""
import math
from pathlib import Path

import pytest
import torch

from gem.closedloop.dppo.gaussian_probability import (
    gaussian_statistics,step_statistics,stepwise_surrogate,stepwise_diagnostics)
from gem.closedloop.dppo.objective_audit import grouped_objective
from gem.closedloop.dppo.actor_backtracking import check_candidate_kl
from gem.closedloop.dppo.actor_optimization import require_current_actor_objective
from gem.closedloop.dppo.training_config import configuration


def statistics(log,counts=None):
    counts=torch.ones_like(log) if counts is None else counts
    return step_statistics(log,log.new_zeros(log.shape),counts)


@pytest.mark.parametrize('advantage',[1.,-1.,0.])
@pytest.mark.parametrize('ratio',[.5,.995,1.,1.005,1.5])
def test_sign_clip_joint_derivative_and_global_normalization(advantage,ratio):
    count=torch.tensor([[3.,17.,2.,1000.]],dtype=torch.float64)
    log=(count*math.log(ratio)).requires_grad_()
    adv=torch.tensor([advantage],dtype=torch.float64)
    out=stepwise_surrogate(statistics(log,count),adv,clip=.01)
    reference=torch.minimum((log/count).exp()*adv[:,None],
        (log/count).exp().clamp(.99,1.01)*adv[:,None])
    expected=torch.autograd.grad(-reference.sum()/(4*7),log,retain_graph=True)[0]
    actual=torch.autograd.grad(-out['objective'].sum()/(4*7),log)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(-out['joint_log_gradient']/(4*7),expected,rtol=1e-13,atol=1e-13)
    assert not out['ratio_is_joint_probability']


def test_steps_clip_independently_even_when_joint_product_would_clip():
    log=torch.full((1,4),math.log(1.004),dtype=torch.float64,requires_grad=True)
    out=stepwise_surrogate(statistics(log),torch.ones(1),clip=.01)
    assert not out['objective_clipped'].any()
    assert math.exp(float(log.sum().detach()))>1.01
    out['objective'].mean().backward()
    assert bool((log.grad>0).all())


def test_never_exponentiates_chain_ratio_and_rejects_actual_proxy_overflow():
    # 大joint log-ratio本身不要求求其指数；生产目标只对joint/D求指数。
    log=torch.full((1,4),1000.,dtype=torch.float64)
    out=stepwise_surrogate(statistics(log,torch.full_like(log,1000.)),torch.ones(1),clip=.01)
    assert torch.isfinite(out['ratio']).all() and not out['ratio_is_joint_probability']
    for bad in (float('nan'),float('inf'),1000.,-1000.):
        log=torch.full((1,4),bad,dtype=torch.float64)
        with pytest.raises(FloatingPointError):stepwise_surrogate(statistics(log),torch.ones(1),clip=.01)


def test_empty_step_keeps_four_slots_and_protected_coordinates_have_no_contribution():
    log=torch.zeros(2,4,dtype=torch.float64,requires_grad=True)
    counts=torch.tensor([[0.,4.,9.,1.],[3.,0.,0.,2.]],dtype=torch.float64)
    advantage=torch.tensor([1.,-2.],dtype=torch.float64)
    out=stepwise_surrogate(statistics(log,counts),advantage,clip=.01)
    assert float(out['objective'].mean().detach())==pytest.approx((3-4)/8)
    (-out['objective'].mean()).backward()
    expected=-advantage[:,None]/(8*counts.clamp_min(1))*(counts>0)
    torch.testing.assert_close(log.grad,expected)
    assert not log.grad[counts==0].count_nonzero()


@pytest.mark.parametrize('offset',[0.,.03,-.1,.6])
@pytest.mark.parametrize('microbatch',[1,3,11])
def test_two_pass_coefficients_match_independent_heteroscedastic_autograd(offset,microbatch):
    rng=torch.Generator().manual_seed(892)
    shape=(5,4,7,3);old=torch.randn(shape,generator=rng,dtype=torch.float64)
    std=torch.rand(5,4,1,1,generator=rng,dtype=torch.float64)+.2
    noise=torch.randn(shape,generator=rng,dtype=torch.float64)
    value=old+std*noise;feature=torch.randn(shape,generator=rng,dtype=torch.float64)
    mask=torch.rand(shape,generator=rng)>.4;mask[:,:,:2]=False;mask[:,:,-1]=False
    mask[0,1]=False
    adv=torch.tensor([1.,-1.,.4,-.7,.2],dtype=torch.float64)
    parameters=torch.tensor([offset,offset/3],dtype=torch.float64,requires_grad=True)
    new=old+parameters[0]*feature;new_std=std*parameters[1].exp()
    independent=(torch.distributions.Normal(new,new_std).log_prob(value)-
        torch.distributions.Normal(old,std).log_prob(value)).masked_fill(~mask,0.).sum((-2,-1))
    count=mask.sum((-2,-1));mean=independent/count.clamp_min(1)
    reference=-torch.minimum(mean.exp()*adv[:,None],mean.exp().clamp(.99,1.01)*adv[:,None]).masked_fill(count==0,0.).sum()/(4*8)
    expected=torch.autograd.grad(reference,parameters,retain_graph=True)[0]
    if offset==0.:
        score=-(mean*adv[:,None]).masked_fill(count==0,0.).sum()/(4*8)
        torch.testing.assert_close(torch.autograd.grad(score,parameters)[0],expected,rtol=1e-12,atol=1e-12)
    flat=lambda v:v.reshape(20,*v.shape[2:])
    with torch.no_grad():
        s=gaussian_statistics(flat(value),flat(old),flat(std),flat(old+parameters[0]*feature),
            flat(std*parameters[1].exp()),flat(mask))
        step=step_statistics(*(s[k].reshape(5,4) for k in ('joint_log_ratio','joint_kl','free_count')))
        coef=-stepwise_surrogate(step,adv,clip=.01)['joint_log_gradient']/(4*8)
    torch.testing.assert_close(step['joint_log_ratio'],independent,rtol=1e-11,atol=1e-11)
    for begin in range(0,20,microbatch):
        end=min(20,begin+microbatch)
        s=gaussian_statistics(flat(value)[begin:end],flat(old)[begin:end],flat(std)[begin:end],
            flat(old+parameters[0]*feature)[begin:end],flat(std*parameters[1].exp())[begin:end],flat(mask)[begin:end])
        (s['joint_log_ratio']*coef.flatten()[begin:end]).sum().backward()
    torch.testing.assert_close(parameters.grad,expected,rtol=1e-11,atol=1e-11)


def test_stepwise_grouped_audit_and_separate_chain_kl_are_additive():
    count=torch.tensor([[4.,10.,2.,8.],[1.,2.,4.,8.]],dtype=torch.float64)
    log=count*torch.tensor([[.005,.03,-.1,.002],[-.05,.002,.03,.001]],dtype=torch.float64)
    kl=torch.arange(8,dtype=torch.float64).reshape(2,4)/100
    step=step_statistics(log,kl,count);adv=torch.tensor([1.,-1.],dtype=torch.float64)
    report=stepwise_diagnostics(step,adv,clip=.01)
    groups=grouped_objective(step,adv,['A','B'],clip=.01,before_statistics=statistics(torch.zeros_like(log),count))
    assert groups['all']['objective_mean']==pytest.approx(report['stepwise_objective_mean'])
    assert groups['all']['gain_from_previous_same_membership']==pytest.approx(report['stepwise_objective_gain'])
    assert report['chain_joint_kl']['mean']==pytest.approx(float(kl.sum(1).mean()))
    assert report['mean_coordinate_kl']==pytest.approx(float((kl/count).mean()))
    assert report['proxy_clip_fraction']>report['proxy_active_clip_fraction']
    assert len(report['per_denoising_step'])==4 and not report['ratio_is_joint_probability']
    assert check_candidate_kl(report,{'mean_coordinate':1.,'chain_joint_max':.01})['failures'].keys()=={'chain_joint_max'}
    report['chain_joint_kl']['max']=float('nan')
    assert not check_candidate_kl(report,{})['passed']


def test_new_formal_configuration_has_explicit_proxy_and_uncalibrated_joint_tail():
    root=Path(__file__).resolve().parents[3]
    c=configuration(root/'configs/closedloop/stage2_stepwise_mean.yaml')
    old=configuration(root/'configs/closedloop/stage2_random20s_gamma095.yaml')
    s=c['stage10'];t=c['stage9'];require_current_actor_objective(t)
    assert t['ppo_clip']==.01 and t['gamma_denoising']==1. and t['bc_enabled'] is False
    assert t['objective_logprob_reduction']=='free_coordinate_mean' and t['actor_trainable_scope']=='all'
    assert s['kl']['mean_coordinate']==.0009 and s['kl']['maximum_step_mean_coordinate']==.0018
    assert s['kl']['p95_coordinate']==.0021
    assert all(s['kl']['chain_joint_'+k] is None for k in ('mean','p95','p99','max'))
    assert s['kl']['calibration']['joint_tail_status']=='uncalibrated_diagnostic_only'
    for key in ('critic','reward','physical_episode','generation_horizon','dataset','precision','ppo_rollout','scale'):
        assert s[key]==old['stage10'][key],key
    for key in ('actor_lr','actor_lr_multipliers','std_schedule','gamma_upper','lambda_upper'):
        assert t[key]==old['stage9'][key],key
    with pytest.raises(ValueError,match='read-only'):require_current_actor_objective(old['stage9'])


def test_explicit_new_objective_transfer_keeps_nonempty_adam_and_rejects_behavior_changes(tmp_path):
    import copy
    from gem.closedloop.dppo.training_checkpoint import SCHEMA,load_training_checkpoint
    from gem.closedloop.dppo.gradient_merge import BCGradientController
    from tests.closedloop.dppo.test_actor_backtracking import equal_state
    root=Path(__file__).resolve().parents[3]
    def identity(name):
        c=configuration(root/'configs/closedloop'/name);s=c['stage10']
        return {key:copy.deepcopy(s[key]) for key in ('training','critic','kl','scale','dataset','physical_episode',
            'reward','ppo_rollout','generation_horizon','bc_gradient')}
    old=identity('stage2_random20s_gamma095.yaml');new=identity('stage2_stepwise_mean.yaml')
    actor=torch.nn.Linear(2,1);critic=torch.nn.Linear(2,1)
    ao=torch.optim.AdamW(actor.parameters(),lr=1e-6);co=torch.optim.AdamW(critic.parameters(),lr=1e-4)
    for model,opt in ((actor,ao),(critic,co)):
        model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad()
    control=BCGradientController()
    before=copy.deepcopy((actor.state_dict(),critic.state_dict(),ao.state_dict(),co.state_dict(),control.state_dict()))
    path=tmp_path/'old.pt'
    torch.save(dict(schema=SCHEMA,identity=old,actor=before[0],critic=before[1],actor_optimizer=before[2],
        critic_optimizer=before[3],bc_controller=before[4],state={'iteration':90}),path)
    args=dict(identity=new,actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,controller=control)
    with pytest.raises(ValueError,match='configuration'):load_training_checkpoint(path,**args)
    restored=load_training_checkpoint(path,**args,new_experiment=True,stepwise_transfer=True)
    assert restored['state']['iteration']==90
    assert equal_state(before,(actor.state_dict(),critic.state_dict(),ao.state_dict(),co.state_dict(),control.state_dict()))
    for section,field,value in [('training','gamma_upper',.99),('training','actor_lr',3e-6),
            ('dataset','standing_start_probability',.2),('reward','track_weight',9.)]:
        changed=copy.deepcopy(new);changed[section][field]=value
        with pytest.raises(ValueError,match='configuration'):
            load_training_checkpoint(path,**dict(args,identity=changed),new_experiment=True,stepwise_transfer=True)
