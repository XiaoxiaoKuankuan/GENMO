"""完整链PPO的独立概率、裁剪与流式梯度数值验收。

先用独立Normal公式检查四步概率/KL、变长自由坐标和前缀/padding排除，再穷举正负
优势与裁剪分支，最后用真实小型Stage1网络对比完整Autograd图、生产流式反传与
行为起点joint-score梯度。微批切分不改变数学目标，测试不使用物理收益作为断言。
测试资产由已有pytest夹具放到临时目录，禁止向正式模型或训练目录写入。
"""
import copy
import math

import pytest
import torch

from gem.closedloop.dppo.gaussian_probability import (
    gaussian_statistics,chain_statistics,ppo_surrogate,chain_diagnostics)
from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,backward_chain_surrogate
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.optimization_support import _parameters,_old_kernel
from gem.closedloop.dppo.execution_checks import policy_phase
from tests.closedloop.dppo.test_clipped_dppo import case
from tests.closedloop.test_stage1_actor import actor_factory


def test_independent_four_transition_normal_formula_mask_and_equal_policy():
    generator=torch.Generator().manual_seed(372)
    shape=(7,4,9,3)
    old=torch.randn(shape,generator=generator,dtype=torch.float64)
    sigma=torch.rand(7,4,1,1,generator=generator,dtype=torch.float64)+.1
    value=old+sigma*torch.randn(shape,generator=generator,dtype=torch.float64)
    new=old+.03*torch.randn(shape,generator=generator,dtype=torch.float64)
    new_sigma=sigma*1.00001
    mask=torch.rand(shape,generator=generator)>.25
    mask[:,:,:2]=False;mask[:,:,-1]=False;mask[0]=False
    flat=lambda v:v.reshape(28,*v.shape[2:])
    actual=gaussian_statistics(flat(value),flat(old),flat(sigma),flat(new),flat(new_sigma),flat(mask))
    chain=chain_statistics(*(actual[k].reshape(7,4) for k in ('joint_log_ratio','joint_kl','free_count')))
    a=torch.distributions.Normal(old,sigma);b=torch.distributions.Normal(new,new_sigma)
    reference=(b.log_prob(value)-a.log_prob(value)).masked_fill(~mask,0.).sum((1,2,3))
    kl=torch.distributions.kl_divergence(a,b).masked_fill(~mask,0.).sum((1,2,3))
    torch.testing.assert_close(chain['joint_log_ratio'],reference,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(chain['joint_kl'],kl,rtol=1e-9,atol=1e-11)
    same=gaussian_statistics(flat(value),flat(old),flat(sigma),flat(old),flat(sigma),flat(mask))
    assert torch.equal(same['joint_log_ratio'],torch.zeros(28,dtype=torch.float64))
    assert torch.equal(same['joint_kl'],torch.zeros(28,dtype=torch.float64))
    assert torch.equal(same['joint_log_ratio'].exp(),torch.ones(28,dtype=torch.float64))
    ignored=new.clone();ignored[~mask]=float('nan')
    protected=gaussian_statistics(flat(value),flat(old),flat(sigma),flat(ignored),flat(new_sigma),flat(mask))
    torch.testing.assert_close(protected['joint_log_ratio'],actual['joint_log_ratio'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='15..18'):
        chain_statistics(torch.zeros(2,4),torch.zeros(2,4),torch.ones(2,4),steps=(16,17,18,19))


@pytest.mark.parametrize('advantage',[1.,-1.,0.])
@pytest.mark.parametrize('ratio',[.5,.95,1.,1.05,1.5])
def test_sign_dependent_chain_clip_and_exact_coefficient(advantage,ratio):
    log=torch.tensor([math.log(ratio)],dtype=torch.float64,requires_grad=True)
    stats=dict(joint_log_ratio=log,free_count=torch.tensor([4000]),joint_kl=log*0)
    out=ppo_surrogate(stats,torch.tensor([advantage]),clip=.1)
    reference=torch.minimum(log.exp()*advantage,log.exp().clamp(.9,1.1)*advantage)
    actual_grad=torch.autograd.grad(out['objective'].sum(),log,retain_graph=True)[0]
    ref_grad=torch.autograd.grad(reference.sum(),log)[0]
    torch.testing.assert_close(out['objective'],reference)
    torch.testing.assert_close(actual_grad,ref_grad)
    torch.testing.assert_close(out['objective_log_gradient'],ref_grad)
    assert out['ratio_is_joint_probability']


def test_joint_ratio_is_sum_not_coordinate_or_per_step_clip():
    # 每一步ratio在clip内，四步之积已越上界；必须整链失去正优势梯度。
    step=torch.full((1,4),math.log(1.04),dtype=torch.float64,requires_grad=True)
    stats=chain_statistics(step,torch.zeros_like(step),torch.full_like(step,1500))
    out=ppo_surrogate(stats,torch.ones(1),clip=.1)
    assert out['objective_clipped'].item()
    assert out['ratio'].item()==pytest.approx(1.04**4)
    out['objective'].sum().backward();assert torch.equal(step.grad,torch.zeros_like(step))
    report=chain_diagnostics(stats,torch.ones(1),clip=.1)
    assert report['active_clip_fraction']==1


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),1000.,-1000.])
def test_extreme_probability_is_explicit_failure_without_silent_clamp(bad):
    stats=dict(joint_log_ratio=torch.tensor([bad],dtype=torch.float64),free_count=torch.tensor([3]))
    with pytest.raises(FloatingPointError,match='rolled back'):
        ppo_surrogate(stats,torch.tensor([-1.]),clip=.1)


@pytest.mark.parametrize('offset',[0.,.015,-.02,.15])
@pytest.mark.parametrize('microbatch',[1,3,8])
def test_fp64_streamed_coefficient_matches_full_autograd_and_score(offset,microbatch):
    """严格数学对照用FP64解析网络；覆盖非行为起点和正负优势不同裁剪分支。"""
    generator=torch.Generator().manual_seed(831)
    x=torch.randn(7,4,3,2,generator=generator,dtype=torch.float64)
    feature=torch.randn(x.shape,generator=generator,dtype=torch.float64)
    mask=torch.rand(x.shape,generator=generator)>.2
    advantage=torch.tensor([1.,-1.,.7,-.3,1.4,-.5,.2],dtype=torch.float64)
    old=torch.zeros_like(x);sigma=torch.full((7,4,1,1),.5,dtype=torch.float64)
    theta=torch.tensor(offset,dtype=torch.float64,requires_grad=True)
    def stats(i):
        b,k=divmod(i,4)
        return gaussian_statistics(x[b:b+1,k],old[b:b+1,k],sigma[b:b+1,k],
            theta*feature[b:b+1,k],sigma[b:b+1,k],mask[b:b+1,k])
    values=[stats(i) for i in range(28)]
    chain=chain_statistics(*(torch.cat([v[k] for v in values]).reshape(7,4)
        for k in ('joint_log_ratio','joint_kl','free_count')))
    expected=-torch.minimum(chain['joint_log_ratio'].exp()*advantage,
        chain['joint_log_ratio'].exp().clamp(.9,1.1)*advantage).mean()
    reference=torch.autograd.grad(expected,theta,retain_graph=True)[0]
    if offset==0.:
        score=torch.autograd.grad(-(chain['joint_log_ratio']*advantage).mean(),theta)[0]
        torch.testing.assert_close(score,reference,rtol=1e-12,atol=1e-12)
    coeff=-ppo_surrogate(chain,advantage,clip=.1)['objective_log_gradient']/7
    for begin in range(0,28,microbatch):
        loss=sum(stats(i)['joint_log_ratio'].sum()*coeff[i//4] for i in range(begin,min(28,begin+microbatch)))
        loss.backward()
    torch.testing.assert_close(theta.grad,reference,rtol=1e-12,atol=1e-12)


def _compare_cpu_gradients(actor,reference):
    # CPU native_gemm不是生产的固定tile核。改变批量会使FP32前向出现舍入差，
    # 高维joint ratio会放大它；用整梯度相对L2报告，不对接近零的坐标作相对除法。
    # 上述FP64用例负责严格公式证明；生产stable_tf32x3另在真实八卡rollout上验收。
    delta=sum((p.grad.double()-reference[n].double()).square().sum()
        for n,p in actor.named_parameters() if n in reference)
    scale=sum(g.double().square().sum() for g in reference.values())
    relative=float((delta/scale).sqrt())
    assert relative<2e-4,relative


@pytest.mark.parametrize('microbatch',[1,3,11])
def test_real_small_network_streamed_full_autograd_and_initial_score_gradients(actor_factory,microbatch):
    policy,rows,targets=case(actor_factory)
    cache=RolloutTensorCache(rows,targets,'cpu')
    try:
        # 同一行为起点，完整Autograd参考保留四步图，用独立Normal公式而非流式系数。
        with policy_phase(policy):
            terms=[]
            for step in (15,16,17,18):
                params,mask=_parameters(policy,rows,[step]*len(rows),torch.device('cpu'),cache)
                old_mean,old_std=_old_kernel(rows,[step]*len(rows),torch.device('cpu'),cache)
                value=cache.get('chain',rows,[step+1]*len(rows)).double()
                new=torch.distributions.Normal(params['mean'].double(),params['std'].double())
                old=torch.distributions.Normal(old_mean,old_std)
                terms.append((new.log_prob(value)-old.log_prob(value)).masked_fill(~mask,0.).sum((-2,-1)))
            log=torch.stack(terms,1).sum(1)
            advantage=targets['advantages'].double()
            loss=-torch.minimum(log.exp()*advantage,log.exp().clamp(.9,1.1)*advantage).mean()
            loss.backward()
        reference={n:p.grad.clone() for n,p in policy.actor.named_parameters() if p.grad is not None}
        policy.actor.zero_grad(set_to_none=True)
        with policy_phase(policy):
            records=evaluate_chain_records(policy,rows,cache,microbatch=microbatch)
            backward_chain_surrogate(policy,rows,cache,records,clip=.1,global_chain_count=len(rows),microbatch=microbatch)
        _compare_cpu_gradients(policy.actor,reference)
        policy.actor.zero_grad(set_to_none=True)
        with policy_phase(policy):
            for step in (15,16,17,18):
                params,mask=_parameters(policy,rows,[step]*len(rows),torch.device('cpu'),cache)
                old_mean,old_std=_old_kernel(rows,[step]*len(rows),torch.device('cpu'),cache)
                score=gaussian_statistics(cache.get('chain',rows,[step+1]*len(rows)),old_mean,old_std,params['mean'],params['std'],mask)
                (-score['joint_log_ratio']*advantage/len(rows)).sum().backward()
        _compare_cpu_gradients(policy.actor,reference)
    finally:cache.close()
