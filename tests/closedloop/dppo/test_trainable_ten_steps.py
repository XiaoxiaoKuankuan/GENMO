"""独立10步候选的路由、真实旧链重放、梯度和更新链路验收。

使用真实小型Stage1 Actor与保存完整19随机转移的4步行为链。仅在Actor仍逐张量
等于冻结先验时建立等分布诊断，验证没有重写行为元数据；10个转移各自clip、按
10×全局链数归一化。独立Normal公式及完整Autograd对照流式微批反向，覆盖尾批。
生产更新器与最终KL必须实际遍历9..18。测试不运行PhysX，不声明运动性能改善。
"""
import copy
from pathlib import Path
import pytest
import torch
import torch.distributed as dist

from tests.closedloop.test_stage1_actor import actor_factory
from tests.closedloop.dppo.test_clipped_dppo import case,Anchor,close_cpu_group
from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
from gem.closedloop.dppo.step_expansion_diagnostic import bind_stage1_step_expansion
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,backward_chain_surrogate,from_records
from gem.closedloop.dppo.gaussian_probability import stepwise_diagnostics
from gem.closedloop.dppo.optimization_support import probability_check_local


def expanded(actor_factory):
    old,rows,targets=case(actor_factory)
    from gem.closedloop.dppo.rollout_helpers import split_trace
    # CPU native GEMM对批形不保证逐位一致；夹具按单条真实采样，并固定条件图批形。
    # 正式GPU测试仍使用原始真实行为文件与stable_tf32x3，不改生产容差或数据。
    for i,row in enumerate(rows):
        trace=old.sample_rollout(row.context,generator=torch.Generator().manual_seed(17+i))
        row.chain=trace['chain'][0].clone();row.old_log_prob=trace['old_log_probs'][0].clone()
        row.free_mask=trace['free_mask'][0].clone();row.metadata['sampler_trace']=split_trace(trace,0,1)
    new=FrozenPrefixPolicy(copy.deepcopy(old.actor),frozen_prefix_steps=9,
        std_schedule=old.std_schedule,generation_frames=60)
    new.condition_cache_max_chains=1
    for row in rows:row.transition_valid=True
    proof=bind_stage1_step_expansion(new,old.kernel_config)
    return old,new,rows,targets,proof


def test_expansion_keeps_real_behavior_and_rejects_changed_actor_or_kernel(actor_factory):
    old,new,rows,targets,proof=expanded(actor_factory)
    assert proof['behavior_data_rewritten'] is False
    assert rows[0].metadata['sampler_trace']['kernel_config']==old.kernel_config
    assert new.trainable_steps==tuple(range(9,19))
    assert new.assert_frozen_identity()['stochastic_frozen_steps']==list(range(9))
    cache=RolloutTensorCache(rows,targets,'cpu')
    try:
        result=probability_check_local(new,rows,tensor_cache=cache,denoising_microbatch=1)
        assert result['passed'] and result['checked_steps']==list(range(19))
        from gem.closedloop.dppo.actor_step_diagnostics import output_probe
        a=output_probe(old,rows,cache);b=output_probe(new,rows,cache)
        torch.testing.assert_close(a['final'],b['final'],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(a['qpos'],b['qpos'],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(a['contact'],b['contact'],rtol=1e-6,atol=1e-6)
    finally:cache.close()
    bad=copy.deepcopy(old.kernel_config);bad['unexpected_noise_change']=True
    with pytest.raises(ValueError,match='actual Gaussian kernel'):bind_stage1_step_expansion(new,bad)
    with torch.no_grad():next(new.actor.parameters()).add_(.01)
    with pytest.raises(ValueError,match='original Stage1'):bind_stage1_step_expansion(new,old.kernel_config)


@pytest.mark.parametrize('microbatch',[3,11])
def test_ten_step_streamed_gradient_matches_independent_normal_autograd(actor_factory,microbatch):
    old,policy,rows,targets,_=expanded(actor_factory)
    cache=RolloutTensorCache(rows,targets,'cpu');global_count=11
    try:
        with policy_phase(policy):
            records=evaluate_chain_records(policy,rows,cache,microbatch=microbatch)
            report=backward_chain_surrogate(policy,rows,cache,records,clip=.01,
                global_chain_count=global_count,microbatch=microbatch)
            grads={n:p.grad.clone() for n,p in policy.actor.named_parameters() if p.grad is not None}
            policy.actor.zero_grad(set_to_none=True)
            context=cache.context(rows);prepared=policy.prepare_conditions(context,include_frozen=False)
            terms=[];mask=cache.get('free_mask',rows);count=mask.sum((-2,-1));adv=cache.get('advantages',rows).double()
            for step in policy.trainable_steps:
                params=policy.transition_parameters(context,cache.get('chain',rows,[step]*len(rows)),step,prepared=prepared)
                x=cache.get('chain',rows,[step+1]*len(rows)).double()
                new=torch.distributions.Normal(params['mean'].double(),params['std'].double()).log_prob(x)
                old=torch.distributions.Normal(cache.get('old_means',rows,[step]*len(rows)).double(),
                    cache.get('old_stds',rows,[step]*len(rows)).double()).log_prob(x)
                ratio=((new-old).masked_fill(~mask,0.).sum((-2,-1))/count).exp()
                terms.append(torch.minimum(ratio*adv,ratio.clamp(.99,1.01)*adv))
            loss=-torch.stack(terms,1).sum()/(10*global_count);loss.backward()
        torch.testing.assert_close(report['loss'],loss.detach(),rtol=1e-6,atol=1e-8)
        for n,p in policy.actor.named_parameters():
            if n in grads:torch.testing.assert_close(p.grad,grads[n],rtol=2e-4,atol=2e-6)
        diagnostics=stepwise_diagnostics(from_records(records),adv,clip=.01)
        assert [r['step'] for r in diagnostics['per_denoising_step']]==list(range(9,19))
        assert diagnostics['denominator']=='10_times_global_valid_behavior_chain_count'
    finally:cache.close()
