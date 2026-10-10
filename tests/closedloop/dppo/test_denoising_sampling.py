"""服务器1执行的去噪时间步抽样损失、随机恢复与全量KL边界测试。

用真实联合高斯PPO公式和可控小策略验证：全步模式不增加RNG消耗，抽样身份与rank
分片无关，重复种子精确重建；轮转无放回子集平均梯度等于原完整目标。4/8步只减少
内部loss样本，old链和概率不变、链minibatch和BC频率不变，最终KL仍覆盖所有20步。
这不是把抽样梯度等同单次全梯度：单次抽样方差需由真实rollout诊断单独报告。
"""
import copy
import pytest
import torch
from gem.closedloop.dppo.denoising_sampling import chain_steps,plan_record
from gem.closedloop.dppo.updater_v2 import actor_update_v2,analytic_kl_local
from gem.closedloop.dppo.policy import masked_joint_log_prob
from tests.closedloop.dppo.test_updater_v2 import BatchGaussianPolicy, rows, Anchor


class TwentySteps(BatchGaussianPolicy):
    steps=20
    timestep_map=tuple(range(999,0,-50))


def samples(policy):
    result=rows(BatchGaussianPolicy(),2)
    generator=torch.Generator().manual_seed(135)
    for row in result:
        chain=torch.zeros(21,120,30);means=[];stds=[];prob=[]
        with torch.no_grad():
            for step in range(20):
                parameters=policy.transition_parameters(row.context,chain[step:step+1],step)
                chain[step+1]=parameters['mean'][0]+.7*torch.randn(120,30,generator=generator)
                means.append(parameters['mean']);stds.append(parameters['std'])
                prob.append(masked_joint_log_prob(chain[step+1:step+2],parameters['mean'],parameters['std'],row.free_mask[None])[0])
        row.chain=chain;row.old_log_prob=torch.stack(prob)
        row.metadata['sampler_trace'].update(kernel_config=policy.kernel_config,timestep_map=torch.tensor(policy.timestep_map),
            old_means=torch.stack(means,1),old_stds=torch.stack(stds,1))
    return result


def test_plan_is_independent_of_owner_and_full_steps_do_not_need_seed():
    for count in (4,8,20):
        entire={i:chain_steps(91,i,20,count) for i in range(33)}
        merged={i:chain_steps(91,i,20,count) for rank in range(8) for i in range(rank,33,8)}
        assert entire==merged
        assert all(len(set(value))==count for value in entire.values())
        report=plan_record(91 if count<20 else None,list(range(33)),20,count)
        assert sum(report['step_histogram'])==33*count and report['inverse_probability_weight']==20/count
    assert chain_steps(None,0,20,20)==list(range(20))


def test_sampled_ppo_is_unbiased_keeps_old_and_full_kl(monkeypatch):
    torch.set_num_threads(1)
    policy=TwentySteps();data=samples(policy);base=copy.deepcopy(policy.actor.state_dict())
    old=torch.stack([row.old_log_prob.clone() for row in data]);gradients=[]
    def update(count):
        policy.actor.load_state_dict(base);optimizer=torch.optim.SGD(policy.actor.parameters(),lr=1e-5)
        generator=torch.Generator().manual_seed(731);before=generator.get_state().clone();anchor=Anchor();observed={}
        def observe(actor,step):
            observed.update({n:p.grad.clone() for n,p in actor.named_parameters() if p.grad is not None})
        report=actor_update_v2(policy,optimizer,data,dict(advantages=torch.tensor([1.,-.6])),ppo_epochs=1,
            epoch_orders=[[0,1]],actor_minibatch_internal_transitions=40,denoising_microbatch=7,
            max_optimizer_steps=1,soft_kl_limit=None,denoising_steps_per_chain=count,
            generator=generator,gradient_observer=observe,gradient_diagnostics=False,
            bc=anchor,bc_weight=.1,kl_check_mode='pre_step_plus_final')
        assert report['optimizer_steps']==1 and anchor.calls==1
        assert report['applied_internal_sample_visits']==2*count
        assert torch.equal(torch.stack([row.old_log_prob for row in data]),old)
        if count==20:assert torch.equal(generator.get_state(),before)
        kl=analytic_kl_local(policy,data,denoising_microbatch=13)
        assert kl['fresh_internal_forwards']==40 and len(kl['per_denoising_step'])==20
        return observed
    expected=update(20)
    for count in (4,8):
        gradients=[]
        for offset in range(20):
            monkeypatch.setattr('gem.closedloop.dppo.denoising_sampling.chain_steps',
                lambda seed,index,total,sampled,offset=offset:sorted((offset+j)%total for j in range(sampled)))
            gradients.append(update(count))
        for name,value in expected.items():
            torch.testing.assert_close(torch.stack([g[name] for g in gradients]).mean(0),value,atol=3e-5,rtol=2e-4)


def test_explicit_finite_launcher_preserves_full_behavior_and_resume_objective():
    from tools.validate_stage10_scale import select_learning_experiment
    from gem.closedloop.dppo.training_scale import derive_training_scale
    import copy
    config=dict(runtime=dict(num_envs=64),stage10=dict(performance={},distributed=dict(world_size=8),
        scale=dict(decisions_per_environment=16,actor_minibatch_chains=2048,critic_epochs=4),
        training=dict(denoising_steps=20,ppo_epochs=2,critic_batch=1024,bc_batch=128,actor_lr=5e-9)))
    select_learning_experiment(config,steps=4,actor_lr=2.5e-9,sensitive_output_fp64=True)
    derive_training_scale(config);stage=config['stage10']
    assert stage['training']['denoising_steps']==20
    assert stage['training']['max_actor_optimizer_steps']==8
    assert stage['derived_scale']['global_internal_transitions']==8192*20
    assert stage['derived_scale']['ppo_local_internal_samples']==1024*4*2
    assert stage['derived_scale']['ppo_full_local_internal_samples']==1024*20*2
    saved=copy.deepcopy(config)
    select_learning_experiment(config,resume=True)
    assert config==saved
    for arguments in (dict(steps=8),dict(actor_lr=5e-9)):
        with pytest.raises(ValueError,match='Resume cannot change'):
            select_learning_experiment(config,resume=True,**arguments)
