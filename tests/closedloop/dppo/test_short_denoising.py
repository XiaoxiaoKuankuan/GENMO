"""十步扩散与短动作、完整音乐上下文的生产路径回归。

采用原Stage1小型真实网络，检查15/30帧动作与60帧音乐各自保持30Hz，未来音乐
确实影响输出、前缀不变、完整十步概率可重算，以及五个可训练随机步骤的真实Adam
更新、Critic缓存及整轮末尾的完整KL/实际参考变化检查，防止优化结束后仍访问旧二十步
索引而无法保存。另验证有限log-ratio下溢只拒绝试更新，NaN仍视为真正异常。
所有临时模型和Gloo文件由pytest专用临时目录管理，不写正式训练资产；不替代PhysX。
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
import torch
import torch.distributed as dist
from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions
from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
from tests.closedloop.dppo.test_actor_backtracking import setup, Collective, equal_state
from gem.closedloop.contracts import validate_stage1_condition_batch
from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
from gem.closedloop.dppo.gaussian_probability import CandidateRatioRangeError
from gem.closedloop.dppo.actor_backtracking import backtracked_step


def short_case(actor_factory, frames, music_frames=60):
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    c=_conditions(batch)
    for key in ('known_qpos30','known_qpos30_mask','future_valid','future_times'):
        c[key]=c[key][:,:frames].clone()
    for key in ('music_features','music_valid'):
        c[key]=c[key][:,:music_frames].clone()
    validate_stage1_condition_batch(c)
    policy=FrozenPrefixPolicy(actor,steps=10,frozen_prefix_steps=4,
        std_schedule=[.001]*4+[.02,.04,.06,.08,.1,0.],generation_frames=frames,music_context_frames=music_frames)
    return policy,c


@pytest.mark.parametrize('frames',[15,30,60])
def test_ten_steps_and_music_context_survive_replay(actor_factory,frames):
    from gem.closedloop.dppo.tensor_cache import ConditionGraphCache
    policy,c=short_case(actor_factory,frames)
    trace=policy.sample_rollout(c,generator=torch.Generator().manual_seed(17))
    assert trace['chain'].shape==(2,11,frames,30)
    assert trace['old_log_probs'].shape==(2,9)
    assert policy.trainable_steps==tuple(range(4,9))
    rows=[SimpleNamespace(context={k:v[i:i+1] for k,v in c.items()}) for i in range(2)]
    with torch.no_grad():
        cache=ConditionGraphCache(policy);cache.prime(rows)
        p=cache.transition(rows,trace['chain'][:,7],torch.tensor([7,7]))
        from gem.closedloop.dppo.policy import masked_joint_log_prob
        actual=masked_joint_log_prob(trace['chain'][:,8],p['mean'],p['std'],p['free_mask'])
        torch.testing.assert_close(actual,trace['old_log_probs'][:,7],atol=2e-3,rtol=1e-7)
        final=policy.transition_parameters(c,trace['chain'][:,9],9)
        assert final['std'] is None
        torch.testing.assert_close(final['mean'],trace['chain'][:,10],rtol=0,atol=0)
        if frames<60:
            changed={k:v.clone() for k,v in c.items()}
            changed['music_features'][:,frames:]+=1.
            original=policy.transition_parameters(c,trace['chain'][:,7],7)['mean']
            different=policy.transition_parameters(changed,trace['chain'][:,7],7)['mean']
            assert not torch.equal(original,different), 'Future music must not be silently cropped'
    policy.assert_frozen_identity()


@pytest.mark.parametrize('frames,music_frames',[(15,60),(30,60),(60,60),(30,30)])
def test_short_motion_critic_and_five_step_optimizer(actor_factory,tmp_path,frames,music_frames):
    from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
    from gem.closedloop.dppo.rollout_helpers import split_trace
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.collectives import DistributedCollectives
    from gem.closedloop.dppo.continuation import DisabledBC
    from gem.closedloop.dppo.gradient_merge import BCGradientController
    from gem.closedloop.dppo.updater import actor_update, final_kl
    policy,c=short_case(actor_factory,frames,music_frames);priv=tracking_context(c)
    critic=PretrainedConditionCritic(policy.actor,dimension=32,layers=2,heads=4)
    values=critic(priv,[3.,4.]);values.square().mean().backward()
    assert torch.isfinite(values).all() and critic.history_encoder.cell.weight_ih.grad is not None
    trace=policy.sample_rollout(c,generator=torch.Generator().manual_seed(17))
    rows=[SimpleNamespace(context={k:v[i:i+1].clone() for k,v in c.items()},identity={},
        chain=trace['chain'][i],old_log_prob=trace['old_log_probs'][i],free_mask=trace['free_mask'][i],
        metadata=dict(critic_context={k:v[i:i+1].clone() for k,v in priv.items()},
            sampler_trace=split_trace(trace,i,2),remaining_music_seconds=3.)) for i in range(2)]
    targets=dict(advantages=torch.tensor([1.,-1.]),valid=torch.ones(2,dtype=torch.bool))
    cache=RolloutTensorCache(rows,targets,'cpu')
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(2)]
    dist.init_process_group('gloo',init_method=(tmp_path/'gloo').as_uri(),rank=0,world_size=1)
    opt=torch.optim.AdamW([p for p in policy.actor.parameters() if p.requires_grad],lr=1e-9,weight_decay=0.)
    try:
        report=actor_update(policy,opt,rows,targets,cache,manifest,BCGradientController(),
            distributed=DistributedCollectives(0,1,device='cpu'),anchor=DisabledBC(),
            generator=torch.Generator().manual_seed(4),minibatch_chains=2,microbatch=3,
            sampled_steps=5,step_strategy='intermediate5',bc_enabled=False,bc_samples_per_rollout=0,
            audit_full_rollout_objective=True)
        assert report['optimizer_steps']==1 and report['internal_sample_visits']==10
        assert report['steps'][0]['denoising_steps']==[4,5,6,7,8]
        assert opt.state
        final=final_kl(policy,rows,cache,manifest,
            distributed=DistributedCollectives(0,1,device='cpu'),microbatch=3)
        assert final['covered_internal_transitions']==10
        assert final['reference_change']['covered_chains']==2
        assert final['final_reference_mean_change']['scope']==(
            'denormalized_index8_Gaussian_mean_not_final_reference')
        for sample in final['reference_change']['samples']:
            assert sample['protected_coordinates_unchanged']
            assert sample['normalized_rms']>=0.
    finally:cache.close();dist.destroy_process_group()


def test_aligned_one_second_enters_transformer_as_thirty_positions(actor_factory):
    policy,c=short_case(actor_factory,30,30)
    observed=[]
    def observe(module,args,kwargs):
        observed.append((args[0].shape[1],kwargs['y']['f_cond'].shape[1]))
    handles=[model.denoiser.register_forward_pre_hook(observe,with_kwargs=True)
        for model in (policy.actor,policy.frozen_policy.actor)]
    try:
        trace=policy.sample_rollout(c,generator=torch.Generator().manual_seed(71))
    finally:
        for handle in handles:handle.remove()
    assert observed and all(item==(30,30) for item in observed)
    assert trace['chain'].shape==(2,11,30,30)
    adapted=policy.actor.adapt_conditions(c)
    mask=adapted['known_mask']
    torch.testing.assert_close(trace['chain'][:,-1][mask],adapted['known_x'][mask],rtol=0,atol=0)
    from gem.closedloop.dppo.training_config import configuration
    from gem.closedloop.dppo.actor_optimization import optimization_binding
    root=Path(__file__).resolve().parents[3]
    config=configuration(root/'configs/closedloop/stage2_ddim10_last5_motion30_music30_klrelaxed_server2.yaml')
    stage=config['stage10'];t=stage['training']
    assert t['music_context_frames']==t['generation_frames']==30
    assert stage['kl']['chain_joint_mean']==2. and t['kl_soft_stop_chain_joint']==1.
    assert optimization_binding(stage)['candidate_acceptance']=='finite_KL_only_first_qualified.v1'
    assert set(t['trust_region'])=={'maximum_lr','growth','margin','max_attempts','minimum_trial_lr_ratio'}


@pytest.mark.parametrize('prefix',[9,15])
def test_old_twenty_step_final_diagnostics_remain_compatible(actor_factory,tmp_path,prefix):
    from tests.closedloop.dppo.test_clipped_dppo import case
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.collectives import DistributedCollectives
    from gem.closedloop.dppo.updater import final_kl
    policy,rows,targets=case(actor_factory,frozen_prefix_steps=prefix)
    cache=RolloutTensorCache(rows,targets,'cpu')
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(len(rows))]
    dist.init_process_group('gloo',init_method=(tmp_path/'gloo').as_uri(),rank=0,world_size=1)
    try:
        result=final_kl(policy,rows,cache,manifest,
            distributed=DistributedCollectives(0,1,device='cpu'),microbatch=3)
        assert result['covered_internal_transitions']==len(rows)*len(policy.trainable_steps)
        assert result['final_reference_mean_change']['scope']==(
            'denormalized_index18_Gaussian_mean_not_final_reference')
        assert result['reference_change']['covered_chains']==len(rows)
        for sample in result['reference_change']['samples']:
            assert sample['protected_coordinates_unchanged']
            assert sample['normalized_rms']<1e-5
    finally:cache.close();dist.destroy_process_group()


def test_range_rejections_are_recoverable_but_nan_is_fatal():
    for error_type,recoverable in [(CandidateRatioRangeError,True),(FloatingPointError,False)]:
        actor,opt,c=setup();before=copy.deepcopy((actor.state_dict(),opt.state_dict()))
        def reject():raise error_type('synthetic range or nonfinite failure')
        report=backtracked_step(actor,opt,c,evaluate=reject,limits={},distributed=Collective(),max_attempts=2)
        assert not report['accepted'] and report['finite_trust_region_exhausted']==recoverable
        assert equal_state(before,(actor.state_dict(),opt.state_dict()))


@pytest.mark.parametrize('logratio,expected',[(-800.,CandidateRatioRangeError),
    (800.,CandidateRatioRangeError),(float('nan'),FloatingPointError),
    (float('inf'),FloatingPointError)])
def test_production_ratio_check_distinguishes_finite_range_from_corruption(logratio,expected):
    from gem.closedloop.dppo.gaussian_probability import ppo_surrogate
    statistics=dict(joint_log_ratio=torch.tensor([0.,logratio],dtype=torch.float64),
        free_count=torch.ones(2,dtype=torch.long))
    original=statistics['joint_log_ratio'].clone()
    with pytest.raises(expected) as error:
        ppo_surrogate(statistics,torch.tensor([1.,-1.]),clip=.1)
    assert type(error.value) is expected
    torch.testing.assert_close(statistics['joint_log_ratio'],original,equal_nan=True)


@pytest.mark.parametrize('frames',[15,30,60])
def test_new_config_and_old_config_remain_distinct(frames):
    from gem.closedloop.dppo.training_config import configuration
    root=Path(__file__).resolve().parents[3]
    c=configuration(root/f'configs/closedloop/stage2_ddim10_last5_motion{frames}_server2.yaml')
    assert c['stage10']['training']['denoising_steps']==10
    assert c['stage10']['training']['ppo_steps_per_chain']==5
    assert c['stage10']['training']['actor_lr']==3e-6
    assert c['stage10']['generation_horizon']['frames']==frames
    assert configuration(root/'configs/closedloop/stage2_continuous_train10_lr10x_server2.yaml')['model']['ddim_steps']==20
