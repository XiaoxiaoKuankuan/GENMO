"""新20步链的15冻结＋4可训练随机＋1冻结确定性合同验收。

验证实际采样的19个高斯转移、末步无噪声且contact来自冻结预测，条件缓存、前缀保护、
非空Adam两次更新及冻结身份检测。只更新中间四步的独立Actor，保留全部20次生成；
CPU小网络误差门槛不替代真实GPU批量一致性门槛，测试产物仅写pytest临时目录。
"""
import pytest
import torch
from types import SimpleNamespace
from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions
from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
from gem.closedloop.dppo.policy import masked_joint_log_prob
from gem.closedloop.dppo.tensor_cache import ConditionGraphCache


def test_intermediate4_gradients_and_deterministic_final(actor_factory):
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    policy=FrozenPrefixPolicy(actor,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
    context=_conditions(batch)
    rollout=policy.sample_rollout(context,generator=torch.Generator().manual_seed(17))
    assert rollout['old_log_probs'].shape==(2,19)
    assert rollout['sampling_noise'].shape==(2,19,60,30)
    assert rollout['probability_steps'].tolist()==list(range(19))
    rows=[SimpleNamespace(context={k:v[i:i+1] for k,v in context.items()}) for i in range(2)]
    steps=torch.tensor([0,15,18,8]);indices=torch.tensor([1,0,1,0])
    with torch.no_grad():
        cache=ConditionGraphCache(policy);cache.prime(rows)
        params=cache.transition([rows[i] for i in indices],rollout['chain'][indices,steps],steps)
        probability=masked_joint_log_prob(rollout['chain'][indices,steps+1],params['mean'],params['std'],params['free_mask'])
        torch.testing.assert_close(probability,rollout['old_log_probs'][indices,steps],atol=2e-3,rtol=1e-7)
        early=policy.transition_parameters(context,rollout['chain'][:,0],0)['mean'].clone()
        final=policy.transition_parameters(context,rollout['chain'][:,19],19)
        assert final['std'] is None
        with pytest.raises(ValueError,match='no Gaussian'):
            policy.evaluate_log_probs(context,rollout['chain'][:,19],rollout['chain'][:,20],19)
        torch.testing.assert_close(rollout['chain'][:,20],final['mean'],atol=0,rtol=0)
        old_last=final['mean'].clone()
        late=policy.transition_parameters(context,rollout['chain'][:,18],18)['mean'].clone()
    opt=torch.optim.Adam(actor.parameters(),lr=1e-4)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        cache=ConditionGraphCache(policy);cache.prime(rows)
        params=cache.transition(rows,rollout['chain'][:,18],torch.tensor([18,18]))
        params['mean'].square().mean().backward();cache.backward();opt.step()
    assert opt.state
    assert all(p.grad is None and not p.requires_grad for p in policy.frozen_policy.actor.parameters())
    policy.assert_frozen_identity()
    with torch.no_grad():
        torch.testing.assert_close(policy.transition_parameters(context,rollout['chain'][:,0],0)['mean'],early,atol=0,rtol=0)
        torch.testing.assert_close(policy.transition_parameters(context,rollout['chain'][:,19],19)['mean'],old_last,atol=0,rtol=0)
        assert not torch.equal(policy.transition_parameters(context,rollout['chain'][:,18],18)['mean'],late)
    with torch.no_grad():next(policy.frozen_policy.actor.parameters()).add_(1)
    with pytest.raises(RuntimeError):policy.assert_frozen_identity()


def test_final_kl_covers_four_steps_and_regenerates_reference(actor_factory):
    from gem.closedloop.dppo.rollout_helpers import split_trace
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.updater import final_kl
    from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
    from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    privileged=tracking_context(_conditions(batch))
    context={k:privileged[k] for k in STAGE1_CONDITION_KEYS}
    policy=FrozenPrefixPolicy(actor,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
    trace=policy.sample_rollout(context,generator=torch.Generator().manual_seed(17))
    rows=[]
    for i in range(2):
        rows.append(SimpleNamespace(context={k:v[i:i+1] for k,v in context.items()},identity={},
            chain=trace['chain'][i],old_log_prob=trace['old_log_probs'][i],free_mask=trace['free_mask'][i],
            metadata=dict(critic_context={k:v[i:i+1] for k,v in privileged.items()},sampler_trace=split_trace(trace,i,2),remaining_music_seconds=5.)))
    cache=RolloutTensorCache(rows,dict(advantages=torch.tensor([1.,-1.])),device='cpu')
    manifest=[dict(valid=True,has_free=True,owner_rank=0,local_index=i) for i in range(2)]
    d=SimpleNamespace(rank=0,world_size=1,all_gather_object=lambda x:[x],gather_rows=lambda x,indices,n:x)
    report=final_kl(policy,rows,cache,manifest,distributed=d,microbatch=3)
    assert report['gaussian_kl_steps']==[15,16,17,18] and report['covered_internal_transitions']==8
    assert report['mean_coordinate_kl']<1e-8
    assert len(report['per_denoising_step'])==4
    assert all(x['protected_coordinates_unchanged'] for x in report['reference_change']['samples'])
    assert all(p.grad is None for p in actor.parameters())
    cache.close()
