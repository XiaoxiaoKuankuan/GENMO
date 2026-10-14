"""真实小型Actor的冻结前16步与末4步优化合同测试。

在CPU使用原Stage1条件编码器、扩散均值和Adam，检查完整20步采样概率复算、
混合去噪步的缓存索引、非空Adam后前段不漂移而末段变化，以及冻结参数绝不接收
梯度。临时数据由pytest basetemp回收；本测试不冒充正式GPU数值或物理验收。
"""
from types import SimpleNamespace
import pytest
import torch
from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
from gem.closedloop.dppo.tensor_cache import ConditionGraphCache
from gem.closedloop.dppo.policy import masked_joint_log_prob
from tests.closedloop.test_stage1_actor import actor_factory, _conditions, _activate_branches


def test_full_chain_mixed_replay_and_nonempty_adam_freeze(actor_factory):
    actor,batch=actor_factory(motion_frames=60)
    _activate_branches(actor)
    policy=FrozenPrefixPolicy(actor,std_schedule=[.001]*16+[.08,.1,.12,.12],generation_frames=60)
    context=_conditions(batch)
    rollout=policy.sample_rollout(context,generator=torch.Generator().manual_seed(17))
    rows=[SimpleNamespace(context={k:v[i:i+1] for k,v in context.items()}) for i in range(2)]
    steps=torch.tensor([0,16,19,8]);indices=torch.tensor([1,0,1,0])
    with torch.no_grad():
        cache=ConditionGraphCache(policy);cache.prime(rows)
        parameters=cache.transition([rows[i] for i in indices],rollout['chain'][indices,steps],steps)
        probability=masked_joint_log_prob(rollout['chain'][indices,steps+1],parameters['mean'],parameters['std'],parameters['free_mask'])
        torch.testing.assert_close(probability,rollout['old_log_probs'][indices,steps],atol=2e-3,rtol=1e-7)
        early_before=policy.transition_parameters(context,rollout['chain'][:,0],0)['mean'].clone()
        late_before=policy.transition_parameters(context,rollout['chain'][:,19],19)['mean'].clone()
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        cache=ConditionGraphCache(policy);cache.prime(rows)
        params=cache.transition(rows,rollout['chain'][:,19],torch.tensor([19,19]))
        params['mean'].square().mean().backward();cache.backward();optimizer.step()
    assert optimizer.state
    assert all(p.grad is None and not p.requires_grad for p in policy.frozen_policy.actor.parameters())
    with torch.no_grad():
        torch.testing.assert_close(policy.transition_parameters(context,rollout['chain'][:,0],0)['mean'],early_before,atol=0,rtol=0)
        assert not torch.equal(policy.transition_parameters(context,rollout['chain'][:,19],19)['mean'],late_before)
    with pytest.raises(ValueError,match='diagnostic only'):
        policy.transition_parameters(context,rollout['chain'][:,0],0)


def test_frozen_weights_are_independent_and_part_of_identity(actor_factory):
    actor,_=actor_factory(motion_frames=60)
    policy=FrozenPrefixPolicy(actor,std_schedule=[.001]*20,generation_frames=60)
    assert policy.kernel_config['frozen_prefix']['steps']==16
    signature=policy._parameter_signature()
    with torch.no_grad():next(policy.frozen_policy.actor.parameters()).add_(1)
    assert signature!=policy._parameter_signature()
