"""非对称三编码器Critic的初始化、因果隔离及跨优化步缓存回归。

使用缩小的真实Stage1网络与合成50Hz过去历史验证90+1维GRU权重映射。检查冻结特征
可复用但历史GRU每次建立新梯度图；独立Actor参数/存储和GMT没有优化入口。注入未来
时间、历史目标错位及伪造proprio时必须拒绝。这是服务器2 CPU合约测试，不替代MC质量。
"""
import copy
import pytest
import torch
from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
from gem.closedloop.dppo.tracking_context import validate_tracking_context


def tracking_context(context):
    c={k:v.clone() for k,v in context.items()}
    # 合成试验明确按50Hz构造过去1秒，内容仍使用Stage1测试夹具原proprio。
    c['proprio_history_times']=c['decision_time'][:,None]+torch.arange(-49,1,dtype=torch.float64)[None]*.02
    valid=c['proprio_history_valid']
    c['tracking_history']=torch.cat((c['proprio_history'],torch.randn(len(valid),50,42)),dim=-1)
    c['tracking_history']=torch.where(valid[...,None],c['tracking_history'],0.)
    c['tracking_history_valid']=valid.clone()
    c['tracking_history_times']=c['proprio_history_times'].clone()
    c['tracking_reference_times']=c['proprio_history_times'].clone()
    return validate_tracking_context(c)


def test_stage1_initialization_and_trainable_history_cache(actor_factory):
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    before=copy.deepcopy(actor.state_dict())
    critic=PretrainedConditionCritic(actor,dimension=32,layers=2,heads=4)
    c=tracking_context(_conditions(batch));f=critic.encode_frozen(c)
    old=actor.history_encoder.cell;new=critic.history_encoder.cell
    torch.testing.assert_close(new.weight_ih[:,:48],old.weight_ih[:,:48],atol=0,rtol=0)
    torch.testing.assert_close(new.weight_ih[:,90],old.weight_ih[:,48],atol=0,rtol=0)
    assert torch.count_nonzero(new.weight_ih[:,48:90])==0
    with torch.no_grad():
        expected=actor.history_encoder(f['history_input'][...,:48],f['history_valid'],f['history_relative'])
        actual=critic.history_encoder(f['history_input'],f['history_valid'],f['history_relative'])
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-6)
    assert not any(v.requires_grad for v in f.values())
    optimizer=torch.optim.Adam([p for p in critic.parameters() if p.requires_grad],lr=1e-3)
    values=[]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value=critic.forward_encoded(f,[4.,5.]);values.append(value.detach().clone())
        (value-torch.tensor([2.,-2.])).square().mean().backward()
        assert new.weight_ih.grad[:,48:90].abs().sum()>0
        assert all(p.grad is None for m in critic.encoder_modules() for p in m.parameters())
        assert all(p.grad is None for p in actor.parameters())
        optimizer.step()
    assert optimizer.state and not torch.equal(values[0],values[1])
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
    assert not ({p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()})


@pytest.mark.parametrize('fault',['future','reference','actor','padding','extra'])
def test_causal_contract_rejects_privilege_leaks(actor_factory,fault):
    actor,batch=actor_factory(motion_frames=60)
    c=tracking_context(_conditions(batch))
    if fault=='future':c['tracking_history_times'][:,-1]+=.02
    elif fault=='reference':c['tracking_reference_times'][:,10]+=.02
    elif fault=='actor':c['tracking_history'][:,-1,0]+=1
    elif fault=='padding':
        for k in ('proprio_history_valid','tracking_history_valid'):c[k][:,0]=False
        c['tracking_history'][:,0]=1
    else:c['future_reward']=torch.ones(2)
    with pytest.raises(ValueError):validate_tracking_context(c)
