"""教师奖励的纯张量回归测试：稳健损失、Critic目标输入、冻结输入导数及保存合同。

本文件只验证数学/数据合同，不用张量夹具冒充原Stage1资产、Newton执行或训练收益。
真实教师参数同一性、DDIM梯度和物理方向由服务器专门审计入口验证。
"""
from dataclasses import replace
import pytest
import torch

from gem.closedloop.mb_control.dance_losses import robust
from gem.closedloop.mb_control.critic import PlanAwareCritic
from gem.closedloop.mb_control.scheduler import Scheduler
from gem.closedloop.mb_control.trainer import frozen_parameters
from gem.closedloop.mb_control.checkpoint import pack_state


def test_large_errors_keep_nonzero_robust_gradient():
    x=torch.tensor([0.,.1,1.,10.,100.],requires_grad=True)
    y=robust(x,.3)
    y.sum().backward()
    assert y[0]==0 and torch.all(y[1:]>y[:-1])
    assert torch.all(x.grad[1:]>0) and x.grad[-1]>3.


def test_teacher_critic_requires_target_and_keeps_state_derivatives(tensor_state):
    critic=PlanAwareCritic(Scheduler(120),width=32,layers=1,heads=4,teacher_conditioned=True)
    with pytest.raises(ValueError,match="固定音乐目标"):
        critic(tensor_state)
    s=tensor_state.state.clone().requires_grad_(True)
    q=tensor_state.reference.world_qpos.clone().requires_grad_(True)
    z=replace(tensor_state,state=s,reference=replace(tensor_state.reference,world_qpos=q),
        teacher_reference=tensor_state.reference,task_id="fixed_music",task_source="AIST++")
    with frozen_parameters(critic):
        critic(z).sum().backward()
    assert s.grad.abs().sum()>0 and q.grad.abs().sum()>0
    assert all(p.grad is None for p in critic.parameters())
    payload=pack_state(z)
    assert payload["task_id"]=="fixed_music" and payload["task_source"]=="AIST++"
    assert torch.equal(payload["teacher_reference"]["world_qpos"],z.teacher_reference.world_qpos)


def test_old_critic_not_shape_compatible_with_teacher_critic():
    old=PlanAwareCritic(Scheduler(120),width=32,layers=1,heads=4)
    new=PlanAwareCritic(Scheduler(120),width=32,layers=1,heads=4,teacher_conditioned=True)
    with pytest.raises(RuntimeError):
        new.load_state_dict(old.state_dict(),strict=True)


def test_anchor_alignment_fixed_once_and_teacher_detached(tensor_state):
    from gem.closedloop.mb_control.dance_losses import AnchorAlignment
    from supermimic.motion import ReferenceBatch
    p=tensor_state.state[:,:3];rotation=torch.eye(3)[None];v=torch.zeros_like(p);q=torch.zeros(1,21)
    target=ReferenceBatch(p,rotation,v,v,q,q,p[:,None],rotation[:,None],v[:,None],v[:,None],v,v,torch.ones(1))
    state=tensor_state.state.clone().requires_grad_(True)
    with torch.no_grad():state[:,:2]+=state.new_tensor([1.,2.])
    alignment=AnchorAlignment.at_anchor(state,target)
    assert not alignment.rotation.requires_grad and not alignment.translation.requires_grad
    first=alignment.reference(target)
    future=target._replace(root_p=target.root_p+target.root_p.new_tensor([.2,.3,0.]))
    moved=alignment.reference(future)
    assert torch.allclose(moved.root_p-first.root_p,future.root_p-target.root_p,atol=1e-6)
    assert torch.allclose(alignment.reference(target).root_p,first.root_p)
    # 参考未改变但机器人已落后时，参考保持不能继承机器人的累计位置误差。
    preservation=AnchorAlignment.at_reference_anchor(target,target)
    assert torch.allclose(preservation.reference(future).root_p,future.root_p)
    assert not torch.allclose(alignment.reference(future).root_p,preservation.reference(future).root_p)
    assert not preservation.rotation.requires_grad and not preservation.translation.requires_grad


def test_critic_direct_mutable_reference_gradient_is_auditable(tensor_state):
    critic=PlanAwareCritic(Scheduler(120),width=32,layers=1,heads=4,teacher_conditioned=True)
    z=replace(tensor_state,teacher_reference=tensor_state.reference,protected_until=120)
    with frozen_parameters(critic):report=critic.audit_mutable_reference_gradient(z)
    assert report['finite'] and report['mutable_reference_gradient_norm']>0
    assert report['protected_reference_gradient_max_abs']==0


def test_delivery_replay_preserves_observed_commit_and_censors_last_request():
    from gem.closedloop.mb_control.online_teacher import recorded_delivery_events
    payload={"events":[{"request_tick":0,"actual_ready_tick":93,"commit_tick":108},
        {"request_tick":60,"actual_ready_tick":111,"commit_tick":None,"rejection_tick":120}],
        "ticks":torch.tensor([0,12,120]),"controls":{"data":[{"data":{"start":{"data":{
            "pending_request":{"data":{"ticket":{"data":{"request_tick":120}}}}}}}}]}}
    events=recorded_delivery_events(payload)
    assert events["0"]["actual_ready_tick"]==93 and events["0"]["commit_tick"]==108
    assert events["60"]["delivery_tick"]==120 and events["60"]["commit_tick"] is None
    assert events["120"]=={"delivered":False,"censored_at_tick":120}
