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
