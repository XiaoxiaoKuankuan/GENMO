"""episode初态累计坐标合同的纯数值单元测试。

只检查固定变换的保持、梯度边界及积分尺度，不把合成张量测试当作Newton物理证据。
真实模型效果另由独立固定8曲配对评价给出；本文件不启动训练、不改资产或旧检查点。
"""
from dataclasses import replace
from types import SimpleNamespace
import torch
import pytest
from gem.closedloop.mb_control.contracts import ClosedLoopState
from gem.closedloop.mb_control.dance_losses import AnchorAlignment
from gem.closedloop.mb_control.episode_objective import with_episode_alignment,episode_alignment,execution_mass


def state():
    x=torch.zeros(1,55);x[:,3]=1
    return ClosedLoopState(0,x,torch.zeros(1,25),torch.zeros(1,21),torch.zeros(1,10,75),
        torch.zeros(1,50,48),torch.ones(1,50,dtype=torch.bool),None,torch.zeros(1,60,35),0,1200)


def test_episode_transform_does_not_follow_later_anchor():
    r=torch.eye(3)[None].requires_grad_();t=torch.tensor([[1.,2.,0.]],requires_grad=True)
    z=with_episode_alignment(state(),AnchorAlignment(r,t))
    later=replace(z,tick=600,state=z.state+3.)
    assert torch.equal(episode_alignment(later,True).translation,torch.tensor([[1.,2.,0.]]))
    assert not later.episode_teacher_translation.requires_grad
    assert not later.episode_teacher_rotation.requires_grad


def test_missing_episode_initial_state_cannot_be_silently_reanchored():
    with pytest.raises(ValueError,match='禁止按当前anchor'):episode_alignment(state(),True)
    assert episode_alignment(state(),False) is None


def test_horizon_scale_changes_only_when_explicitly_requested():
    engine=SimpleNamespace(horizon_steps=16,gamma=.99)
    assert execution_mass(engine,{})==1.
    assert execution_mass(engine,{'normalize_execution_horizon':True})==pytest.approx(.29708445781024874)


def test_local_alignment_can_hide_drift_but_episode_loss_has_correct_gradient():
    from supermimic.motion import ReferenceBatch
    from gem.closedloop.mb_control.dance_losses import execution_loss
    class TensorKinematics:
        def forward(self,p,r,v,w,q,qd,**kwargs):return p[:,None],r[:,None],v[:,None],w[:,None]
    z=state();p=torch.zeros(1,3,requires_grad=True);r=torch.eye(3)[None]
    v=torch.zeros(1,3);q=torch.zeros(1,21)
    target=ReferenceBatch(p,r,v,v,q,q,p[:,None],r[:,None],v[:,None],v[:,None],v,v,torch.ones(1))
    initial=AnchorAlignment.at_anchor(z.state,target)
    moved=z.state.clone();moved[:,0]=1.;moved.requires_grad_()
    local=AnchorAlignment.at_anchor(moved,target)
    local_loss,_=execution_loss(moved,local.reference(target),TensorKinematics())
    episode_loss,_=execution_loss(moved,initial.reference(target),TensorKinematics())
    assert local_loss.item()==0 and episode_loss.item()>0
    episode_loss.sum().backward()
    assert moved.grad[0,0]>0  # 梯度下降确实把累计位置偏移拉向原episode目标。
    assert p.grad is None  # 固定音乐教师不能被学生损失更新。
