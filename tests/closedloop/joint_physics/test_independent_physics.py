"""独立物理请求与认证的关键边界测试。

覆盖静态H支持、课程未激活与真实学生有效率门槛、不同H类型认证隔离、无歌曲白名单放行、
以及动作不合法时必须在WM调用前拒绝。这里使用小型显式张量验证决策逻辑，不冒充Newton
真实方向或学生训练成功；真实资产与完整恢复由有界运行另行记录。
"""
import pytest
import torch

from gem.closedloop.joint_physics.config import ROOT, read_config
from gem.closedloop.joint_physics.contracts import PhysicsRejected
from gem.closedloop.joint_physics.curriculum import Readiness
from gem.closedloop.joint_physics.requests import static_request
from gem.closedloop.joint_physics.execution import ValidatedSupport


def test_static_long_halo_and_real_support():
    episode = {'source_frames':300,'state':torch.zeros(500,55),
               'offline_requests':{'frames':torch.tensor([0]),'prefix_frames':torch.tensor([24]),
                                   'valid':torch.tensor([True]),'forecast_ticks':torch.tensor([348])}}
    assert static_request(episode,0,32)[1] is None
    assert static_request(episode,0,64)[1]=='candidate_right_halo_insufficient'
    episode['offline_requests']['forecast_ticks'][0]=240
    assert static_request(episode,0,64)[1] is None
    episode['state']=torch.zeros(30,55)
    assert static_request(episode,0,64)[1]=='incomplete_real_window'


def test_readiness_requires_step_rate_sources_and_restores():
    cfg = read_config(ROOT/'configs/closedloop/joint_physics_from_scratch.yaml')['physics']
    state = Readiness(cfg)
    rows = [{'reason':'execution_gradient_completed','source':str(i%3),'tags':['standing','arm_swing','double_support']} for i in range(16)]
    state.observe(rows,999)
    assert state.weight(999)==0
    state.observe(rows,1000)
    assert 0 < state.weight(1000) < cfg['lambda_exec']
    restored = Readiness(cfg)
    restored.load_state_dict(state.state_dict())
    assert restored.weight(1001)==state.weight(1001)
    failed = Readiness(cfg)
    failed.observe([dict(r,reason='wm_state_out_of_domain') for r in rows],5000)
    assert failed.weight(5000)==0


def test_type_certification_and_action_boundary():
    guard = object.__new__(ValidatedSupport)
    guard.certificates={'standing':{'passed':True,'audited_timesteps':[100]},'landing':{'passed':False}}
    assert guard.certified(['standing'])
    assert not guard.certified(['standing','landing'])
    assert guard.timesteps_for(['standing'])==[100]
    guard.profiles={'double_support|standing':{'passed':True,'audited_timesteps':[100]}}
    assert guard.certified(['standing','double_support'])
    assert not guard.certified(['standing','single_support'])
    assert not guard.certified(['standing','double_support','turning'])
    assert guard.timesteps_for(['standing','double_support'])==[100]
    guard.lower,guard.upper=torch.full((21,),-1.),torch.ones(21)
    with pytest.raises(PhysicsRejected,match='action_out_of_validated_domain') as error:
        guard.check_action(torch.full((1,21),2.))
    assert len(error.value.details['components'])==21
    with pytest.raises(PhysicsRejected,match='nonfinite_sm_action'):
        guard.check_action(torch.full((1,21),float('nan')))
