"""验证优势只读拆解的真实时间折扣、终止事件和Actor版本边界。

合成小轨迹只用于证明重建公式与线性拆解正确，不当作训练或物理效果证据；特别
确保已积分控制奖励不重复乘dt，跨Actor版本不递推GAE，终止状态不使用next value。
"""
from types import SimpleNamespace
import pytest
import torch
from tools.eval.analyze_stage2_actor_lr_advantages import decompose


def row(index,*,version=0,terminal=False):
    return SimpleNamespace(identity=dict(backend_session_id='one',episode_id='e',policy_version=version,env_id=0),
        transition_valid=True,terminated=terminal,truncated=False,control_tick_begin=12*index,
        control_tick_end=12*(index+1),executed_control_steps=1,rewards=[2.],old_value=3.,next_value=4.,
        next_context={'present':True},metadata={'event_reward':-5. if terminal else 0.})


def test_linear_decomposition_and_no_gae_across_actor_versions():
    rows=[row(0),row(1,version=1,terminal=True)]
    values=decompose(rows);gamma=.99**(1/25)
    assert values['advantages_raw'][0]==pytest.approx(2.+gamma*4.-3.)
    assert values['advantages_raw'][1]==pytest.approx(2.-5.-3.)
    torch.testing.assert_close(values['advantages_raw'],values['gae_reward_component']+values['gae_value_component'])
    rows[1].identity['policy_version']=0
    connected=decompose(rows)
    assert connected['advantages_raw'][0]==pytest.approx(values['advantages_raw'][0]+gamma*.95**(1/25)*values['advantages_raw'][1])
