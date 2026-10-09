"""共享GPU场景中GAE环境身份隔离的回归测试，仅在服务器1执行。

故意令相邻两条记录共享session、episode、策略版本和连续时钟，只改变env_id；
检查第二条奖励不能传播回第一条优势。该反例能发现仅依赖旧单环境身份的错误，
同时对照同环境连续区间仍然传播GAE，不通过把所有记录切断来规避问题。
"""
from types import SimpleNamespace as NS
import torch
from gem.closedloop.dppo.trainer import fixed_targets


def test_shared_session_requires_same_environment_for_gae():
    def row(env,begin,reward):
        return NS(identity=dict(env_id=env,backend_session_id='shared',episode_id='episode1',policy_version=0),
            metadata=dict(value_snapshot_version=0),old_value=0.,next_value=0.,rewards=torch.tensor([reward]*25),
            executed_control_steps=25,control_tick_begin=begin,control_tick_end=begin+300,
            transition_valid=True,terminated=False,truncated=False,next_context={})
    a,b=row(0,0,0.),row(1,300,1.)
    separated=fixed_targets([a,b],None,'cpu',reuse_values=True,critic_version=0,normalize=False)
    assert separated['advantages'][0]==0
    b.identity['env_id']=0
    joined=fixed_targets([a,b],None,'cpu',reuse_values=True,critic_version=0,normalize=False)
    assert joined['advantages'][0]>0
