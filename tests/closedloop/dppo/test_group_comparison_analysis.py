"""对三模型结果分析的统计和物理积分核对做独立小样本验证。

用不等长执行、独立零控制步罚分和交叉变化的tracking/活动门控，确认分解
可以重构真实总奖励差；同时验证按音乐组抽样不会把重复种子当作新音乐，
关节软硬边界采用实际关节×时间分母，以及MC偏差的方向约定。
"""
import numpy as np
import pytest

from tools.eval.analyze_stage10_group_comparison import (
    COMPONENTS, bootstrap_groups, boundary, boundary_cost, critic_statistics, paired_prefix,
)


def trajectory(score, gate, event=0.):
    n=len(score)
    a=dict(tick=np.arange(n)*12+612,activity_gate=np.array(gate),score_track=np.array(score),
        activity=np.ones(n),target_activity=np.ones(n),actual_joint_gmt=np.zeros((n,21)),
        reference_joint_gmt=np.zeros((n,21)),joint_velocity=np.zeros((n,21)),
        safe_limits=np.tile([[-.9,.9]],(1,21,1)),hard_limits=np.tile([[-1.,1.]],(1,21,1)),
        upper_event_reward=np.array([event]))
    for c in COMPONENTS:a['reward_'+c]=np.zeros(n)
    a['reward_track']=2.5*.02*a['score_track']*a['activity_gate']
    a['reward_alive']=np.full(n,.01)
    a['reward']=a['reward_track']+a['reward_alive']
    return a


def test_unequal_prefix_reward_decomposition_and_separate_failure_event():
    a=trajectory([.7,.9],[.8,.5],-4.)
    b=trajectory([.8,.6,.9],[.6,.9,.4],-2.)
    result=paired_prefix(a,b)
    expected=float(b['reward'].sum()-a['reward'].sum()+2)
    assert result['total_reward_delta']==pytest.approx(expected)
    assert result['tracking_score_contribution']+result['tracking_gate_contribution']==pytest.approx(
        b['reward_track'][:2].sum()-a['reward_track'].sum())
    assert result['unequal_tail_reward_delta']==pytest.approx(b['reward'][-1])
    assert result['zero_control_event_delta']==2.
    b['target_activity'][0]=2.
    with pytest.raises(ValueError,match='target activity'):paired_prefix(a,b)


def test_group_bootstrap_keeps_seed_replicas_inside_music_group():
    rows=[dict(dataset=s,group_id=f'{s}{i}',delta=float(i)) for s in ('a','b') for i in range(3)]
    original=bootstrap_groups(rows,'delta',draws=200)
    duplicated=bootstrap_groups(rows*4,'delta',draws=200)
    assert original==duplicated
    assert original['mean']==1.
    assert original['groups_by_source']=={'a':3,'b':3}


def test_joint_boundary_and_mc_bias_units():
    result=boundary(np.array([[-1.,.4],[-1.00001,1.2]]),np.array([[-1.,1.],[-1.,1.]]),1e-6)
    assert result.tolist()==[[False,False],[True,True]]
    assert result.mean()==.5
    stats=critic_statistics([3.,5.],[1.,3.])
    assert stats['bias']==2.
    assert stats['mse']==4.
    assert stats['explained_variance']==1.


def test_zero_width_soft_margin_counts_touch_without_claiming_hard_excess():
    hard=np.array([[-1.,1.],[0.,2.]])
    q=np.array([[0.,0.],[1.,1.]])
    assert boundary_cost(q,hard,hard).tolist()==[[0.,1.],[1.,0.]]
    assert not boundary(q,hard,1e-6).any()
