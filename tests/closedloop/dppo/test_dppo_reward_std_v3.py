"""DPPO新奖励尺度的标量、GPU数学和列式消费一致性测试。

使用冻结GMT八项原始平方误差，核验source_std保持权威原值，reward_std独立用于
exp(-error/std²)，并验证活动度很低时新v3奖励也不乘门控。旧v1仍用于历史
复现，三种数值路径必须得到相同积分，不能重复乘dt或改变真实终止事件。
"""
import copy
import math
import numpy as np
import pytest
from gem.closedloop.dppo.rewards import ExecutionReward,TRACKING_STD,STAGE2_REWARD_STD,REWARD_STD_VERSION,ROOT_REWARD_STD,ROOT_REWARD_STD_VERSION
from gem.closedloop.dppo.vector_reward_math import VectorRewardMath
from gem.closedloop.dppo.vector_reward_adapter import VectorExecutionReward,compare_reward_evidence
from gem.runtime.trajectory_blocks import pack_trace,ColumnarTrace
from tests.closedloop.dppo.test_stage2_reward import device_packet
from test_data_learning import actual_step,target_activity_fixture


@pytest.mark.parametrize('version,std',[(REWARD_STD_VERSION,STAGE2_REWARD_STD),(ROOT_REWARD_STD_VERSION,ROOT_REWARD_STD)])
def test_new_reward_std_preserves_source_and_scalar_vector_formula(version,std):
    scalar=ExecutionReward({'version':version},target_activity=target_activity_fixture)
    vector=VectorExecutionReward({'version':version},target_activity=target_activity_fixture)
    engine=VectorRewardMath(scalar.config,1,'cpu')
    import numpy as np
    scalar.seed_previous_target(np.zeros(21));vector.seed_previous_target(np.zeros(21));engine.previous_valid.fill_(True)
    row=actual_step(0,speed=.01);row['env_id']=0
    for name,item in row['motion_tracking']['terms'].items():
        item['error']=.04
        item['source_score']=math.exp(-.04/TRACKING_STD[name]**2)
    for substep in row['physics_substeps']:
        d=substep['physical_diagnostics']
        for field in ('foot_support_sphere_tangent_speed_m_s','foot_support_sphere_clearance_m'):
            import numpy as np
            d[field]={n:np.zeros(1) for n in d['foot_body_names']}
    row['physical_diagnostics']=copy.deepcopy(row['physics_substeps'][-1]['physical_diagnostics'])
    row['reward_primitives']=device_packet(row,engine)
    expected=scalar.evaluate_step(row);observed=vector.evaluate_batch(ColumnarTrace([pack_trace([row])]))[0]
    assert expected['transition_valid'],expected['errors']
    compare_reward_evidence(observed,expected)
    assert expected['activity']['gate'] < 1.
    assert expected['components']['track']['activity_gate'] == 1.
    scores=expected['components']['track']['normalized']['scores']
    assert scores['anchor_pos']==pytest.approx(math.exp(-.04/std['anchor_pos']**2))
    assert scores['body_pos']==pytest.approx(math.exp(-.04/.15**2))
    assert scalar.config['tracking']['source_std']==TRACKING_STD
    assert scalar.config['tracking']['reward_std']==std

def test_reward_versions_reject_ambiguous_or_changed_source_std():
    assert ExecutionReward().config['tracking']['std']==TRACKING_STD
    with pytest.raises(ValueError):ExecutionReward({'tracking':{'std':STAGE2_REWARD_STD}})
    with pytest.raises(ValueError):ExecutionReward({'version':REWARD_STD_VERSION,'tracking':{'std':TRACKING_STD}})
    bad=copy.deepcopy(TRACKING_STD);bad['anchor_pos']=.2
    with pytest.raises(ValueError):ExecutionReward({'version':REWARD_STD_VERSION,'tracking':{'source_std':bad}})
