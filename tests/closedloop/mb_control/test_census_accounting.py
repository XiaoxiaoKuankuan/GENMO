"""全量音乐评价统计和任务持久化的回归测试。

使用可解析的数值轨迹验证RMSE单位、提前失败分母、真实参考速度和队列恰好领取一次。
这些测试只验证统计与恢复工程，不代表真实机器人完成舞蹈；实际Newton结果另行保存。
"""
import json
import math
import sqlite3

import numpy as np
import pytest

from gem.closedloop.mb_control.census_metrics import aggregate, episode_statistics
from gem.closedloop.mb_control.stage1_census import claim, connect, finish


def fixture_row(complete=False):
    metrics={f'error/{k}':0. for k in ['anchor_pos','anchor_ori','body_pos','body_ori','body_lin_vel','foot_height','leg_joint_pos']}
    metrics.update(upper_eight_tracking=.8,root_xy_error_sq=0.,sm_six_reward_dt=.01,upper_eight_reward_dt=.016)
    row=dict(complete=complete,physical_seconds=5.,music_seconds=10.,physical_metrics_mean=metrics,
             status='physical_result',reason='music_end' if complete else 'physical_failure')
    state=np.zeros((3,55),dtype=np.float32);state[:,3]=1.;state[:,13:34]=.1;state[:,34:55]=2.
    arrays=dict(state=state,ref_q=np.zeros((3,21)),ref_qd=np.ones((3,21)),
                ref_root_R=np.broadcast_to(np.eye(3),(3,3,3)),ref_root_w=np.zeros((3,3)))
    names=[f'l_arm_{i}' for i in range(8)]+[f'l_leg_{i}' for i in range(13)]
    row['statistics']=episode_statistics(row,arrays,names)
    return row


def test_failed_tail_is_not_omitted_from_coverage_score():
    row=fixture_row()
    assert row['statistics']['tracking_score_100']==80
    assert row['statistics']['full_song_coverage_score_100']==40
    assert row['statistics']['duration_fraction']==.5


def test_original_reference_velocity_not_inferred_from_position_difference():
    s=fixture_row()['statistics']
    assert s['joint_rmse_rad']==pytest.approx(.1)
    assert s['joint_velocity_rmse_rad_s']==pytest.approx(1.)
    assert s['reference_joint_speed_rms_rad_s']==1.
    assert s['upper']['speed_ratio']==2.


def test_engineering_error_stays_visible_and_is_not_physical_success():
    r=aggregate([fixture_row(),fixture_row(True),dict(status='engineering_error',reason='invalid_asset')])
    assert r['physical_episodes']==2 and r['completed']==1
    assert r['completion_rate']==.5 and r['engineering_errors']==1
    assert r['results']==3
    assert r['executed_time_weighted_rmse']['joint_rmse_rad']==pytest.approx(.1)


def test_actual_step_weighting_uses_squared_error_before_square_root():
    a,b=fixture_row(),fixture_row();a['statistics']['joint_rmse_rad']=1.;b['statistics']['joint_rmse_rad']=3.
    a['statistics']['steps']=1;b['statistics']['steps']=3
    r=aggregate([a,b])
    assert r['executed_time_weighted_rmse']['joint_rmse_rad']==pytest.approx(math.sqrt(7.))


def test_queue_claim_retry_and_completion_never_duplicate_physics_result(tmp_path):
    task=dict(id='music',census_key='same_music_feature')
    with connect(tmp_path) as db:
        db.execute('CREATE TABLE jobs(key TEXT PRIMARY KEY, ordinal INTEGER, task TEXT,status TEXT,attempts INTEGER DEFAULT 0,owner INTEGER,started TEXT,result TEXT)')
        db.execute('INSERT INTO jobs(key,ordinal,task,status) VALUES(?,?,?,?)',(task['census_key'],0,json.dumps(task),'pending'))
    assert claim(tmp_path)==(task,1)
    assert claim(tmp_path) is None
    finish(tmp_path,task,dict(status='engineering_error'),retry=True)
    assert claim(tmp_path)==(task,2)
    finish(tmp_path,task,dict(status='physical_result',complete=False))
    assert claim(tmp_path) is None
