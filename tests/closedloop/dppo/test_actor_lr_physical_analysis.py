"""物理分析中的真实MC标签、终止事件和权威控制记录缺失回归。

测试刻意让存储GAE标签与真实回报不同，证明分析没有偷用旧targets；另以极小
二进制journal验证每个计奖控制必须对应四个真实子步，缺失不能静默忽略。
这些合成样本只验分析正确性，不提供机器人或学习效果证据。
"""
import json
import sqlite3
import numpy as np
import pytest
import torch
from gem.closedloop.dppo.journal_codec import encode_binary
from gem.closedloop.dppo.training_checkpoint import file_sha256
from tools.eval.analyze_stage2_actor_lr_physics import mc_predictions,physical_controls


def test_mc_uses_integrated_rewards_and_event_not_saved_gae(tmp_path):
    identity=dict(backend_session_id='world',episode_id='episode',decision_id=0)
    snapshot=tmp_path/'critic.pt'
    torch.save(dict(records=[dict(identity=identity,initial_prediction=3.,target=99999.)]),snapshot)
    (tmp_path/'fixed_diagnostic_reference.json').write_text(json.dumps(dict(ranks=[dict(
        critic=dict(path=str(snapshot),sha256=file_sha256(snapshot)))])))
    episode=dict(task_id='one',dataset='source',truncated=False,decisions=[dict(
        identity=identity,decision_index=0,terminated=True,controls=[{'reward':1.},{'reward':2.}],
        zero_step_event_reward=-5.)])
    result=mc_predictions(tmp_path/'report.json',[episode])
    expected=1.+.99**(1/25)*(2.-5.)
    assert result['rows'][0]['mc_return']==pytest.approx(expected)
    assert result['overall']['mse']==pytest.approx((3.-expected)**2)
    episode['truncated']=True
    assert mc_predictions(tmp_path/'report.json',[episode])['excluded_administrative_or_incomplete_tasks']==['one']


def test_physics_requires_every_counted_control_and_four_substeps(tmp_path):
    import hashlib
    path=tmp_path/'world.sqlite'
    row=dict(backend_session_id='world',episode_id='episode',tick=12,
             completed_physics_steps=4,state_valid=True,errors={'yaw_error_rad':.2})
    envelope=dict(backend_session_id='world',mutation_seq=1,operation='advance',
                  result=dict(episode_id='episode',trace=[row]))
    payload=encode_binary(dict(result=dict(replies=[dict(result=envelope)])))
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE replies(sha256 TEXT,payload BLOB)')
        connection.execute('INSERT INTO replies VALUES(?,?)',(hashlib.sha256(payload).hexdigest(),payload))
    episode=dict(task_id='one',dataset='source',sample_id='sample',decisions=[dict(
        identity=dict(backend_session_id='world',episode_id='episode'),
        authoritative_journal_path=str(path),controls=[dict(tick=12)])])
    result=physical_controls([episode],tmp_path)
    assert result['physics_substeps']==4 and result['overall']['mean']==.2
    episode['decisions'][0]['controls'].append(dict(tick=24))
    with pytest.raises(ValueError,match='Missing actual physical controls'):
        physical_controls([episode],tmp_path)
