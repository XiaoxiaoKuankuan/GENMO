"""完整MC局部价值诊断的独立可计算验收。

用每半秒一段的真实控制格式合成完整episode，比较精确价值、固定高估、真正失败与
正常歌曲结束。验证八秒普通截断仍自举、最终终止不自举、事件奖励不丢失，以及
不完整轨迹/错标签明确报错。这里不启动物理世界，不以测试样本冒充训练证据。
"""
from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
from gem.closedloop.dppo.complete_episode import complete_mc_returns
from gem.closedloop.dppo.value_credit_diagnostics import episode_value_records, summarize_value_credit


def episode(reason='music_end',gamma=1.):
    rows=[]
    for i in range(20):
        rows.append(SimpleNamespace(identity=dict(backend_session_id='s',env_id=0,episode_id='e',policy_version=6),
            rewards=torch.full((25,),.02,dtype=torch.float64),executed_control_steps=25,
            control_tick_begin=i*300,control_tick_end=(i+1)*300,terminated=i==19,truncated=False,
            metadata={'event_reward':-5. if i==19 and reason!='music_end' else 0.}))
    return dict(rows=rows,returns=complete_mc_returns(rows,gamma_upper=gamma),
        task=dict(task_id='task',dataset='Mine'),end_reason=reason)


def test_exact_MC_value_remains_exact_on_normal_cuts_and_terminal():
    data=episode()
    records=episode_value_records(data,data['returns'],gamma_upper=1.,lambda_upper=.98)
    assert max(abs(r['value_error']) for r in records)==0.
    assert max(abs(r['gae_error']) for r in records)<1e-12
    assert max(abs(r['bootstrap_error']) for r in records)<1e-12
    assert any(r['before_reconstructed_cut'] for r in records)
    assert any(r['after_reconstructed_cut'] for r in records)
    assert summarize_value_credit(records)['physical_failure_tasks']==0


def test_optimism_contaminates_cut_bootstrap_but_not_true_terminal():
    data=episode('non_yaw_anchor_ori')
    records=episode_value_records(data,data['returns']+10,gamma_upper=1.)
    assert records[15]['bootstrap_error']==pytest.approx(10.)
    assert records[16]['bootstrap_error']==pytest.approx(0.,abs=1e-12)
    assert records[-1]['bootstrap_error']==pytest.approx(0.,abs=1e-12)
    summary=summarize_value_credit(records)
    assert summary['physical_failure_tasks']==1
    assert summary['groups']['near_physical_failure']['value_error']['bias']==pytest.approx(10.)
    assert summary['worst_overestimates'][0]['identity']['policy_version']==6


def test_real_discount_and_failure_event_match_the_production_MC():
    data=episode('fallen',gamma=.99)
    records=episode_value_records(data,data['returns'],gamma_upper=.99)
    assert max(abs(r['bootstrap_error']) for r in records)<1e-12
    assert records[-1]['mc']<0


def test_incomplete_episode_and_modified_targets_are_rejected():
    data=episode();data['rows'][-1].terminated=False
    with pytest.raises(ValueError,match='incomplete episode'):
        episode_value_records(data,data['returns'])
    data=episode();data['returns'][0]+=1
    with pytest.raises(ValueError,match='Stored MC differs'):
        episode_value_records(data,data['returns'],gamma_upper=1.)


def test_nonfinite_and_misaligned_predictions_are_rejected():
    for values in ([0.]*19,[float('nan')]*20):
        with pytest.raises(ValueError,match='Finite aligned'):
            episode_value_records(episode(),values,gamma_upper=1.)
