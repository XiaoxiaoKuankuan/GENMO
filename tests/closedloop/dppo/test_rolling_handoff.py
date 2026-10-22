"""连续滚动参考耗尽与接管的回归，使用逐控制计数替身而不冒充PhysX验证。

将真实故障的bootstrap末尾1180/1164 tick及1.76秒前瞻越界还原成有限状态机。
所有等待必须继续执行旧Motion；准备拒绝、提交拒绝和延迟超预算都不能读出
旧参考范围。成功ACK必须立即更新参考范围，尤其是提交恰好发生在0.1秒格时。
同时验证故障证据只保存有界标量事件，测试产物全部位于pytest临时目录。
"""
import json
import pytest
from gem.closedloop.dppo.vector_environment import VectorUpperEnvironment, bounded_reference_deadline
from gem.closedloop.dppo.reference_lifecycle import record_reference_event, flush_reference_events
from tests.closedloop.dppo.test_env_adapter import adapter
from tests.closedloop.dppo.test_data_learning import actual_step


def rolling_env(tmp_path, *, elapsed=.26, prepared=True, reject=False, start=960, end=1180):
    env,backend=adapter(tmp_path,elapsed=elapsed,prepared=prepared)
    env.__class__=VectorUpperEnvironment
    env.rolling=True;env.decision_ticks=60
    env.config['runtime']={'generation_schedule':'continuous_rolling.v1'}
    backend.tick=start;backend.reject_commit=reject
    backend.reference_end=end
    snapshot=backend.snapshot
    def current():
        return dict(snapshot(),plan_id='old' if backend.reference_end==end else 'p',
            source_end_tick=backend.reference_end,reference_valid_end_tick=backend.reference_end-16)
    backend.snapshot=current;env.snapshot=current()
    # 从1.6秒开始的替身需补齐之前的真实格式活动度历史，不能靠关闭奖励检查通过。
    env.reward.evaluate_batch([actual_step(i) for i in range((start-600)//12)])
    call=backend.call
    def guarded(method,**payload):
        if method=='advance':
            # 保守核验整个控制区间都保有10步前瞻及右差分，不允许替身掩盖越界。
            assert backend.tick+12*payload['control_steps']+120<=backend.reference_end-16
        result=call(method,**payload)
        if method=='commit_plan':
            backend.reference_end=start+1180
            result['reference_state']={k:current()[k] for k in ('episode_id','env_id','plan_id',
                'source_end_tick','reference_valid_end_tick')}
        return result
    backend.call=guarded
    generate=env.generate
    def generation(**kwargs):
        value=generate(**kwargs)
        value['meta'].update(deadline_tick=env._request()['deadline_tick'],prefix_frames=6)
        return value
    env.generate=generation
    return env,backend


def test_rolling_wait_retains_lookahead_and_a_causal_common_grid():
    deadline,diagnostic=bounded_reference_deadline(dict(tick=960,source_end_tick=1180,
        reference_valid_end_tick=1164),1140,minimum_prefix=6,rolling=True)
    assert deadline==1020 and deadline+120<=1164
    assert diagnostic['waiting_lookahead_ticks']==120
    assert deadline%60==0


@pytest.mark.parametrize('prepared',[True,False])
def test_initial_reference_exhaustion_is_a_valid_truncation_not_an_underrun(tmp_path,prepared):
    env,backend=rolling_env(tmp_path,prepared=prepared)
    row=env.step()
    assert row.control_tick_end==1020 and row.executed_control_steps==5
    assert row.truncated and not row.terminated and row.reason=='reference_horizon_truncated'
    assert row.next_context is not None and not any(k=='commit_plan' for k,_ in backend.calls)
    assert backend.reference_end==1180
    evidence=json.loads(next(tmp_path.glob('reference_lifecycle_*.json')).read_text())
    assert evidence['reason']=='reference_horizon_truncated'
    assert any(e['event']=='generation_ready' for e in evidence['events'])


def test_success_at_next_grid_refreshes_authority_without_an_extra_physical_step(tmp_path):
    env,backend=rolling_env(tmp_path,elapsed=.10)
    row=env.step()
    assert row.control_tick_end==1020 and row.executed_control_steps==5
    assert not row.truncated and not row.terminated
    assert env.snapshot['source_end_tick']==2140 and env.snapshot['plan_id']=='p'
    assert env.snapshot['tick']==1020
    assert env._request()['deadline_tick']>1020


def test_commit_rejection_uses_old_motion_and_truncates_only_at_its_real_boundary(tmp_path):
    env,backend=rolling_env(tmp_path,elapsed=.10,reject=True)
    row=env.step()
    assert row.truncated and row.control_tick_end==1020
    assert row.metadata['rejection']['code']=='late_plan'
    assert row.executed_control_steps==5 and backend.reference_end==1180
    assert any(e['event']=='commit_rejected' for e in env._reference_events)


def test_valid_old_two_second_motion_is_not_discarded_after_point_three_seconds(tmp_path):
    env,backend=rolling_env(tmp_path,start=600,end=2380,reject=True,elapsed=.66)
    row=env.step()
    assert row.executed_control_steps==35 and row.control_tick_end==1020
    assert not row.truncated and not row.terminated and backend.reference_end==2380


def test_reference_events_are_bounded_and_do_not_store_observation_arrays(tmp_path):
    env,_=rolling_env(tmp_path)
    for i in range(100):record_reference_event(env,'test',sequence=i)
    flush_reference_events(env,'test')
    content=next(tmp_path.glob('reference_lifecycle_*.json')).read_text()
    events=json.loads(content)['events']
    assert len(events)==32 and events[0]['sequence']==68
    assert 'tracking_history_values' not in content
