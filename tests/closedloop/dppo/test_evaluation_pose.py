"""评估 anchor 姿态取证的无副作用、真实侧车往返及旧证据兼容回归。

测试从既有控制步/真实奖励夹具取得 motion_tracking，再经过生产评估提取函数和
compact_evidence.write_episode/read_episode。覆盖实际列式只读行、跨25步分块、
float32/float64原值和四元数符号保留，验证仅增加姿态字段，原奖励、事件、终止、
决策及输入不变。无姿态旧侧车/旧内嵌JSON必须仍可读取且明确 unknown。

畸形shape、非有限数、非单位四元数、错误坐标或时间只产生不可用诊断，不增加
物理/评估终止；后处理需另行拒绝无效归因证据。本测试只在CPU和pytest临时目录
运行，不启动Isaac/GPU、不加载正式权重，不代表新物理评估或真实机器人验收。
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gem.closedloop.dppo.compact_evidence import read_episode, write_episode
from gem.closedloop.dppo.evaluation import _decision_evidence, _publish_json
from gem.closedloop.dppo.evaluation_pose import (
    FIELD, SCHEMA, extract_anchor_pose_evidence, read_anchor_pose_evidence,
)
from gem.closedloop.dppo.journal_codec import encode_binary
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.runtime.trajectory_blocks import ColumnarTrace, pack_trace
from tests.closedloop.dppo.test_data_learning import actual_step, target_activity_fixture


def _transition(count=3):
    rows = [actual_step(i, speed=.3) for i in range(count)]
    reward = ExecutionReward(dict(version='stage2.gmt8_reference_limits.v1',
        music_reward_enabled=False, music_weight=0.), target_activity=target_activity_fixture)
    details = [reward.evaluate_step(row) for row in rows]
    metadata = dict(latency_seconds=.2, reward_details=details,
        generated=dict(prefix_frames=0, joint_limit_diagnostics={}),
        event_penalty_total=-5., event_reward=0. if count else -5.,
        terminal_snapshot=dict(terminated=True), execution_trace=rows)
    values = [d['reward'] for d in details]
    if values:
        values[-1] -= 5.
    return SimpleNamespace(transition_valid=True, metadata=metadata, rewards=values,
        executed_control_steps=count, executed_physics_steps=count*4,
        terminated=True, truncated=False, reason='physical_failure')


def _without_pose(decision):
    decision = copy.deepcopy(decision)
    for row in decision['controls']:
        row.pop(FIELD, None)
        row.pop('feasibility', None)
    return decision


@pytest.mark.parametrize('columnar', [False, True])
def test_pose_uses_existing_anchor_and_preserves_all_original_decision_fields(columnar):
    result = _transition()
    legacy = copy.deepcopy(result)
    del legacy.metadata['execution_trace']
    baseline = _decision_evidence(legacy, 7)
    if columnar:
        result.metadata['execution_trace'] = ColumnarTrace([pack_trace(result.metadata['execution_trace'])])
    snapshot = lambda: encode_binary(dict(result.metadata,
        execution_trace=list(result.metadata['execution_trace'])))
    before = snapshot()
    decision = _decision_evidence(result, 7)
    assert snapshot() == before
    assert encode_binary(_without_pose(decision)) == encode_binary(_without_pose(baseline))
    for control, source in zip(decision['controls'], result.metadata['execution_trace']):
        pose = read_anchor_pose_evidence(control)
        assert pose['available'] is True and pose['schema'] == SCHEMA
        assert pose['control_tick'] == pose['reference_tick'] == control['tick']
        assert pose['anchor_body_name'] == 'base_link'
        assert pose['coordinate_frame'] == 'world' and pose['quaternion_order'] == 'wxyz'
        assert set(pose['actual']) == set(pose['reference']) == {'anchor_pos_w', 'anchor_quat_w'}
        for side in ('actual', 'reference'):
            for key in pose[side]:
                np.testing.assert_array_equal(pose[side][key], source['motion_tracking'][side][key])
                assert not np.shares_memory(pose[side][key], source['motion_tracking'][side][key])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_pose_is_not_normalized_reframed_or_sign_canonicalized(dtype):
    trace = actual_step()
    tracking = trace['motion_tracking']
    tracking['anchor_body_name'] = 'explicit_non_root_anchor'
    for side in ('actual', 'reference'):
        tracking[side]['anchor_pos_w'] = np.array([123., -45., .7], dtype=dtype)
        tracking[side]['anchor_quat_w'] = np.array([-1.00001, 0., 0., 0.], dtype=dtype)
    before = encode_binary(trace)
    pose = extract_anchor_pose_evidence(trace, trace['tick'])
    assert pose['available'] and pose['anchor_body_name'] == 'explicit_non_root_anchor'
    for side in ('actual', 'reference'):
        for key in pose[side]:
            assert pose[side][key].dtype == dtype
            np.testing.assert_array_equal(pose[side][key], tracking[side][key])
    pose['actual']['anchor_pos_w'][0] = -99.
    assert encode_binary(trace) == before


@pytest.mark.parametrize('damage', [
    'position_shape', 'quaternion_shape', 'nan', 'infinity', 'nonunit', 'object_dtype',
    'missing_actual', 'missing_vector', 'tracking_schema', 'coordinate_frame',
    'quaternion_order', 'body_name', 'control_tick', 'reference_tick', 'trace_tick',
    'trace_count', 'trace_type',
])
def test_invalid_pose_is_nonfatal_and_cannot_change_rewards_or_input(damage):
    result = _transition()
    baseline = _without_pose(_decision_evidence(result, 0))
    trace = result.metadata['execution_trace'][0]
    tracking = trace['motion_tracking']
    if damage == 'position_shape': tracking['actual']['anchor_pos_w'] = np.zeros(4)
    elif damage == 'quaternion_shape': tracking['reference']['anchor_quat_w'] = np.zeros((1, 4))
    elif damage == 'nan': tracking['actual']['anchor_pos_w'][1] = np.nan
    elif damage == 'infinity': tracking['reference']['anchor_quat_w'][2] = np.inf
    elif damage == 'nonunit': tracking['actual']['anchor_quat_w'] *= 2.
    elif damage == 'object_dtype': tracking['actual']['anchor_pos_w'] = np.array(['bad']*3)
    elif damage == 'missing_actual': del tracking['actual']
    elif damage == 'missing_vector': del tracking['reference']['anchor_quat_w']
    elif damage == 'tracking_schema': tracking['schema'] = 'unexpected'
    elif damage == 'coordinate_frame': tracking['coordinate_frame'] = 'local'
    elif damage == 'quaternion_order': tracking['quaternion_order'] = 'xyzw'
    elif damage == 'body_name': tracking['anchor_body_name'] = ''
    elif damage == 'control_tick': tracking['control_tick'] += 12
    elif damage == 'reference_tick': tracking['reference_tick'] += 12
    elif damage == 'trace_tick': trace['tick'] += 12
    elif damage == 'trace_count': result.metadata['execution_trace'].pop()
    elif damage == 'trace_type': result.metadata['execution_trace'] = {'invalid': 'mapping'}
    before = encode_binary(result.metadata)
    decision = _decision_evidence(result, 0)
    assert encode_binary(result.metadata) == before
    assert encode_binary(_without_pose(decision)) == encode_binary(baseline)
    pose = read_anchor_pose_evidence(decision['controls'][0])
    assert pose['available'] is False and pose['reason'].startswith('invalid_')
    assert pose['validation_error']


@pytest.mark.parametrize('missing', ['trace', 'tracking', 'pose'])
def test_missing_pose_is_explicit_unknown_without_zero_substitution(missing):
    result = _transition(1)
    if missing == 'trace': del result.metadata['execution_trace']
    elif missing == 'tracking': del result.metadata['execution_trace'][0]['motion_tracking']
    else:
        tracking = result.metadata['execution_trace'][0]['motion_tracking']
        del tracking['actual'], tracking['reference']
    pose = read_anchor_pose_evidence(_decision_evidence(result, 0)['controls'][0])
    assert pose['available'] is False and pose['reason'].endswith('_unknown')
    assert 'actual' not in pose and 'reference' not in pose


def test_real_compact_archive_roundtrip_preserves_pose_and_original_fields(tmp_path):
    decision = _decision_evidence(_transition(40), 0)
    # 混合有证据/unknown，覆盖列式异构回退以及25步分块边界。
    decision['controls'][25][FIELD] = extract_anchor_pose_evidence(None, decision['controls'][25]['tick'])
    episode = dict(task_id='pose-fixture', reward_sum=decision['reward_sum'], decisions=[decision])
    before = encode_binary(episode)
    path = tmp_path/'episode.json'
    write_episode(path, episode, publish_json=_publish_json)
    assert encode_binary(episode) == before
    restored = read_episode(path)
    assert encode_binary(restored['decisions']) == encode_binary(episode['decisions'])
    document = json.loads(path.read_text())
    assert 'controls' not in document['decisions'][0]
    assert document['control_archive']['control_count'] == 40
    assert read_anchor_pose_evidence(restored['decisions'][0]['controls'][25])['available'] is False
    archive = tmp_path/document['control_archive']['path']
    archive.write_bytes(archive.read_bytes()+b'corrupted_pose_archive')
    with pytest.raises(ValueError, match='SHA'):
        read_episode(path)


@pytest.mark.parametrize('compact', [False, True])
def test_old_sidecars_and_inline_episodes_remain_unchanged_and_read_unknown(tmp_path, compact):
    decision = _without_pose(_decision_evidence(_transition(1), 0))
    episode = dict(task_id='old', decisions=[decision])
    path = tmp_path/'old.json'
    if compact: write_episode(path, episode, publish_json=_publish_json)
    else: _publish_json(path, episode)
    restored = read_episode(path)
    assert restored['decisions'] == episode['decisions']
    control = restored['decisions'][0]['controls'][0]
    before = encode_binary(control)
    assert read_anchor_pose_evidence(control) == dict(schema=SCHEMA, available=False,
                                                     reason='not_recorded_unknown')
    assert FIELD not in control and encode_binary(control) == before


def test_zero_control_terminal_has_no_invented_pose_or_changed_event_reward():
    evidence = _decision_evidence(_transition(0), 0)
    assert evidence['controls'] == [] and evidence['reward_sum'] == -5.
    assert evidence['terminated'] is True and evidence['reason'] == 'physical_failure'
