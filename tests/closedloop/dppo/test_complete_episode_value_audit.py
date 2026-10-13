"""完整物理 episode 价值诊断的独立回报和边界回归。

用可手算的变长50Hz控制序列检查独立Monte Carlo回报、实际时间折扣和零步终止
事件。行政截断、跨episode拼接、缺少物理子步、控制tick缺口与重复条件身份必须
拒绝。这些CPU测试只核验新诊断逻辑，不能替代实际八卡完整音乐执行和Critic质量。
"""
from __future__ import annotations

import copy
import pytest

from tools.audit_stage10_v3_complete_returns import complete_returns, identity_key, validate_value_task


@pytest.mark.parametrize('changed', [None, 'environment', 'termination', 'timing'])
def test_value_audit_compares_physical_task_but_allows_training_horizon(changed):
    model = dict(environment=dict(gravity=-9.81), termination=dict(yaw=1.5),
                 timing=dict(prefix=.3), stage10=dict(episode_seconds=10.))
    evaluation = copy.deepcopy(model)
    evaluation['stage10']['episode_seconds'] = 180.
    if changed:
        evaluation[changed][next(iter(evaluation[changed]))] *= 2
        with pytest.raises(ValueError, match='Value task'):
            validate_value_task(model, evaluation)
    else:
        validate_value_task(model, evaluation)


def example():
    decisions = []
    tick = 600
    for index, rewards in enumerate(([1., 2.], [3.], [])):
        controls = []
        for reward in rewards:
            tick += 12
            controls.append(dict(tick=tick, dt_s=.02, reward=reward))
        event = -5. if index == 2 else 0.
        decisions.append(dict(identity=dict(backend_session_id='s', env_id=0, episode_id=1,
            policy_version=0, decision_id=index), controls=controls, executed_control_steps=len(rewards),
            executed_physics_steps=4*len(rewards), reward_sum=sum(rewards)+event,
            zero_step_event_reward=event, terminated=index == 2, truncated=False))
    return dict(status='passed', task_count=1, episodes=[dict(task_id='t', dataset='Mine',
        transition_valid=True, truncated=False, physical_failure=True, end_reason='fall', decisions=decisions)])


def test_actual_control_discount_and_zero_step_terminal_event():
    report = example()
    targets, episodes = complete_returns(report, .9)
    gamma = .9**(1/25)
    rows = report['episodes'][0]['decisions']
    assert targets[identity_key(rows[2]['identity'])]['target'] == -5.
    assert targets[identity_key(rows[1]['identity'])]['target'] == pytest.approx(3.-5.*gamma)
    assert targets[identity_key(rows[0]['identity'])]['target'] == pytest.approx(1.+2.*gamma+3.*gamma**2-5.*gamma**3)
    assert episodes[0]['control_steps'] == 3


@pytest.mark.parametrize('gamma', [.8, .99, 1.])
def test_complete_returns_match_lambda_one_gae_independent_of_initial_value(gamma):
    from gem.closedloop.dppo.returns import compute_gae
    report = example()
    targets, _ = complete_returns(report, gamma)
    # 相邻旧价值任意且彼此差异很大；完整轨迹lambda=1应抵消所有内部自举项。
    actual = compute_gae([[1., 2.], [3.], []], [100., -7., 23.], [-7., 23., 999.],
        [2, 1, 0], [True, True, False], [True, True, False],
        gamma_upper=gamma, lambda_upper=1., event_rewards=[0., 0., -5.], normalize=False)
    expected = [targets[identity_key(r['identity'])]['target'] for r in report['episodes'][0]['decisions']]
    assert actual['returns'].tolist() == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('corruption', ['truncated', 'early_end', 'tick', 'physics', 'identity', 'duplicate', 'event'])
def test_incomplete_or_inconsistent_episodes_are_rejected(corruption):
    report = copy.deepcopy(example())
    episode = report['episodes'][0]
    rows = episode['decisions']
    if corruption == 'truncated': episode['truncated'] = True
    elif corruption == 'early_end': rows[0]['terminated'] = True
    elif corruption == 'tick': rows[1]['controls'][0]['tick'] += 12
    elif corruption == 'physics': rows[0]['executed_physics_steps'] -= 1
    elif corruption == 'identity': rows[1]['identity']['episode_id'] += 1
    elif corruption == 'duplicate': rows[1]['identity'] = dict(rows[0]['identity'])
    elif corruption == 'event': rows[2]['reward_sum'] = 0.
    with pytest.raises(ValueError): complete_returns(report, .99)
