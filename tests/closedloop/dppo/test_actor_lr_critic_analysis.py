"""离线已预热Critic诊断的数据对齐回归。

用故意错误的旧预测和GAE标签证明MC仍只来自真实积分奖励；同时拒绝缺失、重复、
策略版本不一致及行政截断样本。合成记录只验证分析器，不能替代真实价值预测验收。
"""
import copy
import pytest

from tools.eval.analyze_stage2_actor_lr_critic import join_causal_records


def fixture():
    identity = dict(backend_session_id='world', episode_id='episode', decision_id=0, policy_version=0)
    episode = dict(task_id='song_seed', dataset='music', truncated=False, transition_valid=True,
        decisions=[dict(identity=identity, decision_index=0, terminated=True,
                        controls=[dict(reward=1.), dict(reward=2.)],
                        executed_control_steps=2, zero_step_event_reward=-5.)])
    record = dict(identity=identity, control_tick_begin=600, control_tick_end=624,
                  target=99999., initial_prediction=-99999.)
    return episode, record


def test_posthoc_critic_uses_complete_real_reward_only():
    episode, record = fixture()
    result = join_causal_records([episode], [record], gamma_upper=.99)
    assert result[0]['target'] == pytest.approx(1. + .99 ** (1 / 25) * (2. - 5.))
    assert 'prediction' not in result[0]
    assert result[0]['record'] is record


def test_posthoc_critic_rejects_wrong_or_incomplete_states():
    episode, record = fixture()
    with pytest.raises(ValueError, match='Missing'):
        join_causal_records([episode], [], gamma_upper=.99)
    with pytest.raises(ValueError, match='Duplicate'):
        join_causal_records([episode], [record, record], gamma_upper=.99)
    wrong = copy.deepcopy(record)
    wrong['identity']['policy_version'] = 1
    with pytest.raises(ValueError, match='identity differs'):
        join_causal_records([episode], [wrong], gamma_upper=.99)
    wrong = copy.deepcopy(record)
    wrong['control_tick_end'] += 12
    with pytest.raises(ValueError, match='interval differs'):
        join_causal_records([episode], [wrong], gamma_upper=.99)
    episode['truncated'] = True
    with pytest.raises(ValueError, match='complete nonadministrative'):
        join_causal_records([episode], [record], gamma_upper=.99)
