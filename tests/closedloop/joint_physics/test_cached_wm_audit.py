"""离线WM精度评价的时间合同与多步回放测试，不把合成夹具当真实物理证据。

构造连续状态和逐步变化的动作，验证末动作缺少下一状态时明确排除、动作错位及reset
拼接被拒绝、多步回放递归使用预测状态和实际每步动作、不同H使用相同合法锚点。
另外验证按帧加权与按来源等权的汇总不同，非有限结果保留失败计数。
"""

from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.joint_physics.cached_wm_audit import (
    aggregate, evaluate_episode, matched_starts, pack_errors, paired_transitions,
)


def episode(n=41):
    action = torch.zeros(n-1, 21)
    action[:, 0] = torch.arange(1, n)*.001
    state = torch.zeros(n, 55)
    state[:, 3] = 1
    state[1:, 0] = action[:, 0].cumsum(0)
    previous = torch.cat((torch.zeros(1, 21), action))
    return dict(state=state, action=action, previous_action=previous, xi=torch.zeros(25),
                reset_count=1, state_origin='continuous_newton_observation_not_data_reference',
                frame_origin_tick=0, control_steps=n-1, source_key='test', dataset_id='test',
                termination_reason='sm_physical_termination', contacts=[{}]*(n-1))


class RecordedActionModel:
    def __init__(self, bias=0.):
        self.world_model = torch.nn.Linear(1, 1)
        self.spec = SimpleNamespace(joint_names=[f'joint_{i}' for i in range(21)])
        self.bias = bias

    def step_model(self, state, action, xi):
        result = state.clone()
        result[:, 0] += action[:, 0]+self.bias
        return result


def test_transition_alignment_and_orphan_tail():
    e = episode()
    assert paired_transitions(e) == 40
    e['state'] = e['state'][:-1]
    e['previous_action'] = e['previous_action'][:-1]
    assert paired_transitions(e) == 39
    e['previous_action'][4, 0] += .1
    with pytest.raises(ValueError, match='错位'):
        paired_transitions(e)
    e = episode()
    e['reset_count'] = 2
    with pytest.raises(ValueError, match='reset'):
        paired_transitions(e)


def test_multistep_uses_recorded_changing_actions_and_no_gt_reanchor():
    e = episode()
    exact = evaluate_episode(e, RecordedActionModel(), batch_size=7, anchors=4, horizons=(1, 8, 16, 32))
    assert exact['scores']['one_step_all']['count'] == 40
    assert exact['scores']['rollout_endpoint_h32']['metrics']['root_xy_m']['rmse'] < 1e-7
    biased = evaluate_episode(e, RecordedActionModel(.01), batch_size=7, anchors=4, horizons=(1, 8, 16, 32))
    h1 = biased['scores']['rollout_endpoint_h1']['metrics']['root_xy_m']['rmse']
    h32 = biased['scores']['rollout_endpoint_h32']['metrics']['root_xy_m']['rmse']
    assert h32 == pytest.approx(32*h1, rel=1e-4)
    assert {a['start_control_step'] for a in biased['anchors'] if a['horizon']==1} == {
        a['start_control_step'] for a in biased['anchors'] if a['horizon']==32}


def test_short_sequences_and_failure_denominator():
    starts, hs = matched_starts(12, (1, 8, 16, 32), 8)
    assert hs == [1, 8] and max(starts)+8 == 12
    score = pack_errors({'error': torch.tensor([1., 9., float('nan')])}, 3)
    assert score['count'] == 3 and score['finite_count'] == 2 and score['nonfinite_count'] == 1
    records = [dict(dataset='d', anchors=[], scores={'one_step_all': score}),
               dict(dataset='d', anchors=[], scores={'one_step_all': pack_errors({'error': torch.tensor([16.])}, 1)})]
    value = aggregate(records)['groups']['all']['one_step_all']
    assert value['count'] == 4 and value['nonfinite_count'] == 1
    assert value['metrics']['error']['pooled_rmse'] == pytest.approx((26/3)**.5)
    assert value['metrics']['error']['source_mean_rmse'] == pytest.approx((5**.5+4)/2)
