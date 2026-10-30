"""执行稳定性辅助奖励的合同、因果历史和真实冻结网络梯度回归。

纯张量用例检查jerk的物理单位、复位mask、三个时刻的有限差分梯度，以及名义
支撑代理对摆腿的排除。真实SM权重夹具验证奖励开关不改变Goal、Actor或原六项
Tracking，新增项仅乘一次dt，BPTT内部历史没有detach。合成名义数据只验证数值
正确性，不代替Newton真实接触检查或整歌改善证据。测试不启动仿真或正式训练。
"""

from pathlib import Path
import torch

from gem.closedloop.reference_adapter.execution_costs import execution_jerk, stance_foot_velocity
from gem.closedloop.reference_adapter.precision import PrecisionObjective
from gem.closedloop.reference_adapter.application import direct_reference_config
from gem.closedloop.reference_adapter.vector_replay import VectorReplay
from test_direct_reference import direct_system


def test_four_conditions_two_seeds_keep_common_physical_score():
    import yaml
    directory = Path(__file__).resolve().parents[3]/'configs/closedloop/stage2_root_execution_stability_20261030'
    configs = [direct_reference_config(yaml.safe_load((directory/f'gpu{i}.yaml').read_text())) for i in range(8)]
    for i,c in enumerate(configs):
        assert c['bptt_steps'] == 32 and c['vector']['target_accepted_updates'] == 10000
        assert c['vector']['num_envs'] == 2048 and c['training']['wm_fit_window'] == 8
        assert c['training']['seed'] == c['vector']['sampling_seed'] == 42+i%2
        assert c['alternating']['critic_enabled'] is False
        assert c['alternating']['root_xy'] is False and c['alternating']['action_noise'] == 0
        assert c['sm_mdp'] == configs[0]['sm_mdp']
        assert c['sm_mdp']['root_pos_weight'] == 1.2 and c['sm_mdp']['root_ori_weight'] == 1.
        p=c['alternating']['precision']; group=i//2
        assert p['root_tracking_bonus_weight'] == (1.2 if group in (1,3) else 0.)
        assert (p['execution_jerk_weight'] > 0) == (group in (2,3))
        assert (p['stance_foot_weight'] > 0) == (group in (2,3))
        assert p == configs[i//2*2]['alternating']['precision']
        assert p['root_xy_weight'] == configs[0]['alternating']['precision']['root_xy_weight']
        assert p['soft_reference_weights'] == configs[0]['alternating']['precision']['soft_reference_weights']


def test_jerk_units_reset_and_three_state_gradients():
    x = torch.zeros(2, 55, dtype=torch.float64, requires_grad=True)
    y = torch.zeros_like(x, requires_grad=True)
    previous = torch.full((2, 21), .002, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([True, False])
    cost, rms = execution_jerk(x, y, previous, valid, 10.)
    torch.testing.assert_close(rms, torch.tensor([5., 0.], dtype=torch.float64))
    torch.testing.assert_close(cost, torch.tensor([.125, 0.], dtype=torch.float64))
    gx, gy, gp = torch.autograd.grad(cost.sum(), (x, y, previous))
    assert gx[0, 34:].norm() > 0 and gx[1].norm() == 0
    torch.testing.assert_close(gx[:, 34:], gp)
    torch.testing.assert_close(gy[:, 34:], -2*gp)
    assert torch.autograd.gradcheck(lambda a,b,c: execution_jerk(a,b,c,valid,10.)[0], (x,y,previous))


def test_stance_proxy_keeps_nominal_swing_and_actual_velocity_gradient():
    v = torch.tensor([[[.15,.02,0.],[.2,0.,0.]]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[[0.,0.,0.],[0.,0.,.3]]], dtype=torch.float64)
    target = torch.zeros_like(v); target[:, 1, 0] = .8
    floor = torch.zeros(1,2,dtype=torch.float64)
    cost, raw = stance_foot_velocity(v,p,target,floor)
    assert raw['stance_weight'][0,0] > .8 and raw['stance_weight'][0,1] == 0
    grad, = torch.autograd.grad(cost.sum(), v)
    assert grad[0,0,:2].norm() > 0 and grad[0,1].norm() == 0
    assert torch.autograd.gradcheck(lambda a: stance_foot_velocity(a,p,target,floor)[0], (v,))
    assert stance_foot_velocity(target,p,target,floor)[0].item() == 0


def test_actual_network_reward_single_dt_history_replay_and_frozen_bundle(system):
    engine,_,c,z = direct_system(system)
    settings = dict(root_tracking_bonus_weight=1.2, execution_jerk_weight=.03,
                    execution_jerk_scale=500., stance_foot_weight=.02, stance_foot_scale=.1)
    cfg = direct_reference_config(dict(alternating=dict(precision=settings)))
    assert all(cfg['alternating']['precision'][k] == v for k,v in settings.items())
    assert cfg['alternating']['objective_version'].endswith('v10')
    c.precision = PrecisionObjective(**settings)
    snapshot = c.bundle.frozen_snapshot()
    with torch.no_grad(): c.adapter.net[-1].bias.normal_(std=.01)
    with engine.dynamics.adapter_phase():
        first = c.propose(z)
        state1 = engine.dynamics.step(z['state'],first['action'],z['xi'])
        z1,_,raw0,terms0,*_ = c.advance(z,first,state1)
        assert terms0['execution_jerk'].item() == 0
        torch.testing.assert_close(terms0['root_tracking_bonus'],.02*1.2*raw0['root_pos'])
        second = c.propose(z1)
        state2 = engine.dynamics.step(state1,second['action'],z1['xi'])
        z2,reward,raw,terms,*_ = c.advance(z1,second,state2)
        expected,_ = execution_jerk(state2,state1,z['state'][:,34:],z1['execution_jerk_valid'],500.)
        torch.testing.assert_close(terms['execution_jerk'],-.02*.03*expected)
        assert z2['execution_previous_qd'].requires_grad
        g, = torch.autograd.grad(terms['execution_jerk'].sum()+terms['stance_foot_velocity'].sum(),
                                 c.adapter.net[-1].bias,retain_graph=True)
        assert g.isfinite().all() and g.norm() > 0
        replay = VectorReplay(40,1,'cpu',compute_mc=False)
        replay.add(z1,second['action'],state2,reward,torch.ones(1),torch.zeros(1,dtype=torch.bool),
                   torch.zeros(1,dtype=torch.bool),0)
        recovered = replay.starts(1)
        torch.testing.assert_close(recovered['execution_previous_qd'],z1['execution_previous_qd'])
        assert recovered['execution_jerk_valid'].all()
        c.precision = None
        unchanged = c.propose(z1)
        torch.testing.assert_close(second['goal'],unchanged['goal'],atol=0,rtol=0)
        torch.testing.assert_close(second['action'],unchanged['action'],atol=0,rtol=0)
        plain_terms = c.advance(z1,unchanged,state2)[3]
        torch.testing.assert_close(terms['sm'],plain_terms['sm'],atol=0,rtol=0)
    c.bundle.assert_unchanged(snapshot)
