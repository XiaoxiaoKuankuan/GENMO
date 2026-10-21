"""独立Stage2奖励尺度、真正关闭BC、扩散时间及来源权重的数学与恢复回归。

使用同一具名GMT格式的控制证据，分别计算原source_std与新reward_std下的指数分数，
比较CPU标量、设备张量及列式消费；篡改来源仍须失败。训练测试用真实小型Stage1，
确认关闭BC不创建采样器、不更新Contact head，不调用DDIM链，尾批/恢复和microbatch
对同一固定目标给出一致梯度。合成反馈仅验证实现，不代替真实PhysX收益。
"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.closedloop.common.rewards import (ExecutionReward, GMT_SOURCE_STD, TRACKING_WEIGHTS,
    NO_ACTIVITY_REWARD_VERSION, STAGE2_REWARD_STD_VERSION, resolve_reward_config)
from gem.closedloop.common.vector_reward_math import VectorRewardMath, reward_config_sha
from gem.closedloop.common.vector_reward_adapter import VectorExecutionReward, compare_reward_evidence
from gem.closedloop.awr.objective import timestep_weights, source_weights
from gem.closedloop.awr.trainer import ActorTrainer
from gem.closedloop.awr.dataset import ReplayDataset
from gem.closedloop.awr.distributed import Distributed
from gem.closedloop.awr.config import load_config
from gem.runtime.trajectory_blocks import ColumnarTrace, pack_trace
from tests.closedloop.dppo.test_data_learning import actual_step, target_activity_fixture
from tests.closedloop.dppo.test_stage2_reward import device_packet


NEW_STD = dict(GMT_SOURCE_STD, anchor_pos=.20, anchor_ori=.30, body_pos=.15, body_ori=.20)


def reward_config():
    return dict(version=STAGE2_REWARD_STD_VERSION,
                tracking=dict(source_std=GMT_SOURCE_STD.copy(), reward_std=NEW_STD.copy()))


def test_new_scales_preserve_sources_and_three_reward_paths():
    original = ExecutionReward(dict(version=NO_ACTIVITY_REWARD_VERSION), target_activity=target_activity_fixture)
    scalar = ExecutionReward(reward_config(), target_activity=target_activity_fixture)
    vector = VectorExecutionReward(reward_config(), target_activity=target_activity_fixture)
    engine = VectorRewardMath(scalar.config, 1, 'cpu')
    for c in (original, scalar, vector): c.seed_previous_target(np.zeros(21))
    engine.previous_valid.fill_(True)
    for index in range(3):
        row = actual_step(index, speed=.1); row['env_id'] = 0
        for i, (name, item) in enumerate(row['motion_tracking']['terms'].items()):
            item['error'] = (.03*(i+1))**2
            item['source_score'] = math.exp(-item['error']/GMT_SOURCE_STD[name]**2)
        for substep in row['physics_substeps']:
            d = substep['physical_diagnostics']
            for field in ('foot_support_sphere_tangent_speed_m_s', 'foot_support_sphere_clearance_m'):
                d[field] = {n: np.zeros(1) for n in d['foot_body_names']}
        row['physical_diagnostics'] = copy.deepcopy(row['physics_substeps'][-1]['physical_diagnostics'])
        source_before = copy.deepcopy(row['motion_tracking']['terms'])
        row['reward_primitives'] = device_packet(row, engine)
        old = original.evaluate_step(row); new = scalar.evaluate_step(row)
        assert old['transition_valid'] and new['transition_valid'], new['errors']
        result = vector.evaluate_batch(ColumnarTrace([pack_trace([row])]))[0]
        compare_reward_evidence(result, new)
        scores = new['components']['track']['normalized']['scores']
        for name, item in source_before.items():
            assert scores[name] == pytest.approx(math.exp(-item['error']/NEW_STD[name]**2))
            assert row['motion_tracking']['terms'][name]['source_std'] == GMT_SOURCE_STD[name]
            if name in ('body_lin_vel', 'body_ang_vel', 'foot_height', 'leg_joint_pos'):
                assert scores[name] == old['components']['track']['normalized']['scores'][name]
        for name in old['components'].keys()-{'track'}:
            compare_reward_evidence(new['components'][name], old['components'][name])
        assert new['components']['track']['gate'] == 1.
        expected_delta = .02*2.5*sum(TRACKING_WEIGHTS[k]*(scores[k]-source_before[k]['source_score']) for k in scores)
        assert new['reward']-old['reward'] == pytest.approx(expected_delta, abs=1e-14)
    assert reward_config_sha(original.config) != reward_config_sha(scalar.config)
    assert scalar.config['tracking']['source_std'] == GMT_SOURCE_STD
    assert 'std' not in scalar.config['tracking']


@pytest.mark.parametrize('bad', ['source_config', 'source_evidence', 'legacy_std', 'incomplete', 'nan', 'zero'])
def test_scale_split_never_bypasses_authoritative_checks(bad):
    c = reward_config()
    if bad == 'source_evidence':
        row = actual_step(); row['motion_tracking']['terms']['anchor_pos']['source_std'] = .2
        result = ExecutionReward(c, target_activity=target_activity_fixture).evaluate_step(row)
        assert not result['transition_valid'] and any('authoritative' in e for e in result['errors'])
        with pytest.raises(ValueError, match='authoritative'):
            device_packet(row, VectorRewardMath(c, 1, 'cpu'))
        return
    if bad == 'source_config': c['tracking']['source_std']['anchor_pos'] = .2
    if bad == 'legacy_std': c['tracking']['std'] = GMT_SOURCE_STD.copy()
    if bad == 'incomplete': c['tracking']['reward_std'].pop('body_ang_vel')
    if bad == 'nan': c['tracking']['reward_std']['anchor_pos'] = float('nan')
    if bad == 'zero': c['tracking']['reward_std']['anchor_pos'] = 0.
    with pytest.raises(ValueError): resolve_reward_config(c)
    with pytest.raises(ValueError):
        resolve_reward_config(dict(version=NO_ACTIVITY_REWARD_VERSION, tracking=dict(std=NEW_STD)))


def test_min_snr_x0_formula_and_fixed_global_normalizer():
    alpha = torch.tensor([.99, .8, .5, .01], dtype=torch.float64)
    raw = torch.tensor([5., 4., 1., 1./99.], dtype=torch.float64)
    torch.testing.assert_close(timestep_weights(alpha, dict(mode='min_snr', gamma=5., mean_normalize=False)), raw)
    table = timestep_weights(alpha, dict(mode='min_snr', gamma=5., mean_normalize=True))
    torch.testing.assert_close(table, raw/raw.mean())
    assert table.mean() == pytest.approx(1.)
    assert table[0]/table[-1] == pytest.approx(495.)
    torch.testing.assert_close(timestep_weights(alpha), torch.ones(4, dtype=torch.float64))
    for gamma in (-1., 0., float('inf'), True):
        with pytest.raises(ValueError): timestep_weights(alpha, dict(mode='min_snr', gamma=gamma))
    factors = source_weights(['long']*6+['short']*2, 'uniform_sources')
    assert factors[:6].sum() == pytest.approx(4., abs=1e-14)
    assert factors[6:].sum() == pytest.approx(4., abs=1e-14)
    assert factors.mean() == pytest.approx(1., abs=1e-14)


def no_bc_config(**kwargs):
    return dict(epochs=2, global_batch=3, microbatch=1, bc_enabled=False, bc_global_batch=0,
                lr=1e-4, warmup_steps=1, source_balance='uniform_sources',
                timestep_weighting=dict(mode='min_snr', gamma=5., mean_normalize=True), **kwargs)


def test_no_bc_real_actor_resume_contact_unused_and_no_sampling(episode_factory, monkeypatch):
    actor, _, make = episode_factory
    data = ReplayDataset([make(episode_id='a', source='one'), make(episode_id='b', source='two')], unit_weights=True)
    def forbidden(*a, **k): raise AssertionError('Disabled BC/DDIM was called')
    monkeypatch.setattr(actor, 'sample', forbidden)
    from gem.closedloop.awr import runtime
    monkeypatch.setattr(runtime, 'Stage1Supervision', forbidden)
    run = runtime.AWRRun.__new__(runtime.AWRRun); run.config = dict(awr=dict(actor=dict(bc_enabled=False)))
    assert run.training_bc() is None
    original = copy.deepcopy(actor.state_dict())
    trainer = ActorTrainer(actor, data, None, no_bc_config(), Distributed())
    assert trainer.calibrate()['coefficient'] == 0.
    saved = []
    result = trainer.run(on_step=lambda t, _: saved.append(t.state_dict()) if t.steps == 1 else None)
    assert result['steps'] == 4 and result['bc_optimizer_steps'] == 0 and not result['bc_enabled']
    assert all(h['bc_loss'] == h['bc_weight'] == 0. for h in trainer.history)
    assert saved[0]['bc'] is None
    final = copy.deepcopy(actor.state_dict())
    contact = [k for k in final if 'static_conf_head' in k]
    assert contact and all(torch.equal(original[k], final[k]) for k in contact)
    assert any(not torch.equal(original[k], v) for k, v in final.items())
    resumed = ActorTrainer(actor, data, None, no_bc_config(), Distributed())
    resumed.load_state_dict(saved[0]); resumed.run()
    for key in final: torch.testing.assert_close(actor.state_dict()[key], final[key], atol=0, rtol=0)
    bad = copy.deepcopy(saved[0]); bad['bc_weight'] = .1
    with pytest.raises(ValueError, match='Disabled BC'): resumed.load_state_dict(bad)


def test_time_source_weights_microbatch_gradient_equivalence(episode_factory):
    actor, _, make = episode_factory
    data = ReplayDataset([make(episode_id='a', source='one', lengths=(2, 2, 1)),
                          make(episode_id='b', source='two')], unit_weights=True)
    grads = []
    for micro in (1, 3, 5):
        config = no_bc_config(); config['microbatch'] = micro
        trainer = ActorTrainer(actor, data, None, config, Distributed())
        actor.zero_grad(set_to_none=True); trainer.backward_awr(list(range(5)), step=0)
        grads.append({n: p.grad.clone() for n, p in actor.named_parameters() if p.grad is not None})
    for other in grads[1:]:
        assert grads[0].keys() == other.keys()
        for key in other: torch.testing.assert_close(other[key], grads[0][key], atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize("filename,mode,uniform_control", [
    ("stage2_awr_rewardstd_nobc_minsnr.yaml", "min_snr", True),
    ("stage2_awr_rewardstd_nobc.yaml", "uniform", False),
])
def test_experiment_config_explicitly_disables_bc_and_keeps_source_std(filename,mode,uniform_control):
    root = Path(__file__).resolve().parents[3]
    c = load_config(root/'configs/closedloop'/filename)
    assert c['reward']['tracking']['source_std'] == GMT_SOURCE_STD
    assert c['reward']['tracking']['reward_std'] == NEW_STD
    assert c['awr']['actor']['bc_enabled'] is False and c['awr']['actor']['bc_global_batch'] == 0
    assert c['awr']['actor']['reference_weight'] == 0
    assert c['awr']['uniform_ablation'] is uniform_control and c['awr']['run_ablation']
    assert c['awr']['actor']['timestep_weighting']['mode'] == mode
    assert c['evaluation']['split'] == 'train'


def test_quality_statistics_follow_consumed_reference_not_generation_interval(episode_factory):
    from tests.closedloop.awr.test_learning_signals import with_ledger
    from gem.closedloop.awr.advantages import label_episode_advantages
    _, _, make = episode_factory
    episode = with_ledger(make(lengths=(2, 2)))
    label, _ = label_episode_advantages([episode], {episode.episode_id: torch.zeros(2)}, identity={},
        config=dict(mode='credit_nstep', nstep=4, credit_mode='current_plus_lookahead'))
    # p0在自己的前瞻和下一段中心各占一份；p1仅在下一段前瞻。
    metric = label.statistics['physics']['by_source']['overall']['metrics']['joint_position_rmse_rad']
    assert metric['mean'] == pytest.approx((.7+1.2)/2)
    assert label.statistics['physical_quality_time_basis'].startswith('actual_plan')
