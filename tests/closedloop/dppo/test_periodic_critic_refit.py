"""周期完整MC条件重拟合的纯CPU验证：EV阈值触发、不改善时恢复原Critic、改善时接受。

替身只替换物理采集、独立MC指标和Critic训练，仍执行真实 calibrate_critic 的分支、状态
快照与恢复逻辑；不启动Isaac、不访问GPU。验证EV不低于阈值时只采验证bank不训练，低于
阈值但验证MSE不改善时Critic与Adam逐元素恢复且critic_updates不变，改善时接受并累加步数。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo import mc_calibration


class _Collective:
    rank = 0
    device = torch.device('cpu')
    def all_gather_object(self, value): return [value]
    def broadcast_object(self, value, src=0): return value


class _Bank:
    def __init__(self): self.episodes = []
    def training_data(self, **kwargs): return [], torch.zeros(0, dtype=torch.float64)


def _context(tmp_path):
    critic = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    critic(torch.ones(1, 2)).sum().backward(); optimizer.step(); critic.zero_grad(set_to_none=True)
    (tmp_path/'mc_calibration').mkdir()
    cfg = dict(samples_per_source=1, validation_seeds=[1], training_seeds=[2], mix_candidates=[0., 1.],
        epochs=1, quality_gate={})
    return SimpleNamespace(stage=dict(critic=dict(complete_mc=cfg, global_batch=4), ppo_rollout=dict(world_seconds=4.)),
        state=dict(policy_version=3, critic_updates=10), distributed=_Collective(), critic=critic,
        critic_optimizer=optimizer, generators=dict(critic=torch.Generator().manual_seed(1)),
        policy=SimpleNamespace(kernel_config={}), settings=dict(gamma_upper=.99, lambda_upper=.95), session=tmp_path)


def _patch(monkeypatch, metrics, collected):
    sequence = iter(metrics)
    monkeypatch.setattr(mc_calibration, 'collect_complete_bank', lambda c, label, *a, **k: collected.append(label) or _Bank())
    monkeypatch.setattr(mc_calibration, 'independent_mc_metrics', lambda c, bank: next(sequence))
    monkeypatch.setattr(mc_calibration, 'ValueConditionCache', lambda rows, device: None)
    def train(critic, optimizer, *args, **kwargs):
        with torch.no_grad(): critic.weight.add_(1.)
        return dict(optimizer_steps=5)
    monkeypatch.setattr(mc_calibration, 'critic_update', train)


def _metric(mse, ev): return dict(mse=mse, explained_variance=ev)


def test_high_ev_only_measures_validation_bank(monkeypatch, tmp_path):
    c, collected = _context(tmp_path), []
    _patch(monkeypatch, [_metric(10., .95)], collected)
    report = mc_calibration.calibrate_critic(c, 'policy_3', fit=False, fit_if_ev_below=.9)
    assert not report['fit'] and collected == ['policy_3_val'] and c.state['critic_updates'] == 10


def test_low_ev_refit_without_improvement_restores_critic(monkeypatch, tmp_path):
    c, collected = _context(tmp_path), []
    before = {k: v.clone() for k, v in c.critic.state_dict().items()}
    step = next(iter(c.critic_optimizer.state.values()))['step'].clone()
    _patch(monkeypatch, [_metric(10., .5), _metric(12., .4), _metric(11., .45)], collected)
    report = mc_calibration.calibrate_critic(c, 'policy_3', fit=False, fit_if_ev_below=.9)
    assert report['periodic_refit'] and not report['refit_accepted'] and collected == ['policy_3_val', 'policy_3_train']
    for key, value in c.critic.state_dict().items(): torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    torch.testing.assert_close(next(iter(c.critic_optimizer.state.values()))['step'], step, rtol=0, atol=0)
    assert c.state['critic_updates'] == 10
    assert report['after'] == report['before']
    assert report['candidate_after']['mse'] == 11.
    assert report['selected_mc_ratio'] is None
    assert report['relative_validation_mse_reduction'] == 0.


def test_low_ev_refit_with_improvement_is_accepted(monkeypatch, tmp_path):
    c, collected = _context(tmp_path), []
    _patch(monkeypatch, [_metric(10., .5), _metric(6., .7), _metric(8., .6)], collected)
    report = mc_calibration.calibrate_critic(c, 'policy_3', fit=False, fit_if_ev_below=.9)
    assert report['refit_accepted'] and report['selected_mc_ratio'] == 0. and c.state['critic_updates'] == 15


def test_finite_calibration_music_count_keeps_validation_and_default_unchanged(monkeypatch,tmp_path):
    c,collected=_context(tmp_path),[]
    _patch(monkeypatch,[_metric(10.,.5),_metric(6.,.7),_metric(8.,.6)],collected)
    counts={}
    def collect(c,label,split,*,samples_per_source,seeds):
        counts[split]=samples_per_source
        return _Bank()
    monkeypatch.setattr(mc_calibration,'collect_complete_bank',collect)
    report=mc_calibration.calibrate_critic(c,'finite',fit=False,fit_if_ev_below=1.,training_samples_per_source=8)
    assert counts=={'val':1,'train':8}
    assert report['training_count_override']==8 and report['training_samples_per_source']==8
    assert c.stage['critic']['complete_mc']['samples_per_source']==1
    assert 'training_samples_per_source' not in c.stage['critic']['complete_mc']
