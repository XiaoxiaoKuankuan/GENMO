"""学习信号诊断开关的纯CPU验证：优势置换负对照、半批梯度余弦、梯度抽样与配对告警。

优势置换只在全局有效且含自由坐标的转移之间进行，数值多重集与returns/价值目标必须不变，
同一种子跨调用可复现。SNR诊断用两进程Gloo真实归约验证：开启后p.grad与关闭时逐位
相同，两半rank余弦等于按各rank原梯度直接计算的结果，抽样等于全局PPO梯度的固定步长
坐标。配对告警只读取逐任务回报差，不改变选择。测试不导入Isaac、不使用GPU。
"""
from __future__ import annotations

from datetime import timedelta
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import yaml

from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.gradient_merge import BCGradientController, merge_global_gradients
from gem.closedloop.dppo.periodic_monitor import _regression_alert
from gem.closedloop.dppo.returns import shuffle_advantages_global
from gem.closedloop.dppo.training import _sketch_cosine
from gem.closedloop.dppo.training_config import configuration

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'configs/closedloop/stage2_asymmetric_server2.yaml'


def _targets():
    advantages = torch.tensor([.5, -1., 2., 3., -.25, 7.], dtype=torch.float64)
    manifest = [dict(owner_rank=0, local_index=i, valid=i != 1, has_free=i != 5) for i in range(6)]
    return dict(advantages=advantages, returns=torch.arange(6.), advantages_raw=advantages*3), manifest


def test_shuffle_preserves_multiset_returns_and_ineligible_rows():
    targets, manifest = _targets()
    result = shuffle_advantages_global(targets, manifest, seed=11)
    eligible = [0, 2, 3, 4]
    assert sorted(result['advantages'][eligible].tolist()) == sorted(targets['advantages'][eligible].tolist())
    assert result['advantages'][1] == targets['advantages'][1] and result['advantages'][5] == targets['advantages'][5]
    assert torch.equal(result['returns'], targets['returns']) and torch.equal(result['advantages_raw'], targets['advantages_raw'])
    assert result['advantage_control'] == 'shuffle_global' and result['advantage_shuffle_count'] == 4
    assert torch.equal(shuffle_advantages_global(targets, manifest, seed=11)['advantages'], result['advantages'])
    assert torch.equal(targets['advantages'], torch.tensor([.5, -1., 2., 3., -.25, 7.], dtype=torch.float64))


def test_regression_alert_uses_one_sided_upper_bound():
    assert _regression_alert([-1., -1.2, -.8, -1.1])['triggered']
    assert not _regression_alert([-1., 1.2, -.8, 1.1])['triggered']
    assert not _regression_alert([-1.])['available']


def test_sketch_cosine_matches_flattened_vectors():
    a = dict(w=torch.tensor([1., 0.]), b=torch.tensor([1.]))
    b = dict(w=torch.tensor([1., 1.]), b=torch.tensor([0.]))
    assert math.isclose(_sketch_cosine(a, b), 1/(math.sqrt(2)*math.sqrt(2)))
    assert _sketch_cosine(None, b) is None


@pytest.mark.parametrize('field,value', [('advantage_control', 'random'), ('gradient_snr_diagnostic', 1)])
def test_invalid_diagnostic_switch_is_rejected(tmp_path, field, value):
    c = yaml.safe_load(CONFIG.read_text())
    c['stage10']['training'][field] = value
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match=field):
        configuration(path)


def _merge(distributed, half_split):
    torch.manual_seed(0)
    actor = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    generator = torch.Generator().manual_seed(100 + distributed.rank)
    ppo = {n: torch.randn(p.shape, generator=generator) for n, p in actor.named_parameters()}
    bc = {n: torch.randn(p.shape, generator=torch.Generator().manual_seed(7)) for n, p in actor.named_parameters()}
    local_ppo = {n: v.clone() for n, v in ppo.items()}
    sketch = {} if half_split else None
    report = merge_global_gradients(actor, ppo, bc, BCGradientController(initial=.1, mode='fixed'),
        distributed=distributed, half_split=half_split, sketch=sketch, sketch_budget=5)
    return actor, report, local_ppo, sketch


def _worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory/'rendezvous').as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        distributed = DistributedCollectives(rank, 2, device='cpu')
        plain, _, _, _ = _merge(distributed, False)
        split, report, local, sketch = _merge(distributed, True)
        gathered = [None, None]
        dist.all_gather_object(gathered, {n: v for n, v in local.items()})
        torch.save(dict(plain={n: p.grad for n, p in plain.named_parameters()},
            split={n: p.grad for n, p in split.named_parameters()}, half=report['half_split'],
            locals=gathered, sketch=sketch), directory/f'rank_{rank}.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def snr_results(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo is unavailable')
    directory = tmp_path_factory.mktemp('snr_gloo')
    from tests.closedloop.dppo.test_optimization_support import _spawn_bounded
    _spawn_bounded(_worker, directory)
    return [torch.load(directory/f'rank_{r}.pt', weights_only=False) for r in range(2)]


def test_half_split_changes_no_gradient_and_reports_rank_half_cosine(snr_results):
    for result in snr_results:
        for name, value in result['plain'].items():
            assert torch.equal(value, result['split'][name])
    first, second = snr_results[0]['locals']
    names = sorted(first)
    f = torch.cat([first[n].reshape(-1) for n in names]).double()
    s = torch.cat([second[n].reshape(-1) for n in names]).double()
    expected = float(f @ s / (f.norm()*s.norm()))
    assert math.isclose(snr_results[0]['half']['cosine'], expected, rel_tol=1e-5)
    assert snr_results[0]['half'] == snr_results[1]['half']


def test_sketch_samples_global_ppo_gradient_with_fixed_stride(snr_results):
    first, second = snr_results[0]['locals']
    sketch = snr_results[0]['sketch']
    total = sum(v.numel() for v in first.values())
    stride = max(1, math.ceil(total/5))
    for name, value in sketch.items():
        expected = (first[name]+second[name]).reshape(-1)[::stride].double()
        torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-6)


def test_kl_adaptive_learning_rate_is_bounded_and_monotone():
    from types import SimpleNamespace
    from gem.closedloop.dppo.training import _kl_adapted_lr, _set_actor_lr
    assert math.isclose(_kl_adapted_lr(1e-7, 2e-4, 8e-4), 1.5e-7)
    assert math.isclose(_kl_adapted_lr(1e-7, 3.2e-3, 8e-4), .5e-7)
    assert math.isclose(_kl_adapted_lr(1e-7, 8e-4, 8e-4), 1e-7)
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-7)
    c = SimpleNamespace(actor_optimizer=optimizer, state={}, settings=dict(actor_lr=1e-7, actor_lr_bounds=[5e-8, 4e-7]))
    assert _set_actor_lr(c, 1e-6) == 4e-7 and optimizer.param_groups[0]['lr'] == 4e-7 and c.state['selected_actor_lr'] == 4e-7
    assert _set_actor_lr(c, 1e-9) == 5e-8


@pytest.mark.parametrize('update', [dict(actor_lr_control='kl_adaptive'),
    dict(actor_lr_control='kl_adaptive', actor_lr_bounds=[1e-7, 1e-6], kl_target_coordinate=.01),
    dict(kl_rejection='retry')])
def test_invalid_learning_rate_control_is_rejected(tmp_path, update):
    c = yaml.safe_load(CONFIG.read_text())
    c['stage10']['training'].update(update)
    path = tmp_path / 'invalid_lr.yaml'
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        configuration(path)


def test_kl_adaptive_accepts_maximum_step_metric_below_its_limit(tmp_path):
    c = yaml.safe_load(CONFIG.read_text())
    c['stage10']['training'].update(actor_lr_control='kl_adaptive', actor_lr_bounds=[5e-8, 2e-6],
        kl_target_coordinate=.0004, kl_control_metric='maximum_step_mean_coordinate_kl')
    path = tmp_path / 'max_step.yaml'
    path.write_text(yaml.safe_dump(c))
    assert configuration(path)['stage10']['training']['kl_control_metric'] == 'maximum_step_mean_coordinate_kl'
    c['stage10']['training']['kl_control_metric'] = 'joint'
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='kl_control_metric'):
        configuration(path)
