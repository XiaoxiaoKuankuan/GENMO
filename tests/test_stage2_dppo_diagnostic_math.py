"""验证固定真实数据诊断所用数学，而不启动物理环境或修改正式算法。

使用独立Normal分布公式核对非零策略偏移下的原四步坐标平均PPO目标和梯度，
显式覆盖优势正负号、实际生效裁剪、Microbatch全局分母与真实参数增量方向。
这些是诊断工具自身的正确性证据，不能代替服务器上的真实rollout与GMT验证。
测试只在内存中构造张量，不写入正式模型、日志或数据目录。
"""
import pytest
import torch

from gem.closedloop.dppo.gaussian_probability import gaussian_statistics
from tools.stage2_dppo_diagnostic_math import (
    STEPS, actual_update_direction, production_surrogate, summarize_records, vector_relation,
)


def independent_records(mean, values):
    old = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    new = torch.distributions.Normal(mean, torch.ones_like(mean))
    log_ratio = (new.log_prob(values)-old.log_prob(values)).sum(-1)
    kl = .5*mean.square().sum(-1)
    return torch.stack((log_ratio, kl, torch.full_like(kl, mean.shape[-1])), -1)


@pytest.mark.parametrize('clip', [.01, .005, .003])
def test_independent_normal_full_objective_and_gradient(clip):
    gen = torch.Generator().manual_seed(641)
    values = torch.randn(7, 4, 11, generator=gen, dtype=torch.float64)
    mean = (torch.randn(7, 4, 11, generator=gen, dtype=torch.float64)*.035).requires_grad_()
    advantage = torch.tensor([2., -3., .2, -.1, 0., .7, -.8], dtype=torch.float64)
    records = independent_records(mean, values)
    actual = production_surrogate(records, advantage, clip=clip)['objective'].mean()
    ratio = torch.exp(records[..., 0]/11)
    discounted = advantage[:, None]*mean.new_tensor([.99**(19-step) for step in STEPS])
    expected = torch.minimum(ratio*discounted, ratio.clamp(1-clip, 1+clip)*discounted).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, mean, retain_graph=True)[0],
                               torch.autograd.grad(expected, mean)[0])


def test_active_clip_is_sign_dependent_and_changes_shifted_gradient():
    log_ratio = torch.tensor([[.006]*4, [-.006]*4, [.006]*4, [-.006]*4], dtype=torch.float64, requires_grad=True)
    records = torch.stack((log_ratio*100, torch.ones_like(log_ratio), torch.full_like(log_ratio, 100)), -1)
    adv = torch.tensor([1., -1., -1., 1.], dtype=torch.float64)
    broad = summarize_records(records, adv, clip=.01)
    narrow = summarize_records(records, adv, clip=.003)
    assert broad['active_clip_fraction'] == 0
    assert narrow['ordinary_clip_fraction'] == 1
    assert narrow['active_clip_fraction'] == .5
    grad = []
    for clip in [.01, .005, .003]:
        obj = production_surrogate(records, adv, clip=clip)['objective'].mean()
        grad.append(torch.autograd.grad(obj, log_ratio, retain_graph=True)[0])
    assert not torch.equal(grad[0], grad[2])
    torch.testing.assert_close(grad[1], grad[2])


def test_microbatches_use_full_denominator_and_padding_has_zero_gradient():
    values = torch.linspace(-1., 1., 120, dtype=torch.float64).reshape(5, 4, 6)
    mean = torch.full_like(values, .02, requires_grad=True)
    advantage = torch.tensor([1., -1., 2., .3, -.7], dtype=torch.float64)
    records = independent_records(mean, values)
    whole = -production_surrogate(records, advantage, clip=.005)['objective'].sum()/20
    split = sum(-production_surrogate(records[a:b], advantage[a:b], clip=.005)['objective'].sum()/20
                for a,b in [(0,1), (1,3), (3,5)])
    torch.testing.assert_close(whole, split)
    torch.testing.assert_close(torch.autograd.grad(whole, mean, retain_graph=True)[0],
                               torch.autograd.grad(split, mean)[0])
    free = torch.ones(2, 1, 6, dtype=torch.bool);free[:, :, :3] = False
    target = torch.full((2,1,6), .2, dtype=torch.float64, requires_grad=True)
    stats = gaussian_statistics(torch.ones_like(target), torch.zeros_like(target), torch.ones(2,1,1),
                                target, torch.ones(2,1,1), free)
    stats['joint_log_ratio'].sum().backward()
    assert torch.count_nonzero(target.grad[:, :, :3]) == 0


def test_actual_adam_direction_and_gradient_relation():
    actor = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():actor.weight.copy_(torch.tensor([[1., 2.]]))
    before = {'weight':actor.weight.detach().clone()}
    gradient = {'weight':torch.tensor([[2., 4.]], dtype=torch.float64)}
    with torch.no_grad():actor.weight.add_(gradient['weight'], alpha=-.01)
    report = actual_update_direction(actor, before, gradient)
    assert report['full_objective_first_order_gain'] == pytest.approx(.2)
    assert report['update_cosine_with_objective_ascent'] == pytest.approx(1.)
    assert vector_relation(gradient, gradient)['relative_difference'] == 0
    assert vector_relation(gradient, {'weight':-gradient['weight']})['cosine'] == pytest.approx(-1.)
