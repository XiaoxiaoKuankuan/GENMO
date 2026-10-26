"""音乐代理的时钟、延迟、mask、信号排序和有限差分测试。

纯合成执行速度独立于参考速度，检查30/50/600Hz及平移等价；只支付完整因果窗口，
静止/恒速与无beat不能获得虚假满分。音乐排序是代理数值性质，不能外推物理音乐性。
"""

import pytest
import torch

from gem.closedloop.reference_adapter.music_reward import AudioTrack, MusicReward, MusicState


def audio(start=0, beats=True):
    features = torch.zeros(1, 150, 35)
    if beats:
        features[:, 15::15, 34] = 1
    return AudioTrack.from_features(features, start_tick=start)


def test_clock_preview_shift_and_no_beat():
    a, shifted = audio(), audio(612)
    t = torch.arange(100)*12
    torch.testing.assert_close(a.target(t)[0], shifted.target(t+612)[0])
    torch.testing.assert_close(a.mean_target, shifted.mean_target)
    assert a.beat_ticks[0][0].item() == 300
    assert a.target([300])[0].item() == 1
    values, valid, times = a.preview(-12)
    assert not valid[0, 0] and times[0, 0] == pytest.approx(-8/600)
    assert values.shape == (1, 16, 35)
    empty = audio(beats=False)
    assert not empty.has_beats.any() and torch.isfinite(empty.mean_target).all()
    assert not empty.target(t)[1].any()


def run_signal(qd, a):
    reward = MusicReward(torch.ones(21))
    memory = MusicState.empty(qd[0:1])
    rewards, centers, probabilities = [], [], []
    for i in range(len(qd)):
        memory = reward.push(memory, qd[i:i+1], torch.ones_like(qd[i:i+1]), i*12+a.start_tick)
        memory, value, _, info = reward.settle(memory, a)
        if info["music_valid"].item():
            rewards.append(value)
            centers.append(info["music_center_tick"])
            probabilities.append(info["music_probability"])
    return torch.cat(rewards) if rewards else qd.new_zeros(0), centers, memory


def test_signal_ordering_delay_and_activity():
    a = audio()
    t = torch.arange(240)*.02
    def sync(shift, amplitude=.2, width=.08):
        phase = ((t-shift+.25) % .5)-.25
        return (.8-amplitude*torch.exp(-.5*(phase/width)**2))[:, None].expand(-1, 21)
    torch.manual_seed(42)
    signals = dict(sync=sync(0), quarter=sync(.125), half=sync(.25),
                   static=torch.zeros(240, 21), constant=torch.ones(240, 21)*.8,
                   high_frequency=torch.rand(240, 1).expand(-1, 21)*1.6)
    scores = {k: float(run_signal(v, a)[0].mean()) for k, v in signals.items()}
    print("music_signal_scores", scores)
    assert all(scores["sync"] > scores[k] for k in scores if k != "sync")
    # 保留首轮真实发现的反例：固定公式对尖锐谷的正目标肩部惩罚过大。
    # 这不是放宽验收门槛；通用反站立/音乐性仍明确未通过。
    sharp = float(run_signal(sync(0, amplitude=.75, width=.045), a)[0].mean())
    print("known_music_proxy_counterexample", {"sharp_sync": sharp, "static": scores["static"]})
    assert sharp < scores["static"]
    _, centers, memory = run_signal(signals["sync"], a)
    assert centers[0] == 24 and centers[-1] == (239-2)*12
    assert len(set(centers)) == len(centers)
    reward = MusicReward(torch.ones(21))
    with pytest.raises(ValueError):
        reward.settle(memory, a)
    _, _, stationary = run_signal(signals["static"], a)
    stationary = reward.push(stationary, torch.zeros(1, 21), torch.ones(1, 21), 240*12)
    _, _, cost, _ = reward.settle(stationary, a)
    assert cost.item() == pytest.approx(.49, rel=1e-5)
    assert len(run_signal(signals["sync"][:4], a)[0]) == 0
    assert len(run_signal(signals["sync"], audio(beats=False))[0]) == 0


def test_music_gradient_and_finite_difference_uses_executed_velocity():
    a = audio()
    qd = (.3+torch.arange(9)[:, None]*.03+torch.arange(21)[None]*.001).requires_grad_()
    def objective(x):
        return run_signal(x, a)[0].sum()
    value = objective(qd)
    grad, = torch.autograd.grad(value, qd)
    assert torch.isfinite(grad).all() and grad.norm() > 0
    direction = grad/grad.norm()
    eps = 1e-3
    finite = (objective(qd.detach()+eps*direction)-objective(qd.detach()-eps*direction))/(2*eps)
    torch.testing.assert_close(finite, (grad*direction).sum(), atol=2e-3, rtol=.02)
    assert not objective(qd.detach()).requires_grad
