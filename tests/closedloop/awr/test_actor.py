"""真实小型Stage1网络的AWR单时间去噪、完整DDIM和无伪Contact回归。

同一条件/初始噪声直接比较原生Stage1采样；单位权重比较普通监督梯度；不同切分
按全局分母求梯度，Padding/已知前缀必须没有重建梯度。不会模拟物理改善结论。
"""
import copy
import pytest
import torch
from tests.closedloop.test_stage1_actor import actor_factory, _conditions
from gem.closedloop.awr.sampler import AWRSampler
from gem.closedloop.awr.losses import weighted_denoising, per_sample_denoising
from gem.closedloop.awr.critic import assert_independent
from gem.closedloop.common.pretrained_critic import PretrainedConditionCritic


def test_sampler_exactly_stage1(actor_factory):
    actor, batch = actor_factory(motion_frames=60)
    c = _conditions(batch)
    noise = torch.randn_like(c["known_qpos30"])
    original = actor.sample(c, steps=20, guidance_scale=2.5, noise=noise)
    actual = AWRSampler(actor).sample(c, noise=noise)
    for key in ("qpos", "qpos30", "contact"):
        torch.testing.assert_close(original[key], actual[key], atol=0, rtol=0)
    other = AWRSampler(actor).sample(c, seeds=[12, 13])
    assert not torch.equal(actual["qpos30"], other["qpos30"])


def test_shared_forward_no_contact_and_original_stage1(actor_factory):
    actor, batch = actor_factory(motion_frames=60)
    c = _conditions(batch)
    t = torch.tensor([13, 700]); noise = torch.randn_like(batch["target_qpos30"])
    original = actor.training_forward(batch, timesteps=t, noise=noise)
    replay = actor.denoising_forward(c, batch["target_qpos30"], batch["target_qpos30_valid"],
                                    timesteps=t, noise=noise)
    assert "contact" not in c
    for key in ("pred_x_start", "target_x_start", "static_conf_logits", "xt"):
        torch.testing.assert_close(original[key], replay[key], atol=0, rtol=0)
    with pytest.raises((ValueError, KeyError)):
        actor.training_forward(dict(c, target_qpos30=batch["target_qpos30"],
                                    target_qpos30_valid=batch["target_qpos30_valid"]))


def test_samplewise_weights_mask_and_microbatch_gradient():
    torch.manual_seed(7)
    prediction = torch.randn(5, 60, 30, requires_grad=True)
    target = torch.randn_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, :6] = False; mask[0, 40:] = False
    weight = torch.tensor([1., 2., .1, 3., 1.], dtype=torch.float64)
    full, per = weighted_denoising(prediction, target, mask, weight)
    full.backward()
    fullgrad = prediction.grad.clone()
    assert (fullgrad[~mask] == 0).all()
    manual = torch.stack([(prediction[i][mask[i]]-target[i][mask[i]]).square().mean() for i in range(5)])
    torch.testing.assert_close(per, manual)
    prediction.grad = None
    for sl in (slice(0, 1), slice(1, 3), slice(3, 5)):
        weighted_denoising(prediction[sl], target[sl], mask[sl], weight[sl], global_count=5)[0].backward()
    torch.testing.assert_close(prediction.grad, fullgrad)
    prediction.grad = None
    weighted_denoising(prediction, target, mask, torch.ones(5))[0].backward()
    unit = prediction.grad.clone(); prediction.grad = None
    per_sample_denoising(prediction, target, mask).mean().backward()
    torch.testing.assert_close(prediction.grad, unit)


def test_two_critics_have_independent_storage(actor_factory):
    actor, _ = actor_factory(motion_frames=60)
    a = PretrainedConditionCritic(actor, dimension=32, layers=1)
    b = PretrainedConditionCritic(actor, dimension=32, layers=1)
    assert_independent(a, b)
    assert all(not p.requires_grad for m in a.encoder_modules() for p in m.parameters())
