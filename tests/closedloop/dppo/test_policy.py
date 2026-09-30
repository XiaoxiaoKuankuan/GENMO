"""第九步随机扩散策略的 CPU 契约与概率验收。

测试复用现有 Stage1Actor 的小型真实 Transformer/BumiEndecoder 夹具，不复制网络，
不加载正式 checkpoint、不启动 GPU/Isaac 或训练任务。所有测试数据通过 pytest 指定
临时目录生成，由调用者在测试结束后精确清理。本文件检查完整 20 步链的新旧等概率、
独立 Normal 密度 oracle、原始 diffusion 时间映射和 DDIM 公式、末步真实随机性、
逐坐标前缀与 padding 隔离、仅音乐 CFG、条件负例，以及梯度只回到策略参数的边界。

这些结果只证明采样与概率接口一致，不证明实际动作可执行、奖励质量或训练收敛。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.policy import DPPODiffusionPolicy, masked_joint_log_prob
from gem.diffusion_utils import gaussian_diffusion as gd
from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _sample(policy, conditions, seed=291):
    return policy.sample_rollout(conditions, generator=torch.Generator().manual_seed(seed))


def test_full_twenty_step_unchanged_policy_ratio_and_independent_density(actor_factory):
    actor, batch = actor_factory()
    policy = DPPODiffusionPolicy(actor)
    trace = _sample(policy, _conditions(batch))
    assert trace["chain"].shape == (2, 21, 120, 30)
    assert trace["old_means"].shape == (2, 20, 120, 30)
    assert trace["old_stds"].shape == (2, 20, 1, 1)
    assert trace["chain"].dtype == torch.float32
    assert trace["old_log_probs"].dtype == torch.float64
    assert not trace["chain"].requires_grad
    for index in range(20):
        actual = policy.evaluate_log_probs(trace["conditions"], trace["chain"][:, index],
                                          trace["chain"][:, index + 1], index)
        torch.testing.assert_close(actual, trace["old_log_probs"][:, index], rtol=0, atol=1e-9)
        torch.testing.assert_close((actual - trace["old_log_probs"][:, index]).exp(),
                                   torch.ones(2, dtype=torch.float64), rtol=0, atol=1e-9)
        distribution = torch.distributions.Normal(trace["old_means"][:, index].double(),
                                                   trace["old_stds"][:, index].double())
        independent = distribution.log_prob(trace["chain"][:, index + 1].double())
        independent = independent.masked_fill(~trace["free_mask"], 0).sum((-2, -1))
        torch.testing.assert_close(actual, independent, rtol=0, atol=1e-9)
    assert trace["timestep_map"].tolist() == list(reversed(policy.diffusion.timestep_map))
    assert trace["kernel_config"]["log_prob_reduction"] == "joint_sum_fp64"


def test_ddim_mean_matches_original_schedule_and_floor_never_changes_mean(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    conditions = _conditions(batch)
    policy = DPPODiffusionPolicy(actor, steps=20)
    larger_floor = DPPODiffusionPolicy(actor, steps=20, std_floor=0.3)
    state = torch.randn_like(conditions["known_qpos30"])
    alpha_original = np.cumprod(1 - gd.get_named_beta_schedule("cosine", 1000, 1.0))
    for index in (0, 9, 19):
        result = policy.transition_parameters(conditions, state, index)
        mapped = policy.timestep_map[index]
        alpha = float(alpha_original[mapped])
        alpha_previous = float(alpha_original[policy.timestep_map[index + 1]]) if index < 19 else 1.0
        sigma = 0.1 * np.sqrt((1-alpha_previous)/(1-alpha)) * np.sqrt(1-alpha/alpha_previous)
        x0 = result["pred_x_start"].double()
        expected = np.sqrt(alpha_previous)*x0 + np.sqrt(1-alpha_previous-sigma*sigma) * (
            (state.double()-np.sqrt(alpha)*x0)/np.sqrt(1-alpha))
        mask = result["free_mask"]
        torch.testing.assert_close(result["mean"][mask].double(), expected[mask], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(result["base_std"].double(),
                                   torch.full((1, 1, 1), sigma, dtype=torch.float64), atol=2e-6, rtol=2e-5)
        other = larger_floor.transition_parameters(conditions, state, index)
        torch.testing.assert_close(result["mean"], other["mean"], atol=0, rtol=0)
    assert result["base_std"].item() == 0
    assert result["std"].item() == pytest.approx(0.001)


def test_last_transition_really_samples_noise_and_seed_replays(actor_factory):
    actor, batch = actor_factory(prefix=0)
    policy = DPPODiffusionPolicy(actor, steps=3)
    conditions = _conditions(batch)
    first, second = _sample(policy, conditions), _sample(policy, conditions)
    torch.testing.assert_close(first["chain"], second["chain"], atol=0, rtol=0)
    different = _sample(policy, conditions, seed=292)
    assert not torch.equal(first["chain"], different["chain"])
    residual = ((first["chain"][:, -1] - first["old_means"][:, -1])
                / first["old_stds"][:, -1])[first["free_mask"]]
    assert abs(float(residual.mean())) < 0.05
    assert 0.95 < float(residual.std()) < 1.05
    assert not torch.equal(first["chain"][:, -1], first["old_means"][:, -1])


def test_prefix_padding_and_invalid_placeholder_isolation(actor_factory):
    actor, batch = actor_factory(starts=(45,), prefix=6)
    _activate_branches(actor)
    conditions = _conditions(batch)
    conditions["future_valid"][:, 103:] = False
    conditions["music_valid"][:, 103:] = False
    conditions["proprio_history_valid"][:, :7] = False
    policy = DPPODiffusionPolicy(actor, steps=3)
    first = _sample(policy, conditions)
    known = conditions["known_qpos30_mask"]
    adapted = actor.adapt_conditions(conditions)
    assert known[0, 5, 2:].all() and not known[0, 5, :2].any()
    for state in first["chain"].unbind(1):
        torch.testing.assert_close(state[known], adapted["known_x"][known], rtol=0, atol=0)
        assert torch.count_nonzero(state[:, 103:]) == 0
    torch.testing.assert_close(first["qpos30"][known], conditions["known_qpos30"][known], rtol=0, atol=0)
    modified = copy.deepcopy(conditions)
    modified["known_qpos30"][~known] = 3210
    modified["music_features"][:, 103:] = -800
    modified["proprio_history"][:, :7] = 930
    second = _sample(policy, modified)
    torch.testing.assert_close(first["chain"], second["chain"], rtol=0, atol=0)
    torch.testing.assert_close(first["old_log_probs"], second["old_log_probs"], rtol=0, atol=0)
    poisoned_next = first["chain"][:, 1].clone()
    poisoned_next[~first["free_mask"]] = 1e6
    actual = policy.evaluate_log_probs(conditions, first["chain"][:, 0], poisoned_next, 0)
    torch.testing.assert_close(actual, first["old_log_probs"][:, 0], rtol=0, atol=1e-9)
    # 输入 snapshot 不可随调用者后续改动而改变。
    conditions["proprio_history"].fill_(222)
    assert not torch.equal(first["conditions"]["proprio_history"], conditions["proprio_history"])


def test_eval_dropout_cfg_and_conditions_negative_cases(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    _activate_branches(actor)
    for module in actor.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.6
    actor.music_mask_prob = 0.9
    policy = DPPODiffusionPolicy(actor, steps=3)
    conditions = _conditions(batch)
    actor.train()
    trace = _sample(policy, conditions)
    actor.train()
    actual = policy.evaluate_log_probs(conditions, trace["chain"][:, 0], trace["chain"][:, 1], 0)
    assert not any(module.training for module in actor.modules())
    torch.testing.assert_close(actual, trace["old_log_probs"][:, 0], rtol=0, atol=1e-9)
    for key, amount in (("music_features", 5), ("proprio_history", 3), ("known_qpos30", 0.1)):
        modified = copy.deepcopy(conditions)
        if key == "known_qpos30":
            modified[key][conditions["known_qpos30_mask"]] += amount
        else:
            modified[key] += amount
        changed = policy.evaluate_log_probs(modified, trace["chain"][:, 0], trace["chain"][:, 1], 0)
        assert not torch.allclose(changed, actual, rtol=0, atol=1e-3), key
    for option in ({"guidance_scale": 1.0}, {"eta": 0.3}):
        changed_policy = DPPODiffusionPolicy(actor, steps=3, **option)
        changed = changed_policy.evaluate_log_probs(conditions, trace["chain"][:, 0], trace["chain"][:, 1], 0)
        assert not torch.allclose(changed, actual, rtol=0, atol=1e-3)


def test_cfg_shared_history_prefix_and_guided_x0_formula(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    _activate_branches(actor)
    conditions = _conditions(batch)
    policy = DPPODiffusionPolicy(actor, steps=3, guidance_scale=2.5)
    state = torch.randn_like(conditions["known_qpos30"])
    adapted = actor.adapt_conditions(conditions)
    residual = actor._residual(adapted)
    seen = []
    original = actor._denoise
    def observe(value, time, current, embedding):
        output = original(value, time, current, embedding)
        seen.append((current, embedding, output))
        return output
    actor._denoise = observe
    result = policy.transition_parameters(conditions, state, 0)
    assert len(seen) == 2 and seen[0][0] is seen[1][0]
    torch.testing.assert_close(seen[0][1], actor._music_condition(adapted)+residual)
    torch.testing.assert_close(seen[1][1], actor._music_condition(adapted, torch.ones(1, dtype=torch.bool))+residual)
    expected = actor._constrain(seen[1][2]["pred_x_start"] + 2.5 * (
        seen[0][2]["pred_x_start"]-seen[1][2]["pred_x_start"]), adapted)
    torch.testing.assert_close(result["pred_x_start"], expected)


def test_mixed_step_microbatch_and_parameter_gradient_only(actor_factory):
    actor, batch = actor_factory()
    _activate_branches(actor)
    policy = DPPODiffusionPolicy(actor, steps=3)
    conditions = _conditions(batch)
    trace = _sample(policy, conditions)
    first = torch.stack((trace["chain"][0, 0], trace["chain"][1, 2])).requires_grad_()
    following = torch.stack((trace["chain"][0, 1], trace["chain"][1, 3])).requires_grad_()
    actual = policy.evaluate_log_probs(conditions, first, following, torch.tensor([0, 2]))
    expected = torch.stack((trace["old_log_probs"][0, 0], trace["old_log_probs"][1, 2]))
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=0)
    (-actual.mean()).backward()
    gradients = [p.grad for p in actor.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    assert first.grad is None and following.grad is None
    assert all(p.grad is None for p in actor.denoiser.static_conf_head.parameters())


def test_all_padding_has_zero_joint_probability(actor_factory):
    actor, batch = actor_factory(prefix=0, starts=(45,))
    conditions = _conditions(batch)
    conditions["future_valid"].fill_(False)
    conditions["music_valid"].fill_(False)
    trace = _sample(DPPODiffusionPolicy(actor, steps=3), conditions)
    assert not trace["free_mask"].any()
    assert torch.count_nonzero(trace["chain"]) == 0
    assert torch.count_nonzero(trace["old_log_probs"]) == 0


def test_contract_rejects_targets_bad_state_and_zero_variance(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    policy = DPPODiffusionPolicy(actor, steps=3)
    conditions = _conditions(batch)
    state = torch.randn_like(conditions["known_qpos30"])
    with pytest.raises(ValueError, match="exactly"):
        policy.transition_parameters(batch, state, 0)
    with pytest.raises(TypeError, match="float32"):
        policy.transition_parameters(conditions, state.double(), 0)
    with pytest.raises(ValueError, match="outside"):
        policy.transition_parameters(conditions, state, 3)
    with pytest.raises(ValueError, match="strictly positive"):
        DPPODiffusionPolicy(actor, std_floor=0)
    with pytest.raises(ValueError, match="strictly positive"):
        masked_joint_log_prob(state, state, torch.zeros((1, 1, 1)), torch.ones_like(state, dtype=torch.bool))
