"""验证多卡 DPPO 使用同一个全局目标，而不是各卡独立优化。

本文件在两个 CPU 进程中建立真实 Gloo 通信组，调用生产分布式归约器及 Actor／Critic
更新函数。小型高斯策略保留真实的逐去噪概率、自由动作掩码与解析 KL；对照组用相同
样本执行单进程更新。通过 SGD 避免 Adam 首步的尺度抵消掩盖错误的 world-size 因子，
并覆盖不均匀分片、某个 rank 没有样本、BC 仅计算一次、只用于 BC 的参数和全局未用
参数。额外检查候选学习率搜索所得模型与优化器，以及 KL 按全局样本汇总的结果。

全部进程只使用 CPU；通信文件和结果由 pytest 临时目录统一清理。不加载正式模型、
音乐数据或 GMT，不启动长训；这些数学等价测试不替代服务器 NCCL／GPU 启动验收。
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.trainer import actor_update, analytic_kl, critic_update


class GaussianActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = torch.nn.Linear(1, 1, bias=False)
        self.bc_only = torch.nn.Parameter(torch.tensor(.35))
        self.always_unused = torch.nn.Parameter(torch.tensor(.7))
        self.register_buffer("frozen_stats", torch.tensor([.4, .8]))
        with torch.no_grad():
            self.denoiser.weight.fill_(.2)


class GaussianPolicy:
    steps = 2
    timestep_map = (999, 0)
    kernel_config = {"test": "distributed_scalar_gaussian"}

    def __init__(self):
        self.actor = GaussianActor()

    def transition_parameters(self, context, state, step):
        mean = self.actor.denoiser(context["feature"]).reshape(1, 1, 1) * (step + 1)
        return {"mean": mean.expand_as(state), "std": mean.new_full((1, 1, 1), .7)}

    def evaluate_log_probs(self, context, first, second, step):
        parameters = self.transition_parameters(context, first, step)
        density = torch.distributions.Normal(parameters["mean"].double(), parameters["std"].double())
        mask = context["future_valid"][..., None] & ~context["known_qpos30_mask"]
        return density.log_prob(second.double()).masked_fill(~mask, 0.).sum((-2, -1))


class CountingAnchor:
    def __init__(self):
        self.calls = 0

    def backward(self, actor, weight):
        self.calls += 1
        loss = weight * (actor.bc_only.square() + .3 * actor.denoiser.weight.square().mean())
        loss.backward()
        return {"calls": self.calls, "loss": float(loss.detach())}


class SmallCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = torch.nn.Linear(2, 1)
        self.always_unused = torch.nn.Parameter(torch.tensor(.9))
        with torch.no_grad():
            self.value_head.weight.copy_(torch.tensor([[.2, -.1]]))
            self.value_head.bias.fill_(.05)

    def forward(self, context, remaining_music_seconds):
        inputs = torch.cat((context["feature"], remaining_music_seconds[:, None]), dim=1)
        return self.value_head(inputs).squeeze(-1)


def _rows(policy, count):
    rows = []
    generator = torch.Generator().manual_seed(925)
    for index in range(count):
        known = torch.ones((1, 120, 30), dtype=torch.bool)
        known[:, 0, :2] = False
        context = {"feature": torch.tensor([[.5 + index * .3]]),
                   "future_valid": torch.ones((1, 120), dtype=torch.bool),
                   "known_qpos30_mask": known}
        chain = torch.zeros((3, 120, 30))
        means, stds, probabilities = [], [], []
        with torch.no_grad():
            for step in range(2):
                parameters = policy.transition_parameters(context, chain[step:step + 1], step)
                chain[step + 1] = parameters["mean"][0] + .7 * torch.randn((120, 30), generator=generator)
                means.append(parameters["mean"].clone())
                stds.append(parameters["std"].clone())
                probabilities.append(policy.evaluate_log_probs(context, chain[step:step + 1],
                                                               chain[step + 1:step + 2], step)[0])
        rows.append(SimpleNamespace(context=context, chain=chain, old_log_prob=torch.stack(probabilities),
            free_mask=~known[0], transition_valid=True, metadata={"remaining_music_seconds": float(index + 1),
                "sampler_trace": {"kernel_config": policy.kernel_config,
                    "timestep_map": torch.tensor(policy.timestep_map),
                    "old_means": torch.stack(means, dim=1), "old_stds": torch.stack(stds, dim=1)}}))
    return rows


def _snapshot(model, optimizer, report, **extra):
    return {"state": model.state_dict(), "optimizer": optimizer.state_dict(), "report": report,
            "gradients": {name: None if value.grad is None else value.grad.detach().clone()
                          for name, value in model.named_parameters()}, **extra}


def _run_actor(distributed, *, count=3, use_bc=False, candidates=False):
    policy = GaussianPolicy()
    rows = _rows(policy, count)
    anchor = CountingAnchor() if use_bc else None
    optimizer = torch.optim.SGD(policy.actor.parameters(), lr=.005, momentum=.7, weight_decay=.12)
    report = actor_update(policy, optimizer, rows,
        {"advantages": torch.tensor([1., -.3, .4][:count]), "valid": torch.ones(count, dtype=torch.bool)},
        bc=anchor, bc_weight=.17, grad_clip_norm=2., gamma_denoising=.9,
        learning_rate_candidates=[1e-7, 3e-7, 1e-6] if candidates else None,
        distributed=distributed)
    return _snapshot(policy.actor, optimizer, report,
        kl=analytic_kl(policy, rows, distributed=distributed), bc_calls=0 if anchor is None else anchor.calls)


def _run_critic(distributed, *, batch_size):
    critic = SmallCritic()
    optimizer = torch.optim.SGD(critic.parameters(), lr=.03, momentum=.7, weight_decay=.12)
    report = critic_update(critic, optimizer, _rows(GaussianPolicy(), 3),
        {"returns": torch.tensor([1., -.2, .7])}, steps=3, batch_size=batch_size,
        generator=torch.Generator().manual_seed(913), grad_clip_norm=2., distributed=distributed)
    return _snapshot(critic, optimizer, report)


def _worker(rank, directory):
    import faulthandler
    faulthandler.dump_traceback_later(30, repeat=True)
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        distributed = DistributedCollectives(rank, 2, device="cpu")
        result = {}
        for name, options in ACTOR_CASES.items():
            result[name] = _run_actor(distributed, **options)
        for size in (1, 3):
            result[f"critic_{size}"] = _run_critic(distributed, batch_size=size)
        torch.save(result, directory / f"rank_{rank}.pt")
        # 先让两个rank完成结果落盘，再共同销毁Gloo连接。否则较快的子进程可能
        # 已退出，较慢进程仍在ProcessGroup析构中等待；这与训练数值断言无关。
        # 此屏障仅在成功路径执行，异常仍交给有界spawn传播，不能吞掉失败。
        dist.barrier()
    finally:
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


ACTOR_CASES = {
    "actor_uneven": {"count": 3},
    "actor_bc": {"count": 3, "use_bc": True},
    "actor_empty_rank": {"count": 1, "use_bc": True},
    "actor_candidates": {"count": 3, "use_bc": True, "candidates": True},
}


@pytest.fixture(scope="module")
def distributed_results(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo is unavailable")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        directory = tmp_path_factory.mktemp("distributed_training")
        from tests.closedloop.dppo.test_updater_v2 import _spawn_bounded
        _spawn_bounded(_worker, directory)
        ranks = [torch.load(directory / f"rank_{rank}.pt", weights_only=False) for rank in range(2)]
        single = {name: _run_actor(None, **options) for name, options in ACTOR_CASES.items()}
        single.update({f"critic_{size}": _run_critic(None, batch_size=size) for size in (1, 3)})
        yield single, ranks
    finally:
        torch.set_num_threads(previous)


def _assert_nested_close(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, other in zip(actual, expected):
            _assert_nested_close(value, other)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=2e-7, rel=2e-6)
    else:
        assert actual == expected


@pytest.mark.parametrize("name", ACTOR_CASES)
def test_two_rank_actor_matches_single_global_update(distributed_results, name):
    single, ranks = distributed_results
    expected = single[name]
    for rank, outputs in enumerate(ranks):
        actual = outputs[name]
        for field in ("state", "optimizer", "gradients", "kl"):
            _assert_nested_close(actual[field], expected[field])
        for field in ("ppo_loss", "ppo_only_gradient_norm", "total_gradient_norm", "included_upper_transitions",
                      "excluded_upper_transitions", "clip_fraction", "per_denoising_step"):
            _assert_nested_close(actual["report"][field], expected["report"][field])
        assert actual["bc_calls"] == (expected["bc_calls"] if rank == 0 else 0)
        assert actual["gradients"]["always_unused"] is None
        assert actual["state"]["always_unused"].item() == pytest.approx(.7)
        if name == "actor_candidates":
            assert actual["report"]["lr_calibration"]["selected_lr"] == 1e-6
            assert actual["report"]["lr_calibration"]["attempt_count"] == 3
    _assert_nested_close(ranks[0][name]["state"], ranks[1][name]["state"])


@pytest.mark.parametrize("batch_size", (1, 3))
def test_two_rank_critic_matches_single_global_minibatch(distributed_results, batch_size):
    single, ranks = distributed_results
    name = f"critic_{batch_size}"
    for outputs in ranks:
        _assert_nested_close(outputs[name], single[name])
        assert outputs[name]["gradients"]["always_unused"] is None
