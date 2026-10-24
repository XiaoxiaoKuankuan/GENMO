"""真实权重的纯张量梯度审计，明确不代表物理方向或训练收益。

使用名义仿真状态构造有限启动缓存，比较零投影Stage1采样、grad/no-grad前向及完整
多决策链。可选对初始噪声作方向有限差分；冻结SM和终点Critic参数但检查输入链。
所有结果仅在显式执行后生成，不运行真实物理、不把测试输入当采集数据。
"""

from dataclasses import replace

import torch

from .contracts import ClosedLoopState, ForecastContext
from .reference_buffer import ReferenceBuffer
from .forecast import forecast_rollout
from .online_conditions import build_conditions
from .returns import actor_return
from .trainer import frozen_parameters


def tensor_fixture(adapter, music, initial):
    xi = initial.new_zeros((1, 25))
    xi[:, 0] = 1.
    prev = initial.new_zeros((1, 21))
    qpos = torch.cat((initial[:, :7], initial[:, 13:34][:, adapter.to_genmo]), -1)
    ticks = torch.arange(60, device=initial.device, dtype=torch.long) * 20
    cache = ReferenceBuffer(ticks, qpos[:, None].expand(-1, 60, -1).clone(), initial.new_zeros((1, 60, 2)),
                            torch.zeros_like(ticks), adapter.runtime.kin, adapter.to_sm)
    current, _, _ = cache.read_current_future(0)
    obs = adapter.runtime.observation(initial, prev, xi, current)
    valid = torch.zeros(1, 50, dtype=torch.bool, device=initial.device)
    valid[:, -1] = True
    history = torch.cat((initial.new_zeros((1, 49, 48)), adapter.proprio(initial, xi)[:, None]), 1)
    return ClosedLoopState(0, initial, xi, prev, obs[:, None].expand(-1, 10, -1).clone(), history, valid,
                           cache, music, 0, music.shape[1] * 20)


def audit_gradients(actor, critic, engine, z, latencies, *, finite_difference=False, epsilon=1e-3):
    before = engine.adapter.bundle.frozen_snapshot()
    ticket = engine.scheduler.reserve(z)
    conditions, _ = build_conditions(z, ticket, actor.stage1.endecoder.codec)
    context = forecast_rollout(z, engine.adapter, ticket.first_influence_tick).context
    noise = torch.randn(1, 60, 30, device=z.state.device)
    actor.eval()
    with torch.no_grad():
        original = actor.stage1.sample(conditions, steps=20, guidance_scale=actor.guidance_scale, noise=noise)
        zero = actor.sample(conditions, context, noise)
    zero_error = float((original["qpos30"] - zero["qpos30"]).abs().max())
    if zero_error > 1e-5:
        raise AssertionError(f"零未来分支不满足原Stage1同噪声输出精度: {zero_error}")
    enabled = actor.activation_checkpointing
    actor.activation_checkpointing = False
    plain = actor.sample_differentiable(conditions, context, noise)
    plain_grad = torch.autograd.grad(plain["qpos30"].square().mean(), actor.future_encoder.output.weight)[0]
    actor.activation_checkpointing = True
    recomputed = actor.sample_differentiable(conditions, context, noise)
    checkpoint_grad = torch.autograd.grad(recomputed["qpos30"].square().mean(), actor.future_encoder.output.weight)[0]
    actor.activation_checkpointing = enabled
    noises = [noise.detach().clone().requires_grad_(True)] + [torch.randn_like(noise) for _ in range(engine.decisions - 1)]
    old_limit = engine.max_model_steps
    engine.max_model_steps = None
    try:
        with frozen_parameters(critic):
            intervals = engine.rollout(z, noises, latencies)
            final = intervals[-1]
            value = torch.zeros_like(final.reward) if final.terminal else critic(final.end)
            objective = actor_return(intervals, value).mean()
            gradient, = torch.autograd.grad(objective, noises[0])
            with torch.no_grad():
                plain_intervals = engine.rollout(z, [n.detach() for n in noises], latencies)
                final0 = plain_intervals[-1]
                value0 = torch.zeros_like(final0.reward) if final0.terminal else critic(final0.end)
                forward_error = float((actor_return(plain_intervals, value0).mean() - objective.detach()).abs())
            fd_report = None
            if finite_difference:
                direction = torch.randn_like(noise)
                direction = direction / direction.norm()
                outputs = []
                for sign in (-1, 1):
                    with torch.no_grad():
                        perturbed = [noises[0].detach() + sign * epsilon * direction, *[n.detach() for n in noises[1:]]]
                        seq = engine.rollout(z, perturbed, latencies)
                        last = seq[-1]
                        boot = torch.zeros_like(last.reward) if last.terminal else critic(last.end)
                        outputs.append(float(actor_return(seq, boot).mean()))
                fd_report = {"autograd_direction": float((gradient * direction).sum()),
                             "model_finite_difference": (outputs[1] - outputs[0]) / (2 * epsilon), "epsilon": epsilon}
        engine.adapter.bundle.assert_unchanged(before)
        return {"evidence": "tensor_autograd_only_not_physical_gradient", "zero_forecast_stage1_max_abs": zero_error,
                "checkpoint_output_max_abs": float((plain["qpos30"] - recomputed["qpos30"]).abs().max()),
                "checkpoint_gradient_max_abs": float((plain_grad - checkpoint_grad).abs().max()),
                "first_noise_gradient_norm": float(gradient.norm()), "first_noise_gradient_finite": bool(torch.isfinite(gradient).all()),
                "forecast_output_gradient_norm": float(plain_grad.norm()), "grad_no_grad_forward_abs": forward_error,
                "completed_decisions": len(intervals), "accepted_commits": sum(x.event.get("accepted", False) for x in intervals),
                "frozen_parameters_and_buffers_unchanged": True, "finite_difference": fd_report}
    finally:
        engine.max_model_steps = old_limit
