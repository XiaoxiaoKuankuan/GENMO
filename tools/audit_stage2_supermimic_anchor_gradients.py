"""用已保存的真实首次接管样本验证新短时BPTT执行梯度及检查点数据恢复。

读取真实物理审计产物中的原请求/ForecastContext/noise/anchor/参考覆盖轨迹，重生成
同一20步DDIM计划并检查纯执行状态到GENMO的导数、模型内噪声方向FD、原BC分项。
不运行物理、不更新参数、不预热或伪造Critic质量；此报告仅是新方法数值连通性证据，
正式启动仍必须另有独立短时真实物理方向和真实回报Critic质量。只在服务器2显式运行。
"""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sys.path.insert(0, cfg["supermimic"]["repo"])
    from gem.closedloop.mb_control.application import build_components, SupervisionStream, build_stage1_losses
    from gem.closedloop.mb_control.checkpoint import unpack_value, pack_value
    from gem.closedloop.mb_control.execution_audit import state_probe
    from gem.closedloop.mb_control.trainer import frozen_parameters
    cfg["imagination"].update(mode="physical_anchor_short_bptt", horizon_steps=args.horizon)
    actor, adapter, scheduler, rewards, critic, engine, original = build_components(cfg, "cuda:0")
    payload = torch.load(args.anchor, weights_only=False, map_location="cpu")
    sample = unpack_value(payload["sample"], adapter, "cuda:0")
    sample.validate(args.horizon)
    # 使用同一显式序列化路径再次roundtrip，验证恢复原请求、噪声和真实anchor。
    restored = unpack_value(pack_value(sample), adapter, "cuda:0")
    restored.validate(args.horizon)
    assert torch.equal(restored.generation.request.noise, sample.generation.request.noise)
    assert torch.equal(restored.state.state, sample.state.state)
    before = adapter.bundle.frozen_snapshot()
    interval, plan, noise = engine.rollout_anchor(sample)
    probe = state_probe(interval.end.state)
    params = [p for p in actor.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(probe, [noise, *params], retain_graph=True, allow_unused=True)
    norms = [g.detach().float().square().sum() for g in gradients[1:] if g is not None]
    param_norm = float(torch.stack(norms).sum().sqrt())
    noise_norm = float(gradients[0].detach().norm())
    finite = all(bool(torch.isfinite(g).all()) for g in gradients if g is not None)
    direction = gradients[0].detach()/gradients[0].detach().norm()
    analytic = float((gradients[0].detach()*direction).sum())
    values = []
    with torch.no_grad():
        for sign in (-1, 1):
            end, _, _ = engine.rollout_anchor(sample, noise=noise.detach()+sign*.001*direction)
            values.append(float(state_probe(end.end.state)))
    fd = (values[1]-values[0])/.002
    parity = float((plan.world_qpos.detach()-sample.generation.plan.world_qpos).abs().max())
    state_input = interval.end.state
    cache_input = interval.end.reference.world_qpos
    with frozen_parameters(critic):
        value = critic(interval.end).sum()
        state_grad, cache_grad = torch.autograd.grad(value, (state_input, cache_input), allow_unused=True)
    del interval, plan, gradients
    stream = SupervisionStream(cfg["stage1"]["data_config"], 1, 42, "cuda:0")
    batch = stream.next()
    out = actor.original_supervision(batch)
    loss, _ = build_stage1_losses(actor.stage1, original).to("cuda:0")(
        batch, out["pred_x_start"], out["static_conf_logits"], global_step=595000)
    bc_gradients = torch.autograd.grad(loss, params, allow_unused=True)
    bc_norm = float(torch.stack([g.detach().float().square().sum() for g in bc_gradients if g is not None]).sum().sqrt())
    adapter.bundle.assert_unchanged(before)
    report = {"method": "physical_anchor_short_bptt", "anchor_file": args.anchor, "horizon": args.horizon,
        "plan_regeneration_max_abs": parity, "state_only_genmo_parameter_gradient_norm": param_norm,
        "state_only_noise_gradient_norm": noise_norm, "finite": finite,
        "noise_finite_difference": {"autograd": analytic, "centered_difference": fd, "epsilon": .001,
            "passed": abs(analytic-fd) <= .002+.1*abs(analytic)},
        "critic_terminal_state_gradient": float(state_grad.norm()) if state_grad is not None else 0.,
        "critic_terminal_cache_gradient": float(cache_grad.norm()) if cache_grad is not None else 0.,
        "critic_quality": "not_evaluated_by_this_numerical_probe", "original_bc_loss": float(loss.detach()),
        "original_bc_gradient_norm": bc_norm, "frozen_unchanged": True, "data_roundtrip_passed": True,
        "actor_updates": 0, "frozen_sha256": adapter.bundle.metadata["sha256"],
        "evidence": "real_anchor_numerical_derivatives_not_physical_improvement"}
    report["passed"] = finite and param_norm > 0 and noise_norm > 0 and parity <= 1e-5 and report["noise_finite_difference"]["passed"]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    (output/"anchor_gradients.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
