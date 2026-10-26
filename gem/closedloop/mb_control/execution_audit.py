"""真实权重下逐段审计执行梯度，并用主动断链检查验证审计能发现错误。

从明确张量快照执行多决策GENMO→参考→SM→WM。分别检查奖励直接参考导数、SM动作、
WM状态、第二/三决策、Critic输入和原监督梯度。执行状态探针不含BC、Critic或参考直接
奖励；有限差分仍是模型内数值证据，不能替代Newton物理方向验证。故意detach只在
本审计的try/finally范围注入，训练不提供关闭执行梯度的选项。
"""

import torch

from .trainer import frozen_parameters


def state_probe(state):
    # 只依赖执行状态，避开四元数单位模这一恒等量。
    return state[:, 2].mean() + state[:, 7:].square().mean()


def audit_execution_paths(actor, critic, engine, initial, noises, latencies, *,
                          supervised_batch=None, stage1_losses=None, finite_difference=False):
    plans, transitions = [], []
    limit = engine.max_model_steps
    engine.max_model_steps = None  # 只测量实际需要的图；绝不授权越界训练。
    noise_values = [n.detach().clone().requires_grad_(True) for n in noises]
    try:
        with frozen_parameters(critic):
            intervals = engine.rollout(initial, noise_values, latencies, on_plan=plans.append,
                                       on_transition=lambda *values: transitions.append(values))
            if len(plans) < 3 or len(intervals) < 3:
                raise AssertionError("梯度验收需要实际完成三次生成和决策")
            first = plans[0].world_qpos
            first_version = initial.reference.version + 1
            def consumes_first_reference(transition):
                snapshot = transition[0]
                support = snapshot.reference.dependency_support(snapshot.tick)
                indices = ((support - snapshot.reference.lower) // 20).long()
                return bool((snapshot.reference.source_versions[indices] == first_version).any())
            # 提交只更新缓存；安全裕量内控制仍然读取旧点，此时梯度为0是正确行为。
            # 用真正进入本步目标/速度halo的第一份参考版本选择执行梯度探针。
            executed = [t for t in transitions if consumes_first_reference(t)]
            if not executed:
                raise AssertionError("没有合法接管后的执行步骤")
            z, action, state1, reference = executed[0]
            paths = {}
            def inspect(name, output, source):
                gradient = (torch.autograd.grad(output, source, retain_graph=True, allow_unused=True)[0]
                            if output.requires_grad and source.requires_grad else None)
                norm = float(gradient.norm()) if gradient is not None else 0.
                paths[name] = {"norm": norm, "passed": gradient is not None and bool(torch.isfinite(gradient).all()) and norm > 1e-12}
                return gradient
            direct_z, direct_action, direct_state, direct_ref = next(
                t for t in executed if t[0].tick + 12 >= plans[0].ticket.first_mutable_source_tick)
            direct, _ = engine.rewards(direct_state.detach(), direct_action.detach(), direct_z.previous_action.detach(), direct_ref)
            inspect("reference_direct_reward", direct.mean(), first)
            inspect("reference_to_sm_action", action.square().mean(), first)
            pre_influence = [t for t in transitions if t[0].tick < z.tick]
            before_norm = 0.
            if pre_influence:
                before_action = pre_influence[-1][1].square().mean()
                before_gradient = (torch.autograd.grad(before_action, first, retain_graph=True, allow_unused=True)[0]
                                   if before_action.requires_grad else None)
                before_norm = float(before_gradient.norm()) if before_gradient is not None else 0.
            paths["unconsumed_reference_has_no_control_gradient"] = {"norm": before_norm,
                "passed": before_norm == 0., "first_consumption_tick": z.tick}
            inspect("sm_action_to_wm_state", state_probe(state1), action)
            inspect("first_reference_to_second_decision", state_probe(intervals[1].end.state), first)
            inspect("first_reference_to_third_decision", state_probe(intervals[2].end.state), first)
            last = intervals[-1]
            value = critic(last.end)
            inspect("critic_terminal_state_input", value.mean(), last.end.state)
            inspect("critic_terminal_cache_input", value.mean(), last.end.reference.world_qpos)
            execution_loss = state_probe(intervals[1].end.state) + state_probe(last.end.state)
            inspect("state_only_first_reference_no_bc_no_critic", execution_loss, first)
            noise_gradient = inspect("state_only_first_noise", execution_loss, noise_values[0])
            parameters = [p for p in actor.stage1.parameters() if p.requires_grad]
            gradients = torch.autograd.grad(execution_loss, parameters, allow_unused=True)
            finite = all(bool(torch.isfinite(g).all()) for g in gradients if g is not None)
            norm = sum(float(g.detach().double().square().sum()) for g in gradients if g is not None) ** .5
            paths["state_only_genmo_parameters_no_bc_no_critic"] = {"norm": norm, "passed": finite and norm > 1e-12}
            del gradients
        fd = None
        if finite_difference and noise_gradient is not None:
            direction = noise_gradient.detach() / noise_gradient.detach().norm().clamp_min(1e-20)
            epsilon = 1e-3
            values = []
            with torch.no_grad():
                for sign in (-1, 1):
                    perturbed = [noise_values[0].detach() + sign * epsilon * direction,
                                 *[n.detach() for n in noise_values[1:]]]
                    seq = engine.rollout(initial, perturbed, latencies)
                    values.append(float(state_probe(seq[1].end.state) + state_probe(seq[-1].end.state)))
            numeric = (values[1] - values[0]) / (2 * epsilon)
            analytic = float((noise_gradient * direction).sum())
            error = abs(numeric - analytic)
            fd = {"epsilon": epsilon, "autograd": analytic, "finite_difference": numeric,
                  "absolute_error": error, "passed": error <= 5e-4 + .05 * abs(analytic)}
        original_step = engine.adapter.bundle.step_model
        try:
            engine.adapter.bundle.step_model = lambda *args, **kwargs: original_step(*args, **kwargs).detach()
            broken = engine.rollout(initial, [n.detach().requires_grad_(True) for n in noise_values], latencies)
            broken_loss = state_probe(broken[-1].end.state)
            detected = not broken_loss.requires_grad
        finally:
            engine.adapter.bundle.step_model = original_step
        bc_report = {"passed": False, "status": "not_supplied"}
        if supervised_batch is not None and stage1_losses is not None:
            prediction = actor.original_supervision(supervised_batch)
            bc, _ = stage1_losses(supervised_batch, prediction["pred_x_start"], prediction["static_conf_logits"], global_step=595000)
            gs = torch.autograd.grad(bc, [p for p in actor.stage1.parameters() if p.requires_grad], allow_unused=True)
            norm = sum(float(g.detach().double().square().sum()) for g in gs if g is not None) ** .5
            bc_report = {"loss": float(bc.detach()), "gradient_norm": norm,
                         "passed": norm > 0 and all(bool(torch.isfinite(g).all()) for g in gs if g is not None)}
        passed = all(p["passed"] for p in paths.values()) and detected and bc_report["passed"] and (fd is None or fd["passed"])
        return {"paths": paths, "deliberate_wm_detach_detected": detected, "state_only_finite_difference": fd,
                "original_supervision": bc_report, "passed": passed,
                "main_control_steps": sum(x.steps for x in intervals),
                "longest_model_dependency_steps": max(x.event.get("longest_model_dependency_steps", 0) for x in intervals),
                "forecast_branches": intervals[-1].event.get("forecast_branches", []),
                "evidence": "model_derivative_and_negative_control_not_real_physics"}
    finally:
        engine.max_model_steps = limit
