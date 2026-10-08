"""真实 Actor 的计算微批与 CFG 执行方式预检。

预检不更新权重，不放宽已有概率阈值。使用固定条件、同一显式随机种子分别生成
标量参考链与候选执行链，验证候选在自己采集的数据上复算概率，也在同一参考链上
检查 CFG 合批的数值变化。每个候选再做一次有梯度的微批前向反传检查显存与有限性。
所有 rank 的报告汇总后选择共同通过的配置，选择结果进入完整 checkpoint，恢复时
不得因为硬件或舍入变化静默选择另一条路径。校准的生成次数由调用方持久预算记录。
"""
from __future__ import annotations

import math
import time

import torch
from .policy import masked_joint_log_prob


def _log_probs(policy, context, trace, microbatch, *, details=False):
    values, parameters = [], []
    for start in range(0, policy.steps, microbatch):
        count = min(microbatch, policy.steps-start)
        batch = {key: value.expand(count, *value.shape[1:]).contiguous() for key, value in context.items()}
        steps = torch.arange(start, start+count, device=trace['chain'].device)
        prepared = policy.prepare_conditions(batch)
        result = policy.transition_parameters(batch, trace['chain'][0, start:start+count], steps,
                                              prepared=prepared, diagnostics=details)
        observed = trace['chain'][0, start+1:start+count+1]
        values.append(masked_joint_log_prob(observed, result['mean'], result['std'], result['free_mask']))
        if details:
            coordinate = torch.distributions.Normal(result['mean'].double(), result['std'].double()).log_prob(observed.double())
            result['coordinate_log_probability'] = coordinate.masked_fill(~result['free_mask'], 0.)
            parameters.append({key: value.detach() for key, value in result.items() if value is not None})
    probability = torch.cat(values)
    if not details:
        return probability
    return probability, {key: torch.cat([item[key] for item in parameters]) for key in parameters[0]}


def _step_differences(candidate, reference):
    """固定输入下逐层误差；掩码为精确比较，浮点量报告每步最大绝对差。"""
    result = {}
    for key in candidate.keys() & reference.keys():
        left, right = candidate[key], reference[key]
        error = (left != right).double() if left.dtype == torch.bool else (left.double()-right.double()).abs()
        result[key] = error.reshape(len(error), -1).amax(1).cpu().tolist()
    return result


def probe_profiles(policy, context, *, maximum_microbatch=4, seed=12345, reserve_generation=None,
                   fixed_execution_shape=None, optimizer_probe=False, actor_lr=5e-9):
    actor = policy.actor
    device = next(actor.parameters()).device
    initial_cfg = policy.cfg_batch
    initial_shape = policy.execution_batch_size
    was_training = actor.training
    prior_gradients = {name: parameter.grad for name, parameter in actor.named_parameters()}
    actor.eval()
    reports, chains = [], {}
    saved_weights = ({name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
                     if optimizer_probe else None)
    try:
        policy.execution_batch_size = None
        policy.cfg_batch = False
        if fixed_execution_shape is not None:
            if reserve_generation is not None:
                reserve_generation()
            reference = policy.sample_rollout(context, generator=torch.Generator(device=device).manual_seed(seed))
            with torch.no_grad():
                _, reference_parameters = _log_probs(policy, context, reference, 1, details=True)
        policy.execution_batch_size = fixed_execution_shape
        for cfg in (False, True):
            policy.cfg_batch = cfg
            if reserve_generation is not None:
                reserve_generation()
            generator = torch.Generator(device=device).manual_seed(seed)
            chains[cfg] = policy.sample_rollout(context, generator=generator)
        if fixed_execution_shape is None:
            reference = chains[False]
            policy.cfg_batch = False
            with torch.no_grad():
                _, reference_parameters = _log_probs(policy, context, reference, 1, details=True)
        for micro in (size for size in (4, 2, 1) if size <= maximum_microbatch):
            for cfg in (True, False):
                policy.cfg_batch = cfg
                report = dict(microbatch=micro, cfg_batch=cfg, passed=False,
                    execution_batch_size=fixed_execution_shape, execution_identity=dict(policy.kernel_config))
                started = time.perf_counter()
                try:
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    trace = chains[cfg]
                    with torch.no_grad():
                        own = _log_probs(policy, context, trace, micro)
                        base, parameters = _log_probs(policy, context, reference, micro, details=True)
                        own_delta = own - trace['old_log_probs'][0]
                        base_delta = base - reference['old_log_probs'][0]
                        density = torch.distributions.Normal(trace['old_means'].double(), trace['old_stds'].double())
                        oracle = density.log_prob(trace['chain'][:, 1:].double())
                        oracle = oracle.masked_fill(~trace['free_mask'][:, None], 0.).sum((-2, -1))
                        oracle_error = float((oracle-trace['old_log_probs']).abs().max())
                        # 新执行身份允许与旧路径有已报告的舍入差异；自身采样/重算门槛保持不变。
                        # 旧执行身份仍保留原来的跨路径门槛，不能用这个分支放行旧checkpoint。
                        gate_delta = own_delta if fixed_execution_shape is not None else torch.cat((own_delta, base_delta))
                        max_log = float(gate_delta.abs().max())
                        max_ratio = float(torch.expm1(gate_delta).abs().max())
                        report.update(self_consistency=dict(max_logprob_error=float(own_delta.abs().max()),
                            max_ratio_error=float(torch.expm1(own_delta).abs().max()),
                            per_step_logprob_error=own_delta.abs().cpu().tolist()),
                            cross_execution=dict(max_logprob_error=float(base_delta.abs().max()),
                                max_ratio_error=float(torch.expm1(base_delta).abs().max()),
                                per_step_logprob_error=base_delta.abs().cpu().tolist(),
                                layer_max_abs_error=_step_differences(parameters, reference_parameters)),
                            validation_scope='all_denoising_steps_separate_self_and_cross_execution')
                    actor.zero_grad(set_to_none=True)
                    losses, training_probabilities = [], []
                    for start in range(0, policy.steps, micro):
                        count = min(micro, policy.steps-start)
                        batch = {key: value.expand(count, *value.shape[1:]).contiguous() for key, value in context.items()}
                        steps = torch.arange(start, start+count, device=device)
                        with torch.enable_grad():
                            prepared = policy.prepare_conditions(batch)
                            probability = policy.evaluate_log_probs(batch, trace['chain'][0, start:start+count],
                                trace['chain'][0, start+1:start+count+1], steps, prepared=prepared)
                            loss = -probability.sum()/policy.steps
                            loss.backward()
                            losses.append(loss.detach())
                            training_probabilities.append(probability.detach())
                    training_delta = torch.cat(training_probabilities)-trace['old_log_probs'][0]
                    max_log = max(max_log, float(training_delta.abs().max()))
                    max_ratio = max(max_ratio, float(torch.expm1(training_delta).abs().max()))
                    gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
                    finite = bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
                    report.update(max_logprob_error=max_log, max_ratio_error=max_ratio,
                                  backward_denoising_steps=policy.steps,
                                  backward_loss=float(torch.stack(losses).sum()),
                                  training_self_consistency=dict(max_logprob_error=float(training_delta.abs().max()),
                                      per_step_logprob_error=training_delta.abs().cpu().tolist()),
                                  independent_gaussian_error=oracle_error, finite_gradients=finite,
                                  peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                                  passed=finite and all(math.isfinite(v) for v in (max_log, max_ratio, oracle_error))
                                         and max_log <= 1e-4 and max_ratio <= 1e-3 and oracle_error <= 1e-8)
                    if optimizer_probe and finite:
                        optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
                                                       lr=actor_lr, weight_decay=0.)
                        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        with torch.no_grad():
                            updated = _log_probs(policy, context, trace, micro)
                        report['optimizer_probe'] = dict(learning_rate=actor_lr, gradient_norm_before_clip=float(norm),
                            loss_before=float(torch.stack(losses).sum()), loss_after=float(-updated.mean()),
                            max_logprob_change=float((updated-own).abs().max()),
                            finite=bool(torch.isfinite(updated).all()), scope='temporary_adam_first_step_negative_logprob_restored')
                        report['passed'] = report['passed'] and report['optimizer_probe']['finite']
                        del optimizer
                        actor.load_state_dict(saved_weights)
                except torch.cuda.OutOfMemoryError as error:
                    report['error'] = f'CUDA memory: {error}'
                    actor.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                if saved_weights is not None:
                    actor.load_state_dict(saved_weights)
                reports.append(report)
                report['probe_wall_seconds'] = time.perf_counter()-started
                actor.zero_grad(set_to_none=True)
        return reports
    finally:
        if saved_weights is not None:
            actor.load_state_dict(saved_weights)
        policy.execution_batch_size = initial_shape
        policy.cfg_batch = initial_cfg
        actor.train(was_training)
        for name, parameter in actor.named_parameters():
            parameter.grad = prior_gradients[name]


def select_profile(reports_by_rank, required=None):
    choices = [(size, cfg) for size in (4, 2, 1) for cfg in (True, False)]
    if required is not None:
        choices = [(required['microbatch'], required['cfg_batch'])]
    for size, cfg in choices:
        if all(any(item['microbatch'] == size and item['cfg_batch'] == cfg and item['passed']
                   for item in reports) for reports in reports_by_rank):
            selected = [next(item for item in reports if item['microbatch'] == size and item['cfg_batch'] == cfg and item['passed'])
                        for reports in reports_by_rank]
            shapes = {item.get('execution_batch_size') for item in selected}
            if len(shapes) != 1:
                continue
            shape = next(iter(shapes))
            if required is not None and required.get('execution_batch_size') != shape:
                continue
            result = dict(microbatch=size, cfg_batch=cfg,
                          probability_tolerances=dict(logprob=1e-4, ratio=1e-3, gaussian=1e-8))
            if shape is not None:
                result.update(execution_batch_size=shape, execution_contract='fixed_shape_step_lane_single_condition_fp32.v1')
            return result
    raise RuntimeError('No common microbatch/CFG profile passes unchanged probability gates')
