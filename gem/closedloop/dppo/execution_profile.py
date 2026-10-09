"""真实 Actor 的计算微批与 CFG 执行方式预检。

预检不留下权重更新，不放宽已有概率阈值。可选临时Adam诊断会精确恢复原状态。使用固定条件、同一显式随机种子分别生成
标量参考链与候选执行链，验证候选在自己采集的数据上复算概率，也在同一参考链上
检查 CFG 合批的数值变化。每个候选再做一次有梯度的微批前向反传检查显存与有限性。
所有 rank 的报告汇总后选择共同通过的配置，选择结果进入完整 checkpoint，恢复时
不得因为硬件或舍入变化静默选择另一条路径。校准的生成次数由调用方持久预算记录。
"""
from __future__ import annotations

import math
import time

import torch
from .batch_execution import ROW_BMM
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
        limit = min(maximum_microbatch, fixed_execution_shape or maximum_microbatch)
        candidates = (32, 16, 8, 4, 2, 1) if policy.numerical_layout != 'legacy_step_lane' else (4, 2, 1)
        for micro in (size for size in candidates if size <= limit):
            for cfg in (True, False):
                policy.cfg_batch = cfg
                report = dict(microbatch=micro, cfg_batch=cfg, passed=False,
                    execution_batch_size=fixed_execution_shape, numerical_layout=policy.numerical_layout,
                    execution_identity=dict(policy.kernel_config))
                started = time.perf_counter()
                try:
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    trace = chains[cfg]
                    sampled_identity = {key: value for key, value in trace['kernel_config'].items()
                                        if key != 'timestep_map'}
                    if sampled_identity != policy.kernel_config:
                        raise ValueError('Profile must recompute the actual sampled execution identity')
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
                        separate_cross = fixed_execution_shape is not None or policy.numerical_layout == ROW_BMM
                        gate_delta = own_delta if separate_cross else torch.cat((own_delta, base_delta))
                        max_log = float(gate_delta.abs().max())
                        max_ratio = float(torch.expm1(gate_delta).abs().max())
                        report.update(self_consistency=dict(max_logprob_error=float(own_delta.abs().max()),
                            max_ratio_error=float(torch.expm1(own_delta).abs().max()),
                            per_step_logprob_error=own_delta.abs().cpu().tolist()),
                            cross_execution=dict(max_logprob_error=float(base_delta.abs().max()),
                                max_ratio_error=float(torch.expm1(base_delta).abs().max()),
                                per_step_logprob_error=base_delta.abs().cpu().tolist(),
                                layer_max_abs_error=_step_differences(parameters, reference_parameters)),
                            validation_scope='all_denoising_steps_separate_self_and_cross_execution',
                            probability_gate_scope='sampled_execution_identity' if separate_cross else 'self_and_legacy_cross_execution',
                            sampled_execution_identity=sampled_identity,
                            cross_reference_identity=reference['kernel_config'])
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
    choices = [(size, cfg) for size in (32, 16, 8, 4, 2, 1) for cfg in (True, False)]
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
            layouts = {item.get('numerical_layout', 'legacy_step_lane') for item in selected}
            if len(layouts) != 1:
                continue
            layout = next(iter(layouts))
            if required is not None and required.get('numerical_layout', 'legacy_step_lane') != layout:
                continue
            result = dict(microbatch=size, cfg_batch=cfg,
                          probability_tolerances=dict(logprob=1e-4, ratio=1e-3, gaussian=1e-8))
            if layout != 'legacy_step_lane':
                result.update(numerical_layout=layout, execution_contract=layout,
                    sampling_environment_batch=1, learner_effective_microbatch=size,
                    network_rows=size*(2 if cfg else 1), padding_rows=0)
            elif shape is not None:
                result.update(execution_batch_size=shape, execution_contract='fixed_shape_step_lane_single_condition_fp32.v1')
            return result
    raise RuntimeError('No common microbatch/CFG profile passes unchanged probability gates')


def compare_learning_execution(policy, context, *, microbatch=4, actor_lr=5e-9):
    """独立诊断同一新核下标量重算与设备/梯度缓存的所有模块及一次Adam结果。"""
    from collections import defaultdict
    from types import SimpleNamespace
    from .parallel_support import cpu_snapshot
    from .tensor_cache import RolloutTensorCache, ConditionGraphCache
    from .updater_v2 import _parameters
    actor = policy.actor
    saved = cpu_snapshot(actor.state_dict())
    gradients_before = {name: p.grad for name, p in actor.named_parameters()}
    was_training = actor.training
    device = next(actor.parameters()).device
    actor.eval()
    trace = policy.sample_rollout(context, generator=torch.Generator(device=device).manual_seed(8173))
    row = SimpleNamespace(context=trace['conditions'], chain=trace['chain'][0],
        old_log_prob=trace['old_log_probs'][0], free_mask=trace['free_mask'][0],
        next_context=None, transition_valid=True, identity=dict(policy_version=0),
        metadata=dict(sampler_trace=trace, remaining_music_seconds=10., next_remaining_music_seconds=0.))
    results = []
    try:
        for size, cached in ((1, False), (microbatch, True)):
            actor.load_state_dict(saved); actor.zero_grad(set_to_none=True)
            tensor_cache = RolloutTensorCache([row], dict(advantages=torch.ones(1), returns=torch.ones(1)), device) if cached else None
            graph = ConditionGraphCache(policy, tensor_cache) if cached else None
            probabilities = []
            try:
                for start in range(0, policy.steps, size):
                    steps = list(range(start, min(start+size, policy.steps)))
                    parameters, mask = _parameters(policy, [row]*len(steps), steps, device, cache=tensor_cache,
                                                   conditions=graph)
                    observed = torch.stack([row.chain[step+1] for step in steps])
                    values = masked_joint_log_prob(observed, parameters['mean'], parameters['std'], mask)
                    (-values.sum()/policy.steps).backward()
                    probabilities.append(values.detach())
                if graph is not None:
                    graph.backward()
                gradients = {name: p.grad.detach().cpu().clone() for name, p in actor.named_parameters() if p.grad is not None}
                optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=actor_lr, weight_decay=0.)
                norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                results.append(dict(gradients=gradients, weights=cpu_snapshot(actor.state_dict()),
                    optimizer=cpu_snapshot(optimizer.state_dict()), probability=torch.cat(probabilities).cpu(), norm=float(norm)))
                del optimizer
            finally:
                if tensor_cache is not None:
                    tensor_cache.close()
        scalar, batch = results
        groups = defaultdict(lambda: dict(reference_squared=0., difference_squared=0., max_abs=0., parameters=0))
        if scalar['gradients'].keys() != batch['gradients'].keys():
            raise ValueError('Batched gradient cache changed trainable gradient presence')
        for name, reference in scalar['gradients'].items():
            difference = batch['gradients'][name].double()-reference.double()
            group = groups[name.split('.')[0]]
            group['parameters'] += reference.numel()
            group['reference_squared'] += float(reference.double().square().sum())
            group['difference_squared'] += float(difference.square().sum())
            group['max_abs'] = max(group['max_abs'], float(difference.abs().max()))
        for group in groups.values():
            group['relative_l2'] = math.sqrt(group['difference_squared']/max(group['reference_squared'], 1e-300))
        optimizer_error = max(float((item.double()-batch['optimizer']['state'][index][key].double()).abs().max())
            for index, state in scalar['optimizer']['state'].items() for key, item in state.items() if torch.is_tensor(item))
        return dict(scope='same_fixed_shape_kernel_scalar_vs_cached_all_steps_temporary_adam',
            denoising_steps=policy.steps, learning_rate=actor_lr, modules=dict(groups),
            scalar_loss=float(-scalar['probability'].mean()), cached_loss=float(-batch['probability'].mean()),
            max_logprob_difference=float((scalar['probability']-batch['probability']).abs().max()),
            scalar_gradient_norm=scalar['norm'], cached_gradient_norm=batch['norm'],
            max_parameter_difference=max(float((value.double()-batch['weights'][name].double()).abs().max()) for name, value in scalar['weights'].items()),
            max_optimizer_state_difference=optimizer_error)
    finally:
        actor.load_state_dict(saved); actor.train(was_training)
        for name, parameter in actor.named_parameters():
            parameter.grad = gradients_before[name]
