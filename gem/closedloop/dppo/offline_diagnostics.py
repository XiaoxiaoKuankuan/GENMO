"""基于已经封存的 rollout 执行显式离线诊断，不运行任何环境或正式训练。

输入必须是已有 RolloutWriter 发布的完整清单和对应 fixed_targets；先核对清单、分块、
转移文件的 SHA／大小／身份，旧概率和目标保持只读。诊断可在独立小 Critic 副本上比较
20/40/80 次更新；每组从同一参数、Adam 状态及索引随机种子开始，不能把当前批拟合误差
当作泛化结论。Actor 的逐去噪步梯度模式只反向、不执行 optimizer.step，结束后恢复
原梯度引用和模块模式。它有额外反向成本，不能隐式加入每轮正式训练。

本文所有链 KL 均沿旧策略采样的中间状态计算；逐步范数是各步对整批 PPO 均值目标的
梯度贡献，不是 KL 占比，也不能直接相加得到合成范数。本模块不改变噪声核、不扫描
学习率、不写模型权重、不消费训练随机数或训练预算。
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch

from .buffer import UpperTransition
from .position_repair import file_sha256
from .policy import masked_joint_log_prob
from .updater_v2 import _parameters, critic_update_local


def _inside(base, relative):
    path = (base / relative).resolve(strict=True)
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Diagnostic input reference escapes its immutable rollout directory')
    return path


def load_immutable_rollouts(manifests, target_paths, *, max_chains=8):
    """读取显式选中的旧链与原 targets，不重算 GAE，不伪造另一套采样数据格式。"""
    if type(max_chains) is not int or not 1 <= max_chains <= 160:
        raise ValueError('Offline diagnostic chain limit must lie in [1,160]')
    if len(manifests) != len(target_paths) or not manifests:
        raise ValueError('Each rollout manifest requires its explicitly paired fixed_targets file')
    rows, targets, references, seen, versions = [], {'advantages': [], 'returns': [], 'valid': []}, [], set(), set()
    available = 0
    for manifest_path, target_path in zip(manifests, target_paths):
        manifest_path, target_path = Path(manifest_path).resolve(strict=True), Path(target_path).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema') != 'genmo.closedloop.stage10.rollout.v1' or manifest.get('complete') is not True:
            raise ValueError('Diagnostic rollout must have a complete publication')
        versions.add(manifest['policy_version'])
        fixed = torch.load(target_path, map_location='cpu', weights_only=False)
        count = manifest['transition_count']
        available += count
        for key in targets:
            if torch.as_tensor(fixed[key]).shape != (count,):
                raise ValueError('Fixed targets must exactly align with their published rollout')
        references.extend([dict(path=str(manifest_path), sha256=file_sha256(manifest_path)),
                           dict(path=str(target_path), sha256=file_sha256(target_path))])
        local_index = 0
        for reference in manifest['chunks']:
            chunk_path = _inside(manifest_path.parent, reference['path'])
            if file_sha256(chunk_path) != reference['sha256']:
                raise ValueError('Diagnostic rollout chunk SHA mismatch')
            chunk = json.loads(chunk_path.read_text())
            if (chunk.get('schema') != 'genmo.closedloop.stage10.rollout_chunk.v1'
                    or chunk.get('policy_version') != manifest['policy_version']
                    or len(chunk['records']) != chunk['record_count'] or chunk['record_count'] != reference['record_count']):
                raise ValueError('Diagnostic rollout chunk identity/count mismatch')
            references.append(dict(path=str(chunk_path), sha256=reference['sha256']))
            for record in chunk['records']:
                if len(rows) < max_chains:
                    path = _inside(manifest_path.parent, str(chunk_path.parent.relative_to(manifest_path.parent) / record['path']))
                    if path.stat().st_size != record['size_bytes'] or file_sha256(path) != record['sha256']:
                        raise ValueError('Diagnostic transition size/SHA mismatch')
                    row = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                    if not isinstance(row, UpperTransition):
                        raise TypeError('Diagnostic input must contain original UpperTransition objects')
                    row.validate()
                    identity = {key: row.identity[key] for key in record['identity']}
                    if identity != record['identity'] or row.identity['policy_version'] != manifest['policy_version']:
                        raise ValueError('Diagnostic transition identity differs from publication')
                    identity_key = tuple(sorted(identity.items()))
                    if identity_key in seen:
                        raise ValueError('Duplicate transition in diagnostic inputs')
                    seen.add(identity_key)
                    if not row.transition_valid or not bool(fixed['valid'][local_index]):
                        raise ValueError('Diagnostic selected an invalid published transition')
                    rows.append(row)
                    for key in targets:
                        targets[key].append(torch.as_tensor(fixed[key])[local_index])
                    references.append(dict(path=str(path), sha256=record['sha256']))
                local_index += 1
        if local_index != count:
            raise ValueError('Rollout manifest total count differs from its chunks')
    if not rows or len(versions) != 1 or len({row.identity['run_id'] for row in rows}) != 1:
        raise ValueError('Offline diagnostic requires one nonempty frozen rollout policy/run')
    targets = {key: torch.stack(values) for key, values in targets.items()}
    if any(not torch.isfinite(targets[key]).all() for key in ('advantages', 'returns')):
        raise FloatingPointError('Nonfinite saved diagnostic targets')
    return rows, targets, dict(inputs=references, available_chains=available, selected_chains=len(rows),
        policy_version=next(iter(versions)), target_binding='explicit_paired_paths_and_recorded_SHA',
        audit_scope='all_manifest_chunks_and_selected_transition_files')


def critic_step_comparison(critic, optimizer_state, transitions, targets, *, steps=(20, 40, 80),
                           batch_size=32, learning_rate=1e-4, seed=42, reference=None):
    """每组从同一只读快照开始，原 Critic／原优化器状态不被训练或回写。"""
    if not steps or any(type(value) is not int or not 1 <= value <= 80 for value in steps):
        raise ValueError('Explicit diagnostic Critic steps must lie in [1,80]')
    reports = []
    for count in steps:
        trial = copy.deepcopy(critic)
        optimizer = torch.optim.AdamW(trial.parameters(), lr=learning_rate, weight_decay=0.)
        if optimizer_state is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        report = critic_update_local(trial, optimizer, transitions, targets, steps=count,
            batch_size=batch_size, generator=torch.Generator().manual_seed(seed))
        diagnostic = None
        if reference is not None:
            from .periodic_monitor import fixed_critic_diagnostic
            diagnostic = fixed_critic_diagnostic(trial, reference)
        reports.append(dict(steps=count, training_batch=report, independent_fixed_reference=diagnostic,
                            learning_rates=[group['lr'] for group in optimizer.param_groups]))
    return dict(cases=reports, same_initial_critic_and_optimizer=True, same_minibatch_seed=seed,
                optimizer_state_source='provided_snapshot' if optimizer_state is not None else 'new_for_each_diagnostic',
                production_state_modified=False)


def denoising_gradient_diagnostic(policy, transitions, targets, *, denoising_microbatch=4,
                                 clip=.01, gamma_denoising=.99, objective_logprob_reduction='joint_sum'):
    """显式逐步反向得到真正梯度贡献；无 optimizer、无采样，结束恢复既有 grad/mode。"""
    from .trainer import _validate_actor_transitions
    if type(denoising_microbatch) is not int or denoising_microbatch < 1:
        raise ValueError('Diagnostic microbatch must be a positive integer')
    if not 0 < clip < 1 or not 0 < gamma_denoising <= 1:
        raise ValueError('Invalid diagnostic PPO parameters')
    if objective_logprob_reduction not in ('joint_sum', 'free_coordinate_mean'):
        raise ValueError('Unknown diagnostic probability objective')
    valid = torch.as_tensor(targets.get('valid', [True] * len(transitions))).bool()
    advantages = torch.as_tensor(targets['advantages']).detach()
    if valid.shape != (len(transitions),) or advantages.shape != valid.shape:
        raise ValueError('Diagnostic advantages/validity do not align with the immutable rollout')
    selected = [(index, row) for index, row in enumerate(transitions) if valid[index] and row.free_mask.any()]
    if not selected or not torch.isfinite(advantages[valid]).all():
        raise ValueError('Gradient diagnostic requires valid free chains and finite fixed advantages')
    _validate_actor_transitions(policy, [row for _, row in selected])
    actor = policy.actor
    device = next(actor.parameters()).device
    gradients = {name: parameter.grad for name, parameter in actor.named_parameters()}
    modes = {name: module.training for name, module in actor.named_modules()}
    results = []
    try:
        actor.eval()
        for step in range(policy.steps):
            actor.zero_grad(set_to_none=True)
            loss_sum = 0.
            for start in range(0, len(selected), denoising_microbatch):
                items = selected[start:start + denoising_microbatch]
                rows = [row for _, row in items]
                steps = [step] * len(rows)
                parameters, mask = _parameters(policy, rows, steps, device)
                observed = torch.stack([row.chain[step + 1] for row in rows]).to(device)
                current = masked_joint_log_prob(observed, parameters['mean'], parameters['std'], mask)
                old = torch.stack([row.old_log_prob[step] for row in rows]).to(device).detach()
                difference = current - old
                if objective_logprob_reduction == 'free_coordinate_mean':
                    difference = difference / mask.sum((-2, -1))
                ratio = difference.exp()
                advantage = torch.stack([advantages[index] for index, _ in items]).to(device) * gamma_denoising ** (policy.steps - 1 - step)
                loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).sum() / (len(selected) * policy.steps)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite offline PPO gradient diagnostic')
                loss.backward()
                loss_sum += float(loss.detach())
            module_squares = {}
            for name, parameter in actor.named_parameters():
                if parameter.grad is None:
                    continue
                square = float(parameter.grad.double().square().sum())
                if not math.isfinite(square):
                    raise FloatingPointError('Nonfinite offline PPO gradient')
                prefix = name.split('.')[0]
                module_squares[prefix] = module_squares.get(prefix, 0.) + square
            results.append(dict(step_index=step, loss_contribution=loss_sum,
                gradient_norm=sum(module_squares.values()) ** .5,
                module_gradient_norms={key: value ** .5 for key, value in module_squares.items()}))
    finally:
        for name, parameter in actor.named_parameters():
            parameter.grad = gradients[name]
        for name, module in actor.named_modules():
            module.training = modes[name]
    return dict(per_denoising_step=results, optimizer_steps=0,
        gradient_scope='each_step_contribution_to_mean_over_all_selected_chains_and_steps',
        individual_norms_are_not_additive=True, objective_logprob_reduction=objective_logprob_reduction)
