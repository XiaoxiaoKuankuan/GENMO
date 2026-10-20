"""当前第二阶段的Stage1装配、因果价值快照、连续GAE与原监督BC共享实现。

Actor从s595000装配并保留固定位置表；Critic价值按对应历史和版本计算，真实终止
不自举，普通片段边界不拼接跨episode或行为版本的GAE。BC使用Stage1原数据和
损失以及独立可恢复RNG。Actor更新唯一入口为full_batch_actor，本模块不再保留
旧Stage9、坐标PPO或其他单卡训练器；资源和物理推进由当前主循环负责。
"""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from gem.closedloop.checkpoint import load_stage1_checkpoint
from .returns import compute_gae
from gem.closedloop.training import build_stage1_actor, build_stage1_losses, batch_to_device
from .performance import measure


from gem.closedloop.common.supervision import load_actor


from gem.closedloop.common.supervision import trainable_actor_parameters


def populate_values(transitions, critic, device, *, critic_version=None, batch_size=1):
    """固定版本价值快照；终止置零，自举条件合批但不跨环境重组GAE。"""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('value batch_size must be positive')
    critic.eval()
    work = []
    for row in transitions:
        work.append((row, 'old_value', row.metadata['critic_context'], row.metadata['remaining_music_seconds']))
        row.next_value = 0.
        if row.next_context is not None and not row.terminated:
            work.append((row, 'next_value', row.metadata['next_critic_context'], row.metadata['next_remaining_music_seconds']))
    with torch.no_grad(), measure('critic.fixed_value_snapshot', gpu=True):
        for start in range(0, len(work), batch_size):
            chunk = work[start:start+batch_size]
            context = {key: torch.cat([item[2][key] for item in chunk]).to(device) for key in chunk[0][2]}
            remaining = torch.tensor([item[3] for item in chunk], device=device)
            values = critic(context, remaining).detach().cpu().tolist()
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError('Nonfinite frozen Critic snapshot')
            for item, value in zip(chunk, values):
                setattr(item[0], item[1], value)
    if critic_version is not None:
        for row in transitions:
            row.metadata['value_snapshot_version'] = critic_version


def fixed_targets(transitions, critic, device, *, gamma_upper=.99, lambda_upper=.95,
                  reuse_values=False, critic_version=None, normalize=False):
    if reuse_values:
        if critic_version is None:
            raise ValueError('Reusing populated values requires an explicit Critic version')
        for row in transitions:
            if row.metadata.get('value_snapshot_version') != critic_version:
                raise ValueError('Stale or missing Critic value snapshot version')
            if not math.isfinite(row.old_value) or not math.isfinite(row.next_value):
                raise FloatingPointError('Nonfinite cached Critic values')
    else:
        populate_values(transitions, critic, device, critic_version=critic_version)
    continuation = [False]*len(transitions)
    for i in range(len(transitions)-1):
        a,b=transitions[i:i+2]
        continuation[i] = (a.transition_valid and b.transition_valid and not a.terminated and not a.truncated
            and all(a.identity[k]==b.identity[k] for k in ('backend_session_id','episode_id','policy_version'))
            and a.identity.get('env_id',0)==b.identity.get('env_id',0)
            and a.control_tick_end==b.control_tick_begin)
    return compute_gae(rewards=[t.rewards for t in transitions],values=[t.old_value for t in transitions],
        next_values=[t.next_value for t in transitions],executed_steps=[t.executed_control_steps for t in transitions],
        bootstrap_mask=[not t.terminated and t.next_context is not None for t in transitions],
        continuation_mask=continuation,valid=[t.transition_valid for t in transitions],
        event_rewards=[t.metadata.get('event_reward',0.) for t in transitions],
        gamma_upper=gamma_upper,lambda_upper=lambda_upper,normalize=normalize)


def _validate_actor_transitions(policy, transitions):
    """拒绝不属于当前随机核的链；不允许用参数版本相同掩盖条件或掩码漂移。"""
    if not transitions:
        raise ValueError('Actor diagnostics require at least one transition')
    for item in transitions:
        if not item.transition_valid:
            raise ValueError('Invalid transitions cannot enter Actor optimization')
        if item.chain.shape != (policy.steps + 1, item.context["future_valid"].shape[1], 30):
            raise ValueError('Rollout chain differs from the configured denoising steps')
        if item.chain.dtype != torch.float32 or item.old_log_prob.dtype != torch.float64:
            raise TypeError('Rollout chain/log probability precision changed')
        if item.old_log_prob.shape != (len(policy.stochastic_steps),):
            raise ValueError('Old log probabilities must match the nineteen stochastic transitions')
        expected = item.context['future_valid'][..., None] & ~item.context['known_qpos30_mask']
        if not torch.equal(item.free_mask.cpu(), expected[0].cpu()):
            raise ValueError('Stored free mask differs from the original conditions')
        trace = item.metadata['sampler_trace']
        if trace['kernel_config'] != policy.kernel_config:
            raise ValueError('Rollout and learner stochastic kernel configuration differs')
        if trace['probability_steps'].tolist() != list(policy.stochastic_steps):
            raise ValueError('Deterministic final step cannot have a probability density')
        if trace['timestep_map'].tolist() != list(policy.timestep_map):
            raise ValueError('Rollout timestep map differs from the learner')


from gem.closedloop.common.supervision import SupervisedAnchor
