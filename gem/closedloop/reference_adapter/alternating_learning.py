"""SuperMimic原式交替训练：真实H8世界模型监督、Adapter BPTT和想象GAE Critic。

本文件只实现用户指定的正式学习循环，不调用Oracle、物理方向门禁、候选缩放或
信任域回滚。世界模型沿真实动作递归八步，每步都按原SM物理归一化监督，损失先
按batch求平均、再对时间求和，保持supermimic/train.py的梯度尺度。WM归一化只
从新采集的真实转移按原RunningNorm更新，保留预训练mean/var/count及其上限。

策略仍为2654维可观测Adapter经十帧参考投影、冻结Actor后产生动作；每一步重算
Adapter/Actor/WM，完整内部计算图通过梯度检查点保留。只执行一次有限Adam更新，
输出变化只报告，不作物理门禁。开启Critic时使用H+1想象输入、更新归一化后重新
计算GAE(lambda)标签，再训练Critic；Critic关闭时不产生价值损失或终端梯度。
NaN/Inf、不可执行投影和异常仍抛出，由正式入口保存现场并停止对应独立实验。
"""

import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from supermimic.mdp import unflatten_state, normalize_xi
from .critic import lambda_returns
from .dynamics import frozen_parameters
from .vector_control import detach_state, select_state
from .update_diagnostics import fixed_probes, update_effect


def enable_wm_normalization(dynamics):
    """仅独立WM换回原RunningNorm类，逐张量继承预训练统计，不触碰冻结bundle。"""
    from supermimic.networks import RunningNorm
    for name in ('in_norm', 'out_norm'):
        old = getattr(dynamics.model, name)
        norm = RunningNorm(old.mean.numel(), max_count=old.max_count, eps=old.eps).to(old.mean)
        norm.load_state_dict(old.state_dict(), strict=True)
        setattr(dynamics.model, name, norm)


@torch.no_grad()
def update_wm_normalization(dynamics, row):
    if dynamics._graph_active:
        raise RuntimeError('BPTT图存活期间不得更新WM归一化')
    model = dynamics.model
    source = unflatten_state(row['start']['state'])
    target = unflatten_state(row['end']['state'])
    model.in_norm.update(model.inputs(source, row['proposal']['action'], normalize_xi(row['start']['xi'])))
    model.out_norm.update(model.targets(source, target))


def wm_loss(dynamics, windows):
    if windows.get('origin') != 'real' or windows['action'].shape[1] != 8:
        raise ValueError('正式WM仅使用真实连续H8窗口')
    state = windows['state'][:, 0]
    scale = torch.maximum(dynamics.model.out_norm.std(), dynamics.error_floor)
    errors = []
    for t in range(8):
        state = dynamics.step(state, windows['action'][:, t], windows['xi'][:, t])
        error = dynamics.model.state_error(unflatten_state(state), unflatten_state(windows['next_state'][:, t]))
        errors.append((error/scale).square().mean(-1))
    errors = torch.stack(errors, 1)
    mask = windows['mask']
    loss = (errors*mask).sum()/len(state)
    return loss, (errors[:, -1]*mask[:, -1]).sum()/mask[:, -1].sum().clamp_min(1)


def wm_update(dynamics, optimizer, windows):
    if dynamics._graph_active:
        raise RuntimeError('Adapter反传期间不得训练WM')
    optimizer.zero_grad(set_to_none=True)
    loss, last = wm_loss(dynamics, windows)
    if not torch.isfinite(loss):
        raise FloatingPointError('WM H8 loss非有限')
    loss.backward()
    norm = nn.utils.clip_grad_norm_(dynamics.model.net.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.stack([torch.isfinite(p).all() for p in dynamics.model.net.parameters()]).all()):
        raise FloatingPointError('WM Adam后参数非有限')
    dynamics.model_version += 1
    return dict(wm_loss=float(loss.detach()/8), wm_loss_backward=float(loss.detach()),
                wm_error8=float(last.detach()), wm_gradient_norm=float(norm))


def imagine(control, dynamics, critic, start, horizon, *, terminal_value, gamma=.99, checkpoint_steps=True):
    def step(z, proposal):
        state = dynamics.step(z['state'], proposal['action'], z['xi'])
        end, reward, _, terms, continuation, _, _ = control.advance(z, proposal, state)
        return end, reward, continuation, terms

    z = start
    alive = z['state'].new_ones(len(z['state']))
    score = alive*0
    features, rewards, continuations, terms_sum = [], [], [], {}
    for t in range(horizon):
        # 活跃SO(3)投影使用torch.func.jacrev，不允许放进checkpoint的saved_tensor
        # hooks。投影原图保留；Actor及WM/奖励各自检查点，既不detach也不缩短H。
        proposal = control.propose(z, checkpoint_actor=checkpoint_steps)
        with torch.no_grad():
            certificate = control.constraint_certificate(z, proposal)
            if certificate is not None and not bool(certificate['feasible'].all()):
                control.last_imagination_failure = dict(state=detach_state(z),
                    certificate={k: v.detach() for k, v in certificate.items()},
                    residual=proposal['residual'].detach())
                raise RuntimeError('想象参考投影不可执行；不降低约束')
        if terminal_value:
            # 中间Critic特征只供后续MSE，终端特征另外保留完整输入梯度。
            with torch.no_grad():
                features.append(control.critic_features(detach_state(z)))
        end, reward, cont, terms = (checkpoint(step, z, proposal, use_reentrant=False, preserve_rng_state=True)
                                   if checkpoint_steps and torch.is_grad_enabled() else step(z, proposal))
        score = score+gamma**t*alive*reward
        for key, value in terms.items():
            terms_sum[key] = terms_sum.get(key, score*0)+gamma**t*alive*value
        rewards.append(reward.detach()); continuations.append(cont.detach())
        alive = alive*cont
        z = end
    terminal = alive*0
    if terminal_value:
        x = control.critic_features(z)
        terminal = gamma**horizon*alive*critic(x)
        score = score+terminal
        features.append(x.detach())
    return dict(objective=score, terms=terms_sum, terminal=terminal, survival=alive.detach(),
                features=torch.stack(features) if terminal_value else None,
                rewards=torch.stack(rewards), continuation=torch.stack(continuations))


def policy_update(control, dynamics, critic, optimizer, starts, horizon, *, terminal_value,
                  microbatch=32, gamma=.99, diagnostics=False, on_step=None):
    optimizer.zero_grad(set_to_none=True)
    count = len(starts['state'])
    report, features, rewards, continuations = {}, [], [], []
    probes = fixed_probes(starts, per_song=2) if diagnostics else None
    with torch.no_grad():
        before_proposal = control.propose(probes) if diagnostics else None
    params = tuple(control.adapter.parameters())
    before = [p.detach().clone() for p in params]
    # 仅为非有限Adam写入恢复，绝不用于真实物理收益回滚。
    adam_before = copy.deepcopy(optimizer.state_dict())
    with dynamics.adapter_phase(), frozen_parameters(critic):
        for offset in range(0, count, microbatch):
            start = detach_state(select_state(starts, slice(offset, offset+microbatch)))
            trajectory = imagine(control, dynamics, critic, start, horizon,
                terminal_value=terminal_value, gamma=gamma)
            loss = -trajectory['objective'].sum()/count
            if not torch.isfinite(loss):
                raise FloatingPointError('Adapter想象累计目标非有限')
            loss.backward()
            for key, value in dict(trajectory['terms'], objective=trajectory['objective'],
                                  terminal=trajectory['terminal'], survival=trajectory['survival']).items():
                report[key] = report.get(key, 0.)+float(value.detach().sum())/count
            if terminal_value:
                features.append(trajectory['features'])
                rewards.append(trajectory['rewards']); continuations.append(trajectory['continuation'])
    norm = nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    nn.utils.clip_grad_value_(params, .1)  # 原SuperMimic policy_update的第二层数值裁剪。
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in params]).all()):
        with torch.no_grad():
            for p, old in zip(params, before):
                p.copy_(old)
        optimizer.load_state_dict(adam_before)
        raise FloatingPointError('Adapter Adam非有限；恢复最后有限参数和优化器')
    optimizer.zero_grad(set_to_none=True)
    delta = sum((p.detach()-old).square().sum() for p, old in zip(params, before)).sqrt()
    report.update(adapter_gradient_norm=float(norm), parameter_delta=float(delta),
                  optimizer_step_completed=True, horizon=horizon, critic_enabled=terminal_value)
    if on_step is not None:
        on_step(report)
    # 只有Adam完成才由入口增加计数；探针仅诊断，不将零输出变化当物理门禁。
    if diagnostics:
        report['update_effect'] = update_effect(control, probes, before_proposal, delta)
    dataset = (dict(features=torch.cat(features, 1), rewards=torch.cat(rewards, 1),
                    continuation=torch.cat(continuations, 1)) if terminal_value else None)
    return report, dataset


def critic_update(critic, optimizer, trajectory, *, gamma=.99, lam=.95, epochs=5, minibatches=4):
    if trajectory is None:
        return dict(critic_loss=None, critic_optimizer_steps=0)
    x = trajectory['features'].detach()
    critic.norm.update(x)
    with torch.no_grad():
        values = critic(x)
        target = lambda_returns(trajectory['rewards'], trajectory['continuation'], values, gamma, lam)
    xf, yf = x[:-1].flatten(0, 1), target.flatten()
    losses, steps = [], 0
    for _ in range(epochs):
        for ids in torch.randperm(len(xf), device=xf.device).tensor_split(minibatches):
            if not len(ids):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = (critic(xf[ids])-yf[ids]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Critic想象GAE loss非有限')
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach())); steps += 1
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.stack([torch.isfinite(p).all() for p in critic.parameters()]).all()):
        raise FloatingPointError('Critic Adam后参数非有限')
    with torch.no_grad():
        error = critic(xf)-yf
        variance = yf.var(unbiased=False)
    return dict(critic_loss=sum(losses)/len(losses), critic_optimizer_steps=steps,
                critic_target_mean=float(yf.mean()), critic_target_std=float(variance.sqrt()),
                critic_imagined_mse=float(error.square().mean()),
                critic_imagined_ev=float(1-error.var(unbiased=False)/variance) if variance > 1e-12 else None,
                critic_norm_count=float(critic.norm.count), critic_target_source='imagined_GAE_lambda_not_real_MC')
