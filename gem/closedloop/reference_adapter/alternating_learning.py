"""SuperMimic原式交替训练：真实H8世界模型监督、Adapter BPTT和想象GAE Critic。

本文件只实现用户指定的正式学习循环，不调用Oracle、物理方向门禁、候选缩放或
信任域回滚。世界模型沿真实动作递归八步，每步都按原SM物理归一化监督，损失先
按batch求平均、再对时间求和，保持supermimic/train.py的梯度尺度。WM归一化只
从新采集的真实转移按原RunningNorm更新，保留预训练mean/var/count及其上限。

策略仍为2654维可观测Adapter经十帧参考投影、冻结Actor后产生动作；每一步重算
Adapter/Actor/WM，完整内部计算图通过梯度检查点保留。只执行一次有限Adam更新，
输出变化只报告，不作物理门禁。开启Critic时使用H+1想象输入、更新归一化后重新
计算GAE(lambda)标签，再训练Critic；Critic关闭时不产生价值损失或终端梯度。
不可执行投影在物理发送前拒绝；想象中的孤立数值失败只拒绝对应microbatch，不计
该批梯度及有效样本。全部样本失败由入口有限重试后保存退出，不能无限训练WM。
NaN/Inf等硬故障仍抛出并保存现场。正式修复配置关闭未经真实校准的Critic终端梯度。
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


class PolicyProjectionRejected(RuntimeError):
    """本轮没有任何可行想象样本，未执行Adam；入口记录并按有限预算重试。"""
    def __init__(self, report):
        self.report = report
        super().__init__('本轮全部想象样本的参考投影被拒绝，Adapter未更新')


def _projection_failure(error):
    # 只识别投影器明确携带失败环境mask的异常，不能把OOM/NaN/程序错误吞成样本拒绝。
    return isinstance(error, RuntimeError) and hasattr(error, 'bad_mask')


def _accumulate_projection_statistics(target, source):
    """只累计已分离的GPU标量，统一在更新结束转CPU，避免每个控制步逐字段同步。"""
    for name in ('qp_active_count', 'qp_channel_count', 'sqp_active_count', 'sqp_batch_count'):
        if name in source:
            target[name] = target.get(name, 0)+source[name].detach()
    iterations = source.get('sqp_iterations')
    if iterations is not None and iterations.numel():
        target['sqp_iterations_sum'] = target.get('sqp_iterations_sum', 0)+iterations.detach().sum()
        target['sqp_iterations_count'] = target.get('sqp_iterations_count', 0)+iterations.new_tensor(iterations.numel())
        previous = target.get('sqp_iterations_max')
        target['sqp_iterations_max'] = iterations.max().detach() if previous is None else torch.maximum(previous, iterations.max().detach())
    for name in ('sqp_stationarity_residual64', 'sqp_constraint_residual64', 'sqp_step_residual64',
                 'so3_float64_excess', 'so3_float32_excess'):
        if name in source and source[name].numel():
            value = source[name].detach().amax()
            key = name+'_max'
            target[key] = value if key not in target else torch.maximum(target[key], value)


def _merge_projection_statistics(target, source):
    for name, value in source.items():
        if name.endswith('_max'):
            target[name] = value if name not in target else torch.maximum(target[name], value)
        else:
            target[name] = target.get(name, 0)+value


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


def imagine(control, dynamics, critic, start, horizon, *, terminal_value, gamma=.99, checkpoint_steps=True,
            gradient_diagnostics=False):
    def step(z, proposal):
        with torch.profiler.record_function('stage2_WM_step'):
            state = dynamics.step(z['state'], proposal['action'], z['xi'])
        with torch.profiler.record_function('stage2_reward_and_history'):
            end, reward, _, terms, continuation, _, _ = control.advance(z, proposal, state)
        return end, reward, continuation, terms

    z = start
    alive = z['state'].new_ones(len(z['state']))
    score = alive*0
    features, rewards, continuations, terms_sum = [], [], [], {}
    gradient_probe = None
    projection_statistics = {}
    for t in range(horizon):
        # 活跃SO(3)投影使用torch.func.jacrev，不允许放进checkpoint的saved_tensor
        # hooks。投影原图保留；Actor及WM/奖励各自检查点，既不detach也不缩短H。
        with torch.profiler.record_function('stage2_Adapter_projection_Actor'):
            proposal = control.propose(z, checkpoint_actor=checkpoint_steps)
        if control.projection is not None:
            _accumulate_projection_statistics(projection_statistics, control.projection.last_statistics)
        with torch.no_grad():
            certificate = control.constraint_certificate(z, proposal)
            if certificate is not None and not bool(certificate['feasible'].all()):
                control.last_imagination_failure = dict(state=detach_state(z),
                    certificate={k: v.detach() for k, v in certificate.items()},
                    residual=proposal['residual'].detach())
                error = RuntimeError('想象参考投影不可执行；拒绝该批样本，不降低约束')
                error.bad_mask = ~certificate['feasible']
                error.certificate = certificate
                raise error
        if gradient_diagnostics and t == 0:
            gradient_probe = {key: proposal[key] for key in ('raw_residual', 'residual', 'goal', 'action')}
            for tensor in gradient_probe.values():
                if tensor.requires_grad:
                    tensor.retain_grad()
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
                rewards=torch.stack(rewards), continuation=torch.stack(continuations), gradient_probe=gradient_probe,
                projection_statistics=projection_statistics)


def policy_update(control, dynamics, critic, optimizer, starts, horizon, *, terminal_value,
                  microbatch=32, gamma=.99, diagnostics=False, on_step=None):
    optimizer.zero_grad(set_to_none=True)
    count = len(starts['state'])
    report, features, rewards, continuations = {}, [], [], []
    accepted, rejected, rejection_rows = 0, 0, []
    gradient_square = dict(raw_residual=0., residual=0., goal=0., action=0.)
    projection_gradient_dot = projection_inactive_coordinates = projection_coordinates = 0.
    projection_statistics = {}
    probes = fixed_probes(starts, per_song=1) if diagnostics else None
    with torch.no_grad():
        try:
            before_proposal = control.propose(probes) if diagnostics else None
        except RuntimeError as error:
            if not _projection_failure(error):
                raise
            # 诊断探针失败不能误作Adam失败；真实训练样本继续逐批独立检查。
            before_proposal = None
            report['probe_failure_before'] = str(error)
    params = tuple(control.adapter.parameters())
    before = [p.detach().clone() for p in params]
    # 仅为非有限Adam写入恢复，绝不用于真实物理收益回滚。
    adam_before = copy.deepcopy(optimizer.state_dict())
    with dynamics.adapter_phase(), frozen_parameters(critic):
        for offset in range(0, count, microbatch):
            start = detach_state(select_state(starts, slice(offset, offset+microbatch)))
            try:
                trajectory = imagine(control, dynamics, critic, start, horizon,
                    terminal_value=terminal_value, gamma=gamma, gradient_diagnostics=diagnostics)
            except RuntimeError as error:
                if not _projection_failure(error):
                    raise
                rejected += len(start['state'])
                rejection_rows.append(dict(offset=offset, samples=len(start['state']),
                    infeasible_samples=int(error.bad_mask.sum()), reason=str(error)))
                control.last_imagination_failure = dict(state=detach_state(start),
                    certificate={k: v.detach() for k, v in getattr(error, 'certificate', {}).items()},
                    diagnostic=getattr(error, 'diagnostic', {}), error=str(error))
                continue
            loss = -trajectory['objective'].sum()/count
            if not torch.isfinite(loss):
                raise FloatingPointError('Adapter想象累计目标非有限')
            with torch.profiler.record_function('stage2_BPTT_backward'):
                loss.backward()
            accepted += len(start['state'])
            _merge_projection_statistics(projection_statistics, trajectory['projection_statistics'])
            gp = trajectory['gradient_probe']
            if gp is not None:
                for key, value in gp.items():
                    if value.grad is not None:
                        gradient_square[key] += float(value.grad.detach().square().sum())
                raw_grad, projected_grad = gp['raw_residual'].grad, gp['residual'].grad
                if raw_grad is not None and projected_grad is not None:
                    raw_grad, projected_grad = raw_grad.detach(), projected_grad.detach()
                    projection_gradient_dot += float((raw_grad*projected_grad).sum())
                    nonzero = projected_grad.abs() > 1e-12
                    projection_inactive_coordinates += int(((raw_grad.abs() <= 1e-12) & nonzero).sum())
                    projection_coordinates += int(nonzero.sum())
            for key, value in dict(trajectory['terms'], objective=trajectory['objective'],
                                  terminal=trajectory['terminal'], survival=trajectory['survival']).items():
                report[key] = report.get(key, 0.)+float(value.detach().sum())/count
            if terminal_value:
                features.append(trajectory['features'])
                rewards.append(trajectory['rewards']); continuations.append(trajectory['continuation'])
    if accepted == 0:
        optimizer.zero_grad(set_to_none=True)
        raise PolicyProjectionRejected(dict(accepted_samples=0, rejected_samples=rejected,
            rejected_microbatches=rejection_rows, optimizer_step_completed=False))
    # 仅对实际有效样本取均值；被拒绝批没有backward，不能稀释有效梯度。
    normalization = count/accepted
    if accepted != count:
        for param in params:
            if param.grad is not None:
                param.grad.mul_(normalization)
        report = {k: v*normalization if isinstance(v, (float, int)) else v for k, v in report.items()}
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
                  optimizer_step_completed=True, horizon=horizon, critic_enabled=terminal_value,
                  accepted_samples=accepted, rejected_samples=rejected, rejected_microbatches=rejection_rows)
    if projection_statistics:
        values = {k: float(v) for k, v in projection_statistics.items()}
        values.update(qp_active_fraction=values.get('qp_active_count', 0)/max(values.get('qp_channel_count', 0), 1),
            sqp_active_fraction=values.get('sqp_active_count', 0)/max(values.get('sqp_batch_count', 0), 1),
            sqp_iterations_mean=values.get('sqp_iterations_sum', 0)/max(values.get('sqp_iterations_count', 0), 1))
        report['projection_solver'] = values
    if diagnostics:
        raw_norm, projected_norm = gradient_square['raw_residual']**.5, gradient_square['residual']**.5
        report['projection_gradient'] = dict(
            **{k+'_norm': v**.5*normalization for k, v in gradient_square.items()},
            raw_over_projected_norm=raw_norm/max(projected_norm, 1e-30),
            raw_projected_cosine=projection_gradient_dot/max(raw_norm*projected_norm, 1e-30),
            nonzero_projected_gradient_blocked_fraction=projection_inactive_coordinates/max(projection_coordinates, 1),
            scope='full_objective_initial_imagination_step_all_accepted_samples',
            physical_gain_proven=False)
    if on_step is not None:
        on_step(report)
    # 只有Adam完成才由入口增加计数；探针仅诊断，不将零输出变化当物理门禁。
    if diagnostics and before_proposal is not None:
        try:
            report['update_effect'] = update_effect(control, probes, before_proposal, delta)
        except RuntimeError as error:
            if not _projection_failure(error):
                raise
            report['probe_failure_after'] = str(error)
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
