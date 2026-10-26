"""单GPU独立实验的批量想象、训练集归一化Critic和真实WM局部响应监督。

每个起点只在进入BPTT时detach；展开内部状态、控制历史、残差和音乐记忆保留计算图。
Actor/WM/Critic在Adapter阶段冻结参数，仍可对输入求导。Critic输入按具名字段组使用
训练样本统计，输出在真实回报单位反归一化；不使用验证歌曲拟合统计或网络。
真实成对动作标签只来自训练歌曲的同初态物理推进，切向损失不读取未来真实执行状态。
"""

import copy
import torch
from torch import nn

from supermimic.mdp import unflatten_state
from .critic import calibration, lambda_returns
from .dynamics import frozen_parameters
from .quality import critic_gate
from .vector_control import select_state, detach_state


class GroupNormalizedCritic(nn.Module):
    def __init__(self, dimension, input_spec):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 1))
        self.register_buffer('mean', torch.zeros(dimension))
        self.register_buffer('scale', torch.ones(dimension))
        self.register_buffer('return_mean', torch.zeros(()))
        self.register_buffer('return_scale', torch.ones(()))
        self.register_buffer('normalized', torch.tensor(False))
        self.groups = []
        import math
        cursor = 0
        fields = [(f.name, math.prod(f.shape)) for f in input_spec.fields]
        fields += [('physical_state', 55), ('xi', 25), ('previous_action', 21), ('heading', 9),
                   ('clock', 8), ('fixed_scheduling', 6), ('nominal_current', 57)]
        for name, width in fields:
            self.groups.append((name, cursor, cursor+width)); cursor += width
        if cursor != dimension:
            raise ValueError('Critic字段分组维数不符')

    @torch.no_grad()
    def fit_normalization(self, x, y, *, split):
        if split != 'train':
            raise ValueError('Critic归一化只允许训练集')
        if bool(self.normalized):
            return
        for _, start, end in self.groups:
            values = x[:, start:end]
            std = values.std(0, unbiased=False)
            floor = std.median().clamp_min(.02)*.05
            self.mean[start:end] = values.mean(0)
            self.scale[start:end] = torch.where(std > 1e-6, std.clamp_min(floor), torch.ones_like(std))
        self.return_mean.copy_(y.mean())
        self.return_scale.copy_(y.std(unbiased=False).clamp_min(.1))
        self.normalized.fill_(True)

    def forward(self, x):
        return self.net((x-self.mean)/self.scale).squeeze(-1)*self.return_scale+self.return_mean


class TangentReplay:
    def __init__(self, capacity=65536):
        self.capacity, self.data = capacity, {}
        self.total_pairs = 0

    def add(self, pairs):
        if pairs['origin'] != 'real_paired_training_actions':
            raise ValueError('局部响应只允许真实训练歌曲配对标签')
        count = len(pairs['state'])
        self.total_pairs += count
        for k, v in pairs.items():
            if isinstance(v, torch.Tensor):
                value = v.detach()
                self.data[k] = torch.cat((self.data[k], value))[-self.capacity:] if k in self.data else value[-self.capacity:]

    def sample(self, batch):
        if not self.data or not len(self.data['state']):
            return None
        idx = torch.randint(len(self.data['state']), (batch,), device=self.data['state'].device)
        return {k: v[idx] for k, v in self.data.items()}


def tangent_loss(dynamics, pairs):
    state = pairs['state']
    plus = dynamics.step(state, pairs['plus_action'], pairs['xi'])
    minus = dynamics.step(state, pairs['minus_action'], pairs['xi'])
    model = dynamics.model
    source = unflatten_state(state)
    predicted = model.targets(source, unflatten_state(plus))-model.targets(source, unflatten_state(minus))
    actual = model.targets(source, unflatten_state(pairs['plus_next']))-model.targets(source, unflatten_state(pairs['minus_next']))
    action_size = (pairs['plus_action']-pairs['minus_action']).norm(dim=-1).clamp_min(1e-6)
    scale = torch.maximum(model.out_norm.std(), dynamics.error_floor)
    error = (predicted-actual)/scale/action_size[:, None]
    return error.square().mean(), dict(tangent_target_rms=float((actual/action_size[:, None]).square().mean().sqrt()),
                                       tangent_prediction_rms=float((predicted.detach()/action_size[:, None]).square().mean().sqrt()))


def fit_wm(dynamics, optimizer, windows, pairs, weight):
    if dynamics._graph_active:
        raise RuntimeError('想象图存活时禁止修改WM')
    optimizer.zero_grad(set_to_none=True)
    supervised, report = dynamics.supervised_loss(windows)
    local, details = tangent_loss(dynamics, pairs) if pairs is not None else (supervised*0, {})
    loss = supervised+weight*local
    loss.backward()
    norm = nn.utils.clip_grad_norm_(dynamics.model.net.parameters(), 1., error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    dynamics.model_version += 1
    return {k: float(v) for k, v in report.items()} | details | dict(wm_loss=float(loss.detach()),
        tangent_loss=float(local.detach()), tangent_weight=weight, wm_gradient_norm=float(norm), wm_updates=dynamics.model_version)


def rollout(control, dynamics, critic, start, horizon, *, terminal_value, gamma=.99, intervention=None):
    z = start
    alive = z['state'].new_ones(len(z['state']))
    score = alive*0
    rewards, continuations, features, values, weighted, actions = [], [], [], [], {}, []
    states = []
    execution_score = score*0
    for t in range(horizon):
        x = control.critic_features(z) if terminal_value else z['state'].new_empty(len(z['state']), 0)
        value = critic(x) if terminal_value else alive*0
        proposal = control.propose(z, **(intervention if t == 0 and intervention else {}))
        state = dynamics.step(z['state'], proposal['action'], z['xi'])
        end, reward, raw, terms, cont, done, _ = control.advance(z, proposal, state)
        valid = (z['frame'] < control.bank.ends[z['song']]).to(alive)
        alive = alive*valid
        score = score+gamma**t*alive*reward
        execution_score = execution_score+gamma**t*alive*(raw['track']+terms['music'])
        for k, v in terms.items():
            weighted[k] = weighted.get(k, score*0)+gamma**t*alive*v
        features.append(x); values.append(value); rewards.append(reward); continuations.append(cont*valid)
        actions.append(proposal['action']); states.append(state)
        alive = alive*cont
        z = end
    x = control.critic_features(z) if terminal_value else z['state'].new_empty(len(z['state']), 0)
    value = critic(x) if terminal_value else alive*0
    score = score+gamma**horizon*alive*value
    features.append(x); values.append(value)
    return dict(objective=score, rewards=torch.stack(rewards), continuation=torch.stack(continuations),
        features=torch.stack(features), values=torch.stack(values), weighted=weighted, actions=actions,
        states=states, end=z, terminal_contribution=gamma**horizon*alive*value, execution_objective=execution_score)


def fit_critic(critic, optimizer, dataset, epochs=3, batch=2048):
    x, y = dataset['x'], dataset['y']
    if dataset['split'] != 'train':
        raise ValueError('验证集不能用于Critic优化')
    critic.fit_normalization(x, y, split='train')
    loss_sum, steps = 0., 0
    for _ in range(epochs):
        order = torch.randperm(len(x), device=x.device)
        for ids in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            loss = ((critic(x[ids])-y[ids])/critic.return_scale).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()); steps += 1
    optimizer.zero_grad(set_to_none=True)
    return dict(critic_loss=loss_sum/max(steps, 1), critic_optimizer_steps=steps)


@torch.no_grad()
def validate_critic(critic, dataset, constant, thresholds, train_count, tasks):
    x, y, near, songs = (dataset[k] for k in ('x', 'y', 'near', 'song'))
    predicted = torch.cat([critic(chunk) for chunk in x.split(2048)])
    report = calibration(predicted, y, near, constant_value=float(constant))
    report['per_song'] = {tasks[int(s)]['id']: calibration(predicted[songs == s], y[songs == s], near[songs == s],
                          constant_value=float(constant)) for s in songs.unique()}
    report['gate'] = critic_gate(report, thresholds, train_samples=train_count)
    return report


def adapter_update(control, dynamics, critic, optimizer, starts, horizon, *, terminal_value, microbatch=32):
    parameters = tuple(control.adapter.parameters())
    before = [p.detach().clone() for p in parameters]
    adam = copy.deepcopy(optimizer.state_dict())
    with torch.no_grad():
        old = control.propose(select_state(starts, slice(0, min(32, len(starts['state'])))))
    optimizer.zero_grad(set_to_none=True)
    totals, critic_x, critic_y = {}, [], []
    count = len(starts['state'])
    try:
        with dynamics.adapter_phase(), frozen_parameters(critic):
            for offset in range(0, count, microbatch):
                z = detach_state(select_state(starts, slice(offset, offset+microbatch)))
                trajectory = rollout(control, dynamics, critic, z, horizon, terminal_value=terminal_value)
                loss = -trajectory['objective'].sum()/count
                if offset == 0:
                    execution = trajectory['execution_objective']
                    grads = torch.autograd.grad(execution.mean(), parameters, retain_graph=True, allow_unused=True)
                    totals['execution_gradient_norm'] = float(sum(g.square().sum() for g in grads if g is not None).sqrt())
                    for name, term in trajectory['weighted'].items():
                        grads = torch.autograd.grad(term.mean(), parameters, retain_graph=True, allow_unused=True)
                        totals['gradient_'+name] = float(sum((g.square().sum() for g in grads if g is not None), loss.new_zeros(())).sqrt())
                loss.backward()
                for name, term in trajectory['weighted'].items():
                    totals['value_'+name] = totals.get('value_'+name, 0.)+float(term.detach().sum())/count
                if terminal_value:
                    targets = lambda_returns(trajectory['rewards'], trajectory['continuation'], trajectory['values'])
                    critic_x.append(trajectory['features'][:-1].detach().flatten(0, 1)); critic_y.append(targets.flatten())
        norm = nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if totals.get('execution_gradient_norm', 0.) <= 1e-9:
            raise FloatingPointError('没有执行项梯度，不接受仅正则更新')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('Adapter参数非有限')
        with torch.no_grad():
            new = control.propose(select_state(starts, slice(0, min(32, count))))
            delta = sum((a-b).square().sum() for a, b in zip(parameters, before)).sqrt()
            action_delta = (new['action']-old['action']).square().mean().sqrt()
            reference_delta = (new['goal']-old['goal']).square().mean().sqrt()
            if delta <= 0 or action_delta <= 0 or reference_delta <= 0:
                raise FloatingPointError('参数/参考/实际控制动作未变化')
        totals.update(adapter_gradient_norm=float(norm), parameter_delta=float(delta), action_rms_delta=float(action_delta),
                      goal_rms_delta=float(reference_delta), horizon=horizon,
                      phase='lambda_return_with_calibrated_critic' if terminal_value else 'dense_tracking_no_terminal_value')
        return totals, (dict(x=torch.cat(critic_x), y=torch.cat(critic_y), split='train') if critic_x else None)
    except Exception:
        with torch.no_grad():
            for p, saved in zip(parameters, before):
                p.copy_(saved)
        optimizer.load_state_dict(adam)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
