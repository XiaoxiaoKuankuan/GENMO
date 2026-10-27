"""C消融：用真实物理回报训练270维参考Adapter，完全切断WM到策略的反向传播。

策略先在未压缩的270维latent上采样独立高斯，再经过Tanh、根XY屏蔽和确定性参考
约束。保存并重用同一个latent计算新旧联合logprob；不对投影后的残差假设可逆密度，
也不对冻结SM动作虚构概率。被关闭的20个根XY分量是确定性零，严格从概率维度排除。
这种潜变量动作策略的PPO比率是精确的，即使后续投影非单射也成立。Critic仅拟合
真实GAE目标；Actor、GENMO和WM都不由PPO更新，训练图不包含真实物理未来状态。
"""

import copy
import math
import torch
from .real_returns import gae_returns
from .vector_control import select_state
from .update_diagnostics import fixed_probes, update_effect


def active_mask(adapter, device):
    mask = torch.ones(10, 27, device=device, dtype=torch.bool)
    if adapter.disable_root_xy:
        mask[:, :2] = False
    return mask.flatten()


def gaussian_logprob(latent, mean, std, mask):
    return (-.5*((latent[..., mask]-mean[..., mask])/std).square()-math.log(std)-.5*math.log(2*math.pi)).sum(-1)


@torch.no_grad()
def collect_on_policy(collector, critic, replay, *, steps, std, gamma=.99, lam=.95):
    control = collector.control
    mask = active_mask(control.adapter, collector.device)
    xs, vs, nvs, rewards, dones, latents, logs, cx = [], [], [], [], [], [], [], []
    probes = fixed_probes(collector.state)
    for _ in range(steps):
        z = collector.state
        x = control.packed(control.fields(z))
        mean = control.adapter.net(x)
        # 探索使用独立的Torch随机流，不消耗任务/初态采样器的随机流；二者均保存恢复。
        latent = mean+std*torch.randn_like(mean)
        latent[:, ~mask] = 0
        log = gaussian_logprob(latent, mean, std, mask)
        features = control.critic_features(z)
        value = critic(features)
        row = collector.step(replay, latent=latent)
        # 真实terminal末态用于形状一致的估值，但由done精确清零bootstrap。
        next_value = critic(control.critic_features(row['end']))
        xs.append(x); vs.append(value); nvs.append(next_value); rewards.append(row['reward'])
        dones.append(row['done']); latents.append(latent); logs.append(log); cx.append(features)
    reward, value, next_value, done = map(torch.stack, (rewards, vs, nvs, dones))
    adv, returns = gae_returns(reward, value, next_value, done, gamma, lam)
    return dict(x=torch.cat(xs), latent=torch.cat(latents), old_logprob=torch.cat(logs),
        advantage=adv.flatten(), target=returns.flatten(), critic_x=torch.cat(cx), probes=probes,
        real_reward_mean=float(reward.mean()), real_transitions=reward.numel(),
        true_terminals=int(done.sum()), horizon=steps, std=std)


def ppo_update(control, optimizer, dataset, *, epochs=1, batch=1024, clip=.2):
    adapter = control.adapter
    before = copy.deepcopy(adapter.state_dict())
    adam = copy.deepcopy(optimizer.state_dict())
    parameters = tuple(adapter.parameters())
    probes = dataset['probes']
    with torch.no_grad():
        old = control.propose(probes)
    mask = active_mask(adapter, dataset['x'].device)
    advantage = dataset['advantage']
    advantage = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
    reports = []
    try:
        if epochs != 1:
            raise ValueError('初次A/B/C预算按一次Adam参数步计数；PPO只能一次全批梯度累积')
        for _ in range(epochs):
            indices = torch.randperm(len(advantage), device=advantage.device)
            optimizer.zero_grad(set_to_none=True)
            for ids in indices.split(batch):
                mean = adapter.net(dataset['x'][ids])
                log = gaussian_logprob(dataset['latent'][ids], mean, dataset['std'], mask)
                logratio = log-dataset['old_logprob'][ids]
                ratio = logratio.exp()
                objective = torch.minimum(ratio*advantage[ids], ratio.clamp(1-clip, 1+clip)*advantage[ids])
                loss = -objective.sum()/len(advantage)
                if not torch.isfinite(loss):
                    raise FloatingPointError('PPO联合概率或目标非有限')
                loss.backward()
                reports.append((float(loss.detach()), float(((ratio-1)-logratio).mean().detach()),
                                float(((ratio-1).abs() > clip).float().mean())))
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if not torch.stack([torch.isfinite(p).all() for p in parameters]).all():
                raise FloatingPointError('PPO参数非有限，回滚最后有效策略及优化器')
        with torch.no_grad():
            delta = sum((v-before[k]).square().sum() for k, v in adapter.state_dict().items()).sqrt()
            report = update_effect(control, probes, old, delta)
        if not report['effective']:
            adapter.load_state_dict(before); optimizer.load_state_dict(adam)
        report.update(phase='real_physics_latent_PPO_no_WM_gradient', likelihood_dimension=int(mask.sum()),
            ppo_minibatches=len(reports), loss=sum(v[0] for v in reports)/len(reports),
            gradient_norm=float(norm), approximate_kl=sum(v[1] for v in reports)/len(reports),
            clip_fraction=sum(v[2] for v in reports)/len(reports), optimizer_steps=1,
            budget_definition='one_full_batch_Adam_step_per_accepted_update')
        return report
    except Exception:
        adapter.load_state_dict(before); optimizer.load_state_dict(adam)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
