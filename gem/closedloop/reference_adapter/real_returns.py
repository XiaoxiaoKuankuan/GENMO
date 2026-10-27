"""独立于WM环形Replay的真实回报数据及可核实episode目录。

完整MC采集使用独立的固定训练/验证歌曲与真实仿真，直到各首歌曲真实失败或结束。
只保留少量真实环境的稀疏Critic特征和逐步真实奖励，所以长歌起点不会被WM环形
缓冲淘汰；行政截断不生成MC。标签记录起点、长度、策略版本、歌曲和真实失败。
在线PPO的GAE来自当前策略真实连续片段；真正终止不bootstrap，行政片段末端可以
bootstrap，所有值都detach，不让Critic或WM梯度流入Adapter。
"""

import torch
from .vector_control import select_state


def gae_returns(reward, value, next_value, done, gamma=.99, lam=.95):
    advantage = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        continuation = (~done[t]).to(reward)
        delta = reward[t]+gamma*continuation*next_value[t]-value[t]
        carry = delta+gamma*lam*continuation*carry
        advantage[t] = carry
    return advantage.detach(), (advantage+value).detach()


@torch.no_grad()
def complete_real_mc(collector, *, split, version, stride=8, repetitions=2, latent_std=0.):
    bank, control = collector.bank, collector.control
    selected = bank.by_split[split].repeat(repetitions)
    count = len(selected)
    if count > collector.num_envs:
        raise ValueError('MC环境数量不足')
    songs = selected[torch.arange(collector.num_envs, device=bank.device)%count]
    frames = torch.zeros_like(songs)
    # 第一份为固定原始初态，第二份采用真实扰动。不是人为失败标签。
    initial = collector.starting_states(songs, frames, perturb=False)
    if repetitions > 1:
        initial[len(bank.by_split[split]):count] = collector.starting_states(
            songs[len(bank.by_split[split]):count], frames[len(bank.by_split[split]):count], perturb=True)
    collector.reset(songs=songs, frames=frames, initial=initial, perturb=False)
    active = torch.ones(count, device=bank.device, dtype=torch.bool)
    rewards, valid, features, saved_t = [], [], [], []
    failures = torch.zeros_like(active)
    episodes = collector.state['episode'][:count].cpu().clone()
    while bool(active.any()):
        t = len(rewards)
        if t % stride == 0:
            features.append(control.critic_features(select_state(collector.state, slice(0, count))).cpu())
            saved_t.append(t)
        latent = None
        if latent_std:
            mean = control.adapter.net(control.packed(control.fields(collector.state)))
            latent = mean+latent_std*torch.randn_like(mean)
        row = collector.step(auto_reset=True, split=split, perturb=False, latent=latent)
        rewards.append((row['reward'][:count]*active).cpu())
        valid.append(active.cpu())
        failures |= row['failed'][:count] & active
        active &= ~row['done'][:count]
        if len(rewards) % 512 == 0:
            print(f'完整真实MC split={split} steps={len(rewards)} active={int(active.sum())}', flush=True)
    rewards, valid = torch.stack(rewards), torch.stack(valid)
    result = torch.zeros_like(rewards)
    carry = torch.zeros(count)
    for t in reversed(range(len(rewards))):
        carry = rewards[t]+.99*carry
        result[t] = carry
    times = torch.tensor(saved_t)
    keep = valid[times]
    lengths = valid.sum(0)
    xs = torch.stack(features)
    ids = selected.cpu().expand(len(times), -1)
    near = (lengths[None]-times[:, None] <= 16) & failures.cpu()[None]
    dataset = dict(x=xs[keep], y=result[times][keep], near=near[keep], song=ids[keep], split=split)
    metadata = []
    for i, song in enumerate(selected.tolist()):
        metadata.append(dict(task=bank.tasks[song]['id'], song=song, split=split, policy_version=version,
            policy_distribution='latent_gaussian' if latent_std else 'deterministic', latent_std=latent_std,
            episode_id=int(episodes[i]), start_frame=0, length=int(lengths[i]),
            failed=bool(failures[i]), full_song=not bool(failures[i]),
            retained_return_from_reset=True, total_reward=float(rewards[:, i].sum()), mc=float(result[0, i])))
    return dataset, metadata


def dataset_to(dataset, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dataset.items()}


def coverage(dataset, episodes, tasks):
    report = dict(samples=len(dataset['y']), episodes=len(episodes),
        failure_fraction=sum(e['failed'] for e in episodes)/max(1, len(episodes)),
        lengths=[e['length'] for e in episodes], source='independent_complete_real_episodes_not_WM_ring',
        per_song={})
    for song in dataset['song'].unique().tolist():
        labels = dataset['y'][dataset['song'] == song]
        es = [e for e in episodes if e['song'] == song]
        report['per_song'][tasks[song]['id']] = dict(samples=len(labels), mean=float(labels.mean()),
            std=float(labels.std(unbiased=False)), minimum=float(labels.min()), maximum=float(labels.max()),
            lengths=[e['length'] for e in es], failures=sum(e['failed'] for e in es),
            complete_full_songs=sum(e['full_song'] for e in es))
    return report
