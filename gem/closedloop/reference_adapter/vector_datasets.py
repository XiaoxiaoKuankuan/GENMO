"""将已保存的真实零Adapter完整轨迹转换为紧凑Critic预热数据。

只导入真实回放标签，不导入旧Critic或旧WM参数；按当前1e-4音乐权重重新计算真实
折扣回报。歌曲归属来自本次固定bank，验证/审计歌曲从不进入训练数据或归一化。
转换仅在启动时执行，旧长曲对象随后释放；在线Replay仍由各实验独立采集。
"""

import torch

from .vector_control import select_state


@torch.no_grad()
def features_in_chunks(control, z, chunk=2048):
    return torch.cat([control.critic_features(select_state(z, slice(i, i+chunk)))
                      for i in range(0, len(z['state']), chunk)])


@torch.no_grad()
def import_zero_mc(path, control):
    ck = torch.load(path, weights_only=False, map_location='cpu')
    replay = ck['replay']
    if not torch.equal(ck['music_s0'].cpu(), control.music.s0.cpu()):
        raise ValueError('导入回放音乐标尺与当前训练基线不一致')
    frozen = control.bundle.state_dict()
    if any(not torch.equal(v.detach().cpu(), ck['frozen_bundle'][k].cpu()) for k, v in frozen.items()):
        raise ValueError('旧真实回放的冻结执行模型与本次资产不一致')
    by_song = {t['song_id']: i for i, t in enumerate(control.bank.tasks)}
    data = {split: {k: [] for k in ('x', 'y', 'near', 'song')} for split in ('train', 'validation')}
    for episode_id, rows in replay['episodes'].items():
        if not rows:
            continue
        meta = replay['episode_metadata'][episode_id]
        if not meta['true_start'] or meta['head_truncated'] or not meta['true_terminal']:
            continue
        song = by_song[rows[0].start.song_id]
        split = control.bank.tasks[song]['split']
        if split not in data or rows[0].start.split != split:
            raise ValueError('旧真实回放划分与固定任务不一致')
        if any(r.policy_version != 0 or (r.start.previous_plan is not None and bool(r.start.previous_plan.residual.any())) for r in rows):
            raise ValueError('预热导入仅接受零Adapter固定基线，禁止混入在线其他实验策略')
        cat = lambda fn: torch.cat([fn(r) for r in rows]).to(control.bank.device)
        count = len(rows)
        z = dict(state=cat(lambda r:r.start.executed_state), xi=cat(lambda r:r.start.xi),
            action=cat(lambda r:r.start.previous_action), history=cat(lambda r:r.start.control_history),
            heading=cat(lambda r:r.start.heading), music=cat(lambda r:r.start.music_state.features()),
            song=torch.full((count,), song, dtype=torch.long, device=control.bank.device),
            frame=torch.tensor([r.start.tick//12 for r in rows], device=control.bank.device),
            start_frame=torch.full((count,), rows[0].start.episode_start_tick//12, device=control.bank.device),
            episode=torch.zeros(count, dtype=torch.long, device=control.bank.device),
            plan=torch.zeros(count, 10, 27, device=control.bank.device),
            left=torch.zeros(count, 27, device=control.bank.device), anchor=torch.zeros(count, 27, device=control.bank.device),
            plan_valid=torch.tensor([r.start.previous_plan is not None for r in rows], device=control.bank.device))
        rewards = cat(lambda r: r.raw['sm']+.02*(control.weights['music']*r.raw['music']-
            sum(control.weights[k]*r.raw[k] for k in ('magnitude', 'smooth', 'overlap', 'reference_limit', 'activity'))))
        # 旧完整轨迹仅在真实终止处截断，不给终点bootstrap。
        powers = .99**torch.arange(len(rewards), dtype=torch.float64, device=rewards.device)
        y = (torch.flip(torch.cumsum(torch.flip(rewards.double()*powers, (0,)), 0), (0,))/powers).float()
        near = torch.zeros(count, dtype=torch.bool, device=control.bank.device)
        if rows[-1].reason == 'physical_failure':
            near[-16:] = True
        data[split]['x'].append(features_in_chunks(control, z))
        data[split]['y'].append(y); data[split]['near'].append(near); data[split]['song'].append(z['song'])
    del ck
    return {split: dict(split=split, **{k: torch.cat(v) for k, v in values.items()}) for split, values in data.items()}


@torch.no_grad()
def replay_mc_dataset(replay, control, split, count=16384):
    values = replay.mc_samples(count, control.bank.by_split[split])
    if values is None:
        return None
    z, y, near = values
    return dict(split=split, x=features_in_chunks(control, z), y=y, near=near, song=z['song'])


def combine_mc(base, extra, max_base=32768):
    if extra is None:
        return base
    if base['split'] != extra['split']:
        raise ValueError('训练与验证MC不得混合')
    return dict(split=base['split'], **{k: torch.cat((base[k][:max_base], extra[k])) for k in ('x', 'y', 'near', 'song')})
