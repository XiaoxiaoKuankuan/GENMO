"""百万转移容量的紧凑GPU环形回放，不保存歌曲缓存或逐环境Python对象。

第一维是同步物理步，第二维是环境；每项只保存动态状态张量、歌曲ID、episodeID、
动作和真实终态。窗口按环境列连续索引并校验episode/时间，绝不跨reset。真实终止
时只给保留了完整reset起点的片段计算MC；随机时刻复位是完整的“复位片段回报”，
明确不称完整歌曲。行政截断/回放淘汰的片段不产生MC标签。
"""

import torch

from .vector_control import select_state


class VectorReplay:
    def __init__(self, capacity, num_envs, device, gamma=.99):
        self.slots = max(20, capacity//num_envs)
        self.num_envs, self.capacity = num_envs, self.slots*num_envs
        self.device, self.gamma = device, gamma
        self.data, self.dynamic_keys = {}, ()
        self.cursor = self.size = self.total_vector_steps = 0
        self.completed_segments = self.completed_full_songs = self.head_truncated_terminals = 0

    def __len__(self):
        return self.size*self.num_envs

    @torch.no_grad()
    def add(self, start, action, state_next, reward, continuation, done, failed, version):
        values = {f'z_{k}': v.detach() for k, v in start.items()}
        values.update(next_state=state_next.detach(), applied_action=action.detach(), reward=reward.detach(),
            continuation=continuation.detach(), done=done, failed=failed,
            version=torch.full_like(start['frame'], version),
            vector_time=torch.full_like(start['frame'], self.total_vector_steps),
            mc=reward.new_zeros(len(reward)), mc_valid=torch.zeros_like(done), near_failure=torch.zeros_like(done))
        if not self.data:
            self.dynamic_keys = tuple(start)
            self.data = {k: torch.empty((self.slots, *v.shape), dtype=v.dtype, device=self.device) for k, v in values.items()}
            self.data['vector_time'].fill_(-10**9)
            self.data['z_episode'].fill_(-1)
            self.data['done'].fill_(True)
            self.data['mc_valid'].fill_(False)
        if len(action) != self.num_envs:
            raise ValueError('Replay环境列数改变')
        for k, value in values.items():
            self.data[k][self.cursor].copy_(value)
        written = self.cursor
        self.cursor = (self.cursor+1)%self.slots
        self.size = min(self.size+1, self.slots)
        self.total_vector_steps += 1
        if bool(done.any()):
            self._finish(written, done, failed)

    @torch.no_grad()
    def _finish(self, written, done, failed):
        envs = done.nonzero().flatten()
        times = (torch.arange(self.size, device=self.device)+self.cursor-self.size)%self.slots
        rows, cols = times[:, None], envs[None]
        episode = self.data['z_episode'][written, envs]
        valid = self.data['z_episode'][rows, cols] == episode[None]
        has_start = (valid & (self.data['z_frame'][rows, cols] == self.data['z_start_frame'][rows, cols])).any(0)
        self.completed_segments += int(has_start.sum())
        self.head_truncated_terminals += int((~has_start).sum())
        full_song = self.data['z_start_frame'][written, envs] == 0
        self.completed_full_songs += int((has_start & full_song & ~failed[envs]).sum())
        valid &= has_start[None]
        rewards = self.data['reward'][rows, cols].double()*valid
        # 双精度幂避免长片段的指数缩放损失精度；只有真实done列进入这里。
        powers = torch.pow(torch.tensor(self.gamma, device=self.device, dtype=torch.float64),
                           torch.arange(self.size, device=self.device, dtype=torch.float64))[:, None]
        returns = torch.flip(torch.cumsum(torch.flip(rewards*powers, (0,)), 0), (0,))/powers
        existing = self.data['mc'][rows, cols]
        self.data['mc'][rows, cols] = torch.where(valid, returns.float(), existing)
        self.data['mc_valid'][rows, cols] |= valid
        near = (self.data['z_frame'][written, envs][None]-self.data['z_frame'][rows, cols]) < 16
        self.data['near_failure'][rows, cols] |= valid & near & failed[envs][None]

    def _random(self, count):
        if not self.size:
            raise ValueError('真实回放为空')
        rank = torch.randint(self.size, (count,), device=self.device)
        row = (self.cursor-self.size+rank)%self.slots
        env = torch.randint(self.num_envs, (count,), device=self.device)
        return row, env

    def starts(self, batch, *, horizon=None, ends=None):
        if horizon is not None:
            if ends is None:
                raise ValueError('长BPTT起点必须核对实际名义支持')
            rows, envs, remaining = [], [], batch
            for _ in range(20):
                row, env = self._random(max(remaining*2, 64))
                valid = self.data['z_frame'][row, env]+horizon < ends[self.data['z_song'][row, env]]
                ids = valid.nonzero().flatten()[:remaining]
                rows.append(row[ids]); envs.append(env[ids]); remaining -= len(ids)
                if remaining == 0:
                    break
            if remaining:
                raise ValueError(f'缺少实际支持H{horizon}的真实起点，不能默认退回H8')
            row, env = torch.cat(rows), torch.cat(envs)
            return {k: self.data['z_'+k][row, env] for k in self.dynamic_keys}
        row, env = self._random(batch)
        return {k: self.data['z_'+k][row, env] for k in self.dynamic_keys}

    def windows(self, batch, length=8):
        selected_rows, selected_envs = [], []
        remaining = batch
        for _ in range(20):
            row, env = self._random(max(remaining*3, 128))
            indices = (row[:, None]+torch.arange(length, device=self.device)[None])%self.slots
            column = env[:, None]
            episodes = self.data['z_episode'][indices, column]
            times = self.data['vector_time'][indices, column]
            valid = (episodes == episodes[:, :1]).all(1)
            valid &= (times == times[:, :1]+torch.arange(length, device=self.device)[None]).all(1)
            valid &= (times >= max(0, self.total_vector_steps-self.size)).all(1)
            valid &= (times < self.total_vector_steps).all(1)
            valid &= ~self.data['done'][indices[:, :-1], column].any(1)
            good = valid.nonzero().flatten()[:remaining]
            selected_rows.append(indices[good]); selected_envs.append(env[good])
            remaining -= len(good)
            if remaining == 0:
                break
        if remaining:
            raise ValueError(f'缺少连续真实{length}步窗口，尚缺{remaining}')
        rows, envs = torch.cat(selected_rows), torch.cat(selected_envs)[:, None]
        return dict(origin='real', state=self.data['z_state'][rows, envs], action=self.data['applied_action'][rows, envs],
            xi=self.data['z_xi'][rows, envs], next_state=self.data['next_state'][rows, envs],
            mask=torch.ones(batch, length, device=self.device))

    def mc_samples(self, count, allowed_songs=None):
        valid = self.data['mc_valid'][:self.size]
        if allowed_songs is not None:
            valid = valid & torch.isin(self.data['z_song'][:self.size], allowed_songs)
        pairs = valid.nonzero()
        if not len(pairs):
            return None
        choice = torch.randperm(len(pairs), device=self.device)[:count]
        row, env = pairs[choice].unbind(-1)
        z = {k: self.data['z_'+k][row, env] for k in self.dynamic_keys}
        return z, self.data['mc'][row, env], self.data['near_failure'][row, env]

    def state_dict(self):
        return dict(slots=self.slots, num_envs=self.num_envs, gamma=self.gamma, cursor=self.cursor,
            size=self.size, total_vector_steps=self.total_vector_steps, dynamic_keys=self.dynamic_keys,
            completed_segments=self.completed_segments, completed_full_songs=self.completed_full_songs,
            head_truncated_terminals=self.head_truncated_terminals,
            data={k: v[:self.size].detach().cpu() for k, v in self.data.items()})

    def load_state_dict(self, state):
        if state['slots'] != self.slots or state['num_envs'] != self.num_envs:
            raise ValueError('完整Replay恢复需要相同容量和环境数')
        for key in ('gamma', 'cursor', 'size', 'total_vector_steps', 'dynamic_keys', 'completed_segments',
                    'completed_full_songs', 'head_truncated_terminals'):
            setattr(self, key, state[key])
        self.data = {}
        for k, value in state['data'].items():
            self.data[k] = torch.empty((self.slots, *value.shape[1:]), dtype=value.dtype, device=self.device)
            self.data[k][:self.size].copy_(value)
