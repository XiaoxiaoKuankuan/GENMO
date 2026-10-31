"""单GPU进程的只读歌曲/名义参考库，供所有物理环境和想象批次共享。

加载时一次验证固定名义轨迹、原EDGE35、资产SHA和真实导数halo；按50Hz预计算
名义身体FK、音乐预览、固定拍点高斯和拍点关闭窗口。运行时仅用song/frame张量
索引，不扫描整首歌曲，不按环境循环，不把实际执行状态写入名义库。音乐仍使用
600Hz绝对tick、原30Hz EDGE35和现有v2事件公式；预计算不生成未来真实执行状态。
"""

import torch

from supermimic.bridge.reference import build_reference
from supermimic.motion import ReferenceBatch, select_reference
from .application import task_assets


class VectorBank:
    def __init__(self, tasks, engine, device):
        entries, contracts = [], []
        for task in tasks:
            initial, nominal, audio, contract = task_assets(task, engine, device)
            entries.append((task, initial, nominal, audio))
            contracts.append(contract)
        self._initialize(entries, engine.step.runtime.kin, engine.step.rewards.music)
        self.task_contracts = contracts

    @classmethod
    def from_entries(cls, entries, kin, music):
        result = cls.__new__(cls)
        result._initialize(entries, kin, music)
        return result

    @torch.no_grad()
    def _initialize(self, entries, kin, music):
        self.tasks = [entry[0] for entry in entries]
        self.task_contracts = [task.get('generation_contract') for task in self.tasks]
        self.audio = [entry[3] for entry in entries]
        self.nominal = [entry[2] for entry in entries]
        self.initial = torch.cat([entry[1] for entry in entries])
        self.device = self.initial.device
        from .tasks import condition_kind
        self.task_modalities = torch.tensor([int(condition_kind(t) == 'text') for t in self.tasks],
                                           device=self.device, dtype=torch.long)
        self.by_split = {split: torch.tensor([i for i, t in enumerate(self.tasks) if t['split'] == split],
                          device=self.device, dtype=torch.long)
                         for split in ('train', 'validation', 'audit', 'evaluation')}
        arrays = {k: [] for k in ReferenceBatch._fields}
        extra = {k: [] for k in ('preview', 'preview_mask', 'preview_times', 'target', 'audio_valid',
                                  'close_kernel', 'close_inside', 'close_due')}
        offsets, lengths, ends, starts, densities = [], [], [], [], []
        cursor = 0
        for task, initial, nominal, audio in entries:
            if nominal.lower != 0 or audio.start_tick % 12:
                raise ValueError('向量库需要名义0起点及50Hz对齐的绝对音乐起点')
            grid = torch.arange(0, nominal.upper+1, 12, device=self.device)
            reference = build_reference(nominal.pose_at, grid, nominal.lower, nominal.upper, kin)
            n = len(grid)
            if int(grid[-1]) < (audio.control_end_tick//12)*12+132:
                raise ValueError('名义尾部缺少真实第11点，禁止外推')
            offsets.append(cursor); lengths.append(n); ends.append(audio.control_end_tick//12)
            starts.append(audio.start_tick//12); cursor += n
            for name, value in zip(reference._fields, reference):
                arrays[name].append(value[0].detach())
            idx = ((grid-audio.start_tick)//20)[:, None] + torch.arange(16, device=self.device)[None]
            inside = (idx >= 0) & (idx < audio.features.shape[1])
            clipped = idx.clamp(0, audio.features.shape[1]-1)
            valid = audio.valid[0, clipped] & inside
            extra['preview'].append(audio.features[0, clipped]*valid[..., None])
            extra['preview_mask'].append(valid)
            times = (audio.start_tick+idx*20-grid[:, None]).float()/600
            extra['preview_times'].append(times if audio.condition_kind == 'music' else torch.zeros_like(times))
            target, target_valid = audio.target(grid)
            extra['target'].append(target[0]); extra['audio_valid'].append(target_valid[0])
            beats = audio.beat_ticks[0]
            density = len(beats)/max(1, (audio.end_tick-audio.start_tick)//12+1)
            densities.append(max(density, 1e-4))
            kernel = initial.new_zeros(n, 16)
            window = torch.zeros(n, 16, device=self.device, dtype=torch.bool)
            due = torch.zeros(n, device=self.device, dtype=torch.bool)
            if len(beats):
                closed_at = torch.div(beats+music.tolerance_ticks+11, 12, rounding_mode='floor')
                keep = (closed_at >= 0) & (closed_at < n)
                beats, closed_at = beats[keep], closed_at[keep]
                # EDGE30拍点最少相隔20tick，50Hz一个关闭时刻至多包含一个拍点。
                if len(torch.unique(closed_at)) != len(closed_at):
                    raise ValueError('一个50Hz关闭时刻对应多个音乐拍点')
                times = closed_at[:, None]*12 + torch.arange(-15, 1, device=self.device)[None]*12
                mask = (times-beats[:, None]).abs() <= music.tolerance_ticks
                weights = torch.exp(-.5*((times-beats[:, None])/(600*audio.sigma_s))**2)*mask
                kernel[closed_at], window[closed_at], due[closed_at] = weights, mask, True
            extra['close_kernel'].append(kernel); extra['close_inside'].append(window); extra['close_due'].append(due)
        self.reference = ReferenceBatch(*(torch.cat(arrays[k]) for k in ReferenceBatch._fields))
        for name, values in extra.items():
            setattr(self, name, torch.cat(values))
        self.offsets, self.lengths, self.ends, self.starts = (
            torch.tensor(x, dtype=torch.long, device=self.device) for x in (offsets, lengths, ends, starts))
        self.density = self.initial.new_tensor(densities)
        self.offset11 = torch.arange(11, device=self.device)
        self.offset13 = torch.arange(-1, 12, device=self.device)
        self.event_offsets = torch.arange(-17, -1, device=self.device)

    def sample_tasks(self, count, split, *, generator=None):
        """联合任务按模态等额采样，再在该模态内部均匀抽任务；旧单模态路径保持原状。"""
        ids = self.by_split[split]
        groups = [ids[self.task_modalities[ids] == kind] for kind in (0, 1)]
        groups = [group for group in groups if len(group)]
        if not groups:
            raise ValueError(f'{split}没有可采样任务')
        if len(groups) == 1:
            group = groups[0]
            return group[torch.randint(len(group), (count,), device=self.device, generator=generator)]
        if count % 2 and bool(torch.randint(2, (), device=self.device, generator=generator)):
            groups.reverse()
        chunks = []
        for index, group in enumerate(groups):
            n = count//len(groups)+(index < count%len(groups))
            chunks.append(group[torch.randint(len(group), (n,), device=self.device, generator=generator)])
        chosen = torch.cat(chunks)
        return chosen[torch.randperm(count, device=self.device, generator=generator)]

    def indices(self, songs, frames):
        # 尾部mask由控制器负责；这里只裁剪无效音乐历史，不制造有效参考支持。
        return self.offsets[songs] + torch.minimum(frames.clamp_min(0), self.lengths[songs]-1)

    @torch.no_grad()
    def derivative_certificate(self, adapter, beta):
        """只读名义导数+解析残差预算，给出修正后绝对速度/加速度的逐歌曲上界。

        这里使用名义参考而非未来执行状态；不用于训练统计或Actor输入。SO(3)增量
        在0.5rad球内采用1.1的log Lipschitz上界，角加速度用相邻角速度误差的三角
        不等式给出保守硬上界。若名义轨迹超出证明域，明确拒绝而非继续声称有保证。
        """
        from supermimic.rotations import so3_log
        bounds = adapter.bounds.clone()
        if adapter.disable_root_xy:
            bounds[:2] = 0
        residual_v = 2*bounds*beta/.02
        residual_a = 4*bounds*beta**2/.02**2
        angular_v = 1.1*2*.05*beta/.02
        reports = []
        for i, task in enumerate(self.tasks):
            start, length = int(self.offsets[i]), int(self.lengths[i])
            idx = slice(start, start+length)
            r = self.reference
            qd = r.q[idx].diff(dim=0)/.02
            v = r.root_p[idx].diff(dim=0)/.02
            rotations = r.root_R[idx]
            w = so3_log(rotations[1:]@rotations[:-1].transpose(-1, -2))/.02
            if beta < 1 and float(w.norm(dim=-1).max()*.02)+2*.05*beta >= .5:
                raise ValueError('名义根旋转不在参考导数上界的证明域内')
            # frame=0的左姿态被合法钳制为第0点，额外包含这个零速度段的接缝加速度。
            qdd = torch.cat((torch.zeros_like(qd[:1]), qd)).diff(dim=0)/.02
            acceleration = torch.cat((torch.zeros_like(v[:1]), v)).diff(dim=0)/.02
            alpha = torch.cat((torch.zeros_like(w[:1]), w)).diff(dim=0)/.02
            central_alpha = r.root_w[idx].diff(dim=0)/.02
            nominal = dict(qd_abs_max_per_joint=qd.abs().amax(0).tolist(),
                qdd_abs_max_per_joint=qdd.abs().amax(0).tolist(),
                root_speed_max=float(v.norm(dim=-1).max()), root_acceleration_max=float(acceleration.norm(dim=-1).max()),
                root_angular_speed_max=float(w.norm(dim=-1).max()),
                root_angular_acceleration_max=max(float(alpha.norm(dim=-1).max()), float(central_alpha.norm(dim=-1).max())))
            absolute = dict(qd_abs_max_per_joint=(qd.abs().amax(0)+residual_v[6:]).tolist(),
                qdd_abs_max_per_joint=(qdd.abs().amax(0)+residual_a[6:]).tolist(),
                root_speed_max=nominal['root_speed_max']+float(residual_v[:3].norm()),
                root_acceleration_max=nominal['root_acceleration_max']+float(residual_a[:3].norm()),
                root_angular_speed_max=nominal['root_angular_speed_max']+angular_v,
                root_angular_acceleration_max=nominal['root_angular_acceleration_max']+2*angular_v/.02)
            reports.append(dict(task=task['id'], nominal=nominal, corrected_absolute_bounds=absolute))
        return dict(beta=beta, dt=.02, residual_velocity_bounds=residual_v.tolist(),
            residual_acceleration_bounds=residual_a.tolist(), root_angular_velocity_deviation_bound=angular_v,
            root_angular_acceleration_deviation_bound=2*angular_v/.02, songs=reports,
            halo='d11=2*d10-d9; residual_velocity_11=velocity_10; residual_acceleration_11=0',
            seam='stage1=(anchor-(1-beta)*left)/beta; past_committed_anchors_read_only',
            scope='forward_discrete_derivatives; central_Goal_velocities_are_adjacent_forward_averages')

    def ref(self, songs, frames):
        return select_reference(self.reference, self.indices(songs, frames))

    def pose(self, songs, frames):
        idx = self.indices(songs, frames)
        r = self.reference
        return r.root_p[idx], r.root_R[idx], r.q[idx]

    def music_preview(self, songs, frames):
        idx = self.indices(songs, frames)
        return self.preview[idx], self.preview_mask[idx], self.preview_times[idx]

    def memory_push(self, memory, qd, nominal_qd, music):
        values = torch.stack((music.energy(qd), qd.abs().mean(-1), nominal_qd.abs().mean(-1),
                              qd.new_ones(len(qd))), -1)
        return torch.cat((memory[:, 1:], values[:, None]), 1)

    def music_reward(self, memory, songs, frames, music):
        center = frames-2
        idx = self.indices(songs, center)
        event_frames = frames[:, None]+self.event_offsets
        event_idx = self.indices(songs[:, None], event_frames)
        event_valid = memory[..., 3].unfold(1, 5, 1).bool().all(-1)
        event_valid = event_valid & self.audio_valid[event_idx] & (event_frames >= 0)
        events = music.events(memory[..., 0])*event_valid
        p = events[:, -1]
        valid = event_valid[:, -1]
        target = self.target[idx]
        density = self.density[songs]
        precision = p*(1-target)/density
        inside = self.close_inside[idx]
        complete = ((~inside) | event_valid).all(-1) & inside.any(-1)
        # 预计算窗口仍需真实历史支持；reset之后不能把缺失速度填零补分。
        payable = self.close_due[idx] & complete & (center >= 0)
        matched = (events*self.close_kernel[idx]).clamp(max=1-music.epsilon)
        coverage = (1+torch.expm1(torch.log1p(-matched).sum(-1)))*payable/density
        duplicate = torch.relu(matched.sum(-1)-1).square()*payable/density
        reward = -(precision+coverage+music.extra_event_weight*duplicate)*valid
        count = memory[..., 3].sum(-1).clamp_min(1)
        actual = (memory[..., 1]*memory[..., 3]).sum(-1)/count
        nominal = (memory[..., 2]*memory[..., 3]).sum(-1)/count
        activity = (torch.relu(music.activity_fraction*nominal-actual)/nominal.clamp_min(music.activity_floor)).square()
        activity = activity*memory[..., 3].bool().all(-1)*(nominal > music.activity_floor)
        return reward, activity, dict(music_probability=p, music_target=target, music_valid=valid,
            music_precision_cost=precision, music_coverage_cost=coverage, music_duplicate_cost=duplicate,
            music_settled_beats=payable.to(p), activity_actual=actual, activity_nominal=nominal)
