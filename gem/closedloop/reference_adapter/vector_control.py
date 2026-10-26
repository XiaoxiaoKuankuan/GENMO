"""多环境共用的张量化控制与想象单步，保持原2654维可观测输入合同。

动态状态是批量张量字典，歌曲和绝对50Hz帧分别逐环境保存；只读名义库不进入状态。
缓存已提交锚点/上一十帧残差，在固定名义模式下直接复用其相对坐标，避免反复
矩阵与四元数互转。Goal仅依赖根和关节，修正参考不计算身体FK；原名义Tracking
通过权威SM MDP计算一次实际状态FK。所有函数保留输入梯度，真实采集外层才no_grad。
"""

import torch

from supermimic.mdp import unflatten_state, normalize_xi
from supermimic.motion import ReferenceBatch, select_reference, goal_from_reference
from supermimic.rotations import yaw_mat, mat_to_6d, so3_exp, so3_log


def select_state(z, index):
    return {k: v[index] for k, v in z.items()}


def detach_state(z):
    return {k: v.detach() for k, v in z.items()}


def base_reference(p, R, v, w, q, qd):
    """只构造Goal/观测实际读取的字段，空身体张量不会进入Tracking。"""
    empty = p.new_empty(*p.shape[:-1], 0, 3)
    empty_R = p.new_empty(*p.shape[:-1], 0, 3, 3)
    Rt = R.transpose(-1, -2)
    return ReferenceBatch(p, R, v, w, q, qd, empty, empty_R, empty, empty,
        (Rt@v[..., None]).squeeze(-1), (Rt@w[..., None]).squeeze(-1), R[..., 2, 2])


class VectorControl:
    def __init__(self, engine, bank):
        self.engine, self.bank = engine, bank
        self.adapter, self.bundle, self.runtime = engine.step.adapter, engine.step.bundle, engine.step.runtime
        self.music, self.weights = engine.step.rewards.music, engine.step.rewards.weights
        self.spec = self.adapter.input_spec

    def initial(self, observed, songs, frames, episodes):
        state, xi = observed['state'], observed['xi']
        n = len(state)
        reference = self.bank.ref(songs, frames)
        previous = state.new_zeros(n, 21)
        obs = self.runtime.observation(state, previous, xi, reference)
        if self.engine.step.observation_noise:
            obs = self.runtime.mdp.add_noise(obs)
        memory = state.new_zeros(n, 20, 4)
        memory = self.bank.memory_push(memory, state[:, 34:], reference.qd, self.music)
        return dict(state=state, xi=xi, action=previous, history=obs[:, None].expand(-1, 10, -1).clone(),
            song=songs, frame=frames, start_frame=frames.clone(), episode=episodes,
            heading=yaw_mat(reference.root_R), plan=state.new_zeros(n, 10, 27),
            left=state.new_zeros(n, 27), anchor=state.new_zeros(n, 27),
            plan_valid=torch.zeros(n, device=state.device, dtype=torch.bool), music=memory)

    def nominal_goal(self, z):
        frames = z['frame'][:, None]+self.bank.offset11
        idx = self.bank.indices(z['song'][:, None], frames)
        r = self.bank.reference
        ref = base_reference(r.root_p[idx], r.root_R[idx], r.root_v[idx], r.root_w[idx], r.q[idx], r.qd[idx])
        current = select_reference(ref, (slice(None), 0))
        future = select_reference(ref, (slice(None), slice(1, None)))
        return current, goal_from_reference(current, future)

    def fields(self, z, current=None, goal=None):
        if current is None:
            current, goal = self.nominal_goal(z)
        _, R, _, _, q, qd = unflatten_state(z['state'])
        music, mask, times = self.bank.music_preview(z['song'], z['frame'])
        valid = z['plan_valid'][:, None]
        normalized = z['plan']/self.adapter.bounds
        plan_times = torch.arange(10, device=q.device, dtype=q.dtype)[None]*.02*valid
        fields = dict(control_history=self.bundle.actor.obs_norm(z['history']),
            nominal_goal=self.bundle.actor.goal_norm(goal),
            observable_error=torch.cat((current.q-q, (current.qd-qd)*.1,
                                        mat_to_6d(R.transpose(-1, -2)@current.root_R)), -1),
            previous_plan=normalized, previous_corrected_plan=normalized,
            plan_times=plan_times, plan_valid=valid.expand(-1, 10).to(q),
            committed_anchors=torch.stack((z['left'], z['anchor']), 1)/self.adapter.bounds,
            music=music, music_valid=mask.to(q), music_times=times, music_memory=z['music'])
        return fields

    def packed(self, fields):
        # 顺序仍取权威InputSpec，内部固定张量形状不重复逐字段GPU->CPU同步。
        return torch.cat([fields[f.name].flatten(1) for f in self.spec.fields], -1)

    def reference(self, z, residual):
        frames = (z['frame'][:, None]+self.bank.offset13).clamp_min(0)
        p, R, q = self.bank.pose(z['song'][:, None], frames)
        halo = 2*residual[:, -1:]-residual[:, -2:-1]
        d = torch.cat((z['left'][:, None], z['anchor'][:, None], residual, halo), 1)
        p = p+(z['heading'][:, None]@d[..., :3, None]).squeeze(-1)
        R = R@so3_exp(d[..., 3:6])
        q = q+d[..., 6:]
        dt = ((frames[:, 2:]-frames[:, :-2]).to(p)*.02)[..., None]
        v, qd = (p[:, 2:]-p[:, :-2])/dt, (q[:, 2:]-q[:, :-2])/dt
        w = so3_log(R[:, 2:]@R[:, :-2].transpose(-1, -2))/dt
        ref = base_reference(p[:, 1:-1], R[:, 1:-1], v, w, q[:, 1:-1], qd)
        current = select_reference(ref, (slice(None), 0))
        future = select_reference(ref, (slice(None), slice(1, None)))
        return current, future, goal_from_reference(current, future), halo[:, 0]

    def propose(self, z, *, zero=False, residual_offset=None, action_offset=None):
        residual = z['state'].new_zeros(len(z['state']), 10, 27) if zero else self.adapter.forward_packed(self.packed(self.fields(z)))
        if residual_offset is not None:
            residual = (residual+residual_offset).clamp(-self.adapter.bounds, self.adapter.bounds)
        current, future, goal, halo = self.reference(z, residual)
        action = self.bundle.act(z['history'], goal)
        if action_offset is not None:
            action = action+action_offset
        return dict(residual=residual, current=current, future=future, goal=goal, halo=halo, action=action)

    def advance(self, z, proposal, state_next):
        frames = z['frame']+1
        nominal = self.bank.ref(z['song'], frames)
        r_sm, raw = self.runtime.reward(state_next, proposal['action'], z['action'], nominal, diagnostics=True)
        memory = self.bank.memory_push(z['music'], state_next[:, 34:], nominal.qd, self.music)
        r_music, activity, music_info = self.bank.music_reward(memory, z['song'], frames, self.music)
        d = proposal['residual']/self.adapter.bounds
        overlap = ((proposal['residual'][:, :9]-z['plan'][:, 1:])/self.adapter.bounds).square().mean((1, 2))*z['plan_valid']
        q = proposal['future'].q
        costs = dict(magnitude=d.square().mean((1, 2)), smooth=d.diff(dim=1).square().mean((1, 2)),
            overlap=overlap, reference_limit=(torch.relu(q-self.runtime.mdp.soft_hi)+torch.relu(self.runtime.mdp.soft_lo-q)).sum(-1).mean(-1),
            activity=activity)
        weighted = dict(sm=r_sm, music=.02*self.weights['music']*r_music,
                        **{k: -.02*self.weights[k]*v for k, v in costs.items()})
        raw = dict(raw, sm=r_sm, music=r_music, **costs, **music_info,
            residual_saturation=(d.abs() >= .98).float().mean((1, 2)),
            halo_max_abs=proposal['halo'].abs().amax(-1), target_qd_max_abs=proposal['future'].qd.abs().amax((1, 2)))
        failed = self.runtime.termination(state_next, nominal)
        music_end = frames >= self.bank.ends[z['song']]
        done = failed | music_end
        continuation = (1-self.runtime.termination(state_next, nominal, soft=True))*(~done)
        obs = self.runtime.observation(state_next, proposal['action'], z['xi'],
                                       select_reference(proposal['future'], (slice(None), 0)))
        if self.engine.step.observation_noise:
            obs = self.runtime.mdp.add_noise(obs)
        end = dict(z, state=state_next, action=proposal['action'], frame=frames,
            history=torch.cat((z['history'][:, 1:], obs[:, None]), 1),
            left=z['anchor'], anchor=proposal['residual'][:, 0], plan=proposal['residual'],
            plan_valid=torch.ones_like(z['plan_valid']), music=memory)
        return end, sum(weighted.values()), raw, weighted, continuation, done, failed

    def critic_features(self, z):
        current, goal = self.nominal_goal(z)
        x = self.packed(self.fields(z, current, goal))
        frame, song = z['frame'], z['song']
        clock = torch.stack(((frame-self.bank.starts[song])*.02, (self.bank.ends[song]-frame).clamp_min(0)*.02,
            (self.bank.lengths[song]-1-frame).clamp_min(0)*.02, frame.new_full(frame.shape, -1).float(),
            frame*.02, torch.zeros_like(frame), torch.ones_like(frame), frame.new_full(frame.shape, .04, dtype=torch.float32)), -1)
        ref = torch.cat((current.root_p, mat_to_6d(current.root_R), current.root_v, current.root_w, current.q, current.qd*.1), -1)
        return torch.cat((x, z['state'], normalize_xi(z['xi']), z['action'], z['heading'].flatten(1),
                          clock, x.new_zeros(len(x), 6), ref), -1)
