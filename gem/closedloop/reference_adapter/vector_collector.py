"""单进程管理一个真正的批量物理仿真，逐环境复位而非循环调用单环境。

所有控制、参考、Actor和物理动作均使用[N,...]张量。歌曲采样、合法起点、状态扰动
和真实episode编号分别按环境保存；末态先进入Replay再复位对应索引。训练状态扰动
来自真实reset，关节偏置使用原SM执行器xi接口，不虚构摩擦/失败标签。配对局部响应
采集使用同一真实初态的正负动作，单独标记，不作为完整歌曲；GPU之间没有通信。
"""

import time
import torch

from supermimic.mdp import flatten_state, unflatten_state
from supermimic.rotations import so3_exp
from .vector_control import select_state, detach_state


class VectorCollector:
    def __init__(self, env, control, *, seed=42):
        self.env, self.control, self.bank = env, control, control.bank
        self.device, self.num_envs = self.bank.device, env.N
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        # 探索噪声独立随机流，使不同噪声消融仍可使用同一歌曲/起点采样流。
        self.action_generator = torch.Generator(device=self.device).manual_seed(seed+1489)
        self.state = None
        self.episode_serial = self.real_transitions = self.reset_count = 0
        self.vector_steps = 0
        self.real_by_role = {}
        self.policy_version = 0
        self.timing = {}
        self.uniform_start = False
        self.high_error_starts = None
        self.projection_recovery_attempts = 0
        self.projection_recoveries = 0
        self.projection_recovery_events = []

    def random(self, shape):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def sampling(self, count, split='train', *, random_start=True):
        songs = self.bank.sample_tasks(count, split, generator=self.generator)
        frames = torch.zeros(count, device=self.device, dtype=torch.long)
        if random_start:
            end = self.bank.ends[songs]
            uniform = (self.random((count,))*(end-16).clamp_min(1)).long()
            if self.uniform_start:
                return songs, uniform
            near_end = (end-50-(self.random((count,))*450).long()).clamp_min(0)
            u = self.random((count,))
            frames = torch.where(u < .4, uniform, torch.where(u < .8, near_end, frames))
        return songs, frames

    def starting_states(self, songs, frames, *, perturb=False):
        r = self.bank.ref(songs, frames)
        x = flatten_state((r.root_p, r.root_R, r.root_v, r.root_w, r.q, r.qd))
        x = torch.where((frames == 0)[:, None], self.bank.initial[songs], x)
        if perturb:
            p, R, v, w, q, qd = unflatten_state(x)
            hard = self.random((len(x),)) < .25
            scale = torch.where(hard, .22, .035)[:, None]
            noise = lambda width: 2*self.random((len(x), width))-1
            q = (q+scale*noise(21)).clamp(self.control.runtime.kin.q_lo, self.control.runtime.kin.q_hi)
            qd = qd+scale*5*noise(21)
            R = R@so3_exp(scale*1.5*noise(3))
            p = p+torch.cat((scale*noise(2), .035*noise(1)), -1)
            v = v+scale*3*noise(3)
            x = flatten_state((p, R, v, w, q, qd))
        return x

    @torch.no_grad()
    def reset(self, ids=None, *, split='train', random_start=True, perturb=True,
              songs=None, frames=None, initial=None, joint_bias=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        if not len(ids):
            return
        hard = None
        if songs is None:
            songs, frames = self.sampling(len(ids), split, random_start=random_start)
            if split == 'train' and random_start and self.high_error_starts is not None:
                pool = self.high_error_starts
                hard = self.random((len(ids),)) < .6
                selected = torch.randint(len(pool['state']), (len(ids),), device=self.device, generator=self.generator)
                songs = torch.where(hard, pool['song'][selected], songs)
                frames = torch.where(hard, pool['frame'][selected], frames)
        if initial is None:
            initial = self.starting_states(songs, frames, perturb=perturb)
            if hard is not None:
                initial = torch.where(hard[:, None], pool['state'][selected], initial)
        self.env.xi[ids, 1:22] = ((self.random((len(ids), 21))-.5)*.01 if perturb else 0) if joint_bias is None else joint_bias
        self.env.reset(ids, initial)
        observed = {k: value[ids] for k, value in self.env.observe().items()}
        episodes = torch.arange(self.episode_serial, self.episode_serial+len(ids), device=self.device)
        self.episode_serial += len(ids)
        new = self.control.initial(observed, songs, frames, episodes)
        if hard is not None:
            # 来源是已经真实发生的因果历史，复位片段不冒充从歌曲开头的MC。
            for key in ('action', 'history', 'heading', 'left', 'anchor', 'plan', 'plan_valid', 'music',
                        'execution_previous_qd', 'execution_jerk_valid'):
                if key in pool:
                    new[key][hard] = pool[key][selected[hard]]
            self.env.a_prev[ids] = new['action']
        if self.state is None:
            if len(ids) != self.num_envs:
                raise ValueError('首次初始化必须覆盖所有环境')
            self.state = new
        else:
            for k, value in new.items():
                self.state[k][ids] = value
        self.reset_count += len(ids)

    @torch.no_grad()
    def step(self, replay=None, *, zero=False, auto_reset=True, split='train', perturb=True,
             residual_offset=None, action_offset=None, replay_ids=None, contact_count=0, latent=None,
             reset_split=None, action_noise=0., exploration_fraction=0., exploration_std=.1):
        z = self.state
        if z is None:
            raise RuntimeError('采集前必须reset')
        cuda = z['state'].is_cuda
        if cuda:
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        recovery_limit = (self.projection_recovery_attempts if split == 'train' and auto_reset and
            residual_offset is None and action_offset is None and latent is None else 0)
        for attempt in range(recovery_limit+1):
            proposal = {}
            try:
                proposal = self.control.propose(z, zero=zero, residual_offset=residual_offset, action_offset=action_offset, latent=latent)
                certificate = self.control.constraint_certificate(z, proposal)
                if certificate is not None and not bool(certificate['feasible'].all()):
                    error = RuntimeError('有限时域参考未通过实际输出精度证书；禁止发送异常Goal')
                    error.bad_mask = ~certificate['feasible']
                    error.certificate = certificate
                    raise error
                break
            except RuntimeError as error:
                if not hasattr(error, 'bad_mask'):
                    raise
                certificate = getattr(error, 'certificate', {})
                self.last_constraint_failure = {k: v.detach().cpu() for k, v in certificate.items()}
                self.last_constraint_state = {k: v.detach().clone() for k, v in z.items()}
                self.last_constraint_proposal = {k: v.detach().cpu() for k, v in proposal.items()
                    if isinstance(v, torch.Tensor)}
                ids = error.bad_mask.to(device=self.device).nonzero().flatten()
                event = dict(vector_step=self.vector_steps, attempt=attempt, rejected_environments=len(ids),
                    episode_ids=z['episode'][ids].detach().cpu().tolist(), error=str(error),
                    action_sent=False, replay_transition_written=False)
                self.projection_recovery_events.append(event)
                if attempt >= recovery_limit or not len(ids):
                    raise
                # 未推进物理，不写入假转移；只重置有问题的环境，使用新episode_id。
                # 旧episode行政终止，不伪装成真实失败或完整MC；Replay窗口由id阻断。
                self.reset(ids, split=reset_split or split, random_start=True, perturb=perturb)
                self.projection_recoveries += len(ids)
                z = self.state
        if action_noise or exploration_fraction:
            # 与SM collect相同：探索施加在Actor输出之后，Replay、奖励及下一观测
            # 都读取这份实际执行动作；评价调用默认0，不改变冻结Actor/normalizer。
            proposal['policy_action'] = proposal['action']
            noise = torch.randn(proposal['action'].shape, device=self.device, generator=self.action_generator)
            large = torch.rand((len(noise), 1), device=self.device, generator=self.action_generator) < exploration_fraction
            std = torch.where(large, exploration_std, action_noise)
            proposal['action'] = (proposal['action']+std*noise).clamp(-20, 20)
            proposal['exploration_std'] = std[:, 0]
        if not bool(torch.isfinite(proposal['action']).all()):
            raise FloatingPointError('实际控制动作出现非有限值，未推进物理')
        if cuda:
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        next_state = self.env.step_physics(proposal['action'])
        if cuda:
            torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
        end, reward, raw, weighted, continuation, done, failed = self.control.advance(z, proposal, next_state)
        if cuda:
            torch.cuda.synchronize(self.device)
        t3 = time.perf_counter()
        contacts = None
        if contact_count:
            from .vector_audit import contact_tensors
            contacts = contact_tensors(self.env, contact_count)
        if cuda and contact_count:
            torch.cuda.synchronize(self.device)
        t3b = time.perf_counter()
        if not bool(torch.isfinite(next_state).all() & torch.isfinite(reward).all()):
            raise FloatingPointError('真实物理状态或奖励出现非有限值')
        if replay is not None:
            ids = slice(None) if replay_ids is None else replay_ids
            replay.add(select_state(z, ids), proposal['action'][ids], next_state[ids], reward[ids],
                       continuation[ids], done[ids], failed[ids], self.policy_version)
        if cuda and replay is not None:
            torch.cuda.synchronize(self.device)
        t4 = time.perf_counter()
        # reset只写采集器下一状态；返回的真实末态和起点不能被原地改成reset状态。
        self.state = {k: v.detach().clone() for k, v in end.items()}
        self.real_transitions += self.num_envs
        self.real_by_role[split] = self.real_by_role.get(split, 0)+self.num_envs
        self.vector_steps += 1
        ended = done.nonzero().flatten()
        if auto_reset and len(ended):
            # 计费角色train_safety不是新的歌曲划分；复位仍显式采样其所属训练集。
            self.reset(ended, split=reset_split or split, random_start=True, perturb=perturb)
        if cuda:
            torch.cuda.synchronize(self.device)
        t5 = time.perf_counter()
        timing = dict(control_ms=(t1-t0)*1000, physics_ms=(t2-t1)*1000,
            reward_and_shared_diagnostics_ms=(t3-t2)*1000, detailed_contact_diagnostics_ms=(t3b-t3)*1000,
            replay_ms=(t4-t3b)*1000,
            reset_ms=(t5-t4)*1000, total_ms=(t5-t0)*1000, duplicate_diagnostic_fk_ms=0.)
        self.timing = timing
        return dict(start=z, end=end, proposal=proposal, reward=reward, raw=raw, weighted=weighted,
                    continuation=continuation, done=done, failed=failed, timing=timing, contacts=contacts)

    @torch.no_grad()
    def tangent_pairs(self, epsilon=.02):
        if self.num_envs % 2:
            raise ValueError('正负真实配对需要偶数环境')
        half = self.num_envs//2
        songs, frames = self.sampling(half, 'train', random_start=True)
        initial = self.starting_states(songs, frames, perturb=True)
        # 相邻两环境有相同实际初态和执行器偏置，只有第一步动作正负扰动不同。
        songs, frames, initial = songs.repeat_interleave(2), frames.repeat_interleave(2), initial.repeat_interleave(2, 0)
        bias = ((self.random((half, 21))-.5)*.01).repeat_interleave(2, 0)
        self.reset(songs=songs, frames=frames, initial=initial, perturb=False, joint_bias=bias)
        z = detach_state(self.state)
        d = 2*self.random((half, 21))-1
        d = d/d.norm(dim=-1, keepdim=True).clamp_min(1e-12)*epsilon
        offsets = torch.stack((d, -d), 1).flatten(0, 1)
        # 局部响应必须以当前策略动作作为中心，不能永远围绕零Adapter拟合导数。
        row = self.step(zero=False, auto_reset=False, action_offset=offsets, split='tangent')
        mismatch = (z['state'][0::2]-z['state'][1::2]).abs().amax(-1)
        valid = (mismatch <= 1e-5) & ~row['done'][0::2] & ~row['done'][1::2]
        action = row['proposal']['action']
        return dict(origin='real_paired_training_actions', state=z['state'][0::2][valid],
            xi=z['xi'][0::2][valid], plus_action=action[0::2][valid], minus_action=action[1::2][valid],
            plus_next=row['end']['state'][0::2][valid], minus_next=row['end']['state'][1::2][valid],
            song=songs[0::2][valid], frame=frames[0::2][valid], epsilon=epsilon,
            rejected_pairs=int((~valid).sum()), initial_max_abs=float(mismatch.max()))
