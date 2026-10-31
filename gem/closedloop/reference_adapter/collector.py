"""真实与想象共用的单控制步，以及显式真实物理采集器。

propose只调用Adapter/参考重建/冻结Actor；advance在实际或独立WM推进后，按名义目标
结算奖励和终止，并保留当时修正参考的75维观测历史。采集分段不触发reset；真实结束
后必须显式reset，绝不补未来状态。滚动计划只能在真实可见时安装，想象在边界截断。
"""

from dataclasses import dataclass, replace
import time

import torch

from supermimic.motion import select_reference
from supermimic.rotations import yaw_mat
from .contracts import AdapterState, Pose, map_tensors
from .features import adapter_fields
from .music_reward import MusicState
from .reference import nominal_window, nominal_pose, reconstruct


@dataclass(frozen=True)
class Proposal:
    corrected_reference: object
    nominal_next: object
    action: torch.Tensor


@dataclass(frozen=True)
class Transition:
    start: AdapterState
    end: AdapterState
    action: torch.Tensor
    reward: torch.Tensor
    raw: dict
    weighted: dict
    continuation: torch.Tensor
    terminated: torch.Tensor
    reason: str
    truncated: bool = False
    policy_version: int = 0
    wall_seconds: float = 0.


class ControlStep:
    def __init__(self, adapter, bundle, runtime, rewards, *, observation_noise=False):
        self.adapter, self.bundle, self.runtime, self.rewards = adapter, bundle, runtime, rewards
        self.observation_noise = observation_noise

    def propose(self, z, *, zero=False, residual_offset=None, action_offset=None):
        z.validate()
        current, future, goal = nominal_window(z.nominal_reference, z.tick, self.runtime.kin)
        values = adapter_fields(z, self.adapter, self.bundle, current, goal)
        residual = z.executed_state.new_zeros(len(z.executed_state), 10, 27) if zero else self.adapter(values)
        if residual_offset is not None:
            residual = (residual+residual_offset).clamp(-self.adapter.bounds, self.adapter.bounds)
        corrected = reconstruct(z, residual, self.runtime.kin)
        action = self.bundle.act(z.control_history, corrected.goal)
        if action_offset is not None:
            action = action+action_offset
        return Proposal(corrected, select_reference(future, (slice(None), 0)), action)

    def advance(self, z, proposal, state_next, *, origin):
        if state_next.shape != z.executed_state.shape or not torch.isfinite(state_next).all():
            raise FloatingPointError("执行/预测终态state55非有限或形状错误")
        corrected, nominal = proposal.corrected_reference, proposal.nominal_next
        reward, raw, weighted, music = self.rewards(z, corrected, state_next, proposal.action, nominal, self.adapter.bounds)
        failed = self.runtime.termination(state_next, nominal)
        tick = z.tick+12
        task_end = tick+12 > z.audio.control_end_tick
        reference_end = tick+132 > z.nominal_reference.upper
        terminated = failed | task_end | (reference_end and z.mode == "fixed")
        end_reason = "task_end" if z.audio.condition_kind == "text" else "music_end"
        reason = "physical_failure" if bool(failed.any()) else end_reason if task_end else "reference_end" if reference_end else "running"
        truncated = z.mode == "rolling" and ((z.plan_boundary_tick is not None and tick >= z.plan_boundary_tick) or reference_end)
        # 没有下一Goal支持的滚动缓存必须更早结束想象；不能以无效输入bootstrap。
        continuation = (1-self.runtime.termination(state_next, nominal, soft=True)) * (~terminated)
        obs = self.runtime.observation(state_next, proposal.action, z.xi, select_reference(corrected.future, (slice(None), 0)))
        if self.observation_noise:
            obs = self.runtime.mdp.add_noise(obs)
        end = replace(z, tick=tick, executed_state=state_next, previous_action=proposal.action,
                      control_history=torch.cat((z.control_history[:, 1:], obs[:, None]), 1),
                      left_pose=z.current_pose, current_pose=corrected.plan.corrected_pose.select((slice(None), 0)),
                      previous_plan=corrected.plan, music_state=music, origin=origin)
        return Transition(z, end, proposal.action, reward, raw, weighted, continuation, terminated, reason, truncated)


def initial_state(observed, nominal, audio, step, *, episode_id, song_id, split="train", tick=0, mode="fixed", boundary_tick=None):
    current, _, _ = nominal_window(nominal, tick, step.runtime.kin)
    pose = Pose(current.root_p, current.root_R, current.q)
    left = nominal_pose(nominal, nominal.ticks.new_tensor([max(nominal.lower, tick-12)])).select((slice(None), 0))
    obs = step.runtime.observation(observed["state"], observed["previous_action"], observed["xi"], current)
    if step.observation_noise:
        obs = step.runtime.mdp.add_noise(obs)
    memory = step.rewards.music.push(MusicState.empty(observed["state"]), observed["state"][:, 34:], current.qd, tick)
    return AdapterState(tick, observed["state"], observed["xi"], observed["previous_action"],
                        obs[:, None].expand(-1, 10, -1).clone(), nominal, left, pose, None,
                        yaw_mat(current.root_R), audio, memory, episode_id, song_id, split,
                        mode=mode, known_at_tick=tick, plan_boundary_tick=boundary_tick, episode_start_tick=tick)


class PhysicalCollector:
    def __init__(self, env, step):
        self.env, self.step = env, step
        self.state, self.terminal = None, True
        self.policy_version, self.steps = 0, 0
        self.reset_count = 0

    @torch.no_grad()
    def reset(self, initial, nominal, audio, **metadata):
        if len(initial) != 1:
            raise ValueError("首版真实物理只使用已验证的单环境接口；batch64为更新样本数")
        self.env.reset(torch.arange(len(initial), device=initial.device), initial)
        self.state = initial_state(self.env.observe(), nominal, audio, self.step, **metadata)
        self.terminal = False
        self.reset_count += 1
        return self.state

    @torch.no_grad()
    def install_nominal(self, nominal, *, known_at_tick, boundary_tick=None):
        if self.state is None or self.state.mode != "rolling" or known_at_tick > self.state.tick:
            raise ValueError("只能安装当前真实可见的滚动计划")
        if nominal.version <= self.state.nominal_reference.version:
            raise ValueError("名义计划版本必须递增")
        old = self.state.nominal_reference
        if nominal.to_sm != old.to_sm or nominal.lower > max(old.lower, self.state.tick-12):
            raise ValueError("滚动计划改变关节坐标合同或缺失左支持")
        past = old.ticks[(old.ticks >= nominal.lower) & (old.ticks <= self.state.tick)]
        if len(past) and not torch.allclose(old.source_at(past), nominal.source_at(past), atol=1e-6, rtol=1e-6):
            raise ValueError("名义计划不得回写过去原始GENMO前缀")
        nominal_window(nominal, self.state.tick, self.step.runtime.kin)
        self.state = replace(self.state, nominal_reference=nominal, known_at_tick=known_at_tick,
                             plan_boundary_tick=boundary_tick)

    @torch.no_grad()
    def collect(self, count, replay=None, *, zero=False, intervention=None):
        if self.terminal or self.state is None:
            raise RuntimeError("episode已结束或未初始化，必须显式reset")
        rows = []
        for _ in range(count):
            if self.state.mode == "rolling" and self.state.tick+144 > self.state.nominal_reference.upper:
                break  # 等待真实可见计划，保留物理episode，不伪造缓存。
            # 名义计划边界只限制想象；真实执行仍可消费当前已知缓存，直到实际新计划到达。
            dev = self.state.executed_state.device
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            started = time.perf_counter()
            proposal = self.step.propose(self.state, zero=zero, **(intervention or {}))
            next_state = self.env.step_physics(proposal.action)
            row = self.step.advance(self.state, proposal, next_state, origin="real")
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            row = replace(row, policy_version=0 if zero else self.policy_version,
                          wall_seconds=time.perf_counter()-started)
            row = map_tensors(row, lambda x: x.detach())
            if replay is not None:
                replay.add(row)
            rows.append(row)
            self.state = row.end
            self.steps += 1
            self.terminal = bool(row.terminated.any())
            if self.terminal or row.truncated:
                break
        return rows
