"""将冻结 GMT 的真实执行组织为上层半马尔可夫训练转移。

一次动作对应一次完整去噪计划；latency 模式沿用第八步实测生成与参考准备耗时映射
到仿真时间的约定，等待期间真实执行旧参考。错过的决策只记事件，不补发过时动作。
新状态只在没有 pending 的共同决策网格读取，通过无副作用 preview 构造 bootstrap。
音乐结束是真终止，采集时长上限在合法边界截断。全部实际轨迹先由 RPC journal 保存
再 ACK；此处再次核对计数与逐步记录，并按完整执行区间计算奖励，不按新计划筛帧。
"""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch

from gem.closedloop.coordinator import ceil_control_tick
from gem.closedloop.dppo.buffer import UpperTransition
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.closedloop.frozen_actor import stable_noise_seed
from gem.runtime.closedloop_protocol import RemoteError


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


class ExecutionIntegrityError(RuntimeError):
    pass


class UpperEnvironment:
    def __init__(self, config, backend, builder, policy, budget, output):
        self.config, self.backend, self.builder, self.policy = config, backend, builder, policy
        self.budget, self.output = budget, Path(output)
        self.snapshot = None
        self.attempt = self.decision = self.episode_count = 0
        self.latency_budget_s = float(config['stage9'].get('latency_budget_s', .5))
        self.mode = config['stage9'].get('execution_mode', 'latency')
        self.phase = 'main'
        self.policy_version = 0
        self.iteration = 0
        self.comparison_noise_index = None
        self.output.joinpath('raw_samples').mkdir(parents=True, exist_ok=True)

    def _advance(self, count):
        count = int(count)
        if not 1 <= count <= 25:
            raise ValueError('advance requires 1..25 controls')
        previous = self.snapshot
        self.budget.reserve(self.phase, control_steps=count, physics_steps=4*count)
        result = self.backend.call('advance', control_steps=count,
            advance_id=f"{previous['episode_id']}:advance:{self.backend.sequence+1}",
            expected_episode_id=previous['episode_id'])
        self.budget.settle_control(self.phase, count, result)
        actual, rows = int(result['executed_control_steps']), result['trace']
        snapshot = result.get('snapshot')
        if len(rows) != actual:
            raise ExecutionIntegrityError('len(trace) differs from executed_control_steps')
        if not result.get('transition_valid', True) or not result.get('physics_count_exact', True):
            raise ExecutionIntegrityError(f"Unusable execution: {result.get('reason', result.get('fault_stage'))}")
        if result['executed_physics_steps'] != actual*4 or snapshot is None:
            raise ExecutionIntegrityError('Incomplete physical control interval or missing snapshot')
        if snapshot['episode_id'] != previous['episode_id'] or snapshot['tick'] != previous['tick']+12*actual:
            raise ExecutionIntegrityError('Execution changed episode or tick unexpectedly')
        for i, row in enumerate(rows):
            if row['episode_id'] != previous['episode_id'] or row['tick'] != previous['tick']+12*(i+1):
                raise ExecutionIntegrityError('Trace is not a complete contiguous actual trajectory')
        if actual < count and not snapshot['done']:
            raise ExecutionIntegrityError('Short advance without terminal state')
        self.snapshot = snapshot
        return rows

    def reset_task(self, sample, music, *, seed, phase='main'):
        self.phase, self.sample, self.music, self.seed = phase, sample, np.array(music, copy=True), int(seed)
        self.episode_count += 1
        self.snapshot = self.backend.call('reset_episode', seed=self.seed,
            episode_spec={'sample_id':sample['row']['sample_id'], 'dataset':sample['dataset'], 'mode':self.mode})
        while self.snapshot['tick'] < 600 and not self.snapshot['done']:
            self._advance(min(25, (600-self.snapshot['tick'])//12))
        if self.snapshot['done']:
            raise ExecutionIntegrityError('Warmup terminated before any policy sample')
        self.music_end_tick = 600 + (len(music)*50//30)*12
        self.soft_end_tick = 600 + int(float(self.config['stage9']['episode_seconds'])*600)
        self.reward = ExecutionReward(self.config['stage9'].get('reward'), self.music, music_start_tick=600)
        return self.preview_context()[0]

    def _request(self):
        tick = int(self.snapshot['tick'])
        return dict(env_id=self.snapshot['env_id'], episode_id=self.snapshot['episode_id'],
            request_id=f"{self.snapshot['episode_id']}:decision:{self.decision}", decision_tick=tick,
            deadline_tick=tick+ceil_control_tick(self.latency_budget_s*600), min_prefix=12)

    def preview_context(self):
        request = self._request()
        reservation = self.backend.call('preview_prefix', request=request)
        return self.builder.build(self.snapshot, reservation, self.music, music_start_tick=600)

    def remaining_music(self, tick=None):
        tick = self.snapshot['tick'] if tick is None else tick
        return max(0., len(self.music)/30.-(tick-600)/600.)

    def generate(self, *, deterministic=False):
        self.budget.reserve(self.phase, generations=1)
        started = time.perf_counter()
        request = self._request()
        reservation = self.backend.call('reserve_prefix', request=request)
        context, meta = self.builder.build(self.snapshot, reservation, self.music, music_start_tick=600)
        self.attempt += 1
        key = f"{self.config['stage9']['run_id']}:{self.iteration}:{self.episode_count}:{self.decision}:{self.attempt}"
        seed = stable_noise_seed(self.config['stage9']['seed'], key)
        if self.comparison_noise_index is not None:
            seed = stable_noise_seed(1729, str(self.comparison_noise_index))
            self.comparison_noise_index += 1
        device = next(self.policy.actor.parameters()).device
        batch = {k: v.to(device) for k,v in context.items()}
        generator = torch.Generator(device=device).manual_seed(seed)
        if deterministic:
            with torch.no_grad():
                noise = torch.randn((1,120,30),device=device,generator=generator)
                sample = self.policy.actor.sample(batch, steps=self.policy.steps,
                    guidance_scale=self.policy.guidance_scale,noise=noise)
            trace = {'conditions':cpu_copy(context), 'deterministic_baseline':True}
        else:
            trace = self.policy.sample_rollout(batch, generator=generator)
            sample = trace
        meta.update(seed=seed, decision_id=self.decision, plan_id=f"{request['request_id']}:plan")
        anchor = torch.as_tensor(meta['world_anchor'], dtype=torch.float32,device=device)
        with torch.no_grad():
            world = self.policy.actor.endecoder.codec.apply_world_anchor(sample['qpos'].to(device),anchor)
        generated = dict(meta)
        for name, value in (('qpos_world',world),('qpos30',sample['qpos30']),('contact',sample['contact'])):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite generated {name}')
            generated[name] = value[0].detach().cpu().numpy().copy()
        trace = cpu_copy(trace)
        raw_path = self.output/'raw_samples'/f'{self.phase}_{self.attempt:06d}.pt'
        torch.save({'trace':trace,'generated':generated,'policy_version':self.policy_version},raw_path)
        rejection, prepared = None, None
        try:
            prepared = self.backend.call('prepare_plan', generated_plan=generated)
        except RemoteError as exc:
            if exc.code not in {'invalid_qpos','invalid_quaternion','invalid_reference_arrays',
                                'invalid_plan_output','known_source_changed','protected_reference_changed',
                                'locked_source_changed','protection_advanced'}:
                raise
            rejection = dict(code=exc.code, message=str(exc))
        elapsed = time.perf_counter()-started
        return dict(context=context,meta=meta,generated=generated,trace=trace,prepared=prepared,
                    rejection=rejection,elapsed=elapsed,seed=seed,raw_path=str(raw_path))

    def step(self, *, deterministic=False):
        start = int(self.snapshot['tick'])
        if start%300:
            raise ExecutionIntegrityError('Upper action requested outside the decision grid')
        generated = self.generate(deterministic=deterministic)
        candidate = generated['prepared']
        arrival = start if self.mode=='paused' else ceil_control_tick(start+600*generated['elapsed'])
        ready_tick = max(start+300, int(math.ceil(arrival/300))*300)
        rows, rewards, details, events = [], [], [], []
        commit, rejection = None, generated['rejection']
        pending = candidate is not None
        while not self.snapshot['done'] and self.snapshot['tick'] < self.music_end_tick:
            tick = int(self.snapshot['tick'])
            if pending and tick >= arrival:
                commit_begin = time.perf_counter()
                try:
                    commit = self.backend.call('commit_plan', prepared_plan_id=candidate['prepared_plan_id'],
                                               expected_control_tick=tick)
                except RemoteError as exc:
                    if exc.code not in {'late_plan','protected_reference_changed','stale_plan',
                                        'locked_source_changed','protection_advanced'}:
                        raise
                    rejection = dict(code=exc.code,message=str(exc))
                generated['commit_seconds'] = time.perf_counter()-commit_begin
                pending = False
            if not pending and tick >= ready_tick:
                break
            stop = min(self.music_end_tick, arrival if pending and arrival>tick else ready_tick,
                       ((tick//300)+1)*300)
            for row in self._advance(min(25,(stop-tick)//12)):
                result = self.reward.evaluate_step(row)
                if not result.get('transition_valid',False):
                    torch.save({'row':row,'reward':result,'raw_sample_path':generated['raw_path']},
                               self.output/'invalid_reward.pt')
                    raise ExecutionIntegrityError(f"Invalid execution reward: {result.get('errors')}")
                rows.append(row)
                details.append(result)
                rewards.append(float(result['reward']))
            if pending and self.snapshot['tick']%300==0:
                events.append(dict(kind='decision_missed',tick=self.snapshot['tick']))
            if self.snapshot['tick'] > self.soft_end_tick+600*max(2.,2*self.latency_budget_s+.5):
                raise ExecutionIntegrityError('Pending drain exceeded finite guard')
        if pending:
            self.backend.call('discard_plan',prepared_plan_id=candidate['prepared_plan_id'])
        end = int(self.snapshot['tick'])
        terminated = bool(self.snapshot.get('terminated',False) or end >= self.music_end_tick)
        truncated = bool(self.snapshot.get('truncated',False) or (not terminated and end>=self.soft_end_tick))
        reason = self.snapshot.get('reason') or ('music_end' if terminated else 'collection_limit' if truncated else None)
        event_reward = (-1. if rejection else 0.)
        if terminated and reason != 'music_end':
            event_reward -= 5.
        if rewards:
            rewards[-1] += event_reward
            zero_step_event = 0.
        else:
            zero_step_event = event_reward
        self.decision += 1
        next_context = None
        if not terminated and not self.snapshot['done']:
            next_context, _ = self.preview_context()
        elif truncated:
            # 不在非法时间网格虚构可bootstrap条件。
            raise ExecutionIntegrityError('Backend truncation lacks a legal trusted next condition')
        trace = generated['trace']
        consumed = sorted({str(p) for row in rows for p in row.get('consumed_plan_ids', [row.get('active_plan_id')]) if p is not None})
        metadata = dict(remaining_music_seconds=self.remaining_music(start),
            next_remaining_music_seconds=self.remaining_music(end), sampler_trace=trace,
            generated=generated['generated'],published=commit,rejection=rejection,reward_details=details,
            events=events,consumed_plan_ids=consumed,event_reward=zero_step_event,event_penalty_total=event_reward,
            raw_sample_path=generated['raw_path'],latency_seconds=generated['elapsed'],
            commit_seconds=generated.get('commit_seconds',0.),terminal_snapshot=cpu_copy(self.snapshot))
        if deterministic:
            return dict(metadata=metadata,rewards=rewards,terminated=terminated,truncated=truncated,
                        executed_control_steps=len(rows),reason=reason)
        identity = dict(run_id=self.config['stage9']['run_id'], backend_session_id=self.backend.session_id,
            env_id=self.snapshot['env_id'],episode_id=self.snapshot['episode_id'],decision_id=self.decision-1,
            request_id=generated['meta']['request_id'],plan_id=generated['meta']['plan_id'],
            parent_plan_id=generated['meta']['parent_plan_id'],policy_version=self.policy_version)
        return UpperTransition(identity=identity,context=generated['context'],next_context=next_context,
            chain=trace['chain'][0],old_log_prob=trace['old_log_probs'][0],free_mask=trace['free_mask'][0],
            rewards=torch.tensor(rewards,dtype=torch.float64),old_value=0.,next_value=0.,
            control_tick_begin=start,control_tick_end=end,executed_control_steps=len(rows),
            executed_physics_steps=4*len(rows),terminated=terminated,truncated=truncated,
            transition_valid=True,reason=reason,metadata=metadata)
