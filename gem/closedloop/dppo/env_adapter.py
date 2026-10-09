"""将冻结 GMT 的真实执行组织为上层半马尔可夫训练转移。

一次动作对应一次完整去噪计划；latency 模式沿用第八步实测生成与参考准备耗时映射
到仿真时间的约定，等待期间真实执行旧参考。错过的决策只记事件，不补发过时动作。
新状态只在没有 pending 的共同决策网格读取，通过无副作用 preview 构造 bootstrap。
音乐结束是真终止，采集时长上限在合法边界截断。全部实际轨迹先由 RPC journal 保存
再 ACK；此处再次核对计数与逐步记录，并按完整执行区间计算奖励，不按新计划筛帧。
第十步可显式指定 music_start_frame，把已核验的完整音乐和奖励配对动作同步从该帧
开始；音乐保留到真实文件末尾，行政 episode 上限不改变 Critic 的真实剩余时长。
可选 disk_guard 将原始去噪样本和异常证据纳入同一运行磁盘预算：先序列化得到精确
字节数并检查容量，再原子发布和记账。旧v1仍把训练证据IO算入生成延迟；显式选择
deployment_critical.v2时仅部署必需条件、推理、输出转换/拷贝和参考RPC计入到达延迟，
训练trace拷贝、原始证据落盘及RPC journal另行记录。证据依然在物理推进前可靠保存，
副作用回复仍先落盘再ACK；新旧延迟合同必须在训练身份和基线中明确区分。
"""
from __future__ import annotations

import io
import hashlib
import math
import os
import tempfile
import time
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from gem.closedloop.coordinator import ceil_control_tick
from gem.closedloop.dppo.buffer import UpperTransition
from gem.closedloop.dppo.performance import measure, profiled
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.closedloop.dppo.target_activity import load_paired_activity
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
        self.disk_guard = None
        runtime = config.get('runtime', {})
        self.timing_contract = runtime.get('timing_contract', 'legacy_audit_inclusive.v1')
        from .deployment_clock import MODELED_CLOCK, clock_for_config
        if self.timing_contract not in {'legacy_audit_inclusive.v1', 'deployment_critical.v2', MODELED_CLOCK}:
            raise ValueError('Unsupported upper environment timing_contract')
        self.deployment_clock = clock_for_config(config)
        if self.deployment_clock is not None:
            self.latency_budget_s = self.deployment_clock.budget_seconds
        self.rank = runtime.get('rank')
        if self.rank is not None and (type(self.rank) is not int or self.rank < 0):
            raise ValueError('runtime.rank must be a nonnegative integer')
        self.output.joinpath('raw_samples').mkdir(parents=True, exist_ok=True)

    @profiled("storage.raw_evidence")
    def _save_evidence(self, value, path):
        """按精确序列化大小预检，完整落盘后发布；容量不足时不触发物理推进。"""
        guard = self.disk_guard
        if guard is None:
            torch.save(value, path)
            content = Path(path).read_bytes()
            return dict(sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
        path = guard._path(path)
        if path.exists():
            raise FileExistsError(f'Evidence already exists: {path}')
        with io.BytesIO() as serialized:
            with measure("storage.raw_serialize"):
                torch.save(value, serialized)
            guard.check(serialized.tell())
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    with measure("storage.raw_write"):
                        with serialized.getbuffer() as content:
                            digest = hashlib.sha256(content).hexdigest()
                            stream.write(content)
                        stream.flush()
                    with measure("storage.raw_fsync"):
                        os.fsync(stream.fileno())
                # 排他原子发布，避免已有正式样本被覆盖。
                os.link(temporary, path)
                temporary.unlink()
                descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
                guard.account_file(path)
            guard.check()
            return dict(sha256=digest, size_bytes=path.stat().st_size)

    @profiled("collection.advance_including_rpc")
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

    def reset_task(self, sample, music, *, seed, phase='main', music_start_frame=0):
        # 配对动作只进入奖励；在reset/物理推进前验证，绝不加入builder的Actor条件。
        music = np.asarray(music)
        if music.ndim != 2 or music.shape[1] != 35 or len(music) < 2 or not np.isfinite(music).all():
            raise ValueError('Task music must contain finite full [T>=2,35] features')
        if (isinstance(music_start_frame, bool) or not isinstance(music_start_frame, Integral)
                or not 0 <= music_start_frame <= len(music)-2):
            raise ValueError('music_start_frame must leave at least two real music frames')
        if sample['row'].get('num_frames', len(music)) != len(music):
            raise ValueError('reset_task requires full music before applying music_start_frame')
        split = sample['row'].get('split', 'train')
        reward_config = self.config['stage9'].get('reward', {})
        target = load_paired_activity(self.config['stage9']['bc_data_root'], sample,
            music_start_tick=600, window_s=reward_config.get('activity', {}).get('window_s', .5),
            dt=reward_config.get('dt', .02), split=split, music_start_frame=music_start_frame)
        self.full_music_num_frames, self.music_start_frame, self.data_split = len(music), int(music_start_frame), split
        self.phase, self.sample, self.music, self.seed = phase, sample, np.array(music[self.music_start_frame:], copy=True), int(seed)
        self.episode_count += 1
        self.snapshot = self.backend.call('reset_episode', seed=self.seed,
            episode_spec={'sample_id':sample['row']['sample_id'], 'dataset':sample['dataset'], 'mode':self.mode})
        last_warmup_row = None
        while self.snapshot['tick'] < 600 and not self.snapshot['done']:
            warmup_rows = self._advance(min(25, (600-self.snapshot['tick'])//12))
            if warmup_rows:
                last_warmup_row = warmup_rows[-1]
        if self.snapshot['done']:
            raise ExecutionIntegrityError('Warmup terminated before any policy sample')
        self.music_end_tick = 600 + (len(self.music)*50//30)*12
        self.soft_end_tick = 600 + int(float(self.config['stage9']['episode_seconds'])*600)
        reward_type = ExecutionReward
        if self.config.get('runtime', {}).get('vector_reward_contract') is not None:
            from .vector_reward_adapter import VectorExecutionReward
            from .vector_reward_math import VECTOR_REWARD_VERSION
            if (self.config['runtime'].get('backend') != 'gpu_vectorized.v1' or
                    self.config['runtime']['vector_reward_contract'] != VECTOR_REWARD_VERSION):
                raise ValueError('Explicit supported GPU reward contract required')
            reward_type = VectorExecutionReward
        self.reward = reward_type(reward_config, self.music, music_start_tick=600,
                                      target_activity=target)
        if last_warmup_row is not None:
            self.reward.seed_previous_target(last_warmup_row.get('joint_position_target'))
        return self.preview_context()[0]

    def _request(self):
        tick = int(self.snapshot['tick'])
        request_id = f"{self.snapshot['episode_id']}:decision:{self.decision}"
        if self.rank is not None:
            request_id += f':rank:{self.rank}'
        return dict(env_id=self.snapshot['env_id'], episode_id=self.snapshot['episode_id'],
            request_id=request_id, decision_tick=tick,
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
        timings = dict(timing_contract=self.timing_contract, journal_seconds=0.)
        def journal_seconds(method):
            record = getattr(self.backend, 'last_call_timing', None)
            return (float(record['journal_seconds']) if isinstance(record, dict)
                    and record.get('method') == method else 0.)
        request = self._request()
        reservation = self.backend.call('reserve_prefix', request=request)
        prefix_end = time.perf_counter()
        timings['prefix_rpc_seconds'] = prefix_end-started
        timings['journal_seconds'] += journal_seconds('reserve_prefix')
        context, meta = self.builder.build(self.snapshot, reservation, self.music, music_start_tick=600)
        condition_end = time.perf_counter()
        timings['condition_build_seconds'] = condition_end-prefix_end
        self.attempt += 1
        key = f"{self.config['stage9']['run_id']}:{self.iteration}:{self.episode_count}:{self.decision}:{self.attempt}"
        if self.rank is not None:
            key += f':rank:{self.rank}'
        if getattr(self, 'collector_env_slot', None) is not None:
            key += f':environment:{self.collector_env_slot}'
        seed = stable_noise_seed(self.config['stage9']['seed'], key)
        if self.comparison_noise_index is not None:
            seed = stable_noise_seed(1729, str(self.comparison_noise_index))
            self.comparison_noise_index += 1
        device = next(self.policy.actor.parameters()).device
        batch = {k: v.to(device) for k,v in context.items()}
        generator = torch.Generator(device=device).manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        transferred = time.perf_counter()
        timings['input_transfer_seconds'] = transferred-condition_end
        if deterministic:
            with torch.no_grad():
                noise = torch.randn((1,120,30),device=device,generator=generator)
                sample = self.policy.actor.sample(batch, steps=self.policy.steps,
                    guidance_scale=self.policy.guidance_scale,noise=noise)
            trace = {'conditions':cpu_copy(context), 'deterministic_baseline':True}
        else:
            trace = self.policy.sample_rollout(batch, generator=generator)
            sample = trace
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        sampled = time.perf_counter()
        timings['actor_seconds'] = sampled-transferred
        if not deterministic:
            timings['actor_phases'] = dict(getattr(self.policy, 'last_sample_timing', {}))
        meta.update(seed=seed, decision_id=self.decision, plan_id=f"{request['request_id']}:plan")
        anchor = torch.as_tensor(meta['world_anchor'], dtype=torch.float32,device=device)
        with torch.no_grad():
            world = self.policy.actor.endecoder.codec.apply_world_anchor(sample['qpos'].to(device),anchor)
        generated = dict(meta)
        for name, value in (('qpos_world',world),('qpos30',sample['qpos30']),('contact',sample['contact'])):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite generated {name}')
            generated[name] = value[0].detach().cpu().numpy().copy()
        outputs_ready = time.perf_counter()
        timings['output_conversion_transfer_seconds'] = outputs_ready-sampled
        raw_path = self.output/'raw_samples'/f'{self.phase}_{self.attempt:06d}.pt'
        def save_trace():
            nonlocal trace
            beginning = time.perf_counter()
            with measure("storage.raw_cpu_copy"):
                trace = cpu_copy(trace)
            copied = time.perf_counter()
            evidence = self._save_evidence({'trace':trace,'generated':generated,'policy_version':self.policy_version},raw_path)
            timings['raw_evidence_identity'] = evidence
            timings['trace_copy_seconds'] = copied-beginning
            timings['raw_evidence_seconds'] = time.perf_counter()-copied
        if self.timing_contract == 'legacy_audit_inclusive.v1':
            save_trace()
        rejection, prepared = None, None
        preparing = time.perf_counter()
        try:
            prepared = self.backend.call('prepare_plan', generated_plan=generated)
        except RemoteError as exc:
            if exc.code not in {'invalid_qpos','invalid_quaternion','invalid_plan_output','known_source_changed'}:
                raise
            rejection = dict(code=exc.code, message=str(exc), policy_penalty=True,
                             category='finite_invalid_reference')
        prepared_at = time.perf_counter()
        timings['prepare_rpc_seconds'] = prepared_at-preparing
        timings['journal_seconds'] += journal_seconds('prepare_plan')
        if self.timing_contract != 'legacy_audit_inclusive.v1':
            # 实际wall时间仍包含journal；仿真到达只映射部署必需路径，ACK没有被扣除。
            critical = max(0., prepared_at-started-timings['journal_seconds'])
            save_trace()
        else:
            critical = prepared_at-started
        elapsed = time.perf_counter()-started
        timings.update(critical_ready_seconds=critical, total_wall_seconds=elapsed,
            excluded_audit_seconds=(timings['trace_copy_seconds']+timings['raw_evidence_seconds']
                +timings['journal_seconds']) if self.timing_contract != 'legacy_audit_inclusive.v1' else 0.)
        return dict(context=context,meta=meta,generated=generated,trace=trace,prepared=prepared,
                    rejection=rejection,elapsed=elapsed,critical_ready_seconds=critical,
                    timing=timings,seed=seed,raw_path=str(raw_path))

    def step(self, *, deterministic=False):
        """程序/RPC故障单独落盘为invalid，不用策略惩罚替代未知执行后果。"""
        self._inflight_sample = None
        start = int(self.snapshot['tick'])
        try:
            return self._step_impl(deterministic=deterministic)
        except Exception as exc:
            invalid = dict(transition_valid=False, reason='infrastructure_failure',
                error_type=type(exc).__name__, error=str(exc), event_penalty_total=0.,
                reward=None, control_tick_begin=start,
                last_trusted_snapshot=cpu_copy(self.snapshot),
                executed_control_steps=None, executed_physics_steps=None,
                execution_evidence='execution_journal.sqlite',
                last_backend_envelope=cpu_copy(getattr(self.backend, 'last_envelope', None)),
                sample=cpu_copy(self._inflight_sample))
            path = self.output/f'invalid_transition_{self.episode_count}_{self.decision}_{self.attempt}.pt'
            self._save_evidence(invalid, path)
            raise

    def _execution_boundary(self, generated, arrival, ready_tick):
        """原单环境不增加行政边界；向量子类仅在显式参考合同下覆盖。"""
        return None

    def _arrival_tick(self, generated, start):
        if self.mode == 'paused':
            return start
        if self.deployment_clock is not None:
            record = self.deployment_clock.sample(seed=self.config['stage10']['seed'],
                sample_id=self.sample['row']['sample_id'], music_start_frame=self.music_start_frame,
                decision_tick=start)
            generated.setdefault('timing', {})['deployment_clock'] = record
            return record['arrival_tick']
        delay = (generated.get('critical_ready_seconds', generated['elapsed'])
                 if self.timing_contract == 'deployment_critical.v2' else generated['elapsed'])
        return ceil_control_tick(start+600*delay)

    def _step_impl(self, *, deterministic=False):
        start = int(self.snapshot['tick'])
        if start%300:
            raise ExecutionIntegrityError('Upper action requested outside the decision grid')
        generated = self.generate(deterministic=deterministic)
        self._inflight_sample = generated
        candidate = generated['prepared']
        arrival = self._arrival_tick(generated, start)
        ready_tick = max(start+300, int(math.ceil(arrival/300))*300)
        boundary = self._execution_boundary(generated, arrival, ready_tick)
        execution_end = min(self.music_end_tick, boundary['tick']) if boundary else self.music_end_tick
        rows, rewards, details, events = [], [], [], []
        commit, rejection = None, generated['rejection']
        pending = candidate is not None
        while not self.snapshot['done'] and self.snapshot['tick'] < execution_end:
            tick = int(self.snapshot['tick'])
            if pending and tick >= arrival:
                commit_begin = time.perf_counter()
                try:
                    commit = self.backend.call('commit_plan', prepared_plan_id=candidate['prepared_plan_id'],
                                               expected_control_tick=tick)
                except RemoteError as exc:
                    if exc.code != 'late_plan':
                        raise
                    # 延迟到达不是有限但非法的动作；保留事件，不让策略替时序故障付罚分。
                    rejection = dict(code=exc.code, message=str(exc), policy_penalty=False,
                                     category='late_arrival')
                generated['commit_seconds'] = time.perf_counter()-commit_begin
                pending = False
            if not pending and tick >= ready_tick:
                break
            stop = min(execution_end, arrival if pending and arrival>tick else ready_tick,
                       ((tick//300)+1)*300)
            for row in self._advance(min(25,(stop-tick)//12)):
                result = self.reward.evaluate_step(row)
                if not result.get('transition_valid',False):
                    self._save_evidence({'row':row,'reward':result,'raw_sample_path':generated['raw_path']},
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
        boundary_reached = bool(boundary and end>=boundary['tick'] and not terminated)
        truncated = bool(self.snapshot.get('truncated',False) or (not terminated and end>=self.soft_end_tick) or boundary_reached)
        reason = self.snapshot.get('reason') or ('music_end' if terminated else
            boundary['reason'] if boundary_reached else 'collection_limit' if truncated else None)
        rejected_action = bool(rejection and rejection.get('policy_penalty',
            rejection.get('code') in {'invalid_qpos','invalid_quaternion','invalid_plan_output','known_source_changed'}))
        event_reward = self.reward.event_reward('reference_rejected') if rejected_action else 0.
        if self.snapshot.get('terminated', False):
            event_reward += self.reward.event_reward('physical_failure')
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
        backend_cpu_totals = {}
        for row in rows:
            for name, seconds in {**row.get('cpu_timing', {}), **{
                    key: row[key] for key in ('gmt_inference_seconds', 'physics_seconds', 'step_seconds')
                    if key in row}}.items():
                backend_cpu_totals[name] = backend_cpu_totals.get(name, 0.) + float(seconds)
        consumed = sorted({str(p) for row in rows for p in row.get('consumed_plan_ids', [row.get('active_plan_id')]) if p is not None})
        metadata = dict(remaining_music_seconds=self.remaining_music(start),
            next_remaining_music_seconds=self.remaining_music(end), sampler_trace=trace,
            music_start_frame=getattr(self, 'music_start_frame', 0),
            full_music_num_frames=getattr(self, 'full_music_num_frames', len(self.music)),
            data_split=getattr(self, 'data_split', 'train'), backend_cpu_totals=backend_cpu_totals,
            generated=generated['generated'],published=commit,rejection=rejection,reward_details=details,
            events=events,consumed_plan_ids=consumed,event_reward=zero_step_event,event_penalty_total=event_reward,
            raw_sample_path=generated['raw_path'],latency_seconds=generated['elapsed'],
            raw_evidence_identity=generated.get('timing', {}).get('raw_evidence_identity'),
            timing_contract=self.timing_contract, timing=generated.get('timing', {}),
            critical_ready_seconds=generated.get('critical_ready_seconds', generated['elapsed']),
            commit_seconds=generated.get('commit_seconds',0.),terminal_snapshot=cpu_copy(self.snapshot))
        if boundary is not None:metadata['reference_execution_boundary']=dict(boundary,reached=boundary_reached)
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
