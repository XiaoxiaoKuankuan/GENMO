"""完整Episode采集和生成目标、提交参考、真实执行区间的绑定。

只复用公共物理事务续体；不调用PPO rollout窗口，也不构造UpperTransition。
当前行为Actor在整轮采集固定。未执行参考保留物理时间账本但不进入Actor回归；失败
Episode正常保存，行政截断另存排除证据。奖励不跨计划搬移，后续消费只更新证据。
"""
from __future__ import annotations

import hashlib
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch

from gem.closedloop.common.vector_environment import VectorUpperEnvironment
from gem.closedloop.common.vector_generation import generate_for_environment_flow
from gem.closedloop.common.world_collector import WorldEnvironmentCollector
from gem.closedloop.common.vector_collector import VectorLaneBackend
from gem.closedloop.common.world_flow import backend_operation
from gem.closedloop.common.env_adapter import WarmupEpisodeEnded
from gem.closedloop.common.reference_credit import ReferenceCreditLedger
from gem.closedloop.dppo.memory_execution import MemoryReceiveJournal
from .records import AWRTransition, AWRCompleteEpisode
from .returns import mc_returns
from .sampler import generate_batch
from .io import cpu_copy, digest, write_tree, atomic_json


def reference_audit(generated, committed):
    """验证提交确认中的实际源参考；只允许既有已知前缀精确恢复。"""
    audit = None if committed is None else committed.get("awr_reference_audit")
    if committed is None:
        return dict(committed=False, target_matches_commit=False)
    if committed.get("accepted") is not True or audit is None:
        raise ValueError("AWR commit lacks authoritative source reference audit")
    original = np.asarray(generated["qpos_world"], dtype="<f8")
    submitted = np.asarray(audit["submitted_qpos_world"], dtype="<f8")
    prefix = int(generated["prefix_frames"])
    if (audit["schema"] != "gmt.awr.reference_audit.v1" or submitted.shape != (60, 28)
            or audit["prefix_frames"] != prefix or not audit["free_source_exact"]
            or audit["original_world_sha256"] != hashlib.sha256(original.tobytes()).hexdigest()
            or audit["submitted_world_sha256"] != hashlib.sha256(submitted.tobytes()).hexdigest()
            or not np.array_equal(original[prefix:], submitted[prefix:])
            or committed["plan_id"] != generated["plan_id"]):
        raise ValueError("AWR generated reference does not match actual committed free source")
    return dict(committed=True, target_matches_commit=True, source_audit=audit,
                root_anchor=generated["world_anchor"],
                coordinate_frame="canonical_local_qpos30_to_world_without_environment_origin")


class AWRUpperEnvironment(VectorUpperEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_credit = ReferenceCreditLedger()
        self.rollout_window_reached = False
        self.actor_fingerprint = None
        self.active_awr_task = {}
        self.awr_decision_origin = 0

    def reset_task_flow(self, *args, **kwargs):
        # 公共decision计数跨Episode累加，不能把全局decision==0误当每集首次生成。
        self.awr_decision_origin = self.decision
        return (yield from super().reset_task_flow(*args, **kwargs))

    def generation_noise_seed(self, decision):
        from .candidates import noise_seed
        return noise_seed(self.active_awr_task, decision-self.awr_decision_origin)

    def generate_flow(self, **kwargs):
        # 评价与采集共享同一个DDIM，不进入原PPO sample_rollout，也不改变采样器。
        result = yield from generate_for_environment_flow(self)
        result["timing"]["reference_deadline"] = dict(self.deadline_diagnostic)
        return result

    def make_transition(self, identity, generated, next_context, trace, rewards, start, end,
                        rows, terminated, truncated, reason, metadata):
        original = generated["generated"]
        execution = reference_audit(original, metadata["published"])
        plan = identity["plan_id"]
        execution.update(observed_by_gmt=plan in metadata["consumed_plan_ids"],
                         reward_interval_ticks=[start, end],
                         consumed_plan_ids=list(metadata["consumed_plan_ids"]),
                         gmt_input_first_tick=metadata["consumed_plan_first_gmt_input_tick"].get(plan))
        target = torch.from_numpy(original["qpos30"]).clone()
        valid = generated["context"]["future_valid"][0, :, None].expand(60, 30).clone()
        # 最后XY位移不影响本窗口任何已解码姿态，沿用Stage1无halo有效性合同。
        valid[-1, :2] = False
        known = generated["context"]["known_qpos30_mask"][0]
        metadata.pop("sampler_trace", None)
        metadata.update(sampler=trace["kernel_config"], generation_seed=generated["seed"],
                        physical_failure=bool(metadata["terminal_snapshot"].get("terminated", False)),
                        execution_trace=cpu_copy(list(rows)), source_commit=self.config["awr"]["source_commits"],
                        initial_target_valid=valid.clone())
        identity.update(actor_version=self.policy_version, actor_fingerprint=self.actor_fingerprint)
        if self.active_awr_task.get("candidate_group_id"):
            for key in ("candidate_group_id", "candidate_index", "candidate_count", "reset_seed",
                        "common_future_noise_key", "comparison_start_tick"):
                identity[key] = self.active_awr_task[key]
            metadata["candidate_initial_state"] = generated.get("candidate_initial_state")
        return AWRTransition(identity=identity, context=cpu_copy(generated["context"]),
            next_context=cpu_copy(next_context), target_qpos30=target, target_valid=valid,
            free_mask=valid & ~known, contact=torch.from_numpy(original["contact"]).clone(),
            control_tick_begin=start, control_tick_end=end, executed_control_steps=len(rows),
            rewards=torch.tensor(rewards, dtype=torch.float64), terminated=terminated,
            truncated=truncated, reason=reason, execution=cpu_copy(execution),
            actor_valid=execution["committed"] and execution["observed_by_gmt"],
            metadata=cpu_copy(metadata))


def finalize_episode(rows, task, contract):
    # 同一计划可能在下个决策区间才进入GMT；仅补充真实消费证据，绝不移动那段奖励。
    observed, center_ticks, reward_ticks = {}, {}, {}
    for row in rows:
        for plan, tick in row.metadata["consumed_plan_first_gmt_input_tick"].items():
            observed[plan] = min(tick, observed.get(plan, tick))
        for control in row.metadata["execution_trace"]:
            plan = str(control["consumed_plan_ids"][10])
            begin = int(control["control_tick_begin"])
            if int(control["consumed_reference_ticks"][10]) != begin:
                raise ValueError("AWR current consumption must use the actual GMT window center")
            center_ticks.setdefault(plan, []).append(begin)
            reward_ticks.setdefault(str(control["reference_plan_id"]), []).append(int(control["reference_tick"]))
    for row in rows:
        plan = row.identity["plan_id"]
        ticks = center_ticks.get(plan, [])
        row.execution.update(observed_by_gmt=plan in observed, gmt_input_first_tick=observed.get(plan),
                             current_point_ticks=np.asarray(ticks, dtype=np.int64),
                             reward_reference_ticks=np.asarray(reward_ticks.get(plan, []), dtype=np.int64),
                             actually_consumed_seconds=len(ticks)*.02,
                             full_window_supervision=True,
                             unused_future_is_not_completed_motion=True)
        row.actor_valid = bool(row.execution["committed"] and plan in observed and row.free_mask.any())
    task = copy.deepcopy(task)
    if task.get("candidate_group_id"):
        task["initial_state_fingerprint"] = (rows[0].metadata.get("candidate_initial_state") or {}).get("fingerprint")
    sample = task["sample"]
    episode = AWRCompleteEpisode(rows=rows, source=sample["dataset"], group_id=sample["group_id"],
        audio_sha256=sample["row"]["source_audio_sha256"], contract=contract, task=task)
    mc_returns(episode)
    from .reference_credit import build_reference_ledger
    episode.reference_ledger = build_reference_ledger(episode)
    return episode


class CompleteCollector:
    def __init__(self, config, policy, builder, world, budget, path, *, num_envs, actor_fingerprint):
        self.config, self.policy, self.builder, self.budget = config, policy, builder, budget
        self.path, self.fingerprint = Path(path), actor_fingerprint
        self.path.mkdir(parents=True, exist_ok=True)
        self.collector = WorldEnvironmentCollector(policy, self._factory, world, num_envs=num_envs,
            generation_batch=num_envs, timeout_seconds=config["awr"]["collection_timeout_seconds"],
            training_storage="ephemeral_device", generation_function=generate_batch)
        for slot in range(num_envs):
            self.collector._resource(slot)

    def _factory(self, slot, proxy):
        path = self.path/f"env{slot:03d}"
        journal = MemoryReceiveJournal(path/"lane")
        backend = VectorLaneBackend(self.collector.lane_client(slot), journal)
        env = AWRUpperEnvironment(self.config, backend, self.builder, proxy, self.budget, path)
        env.actor_fingerprint = self.fingerprint
        env.policy_version = self.config["awr"]["actor_version"]
        return SimpleNamespace(env=env, journal=journal, close=journal.close)

    def collect(self, tasks, catalog, contract, *, on_episode, should_stop=lambda: False):
        from gem.closedloop.dppo.evaluation_world import assign_tasks
        queues = assign_tasks(tasks, self.collector.num_envs)
        outcomes = []
        started = time.perf_counter()
        def flow(slot):
            env = self.collector.states[slot].resource.env
            for index, task in queues[slot]:
                if should_stop():
                    break
                rows = []
                env.active_awr_task = copy.deepcopy(task)
                env.comparison_noise_index = task["noise_index_start"]
                try:
                    yield from env.reset_task_flow(task["sample"], catalog.load_music(task["sample"]),
                        seed=task.get("reset_seed", task["seed"]), phase="train", music_start_frame=task["music_start_frame"],
                        initialization=task["initialization"])
                    guard = math.ceil(task["remaining_music_seconds"]/.02)+2
                    for _ in range(guard):
                        row = yield from env.step_flow(deterministic=False)
                        rows.append(row)
                        if row.terminated or row.truncated:
                            break
                    else:
                        raise RuntimeError("Complete physical task exceeded finite guard")
                    if rows[-1].truncated:
                        write_tree(self.path/"excluded"/task["task_id"],
                                   dict(task=task, rows=[r.to_dict() for r in rows], reason=rows[-1].reason))
                        outcomes.append(dict(task_id=task["task_id"], complete=False, reason=rows[-1].reason))
                    else:
                        episode = finalize_episode(rows, task, contract)
                        publication = on_episode(episode)
                        outcomes.append(dict(task_id=task["task_id"], complete=True, publication=publication,
                            episode_id=episode.episode_id, states=len(rows),
                            actor_samples=sum(r.actor_valid for r in rows), mc_return=float(episode.returns[0]),
                            reward_sum=sum(float(r.rewards.sum())+r.metadata.get("event_reward", 0.) for r in rows),
                            executed_seconds=sum(r.executed_control_steps for r in rows)*.02,
                            physical_failure=rows[-1].metadata["physical_failure"], reason=rows[-1].reason))
                    atomic_json(self.path/"outcomes.json", outcomes)
                except WarmupEpisodeEnded:
                    outcomes.append(dict(task_id=task["task_id"], complete=False, physical_failure=True,
                                         reason="physical_warmup_failure", snapshot=cpu_copy(env.snapshot)))
                    write_tree(self.path/"excluded"/task["task_id"], outcomes[-1])
                    # summary JSON仅小型身份，不重复大数组快照。
                    outcomes[-1].pop("snapshot")
                    atomic_json(self.path/"outcomes.json", outcomes)
            yield from backend_operation("retire")
        with torch.no_grad():
            self.collector._drive({slot: flow(slot) for slot in range(self.collector.num_envs)})
        return dict(tasks=outcomes, seconds=time.perf_counter()-started,
                    timing=self.collector.world_timing, generation_batches=self.collector.batch_reports,
                    actual_budget=self.budget.state_dict())

    def close(self):
        for state in self.collector.states:
            state.resource.close()
