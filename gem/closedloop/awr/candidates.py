"""首次GENMO决策的同条件候选任务、独立随机流和严格可比性审计。

任务重置种子与生成种子分离；只在首次决策改变噪声，后续噪声由组号和决策号决定。
首次真实物理快照、Actor全部条件、前缀与Anchor均保存二进制数值及内容指纹。分组
打标前检查完整成员、冻结物理/行为身份及逐字段绝对误差；缺失或不一致只能回退，
不能伪造Group Advantage。可观测状态在容差内一致不是任意中途PhysX状态克隆，
后续状态分叉也不再被称为同状态。共同窗口使用积分回报和真实终止后的零吸收尾部，
不按存活秒数平均，不改变原始失败事件或完整MC。
"""
from __future__ import annotations

import copy
import hashlib
import math
import numpy as np
import torch

from .io import digest, cpu_copy


def noise_seed(task, decision_index):
    if not task.get("candidate_group_id"):
        return None
    if decision_index == 0:
        return int(task["generation_seed"])
    key = (task["common_future_noise_key"] if task.get("common_future_noise", True)
           else [task["candidate_group_id"], task["candidate_index"], "independent_future"])
    return int(digest([key, int(decision_index)])[:15], 16)


def add_candidates(tasks, config, *, world_size, macro):
    """group_fraction指候选Episode占总Episode配额的比例，扩组不增加采集预算。"""
    tasks = copy.deepcopy(tasks)
    if not config.get("enabled", False):
        return tasks
    count, k = len(tasks), config["candidates_per_group"]
    groups = math.floor(count * config["group_fraction"] / k)
    if not groups:
        return tasks
    # 优先选不同来源与音乐组；选样身份不包含rank/slot，调度仅在最后赋值。
    unique = {}
    for task in tasks:
        key = (task["dataset"], task["group_id"], task["music_start_frame"], task["initialization"])
        unique.setdefault(key, task)
    by_source = {}
    for key, task in sorted(unique.items(), key=lambda item: digest([macro, item[0]])):
        by_source.setdefault(key[0], []).append(task)
    selected = []
    while len(selected) < groups:
        added = False
        for source in sorted(by_source):
            if by_source[source] and len(selected) < groups:
                selected.append(by_source[source].pop(0)); added = True
        if not added:
            raise ValueError("Insufficient distinct candidate conditions for the fixed quota")
    def condition(t):
        return t["dataset"], t["group_id"], t["music_start_frame"], t["initialization"]
    selected_conditions = {condition(t) for t in selected}
    # smoke中原本每条件四个种子，整组替换，保留全部16个音乐折覆盖。
    ordinary_pool = [t for t in tasks if condition(t) not in selected_conditions]
    if len(ordinary_pool) < count-groups*k:
        used = {t["task_id"] for t in selected} | {t["task_id"] for t in ordinary_pool}
        ordinary_pool += [t for t in tasks if t["task_id"] not in used]
    ordinary = ordinary_pool[:count-groups*k]
    result = list(ordinary)
    for base in selected:
        group = digest(dict(schema="genmo.awr.candidate_group.v1", macro=macro,
            dataset=base["dataset"], sample=base["sample_id"], group=base["group_id"],
            start=base["music_start_frame"], reset_seed=base["seed"], initialization=base["initialization"]))
        for index in range(k):
            task = copy.deepcopy(base)
            task.update(candidate_group_id=group, candidate_index=index, candidate_count=k,
                reset_seed=base["seed"], generation_seed=int(digest([group, "target", index])[:15], 16),
                common_future_noise_key=digest([group, "future"]),
                common_future_noise=config["common_future_noise"],
                comparison_start_tick=600, initial_state_fingerprint=None,
                candidate_schema="genmo.awr.first_generation_candidate.v1")
            task["task_id"] = digest(dict(group=group, candidate=index))
            task["noise_index_start"] = task["generation_seed"]  # 旧外围字段；实际使用noise_seed钩子
            result.append(task)
    for index, task in enumerate(result):
        task["planned_index"] = index
        task["rank"] = index % world_size
    if len(result) != count or len({t["task_id"] for t in result}) != count:
        raise ValueError("Candidate expansion changed task quota or duplicated identity")
    return result


def state_fingerprint(state):
    hashes = {}
    for key, value in sorted(state.items()):
        x = np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
        if x.dtype.kind not in "biuf" or not np.isfinite(x).all():
            raise ValueError("Candidate initial state must contain finite numeric arrays")
        hashes[key] = dict(shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(x.tobytes()).hexdigest())
    return digest(hashes)


def initial_state(packet, context, metadata):
    """在实际reserve之后、扩散采样之前记录精确输入，不从执行后的轨迹倒推。"""
    s, task = packet["snapshot"], packet["candidate_task"]
    required = ("robot_qpos", "robot_joint_vel", "candidate_root_lin_vel_w",
                "candidate_root_ang_vel_w", "candidate_contact_forces_w")
    available = (s.get("candidate_state_schema") == "gmt.live_candidate_initial_state.v1"
                 and s.get("candidate_state_is_live") is True and all(k in s for k in required))
    if not available:
        return dict(available=False, reason="backend_missing_live_initial_state", state={}, fingerprint=None)
    state = {k: np.asarray(s[k]).copy() for k in required}
    state.update({"condition."+k: v.detach().cpu().clone() for k, v in context.items()})
    state.update(root_anchor=np.asarray(metadata["world_anchor"], dtype=np.float64),
                 music_start_frame=np.asarray(task["music_start_frame"], dtype=np.int64),
                 decision_tick=np.asarray(s["tick"], dtype=np.int64),
                 prefix_frames=np.asarray(packet["reservation"]["prefix_frames"], dtype=np.int64))
    return dict(available=True, state=state, fingerprint=state_fingerprint(state),
                scope="first_generation_observable_state_not_full_PhysX_clone",
                reset_seed=task["reset_seed"], generation_seed=packet["seed"],
                backend_runtime_fingerprint=s["runtime_fingerprint_sha256"])


def compare_states(reference, current, config):
    if not reference.get("available") or not current.get("available"):
        return dict(valid=False, reason="missing_initial_state", maximum_differences={})
    a, b = reference["state"], current["state"]
    if set(a) != set(b) or any(state_fingerprint(v["state"]) != v["fingerprint"] for v in (reference, current)):
        return dict(valid=False, reason="initial_state_identity_mismatch", maximum_differences={})
    differences, tolerances, failed = {}, {}, []
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or x.dtype != y.dtype:
            return dict(valid=False, reason="initial_state_shape_dtype_mismatch", maximum_differences={})
        tolerance = (config.get("contact_atol", 1e-3) if "contact_forces" in key
                     else config.get("state_atol", 1e-5))
        if x.dtype.kind in "biu" or key.endswith(("time", "times")):
            tolerance = 0.
        delta = float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)), initial=0.))
        differences[key], tolerances[key] = delta, tolerance
        if not np.isfinite(delta) or delta > tolerance:
            failed.append(key)
    return dict(valid=not failed, reason=None if not failed else "initial_state_outside_tolerance",
                failed_fields=failed, maximum_differences=differences, absolute_tolerances=tolerances,
                exact=reference["fingerprint"] == current["fingerprint"])


def window_return(episode, seconds, *, gamma_upper=.99):
    episode.validate()
    start = episode.rows[0].control_tick_begin
    ticks = round(float(seconds)*600)
    if ticks <= 0 or ticks % 12 or abs(ticks-float(seconds)*600) > 1e-8:
        raise ValueError("Candidate window must use exact positive control ticks")
    end, result, count, events = start+ticks, 0., 0, 0.
    gamma = gamma_upper**(1/25)
    for row in episode.rows:
        for j, reward in enumerate(row.rewards.tolist()):
            tick = row.control_tick_begin+j*12
            if tick < end:
                result += gamma**((tick-start)//12)*reward; count += 1
        # 非零控制步事件已在原rewards中；这里只计原始零步事件。
        event = float(row.metadata.get("event_reward", 0.))
        event_tick = row.control_tick_begin+max(row.executed_control_steps-1, 0)*12
        if event and event_tick < end:
            contribution = gamma**((event_tick-start)//12)*event
            result += contribution; events += contribution
    return dict(return_value=result, start_tick=start, end_tick=end, executed_controls=count,
                missing_controls=ticks//12-count, terminal_absorbing_tail_value=0.,
                zero_step_event_discounted=events, physical_failure=bool(episode.rows[-1].metadata.get("physical_failure")),
                scope="fixed_physical_window_discounted_sum_not_survival_average")


def group_advantages(episodes, config, *, gamma_upper=.99, planned_tasks=None):
    grouped, signals, reports = {}, {}, []
    for episode in episodes:
        group = episode.task.get("candidate_group_id")
        if group:
            grouped.setdefault(group, []).append(episode)
    expected = {}
    for task in planned_tasks or []:
        if task.get("candidate_group_id"):
            expected.setdefault(task["candidate_group_id"], []).append(task)
            grouped.setdefault(task["candidate_group_id"], [])
    for group, members in sorted(grouped.items()):
        if not members:
            report = dict(candidate_group_id=group, valid=False, reasons=["all_members_missing"], members=[],
                          expected_task_ids=[t["task_id"] for t in expected[group]])
            report["identity"] = digest(report); reports.append(report)
            continue
        members.sort(key=lambda e: e.task["candidate_index"])
        first = members[0]; k = first.task["candidate_count"]
        reasons, comparisons = [], []
        if (len(members) != k or [e.task["candidate_index"] for e in members] != list(range(k))
                or len({e.episode_id for e in members}) != k or len({e.task["task_id"] for e in members}) != k):
            reasons.append("incomplete_or_duplicate_members")
        if group in expected and {e.task["task_id"] for e in members} != {t["task_id"] for t in expected[group]}:
            reasons.append("planned_members_missing_or_changed")
        def binding(e):
            return dict(contract=e.contract, group=e.group_id, audio=e.audio_sha256,
                sample=e.task["sample_id"], start=e.task["music_start_frame"], reset=e.task["reset_seed"],
                initialization=e.task["initialization"], common=e.task["common_future_noise_key"],
                common_enabled=e.task["common_future_noise"], count=e.task["candidate_count"],
                tick=e.task["comparison_start_tick"], actor=e.rows[0].identity["actor_fingerprint"],
                version=e.rows[0].identity["policy_version"], sampler=e.rows[0].metadata.get("sampler"))
        if any(digest(binding(e)) != digest(binding(first)) for e in members):
            reasons.append("group_physics_behavior_or_task_identity_mismatch")
        if len({e.task["generation_seed"] for e in members}) != len(members):
            reasons.append("target_noise_not_distinct")
        if not first.task["common_future_noise"]:
            reasons.append("future_random_streams_not_shared_single_intervention_unverified")
        environment = first.contract.get("environment", {})
        if any(environment.get(k) is not False for k in
               ("domain_randomization", "external_perturbations", "observation_noise", "reset_randomization")):
            reasons.append("uncontrolled_physics_randomness")
        reference = first.rows[0].metadata.get("candidate_initial_state") or {}
        for e in members:
            evidence = e.rows[0].metadata.get("candidate_initial_state") or {}
            check = compare_states(reference, evidence, config); comparisons.append(check)
            if not check["valid"]:
                reasons.append(check["reason"])
            if evidence.get("backend_runtime_fingerprint") != reference.get("backend_runtime_fingerprint"):
                reasons.append("actual_backend_parameters_differ")
            if e.task.get("initial_state_fingerprint") != evidence.get("fingerprint"):
                reasons.append("task_state_fingerprint_mismatch")
            if not e.rows[0].actor_valid or e.rows[0].control_tick_begin != e.task["comparison_start_tick"]:
                reasons.append("first_action_unexecuted_or_wrong_tick")
            for index, row in enumerate(e.rows):
                if (row.identity["decision_id"] != index or
                        row.metadata.get("generation_seed") != noise_seed(e.task, index)):
                    reasons.append("generation_random_stream_mismatch"); break
        windows = {format(float(seconds), "g"): [window_return(e, seconds, gamma_upper=gamma_upper) for e in members]
                   for seconds in config.get("evaluation_windows_seconds", [2., 4.])}
        for seconds, scores_for_window in windows.items():
            for e, score in zip(members, scores_for_window):
                ledger = e.reference_ledger or {}
                if ledger.get("available"):
                    selected_controls = np.asarray(ledger["control_ticks"]) < score["end_tick"]
                    score["tracking_errors_mean"] = (dict(zip(ledger["error_names"],
                        np.asarray(ledger["tracking_errors"])[selected_controls].mean(axis=0).tolist()))
                        if selected_controls.any() else {})
        selected = windows[format(float(config.get("score_window_seconds", 2.)), "g")]
        scores = np.array([v["return_value"] for v in selected], dtype=np.float64)
        valid = not reasons
        if valid:
            for e, advantage in zip(members, scores-scores.mean()):
                signals[e.rows[0].sample_id] = float(advantage)
        report = dict(candidate_group_id=group, valid=valid, reasons=sorted(set(reasons)),
            members=[dict(episode_id=e.episode_id, task_id=e.task["task_id"],
                          sample_id=e.rows[0].sample_id, candidate_index=e.task["candidate_index"],
                          generation_seed=e.task["generation_seed"], initial_state_fingerprint=e.task.get("initial_state_fingerprint"))
                     for e in members], initial_comparisons=comparisons, windows=windows,
            score_window_seconds=config.get("score_window_seconds", 2.),
            common_future_noise=first.task["common_future_noise"],
            later_decisions_same_state_claim=False,
            group_mean=float(scores.mean()), group_advantages=(scores-scores.mean()).tolist(),
            group_binding_sha256=digest(binding(first)))
        report["identity"] = digest(report); reports.append(report)
    if config.get("enabled") and not config.get("fallback_to_standard_awr", True) and any(not r["valid"] for r in reports):
        raise ValueError("Invalid candidate group and fallback is disabled")
    return signals, reports
