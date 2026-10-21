"""真实GMT参考暴露账本和信用加权N-step信号，使用600Hz整数时间。

只读取已保存的50Hz控制回执，不推进物理、不调整commit、不复制奖励。二进制账本
保留每步21点plan身份、参考tick、奖励参考及误差；中心消费与未来10点观察分开。
current_only把一单位暴露给中心计划；current_plus_lookahead按配置把一单位分给
中心及未来10点，再在每个真实决策区间按控制步平均得到c_ij。bootstrap份额保留，
不把它重归一化给新计划；任何一步总量均为1。该暴露近似不是物理因果贡献或无偏
策略梯度。旧Replay缺逐步证据返回明确invalid，由标签层按配置回退，不产生负标签。
"""
from __future__ import annotations

import numpy as np
import torch
from .io import digest
from .candidates import state_fingerprint
from .returns import interval_rewards


SCHEMA = "genmo.awr.reference_exposure_ledger.v1"


def _ledger_digest(ledger):
    scalars = {k: v for k, v in ledger.items() if not isinstance(v, (np.ndarray, torch.Tensor)) and k != "sha256"}
    arrays = {k: v for k, v in ledger.items() if isinstance(v, (np.ndarray, torch.Tensor))}
    return digest(dict(metadata=scalars, arrays=state_fingerprint(arrays)))


def build_reference_ledger(episode):
    episode.validate()
    if any("execution_trace" not in r.metadata for r in episode.rows):
        return dict(available=False, reason="missing_per_control_reference_evidence")
    plans, lookup = [], {}
    def code(plan):
        plan = str(plan)
        if plan not in lookup:
            lookup[plan] = len(plans); plans.append(plan)
        return lookup[plan]
    commits = {str(r.identity["plan_id"]): (r.metadata.get("published") or {}).get("effective_tick")
               for r in episode.rows}
    starts, intervals, windows, ticks, reward_ids, reward_ticks, errors = [], [], [], [], [], [], []
    names = None
    for index, row in enumerate(episode.rows):
        controls = row.metadata["execution_trace"]
        if len(controls) != row.executed_control_steps:
            raise ValueError("Credit ledger control count differs from actual rewards")
        for offset, control in enumerate(controls):
            begin = row.control_tick_begin+12*offset
            observed = [str(p) for p in control["consumed_plan_ids"]]
            rticks = [int(t) for t in control["consumed_reference_ticks"]]
            if (int(control["control_tick_begin"]) != begin or int(control["tick"]) != begin+12
                    or rticks != list(range(begin-120, begin+121, 12)) or len(observed) != 21
                    or int(control["reference_tick"]) != begin+12
                    or str(control["episode_id"]) != str(row.identity["episode_id"])):
                raise ValueError("Duplicate/gapped/cross-episode or malformed reference credit evidence")
            reward_plan = str(control["reference_plan_id"])
            for plan in set(observed+[reward_plan]):
                if not plan.startswith("bootstrap:") and (commits.get(plan) is None or int(commits[plan]) > begin):
                    raise ValueError("Uncommitted or unregistered plan appears in actual GMT input")
            starts.append(begin); intervals.append(index); windows.append([code(p) for p in observed]); ticks.append(rticks)
            reward_ids.append(code(reward_plan)); reward_ticks.append(begin+12)
            error = control.get("errors", {})
            if names is None:
                names = sorted(error)
            if sorted(error) != names:
                raise ValueError("Tracking error columns changed within complete Episode")
            errors.append([float(error[k]) for k in names])
    n = len(starts)
    value = dict(schema=SCHEMA, available=True, episode_id=episode.episode_id,
        row_sample_ids=[r.sample_id for r in episode.rows], plan_dictionary=plans,
        commit_ticks={p: int(v) for p, v in commits.items() if v is not None}, error_names=names or [],
        control_ticks=np.asarray(starts, dtype=np.int64), interval_indices=np.asarray(intervals, dtype=np.int64),
        window_plan_indices=np.asarray(windows, dtype=np.int32).reshape(n, 21),
        consumed_reference_ticks=np.asarray(ticks, dtype=np.int64).reshape(n, 21),
        reward_plan_indices=np.asarray(reward_ids, dtype=np.int32),
        reward_reference_ticks=np.asarray(reward_ticks, dtype=np.int64),
        tracking_errors=np.asarray(errors, dtype=np.float64).reshape(n, len(names or [])),
        semantics="acknowledged_plan_exposure_including_preserved_prefix_not_causal_credit")
    value["sha256"] = _ledger_digest(value)
    return value


def reference_exposure(episode, *, mode="current_only", current_weight=.5, lookahead_weight=.5,
                       coordinate_scope="all_committed", include_control_weights=False):
    ledger = episode.reference_ledger
    if ledger is None:
        ledger = build_reference_ledger(episode)
    count = len(episode.rows)
    if not ledger.get("available", False):
        return dict(valid=False, reason=ledger.get("reason"), matrix=torch.zeros(count, count, dtype=torch.float64),
                    exposure=torch.zeros(count, dtype=torch.float64), plans=[])
    if (ledger.get("schema") != SCHEMA or ledger.get("sha256") != _ledger_digest(ledger)
            or ledger["episode_id"] != episode.episode_id
            or ledger["row_sample_ids"] != [r.sample_id for r in episode.rows]):
        raise ValueError("Credit ledger identity differs from immutable Episode")
    if mode not in ("current_only", "current_plus_lookahead"):
        raise ValueError("Unknown reference exposure mode")
    if coordinate_scope not in ("all_committed", "free_coordinates"):
        raise ValueError("Unknown reference coordinate scope")
    if (not np.isfinite([current_weight, lookahead_weight]).all() or min(current_weight, lookahead_weight) < 0
            or current_weight+lookahead_weight <= 0):
        raise ValueError("Invalid current/lookahead weights")
    weights = np.zeros(21, dtype=np.float64)
    if mode == "current_only":
        weights[10] = 1.
    else:
        weights[10] = current_weight/(current_weight+lookahead_weight)
        weights[11:] = lookahead_weight/(current_weight+lookahead_weight)/10
    indices, window, ticks = (np.asarray(ledger[k]) for k in
                              ("interval_indices", "window_plan_indices", "control_ticks"))
    expected = np.concatenate([np.arange(r.control_tick_begin, r.control_tick_end, 12) for r in episode.rows])
    if (not np.array_equal(ticks, expected) or
            not np.array_equal(indices, np.repeat(np.arange(count), [r.executed_control_steps for r in episode.rows]))):
        raise ValueError("Credit ledger no longer matches physical reward time intervals")
    table = ledger["plan_dictionary"]
    matrix = np.zeros((count, count), dtype=np.float64)
    exposure, reports, control_weights = np.zeros(count, dtype=np.float64), [], []
    for i, row in enumerate(episode.rows):
        plan = str(row.identity["plan_id"])
        match = (window == table.index(plan)) if plan in table else np.zeros_like(window, dtype=bool)
        full_weight = (match*weights[None]).sum(axis=1)
        coordinate_fraction = np.ones_like(window, dtype=np.float64)
        if coordinate_scope == "free_coordinates":
            # 30Hz源帧映射到实际50Hz窗口查询tick。部分自由坐标按占比计算，插值边界按
            # 相邻帧线性分配；这是暴露近似，不把root积分传播冒充严格物理因果。
            source_ticks = np.asarray(row.context["future_times"][0])*600
            if not np.allclose(source_ticks, np.rint(source_ticks), atol=1e-6, rtol=0):
                raise ValueError("Reference source times are not on the 600Hz clock")
            source_ticks = np.rint(source_ticks).astype(np.int64)
            if np.any(np.diff(source_ticks) != 20) or source_ticks[0] != row.control_tick_begin:
                raise ValueError("Reference source times differ from the physical decision")
            valid = np.asarray(row.target_valid).sum(axis=1)
            fraction = np.divide(np.asarray(row.free_mask).sum(axis=1), valid,
                                 out=np.zeros(60, dtype=np.float64), where=valid > 0)
            coordinate_fraction = np.interp(np.asarray(ledger["consumed_reference_ticks"]),
                                            source_ticks, fraction, left=0., right=0.)
        control_weight = (match*coordinate_fraction*weights[None]).sum(axis=1)
        if include_control_weights:
            control_weights.append(control_weight)
        window_any, current, future = match.any(axis=1), match[:, 10], match[:, 11:].any(axis=1)
        sums = np.bincount(indices, weights=control_weight, minlength=count)
        lengths = np.array([r.executed_control_steps for r in episode.rows])
        matrix[i] = np.divide(sums, lengths, out=np.zeros_like(sums), where=lengths > 0)
        exposure[i] = control_weight.sum()*.02
        def first(mask):
            return int(ticks[mask][0]) if mask.any() else None
        last = int(ticks[current][-1]) if current.any() else None
        reports.append(dict(plan_id=plan, commit_tick=ledger["commit_ticks"].get(plan),
            first_gmt_input_tick=first(window_any), first_current_tick=first(current), first_lookahead_tick=first(future),
            current_point_ticks=ticks[current].tolist(), lookahead_control_count=int(future.sum()),
            reward_reference_ticks=np.asarray(ledger["reward_reference_ticks"])[
                np.asarray(ledger["reward_plan_indices"]) == table.index(plan)].tolist() if plan in table else [],
            last_current_tick=last,
            first_free_exposure_tick=first(control_weight > 0),
            full_plan_exposure_seconds=float(full_weight.sum()*.02),
            excluded_known_coordinate_seconds=float((full_weight-control_weight).sum()*.02),
            replaced_after_tick=last+12 if last is not None and last+12 < episode.rows[-1].control_tick_end else None,
            actually_consumed_seconds=float(current.sum()*.02), weighted_exposure_seconds=float(exposure[i])))
    if (matrix < 0).any() or (matrix > 1+1e-12).any() or (matrix.sum(axis=0) > 1+1e-12).any():
        raise ValueError("Reference exposure exceeds one unit per physical control")
    matrix = np.clip(matrix, 0., 1.)
    return dict(valid=True, mode=mode, coordinate_scope=coordinate_scope,
                ledger_sha256=ledger["sha256"], matrix=torch.from_numpy(matrix),
                exposure=torch.from_numpy(exposure), plans=reports,
                control_weights=control_weights if include_control_weights else None,
                point_weights=weights.tolist(), total_per_control=1.,
                unassigned_semantics="bootstrap_reference_share_is_not_redistributed")


def credit_advantages(episode, values, exposure, *, nstep=4, gamma_upper=.99):
    """所有c_ij=1时TD残差望远镜求和严格等于普通N-step Advantage。"""
    if torch.is_tensor(values) and values.requires_grad:
        raise ValueError("Credit values must be frozen cross-fitted predictions")
    v = torch.as_tensor(values).detach().cpu().double()
    episode.validate()
    r, discount = interval_rewards(episode.rows, gamma_upper)
    c = torch.as_tensor(exposure).double()
    if (v.shape != r.shape or c.shape != (len(r), len(r)) or not torch.isfinite(c).all()
            or not torch.isfinite(v).all() or (c < 0).any() or (c > 1).any()):
        raise ValueError("Invalid frozen values or reference credit matrix")
    if nstep == "full_mc":
        nstep = len(r)
    if type(nstep) is not int or nstep < 1:
        raise ValueError("Invalid Actor credit N-step horizon")
    delta = r+discount*torch.cat((v[1:], v.new_zeros(1)))-v
    result = torch.zeros_like(v)
    for i in range(len(v)):
        scale = 1.
        for j in range(i, min(len(v), i+nstep)):
            result[i] += scale*c[i, j]*delta[j]
            scale *= discount[j]
    return result
