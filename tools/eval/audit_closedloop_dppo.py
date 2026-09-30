#!/usr/bin/env python3
"""只读审计第九步 DPPO 的真实执行、固定目标、网络更新及恢复证据。

输入是包含 acceptance/ 与 resume_check/ 的运行目录。审计器逐条读取 SQLite
ACK 日志，独立核对 session/seq、控制时钟、四个 200Hz 子步，再把 rollout 中
每个实际奖励与同 session、episode、tick 的物理记录关联。奖励只复核已记录分项
的权重与 dt 积分；不启动 Actor、GMT、Isaac 或 GPU，也不声称重新验证完整奖励
算法或物理质量。半马尔可夫 GAE 直接按存储值独立重算，不调用训练用 returns。

大 checkpoint 仅通过 CPU mmap 读取元数据与键集合，不遍历模型或优化器张量。
PyTorch 文件必须来自可信运行，因为 UpperTransition 需要 weights_only=False。
SQLite 原库及 WAL 先复制到临时目录，避免只读 SQLite 连接仍创建源目录 shm；
复制期间源文件变化则拒绝把不一致快照当作证据。临时文件退出时自动清理。

默认完整门禁要求主训练 64 条、恢复 16 条及完整 checkpoint。--allow-incomplete
仅把缺失或尚未运行的阶段标成 not_run，绝不豁免已有 failed、损坏记录或 KL 超限。
所有输入保持不变；只有显式 --output 指定的新文件会被排他创建，已有文件不覆盖。
源码与资产冻结结论来自归档清单、运行末尾复核及恢复身份的交叉核对，不冒充重新
哈希当前工作树，也不证明网络重试场景已覆盖或训练质量已收敛。

显式启用学习率候选时，另核验同一起点/梯度、候选KL与最大合格选择、一次保留更新
及所有候选尝试预算。checkpoint仅额外读取优化器参数组学习率，不扫描大张量；恢复
后的实际学习率必须与胜出候选一致。没有候选配置的既有归档保持原验收语义。
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

import torch
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gem.closedloop.dppo.buffer import UpperTransition


COMPONENTS = ("music", "track", "stable", "actuator", "contact", "consistency")
V2_WEIGHT_KEYS = {"track": "track_weight", "music": "music_weight", "stable": "stable_weight",
    "alive": "alive_weight", "cmd": "cmd_penalty_weight", "torque": "torque_penalty_weight",
    "contact": "contact_penalty_weight", "joint_limit": "joint_limit_penalty_weight"}
HARD_LIMITS = dict(generations=256, control_steps=10000, physics_steps=40000, iterations=3)
PHASE_LIMITS = dict(calibration=150, main=6500, resume=1800, comparison_A=300,
                    comparison_B=300, comparison_C=300, diagnostic=650)


class NotRun(Exception):
    """仅表示缺少后续阶段的证据；不能用来表示实际数据不合法。"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label):
    require(type(value) is int and value >= 0, f"{label}: expected nonnegative integer")
    return value


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}: expected number")
    require(math.isfinite(value), f"{label}: nonfinite number")
    return float(value)


def near(a, b, label, atol=1e-9):
    a, b = number(a, label), number(b, label)
    require(math.isclose(a, b, abs_tol=atol, rel_tol=1e-9), f"{label}: {a} != {b}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def load_torch(path):
    # 禁止退回会一次性复制全部大权重的加载路径。
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


class Audit:
    def __init__(self, root, allow_incomplete):
        self.root, self.allow_incomplete = root, allow_incomplete
        self.checks = []

    def check(self, name, function):
        try:
            result = function()
            self.checks.append(dict(name=name, status="passed", details=result))
            return result
        except (FileNotFoundError, NotRun) as error:
            self.checks.append(dict(name=name, status="not_run" if self.allow_incomplete else "failed", error=str(error)))
        except Exception as error:
            self.checks.append(dict(name=name, status="failed", error=f"{type(error).__name__}: {error}"))
        return None


def need(value, label):
    if value is None:
        raise NotRun(f"{label}: prerequisite artifact unavailable; see its own check")
    return value


def check_summary(summary, mode):
    need(summary, "summary")
    require(summary.get("mode") == mode, f"expected mode {mode}")
    if summary.get("status") == "running":
        raise NotRun("summary still reports running")
    require(summary.get("status") == "passed", f"run status={summary.get('status')}: {summary.get('error')}")
    require(summary.get("exit_code") == 0, "nonzero or missing process exit code")
    return dict(status=summary["status"], mode=mode)


def check_probability(summary):
    need(summary, "summary")
    report = summary.get("probability_check")
    if report is None:
        raise NotRun("zero-update probability check has not been recorded")
    require(report.get("passed") is True, "probability check did not pass")
    tolerances = dict(max_abs_log_probability_difference=1e-4, max_abs_ratio_minus_one=1e-3,
                      max_abs_independent_gaussian_difference=1e-8)
    for key, limit in tolerances.items():
        require(0 <= number(report[key], key) <= limit, f"{key} exceeds {limit}")
    return {key: report[key] for key in tolerances}


def check_lr_calibration(summary, config):
    """独立核对可选学习率试验；拒绝用最后一次试验或累计更新冒充最大合格候选。"""
    configured = config['stage9'].get('actor_lr_candidates')
    calibration = summary.get('actor', {}).get('lr_calibration')
    if not configured and calibration is None:
        return None
    require(isinstance(configured, list) and 1 <= len(configured) <= HARD_LIMITS['iterations'],
            'learning-rate candidates missing or outside bounded attempt count')
    learning_rates = [number(value, 'candidate learning rate') for value in configured]
    require(all(0 < value <= 1e-6 for value in learning_rates)
            and learning_rates == sorted(set(learning_rates)), 'candidate learning rates must be positive, sorted and unique')
    rule = 'largest_candidate_below_joint_kl_limit'
    require(config['stage9'].get('actor_lr_selection') == rule, 'unsupported learning-rate selection rule')
    require(isinstance(calibration, dict), 'configured learning-rate calibration report missing')
    require(calibration.get('selection_rule') == rule, 'reported learning-rate selection rule differs')
    require(calibration.get('base_state_restored_per_candidate') is True, 'candidates did not restore the same initial state')
    require(calibration.get('gradients_reused') is True, 'candidates did not reuse one fixed gradient')
    attempts = integer(calibration.get('attempt_count'), 'optimizer attempt count')
    require(attempts == len(learning_rates), 'candidate attempt count differs from configured rates')
    require(type(calibration.get('accepted_updates')) is int and calibration['accepted_updates'] == 1,
            'candidate search must retain exactly one accepted update')
    limit = number(config['stage9']['kl_stop_joint'], 'KL threshold')
    require(0 < limit <= .02, 'joint KL threshold exceeds bounded acceptance contract')
    near(calibration.get('kl_limit'), limit, 'candidate KL threshold', atol=1e-12)
    candidates = calibration.get('candidates')
    require(isinstance(candidates, list) and len(candidates) == attempts, 'candidate evidence count differs from attempts')
    accepted = []
    for expected_lr, candidate in zip(learning_rates, candidates):
        lr = number(candidate['lr'], 'candidate learning rate')
        near(lr, expected_lr, 'candidate learning-rate order', atol=0.)
        change = candidate['parameter_change']
        changed = integer(change['changed_count'], 'changed parameter count')
        total = integer(change['parameter_count'], 'parameter count')
        require(total > 0 and changed <= total, 'invalid changed parameter count')
        near(change['changed_fraction'], changed / total, 'changed parameter fraction', atol=1e-12)
        l2, maximum = number(change['l2'], 'parameter delta L2'), number(change['max_abs'], 'maximum parameter delta')
        require((l2 > 0 and maximum > 0) if changed else (l2 == 0 and maximum == 0),
                'parameter delta norms disagree with changed count')
        require(maximum <= l2 + 1e-12, 'maximum parameter delta exceeds total L2')
        require(isinstance(change.get('per_module'), dict) and bool(change['per_module']), 'per-module parameter deltas missing')
        kl = candidate['kl']
        require(kl['joint_kl_scope'] == 'sum_free_coordinates_per_internal_transition_then_mean',
                'candidate KL uses wrong reduction')
        mean = number(kl['mean_joint_kl'], 'candidate mean joint KL')
        require(mean >= -1e-10, 'negative candidate mean joint KL')
        for name in ('p95_joint_kl', 'max_joint_kl', 'mean_per_dimension_kl', 'mean_chain_joint_kl'):
            require(number(kl[name], name) >= -1e-10, f'negative candidate {name}')
        near(kl['mean_chain_joint_kl'], mean * config['stage9']['denoising_steps'], 'candidate chain KL', atol=1e-7)
        qualifies = changed > 0 and mean <= limit
        require(type(candidate.get('accepted')) is bool and candidate['accepted'] == qualifies,
                'candidate acceptance differs from finite parameter change and joint KL gate')
        if qualifies:
            accepted.append(candidate)
    require(bool(accepted), 'no changed candidate passed joint KL gate')
    selected = max(accepted, key=lambda item: item['lr'])
    near(calibration.get('selected_lr'), selected['lr'], 'selected learning rate must be largest qualifying candidate', atol=0.)
    final_kl = summary.get('kl')
    require(isinstance(final_kl, dict), 'final restored candidate KL missing')
    require(final_kl.get('joint_kl_scope') == selected['kl']['joint_kl_scope'], 'final KL scope differs from selected candidate')
    for name in ('mean_joint_kl', 'p95_joint_kl', 'max_joint_kl', 'mean_per_dimension_kl', 'mean_chain_joint_kl'):
        near(final_kl[name], selected['kl'][name], f'final KL differs from selected candidate: {name}', atol=1e-10)
    return dict(selected_lr=float(selected['lr']), optimizer_attempts=attempts, accepted_updates=1,
                selection_rule=rule, fixed_gradient_candidates=True)


def optimizer_group_lrs(optimizer, label):
    """只读参数组小元数据；不访问Adam动量或模型张量的实际内容。"""
    groups = optimizer.get('param_groups')
    require(isinstance(groups, list) and bool(groups), f'{label}: optimizer parameter groups missing')
    rates = [number(group.get('lr'), f'{label} learning rate') for group in groups]
    require(all(rate > 0 for rate in rates), f'{label}: nonpositive optimizer learning rate')
    return rates


def check_group_lrs(actual, expected, label):
    require(isinstance(actual, list) and len(actual) == len(expected), f'{label}: optimizer learning-rate groups differ')
    for actual_lr, expected_lr in zip(actual, expected):
        near(actual_lr, expected_lr, f'{label}: optimizer learning rate differs', atol=0.)


def check_frozen(summary, source, preflight):
    need(summary, "summary"); need(source, "source identity"); need(preflight, "preflight")
    require(preflight.get("ready") is True, "preflight was not ready")
    records = source["files"]
    require(bool(records) and source["file_count"] == len(records), "source inventory count mismatch")
    inventory = [(r["repository"], r["relative_path"], r["sha256"]) for r in records]
    require(len(set((r[0], r[1]) for r in inventory)) == len(inventory), "duplicate source inventory entry")
    digest = hashlib.sha256(json.dumps(inventory, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(digest == source["source_manifest_sha256"], "source manifest content digest mismatch")
    proof = summary.get("source_unchanged")
    if proof is None:
        raise NotRun("end-of-run source freeze verification missing")
    require(proof.get("unchanged") is True, "runtime source changed")
    if "initial_manifest_sha256" in proof:
        require(proof["initial_manifest_sha256"] == digest, "source verification refers to different inventory")
    require(summary.get("original_assets_unchanged") is True, "original assets not verified unchanged")
    require(bool(preflight["asset_sha256"]), "empty asset inventory")
    gmt = summary.get("worker_shutdown", {}).get("gmt")
    if gmt is None:
        raise NotRun("GMT shutdown proof missing")
    require(gmt.get("policy_unchanged") is True, "GMT policy changed or was not checked")
    require(gmt.get("runtime_parameters_unchanged") is True, "GMT parameters/normalization changed or were not checked")
    require(gmt.get("closed") is True and gmt.get("process_exit_code") == 0, "GMT did not close cleanly")
    require(not gmt.get("close_error") and not gmt.get("forced_shutdown"), "GMT close error or forced shutdown")
    return dict(source_manifest_sha256=digest, asset_count=len(preflight["asset_sha256"]),
                policy_unchanged=True, runtime_parameters_unchanged=True,
                evidence="archived identities and end-of-run freeze reports; no live source/asset rehash")


def checkpoint_metadata(root, summary):
    need(summary, "main summary")
    if not summary.get("checkpoint"):
        raise NotRun("training checkpoint was not published")
    declared = Path(summary["checkpoint"])
    archived = root / "checkpoints" / declared.name
    path = archived if archived.is_file() else declared
    payload = load_torch(path)
    require(payload.get("version") == "genmo.closedloop.stage9.full_state.v1", "unsupported checkpoint version")
    keys = ("actor", "critic", "actor_optimizer", "critic_optimizer", "state", "identity", "config", "rng",
            "samplers", "optimizer_layout", "restore_environment")
    require(all(key in payload for key in keys), "full-state checkpoint fields missing")
    # 只复制小型 Python 元数据；模型权重和 Adam 状态只检查键存在，不触碰页内容。
    result = {key: copy.deepcopy(payload[key]) for key in ("state", "identity", "config", "restore_environment")}
    result.update(path=str(path), rng_keys=sorted(payload["rng"]), sampler_names=sorted(payload["samplers"]),
                  optimizer_layout_names=sorted(payload["optimizer_layout"]))
    calibration = check_lr_calibration(summary, result['config'])
    if calibration is not None:
        rates = {name: optimizer_group_lrs(payload[f'{name}_optimizer'], name) for name in ('actor', 'critic')}
        check_group_lrs(rates['actor'], [calibration['selected_lr']] * len(rates['actor']), 'checkpoint Actor')
        check_group_lrs(rates['critic'], [result['config']['stage9']['critic_lr']] * len(rates['critic']), 'checkpoint Critic')
        near(result['state'].get('selected_actor_lr'), calibration['selected_lr'], 'checkpoint selected learning rate', atol=0.)
        require(type(result['state'].get('optimizer_attempts')) is int
                and result['state']['optimizer_attempts'] == calibration['optimizer_attempts'],
                'checkpoint optimizer attempt count differs from candidate report')
        result['optimizer_lrs'] = rates
        result['lr_calibration'] = calibration
    del payload
    require(result["restore_environment"] == "new_worker_session_and_reset", "resume must reconstruct physics in a new worker")
    state = result["state"]
    require(state.get("buffer_size") == 0 and state.get("pending_plan") is False, "checkpoint not at empty-buffer/no-pending boundary")
    require(state.get("actor_updates") == 1 and state.get("iteration") == 1, "expected one accepted Actor update")
    require(state.get("critic_updates") == 20, "expected 20 Critic updates")
    require(set(result["rng_keys"]) >= {"python", "numpy", "torch", "cuda", "generators", "generator_devices"}, "full RNG state missing")
    require(set(result["sampler_names"]) == {"music", "bc"}, "music/BC sampler state missing")
    require(set(result["optimizer_layout_names"]) == {"actor", "critic"}, "optimizer name binding missing")
    return result


def _file_signature(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


@contextmanager
def journal_snapshot(path):
    """WAL 感知磁盘快照；不让 SQLite 在输入目录创建/更新共享内存文件。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sources = [path, Path(str(path) + "-wal")]
    before = [_file_signature(p) for p in sources]
    with tempfile.TemporaryDirectory(prefix="stage9-audit-sqlite-") as directory:
        copy_path = Path(directory) / "journal.sqlite"
        for source, suffix, signature in zip(sources, ("", "-wal"), before):
            if signature is not None:
                shutil.copyfile(source, Path(str(copy_path) + suffix))
        require(before == [_file_signature(p) for p in sources], "journal changed while snapshotting; retry at a stable boundary")
        connection = sqlite3.connect(copy_path.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            yield connection
        finally:
            connection.close()


def check_journal(path):
    sequences, ticks, episode_end = {}, {}, {}
    counts = Counter()
    with journal_snapshot(path) as connection:
        require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity failure")
        for row_number, (identity, digest, payload) in enumerate(connection.execute("SELECT identity,sha256,payload FROM replies ORDER BY rowid"), 1):
            require(hashlib.sha256(payload.encode()).hexdigest() == digest, f"reply {row_number}: SHA256 mismatch")
            reply = json.loads(payload)
            require(reply.get("execution_protocol") == "ack.v2", f"reply {row_number}: unsupported ACK protocol")
            session = reply["backend_session_id"]
            sequence = integer(reply["mutation_seq"], "mutation_seq")
            require(bool(session) and sequence == sequences.get(session, 0) + 1, f"reply {row_number}: session sequence gap/reuse")
            sequences[session] = sequence
            require(json.loads(identity) == [session, sequence], f"reply {row_number}: stored identity differs from envelope")
            counts["mutations"] += 1
            if reply.get("operation") != "advance":
                continue
            counts["advances"] += 1
            body = reply.get("result")
            require("__nonfinite__" not in payload, f"advance {sequence}: nonfinite execution evidence")
            require(reply.get("ok") is True and isinstance(body, dict), f"advance {sequence}: missing/failed result")
            require(body.get("transition_valid") is True and body.get("physics_count_exact") is True,
                    f"advance {sequence}: invalid transition or uncertain physics count")
            require(not body.get("partial_control_step"), f"advance {sequence}: partial control step")
            require(body.get("backend_session_id") == session and body.get("mutation_seq") == sequence,
                    f"advance {sequence}: nested identity mismatch")
            episode = body["episode_id"]
            m = integer(body["executed_control_steps"], "executed_control_steps")
            physics = integer(body["executed_physics_steps"], "executed_physics_steps")
            begin, end = integer(body["control_tick_begin"], "begin tick"), integer(body["control_tick_end"], "end tick")
            require(physics == 4*m and end-begin == 12*m, f"advance {sequence}: tick/physics != actual m")
            trace = body["trace"]
            require(len(trace) == m, f"advance {sequence}: len(trace)={len(trace)} != m={m}")
            pair = (session, episode)
            require(begin == episode_end.get(pair, 0), f"advance {sequence}: episode tick gap")
            episode_end[pair] = end
            for i, row in enumerate(trace):
                start, tick = begin + 12*i, begin + 12*(i+1)
                require(row.get("tick") == tick and row.get("control_tick_begin") == start, f"advance {sequence}: trace tick gap")
                require(row.get("backend_session_id") == session and row.get("mutation_seq") == sequence and row.get("episode_id") == episode,
                        f"advance {sequence}: trace identity mismatch")
                require(row.get("transition_valid") is True and row.get("state_valid") is True and row.get("completed_physics_steps") == 4,
                        f"advance {sequence}: incomplete post-physics record at {tick}")
                samples = row["physics_substeps"]
                require(len(samples) == 4, f"advance {sequence}: expected four actual reward substeps")
                for j, sample in enumerate(samples, 1):
                    require(sample["physics_tick"] == start + 3*j, f"advance {sequence}: substep clock mismatch")
                    near(sample["dt_s"], .005, "physics dt")
                    require(all(k in sample for k in ("joint_vel_gmt", "joint_position_target", "physical_diagnostics")), "missing actual substep state")
                    physical = sample["physical_diagnostics"]
                    required = ("body_link_lin_vel_w", "joint_effort_limits_nm",
                                "joint_velocity_limits_rad_s", "foot_net_contact_forces_w_n", "foot_min_support_clearance_m")
                    require(all(k in physical for k in required), f"advance {sequence}: physical proxy evidence missing")
                    require('applied_joint_torque_nm' in physical or 'pd_torque_estimate_nm' in physical, 'PD torque estimate missing')
                    require("__nonfinite__" not in json.dumps(sample), f"advance {sequence}: nonfinite physics substep")
                key = (session, episode, tick)
                require(key not in ticks, "executed control step journaled twice")
                ticks[key] = sequence
            counts["control_steps"] += m
            counts["physics_steps"] += physics
    require(counts["mutations"] > 0, "empty execution journal")
    return dict(counts=dict(counts), sessions=sequences, episodes=len(episode_end), ticks=ticks)


def check_count(rows, expected, partial, summary):
    need(rows, "rollout")
    require(isinstance(rows, (list, tuple)) and all(isinstance(row, UpperTransition) for row in rows), "rollout must contain real UpperTransition records")
    require(len(rows) <= expected, f"rollout has more than {expected} transitions")
    if partial:
        raise NotRun(f"only rollout.partial.pt exists: {len(rows)}/{expected} complete decisions")
    require(len(rows) == expected, f"rollout requires {expected}, got {len(rows)}")
    if summary is not None:
        require(summary.get("collected_upper_transitions") == expected, "summary/rollout count mismatch")
    return dict(upper_transitions=len(rows))


def check_reward_arithmetic(detail, reward_config):
    """版本化复核积分与门控；保留旧v1归档审计，不把旧结果冒充新公式。"""
    version = detail.get('version')
    require(version in ('stage9.execution_reward.v1', 'stage9.execution_reward.v2'), 'unknown reward version')
    v2 = version.endswith('.v2')
    require(v2 == (reward_config.get('version') == 'stage9.execution_reward.v2'), 'reward/config version mismatch')
    names = V2_WEIGHT_KEYS if v2 else COMPONENTS
    require(set(detail['components']) == set(names), 'reward component set mismatch')
    gate, intensity = 1., None
    if v2:
        activity = detail['activity']
        require(activity.get('valid') is True and activity.get('target_source'), 'paired target activity unavailable')
        actual = number(activity['actual_activity_rad_s'], 'actual activity')
        target = number(activity['target_activity_rad_s'], 'target activity')
        require(actual >= 0 and target >= 0, 'negative activity')
        cfg = reward_config['activity']
        gate = 1. if target <= cfg['inactive_target_rad_s'] else min(1., actual/(cfg['full_gate_ratio']*target+cfg['epsilon']))
        near(activity['gate'], gate, 'activity gate formula')
        ratio = math.log((actual+cfg['intensity_epsilon'])/(target+cfg['intensity_epsilon']))/math.log(cfg['intensity_log_ratio'])
        intensity = math.exp(-min(abs(ratio), 1e150)**2)
        near(activity['intensity_score'], intensity, 'paired intensity formula')
        for name in ('consistency', 'power'):
            diagnostic = detail['diagnostics'][name]
            require(diagnostic.get('valid') is True, f'invalid {name} diagnostic')
            near(diagnostic['reward_weight'], 0., f'{name} must not be a reward')
        consistency = detail['diagnostics']['consistency']
        for key, tolerance in reward_config['consistency'].items():
            require(abs(number(consistency['raw'][key], key)) <= tolerance, 'reference consistency gate exceeded')
    counts, unavailable, rate = Counter(), Counter(), 0.
    for name in names:
        component = detail['components'][name]
        require(component['enabled'] is True, f'{name} reward disabled')
        score = number(component['score'], f'{name} score')
        require(0. <= score <= 1., f'{name} must be bounded in [0,1]')
        if v2:
            key = V2_WEIGHT_KEYS[name]
            weight = reward_config[key]*(-1 if 'penalty' in key else 1)
        else:
            weight = reward_config['weights'][name]
        multiplier = gate if v2 and name in ('track', 'music') else 1.
        near(component['weight'], weight, f'{name} weight')
        if v2:
            near(component['gate'], multiplier, f'{name} activity gate')
            require(isinstance(component.get('raw'), dict) and isinstance(component.get('normalized'), dict), 'raw/normalized evidence missing')
            near(component['integrated_reward'], component['weighted_rate']*.02, f'{name} integrated reward')
        near(component['weighted_rate'], score*weight*multiplier, f'{name} weighted rate')
        if not component['valid']:
            reason = component.get('raw', {}).get('reason')
            first_cmd = v2 and name == 'cmd' and component['raw'].get('first_step') is True
            old_music = not v2 and name == 'music' and reason in ('insufficient_causal_history', 'no_music_beats')
            require(first_cmd or old_music, f'invalid {name} reward: {reason}')
            near(score, 0., 'unsupported reward must have zero score')
            unavailable['first_cmd_step' if first_cmd else reason] += 1
        else:
            counts[name] += 1
        rate += component['weighted_rate']
    if v2:
        near(detail['reward_rate'], rate, 'reward rate sum')
    near(detail['reward'], rate*.02, 'reward dt integral')
    return counts, unavailable


def check_rollout(rows, journal, config, *, partial=False):
    need(rows, "rollout"); need(journal, "journal"); need(config, "resolved config")
    require(len(rows) > 0, "empty rollout")
    versions, used, boundaries = set(), set(), []
    counts, music_invalid = Counter(), Counter()
    bootstrap, continuation = [], [False]*len(rows)
    for index, item in enumerate(rows):
        require(isinstance(item, UpperTransition), "rollout item type is not UpperTransition")
        item.validate()
        require(item.transition_valid, f"row {index}: invalid transition entered training rollout")
        require(item.chain.shape[0] == config["stage9"]["denoising_steps"] + 1, "denoising step count mismatch")
        versions.add(item.identity["policy_version"])
        require(item.identity["backend_session_id"] in journal["sessions"], "rollout refers to unknown backend session")
        near(float(item.context["decision_time"][0]), item.control_tick_begin/600, "sample decision time")
        if item.next_context is not None:
            near(float(item.next_context["decision_time"][0]), item.control_tick_end/600, "next decision time")
        remaining = number(item.metadata["remaining_music_seconds"], "remaining music seconds")
        require(remaining >= 0., "negative remaining music duration")
        near(item.metadata["next_remaining_music_seconds"], max(0., remaining-item.executed_control_steps*.02),
             "remaining music duration", atol=1e-7)
        details = item.metadata["reward_details"]
        require(len(details) == item.executed_control_steps, f"row {index}: reward detail count != m")
        reconstructed = []
        for step, detail in enumerate(details, 1):
            tick = item.control_tick_begin + 12*step
            key = (item.identity["backend_session_id"], item.identity["episode_id"], tick)
            require(key in journal["ticks"], f"row {index}: executed reward at {tick} has no journal evidence")
            require(key not in used, "same executed consequence used by multiple upper transitions")
            used.add(key)
            require(detail["tick"] == tick and detail["episode_id"] == item.identity["episode_id"], "reward identity/time mismatch")
            require(detail.get("transition_valid") is True and not detail.get("errors"), "invalid reward diagnostics")
            require(detail.get("reward_is_integrated") is True, "reward is not dt-integrated")
            near(detail["dt_s"], .02, "control reward dt")
            supported, unavailable = check_reward_arithmetic(detail, config['stage9']['reward'])
            counts.update(supported)
            music_invalid.update(unavailable)
            reconstructed.append(detail["reward"])
        reward_config = config['stage9']['reward']
        rejection = item.metadata.get('rejection')
        if reward_config.get('version') == 'stage9.execution_reward.v2':
            rejected_action = bool(rejection and rejection.get('policy_penalty',
                rejection.get('code') in {'invalid_qpos','invalid_quaternion','invalid_plan_output','known_source_changed'}))
            penalty = -reward_config['rejected_plan_penalty'] if rejected_action else 0.
            penalty -= reward_config['failure_penalty'] if item.terminated and item.reason not in ('music_end','task_complete') else 0.
        else:
            penalty = (-1. if rejection else 0.)
            penalty -= 5. if item.terminated and item.reason != 'music_end' else 0.
        near(item.metadata["event_penalty_total"], penalty, "once-only event penalty")
        event = item.metadata.get("event_reward", 0.)
        if reconstructed:
            near(event, 0., "nonzero-step event duplicated separately")
            reconstructed[-1] += penalty
        else:
            near(event, penalty, "zero-step event missing")
        require(torch.allclose(item.rewards.double(), torch.tensor(reconstructed, dtype=torch.float64), atol=1e-9, rtol=1e-9),
                f"row {index}: stored rewards differ from dt-integrated details plus event")
        bootstrap.append(not item.terminated and item.next_context is not None)
        if index:
            previous = rows[index-1]
            same = all(previous.identity[k] == item.identity[k] for k in ("backend_session_id", "episode_id", "policy_version"))
            if not previous.terminated and not previous.truncated:
                require(same and previous.control_tick_end == item.control_tick_begin, "nonterminal episode discontinuity/gap")
                require(previous.next_context is not None and all(torch.equal(previous.next_context[k], item.context[k]) for k in item.context),
                        "next observation/committed-prefix conditions differ from following sample")
                continuation[index-1] = True
            else:
                require(not same, "collection continued in a terminated/truncated episode")
                boundaries.append(index)
    require(len(versions) == 1, "mixed policy versions in one rollout")
    if not partial:
        require(rows[-1].terminated or rows[-1].truncated, "final rollout row has no collection boundary")
    return dict(policy_version=next(iter(versions)), control_steps=len(used), bootstrap_mask=bootstrap,
                continuation_mask=continuation, episode_boundaries=boundaries,
                reward_valid_counts=dict(counts), music_invalid_counts=dict(music_invalid))


def check_targets(rows, targets, rollout, config):
    need(rows, "rollout"); need(targets, "fixed targets"); need(rollout, "rollout integrity"); need(config, "config")
    n = len(rows)
    vectors = {}
    for name in ("old_values", "next_values", "discounted_rewards", "advantages_raw", "returns", "advantages", "valid"):
        value = torch.as_tensor(targets[name]).detach().cpu()
        require(value.shape == (n,) and torch.isfinite(value).all(), f"fixed target {name} shape/nonfinite")
        vectors[name] = value
    require(vectors["valid"].dtype == torch.bool and vectors["valid"].all(), "invalid targets entered optimizer")
    old = torch.tensor([row.old_value for row in rows], dtype=torch.float64)
    nxt = torch.tensor([row.next_value for row in rows], dtype=torch.float64)
    require(torch.equal(old, vectors["old_values"]) and torch.equal(nxt, vectors["next_values"]), "rollout values differ from frozen target values")
    require(torch.allclose(vectors["returns"]-vectors["advantages_raw"], old, atol=1e-10, rtol=1e-9),
            "old_value != fixed target - raw advantage")
    gamma = config["stage9"]["gamma_upper"] ** (1/25)
    lam = config["stage9"]["lambda_upper"] ** (1/25)
    near(targets["gamma_low"], gamma, "gamma per control step")
    near(targets["lambda_low"], lam, "lambda per control step")
    discounted = [sum(gamma**j*float(r) for j, r in enumerate(row.rewards)) +
                  gamma**max(row.executed_control_steps-1, 0)*row.metadata.get("event_reward", 0.) for row in rows]
    raw = [0.]*n
    for i in range(n-1, -1, -1):
        m = rows[i].executed_control_steps
        raw[i] = discounted[i]-float(old[i]) + (gamma**m*float(nxt[i]) if rollout["bootstrap_mask"][i] else 0.)
        if rollout["continuation_mask"][i]:
            raw[i] += (gamma*lam)**m*raw[i+1]
    advantage = torch.tensor(raw, dtype=torch.float64)
    expected = dict(discounted_rewards=torch.tensor(discounted, dtype=torch.float64), advantages_raw=advantage,
                    returns=advantage+old, advantages=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8))
    errors = {}
    for name, value in expected.items():
        errors[name] = float((vectors[name]-value).abs().max())
        require(torch.allclose(vectors[name], value, atol=1e-9, rtol=1e-8), f"independent semi-Markov recomputation differs: {name}")
    return dict(max_errors=errors, gamma_low=gamma, lambda_low=lam, old_values_match=True)


def check_updates(summary, config):
    need(summary, "main summary"); need(config, "config")
    if "actor" not in summary or "critic" not in summary:
        raise NotRun("Actor/Critic update evidence missing")
    actor, critic = summary["actor"], summary["critic"]
    require(actor.get("parameters_changed") is True and actor.get("critic_unchanged") is True, "Actor update changed wrong network or did not occur")
    require(critic.get("parameters_changed") is True and critic.get("actor_unchanged") is True, "Critic update changed wrong network or did not occur")
    require(actor["optimizer_steps"] == 1, "expected one Actor optimizer step")
    for key in ("ppo_only_gradient_norm", "total_gradient_norm"):
        require(number(actor[key], key) > 0, f"{key} is not positive")
    require(actor.get("ratio_scope") == "before_single_optimizer_step", "PPO ratio measurement scope differs")
    modules = actor["ppo_module_gradients"]
    norms = [number(value, key) for key, value in modules.items()]
    require(bool(norms) and min(norms) >= 0 and max(norms) > 0, "no finite positive module PPO-only gradient")
    for key in ("losses", "gradient_norms"):
        require(len(critic[key]) == 20 and all(math.isfinite(number(v, key)) for v in critic[key]), f"Critic {key} missing/nonfinite")
    require(any(v > 0 for v in critic["gradient_norms"]), "Critic value-loss gradients are zero")
    if config["stage9"]["bc_weight"] > 0:
        bc = actor.get("bc")
        require(isinstance(bc, dict) and bc.get("bc_update_steps") == 1, "enabled original-pair supervision was not used")
        near(bc["weight"], config["stage9"]["bc_weight"], "BC weight")
        number(bc["loss"], "BC loss")
    result = dict(ppo_only_gradient_norm=actor["ppo_only_gradient_norm"], total_gradient_norm=actor["total_gradient_norm"],
                  actor_updates=1, critic_updates=20, actor_critic_isolation=True)
    calibration = check_lr_calibration(summary, config)
    if calibration is not None:
        result['lr_calibration'] = calibration
    return result


def check_kl(summary, config):
    need(summary, "main summary"); need(config, "config")
    if "kl" not in summary:
        raise NotRun("post-update KL was not measured")
    kl = summary["kl"]
    limit = number(config["stage9"]["kl_stop_joint"], "KL threshold")
    require(0 < limit <= .02, "joint KL threshold exceeds bounded acceptance contract")
    require(kl["joint_kl_scope"] == "sum_free_coordinates_per_internal_transition_then_mean", "KL uses wrong reduction")
    value = number(kl["mean_joint_kl"], "mean joint KL")
    require(-1e-10 <= value <= limit, f"joint KL {value} exceeds {limit}")
    for name in ("p95_joint_kl", "max_joint_kl", "mean_per_dimension_kl", "mean_chain_joint_kl"):
        require(number(kl[name], name) >= -1e-10, f"negative {name}")
    near(kl["mean_chain_joint_kl"], value*config["stage9"]["denoising_steps"], "chain vs transition KL", atol=1e-7)
    return dict(mean_joint_kl=value, threshold=limit, reduction=kl["joint_kl_scope"])


def check_identity(checkpoint, phases):
    need(checkpoint, "checkpoint metadata")
    identity = checkpoint["identity"]
    for name, phase in phases.items():
        need(phase["source"], f"{name} source"); need(phase["preflight"], f"{name} preflight"); need(phase["config"], f"{name} config")
        require(identity["source_manifest_sha256"] == phase["source"]["source_manifest_sha256"], f"{name}: checkpoint/source mismatch")
        require(identity["assets"] == phase["preflight"]["asset_sha256"], f"{name}: checkpoint/assets mismatch")
        config = phase["config"]
        require(identity["reward"] == config["stage9"]["reward"], f"{name}: checkpoint/reward mismatch")
        if config['stage9'].get('actor_lr_candidates'):
            require(all(key in identity['training_contract'] for key in ('actor_lr_candidates', 'actor_lr_selection')),
                    f'{name}: learning-rate candidate identity missing')
        require(all(config["stage9"][key] == value for key, value in identity["training_contract"].items()), f"{name}: training contract mismatch")
        for key in ("environment", "termination"):
            require(identity[key] == config[key], f"{name}: {key} mismatch")
    require(set(identity["bc_manifests"]) == {"AIST++", "AIOZ-GDANCE", "FineDance", "Mine"}, "BC train provenance missing")
    require(bool(identity["music_selection_sha256"]), "music selection identity missing")
    require(checkpoint["config"]["stage9"]["reward"] == identity["reward"], "checkpoint internal reward mismatch")
    return dict(source_and_assets_match=True, original_paired_sources=sorted(identity["bc_manifests"]))


def check_resume(checkpoint, phases):
    need(checkpoint, "checkpoint metadata")
    main, resume = phases["main"], phases["resume"]
    for label, phase in phases.items():
        need(phase["rollout_audit"], f"{label} rollout integrity"); need(phase["journal"], f"{label} journal")
    summary = need(resume["summary"], "resume summary")
    proof = summary["resume"]
    require(proof.get("restored_full_state") is True and proof.get("old_buffer_discarded") is True, "resume did not restore full state/discard old buffer")
    require(proof["restored_actor_updates"] == checkpoint["state"]["actor_updates"], "restored update count mismatch")
    old, new = main["rollout_audit"]["policy_version"], resume["rollout_audit"]["policy_version"]
    require(new == old+1 == checkpoint["state"]["policy_version"], "resume policy version mismatch")
    require(not (set(main["journal"]["sessions"]) & set(resume["journal"]["sessions"])), "resume reused a physical worker session")
    require(proof["new_backend_session_id"] in resume["journal"]["sessions"], "resume report session not found in journal")
    require("actor" not in summary and "critic" not in summary, "resume-check unexpectedly updated parameters")
    result = dict(main_policy_version=old, resume_policy_version=new, new_physics_session=True)
    if checkpoint.get('lr_calibration') is not None:
        for name in ('actor', 'critic'):
            check_group_lrs(proof.get(f'{name}_optimizer_lrs'), checkpoint['optimizer_lrs'][name], f'restored {name}')
        result['optimizer_lrs'] = checkpoint['optimizer_lrs']
    return result


def check_budget(phases, ledger, checkpoint):
    snapshots = []
    for name in ("main", "resume"):
        summary = phases[name]["summary"]
        if summary is not None:
            snapshots.append((name, summary["budget"]))
    if checkpoint is not None:
        snapshots.insert(1, ("checkpoint", checkpoint["state"]["budget"]))
    if ledger is not None:
        snapshots.append(("ledger", ledger))
    if not snapshots:
        raise NotRun("no budget snapshot")
    main_actor = (phases['main']['summary'] or {}).get('actor')
    attempts = 0
    if main_actor is not None:
        calibration = main_actor.get('lr_calibration')
        attempts = integer(calibration['attempt_count'], 'budget optimizer attempts') if calibration is not None else 1
        require(1 <= attempts <= HARD_LIMITS['iterations'], 'invalid optimizer attempt budget')
    previous = None
    for label, current in snapshots:
        require(set(current["limits"]) == set(HARD_LIMITS) and set(current["used"]) == set(HARD_LIMITS), f"{label}: incomplete budget counters")
        for key, hard in HARD_LIMITS.items():
            limit = integer(current["limits"][key], f"{label} {key} limit")
            used = integer(current["used"][key], f"{label} {key} used")
            require(0 < limit <= hard and used <= limit, f"{label}: {key} hard limit exceeded")
            require(sum(integer(p.get(key, 0), f"phase {key}") for p in current["phases"].values()) == used, f"{label}: phase totals != {key} used")
        require(current['used']['iterations'] >= attempts, f'{label}: budget undercounts optimizer attempts')
        require(current["limits"]["physics_steps"] == 4*current["limits"]["control_steps"], "physics budget limit != 4*control limit")
        for name, cap in PHASE_LIMITS.items():
            require(current["phases"].get(name, {}).get("control_steps", 0) <= cap, f"{label}: {name} phase limit exceeded")
        if previous is not None:
            require(previous["limits"] == current["limits"], "budget limits changed during resume")
            require(all(current["used"][key] >= previous["used"][key] for key in HARD_LIMITS), f"{label}: budget rolled back")
            for name, values in previous["phases"].items():
                require(all(current["phases"].get(name, {}).get(key, 0) >= value for key, value in values.items()), f"{label}: phase budget rolled back")
        previous = current
    observed = Counter(iterations=attempts) if attempts else Counter()
    for phase in phases.values():
        if phase["journal"] is not None:
            for key in ("control_steps", "physics_steps"):
                observed[key] += phase["journal"]["counts"].get(key, 0)
        if phase["rows"] is not None:
            observed["generations"] += len(phase["rows"])
    for key, value in observed.items():
        require(previous["used"][key] >= value, f"budget undercounts observed {key}")
    main, resume = phases["main"], phases["resume"]
    if main["summary"] is not None and resume["summary"] is not None and resume["journal"] is not None:
        for key in ("control_steps", "physics_steps"):
            delta = resume["summary"]["budget"]["used"][key]-main["summary"]["budget"]["used"][key]
            require(delta >= resume["journal"]["counts"].get(key, 0), f"resume budget undercounts its actual {key}")
    return dict(snapshots=[name for name, _ in snapshots], limits=previous["limits"], used=previous["used"],
                observed_lower_bound=dict(observed), unobserved_consumption={key: previous["used"][key]-value for key, value in observed.items()},
                note="shared ledger may include earlier failed attempts/calibration; lower bounds do not identify every generation")


def audit_run(run_dir, allow_incomplete=False, budget_file=None):
    root = Path(run_dir).expanduser().resolve()
    audit = Audit(root, allow_incomplete)
    phases = {}
    for name, subdir, expected, mode in (("main", "acceptance", 64, "train"), ("resume", "resume_check", 16, "resume-check")):
        directory = root/subdir
        phase = {}
        for key, filename in (("summary", "summary.json"), ("source", "source_identity.json"), ("preflight", "preflight.json")):
            phase[key] = audit.check(f"{name}.{key}.read", lambda p=directory/filename: read_json(p))
        phase["config"] = audit.check(f"{name}.config.read", lambda p=directory/"resolved_config.yaml": yaml.safe_load(p.read_text()))
        audit.check(f"{name}.completed", lambda: check_summary(phase["summary"], mode))
        audit.check(f"{name}.probability_before_update", lambda: check_probability(phase["summary"]))
        audit.check(f"{name}.frozen", lambda: check_frozen(phase["summary"], phase["source"], phase["preflight"]))
        phase["journal"] = audit.check(f"{name}.journal", lambda: check_journal(directory/"execution_journal.sqlite"))
        partial = not (directory/"rollout.pt").exists() and (directory/"rollout.partial.pt").exists()
        path = directory/("rollout.partial.pt" if partial else "rollout.pt")
        phase["rows"] = audit.check(f"{name}.rollout.read", lambda: load_torch(path))
        audit.check(f"{name}.rollout.count", lambda: check_count(phase["rows"], expected, partial, phase["summary"]))
        phase["rollout_audit"] = audit.check(f"{name}.rollout.integrity", lambda: check_rollout(phase["rows"], phase["journal"], phase["config"], partial=partial))
        if name == "main":
            targets = audit.check("main.fixed_targets.read", lambda: load_torch(directory/"fixed_targets.pt"))
            audit.check("main.fixed_targets.independent_gae", lambda: check_targets(phase["rows"], targets, phase["rollout_audit"], phase["config"]))
            audit.check("main.network_updates", lambda: check_updates(phase["summary"], phase["config"]))
            audit.check("main.joint_kl", lambda: check_kl(phase["summary"], phase["config"]))
        phases[name] = phase
    checkpoint = audit.check("checkpoint.metadata", lambda: checkpoint_metadata(root, phases["main"]["summary"]))
    audit.check("checkpoint.identity_main", lambda: check_identity(checkpoint, {"main": phases["main"]}))
    audit.check("checkpoint.identity_resume", lambda: check_identity(checkpoint, phases))
    audit.check("resume.full_state_and_new_physics", lambda: check_resume(checkpoint, phases))
    ledger_path = Path(budget_file).expanduser().resolve() if budget_file is not None else root/"budget.json"
    ledger = audit.check("budget.ledger.read", lambda: read_json(ledger_path))
    audit.check("budget.cumulative", lambda: check_budget(phases, ledger, checkpoint))
    # 输出仅保留可复核的小型统计，不把完整 rollout、checkpoint 元数据或 tick 索引再次序列化。
    for check in audit.checks:
        if "details" not in check:
            continue
        if check["name"].endswith(".read"):
            check["details"] = dict(loaded=True)
        elif check["name"].endswith(".journal"):
            check["details"] = {k: v for k, v in check["details"].items() if k != "ticks"}
        elif check["name"] == "checkpoint.metadata":
            check["details"] = {k: checkpoint[k] for k in ("path", "restore_environment", "rng_keys", "sampler_names", "optimizer_layout_names")}
            check["details"]["state"] = checkpoint["state"]
            if checkpoint.get('lr_calibration') is not None:
                check['details'].update(optimizer_lrs=checkpoint['optimizer_lrs'], lr_calibration=checkpoint['lr_calibration'])
    counts = Counter(check["status"] for check in audit.checks)
    status = "failed" if counts["failed"] else "incomplete" if counts["not_run"] else "passed"
    return dict(schema="genmo.closedloop.stage9.audit.v1", run_dir=str(root), budget_file=str(ledger_path), status=status,
                stage9_passed=status == "passed", allow_incomplete=bool(allow_incomplete),
                check_counts=dict(counts), checks=audit.checks,
                evidence_boundary=["No Actor/GMT/simulator/GPU loaded or executed.",
                    "Checkpoint tensor payloads are memory mapped; only metadata is audited.",
                    "Gaussian ratio, PPO gradient and frozen-network reports are cross-checked; not recomputed by rerunning networks.",
                    "Reward component arithmetic and time integration are audited; reward quality/convergence is not established.",
                    "Idempotency identities and recorded consequences are audited; no network retries are injected."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="仅创建新的审计 JSON，拒绝覆盖已有文件")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--budget-file", type=Path, help="显式指定跨多次尝试的共享账本；缺省为 run-dir/budget.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists; refusing to overwrite: {args.output}")
    report = audit_run(args.run_dir, allow_incomplete=args.allow_incomplete, budget_file=args.budget_file)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output), check_counts=report["check_counts"]), ensure_ascii=False))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
