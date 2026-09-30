"""独立审计入口的 CPU 证据验收与故障注入。

测试创建真实 UpperTransition、ExecutionReward 和 StepJournal 产物，而非用字典
冒充训练 Buffer。小型模型状态只用于完整 checkpoint 的元数据容器，测试不构建
Actor/Critic、不初始化 GPU、不运行物理仿真。主采集 64 条与恢复 16 条同时包含
连续转移、行政截断、任务结束和一次性失败代价，以检查 bootstrap 与 GAE 递推分离。

负面样例覆盖落盘后缺失 trace、200Hz 子步时刻错误、SHA/序号篡改、错误 dt、
事件重复惩罚、旧价值未保存、优势递推错误、策略版本串用、GMT 非冻结、联合 KL
超限和预算回退。还检查 WAL 未 checkpoint 时的只读快照、源文件无变化，以及
大 checkpoint 的 mmap 参数和显式输出拒绝覆盖语义。所有文件仅写 pytest 临时目录。
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3

import numpy as np
import pytest
import torch
import yaml

from gem.closedloop.dppo.buffer import StepJournal, UpperTransition
from gem.closedloop.dppo.returns import compute_gae
from gem.closedloop.dppo.rewards import DEFAULT_CONFIG, ExecutionReward
from tools.eval import audit_closedloop_dppo as audit


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def context(tick):
    time = tick/600
    return dict(music_features=torch.zeros(1, 120, 35), music_valid=torch.ones(1, 120, dtype=torch.bool),
                proprio_history=torch.zeros(1, 50, 48), proprio_history_valid=torch.ones(1, 50, dtype=torch.bool),
                proprio_history_times=((torch.arange(50, dtype=torch.float64)-49)/50+time)[None],
                known_qpos30=torch.zeros(1, 120, 30), known_qpos30_mask=torch.zeros(1, 120, 30, dtype=torch.bool),
                future_valid=torch.ones(1, 120, dtype=torch.bool),
                future_times=(torch.arange(120, dtype=torch.float64)/30+time)[None],
                decision_time=torch.tensor([time], dtype=torch.float64))


def physical_step(session, sequence, episode, start):
    names = ["root", "left_foot", "right_foot"]
    physical = dict(body_names=names, foot_body_names=names[1:], body_link_lin_vel_w=np.zeros((3, 3)),
                    applied_joint_torque_nm=np.zeros(21), joint_effort_limits_nm=np.ones(21)*40,
                    joint_velocity_limits_rad_s=np.ones(21)*10, foot_min_support_clearance_m=np.zeros(2),
                    foot_net_contact_forces_w_n=np.array([[0., 0., 10.], [0., 0., 10.]]),
                    contact_body_names=names, net_contact_forces_w_n=np.array([[0., 0., 0.], [0., 0., 10.], [0., 0., 10.]]))
    return dict(backend_session_id=session, mutation_seq=sequence, episode_id=episode,
                tick=start+12, control_tick_begin=start, completed_physics_steps=4, transition_valid=True, state_valid=True,
                actual_joint_pos_gmt=np.zeros(21), actual_joint_vel_gmt=np.zeros(21),
                actual_qpos=np.r_[0., 0., 1., 1., 0., 0., 0., np.zeros(21)], actual_root_ang_vel_b=np.zeros(3),
                joint_position_target=np.zeros(21), reference=dict(joint_pos=np.zeros(21), joint_vel=np.zeros(21),
                    body_pos_w=np.array([[0., 0., 1.]]*3), body_quat_w=np.array([[1., 0., 0., 0.]]*3)),
                errors=dict(root_height_error_m=0., non_yaw_orientation_error_rad=0., yaw_error_rad=0.,
                            end_effector_relative_height_error_m=0.),
                reference_consistency=dict(valid=True, joint_vel_rms_rad_s=0., root_lin_vel_rms_m_s=0., root_ang_vel_rms_rad_s=0.),
                physical_diagnostics=physical,
                physics_substeps=[dict(physics_tick=start+j*3, dt_s=.005, joint_vel_gmt=np.zeros(21),
                                       joint_position_target=np.zeros(21), physical_diagnostics=physical) for j in range(1, 5)])


def envelope(session, sequence, operation, result):
    return dict(execution_protocol="ack.v2", backend_session_id=session, mutation_seq=sequence,
                operation=operation, ok=True, result=result, error=None)


def phase_artifacts(directory, count, version, config):
    directory.mkdir()
    session, rows, seq = f"session-{version}", [], 0
    # 相同张量的共享 storage 减少磁盘夹具体积；真实记录仍使用 UpperTransition 校验。
    chain = torch.zeros(21, 120, 30)
    with StepJournal(directory/"execution_journal.sqlite") as journal:
        for i in range(count):
            episode, start = f"episode-{version}-{i//2}", i%2*12
            if i%2 == 0:
                seq += 1
                journal.append_result(envelope(session, seq, "reset_episode", dict(episode_id=episode)))
                reward = ExecutionReward(music_features=np.zeros((150, 35)))
            seq += 1
            step = physical_step(session, seq, episode, start)
            feedback = dict(backend_session_id=session, mutation_seq=seq, episode_id=episode,
                            transition_valid=True, physics_count_exact=True, partial_control_step=None,
                            control_tick_begin=start, control_tick_end=start+12, executed_control_steps=1,
                            executed_physics_steps=4, trace=[step])
            reply = envelope(session, seq, "advance", feedback)
            assert journal.append_result(reply)
            assert not journal.append_result(reply)  # 正常重发不会重复落盘。
            detail = reward.evaluate_step(step)
            terminal, truncated = i%4 == 1, i%4 == 3
            penalty = -5. if terminal else 0.
            row = UpperTransition(identity=dict(run_id="fixture", backend_session_id=session, episode_id=episode,
                decision_id=i%2, policy_version=version), context=context(start),
                next_context=None if terminal else context(start+12), chain=chain,
                old_log_prob=torch.zeros(20, dtype=torch.float64), free_mask=torch.ones(120, 30, dtype=torch.bool),
                rewards=torch.tensor([detail["reward"]+penalty], dtype=torch.float64), old_value=i/100,
                next_value=0. if terminal else (i+1)/100, control_tick_begin=start, control_tick_end=start+12,
                executed_control_steps=1, executed_physics_steps=4, terminated=terminal, truncated=truncated,
                reason="physical_failure" if terminal else "collection_boundary" if truncated else None,
                metadata=dict(reward_details=[detail], remaining_music_seconds=5-start/600,
                              next_remaining_music_seconds=5-(start+12)/600, rejection=None,
                              event_reward=0., event_penalty_total=penalty))
            row.validate()
            rows.append(row)
    torch.save(tuple(rows), directory/"rollout.pt")
    source_files = [dict(repository="genmo_repo", relative_path="example.py", sha256="a"*64)]
    digest = hashlib.sha256(json.dumps([(r["repository"], r["relative_path"], r["sha256"]) for r in source_files],
                                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    source = dict(files=source_files, file_count=1, source_manifest_sha256=digest)
    write_json(directory/"source_identity.json", source)
    write_json(directory/"preflight.json", dict(ready=True, asset_sha256=dict(gmt="b"*64)))
    (directory/"resolved_config.yaml").write_text(yaml.safe_dump(config))
    probability = dict(passed=True, max_abs_log_probability_difference=0., max_abs_ratio_minus_one=0.,
                       max_abs_independent_gaussian_difference=0.)
    summary = dict(mode="train" if version == 0 else "resume-check", status="passed", exit_code=0,
                   collected_upper_transitions=count, probability_check=probability,
                   source_unchanged=dict(unchanged=True, initial_manifest_sha256=digest), original_assets_unchanged=True,
                   worker_shutdown=dict(gmt=dict(closed=True, policy_unchanged=True, runtime_parameters_unchanged=True, process_exit_code=0)))
    if version == 0:
        targets = compute_gae([r.rewards for r in rows], [r.old_value for r in rows], [r.next_value for r in rows],
                              [1]*count, [not r.terminated for r in rows], [i%2 == 0 for i in range(count)])
        targets.update(old_values=torch.tensor([r.old_value for r in rows], dtype=torch.float64),
                       next_values=torch.tensor([r.next_value for r in rows], dtype=torch.float64))
        torch.save(targets, directory/"fixed_targets.pt")
        summary.update(actor=dict(parameters_changed=True, critic_unchanged=True, optimizer_steps=1,
            ppo_only_gradient_norm=2., total_gradient_norm=3., ppo_module_gradients=dict(denoiser=2., history_encoder=.1),
            ratio_scope="before_single_optimizer_step", bc=dict(loss=.3, weight=.1, bc_update_steps=1)),
            critic=dict(parameters_changed=True, actor_unchanged=True, losses=[1.]*20, gradient_norms=[2.]*20),
            kl=dict(mean_joint_kl=.002, p95_joint_kl=.003, max_joint_kl=.04, mean_per_dimension_kl=1e-6,
                    mean_chain_joint_kl=.04, joint_kl_scope="sum_free_coordinates_per_internal_transition_then_mean"))
    else:
        summary["resume"] = dict(restored_full_state=True, old_buffer_discarded=True, restored_actor_updates=1,
                                  new_backend_session_id=session)
    return rows, source, summary


@pytest.fixture
def run_dir(tmp_path):
    config = dict(stage9=dict(reward=copy.deepcopy(DEFAULT_CONFIG), denoising_steps=20, gamma_upper=.99,
                  lambda_upper=.95, bc_weight=.1, kl_stop_joint=.02, actor_lr=1e-9), environment={}, termination={})
    root = tmp_path/"run"
    root.mkdir()
    _, source, main = phase_artifacts(root/"acceptance", 64, 0, config)
    _, _, resume = phase_artifacts(root/"resume_check", 16, 1, config)
    # 初始共享账本已经消耗 100 步，验证不能错误要求本目录 journal == 整个 campaign。
    def budget(resumed):
        phases = dict(diagnostic=dict(control_steps=100, physics_steps=400, generations=10),
                      main=dict(control_steps=64, physics_steps=256, generations=64), update=dict(iterations=1))
        if resumed:
            phases["resume"] = dict(control_steps=16, physics_steps=64, generations=16)
        return dict(limits=copy.deepcopy(audit.HARD_LIMITS),
                    used={k: sum(phase.get(k, 0) for phase in phases.values()) for k in audit.HARD_LIMITS}, phases=phases)
    main["budget"], resume["budget"] = budget(False), budget(True)
    path = root/"checkpoints"/"stage9_000001.pt"
    path.parent.mkdir()
    main["checkpoint"] = str(path)
    checkpoint = dict(version="genmo.closedloop.stage9.full_state.v1", actor={"weight": torch.zeros(2)},
        critic={"weight": torch.zeros(1)}, actor_optimizer={}, critic_optimizer={},
        state=dict(buffer_size=0, pending_plan=False, actor_updates=1, critic_updates=20, iteration=1,
                   policy_version=1, budget=main["budget"]),
        identity=dict(assets=dict(gmt="b"*64), source_manifest_sha256=source["source_manifest_sha256"],
            reward=config["stage9"]["reward"], training_contract=dict(gamma_upper=.99, lambda_upper=.95, actor_lr=1e-9),
            environment={}, termination={}, music_selection_sha256="c"*64,
            bc_manifests={name: "d"*64 for name in ("AIST++", "AIOZ-GDANCE", "FineDance", "Mine")}),
        config=config, rng={k: None for k in ("python", "numpy", "torch", "cuda", "generators", "generator_devices")},
        samplers=dict(music={}, bc={}), optimizer_layout=dict(actor={}, critic={}), restore_environment="new_worker_session_and_reset")
    torch.save(checkpoint, path)
    write_json(root/"acceptance"/"summary.json", main)
    write_json(root/"resume_check"/"summary.json", resume)
    write_json(root/"budget.json", budget(True))
    return root


def check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


def mutate_json(path, function):
    value = json.loads(path.read_text())
    function(value)
    write_json(path, value)


def mutate_reply(path, function, *, rehash=True):
    with sqlite3.connect(path) as connection:
        rowid, payload = connection.execute("SELECT rowid,payload FROM replies WHERE payload LIKE '%\"operation\":\"advance\"%' LIMIT 1").fetchone()
        value = json.loads(payload)
        function(value)
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if rehash:
            connection.execute("UPDATE replies SET payload=?,sha256=? WHERE rowid=?", (payload, hashlib.sha256(payload.encode()).hexdigest(), rowid))
        else:
            connection.execute("UPDATE replies SET payload=? WHERE rowid=?", (payload, rowid))


def test_complete_audit_mmap_no_input_changes_and_cli(run_dir, monkeypatch, tmp_path):
    before = {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.rglob("*") if p.is_file()}
    original = torch.load
    calls = []
    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", checked_load)
    report = audit.audit_run(run_dir)
    assert report["status"] == "passed", report
    assert all(c["mmap"] is True and c["map_location"] == "cpu" for c in calls)
    assert check(report, "main.rollout.integrity")["details"]["bootstrap_mask"][3]
    assert not check(report, "main.rollout.integrity")["details"]["continuation_mask"][3]
    assert check(report, "main.rollout.integrity")["details"]["music_invalid_counts"] == {"insufficient_causal_history": 64}
    assert check(report, "budget.cumulative")["details"]["unobserved_consumption"]["control_steps"] == 100
    json.dumps(report, allow_nan=False)
    after = {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.rglob("*") if p.is_file()}
    assert after == before
    output = tmp_path/"audit.json"
    assert audit.main(["--run-dir", str(run_dir), "--output", str(output)]) == 0
    with pytest.raises(SystemExit):
        audit.main(["--run-dir", str(run_dir), "--output", str(output)])


@pytest.mark.parametrize("change,error", [
    (lambda r: r["result"]["trace"].clear(), "len(trace)"),
    (lambda r: r["result"].update(executed_physics_steps=3), "tick/physics"),
    (lambda r: r["result"]["trace"][0]["physics_substeps"][2].update(physics_tick=12), "substep clock"),
    (lambda r: r["result"]["trace"][0]["physics_substeps"][0].update(dt_s=.02), "physics dt"),
    (lambda r: r.update(mutation_seq=99), "sequence gap"),
    (lambda r: r["result"]["trace"][0].update(state_valid=False), "incomplete post-physics"),
])
def test_physics_or_journal_corruption_is_not_waived(run_dir, change, error):
    mutate_reply(run_dir/"acceptance"/"execution_journal.sqlite", change)
    report = audit.audit_run(run_dir, allow_incomplete=True)
    assert report["status"] == "failed"
    assert error in check(report, "main.journal")["error"]


def test_payload_hash_detects_tampering(run_dir):
    mutate_reply(run_dir/"acceptance"/"execution_journal.sqlite", lambda r: r.update(extra="tampered"), rehash=False)
    assert "SHA256" in check(audit.audit_run(run_dir), "main.journal")["error"]


@pytest.mark.parametrize("mutation,error", [
    (lambda rows: rows[0].metadata["reward_details"][0].update(dt_s=.04), "control reward dt"),
    (lambda rows: rows[0].metadata.update(event_reward=-1.), "duplicated separately"),
    (lambda rows: rows[0].metadata.update(event_penalty_total=-1.), "once-only event"),
    (lambda rows: rows[0].metadata.update(next_remaining_music_seconds=4.97), "remaining music duration"),
    (lambda rows: rows[0].rewards.add_(.1), "stored rewards differ"),
    (lambda rows: rows[1].identity.update(policy_version=9), "discontinuity"),
    (lambda rows: rows[1].context["music_features"].add_(1), "next observation"),
    (lambda rows: rows[0].metadata["reward_details"][0]["components"]["track"].update(valid=False), "invalid track"),
])
def test_rollout_consequences_and_masks_are_checked(run_dir, mutation, error):
    path = run_dir/"acceptance"/"rollout.pt"
    rows = torch.load(path, weights_only=False)
    mutation(rows)
    torch.save(rows, path)
    report = audit.audit_run(run_dir)
    assert report["status"] == "failed"
    assert error in check(report, "main.rollout.integrity")["error"]


@pytest.mark.parametrize("mutation,error", [
    (lambda data: data["old_values"].zero_(), "rollout values"),
    (lambda data: data["advantages_raw"].add_(1.), "old_value !="),
    (lambda data: data["advantages"].zero_(), "independent semi-Markov"),
    (lambda data: data.update(gamma_low=.99), "gamma per control step"),
])
def test_fixed_targets_are_independently_recomputed(run_dir, mutation, error):
    path = run_dir/"acceptance"/"fixed_targets.pt"
    data = torch.load(path, weights_only=False)
    mutation(data)
    torch.save(data, path)
    assert error in check(audit.audit_run(run_dir), "main.fixed_targets.independent_gae")["error"]


def test_missing_resume_is_not_run_but_existing_kl_failure_still_fails(run_dir):
    import shutil
    shutil.rmtree(run_dir/"resume_check")
    report = audit.audit_run(run_dir, allow_incomplete=True)
    assert report["status"] == "incomplete"
    assert check(report, "resume.summary.read")["status"] == "not_run"
    assert audit.audit_run(run_dir)["status"] == "failed"
    mutate_json(run_dir/"acceptance"/"summary.json", lambda s: s["kl"].update(mean_joint_kl=2964.))
    report = audit.audit_run(run_dir, allow_incomplete=True)
    assert report["status"] == "failed"
    assert "exceeds" in check(report, "main.joint_kl")["error"]


@pytest.mark.parametrize("mutation,name", [
    (lambda s: s["worker_shutdown"]["gmt"].update(runtime_parameters_unchanged=False), "main.frozen"),
    (lambda s: s["actor"].update(ppo_only_gradient_norm=0.), "main.network_updates"),
    (lambda s: s["actor"]["ppo_module_gradients"].update(history_encoder=-1.), "main.network_updates"),
    (lambda s: s["critic"].update(actor_unchanged=False), "main.network_updates"),
    (lambda s: s["probability_check"].update(max_abs_ratio_minus_one=.1), "main.probability_before_update"),
    (lambda s: s.update(status="failed"), "main.completed"),
])
def test_declared_success_cannot_hide_update_failures(run_dir, mutation, name):
    mutate_json(run_dir/"acceptance"/"summary.json", mutation)
    report = audit.audit_run(run_dir, allow_incomplete=True)
    assert report["status"] == "failed" and check(report, name)["status"] == "failed"


def test_shared_budget_override_and_rollback(run_dir, tmp_path):
    external = tmp_path/"campaign.json"
    (run_dir/"budget.json").rename(external)
    assert audit.audit_run(run_dir, budget_file=external)["status"] == "passed"
    def rollback(budget):
        budget["used"]["control_steps"] -= 1
        budget["phases"]["resume"]["control_steps"] -= 1
    mutate_json(external, rollback)
    assert "rolled back" in check(audit.audit_run(run_dir, budget_file=external), "budget.cumulative")["error"]


def test_wal_snapshot_is_read_only_and_includes_uncheckpointed_replies(tmp_path):
    path = tmp_path/"journal.sqlite"
    with StepJournal(path) as journal:
        journal.connection.execute("PRAGMA wal_autocheckpoint=0")
        journal.append_result(envelope("session", 1, "reset_episode", dict(episode_id="e")))
        before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in tmp_path.iterdir()}
        report = audit.check_journal(path)
        assert report["counts"]["mutations"] == 1
        after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in tmp_path.iterdir()}
        assert after == before
