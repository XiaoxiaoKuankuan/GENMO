"""独立审计入口的 CPU 证据验收与故障注入。

测试创建真实 UpperTransition、ExecutionReward 和 StepJournal 产物，而非用字典
冒充训练 Buffer。小型模型状态只用于完整 checkpoint 的元数据容器，测试不构建
Actor/Critic、不初始化 GPU、不运行物理仿真。主采集 64 条与恢复 16 条同时包含
连续转移、行政截断、任务结束和一次性失败代价，以检查 bootstrap 与 GAE 递推分离。

负面样例覆盖落盘后缺失 trace、200Hz 子步时刻错误、SHA/序号篡改、错误 dt、
事件重复惩罚、旧价值未保存、优势递推错误、策略版本串用、GMT 非冻结、联合 KL
超限和预算回退。奖励夹具使用第二版实际执行公式及独立配对活动度，另外验证活动
门控被篡改、一致性超容差、诊断功率被赋奖励权重、版本错配均不能通过；第一版
归档仍按自己的旧公式核验，不把旧验收重新标成第二版。还检查 WAL 未 checkpoint 时的只读快照、源文件无变化，以及
大 checkpoint 的 mmap 参数和显式输出拒绝覆盖语义。所有文件仅写 pytest 临时目录。

可选学习率搜索夹具保留三次同起点候选中的最大合格一步，同时把三次尝试完整记入
预算。故障注入覆盖选择错误、KL与胜出权重不对应、梯度/起点未复用、checkpoint及
恢复学习率错配、候选身份遗漏和少记尝试预算；原无候选归档仍保持兼容。
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
from tests.closedloop.dppo.test_data_learning import actual_step, target_activity_fixture
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
    row = actual_step(start // 12 - 50, speed=1.)
    row.update(backend_session_id=session, mutation_seq=sequence, episode_id=episode)
    return row


def activity_from_tick_zero(tick):
    """把复用目标夹具的600Hz warmup原点平移至本审计夹具的episode零点。"""
    result = target_activity_fixture(tick + 600)
    result["window_begin_tick"] -= 600
    result["window_end_tick"] -= 600
    return result


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
                reward = ExecutionReward(config["stage9"]["reward"], music_features=np.zeros((150, 35)),
                                         music_start_tick=0, target_activity=activity_from_tick_zero)
            seq += 1
            step = physical_step(session, seq, episode, start)
            terminal, truncated = i%4 == 1, i%4 == 3
            step.update(terminated=terminal, truncated=truncated)
            feedback = dict(backend_session_id=session, mutation_seq=seq, episode_id=episode,
                            transition_valid=True, physics_count_exact=True, partial_control_step=None,
                            control_tick_begin=start, control_tick_end=start+12, executed_control_steps=1,
                            executed_physics_steps=4, trace=[step])
            reply = envelope(session, seq, "advance", feedback)
            assert journal.append_result(reply)
            assert not journal.append_result(reply)  # 正常重发不会重复落盘。
            detail = reward.evaluate_step(step)
            assert detail["transition_valid"], detail["errors"]
            assert detail["components"]["alive"]["score"] == float(not terminal)
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
    assert check(report, "main.rollout.integrity")["details"]["music_invalid_counts"] == {"first_cmd_step": 32}
    assert check(report, "main.rollout.integrity")["details"]["reward_valid_counts"]["music"] == 64
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
    (lambda detail: detail["activity"].update(gate=.25), "activity gate formula"),
    (lambda detail: detail["components"]["track"].update(gate=.25), "track activity gate"),
    (lambda detail: detail["diagnostics"]["consistency"]["raw"].update(joint_vel_rms_rad_s=2e-4),
     "reference consistency gate exceeded"),
    (lambda detail: detail["diagnostics"]["power"].update(reward_weight=.1), "power must not be a reward"),
    (lambda detail: detail["activity"].update(valid=False), "paired target activity unavailable"),
])
def test_v2_reward_gates_and_diagnostic_only_contract(run_dir, mutation, error):
    path = run_dir / "acceptance" / "rollout.pt"
    rows = torch.load(path, weights_only=False)
    mutation(rows[0].metadata["reward_details"][0])
    torch.save(rows, path)
    report = audit.audit_run(run_dir)
    assert report["status"] == "failed"
    assert error in check(report, "main.rollout.integrity")["error"]


def test_v2_archive_cannot_be_checked_with_v1_config(run_dir):
    path = run_dir / "acceptance" / "resolved_config.yaml"
    config = yaml.safe_load(path.read_text())
    config["stage9"]["reward"]["version"] = "stage9.execution_reward.v1"
    path.write_text(yaml.safe_dump(config))
    report = audit.audit_run(run_dir)
    assert report["status"] == "failed"
    assert "reward/config version mismatch" in check(report, "main.rollout.integrity")["error"]


def test_v1_archived_reward_still_uses_its_original_formula(tmp_path):
    weights = dict(music=2., track=2., stable=1., actuator=-.2, contact=-.5, consistency=-.5)
    components = {
        name: dict(enabled=True, valid=name != "music", score=.5 if name != "music" else 0., weight=weight,
                   weighted_rate=.5 * weight if name != "music" else 0.,
                   raw={} if name != "music" else {"reason": "insufficient_causal_history"})
        for name, weight in weights.items()
    }
    detail = dict(version="stage9.execution_reward.v1", components=components,
                  reward=.02 * sum(component["weighted_rate"] for component in components.values()))
    archive = tmp_path / "legacy_reward.json"
    write_json(archive, dict(detail=detail, config=dict(weights=weights)))
    restored = json.loads(archive.read_text())
    counts, unavailable = audit.check_reward_arithmetic(restored["detail"], restored["config"])
    assert dict(counts) == {name: 1 for name in weights if name != "music"}
    assert dict(unavailable) == {"insufficient_causal_history": 1}
    with pytest.raises(ValueError, match="reward/config version mismatch"):
        audit.check_reward_arithmetic(restored["detail"], DEFAULT_CONFIG)


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


@pytest.fixture
def calibrated_run_dir(run_dir):
    """三次尝试仅保留第二候选；第三次因KL超限拒绝，所有预算仍累计三次。"""
    paths = [run_dir / phase / 'resolved_config.yaml' for phase in ('acceptance', 'resume_check')]
    for path in paths:
        config = yaml.safe_load(path.read_text())
        config['stage9'].update(actor_lr_candidates=[1e-9, 3e-9, 1e-8],
            actor_lr_selection='largest_candidate_below_joint_kl_limit', critic_lr=1e-4)
        path.write_text(yaml.safe_dump(config))
    main = json.loads((run_dir / 'acceptance' / 'summary.json').read_text())
    candidates = []
    for lr, factor, accepted in ((1e-9, 1., True), (3e-9, 9., True), (1e-8, 100., False)):
        kl = {key: value * factor if isinstance(value, float) else value for key, value in main['kl'].items()}
        change = dict(changed_count=50, parameter_count=100, changed_fraction=.5,
                      l2=lr * 1e6, max_abs=lr * 1e5, per_module=dict(denoiser=dict(changed_count=50)))
        candidates.append(dict(lr=lr, kl=kl, parameter_change=change, accepted=accepted))
    main['actor']['lr_calibration'] = dict(candidates=candidates, selected_lr=3e-9, attempt_count=3,
        accepted_updates=1, selection_rule='largest_candidate_below_joint_kl_limit', kl_limit=.02,
        base_state_restored_per_candidate=True, gradients_reused=True)
    main['kl'] = copy.deepcopy(candidates[1]['kl'])
    resume = json.loads((run_dir / 'resume_check' / 'summary.json').read_text())
    resume['resume'].update(actor_optimizer_lrs=[3e-9], critic_optimizer_lrs=[1e-4])
    checkpoint_path = run_dir / 'checkpoints' / 'stage9_000001.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint['config'] = copy.deepcopy(config)
    checkpoint['identity']['training_contract'].update(
        actor_lr_candidates=config['stage9']['actor_lr_candidates'],
        actor_lr_selection=config['stage9']['actor_lr_selection'])
    checkpoint['state'].update(selected_actor_lr=3e-9, optimizer_attempts=3)
    checkpoint['actor_optimizer'] = dict(param_groups=[dict(lr=3e-9)])
    checkpoint['critic_optimizer'] = dict(param_groups=[dict(lr=1e-4)])
    ledger = json.loads((run_dir / 'budget.json').read_text())
    for budget in (main['budget'], resume['budget'], checkpoint['state']['budget'], ledger):
        budget['used']['iterations'] = 3
        budget['phases']['update']['iterations'] = 3
    torch.save(checkpoint, checkpoint_path)
    write_json(run_dir / 'acceptance' / 'summary.json', main)
    write_json(run_dir / 'resume_check' / 'summary.json', resume)
    write_json(run_dir / 'budget.json', ledger)
    return run_dir


def test_calibrated_update_complete_checkpoint_resume_and_attempts(calibrated_run_dir):
    report = audit.audit_run(calibrated_run_dir)
    assert report['status'] == 'passed', report
    metadata = check(report, 'checkpoint.metadata')['details']
    assert metadata['optimizer_lrs'] == dict(actor=[3e-9], critic=[1e-4])
    assert metadata['state']['actor_updates'] == metadata['state']['iteration'] == 1
    assert check(report, 'main.network_updates')['details']['lr_calibration']['optimizer_attempts'] == 3
    assert check(report, 'budget.cumulative')['details']['observed_lower_bound']['iterations'] == 3
    assert check(report, 'resume.full_state_and_new_physics')['details']['optimizer_lrs']['actor'] == [3e-9]


@pytest.mark.parametrize('mutation,error', [
    (lambda c: c.update(selected_lr=1e-9), 'largest qualifying candidate'),
    (lambda c: c.update(selected_lr=3.1e-9), 'largest qualifying candidate'),
    (lambda c: c.update(attempt_count=2), 'attempt count'),
    (lambda c: c.update(accepted_updates=3), 'exactly one accepted'),
    (lambda c: c.update(base_state_restored_per_candidate=False), 'same initial state'),
    (lambda c: c.update(gradients_reused=False), 'one fixed gradient'),
    (lambda c: c.update(kl_limit=.2), 'candidate KL threshold'),
    (lambda c: c['candidates'][2].update(accepted=True), 'candidate acceptance'),
    (lambda c: c['candidates'][0].update(lr=2e-9), 'learning-rate order'),
    (lambda c: c['candidates'][0]['parameter_change'].update(changed_fraction=.7), 'changed parameter fraction'),
    (lambda c: c['candidates'][0]['parameter_change'].update(l2=0.), 'delta norms'),
    (lambda c: c['candidates'][0]['parameter_change'].update(per_module={}), 'per-module parameter deltas'),
    (lambda c: c['candidates'][0]['kl'].update(mean_joint_kl=float('nan')), 'nonfinite'),
])
def test_lr_candidate_evidence_cannot_fake_an_accepted_update(calibrated_run_dir, mutation, error):
    path = calibrated_run_dir / 'acceptance' / 'summary.json'
    mutate_json(path, lambda s: mutation(s['actor']['lr_calibration']))
    # 非有限JSON在读取层即被拒绝，其余都应在候选独立检查失败。
    report = audit.audit_run(calibrated_run_dir)
    assert report['status'] == 'failed'
    name = 'main.summary.read' if error == 'nonfinite' else 'main.network_updates'
    assert error in check(report, name)['error']


def test_zero_parameter_change_is_a_rejected_candidate(calibrated_run_dir):
    def zero_first(summary):
        candidate = summary['actor']['lr_calibration']['candidates'][0]
        candidate['accepted'] = False
        candidate['parameter_change'].update(changed_count=0, changed_fraction=0., l2=0., max_abs=0.)
    mutate_json(calibrated_run_dir / 'acceptance' / 'summary.json', zero_first)
    report = audit.audit_run(calibrated_run_dir)
    assert report['status'] == 'passed', report


def test_selected_candidate_requires_final_kl_recomputation(calibrated_run_dir):
    mutate_json(calibrated_run_dir / 'acceptance' / 'summary.json', lambda s: s['kl'].update(max_joint_kl=.123))
    report = audit.audit_run(calibrated_run_dir)
    assert report['status'] == 'failed'
    assert 'final KL differs from selected candidate' in check(report, 'main.network_updates')['error']


def test_configured_calibration_cannot_omit_the_report(calibrated_run_dir):
    mutate_json(calibrated_run_dir / 'acceptance' / 'summary.json', lambda s: s['actor'].pop('lr_calibration'))
    assert 'calibration report missing' in check(audit.audit_run(calibrated_run_dir), 'main.network_updates')['error']


@pytest.mark.parametrize('mutation,error', [
    (lambda p: p['state'].update(selected_actor_lr=1e-9), 'checkpoint selected learning rate'),
    (lambda p: p['state'].update(optimizer_attempts=1), 'optimizer attempt count'),
    (lambda p: p['actor_optimizer']['param_groups'][0].update(lr=1e-9), 'checkpoint Actor'),
    (lambda p: p['critic_optimizer']['param_groups'][0].update(lr=2e-4), 'checkpoint Critic'),
    (lambda p: p['actor_optimizer'].update(param_groups=[]), 'parameter groups missing'),
])
def test_checkpoint_must_keep_the_winning_optimizer(calibrated_run_dir, mutation, error):
    path = calibrated_run_dir / 'checkpoints' / 'stage9_000001.pt'
    payload = torch.load(path, weights_only=False)
    mutation(payload)
    torch.save(payload, path)
    report = audit.audit_run(calibrated_run_dir)
    assert report['status'] == 'failed'
    assert error in check(report, 'checkpoint.metadata')['error']


def test_candidate_rules_are_bound_to_checkpoint_identity(calibrated_run_dir):
    path = calibrated_run_dir / 'checkpoints' / 'stage9_000001.pt'
    payload = torch.load(path, weights_only=False)
    payload['identity']['training_contract'].pop('actor_lr_candidates')
    torch.save(payload, path)
    assert 'candidate identity missing' in check(audit.audit_run(calibrated_run_dir), 'checkpoint.identity_main')['error']


@pytest.mark.parametrize('field,rates', [('actor_optimizer_lrs', [1e-9]), ('critic_optimizer_lrs', [2e-4]),
                                      ('actor_optimizer_lrs', [])])
def test_resume_must_restore_actual_winning_learning_rates(calibrated_run_dir, field, rates):
    mutate_json(calibrated_run_dir / 'resume_check' / 'summary.json', lambda s: s['resume'].update({field: rates}))
    report = audit.audit_run(calibrated_run_dir)
    assert report['status'] == 'failed'
    assert 'optimizer learning' in check(report, 'resume.full_state_and_new_physics')['error']


def test_budget_counts_all_candidates_even_when_only_one_is_kept(calibrated_run_dir):
    def undercount(budget):
        budget['used']['iterations'] = 1
        budget['phases']['update']['iterations'] = 1
    for phase in ('acceptance', 'resume_check'):
        mutate_json(calibrated_run_dir / phase / 'summary.json', lambda s: undercount(s['budget']))
    mutate_json(calibrated_run_dir / 'budget.json', undercount)
    path = calibrated_run_dir / 'checkpoints' / 'stage9_000001.pt'
    payload = torch.load(path, weights_only=False)
    undercount(payload['state']['budget'])
    torch.save(payload, path)
    report = audit.audit_run(calibrated_run_dir)
    assert report['status'] == 'failed'
    assert 'budget undercounts optimizer attempts' in check(report, 'budget.cumulative')['error']
