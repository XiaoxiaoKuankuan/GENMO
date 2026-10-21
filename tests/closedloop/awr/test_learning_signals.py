"""候选随机流、真实暴露信用、N-step标签及训练音乐评价的端到端数学回归。

使用明确标注的合成物理控制账本，独立手算折扣和每步参考归属；覆盖缺失成员、提前
失败、跨版本、初态不一致、零步事件与旧标签拒绝。小型真实Stage1用于无教师更新，
CPU测试不宣称PhysX同状态或RL提升；真实验收证据由独立服务器实验保存。
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from gem.closedloop.awr.candidates import (noise_seed, add_candidates, initial_state, state_fingerprint,
                                          group_advantages, window_return)
from gem.closedloop.awr.reference_credit import build_reference_ledger, reference_exposure, credit_advantages
from gem.closedloop.awr.returns import nstep_advantages, nstep_targets, interval_rewards
from gem.closedloop.awr.advantages import label_episode_advantages, label_advantages
from gem.closedloop.awr.dataset import ReplayDataset
from gem.closedloop.awr.config import load_config


def with_ledger(episode):
    for index, row in enumerate(episode.rows):
        row.metadata["published"] = dict(effective_tick=row.control_tick_begin)
        trace = []
        for offset in range(row.executed_control_steps):
            tick = row.control_tick_begin+offset*12
            previous = f"p{index-1}" if index else "bootstrap:test"
            trace.append(dict(control_tick_begin=tick, tick=tick+12, episode_id=row.identity["episode_id"],
                consumed_reference_ticks=list(range(tick-120, tick+121, 12)),
                consumed_plan_ids=[previous]*11+[f"p{index}"]*10,
                reference_tick=tick+12, reference_plan_id=previous,
                errors=dict(root_position_error_m=.1+index, joint_position_rmse_rad=.2+index, yaw_error_rad=.3+index)))
        row.metadata["execution_trace"] = trace
    episode.reference_ledger = build_reference_ledger(episode)
    return episode


def candidate_episodes(make, count=4):
    result = []
    for i in range(count):
        e = make(episode_id=f"candidate{i}", lengths=(25, 25, 25, 25, 25),
                 rewards=tuple([float(i+1)]*25 for _ in range(5)))
        e.contract.update(environment=dict(domain_randomization=False, external_perturbations=False,
            observation_noise=False, reset_randomization=False), sampler={"cfg": 2.5})
        e.task.update(candidate_group_id="group", candidate_index=i, candidate_count=4,
            candidate_schema="genmo.awr.first_generation_candidate.v1", reset_seed=99, seed=99,
            generation_seed=100+i, common_future_noise_key="future-group", common_future_noise=True,
            comparison_start_tick=600, sample_id="song", music_start_frame=0, initialization="standing")
        snapshot = dict(robot_qpos=np.zeros(28), robot_joint_vel=np.zeros(21),
            candidate_root_lin_vel_w=np.zeros(3), candidate_root_ang_vel_w=np.zeros(3),
            candidate_contact_forces_w=np.zeros((4, 3)), candidate_state_schema="gmt.live_candidate_initial_state.v1",
            candidate_state_is_live=True, tick=600, runtime_fingerprint_sha256="physics")
        evidence = initial_state(dict(snapshot=snapshot, candidate_task=e.task, seed=e.task["generation_seed"],
            reservation=dict(prefix_frames=12)), e.rows[0].context, dict(world_anchor=[0., 0., 1., 0.]))
        for j, row in enumerate(e.rows):
            row.metadata.update(generation_seed=noise_seed(e.task, j), sampler={"cfg": 2.5})
        e.rows[0].metadata["candidate_initial_state"] = evidence
        e.task["initial_state_fingerprint"] = evidence["fingerprint"]
        result.append(with_ledger(e))
    return result


def test_candidate_noise_is_rank_slot_task_independent_and_only_first_changes():
    base = dict(candidate_group_id="g", candidate_index=0, generation_seed=10, common_future_noise_key="future")
    other = dict(base, candidate_index=1, generation_seed=20, rank=7, env_slot=31, task_id="different")
    assert noise_seed(base, 0) != noise_seed(other, 0)
    assert [noise_seed(base, i) for i in range(1, 9)] == [noise_seed(other, i) for i in range(1, 9)]
    assert len({noise_seed(base, i) for i in range(1, 9)}) == 8
    assert noise_seed({}, 0) is None


def test_first_generation_is_episode_relative_not_persistent_environment_counter(monkeypatch):
    from gem.closedloop.awr.collection import AWRUpperEnvironment, VectorUpperEnvironment
    def reset(self, *args, **kwargs):
        yield from ()
    monkeypatch.setattr(VectorUpperEnvironment, "reset_task_flow", reset)
    env = AWRUpperEnvironment.__new__(AWRUpperEnvironment)
    env.decision = 173
    env.active_awr_task = dict(candidate_group_id="g", generation_seed=10, common_future_noise_key="future")
    list(env.reset_task_flow())
    assert env.awr_decision_origin == env.decision == 173
    assert env.generation_noise_seed(173) == 10
    assert env.generation_noise_seed(174) == noise_seed(env.active_awr_task, 1)
    env.decision = 299
    list(env.reset_task_flow())
    assert env.generation_noise_seed(299) == 10


def test_candidate_expansion_keeps_budget_all_music_and_independent_task_ids():
    tasks = [dict(task_id=f"t{i}-{j}", dataset=str(i//4), sample_id=f"s{i}", group_id=f"g{i}", seed=42,
                  music_start_frame=10, initialization="paired", rank=0) for i in range(16) for j in range(4)]
    c = dict(enabled=True, candidates_per_group=4, group_fraction=.25, common_future_noise=True)
    one = add_candidates(tasks, c, world_size=1, macro=0)
    eight = add_candidates(tasks, c, world_size=8, macro=0)
    assert len(one) == len(eight) == len({t["task_id"] for t in one}) == 64
    assert {t["group_id"] for t in one} == {t["group_id"] for t in tasks}
    grouped = [t for t in one if t.get("candidate_group_id")]
    assert len(grouped) == 16 and len({t["candidate_group_id"] for t in grouped}) == 4
    assert [{k: v for k, v in t.items() if k != "rank"} for t in one] == [
        {k: v for k, v in t.items() if k != "rank"} for t in eight]


def test_actual_center_and_future_credit_are_distinct_normalized_and_auditable(episode_factory):
    _, _, make = episode_factory
    e = with_ledger(make(lengths=(2, 2, 1)))
    current = reference_exposure(e)
    future = reference_exposure(e, mode="current_plus_lookahead", current_weight=3., lookahead_weight=1.)
    assert current["plans"][0]["first_gmt_input_tick"] == 600
    assert current["plans"][0]["first_current_tick"] == 624
    assert current["matrix"][0].tolist() == [0., 1., 0.]
    torch.testing.assert_close(future["matrix"][0], torch.tensor([.25, .75, 0.], dtype=torch.float64))
    assert torch.all(current["matrix"].sum(0) <= 1) and torch.all(future["matrix"].sum(0) <= 1+1e-12)
    assert current["matrix"][2].sum() == 0  # 最后计划只被前瞻观察，不能伪造当前消费
    assert future["matrix"][2].sum() > 0
    assert current["plans"][0]["replaced_after_tick"] == 648
    bad = copy.deepcopy(e); bad.reference_ledger["window_plan_indices"][0, 10] = 100
    with pytest.raises(ValueError, match="identity"):
        reference_exposure(bad)


@pytest.mark.parametrize("nstep", [4, 8, 16, "full_mc"])
def test_credit_all_one_telescopes_variable_duration_nstep(episode_factory, nstep):
    _, _, make = episode_factory
    e = make(lengths=(2, 7, 1, 25, 3, 0), event=-5.)
    v = torch.tensor([10., 8., -1., 12., 7., 3.], dtype=torch.float64)
    original = e.returns.clone()
    actual = credit_advantages(e, v, torch.ones(6, 6), nstep=nstep)
    expected = nstep_advantages(e, v, nstep)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(e.returns, original, atol=0, rtol=0)
    if nstep == 4:
        r, d = interval_rewards(e.rows)
        manual = r[0]+d[0]*r[1]+d[:2].prod()*r[2]+d[:3].prod()*r[3]+d[:4].prod()*v[4]-v[0]
        assert actual[0] == pytest.approx(manual)
    with pytest.raises(ValueError, match="frozen"):
        credit_advantages(e, v.requires_grad_(), torch.ones(6, 6), nstep=nstep)


def test_fixed_window_failure_counts_event_once_and_never_averages_alive_time(episode_factory):
    _, _, make = episode_factory
    failure = make(lengths=(2,), rewards=((1., -4.),))  # 真实最后控制奖励含一次-5事件
    failure.rows[-1].metadata["physical_failure"] = True
    failure.rows[-1].reason = "yaw_error"
    gamma = .99**(1/25)
    r = window_return(failure, 2.)
    assert r["return_value"] == pytest.approx(1-4*gamma)
    assert r["missing_controls"] == 98 and r["terminal_absorbing_tail_value"] == 0
    survival = make(lengths=(100,), rewards=([.1]*100,))
    assert window_return(survival, 2.)["return_value"] > r["return_value"]
    zero = make(lengths=(1, 0), rewards=((1.,), ()), event=-5.)
    assert window_return(zero, 4.)["return_value"] == pytest.approx(1-5*gamma)


@pytest.mark.parametrize("fault", [None, "state", "contact", "actor", "missing", "seed", "task_sha"])
def test_group_only_marks_first_with_all_members_and_actual_initial_equality(episode_factory, fault):
    _, _, make = episode_factory
    episodes = candidate_episodes(make)
    if fault in ("state", "contact"):
        evidence = episodes[-1].rows[0].metadata["candidate_initial_state"]
        field = "robot_qpos" if fault == "state" else "candidate_contact_forces_w"
        evidence["state"][field].flat[0] += .1
        evidence["fingerprint"] = state_fingerprint(evidence["state"])
        episodes[-1].task["initial_state_fingerprint"] = evidence["fingerprint"]
    if fault == "actor":
        for row in episodes[-1].rows:
            row.identity["actor_fingerprint"] = "different"
    if fault == "missing": episodes.pop()
    if fault == "seed": episodes[-1].task["reset_seed"] += 1
    if fault == "task_sha": episodes[-1].task["initial_state_fingerprint"] = "tampered"
    signal, report = group_advantages(episodes, dict(score_window_seconds=2, evaluation_windows_seconds=[2., 4.]))
    assert report[0]["valid"] is (fault is None)
    if fault:
        assert signal == {} and report[0]["reasons"]
    else:
        assert len(signal) == 4 and abs(sum(signal.values())) < 1e-12
        assert signal[episodes[3].rows[0].sample_id] > signal[episodes[0].rows[0].sample_id]
        assert all(row.sample_id not in signal for e in episodes for row in e.rows[1:])


@pytest.mark.parametrize("mode", ["mc", "nstep", "credit_nstep", "hybrid", "group_hybrid"])
def test_frozen_label_modes_components_scale_fallback_identity_and_statistics(episode_factory, tmp_path, mode):
    _, _, make = episode_factory
    episodes = candidate_episodes(make)
    values = {e.episode_id: torch.arange(len(e.rows), dtype=torch.float64) for e in episodes}
    config = dict(mode=mode, alpha=.7, nstep=4, credit_mode="current_plus_lookahead")
    candidates = dict(enabled=True, advantage_mode="hybrid" if mode == "group_hybrid" else "standard")
    labels, report = label_episode_advantages(episodes, values, identity=dict(replay_snapshot="fixed"),
                                             config=config, candidate_config=candidates)
    assert labels.final_advantage.shape == labels.weights.shape == (20,)
    assert torch.isfinite(labels.final_weight).all() and (labels.final_weight > 0).all()
    assert labels.weights.mean() == pytest.approx(1.)
    assert labels.group_valid.sum() == 4 and labels.statistics["effective_sample_size"] <= 20+1e-10
    assert report["candidate_groups"][0]["valid"]
    data = ReplayDataset(episodes, labels, expected_signal=config, expected_candidates=candidates, expected_snapshot="fixed")
    assert len(data) == 20
    with pytest.raises(ValueError, match="mode/config"):
        ReplayDataset(episodes, labels, expected_signal=dict(config, nstep=8), expected_candidates=candidates, expected_snapshot="fixed")
    if mode == "mc":
        old = label_advantages(labels.sample_ids, labels.returns, labels.values, sources=labels.source, identity={})
        torch.testing.assert_close(labels.weights, old.weights, atol=0, rtol=0)
    # 旧Replay没有消费证据，只回退nstep，不能把缺失赋成负优势。
    legacy = make(); v = {legacy.episode_id: torch.tensor([1., 3.])}
    fallback, _ = label_episode_advantages([legacy], v, identity={}, config=dict(mode="credit_nstep", credit_fallback="nstep"))
    plain, _ = label_episode_advantages([legacy], v, identity={}, config=dict(mode="nstep"))
    assert not fallback.credit_valid.any()
    torch.testing.assert_close(fallback.weights, plain.weights, atol=0, rtol=0)


def test_new_experiment_configs_disable_teacher_gates_use_train_and_preserve_old_files():
    root = Path(__file__).resolve().parents[3]/"configs/closedloop"
    base = load_config(root/"stage2_awr_group_credit.yaml")
    assert base["awr"]["actor"]["reference_weight"] == 0 and base["awr"]["actor"]["lr"] == 1e-5
    assert base["awr"]["retention_gate"] is None and not base["awr"]["stop_on_physical_rejection"]
    assert base["evaluation"]["split"] == "train" and base["evaluation"]["selection_policy"] == "reward_only"
    for name in ("mc", "nstep4", "nstep8", "nstep16", "credit_current", "credit_lookahead", "hybrid0", "hybrid05", "hybrid07", "grouped"):
        candidate = load_config(root/f"stage2_awr_signal_{name}.yaml")
        assert candidate["awr"]["actor"] == base["awr"]["actor"]


def test_reward_only_selection_keeps_failures_as_diagnostics():
    from gem.closedloop.awr.evaluation import paired_comparison
    b = dict(plan_identity="p", physics_identity="same", sampler_identity="same",
        task_summaries=[dict(task_id="t", source="s", group_id="g", reward_sum=10., seconds=20., physical_failure=False)],
        aggregate=dict(overall=dict(music_completion_count=1, physical_failure_count=0, executed_seconds=20.,
                                   activity=dict(actual_activity_rad_s=dict(mean=1.)))))
    c = copy.deepcopy(b)
    c["task_summaries"][0].update(reward_sum=11., seconds=10., physical_failure=True)
    c["aggregate"]["overall"].update(music_completion_count=0, physical_failure_count=1, executed_seconds=10.,
                                    activity=dict(actual_activity_rad_s=dict(mean=.1)))
    result = paired_comparison(b, c, bootstrap=30)
    assert result["acceptable"] and result["new_paired_failures"] == ["t"]
    assert not result["physical_diagnostics"]["activity_retained"]
    assert not paired_comparison(b, c, bootstrap=30, selection_policy="legacy_physical_gates")["acceptable"]


def test_training_music_evaluation_never_uses_or_relabels_holdout(tmp_path):
    from gem.closedloop.awr.train_evaluation_plan import build_train_plan
    from gem.closedloop.common.full_dataset import SOURCES
    pools = {}
    for split in ("train", "val", "test"):
        pools[split] = {s: [dict(dataset=s, split=split, group_id=f"{split}:{s}:{i}",
            row=dict(sample_id=f"{split}:{s}:{i}", split=split, num_frames=900, fps=30,
                     source_audio_sha256=f"audio:{split}:{s}:{i}")) for i in range(3)] for s in SOURCES}
    catalog = SimpleNamespace(data_root=tmp_path, identity=dict(catalog="fixture"), samples=pools)
    plan = build_train_plan(catalog, samples_per_source=2, seeds=[42, 1729], selection_seed=42)
    assert plan == build_train_plan(catalog, samples_per_source=2, seeds=[42, 1729], selection_seed=42)
    assert plan["task_count"] == 16 and plan["split"] == "train" and not plan["generalization_claim"]
    assert not plan["training_buffer_allowed"] and not plan.get("critic_mc_supervision")
    assert all(t["split"] == t["sample"]["row"]["split"] == "train" for t in plan["tasks"])
    source = SOURCES[0]
    catalog.samples["test"][source][0]["group_id"] = catalog.samples["train"][source][0]["group_id"]
    with pytest.raises(ValueError, match="overlaps"):
        build_train_plan(catalog, samples_per_source=2, seeds=[42], selection_seed=42)


def test_teacher_disabled_is_actual_loss_and_gradients_not_hidden_anchor(episode_factory):
    from gem.closedloop.awr.trainer import ActorTrainer
    from gem.closedloop.awr.distributed import Distributed
    from tests.closedloop.awr.test_training import ArithmeticBC
    actor, _, make = episode_factory
    initial = copy.deepcopy(actor.state_dict())
    dataset = ReplayDataset([make()], unit_weights=True)
    trainer = ActorTrainer(actor, dataset, ArithmeticBC(), dict(epochs=1, global_batch=2, microbatch=1,
        bc_global_batch=2, bc_microbatch=1, reference_weight=0., lr=1e-5, warmup_steps=0), Distributed(), bc_weight=.01)
    assert trainer.reference_actor is None and trainer.state_dict()["reference_actor"] is None
    result = trainer.run()
    row = trainer.history[-1]
    assert result["steps"] == 1 and row["lr"] == 1e-5
    assert row["reference_loss"] == row["reference_weight"] == 0.
    assert row["total_loss"] == pytest.approx(row["awr_loss"]+.01*row["bc_loss"])
    assert any(not torch.equal(value, actor.state_dict()[name]) for name, value in initial.items())


def test_missing_whole_candidate_group_and_zero_step_member_have_no_group_labels(episode_factory):
    _, _, make = episode_factory
    planned = [e.task for e in candidate_episodes(make)]
    signals, reports = group_advantages([], dict(enabled=True), planned_tasks=planned)
    assert signals == {} and reports[0]["reasons"] == ["all_members_missing"]


def test_replay_and_label_binary_roundtrip_preserves_reference_ledger_and_candidate_identity(tmp_path, episode_factory):
    from gem.closedloop.awr.replay import EpisodeReplay
    from gem.closedloop.awr.advantages import AdvantageLabelSet
    _, _, make = episode_factory
    episodes = candidate_episodes(make)
    replay = EpisodeReplay(tmp_path/"replay", capacity=4, contract=episodes[0].contract)
    try:
        for episode in episodes:
            replay.append(episode)
        snapshot = replay.snapshot()
        projected = [replay.read(key, training=True) for key in snapshot.episode_ids]
        assert all(e.training_view for e in projected)
        labels, _ = label_episode_advantages(projected, {e.episode_id: torch.zeros(len(e.rows)) for e in projected},
            identity=dict(replay_snapshot=snapshot.snapshot_id), config=dict(mode="group_hybrid"),
            candidate_config=dict(enabled=True, advantage_mode="hybrid"))
        assert labels.group_valid.sum() == 4 and labels.credit_valid.any()
        labels.save(tmp_path/"labels")
        restored = AdvantageLabelSet.load(tmp_path/"labels")
        assert restored.identity == labels.identity
        torch.testing.assert_close(restored.final_weight, labels.weights, atol=0, rtol=0)
        assert replay.read(snapshot.episode_ids[0], training=True).reference_ledger["sha256"] == episodes[0].reference_ledger["sha256"]
    finally:
        replay.close()


def test_offline_all_modes_use_one_frozen_snapshot_and_never_change_run(tmp_path, episode_factory):
    import json
    from gem.closedloop.awr.checkpoint import Checkpoints
    from gem.closedloop.awr.replay import EpisodeReplay
    from gem.closedloop.awr.io import atomic_json
    from tools.audit_stage2_awr_signals import audit
    _, _, make = episode_factory
    episodes = candidate_episodes(make)
    root = tmp_path/"run"
    replay = EpisodeReplay(root/"replay", capacity=4, contract=episodes[0].contract)
    for e in episodes:
        replay.append(e)
    snapshot = replay.snapshot(); replay.close()
    identity = dict(request="test", sources={"genmo": "frozen"})
    ck = Checkpoints(root/"checkpoints", identity)
    reference = ck.save("critics", dict(replay_snapshot=snapshot.snapshot_id,
        crossfit_values={e.episode_id: torch.arange(len(e.rows), dtype=torch.float64) for e in episodes}), progress={})
    atomic_json(root/"critic.json", dict(crossfit_identity="crossfit", fold_identity="fold",
                                        selected_critic_fingerprints=["a", "b"]))
    atomic_json(root/"tasks.json", dict(tasks=[e.task for e in episodes]))
    state = dict(request_sha256="test", source_commits={"genmo": "frozen"},
        config=dict(awr=dict(advantage=dict(mode="group_hybrid", nstep=4, alpha=.5,
                                          credit_mode="current_plus_lookahead"),
                            candidate_sampling=dict(enabled=True, advantage_mode="hybrid"))),
        artifacts=dict(macro000_replay=snapshot.snapshot_id, macro000_critics=reference,
                       macro000_critic_report=str(root/"critic.json"), macro000_tasks=str(root/"tasks.json")))
    atomic_json(root/"state.json", state)
    original = (root/"state.json").read_bytes()
    result = audit(root, tmp_path/"audit")
    assert result["actor_optimizer_updates"] == result["new_physical_controls"] == 0
    assert len(result["variants"]) == 11 and result["full_credit_equals_nstep_max_error"] < 1e-12
    assert result["variants"]["mc"]["weight"] == result["variants"]["hybrid0"]["weight"]
    assert len({v["label_id"] for v in result["variants"].values()}) == 11
    assert (root/"state.json").read_bytes() == original
