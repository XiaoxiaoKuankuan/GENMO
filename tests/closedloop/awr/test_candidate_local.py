"""首次候选独立世界、自由参考信用及失败分层价值拟合的回归测试。

用明确合成任务验证rank调度和恢复不改噪声，用真实记录类型的合成控制账本核对
已知前缀不会获得新动作的信用。检查候选固定目标质量不随microbatch改变，以及
失败子集退化时不能被总体FVI改善掩盖。这里只验证数学/调度，不冒充真实PhysX结果。
"""
import copy
from pathlib import Path
import pytest
import torch

from gem.closedloop.awr.candidates import add_candidates, collection_rounds, noise_seed
from gem.closedloop.awr.config import load_config
from gem.closedloop.awr.critic import regression_weights, validation_score, fvi_selection
from gem.closedloop.awr.objective import candidate_objective_weights
from gem.closedloop.awr.reference_credit import reference_exposure
from tests.closedloop.awr.test_learning_signals import with_ledger


def test_fresh_rounds_keep_every_candidate_in_virgin_slot_and_resume_identity():
    tasks = [dict(task_id=f"t{i}-{j}", dataset=str(i//4), sample_id=f"s{i}", group_id=f"g{i}", seed=42,
                  music_start_frame=10, initialization="paired", rank=0) for i in range(16) for j in range(4)]
    config = dict(enabled=True, candidates_per_group=4, group_fraction=1., common_future_noise=True,
                  world_lifecycle="fresh_candidate_round")
    plan = add_candidates(tasks, config, world_size=8, macro=0)
    assert len(plan) == 64 and len({t["task_id"] for t in plan}) == 64
    assert len({t["candidate_group_id"] for t in plan}) == 16
    for rank in range(8):
        rounds = collection_rounds(plan, config, rank=rank)
        assert len(rounds) == 8 and all(len(t) == 1 for _, t in rounds)
        first = rounds[0][1][0]
        resumed = collection_rounds(plan, config, rank=rank, finished=[first["task_id"]])
        assert resumed[0][1] == [] and resumed[1:] == rounds[1:]
    for group in {t["candidate_group_id"] for t in plan}:
        members = [t for t in plan if t["candidate_group_id"] == group]
        assert len({t["collection_round"] for t in members}) == 1
        assert len({t["rank"] for t in members}) == 4
        assert {t["collection_slot"] for t in members} == {0}
        assert len({noise_seed(t, 0) for t in members}) == 4
        assert len({noise_seed(t, 1) for t in members}) == 1
    with pytest.raises(ValueError, match="divisible"):
        add_candidates(tasks, config, world_size=2, macro=0)


def test_free_credit_removes_known_coordinates_without_redistribution(episode_factory):
    _, _, make = episode_factory
    episode = with_ledger(make(lengths=(25, 25, 25)))
    for row in episode.rows:
        row.context["future_times"] = (row.control_tick_begin+torch.arange(60)*20).double()[None]/600
        row.context["known_qpos30_mask"] = torch.zeros(1, 60, 30, dtype=torch.bool)
        row.context["known_qpos30_mask"][:, :22] = row.target_valid[:22]
        row.context["known_qpos30"] = row.target_qpos30.clone()[None]
        row.free_mask = row.target_valid & ~row.context["known_qpos30_mask"][0]
    all_refs = reference_exposure(episode, mode="current_plus_lookahead")
    free = reference_exposure(episode, mode="current_plus_lookahead", coordinate_scope="free_coordinates")
    assert (free["matrix"] <= all_refs["matrix"]+1e-12).all()
    assert free["matrix"][0, 0] == 0
    assert free["matrix"][0, 1] > 0
    assert free["plans"][0]["excluded_known_coordinate_seconds"] > 0
    assert (free["matrix"].sum(0) <= 1).all()
    episode.rows[0].context["future_times"] += .001
    with pytest.raises(ValueError, match="600Hz"):
        reference_exposure(episode, coordinate_scope="free_coordinates")


def test_fixed_candidate_mass_and_global_microbatch_normalization():
    sources = ["a", "b"]+["a"]*5+["b"]*13
    mask = torch.tensor([True, True]+[False]*18)
    weights = candidate_objective_weights(sources, mask, fraction=.25)
    assert weights.mean() == pytest.approx(1.)
    assert weights[mask].sum()/weights.sum() == pytest.approx(.25)
    assert weights[torch.tensor([s == "a" for s in sources])].sum()/weights.sum() == pytest.approx(.5)
    x = torch.arange(20.).requires_grad_()
    full = (weights*x.square()).sum()/20
    expected, = torch.autograd.grad(full, x)
    for start in range(0, 20, 3):
        (weights[start:start+3]*x[start:start+3].square()).sum().div(20).backward()
    torch.testing.assert_close(x.grad, expected)
    torch.testing.assert_close(candidate_objective_weights(sources, torch.zeros(20, dtype=torch.bool), fraction=.25),
        candidate_objective_weights(sources, mask, fraction=None))


def test_critic_failure_balance_and_fvi_selection_do_not_hide_failure_bias(episode_factory):
    _, _, make = episode_factory
    ok = make(episode_id="ok", source="a", lengths=(25, 25, 25))
    fail = make(episode_id="fail", source="a", lengths=(1,))
    fail.rows[-1].metadata["physical_failure"] = True
    other = make(episode_id="other", source="b", lengths=(25, 25))
    weights = regression_weights([ok, fail, other], "source_episode_type")
    assert weights.mean() == pytest.approx(1.)
    assert weights[:3].sum() == weights[3]  # 失败与自然结束在来源a内各占一半
    mc = dict(overall=dict(mse=20., bias=1.), episode_types={"natural_end":dict(mse=10.),
        "physical_failure":dict(mse=30.)}, source_episode_types={"a":{"natural_end":dict(mse=10.),
        "physical_failure":dict(mse=30.)}})
    fitted = copy.deepcopy(mc)
    fitted["overall"].update(mse=9., bias=.5)
    fitted["episode_types"]["physical_failure"]["mse"] = 60.
    fitted["source_episode_types"]["a"]["natural_end"]["mse"] = 1.
    assert fvi_selection(mc, fitted, "overall")["accepted"]
    decision = fvi_selection(mc, fitted, "source_episode_type")
    assert not decision["accepted"] and "physical_failure_mse_increased" in decision["reasons"]


def test_candidate_local_config_preserves_reward_bc_and_explicit_objective():
    root = Path(__file__).resolve().parents[3]/"configs/closedloop"
    old, new = [load_config(root/name) for name in ("stage2_awr_rewardstd_nobc.yaml", "stage2_awr_candidate_local.yaml")]
    assert new["reward"] == old["reward"] and new["environment"] == old["environment"]
    assert new["awr"]["actor"]["bc_enabled"] is False
    assert new["awr"]["actor"]["reference_weight"] == 0
    assert new["awr"]["actor"]["lr"] == 1e-5
    assert new["awr"]["smoke"]["envs_per_rank"] == 1
    assert new["awr"]["candidate_sampling"]["group_fraction"] == 1
    nstep = load_config(root/"stage2_awr_candidate_nstep.yaml")
    expected = copy.deepcopy(new)
    expected["awr"]["advantage"]["mode"] = "nstep"
    assert nstep == expected


def test_readonly_candidate_audit_binds_snapshot_and_recomputes_weights(episode_factory, tmp_path):
    import json
    from tools.audit_stage2_awr_candidate_local import audit
    from gem.closedloop.awr.io import write_tree, atomic_json, digest
    from gem.closedloop.awr.advantages import label_episode_advantages, signal_config
    from tests.closedloop.awr.test_learning_signals import candidate_episodes
    _, _, make = episode_factory
    episodes = candidate_episodes(make)
    entries = []
    for e in episodes:
        path = tmp_path/"replay/episodes"/e.episode_id
        write_tree(path, e.to_dict())
        entries.append([e.episode_id, json.loads((path/"manifest.json").read_text())["sha256"]])
    contract = digest(episodes[0].contract)
    snapshot = digest(dict(contract=contract, episodes=entries))
    atomic_json(tmp_path/"replay/snapshots"/(snapshot+".json"), dict(snapshot_id=snapshot,
        episode_ids=[e.episode_id for e in episodes], contract_sha256=contract))
    candidate = dict(enabled=True, advantage_mode="grouped", score_window_seconds=2., evaluation_windows_seconds=[2., 4.])
    signal, _ = signal_config(dict(mode="credit_nstep"), candidate)
    labels, _ = label_episode_advantages(episodes,
        {e.episode_id: torch.zeros(len(e.rows)) for e in episodes}, identity=dict(replay_snapshot=snapshot),
        config=signal, candidate_config=candidate)
    labels.save(tmp_path/"labels")
    atomic_json(tmp_path/"critic.json", dict(selected="mc_only", folds=[], selection_diagnostics=[]))
    state = dict(source_commits=dict(genmo="synthetic"), config=dict(awr=dict(replay_capacity=4,
        actor=dict(source_balance="uniform_sources", candidate_objective_fraction=.25),
        advantage=signal, candidate_sampling=candidate)), artifacts=dict(macro000_replay=snapshot,
        macro000_labels=str(tmp_path/"labels"), macro000_critic_report=str(tmp_path/"critic.json")))
    atomic_json(tmp_path/"state.json", state)
    before = (tmp_path/"state.json").read_bytes()
    report = audit(tmp_path)
    assert report["valid_groups"] == 1 and report["verified_candidate_samples"] == 4
    assert report["coefficient_mass_before_advantage"] == pytest.approx(.25)
    assert report["ranking"]["windows"]["2"]["weight"]["pair_accuracy"] == 1.
    assert (tmp_path/"state.json").read_bytes() == before


def test_natural_candidate_control_preserves_scope_and_restores_source_weights():
    from tools.ablate_stage2_awr_uniform import natural_candidate_config
    original = dict(bc_enabled=False, reference_weight=0., source_balance="uniform_sources",
                    candidate_objective_fraction=.25, epochs=5, lr=1e-5,
                    timestep_weighting={"mode": "uniform"})
    control = natural_candidate_config(original)
    assert original["candidate_objective_fraction"] == .25
    assert control == dict(original, candidate_objective_fraction=None)
    sources = ["a"]*10+["b"]*10
    mask = torch.tensor([True]+[False]*9+[True]+[False]*9)
    weights = candidate_objective_weights(sources, mask, fraction=control["candidate_objective_fraction"])
    torch.testing.assert_close(weights, torch.ones(20, dtype=torch.float64))
    for changed in (dict(original, bc_enabled=True), dict(original, reference_weight=.1)):
        with pytest.raises(ValueError):
            natural_candidate_config(changed)


def test_group_only_control_changes_no_group_scores_and_freezes_one_global_scale(episode_factory):
    from tools.ablate_stage2_awr_uniform import group_only_weights
    from gem.closedloop.awr.advantages import label_episode_advantages
    from tests.closedloop.awr.test_learning_signals import candidate_episodes
    _, _, make = episode_factory
    episodes = candidate_episodes(make)
    labels, _ = label_episode_advantages(episodes,
        {e.episode_id: torch.zeros(len(e.rows)) for e in episodes}, identity={},
        config=dict(mode="credit_nstep"), candidate_config=dict(enabled=True, advantage_mode="grouped"))
    before = labels.weights.clone()
    weights, result = group_only_weights(labels, list(reversed(labels.sample_ids)))
    torch.testing.assert_close(weights, result.weights.flip(0))
    torch.testing.assert_close(labels.weights, before)
    assert (result.normalized[~labels.group_valid] == 0).all()
    assert (result.raw_weights[~labels.group_valid] == 1).all()
    torch.testing.assert_close(result.normalized[labels.group_valid], labels.normalized[labels.group_valid])
    assert result.weights.mean() == pytest.approx(1.)
    _, nstep = group_only_weights(labels, labels.sample_ids, ordinary="nstep")
    torch.testing.assert_close(nstep.normalized[labels.group_valid], labels.normalized[labels.group_valid])
    torch.testing.assert_close(nstep.normalized[~labels.group_valid],
        labels.nstep_advantage[~labels.group_valid]/labels.identity["component_scales"]["nstep"]["std"])
    torch.testing.assert_close(nstep.returns, labels.returns)
    torch.testing.assert_close(nstep.group_valid, labels.group_valid)
    assert nstep.identity["label_id"] != result.identity["label_id"]
    torch.testing.assert_close(labels.weights, before)
