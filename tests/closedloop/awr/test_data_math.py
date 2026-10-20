"""完整Episode、FIFO/快照、MC/FVI和优势公式的独立算术回归。

构造可手算奖励，检查截断/重复/行为混合拒绝、校验和及真实条件往返。不用学习结果
反推实现正确，也不以生成Contact作为标签；测试成功仅代表这里覆盖的数据与数学契约。
"""
import copy
import math
import pytest
import torch
from gem.closedloop.awr.returns import mc_returns, nstep_targets
from gem.closedloop.awr.replay import EpisodeReplay
from gem.closedloop.awr.advantages import label_advantages, AdvantageLabelSet
from gem.closedloop.awr.folds import build_folds, partition_episodes


def test_training_projection_preserves_learning_values_and_checks_omitted_evidence(tmp_path, episode_factory):
    import json
    import numpy as np
    _, _, make = episode_factory
    episode = make()
    episode.rows[0].metadata["large_physical_diagnostic"] = np.arange(100, dtype=np.uint8)
    replay = EpisodeReplay(tmp_path / "replay", capacity=2, contract=episode.contract)
    try:
        replay.append(episode)
        full = replay.read(episode.episode_id)
        view = replay.read(episode.episode_id, training=True)
        assert view.training_view and "large_physical_diagnostic" not in view.rows[0].metadata
        assert "large_physical_diagnostic" in full.rows[0].metadata
        torch.testing.assert_close(view.returns, full.returns, atol=0, rtol=0)
        for a, b in zip(full.rows, view.rows):
            torch.testing.assert_close(a.target_qpos30, b.target_qpos30, atol=0, rtol=0)
            for key in a.context:
                torch.testing.assert_close(a.context[key], b.context[key], atol=0, rtol=0)
            for key in a.metadata["critic_context"]:
                torch.testing.assert_close(a.metadata["critic_context"][key], b.metadata["critic_context"][key], atol=0, rtol=0)
        with pytest.raises(ValueError, match="projected"):
            replay.publish(view)
        with pytest.raises(ValueError, match="projected"):
            view.to_dict()
        folder = replay.root / "episodes" / episode.episode_id
        manifest = json.loads((folder / "manifest.json").read_text())
        unused = folder / manifest["files"]["|u1"]["file"]
        data = bytearray(unused.read_bytes()); data[-1] ^= 1; unused.write_bytes(data)
        with pytest.raises(ValueError, match="checksum"):
            replay.read(episode.episode_id, training=True)
    finally:
        replay.close()


def test_variable_mc_fvi_terminal_and_event(episode_factory):
    _, _, make = episode_factory
    episode = make(lengths=(2, 1), rewards=((1., 2.), (-3.,)))
    gamma = .99**(1/25)
    torch.testing.assert_close(episode.returns, torch.tensor([1+gamma*2-gamma**2*3, -3.], dtype=torch.float64))
    previous = torch.tensor([100., 7.], dtype=torch.float64)
    torch.testing.assert_close(nstep_targets(episode, previous, 1),
                               torch.tensor([1+gamma*2+gamma**2*7, -3.], dtype=torch.float64))
    torch.testing.assert_close(nstep_targets(episode, previous, 16), episode.returns)
    zero = make(lengths=(2, 0), rewards=((1., 2.), ()), event=-5.)
    assert zero.returns[0].item() == pytest.approx(1+gamma*2-gamma**2*5)
    assert zero.returns[-1].item() == -5
    with pytest.raises(ValueError, match="frozen"):
        nstep_targets(episode, previous.requires_grad_())


@pytest.mark.parametrize("fault", ["gap", "duplicate", "admin", "unterminated", "version", "early_terminal", "session"])
def test_invalid_episode_rejected(episode_factory, fault):
    _, _, make = episode_factory
    e = make()
    if fault == "gap": e.rows[-1].control_tick_begin += 12
    if fault == "duplicate": e.rows[-1].identity["decision_id"] = 0
    if fault == "admin": e.rows[-1].truncated = True
    if fault == "unterminated": e.rows[-1].terminated = False
    if fault == "version": e.rows[-1].identity["policy_version"] = 1
    if fault == "early_terminal": e.rows[0].terminated = True
    if fault == "session": e.rows[-1].identity["backend_session_id"] = "other"
    with pytest.raises(ValueError):
        mc_returns(e)


def test_replay_binary_roundtrip_fifo_pins_resume_and_holdout(tmp_path, episode_factory):
    _, _, make = episode_factory
    root = tmp_path/"replay"
    replay = EpisodeReplay(root, capacity=2, contract={"reward": "test"}, heldout_groups=["validation"])
    first = make(episode_id="one")
    assert replay.append(first)
    assert not replay.append(first)
    saved = replay.snapshot()
    restored = replay.read(first.episode_id)
    for a, b in zip(first.rows, restored.rows):
        assert torch.equal(a.target_qpos30, b.target_qpos30)
        for key in a.context:
            assert torch.equal(a.context[key], b.context[key])
        assert torch.equal(a.metadata["critic_context"]["tracking_history"], b.metadata["critic_context"]["tracking_history"])
    assert torch.equal(first.returns, restored.returns)
    replay.append(make(episode_id="two")); replay.append(make(episode_id="three"))
    assert replay.report()["active_episodes"] == 2
    assert replay.collect_garbage() == []
    replay.close()
    replay = EpisodeReplay(root, capacity=2, contract={"reward": "test"}, heldout_groups=["validation"])
    assert replay.load_snapshot(saved.snapshot_id) == saved
    replay.release(saved.snapshot_id)
    assert replay.collect_garbage() == [first.episode_id]
    with pytest.raises(ValueError, match="leaked"):
        replay.append(make(episode_id="val", group="validation"))
    snap = replay.snapshot()
    target = next((root/"episodes"/snap.episode_ids[0]).glob("*.npy"))
    with target.open("r+b") as f:
        f.seek(-1, 2); f.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        replay.read(snap.episode_ids[0])
    replay.close()


@pytest.mark.parametrize("normalize", [True, False])
def test_advantage_formula_and_global_cap(tmp_path, normalize):
    g = torch.tensor([-10., -1., 1., 100.], dtype=torch.float64)
    labels = label_advantages(list("abcd"), g, torch.zeros_like(g), sources=["a", "a", "b", "b"],
                              identity={"replay": "test"}, weight_mean_normalize=normalize)
    expected = (g/g.std(correction=0)).exp().clamp_max(20)
    assert torch.equal(labels.normalized, g/g.std(correction=0))  # 不减均值
    torch.testing.assert_close(labels.raw_weights, expected)
    torch.testing.assert_close(labels.weights, expected/expected.mean() if normalize else expected)
    assert torch.all(labels.weights > 0)
    assert labels.weights[-1] > labels.weights[0]
    assert labels.weights.max() <= labels.statistics["weight_upper_bound"]+1e-12
    labels.save(tmp_path/"labels")
    assert torch.equal(AdvantageLabelSet.load(tmp_path/"labels").weights, labels.weights)
    capped = label_advantages([str(i) for i in range(21)], [1.]+[100.]*20, [0.]*21, sources=["s"]*21,
                             identity={}, weight_mean_normalize=normalize)
    assert torch.all(capped.raw_weights[1:] == 20.)


@pytest.mark.parametrize("values", [[0., 0.], [float("nan"), 1.], [-10000., -9999.], [100., 101.]])
def test_degenerate_advantages_fail(values):
    with pytest.raises((ValueError, FloatingPointError)):
        label_advantages(["a", "b"], values, [0., 0.], sources=["s", "s"], identity={})


def test_stable_connected_music_folds(episode_factory):
    _, _, make = episode_factory
    samples = [dict(split="train", group_id=f"g{i}", audio_sha256=f"a{i}", source="s") for i in range(8)]
    samples.append(dict(split="train", group_id="alias", audio_sha256="a0", source="s"))
    mapping = build_folds(samples)
    assert mapping == build_folds(reversed(samples))
    assert mapping["groups"]["alias"] == mapping["groups"]["g0"]
    episodes = [make(episode_id=str(i), group=f"g{i}", source="s") for i in range(8)]
    for fold in (0, 1):
        train, val, label = partition_episodes(episodes, mapping, fold)
        assert not ({e.group_id for e in train+val} & {e.group_id for e in label})
        assert not ({e.group_id for e in train} & {e.group_id for e in val})
