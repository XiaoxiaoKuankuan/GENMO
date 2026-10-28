"""离线批量FutureContext的时间、因果和数值合同回归。

使用小型确定性SM/WM替身及原ReferenceBuffer/forecast_rollout/SMAdapter算法，比较旧单请求
路径与新批量路径，覆盖不同歌曲、绝对请求时刻、已知前缀、真实历史及非有限行隔离。
测试不启动Newton、不加载训练资产，不将这些CPU合同结果宣称为真实物理或吞吐验收。
"""

from copy import deepcopy
from types import SimpleNamespace

import torch

from gem.closedloop.joint_physics import offline
from gem.closedloop.joint_physics.contracts import PhysicsRejected
from gem.closedloop.joint_physics.reference import make_window
from gem.closedloop.mb_control.contracts import ReferenceExhausted
from gem.closedloop.mb_control.forecast import forecast_rollout
from gem.closedloop.mb_control.sm_adapter import SMAdapter


class _Kinematics:
    device = torch.device("cpu")

    def forward(self, p, r, v, w, q, qd):
        return p.unsqueeze(-2), r.unsqueeze(-3), v.unsqueeze(-2), w.unsqueeze(-2)


class _Bundle:
    def __init__(self):
        self.batch_sizes = []

    def act(self, history, goal):
        return .1*history[:, -1, :21]+.01*goal[:, :21]

    def step_model(self, state, action, xi):
        self.batch_sizes.append(len(state))
        result = state.clone()
        result[:, :3] += .02*state[:, 7:10]
        result[:, 13:34] += .02*action
        result[:, 34:] = action
        return result


def _adapter():
    adapter = object.__new__(SMAdapter)
    adapter.bundle = _Bundle()
    adapter.to_sm = adapter.to_genmo = adapter.to_history = tuple(range(21))
    def observation(state, action, xi, reference):
        return torch.cat((state[:, 13:34]-reference.q, state[:, 34:], action,
                          state[:, :3]-reference.root_p, state[:, 7:13], state[:, :3]), -1)
    adapter.runtime = SimpleNamespace(kin=_Kinematics(), observation=observation,
                                     mdp=SimpleNamespace(default_q=torch.zeros(21)))
    return adapter


def _fixture(name="a", offset=0.):
    n, controls = 90, 130
    qpos = torch.zeros(n, 28)
    qpos[:, 0] = torch.arange(n)*.003+offset
    qpos[:, 2], qpos[:, 3] = .7, 1.
    qpos[:, 7:] = torch.sin(torch.arange(n)[:, None]*.03)*.04
    state = torch.zeros(controls, 55)
    state[:, 0] = torch.arange(controls)*.002+offset
    state[:, 2], state[:, 3], state[:, 7] = .71, 1., .1
    state[:, 13:34] = torch.cos(torch.arange(controls)[:, None]*.02)*.03
    episode = dict(source_key=name, sample_id=name, split="train", source_frames=n,
        reference_qpos=qpos, source_versions=torch.arange(n)//30, state=state,
        xi=torch.zeros(25), observation=torch.arange(controls)[:, None].expand(-1, 75)*.0001,
        proprio=torch.arange(controls)[:, None].expand(-1, 48)*.0001,
        previous_action=torch.zeros(controls, 21), visible_upper=torch.full((controls,), (n-1)*20),
        committed_tick=torch.zeros(controls, dtype=torch.long), plan_version=torch.full((controls,), 2),
        ground_height_world_m=0., frame_origin_tick=0)
    dataset = SimpleNamespace(rows=[{"sample_id": name}], split="train", motion_frames=60,
        _requested_prefix_frames=lambda _name, frame: 15 if (frame//3)%2 else 18)
    def forbidden_window(*_):
        raise AssertionError("批量前瞻不应装配完整监督标签或读取未知真实姿态")
    dataset.get_window = forbidden_window
    return episode, dataset, 0


class _Codec:
    def encode_condition_features(self, pose):
        return (pose.new_zeros((*pose.shape[:-1], 30)),
                SimpleNamespace(position_w=pose[:, :1, :3], yaw=pose.new_zeros((len(pose), 1))))


def _scalar_requests(episode, dataset, adapter):
    results = []
    for frame in range(0, episode["source_frames"], 3):
        p = min(dataset._requested_prefix_frames(episode["sample_id"], frame),
                min(60, episode["source_frames"]-frame)-1)
        sample = {"target_qpos30": torch.zeros(1, 60, 30), "prefix_frames": torch.tensor([p]),
                  "known_qpos30_mask": torch.zeros(1, 60, 30, dtype=torch.bool),
                  "meta": [{"decision_frame": frame}]}
        try:
            window, _ = make_window(episode, sample, adapter, _Codec(), 1)
            trace = []
            result = forecast_rollout(window.request, adapter, window.start.tick,
                                      on_step=lambda z: trace.append(z.state))
            results.append((True, result.context.features[0], window.start.tick,
                            torch.cat(trace) if trace else result.predicted.state))
        except (PhysicsRejected, ReferenceExhausted, FloatingPointError):
            results.append((False, None, 0, None))
    return results


def test_batched_matches_scalar_forecast_with_independent_absolute_ticks():
    entries = [_fixture("a"), _fixture("b", .2)]
    expected = [_scalar_requests(episode, dataset, _adapter()) for episode, dataset, _ in entries]
    adapter = _adapter()
    summaries = offline.cache_requests_batched(entries, adapter, batch_size=16)
    assert max(adapter.bundle.batch_sizes) > 1
    assert sum(adapter.bundle.batch_sizes) > len(adapter.bundle.batch_sizes)
    for (episode, _, _), scalar, summary in zip(entries, expected, summaries):
        actual = episode["offline_requests"]
        assert actual["valid"].tolist() == [item[0] for item in scalar]
        assert summary["valid_requests_per_second"] > 0
        for index, (valid, feature, tick, trace) in enumerate(scalar):
            if valid:
                torch.testing.assert_close(actual["features"][index], feature, atol=1e-6, rtol=1e-5)
                assert int(actual["forecast_ticks"][index]) == tick
                start, end = actual["state_offsets"][index:index+2]
                torch.testing.assert_close(actual["predicted_states"][start:end], trace, atol=1e-6, rtol=1e-5)
        assert torch.equal(actual["request_ticks"], actual["frames"]*20)
        assert torch.all(actual["reference_versions"][actual["valid"]] == 2)


def test_future_measurements_and_uncommitted_reference_do_not_enter_first_context():
    entry = _fixture()
    episode = entry[0]
    episode["visible_upper"].fill_(900)
    changed = deepcopy(entry)
    changed[0]["state"][1:] += 17.
    changed[0]["observation"][1:] += 18.
    changed[0]["proprio"][1:] += 19.
    changed[0]["reference_qpos"][46:, :3] += 20.
    offline.cache_requests_batched([entry, changed], _adapter(), batch_size=64)
    first, other = episode["offline_requests"], changed[0]["offline_requests"]
    assert bool(first["valid"][0]) and bool(other["valid"][0])
    torch.testing.assert_close(first["features"][0], other["features"][0], atol=0., rtol=0.)


def test_nonfinite_request_isolated_without_rejecting_good_batch():
    good, bad = _fixture("good"), _fixture("bad")
    bad[0]["state"][:, 13] = float("nan")
    adapter = _adapter()
    offline.cache_requests_batched([good, bad], adapter, batch_size=64)
    assert bool(good[0]["offline_requests"]["valid"].any())
    assert not bool(bad[0]["offline_requests"]["valid"].any())
    assert max(adapter.bundle.batch_sizes) > 1
    assert "offline_forecast_incomplete_or_nonfinite" in bad[0]["offline_requests"]["reasons"]


def test_future_plan_commit_rejected_before_model_call():
    entry = _fixture()
    entry[0]["committed_tick"] = torch.arange(len(entry[0]["state"]))*12+60
    adapter = _adapter()
    offline.cache_requests(*entry, adapter)
    assert not adapter.bundle.batch_sizes
    assert "future_plan_leak" in entry[0]["offline_requests"]["reasons"]


def test_zero_step_forecast_keeps_real_request_and_no_world_model_call():
    entry = _fixture()
    entry[1]._requested_prefix_frames = lambda *_: 1
    adapter = _adapter()
    expected = _scalar_requests(entry[0], entry[1], _adapter())
    offline.cache_requests(*entry, adapter)
    actual = entry[0]["offline_requests"]
    assert bool(actual["valid"][0]) and expected[0][0]
    assert int(actual["forecast_ticks"][0]) == 0
    assert not adapter.bundle.batch_sizes
    torch.testing.assert_close(actual["features"][0], expected[0][1])
    torch.testing.assert_close(actual["predicted_states"], entry[0]["state"][:1])
