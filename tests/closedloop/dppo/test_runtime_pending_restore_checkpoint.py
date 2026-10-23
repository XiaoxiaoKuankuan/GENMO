"""初始化快照保留待应用采样恢复状态的独立 CPU 回归。

使用真实 StratifiedTimePool、FullMusicSampler、执行计数恢复和 runtime.collect
入口，替换唯一物理世界为边界探针：先观察预算阶段已打开，再在第一次物理请求
之前取出下一任务与噪声编号。比较不中断首次采集和“初始化快照序列化、重新
构造运行时、恢复、首次采集”两条路径，验证共享时间池、各环境采样器与执行
计数连续；探针主动结束，不运行后续价值计算、优化、Isaac 或 CUDA。

另外检查 active 世界仍拒绝快照、反复保存不消费 pending、嵌套张量/数组/容器
互不别名，以及 apply_restore 只应用一次、应用后重新保存当前逻辑状态。
所有夹具只写 pytest 的临时目录，完整快照的往返序列化在内存完成。本测试不
保存或恢复 PhysX 状态，也不声称真实八卡恢复或后续物理收益已经验收。
"""
from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.closedloop.dppo import runtime as runtime_module
from gem.closedloop.dppo.full_dataset import FullMusicCatalog, FullMusicSampler
from gem.closedloop.dppo.time_sampling import StratifiedTimePool
from tests.closedloop.dppo.test_full_dataset import make_catalog_data


def _assert_same(left, right):
    """逐叶比较任务中的实际音乐数组；不能依赖字典对 ndarray 的隐式真值。"""
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    else:
        assert left == right


class _FirstPhysicalRequest(Exception):
    """探针在实际物理执行前结束；不伪造后续 rollout 的返回结果。"""

    def __init__(self, requests):
        super().__init__('CPU probe reached the first physical request')
        self.requests = requests


class _CollectorProbe:
    def __init__(self, runtime, catalog):
        self.runtime, self.catalog = runtime, catalog
        self.active = False
        self.states = [None] * runtime.num_envs

    def _resource(self, slot):
        if self.states[slot] is None:
            resource = SimpleNamespace(
                sampler=FullMusicSampler(self.catalog, seed=71 + slot,
                    minimum_remaining_seconds=4., time_pool=self.runtime.time_pool),
                env=SimpleNamespace(decision=0, attempt=0, episode_count=0,
                    latency_budget_s=.2, policy_version=10, iteration=10))
            self.states[slot] = SimpleNamespace(resource=resource, transitions=0, task=None)
        return self.states[slot]

    def collect(self, *, world_seconds, policy_version):
        assert world_seconds == 4. and policy_version == 10
        assert self.runtime.opened_phases == 1
        assert not hasattr(self.runtime, 'saved_logical')
        requests = []
        for slot in range(self.runtime.num_envs):
            state = self._resource(slot)
            assert state.task is None  # 只恢复逻辑，旧episode不得继续物理执行。
            task = state.resource.sampler.next_task()
            env = state.resource.env
            requests.append(dict(slot=slot, task=task, transitions=state.transitions,
                decision=env.decision, attempt=env.attempt, episode_count=env.episode_count,
                iteration=env.iteration, policy_version=env.policy_version))
        raise _FirstPhysicalRequest(requests)


def _runtime(catalog):
    result = runtime_module.InProcessTrainingRuntime.__new__(
        runtime_module.InProcessTrainingRuntime)
    result.num_envs = 2
    result.time_pool = StratifiedTimePool(catalog, seed=42, rank=0, world_size=1,
        minimum_remaining_seconds=4., bin_seconds=1., standing_start_probability=0.)
    result.collector = _CollectorProbe(result, catalog)
    result.c = SimpleNamespace(stage={'generation_horizon': {'frames': 30},
        'ppo_rollout': {'world_seconds': 4.}}, state={'policy_version': 10}, distributed=None)
    result.opened_phases = 0

    def open_phase(label, counts, strict, *, world_seconds):
        assert counts == [0, 0] and world_seconds == 4.
        result.opened_phases += 1
    result._open = open_phase
    return result


@pytest.fixture
def source(tmp_path):
    catalog = FullMusicCatalog(make_catalog_data(tmp_path, frames=300))
    runtime = _runtime(catalog)
    for slot in range(runtime.num_envs):
        state = runtime.collector._resource(slot)
        for _ in range(7 + slot):
            state.task = state.resource.sampler.next_task()
            state.resource.sampler.record_execution(state.task, 25)
        state.transitions = 19 + slot
        env = state.resource.env
        env.decision, env.attempt, env.episode_count = 17 + slot, 23 + slot, 4 + slot
    return catalog, runtime


def _first_request(runtime, monkeypatch):
    monkeypatch.setattr(runtime_module, 'local_call', lambda collective, function: function())
    with pytest.raises(_FirstPhysicalRequest) as reached:
        runtime.collect('first_after_restore')
    return reached.value.requests


@pytest.mark.parametrize('checkpoint_restarts', [1, 2])
def test_pending_initial_snapshot_resume_matches_uninterrupted_first_collect(
        source, monkeypatch, checkpoint_restarts):
    catalog, original = source
    saved = original.checkpoint_state()
    direct = _runtime(catalog)
    direct.restore_logical(copy.deepcopy(saved))
    expected = _first_request(direct, monkeypatch)

    restarted = _runtime(catalog)
    restarted.restore_logical(copy.deepcopy(saved))
    for _ in range(checkpoint_restarts):
        stream = io.BytesIO()
        torch.save(restarted.checkpoint_state(), stream)
        stream.seek(0)
        restored = torch.load(stream, map_location='cpu', weights_only=False)
        restarted = _runtime(catalog)
        restarted.restore_logical(restored)
    _assert_same(_first_request(restarted, monkeypatch), expected)
    _assert_same(restarted.checkpoint_state(), direct.checkpoint_state())
    assert [request['attempt'] for request in expected] == [23, 24]
    assert [request['transitions'] for request in expected] == [19, 20]
    assert restarted.time_pool.state_dict()['motion_draws'] == saved['time_pool']['motion_draws'] + 2


def test_pending_snapshots_do_not_consume_or_alias_nested_state(source):
    catalog, original = source
    saved = original.checkpoint_state()
    saved['opaque_diagnostic'] = dict(tensor=torch.tensor([2., 3.]),
        array=np.array([5, 7]), nested=[{'values': [11, 13]}])
    pending = _runtime(catalog)
    pending.restore_logical(saved)
    fresh_pool = pending.time_pool.state_dict()
    first, second = pending.checkpoint_state(), pending.checkpoint_state()
    assert pending.saved_logical is saved
    assert pending.collector.states == [None, None]
    assert pending.time_pool.state_dict() == fresh_pool
    assert first['time_pool'] == second['time_pool'] == saved['time_pool']
    first['states'][0]['execution']['attempt'] = 999
    first['opaque_diagnostic']['tensor'].fill_(17)
    first['opaque_diagnostic']['array'][:] = 19
    first['opaque_diagnostic']['nested'][0]['values'].append(23)
    for unchanged in (second, saved):
        assert unchanged['states'][0]['execution']['attempt'] == 23
        assert torch.equal(unchanged['opaque_diagnostic']['tensor'], torch.tensor([2., 3.]))
        np.testing.assert_array_equal(unchanged['opaque_diagnostic']['array'], [5, 7])
        assert unchanged['opaque_diagnostic']['nested'][0]['values'] == [11, 13]
    second['time_pool']['cursors']['Mine'] = -1
    assert saved['time_pool']['cursors']['Mine'] >= 0


@pytest.mark.parametrize('pending', [False, True])
def test_active_world_still_rejects_checkpoint_before_pending_branch(source, pending):
    catalog, original = source
    target = _runtime(catalog)
    if pending:
        target.restore_logical(original.checkpoint_state())
    target.collector.active = True
    with pytest.raises(RuntimeError, match='idle world'):
        target.checkpoint_state()
    assert hasattr(target, 'saved_logical') is pending


def test_apply_once_then_checkpoint_captures_current_logical_state(source):
    catalog, original = source
    saved = original.checkpoint_state()
    target = _runtime(catalog)
    target.restore_logical(copy.deepcopy(saved))
    target.apply_restore()
    assert not hasattr(target, 'saved_logical')
    for state in target.collector.states:
        assert state.task is None
        state.resource.sampler.next_task()
        state.resource.env.attempt += 1
        state.transitions += 1
    advanced = target.checkpoint_state()
    target.apply_restore()
    assert target.checkpoint_state() == advanced
    assert advanced['time_pool']['motion_draws'] == saved['time_pool']['motion_draws'] + 2
    assert [s['execution']['attempt'] for s in advanced['states']] == [24, 25]
    assert [s['transitions'] for s in advanced['states']] == [20, 21]
    assert all(s['ended_on_restore_episode'] is None for s in advanced['states'])
    assert saved['states'][0]['ended_on_restore_episode'] is not None


def test_already_initialized_checkpoint_preserves_existing_capture_path(source):
    _, original = source
    before = original.time_pool.state_dict()
    saved = original.checkpoint_state()
    assert not hasattr(original, 'saved_logical')
    assert saved['time_pool'] == before == original.time_pool.state_dict()
    assert saved['restore'] == 'fresh_PhysX_episodes_no_partial_rollout_reuse'
    assert saved['generation_horizon_frames'] == 30
    assert saved['environment_count'] == 2
    for slot, state in enumerate(original.collector.states):
        assert saved['states'][slot]['sampler'] == state.resource.sampler.state_dict()
        assert saved['states'][slot]['ended_on_restore_episode'] == state.task['sample']['row']['sample_id']


def test_pending_empty_lane_remains_empty_until_first_collection(source):
    catalog, original = source
    saved = original.checkpoint_state()
    saved['states'][1] = None
    target = _runtime(catalog)
    target.restore_logical(saved)
    assert target.checkpoint_state()['states'][1] is None
    target.apply_restore()
    assert target.collector.states[1] is None
    assert target.checkpoint_state()['states'][1] is None
