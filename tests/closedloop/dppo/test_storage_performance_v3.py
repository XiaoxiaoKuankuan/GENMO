"""第二阶段CPU采集与存储优化的等价性及故障验收。

使用独立临时文件、socket字节和CPU价值网络检查不可变资产缓存的失效与容量、
RPC去除中间副本后的独立快照、批量旧价值与逐条参考、音乐节拍缓存的重置。
后续持久化格式测试沿用原UpperTransition及预算语义，所有产物由pytest basetemp
隔离，不读取正式训练目录；模拟故障不能当作真实八卡物理验收结果。
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.asset_cache import ImmutableBytesCache
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.closedloop.dppo.trainer import populate_values, fixed_targets
from gem.runtime.closedloop_protocol import _pack, _unpack


def test_asset_cache_bounded_and_rejects_changed_content(tmp_path):
    first, second = tmp_path/'a', tmp_path/'b'
    first.write_bytes(b'abcd'); second.write_bytes(b'12345')
    cache = ImmutableBytesCache(8)
    original = cache.read(first)
    assert cache.read(first) == original and cache.hits == 1
    cache.read(second)
    assert cache.bytes == 5 and first not in cache.entries
    first.write_bytes(b'wxyz')
    assert cache.read(first)[1] != original[1]
    cache.clear()
    assert not cache.entries and cache.bytes == 0


def test_rpc_contiguous_noncontiguous_arrays_remain_independent_after_encode():
    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    message = dict(a=array, b=array[:, ::2], c=array, text=np.array(['中文', 'x']))
    packed = _pack(message)
    array[:] = -1
    decoded = _unpack(*packed)
    np.testing.assert_array_equal(decoded['a'], np.arange(24).reshape(4, 6))
    np.testing.assert_array_equal(decoded['b'], np.arange(24).reshape(4, 6)[:, ::2])
    decoded['a'][:] = 0
    assert decoded['c'][0, 1] == 1  # 不静默引入字段之间的可写别名
    assert decoded['text'][0] == '中文'


def test_music_beats_recomputed_only_when_episode_source_changes(monkeypatch):
    music = np.zeros((100, 35)); music[[3, 7, 20], 34] = 1.
    reward = ExecutionReward(music_features=music)
    expected = 600 + np.rint(np.array([3, 7, 20])*20/12).astype(np.int64)*12
    np.testing.assert_array_equal(reward._beat_ticks, expected)
    reward.reset(music_start_tick=1200)
    np.testing.assert_array_equal(reward._beat_ticks, expected+600)
    music[:, 34] = 0.
    reward.reset(music_features=music)
    assert len(reward._beat_ticks) == 0


def test_batched_values_preserve_fixed_targets_and_terminal_bootstrap():
    class Critic(torch.nn.Module):
        def forward(self, context, remaining):
            return context['x'].sum(-1) + remaining*.25
    rows = []
    for i in range(5):
        rows.append(SimpleNamespace(context={'x': torch.tensor([[float(i), 2.]])},
            next_context={'x': torch.tensor([[float(i+1), 2.]])}, terminated=i==4,
            truncated=i==2, transition_valid=True, rewards=torch.tensor([1., 2.]),
            executed_control_steps=2, control_tick_begin=24*i, control_tick_end=24*(i+1),
            identity=dict(backend_session_id='s', episode_id='e' if i<3 else 'f', policy_version=1),
            metadata=dict(remaining_music_seconds=float(10-i), next_remaining_music_seconds=float(9-i))))
    reference = copy.deepcopy(rows)
    populate_values(reference, Critic(), 'cpu', critic_version=0)
    populate_values(rows, Critic(), 'cpu', critic_version=0, batch_size=3)
    assert [r.old_value for r in rows] == [r.old_value for r in reference]
    assert [r.next_value for r in rows] == [r.next_value for r in reference]
    assert rows[-1].next_value == 0.
    a = fixed_targets(rows, Critic(), 'cpu', reuse_values=True, critic_version=0)
    b = fixed_targets(reference, Critic(), 'cpu', reuse_values=True, critic_version=0)
    for key in a:
        if torch.is_tensor(a[key]):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        else:
            assert a[key] == b[key]
    with pytest.raises(ValueError, match='positive'):
        populate_values(rows, Critic(), 'cpu', batch_size=0)


def test_incremental_budget_batch_resume_reference_and_no_duplicate_refund(tmp_path):
    from gem.closedloop.dppo.budget_ledger import IncrementalBudget, read_budget_state
    from gem.closedloop.dppo.run_management import TrainingBudget, validate_budget_progress
    limits = dict(accepted_iterations=10, optimizer_attempts=40, generations=100,
                  control_steps=1000, physics_steps=4000)
    path = tmp_path/'budget.json'
    old = TrainingBudget(path, limits)
    old.reserve('legacy', generations=1)
    baseline = old.state_dict()
    budget = IncrementalBudget(path, limits)
    entries = [(f'round/rank{i}', dict(generations=5, control_steps=50, physics_steps=200)) for i in range(8)]
    budget.reserve_many(entries)
    assert budget.sequence == 1 and budget.state['used']['generations'] == 41
    credit = dict(generations=3, control_steps=30, physics_steps=120)
    budget.settle_many([(phase, amount, credit, phase) for phase, amount in entries])
    assert budget.sequence == 2 and budget.state['used']['generations'] == 25
    ref = budget.summary()
    for phase, amount in entries:
        budget.settle_lease(phase, amount, credit, lease_id=phase)
    assert budget.sequence == 2
    with pytest.raises(ValueError, match='settled'):
        budget.reserve(entries[0][0], generations=1)
    budget.accept_iteration(identity='session/1')
    with pytest.raises(ValueError, match='Duplicate'):
        budget.accept_iteration(identity='session/1')
    saved = budget.state_dict()
    assert 'phases' not in ref and len(str(ref)) < 600
    budget.close()
    restored = IncrementalBudget(path, limits)
    assert restored.state_dict() == saved
    validate_budget_progress(baseline, restored.state_dict())
    assert read_budget_state(path, ref)['used']['generations'] == 25
    bad = copy.deepcopy(ref); bad['used']['generations'] = 0
    with pytest.raises(ValueError, match='differs'):
        read_budget_state(path, bad)
    restored.close()


def test_incremental_budget_failed_transaction_does_not_authorize_execution(tmp_path):
    from gem.closedloop.dppo.budget_ledger import IncrementalBudget, replay_budget
    limits = dict(accepted_iterations=1, optimizer_attempts=4, generations=8,
                  control_steps=10, physics_steps=40)
    budget = IncrementalBudget(tmp_path/'budget.json', limits)
    budget.connection.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT,'disk injection'); END")
    with pytest.raises(Exception, match='disk injection'):
        budget.reserve('x', generations=1)
    assert budget.state['used']['generations'] == 0 and budget.poisoned
    with pytest.raises(RuntimeError, match='recovery'):
        budget.reserve('x', generations=1)
    budget.close()
    state, seq, _ = replay_budget(tmp_path/'budget.json')
    assert state['used']['generations'] == seq == 0


def test_incremental_budget_corrupt_history_rejected(tmp_path):
    from gem.closedloop.dppo.budget_ledger import IncrementalBudget, replay_budget
    limits = dict(accepted_iterations=1, optimizer_attempts=4, generations=8,
                  control_steps=10, physics_steps=40)
    budget = IncrementalBudget(tmp_path/'budget.json', limits)
    budget.reserve('x', generations=1)
    with budget.connection:
        budget.connection.execute("UPDATE events SET sha256='bad'")
    budget.close()
    with pytest.raises(ValueError, match='SHA'):
        replay_budget(tmp_path/'budget.json')


def test_block_rollout_keeps_one_raw_chain_and_detects_corruption(tmp_path):
    import json
    import hashlib
    from gem.closedloop.dppo.buffer import UpperTransition
    from gem.closedloop.dppo.rollout_storage import BlockRolloutWriter, load_rollout_record
    from gem.closedloop.dppo.offline_diagnostics import load_immutable_rollouts
    from tests.closedloop.dppo.test_stage10_entrypoint import transition
    # 沿用入口测试的标准有效条件与120x30链，不建立另一套概率/执行字段。
    row = transition().snapshot()
    row.metadata["sampler_trace"] = {}
    row.metadata['sampler_trace']['chain'] = row.chain[None]
    row.metadata['sampler_trace']['old_log_probs'] = row.old_log_prob[None]
    row.metadata['sampler_trace']['free_mask'] = row.free_mask[None]
    row.metadata['generated'] = {'test': 'original'}
    raw_dir = tmp_path/'raw_samples'; raw_dir.mkdir()
    path = raw_dir/'one.pt'
    torch.save(dict(trace=row.metadata['sampler_trace'], generated=row.metadata['generated'],
                    policy_version=row.identity['policy_version']), path)
    row.metadata.update(raw_sample_path=str(path), raw_evidence_identity=dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size))
    writer = BlockRolloutWriter(tmp_path/'rollout', policy_version=row.identity['policy_version'], chunk_size=2)
    writer.append(row)
    assert not list((tmp_path/'rollout').rglob('*.pt'))
    second = copy.deepcopy(row); second.identity['decision_id'] += 1
    writer.append(second)
    assert len(list((tmp_path/'rollout').rglob('*.pt'))) == 1
    manifest = writer.finish()
    published = json.loads(manifest.read_text())
    chunk_path = manifest.parent/published['chunks'][0]['path']
    chunk = json.loads(chunk_path.read_text())
    record = chunk['records'][0]
    compact = torch.load(chunk_path.parent/record['path'], weights_only=False)
    assert 'chain' not in compact[0]['fields'] and 'sampler_trace' not in compact[0]['fields']['metadata']
    restored = load_rollout_record(chunk_path.parent/record['path'], record, rank_directory=tmp_path)
    assert isinstance(restored, UpperTransition)
    torch.testing.assert_close(restored.chain, row.chain, rtol=0, atol=0)
    assert restored.metadata['generated'] == row.metadata['generated']
    targets = tmp_path/'fixed_targets.pt'
    torch.save(dict(advantages=torch.ones(2), returns=torch.ones(2), valid=torch.ones(2, dtype=torch.bool)), targets)
    loaded, _, _ = load_immutable_rollouts([manifest], [targets], max_chains=2)
    assert len(loaded) == 2
    path.write_bytes(b'corrupt raw')
    with pytest.raises(ValueError, match='Raw trace size/SHA'):
        load_rollout_record(chunk_path.parent/record['path'], record, rank_directory=tmp_path)


def test_raw_rpc_old_npz_compatible_and_rejects_descriptor_corruption():
    import json
    from gem.runtime.closedloop_protocol import _pack_legacy
    source = dict(x=np.arange(12, dtype='>f4').reshape(3, 4), empty=np.zeros((0, 4)),
                  scalar=np.array(2., dtype=np.float64), labels=np.array(['中文', 'test']))
    for packed in (_pack(source), _pack_legacy(source)):
        restored = _unpack(*packed)
        for key in source:
            assert restored[key].dtype == source[key].dtype
            np.testing.assert_array_equal(restored[key], source[key])
    metadata, payload = _pack(source)
    bad = json.loads(metadata)
    bad['value']['x']['__ndarray_raw__']['size'] += 1
    with pytest.raises(ValueError, match='byte count'):
        _unpack(json.dumps(bad).encode(), payload)
    bad['value']['x']['__ndarray_raw__']['offset'] = len(payload)+1
    with pytest.raises(ValueError, match='bounds'):
        _unpack(json.dumps(bad).encode(), payload)

def test_incremental_partial_control_settlement_is_atomic_idempotent_and_replayable(tmp_path):
    from gem.closedloop.dppo.budget_ledger import IncrementalBudget,replay_budget
    limits=dict(accepted_iterations=2,optimizer_attempts=8,generations=40,control_steps=1000,physics_steps=4000)
    budget=IncrementalBudget(tmp_path/'budget.json',limits)
    budget.reserve('train',control_steps=25,physics_steps=100)
    unknown=dict(physics_count_exact=False)
    budget.settle_control('train',25,unknown)
    assert budget.state['used']['physics_steps']==100
    result=dict(backend_session_id='env3',mutation_seq=5,physics_count_exact=True,
        executed_control_steps=7,executed_physics_steps=28)
    budget.settle_control('train',25,result)
    sequence=budget.sequence
    budget.settle_control('train',25,result)
    assert budget.sequence==sequence
    assert budget.state['used']['control_steps']==7 and budget.state['used']['physics_steps']==28
    with pytest.raises(ValueError,match='payload changed'):
        budget.settle_control('train',25,dict(result,executed_control_steps=8,executed_physics_steps=32))
    expected=budget.state_dict();budget.close()
    reopened=IncrementalBudget(tmp_path/'budget.json',limits)
    assert reopened.state_dict()==expected
    reopened.settle_control('train',25,result)
    assert reopened.sequence==sequence
    reopened.close()
