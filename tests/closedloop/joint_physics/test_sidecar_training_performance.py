"""监督sidecar性能修复的合同与数值回归。

临时真实格式分片检查字节预算LRU、复用反序列化结果、超预算读取和文件篡改拒绝；
小型原GENMO/codec夹具对比原GPU逐样本装配与CPU整批搬运的条件、有效标记及前向。
失败窗口、已知前缀、未来标签隔离和缓存恢复均不得因性能修复改变。这里不运行Newton，
不将CPU或小网络结果宣称为正式八卡吞吐；所有产物限定pytest临时目录。
"""

import copy
import json
import os

import pytest
import torch

from gem.closedloop.joint_physics.actor import JointPhysicsActor
from gem.closedloop.joint_physics.contracts import SIDECAR_SCHEMA, PhysicsRejected, digest, source_key
from gem.closedloop.joint_physics.data import SidecarStore
from gem.closedloop.joint_physics.reference import attach_sidecars, attach_supervised_sidecars
from gem.closedloop.mb_control.contracts import ForecastContext
from gem.closedloop.training import batch_to_device
from tests.closedloop.joint_physics.test_batched_offline import _adapter, _fixture


def _shards(tmp_path):
    motion, music = tmp_path / 'motion', tmp_path / 'music'
    motion.write_bytes(b'motion'); music.write_bytes(b'music')
    index = dict(schema=SIDECAR_SCHEMA, split='train', assets={}, sources={}, shards={})
    metadata = []
    for i in range(3):
        meta = dict(dataset_id='fixture', split='train', sample_id=str(i),
                    motion_path=str(motion), music_feature_path=str(music))
        key, name = source_key(meta), f'shard_{i}.pt'
        torch.save([dict(source_key=key, split='train', motion_sha256=digest(motion),
                         music_sha256=digest(music), state=torch.zeros(4, 55))], tmp_path / name)
        index['sources'][key] = dict(shard=name, episode=0)
        index['shards'][name] = digest(tmp_path / name)
        metadata.append(meta)
    (tmp_path / 'index.json').write_text(json.dumps(index))
    return metadata, [(tmp_path / f'shard_{i}.pt').stat().st_size for i in range(3)]


def test_mmap_shards_reused_with_bounded_lru_and_restore(tmp_path, monkeypatch):
    metadata, sizes = _shards(tmp_path)
    calls, load = [], torch.load
    def counted(*args, **kwargs):
        calls.append(args[0]); return load(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', counted)
    store = SidecarStore(tmp_path, 'train', cache_bytes=sum(sizes))
    for _ in range(2):
        for meta in metadata:
            store.episode(meta)
    assert len(calls) == 3  # 旧两分片缓存的相同访问会加载六次。
    assert store.cache_bytes == sum(sizes)
    restored = SidecarStore(tmp_path, 'train', snapshot=store.snapshot(), cache_bytes=0)
    for meta in metadata:
        torch.testing.assert_close(restored.episode(meta)['state'], store.episode(meta)['state'])
    assert restored.cache_bytes == 0 and not restored.cache
    limited = SidecarStore(tmp_path, 'train', cache_bytes=sum(sizes[:2]))
    limited.episode(metadata[0]); limited.episode(metadata[1]); limited.episode(metadata[0])
    limited.episode(metadata[2])
    assert 'shard_0.pt' in limited.cache and 'shard_1.pt' not in limited.cache
    assert limited.cache_bytes <= limited.max_cache_bytes


def test_cached_shard_mutation_still_rejected(tmp_path):
    metadata, sizes = _shards(tmp_path)
    store = SidecarStore(tmp_path, 'train', cache_bytes=sum(sizes))
    store.episode(metadata[0])
    path = tmp_path / 'shard_0.pt'
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match='分片发生改变'):
        store.episode(metadata[0])


class _Store:
    def __init__(self, episode):
        self.value = episode

    def episode(self, meta):
        if meta['sample_id'] == 'missing':
            raise PhysicsRejected('missing_real_sidecar')
        return self.value

    def request(self, episode, frame, prefix_frames, device, support_guard=None):
        old = episode['reference_qpos']
        from gem.closedloop.joint_physics.reference import episode_reference
        reference = episode_reference(episode, _adapter(), 'cpu', visible_upper=int(episode['visible_upper'][frame*20//12]))
        first = reference.first_influence(frame*20, (frame+prefix_frames)*20)
        return ForecastContext(old.new_zeros(1, 58).to(device), torch.ones(1, dtype=torch.bool, device=device), frame*20, first)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='需要CUDA数值对照'))])
def test_supervised_cpu_assembly_matches_original_conditions_and_forward(actor_factory, device):
    base, batch = actor_factory(motion_frames=60, prefix=15, starts=(0, 30, 60, 3))
    batch['meta'][-1]['sample_id'] = 'missing'
    episode, _, _ = _fixture()
    adapter, store = _adapter(), _Store(episode)
    actor = JointPhysicsActor(base).to(device).eval()
    old, old_contexts, _, old_reasons = attach_sidecars(batch_to_device(batch, device), store, adapter, base.endecoder.codec, 32)
    new, contexts, reasons = attach_supervised_sidecars(batch, store, adapter, base.endecoder.codec, 32, device)
    assert reasons == old_reasons
    assert 'available' in reasons and 'incomplete_real_window' in reasons and 'missing_real_sidecar' in reasons
    for key, value in old.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(new[key], value, rtol=1e-6, atol=1e-6)
        else:
            assert new[key] == value
    for first, second in zip(old_contexts, contexts):
        torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
        torch.testing.assert_close(first.valid, second.valid, rtol=0, atol=0)
        assert (first.request_tick, first.forecast_tick) == (second.request_tick, second.forecast_tick)
    noise = torch.randn(4, 60, 30, device=device)
    k = torch.tensor([50, 100, 250, 999], device=device)
    drop = torch.zeros(4, dtype=torch.bool, device=device)
    with torch.no_grad():
        first = actor.denoise_supervised(old, old_contexts, k, noise, drop)
        second = actor.denoise_supervised(new, contexts, k, noise, drop)
    torch.testing.assert_close(first['pred_x_start'], second['pred_x_start'], rtol=1e-5, atol=1e-6)
    changed = copy.deepcopy(batch)
    changed['target_qpos30'][~changed['known_qpos30_mask']] += 19.
    changed['target_contact'] = 1-changed['target_contact']
    other, other_contexts, _ = attach_supervised_sidecars(changed, store, adapter, base.endecoder.codec, 32, device)
    for key in ('known_qpos30', 'proprio_history', 'proprio_history_valid', 'proprio_history_times'):
        torch.testing.assert_close(new[key], other[key], rtol=0, atol=0)
    for first, second in zip(contexts, other_contexts):
        torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
