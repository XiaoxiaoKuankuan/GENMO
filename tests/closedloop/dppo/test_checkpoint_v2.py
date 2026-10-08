"""第二阶段 v2 多 rank 完整恢复与显式权重评估的故障边界测试。

测试使用很小的 CPU 模型和独立临时目录，不启动分布式训练、物理环境或长训练。
验证共享模型/优化器只保存一套，各 rank 的采样位置和命名 RNG 分别恢复；拓扑、
边界或 sampler 身份错误须在修改真实模型前拒绝。另验证评估权重加载不改变 RNG。
"""
import copy
import random

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.checkpoint import (
    VERSION_V2,
    capture_rank_state,
    capture_rng,
    load_checkpoint,
    load_weights_checkpoint,
    restore_rng,
    save_checkpoint,
)
from gem.closedloop.dppo.long_run import LongRunMaintenance
from gem.closedloop.dppo.run_management import RunManager, _read_json


class Sampler:
    def __init__(self, position=0, identity='catalog'):
        self.position, self.identity = position, identity

    def state_dict(self):
        return dict(position=self.position, identity=self.identity)

    def load_state_dict(self, state):
        if state['identity'] != self.identity:
            raise ValueError('Sampler identity mismatch')
        self.position = state['position']


@pytest.fixture(autouse=True)
def cpu_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    saved = capture_rng()
    yield
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    restore_rng(saved)


def objects():
    actor, critic = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    optimizers = (torch.optim.AdamW(actor.parameters(), lr=.002),
                  torch.optim.AdamW(critic.parameters(), lr=.003))
    for model, optimizer in zip((actor, critic), optimizers):
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    return dict(actor=actor, critic=critic, actor_optimizer=optimizers[0], critic_optimizer=optimizers[1])


def fixture_checkpoint(tmp_path):
    models = objects()
    states = []
    for rank in range(2):
        random.seed(100 + rank)
        np.random.seed(100 + rank)
        torch.manual_seed(100 + rank)
        states.append(capture_rank_state(rank, state=dict(iteration=300, policy_version=4,
            buffer_size=0, pending_plan=False, decision=rank + 1),
            samplers=dict(music=Sampler(rank + 3)),
            generators=dict(collector=torch.Generator().manual_seed(rank + 20))))
    path = tmp_path / 'v2.pt'
    save_checkpoint(path, **models, state=dict(iteration=300, policy_version=4, buffer_size=0,
        pending_plan=False), identity={'assets': 'test'}, config={}, version=VERSION_V2, rank_states=states)
    return models, states, path


def test_shared_state_and_local_rank_exact_resume(tmp_path):
    models, states, path = fixture_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    assert payload['world_size'] == 2 and set(payload) & {'rng', 'samplers'} == set()
    original = copy.deepcopy(models['actor'].state_dict())
    fresh = objects()
    sampler, generator = Sampler(), torch.Generator().manual_seed(999)
    state = load_checkpoint(path, **fresh, identity={'assets': 'test'}, samplers=dict(music=sampler),
                            generators=dict(collector=generator), rank=1, world_size=2)
    assert state['iteration'] == 300 and state['local_rank_state']['decision'] == 2
    assert sampler.position == 4
    assert torch.equal(generator.get_state(), states[1]['rng']['generators']['collector'])
    assert torch.equal(torch.get_rng_state(), states[1]['rng']['torch'])
    assert all(torch.equal(original[name], value) for name, value in fresh['actor'].state_dict().items())
    for model, optimizer in ((fresh['actor'], fresh['actor_optimizer']),
                             (models['actor'], models['actor_optimizer'])):
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
    assert all(torch.equal(models['actor'].state_dict()[name], value)
               for name, value in fresh['actor'].state_dict().items())


@pytest.mark.parametrize('change', ['world_size', 'rank', 'boundary', 'duplicate', 'sampler'])
def test_v2_rejects_bad_restore_before_model_mutation(tmp_path, change):
    _, _, path = fixture_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    kwargs = dict(rank=0, world_size=2)
    if change == 'world_size':
        kwargs['world_size'] = 8
    elif change == 'rank':
        kwargs['rank'] = 2
    elif change == 'boundary':
        payload['rank_states'][1]['state']['pending_plan'] = True
    elif change == 'duplicate':
        payload['rank_states'][1]['rank'] = 0
    elif change == 'sampler':
        payload['rank_states'][0]['samplers']['music']['identity'] = 'wrong'
    torch.save(payload, path)
    fresh = objects()
    before = copy.deepcopy(fresh['actor'].state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(path, **fresh, identity={'assets': 'test'}, samplers=dict(music=Sampler()),
                        generators=dict(collector=torch.Generator()), **kwargs)
    assert all(torch.equal(before[name], value) for name, value in fresh['actor'].state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())


def test_weights_only_evaluation_does_not_restore_rng_or_optimizer(tmp_path):
    models, _, path = fixture_checkpoint(tmp_path)
    fresh = objects()
    rng = torch.get_rng_state().clone()
    optimizer = copy.deepcopy(fresh['actor_optimizer'].state_dict())
    provenance = load_weights_checkpoint(path, actor=fresh['actor'], critic=fresh['critic'],
                                         identity={'assets': 'test'})
    assert provenance['restore_mode'] == 'weights_only'
    assert not provenance['rng_restored'] and not provenance['optimizer_restored']
    assert torch.equal(rng, torch.get_rng_state())
    assert fresh['actor_optimizer'].param_groups[0]['lr'] == optimizer['param_groups'][0]['lr']
    assert all(torch.equal(models['actor'].state_dict()[name], value)
               for name, value in fresh['actor'].state_dict().items())


def test_capture_rank_only_touches_current_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    devices = []
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: devices.append(device) or torch.ones(4, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: pytest.fail('Must not initialize other ranks'))
    state = capture_rank_state(3, state=dict(buffer_size=0, pending_plan=False))
    assert devices == [3] and state['rng']['cuda_device'] == 3


def test_sparse_v2_publications_keep_recent_two_and_audit_rank_metadata(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        maintenance = LongRunMaintenance(manager, dict(run_control={}, storage=dict(
            checkpoint_keep_last=2, checkpoint_keep_every=1000, archive_completed_iterations=False)))
        models = objects()
        paths = {}
        for iteration in (300, 600, 900):
            shared = dict(iteration=iteration, policy_version=iteration, buffer_size=0, pending_plan=False)
            rank_states = [capture_rank_state(rank, state=shared, samplers=dict(music=Sampler(rank)))
                           for rank in range(2)]
            path = manager.run_dir / 'checkpoints' / f'iteration_{iteration:09d}.pt'
            save_checkpoint(path, **models, state=shared, identity={}, config={}, version=VERSION_V2,
                            rank_states=rank_states)
            manager.publish_checkpoint(iteration, path)
            paths[iteration] = path
        removed = maintenance.prune_checkpoints()
        assert len(removed) == 1 and not paths[300].exists()
        assert paths[600].exists() and paths[900].exists()
        assert manager.latest_checkpoint() == paths[900]
        retired = _read_json(manager.run_dir / 'checkpoints/retired' / f'{paths[300].stem}.json')
        assert retired['audit_state']['world_size'] == 2
        assert retired['audit_state']['rank_states'][1]['state']['iteration'] == 300
