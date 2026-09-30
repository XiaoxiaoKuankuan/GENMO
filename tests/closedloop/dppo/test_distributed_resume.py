"""验证八卡运行层的恢复同步不会把非主进程优化器重置成首次更新。

测试使用两个真实Gloo进程和小型带buffer网络，先保存已有AdamW动量与步数的
主进程checkpoint，再让另一个rank从不同权重、不同学习率和空优化器启动。
生产DistributedLearner初始化必须同步完整模型、buffer、AdamW参数组及状态，
随后一次全局梯度更新应与不中断的单进程基准一致。另覆盖全新空状态初始化，
以及初始化不得消耗主进程已恢复的随机状态。checkpoint和交换文件仅位于pytest
临时目录，不读取正式音乐数据或模型，不使用GPU，也不启动任何长期训练。
"""
from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gem.closedloop.dppo.distributed_runtime import DistributedCollectives, DistributedLearner


class BufferedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.register_buffer('normalizer', torch.tensor([.3, .7]), persistent=False)

    def forward(self, inputs):
        return self.layer(inputs / self.normalizer).squeeze(-1)


def _components(seed, *, wrong_worker_configuration=False):
    torch.manual_seed(seed)
    actor, critic = BufferedModel(), BufferedModel()
    options = dict(lr=.4, betas=(.1, .2), weight_decay=.7) if wrong_worker_configuration else dict(
        lr=.003, betas=(.85, .97), weight_decay=.04)
    return actor, critic, torch.optim.AdamW(actor.parameters(), **options), torch.optim.AdamW(critic.parameters(), **options)


def _update(model, optimizer, offset, distributed=None):
    inputs = torch.tensor([[.2, .3], [-.8, .4], [.6, -.9]])
    target = torch.tensor([.4, -.3, .8]) + offset
    indices = list(range(3)) if distributed is None else list(range(distributed.rank, 3, distributed.world_size))
    optimizer.zero_grad(set_to_none=True)
    ((model(inputs[indices]) - target[indices]).square().sum() / len(inputs)).backward()
    if distributed is not None:
        distributed.sum_gradients(model)
    optimizer.step()


def _snapshot(actor, critic, actor_optimizer, critic_optimizer):
    return copy.deepcopy(dict(actor=actor.state_dict(), critic=critic.state_dict(),
        actor_buffer=actor.normalizer, critic_buffer=critic.normalizer,
        actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict()))


def _worker(rank, directory):
    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(directory/'rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        collective = DistributedCollectives(rank, 2, device='cpu')
        results = {}
        for name in ('new', 'resumed'):
            actor, critic, actor_optimizer, critic_optimizer = _components(100 + rank, wrong_worker_configuration=True)
            if rank == 0:
                saved = torch.load(directory/f'{name}.pt', map_location='cpu', weights_only=False)
                actor.load_state_dict(saved['actor'])
                critic.load_state_dict(saved['critic'])
                actor.normalizer.copy_(saved['actor_buffer'])
                critic.normalizer.copy_(saved['critic_buffer'])
                actor_optimizer.load_state_dict(saved['actor_optimizer'])
                critic_optimizer.load_state_dict(saved['critic_optimizer'])
            else:
                actor.normalizer.fill_(5.)
                critic.normalizer.fill_(8.)
            learner = DistributedLearner(collective, directory/name)
            learner.actor, learner.critic = actor, critic
            learner.actor_optimizer, learner.critic_optimizer = actor_optimizer, critic_optimizer
            random_state = torch.get_rng_state().clone()
            learner._synchronize_initial()
            assert torch.equal(random_state, torch.get_rng_state())
            initial = _snapshot(actor, critic, actor_optimizer, critic_optimizer)
            _update(actor, actor_optimizer, .2, collective)
            _update(critic, critic_optimizer, -.1, collective)
            evidence = learner._finish('resume_next_update', {})
            results[name] = dict(initial=initial, after=_snapshot(actor, critic, actor_optimizer, critic_optimizer),
                                 initialization=learner.evidence[0], update=evidence)
        torch.save(results, directory/f'rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            _assert_equal(first, second)
    else:
        assert actual == expected


@pytest.fixture(scope='module')
def resumed_replicas(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('PyTorch Gloo is unavailable')
    directory = tmp_path_factory.mktemp('distributed_resume')
    expected = {}
    for name, previous_steps in (('new', 0), ('resumed', 4)):
        actor, critic, actor_optimizer, critic_optimizer = _components(731)
        for step in range(previous_steps):
            _update(actor, actor_optimizer, step*.1)
            _update(critic, critic_optimizer, -step*.2)
        initial = _snapshot(actor, critic, actor_optimizer, critic_optimizer)
        torch.save(initial, directory/f'{name}.pt')
        _update(actor, actor_optimizer, .2)
        _update(critic, critic_optimizer, -.1)
        expected[name] = dict(initial=initial, after=_snapshot(actor, critic, actor_optimizer, critic_optimizer))
    mp.spawn(_worker, args=(str(directory),), nprocs=2, join=True)
    ranks = [torch.load(directory/f'rank_{rank}.pt', weights_only=False) for rank in range(2)]
    assert not list(directory.glob('*/distributed_exchange/*.pt'))
    return expected, ranks


@pytest.mark.parametrize('name', ('new', 'resumed'))
def test_optimizer_sync_and_next_global_update_match_uninterrupted(resumed_replicas, name):
    expected, ranks = resumed_replicas
    for result in ranks:
        _assert_equal(result[name]['initial'], expected[name]['initial'])
        _assert_equal(result[name]['after'], expected[name]['after'])
        assert result[name]['initialization']['replicas_identical']
        assert result[name]['initialization']['optimizers']['replicas_identical']
        assert result[name]['update']['distributed']['replicas_identical']
        if name == 'resumed':
            for optimizer in ('actor_optimizer', 'critic_optimizer'):
                assert {float(state['step']) for state in result[name]['after'][optimizer]['state'].values()} == {5.}
