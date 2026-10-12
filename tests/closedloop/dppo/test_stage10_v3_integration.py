"""Stage10 v3 评估与完整保存边界的集成回归。

此测试针对真实八卡入口在初始评估完成后发生的接口错误：选择报告含有 eligible
等诊断字段，而保存绑定函数仅接受 best_observed 与 best_saved 两个历史指针。
测试调用真实 v3 包装函数和实际选择/校验逻辑，只替换耗时的仿真评估及模型快照，
在 pytest 独立临时目录核对实际文件 SHA、模型身份、best 指针与报告发布一致性。
错误模型指纹必须拒绝，不能因修复关键字传递而削弱重要评估模型的身份检查。
本测试不运行物理环境，不把小型集成回归解释为 GPU 性能或强化学习质量验收。
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo import parallel_training, training_v3
from gem.closedloop.dppo.checkpoint_v3 import file_sha256
from gem.closedloop.dppo.periodic_monitor import choose_best


@pytest.mark.parametrize('matching_actor', [True, False])
def test_evaluation_binds_only_best_pointers_and_validates_actor(tmp_path, monkeypatch, matching_actor):
    session = tmp_path / 'sessions' / 'finite'
    artifact = session / 'evaluations' / 'initial.json'
    artifact.parent.mkdir(parents=True)
    checkpoint = tmp_path / 'checkpoints' / 'evaluated.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'independent finite checkpoint fixture')
    result = dict(iteration=0, plan_sha256='fixed-plan', actor_identity=dict(model_fingerprint='actor'),
        source_balanced_reward=2., physical_failure_count=0, mean_executed_seconds=10.,
        episode_metrics=[dict(task_id='one', dataset='AIST', seed=42, reward=2.,
                             executed_seconds=10., physical_failure=0)])
    result['selection'] = choose_best(result, result, iteration=0)
    artifact.write_text(json.dumps(result))
    report = dict(evaluation_artifacts=[dict(path=str(artifact.relative_to(tmp_path)),
                                            sha256=file_sha256(artifact))])
    publication = dict(path=str(checkpoint.relative_to(tmp_path)), sha256=file_sha256(checkpoint),
                       iteration=0, metadata=dict(actor_model_fingerprint='actor'))
    context = SimpleNamespace(distributed=None, actor=object(), output=tmp_path, session=session, state={})
    monkeypatch.setattr(parallel_training, '_evaluate_and_record', lambda *args: copy.deepcopy(result))
    monkeypatch.setattr(training_v3, '_snapshot', lambda *args, **kwargs: dict(publication=publication))
    monkeypatch.setattr(training_v3, 'root_call', lambda distributed, call: call())
    monkeypatch.setattr(training_v3, '_model_fingerprint', lambda actor: 'actor' if matching_actor else 'wrong')
    if not matching_actor:
        with pytest.raises(ValueError, match='Actor differs'):
            training_v3._evaluate(context, report, 'initial')
        assert not (tmp_path / 'best_checkpoint.json').exists()
        return
    saved = training_v3._evaluate(context, report, 'initial')
    assert saved['selection']['eligible'] is True
    assert saved['selection']['best_saved']['checkpoint']['sha256'] == file_sha256(checkpoint)
    assert set(context.state) == {'best_observed', 'best_saved'}
    assert not (tmp_path / 'eligible.json').exists()
    assert json.loads((tmp_path / 'best_checkpoint.json').read_text()) == publication
    assert json.loads(artifact.read_text()) == saved
    assert report['evaluation_artifacts'][0]['sha256'] == file_sha256(artifact)


def test_complete_rollout_anomaly_preserves_execution_and_old_statistics(tmp_path):
    from gem.closedloop.dppo.diagnostic_rollout_v3 import load_rollout
    from tests.closedloop.dppo.test_data_learning import transition
    row = transition()
    kernel = dict(schema='finite-test-kernel')
    row.metadata['sampler_trace'] = dict(kernel_config=kernel,
        old_means=torch.zeros(1, 2, 120, 30), old_stds=torch.ones(1, 2, 1, 1))
    row.metadata['execution_trace'] = [dict(control_index=i, physical_substeps=4) for i in range(2)]
    manifest = [dict(owner_rank=0, local_index=0, valid=True, has_free=True)]
    targets = dict(returns=torch.tensor([.3]), advantages=torch.tensor([1.]))
    context = SimpleNamespace(session=tmp_path, distributed=SimpleNamespace(rank=0), guard=None,
        policy=SimpleNamespace(kernel_config=kernel), identity=dict(test='anomaly'))
    buffer = SimpleNamespace(transitions=(row,))
    record = training_v3._preserve_rollout_anomaly(context, buffer, targets, manifest, 'failed_step')
    with torch.no_grad():
        row.chain.add_(2.)
        row.metadata['sampler_trace']['old_means'].add_(3.)
    restored = load_rollout(tmp_path / 'anomalies/failed_step/rank00/rollout')
    saved = restored['transitions'][0]
    assert record['full_execution_evidence'] is True
    assert record['real_control_steps'] == 2 and record['real_physics_steps'] == 8
    assert saved.metadata['execution_trace'] == row.metadata['execution_trace']
    assert torch.count_nonzero(saved.chain) == 0
    assert torch.count_nonzero(saved.metadata['sampler_trace']['old_means']) == 0
    assert saved.old_log_prob.dtype == torch.float64
    assert restored['global_manifest'] == manifest


@pytest.mark.parametrize('frozen', [True, False])
def test_critic_recomputes_trainable_encoder_after_each_update(frozen):
    from gem.closedloop.dppo.updater_v3 import critic_update_v3

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.freeze_encoders = frozen
            self.encoder = torch.nn.Linear(2, 3)
            self.head = torch.nn.Linear(3, 1)
            self.encoder.requires_grad_(not frozen)
            self.encoding_calls = 0

        def encode_frozen(self, inputs):
            assert self.freeze_encoders
            self.encoding_calls += 1
            return dict(feature=self.encoder(inputs['feature']).detach())

        def forward_encoded(self, inputs, remaining):
            return self.head(inputs['feature']).squeeze(-1)

        def forward(self, inputs, remaining):
            self.encoding_calls += 1
            return self.head(self.encoder(inputs['feature'])).squeeze(-1)

    torch.manual_seed(103)
    critic = Critic()
    before = critic.encoder.weight.detach().clone()
    rows = [SimpleNamespace(index=i) for i in range(4)]
    features = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    cache = SimpleNamespace(device=torch.device('cpu'),
        context=lambda selected: dict(feature=features[[row.index for row in selected]]),
        get=lambda key, selected: torch.ones(len(selected)))
    manifest = [dict(owner_rank=0, local_index=i, valid=True, has_free=True) for i in range(4)]
    distributed = SimpleNamespace(rank=0, world_size=1, all_gather_object=lambda value: [value],
        sum_tensor=lambda value: value, broadcast_object=lambda value: value, sum_gradients=lambda module: None)
    optimizer = torch.optim.SGD([p for p in critic.parameters() if p.requires_grad], lr=.1)
    report = critic_update_v3(critic, optimizer, rows, dict(returns=[1., 2., 3., 4.]), cache, manifest,
        distributed=distributed, generator=torch.Generator().manual_seed(3), epochs=2, batch=2)
    assert report['optimizer_steps'] == 4
    assert report['frozen_encoder_passes'] == int(frozen)
    assert critic.encoding_calls == (1 if frozen else 6)
    assert torch.equal(before, critic.encoder.weight) == frozen


@pytest.mark.parametrize('device_trace', [False, True])
def test_generation_delivers_conditions_on_the_sampling_device(device_trace):
    from gem.closedloop.dppo.vector_generation import generate_batch
    from tests.closedloop.dppo.test_data_learning import context
    device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    actor = torch.nn.Linear(1, 1).to(device)
    actor.endecoder = SimpleNamespace(codec=SimpleNamespace(apply_world_anchor=lambda value, anchor: value))

    def sample(conditions, **kwargs):
        count = len(conditions['decision_time'])
        zeros = lambda *shape: torch.zeros(count, *shape, device=device)
        return dict(conditions={key: value.clone() for key, value in conditions.items()},
            chain=zeros(21, 120, 30), qpos=zeros(120, 28), qpos30=zeros(120, 30), contact=zeros(120, 2),
            free_mask=conditions['future_valid'][..., None] & ~conditions['known_qpos30_mask'])

    builder = SimpleNamespace(build_many=lambda *args: ([context(), context()],
        [dict(world_anchor=[0., 0., 0.]), dict(world_anchor=[0., 0., 0.])]))
    policy = SimpleNamespace(actor=actor, sample_rollout=sample, last_sample_timing={})
    packets = [dict(builder=builder, snapshot={}, reservation=dict(request_id=str(i)), music=None,
                    seed=i+17, decision=0) for i in range(2)]
    outputs, _ = generate_batch(policy, packets, shared_storage=True, device_trace=device_trace)
    expected = device if device_trace else torch.device('cpu')
    for item in outputs:
        assert all(value.device == expected for value in item['context'].values())
        assert item['trace']['free_mask'].device == expected
        if device_trace:
            assert item['context'] is item['trace']['conditions']


def test_boundary_tail_is_part_of_complete_memory_execution_evidence():
    from gem.closedloop.dppo.vector_boundary import finish_vector_fragment
    from tests.closedloop.dppo.test_vector_fragment_boundary import fixture_env
    env, row, _, _ = fixture_env(source=6280, actual=25)
    prefix = [dict(tick=5088), dict(tick=5100)]
    row.metadata['execution_trace'] = prefix
    finish_vector_fragment(env, row, continue_episode=True)
    saved = row.metadata['execution_trace']
    assert len(saved) == row.executed_control_steps == 27
    assert saved[:2] == prefix
    assert [step['tick'] for step in saved] == list(range(5088, 5401, 12))
