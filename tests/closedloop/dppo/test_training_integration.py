"""Stage10 评估与完整保存边界的集成回归。

此测试针对真实八卡入口在初始评估完成后发生的接口错误：选择报告含有 eligible
等诊断字段，而保存绑定函数仅接受 best_observed 与 best_saved 两个历史指针。
测试调用真实 当前 包装函数和实际选择/校验逻辑，只替换耗时的仿真评估及模型快照，
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
from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions

from gem.closedloop.dppo import evaluation_runtime, training
from gem.closedloop.dppo.training_checkpoint import file_sha256
from gem.closedloop.dppo.periodic_monitor import choose_best


@pytest.mark.parametrize('used,expected', [(0, 1500000), (1499999, 1500000), (1500000, 3000000), (3199999, 4500000)])
def test_periodic_interaction_schedule_does_not_restart_after_resume(used, expected):
    assert training._next_interaction_boundary(used, 1500000) == expected


def test_actor_free_warmup_periods_create_distinct_actual_phase_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_runtime, 'begin_lease', lambda *args: object())
    c = SimpleNamespace(session=tmp_path / 'session', distributed=SimpleNamespace(rank=0), manager=None, budget=None)
    paths = []
    for rollout, controls in [(12, 1500000), (24, 3000000)]:
        state = dict(iteration=0, rollouts_completed=rollout, training_control_steps=controls)
        label = training._periodic_evaluation_label(state)
        path, _, _ = evaluation_runtime._new_phase(c, 'eval_' + label, [{}])
        paths.append(path)
    assert paths[0] != paths[1] and all(p.is_dir() for p in paths)


@pytest.mark.parametrize('changed', ['environment', 'termination', 'gmt_agent_config', 'isaac_contract'])
def test_physical_identity_binds_task_and_sidecar_assets(tmp_path, changed):
    paths = {}
    for name in ['gmt_agent_config', 'isaac_contract']:
        path = tmp_path / name
        path.write_text('original')
        paths[name] = str(path)
    c = dict(paths=paths, environment=dict(domain_randomization=False), termination=dict(yaw_error_rad=1.5))
    original = training._physical_configuration_identity(c)
    if changed in paths:
        from pathlib import Path
        Path(paths[changed]).write_text('changed')
    else:
        c[changed][next(iter(c[changed]))] = .25
    assert training._physical_configuration_identity(c) != original


def test_source_identity_detects_changed_physical_task_outside_adapter(tmp_path):
    genmo, gmt = tmp_path / 'genmo', tmp_path / 'gmt'
    for name in ('tools/train_closedloop_stage10.py', 'scripts/train_stage10_8gpu_server1.sh'):
        path = genmo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    task = gmt / 'source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/tracking_env_cfg.py'
    task.parent.mkdir(parents=True)
    task.write_text('decimation = 4')
    config = dict(paths=dict(genmo_repo=str(genmo), gmt_repo=str(gmt)))
    original = training._sources(config)
    task.write_text('decimation = 2')
    assert training._sources(config) != original


@pytest.mark.parametrize('changed', [None, 'linear', 'precision', 'attention', 'compiler', 'torch_version', 'assets'])
def test_deployment_delay_rejects_unmeasured_forward_contract(changed):
    from pathlib import Path
    import yaml
    root = Path(training.__file__).resolve().parents[3]
    profile = yaml.safe_load((root / 'configs/closedloop/stage2_asymmetric_server2.yaml').read_text())['timing']['deployment_profile']
    execution = copy.deepcopy(profile['numerical_execution'])
    # 旧标定文件的展示文字不影响真实算子；纯反向实现不属于单请求前向计时。
    execution.update(mode='compensated_tensorcore_fp32_storage_candidate', weight_gradient='new_backward')
    assets = copy.deepcopy(profile['measured_assets'])
    if changed == 'assets':
        assets['checkpoint'] = 'different'
    elif changed:
        execution[changed] = 'unmeasured'
    if changed:
        with pytest.raises(ValueError, match='calibration'):
            training._validate_calibrated_execution(profile, execution, assets)
    else:
        training._validate_calibrated_execution(profile, execution, assets)


def test_four_step_microbatches_keep_epoch_orders_old_chain_and_complete_logs(tmp_path, actor_factory):
    from gem.closedloop.dppo import updater
    from gem.closedloop.dppo.collectives import DistributedCollectives
    from gem.closedloop.dppo.gradient_merge import BCGradientController
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
    from gem.closedloop.dppo.dual_collector import split_trace
    from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
    from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
    import torch.distributed as dist

    class Anchor:
        def __init__(self): self.calls = 0
        def backward(self, actor, weight):
            self.calls += 1
            loss = sum(p.square().mean() for p in actor.parameters() if p.requires_grad)
            (loss * weight).backward()
            return dict(batch_size=2, loss=float(loss.detach()), samples=[[{'dataset_id':'fixture'}]*2])

    actor, batch = actor_factory(motion_frames=60); _activate_branches(actor)
    policy = FrozenPrefixPolicy(actor,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
    privileged = tracking_context(_conditions(batch))
    context = {k:privileged[k] for k in STAGE1_CONDITION_KEYS}
    trace = policy.sample_rollout(context,generator=torch.Generator().manual_seed(17))
    data = [SimpleNamespace(context={k:v[i:i+1].clone() for k,v in context.items()},identity={},
        chain=trace['chain'][i].clone(),old_log_prob=trace['old_log_probs'][i].clone(),free_mask=trace['free_mask'][i].clone(),
        metadata=dict(critic_context={k:v[i:i+1].clone() for k,v in privileged.items()},
            sampler_trace=split_trace(trace,i,2),remaining_music_seconds=5.)) for _ in range(4) for i in range(2)]
    old_chain = torch.stack([row.chain.clone() for row in data])
    base = copy.deepcopy(policy.actor.state_dict())
    targets = dict(advantages=torch.tensor([1., -.6] * 4))
    manifest = [dict(owner_rank=0, local_index=i, valid=True, has_free=True) for i in range(8)]
    reports, states = [], []
    dist.init_process_group('gloo', init_method=(tmp_path / 'one_rank').as_uri(), rank=0, world_size=1)
    cache = RolloutTensorCache(data, targets, 'cpu')
    try:
        for microbatch in (3, 8):
            policy.actor.load_state_dict(base)
            generator = torch.Generator().manual_seed(731)
            anchor = Anchor()
            report = updater.actor_update(policy, torch.optim.AdamW(policy.actor.parameters(), lr=1e-5),
                data, targets, cache, manifest, BCGradientController(initial=.01),
                distributed=DistributedCollectives(0, 1), anchor=anchor, generator=generator,
                sampled_steps=4, minibatch_chains=2, microbatch=microbatch, soft_kl=None)
            assert report['optimizer_steps'] == anchor.calls == 8
            assert report['internal_sample_visits'] == 8 * 4 * 2
            assert report['actual_global_bc_samples'] == 16
            assert report['bc_samples_by_source'] == {'fixture':16}
            for step in report['steps']:
                assert [r['step'] for r in step['per_denoising_step'] if r['visits']] == [15,16,17,18]
            reports.append(report)
            states.append(generator.get_state().clone())
    finally:
        cache.close()
        dist.destroy_process_group()
    assert reports[0]['epoch_orders'] == reports[1]['epoch_orders']
    assert torch.equal(states[0], states[1])
    assert [s['denoising_sampling']['seed'] for s in reports[0]['steps']] == [
        s['denoising_sampling']['seed'] for s in reports[1]['steps']]
    assert torch.equal(torch.stack([row.chain for row in data]), old_chain)


@pytest.mark.parametrize('fault', [None, 'assets', 'kernel', 'schema', 'hard_kl', 'soft_kl', 'partial', 'nonfinite', 'binding'])
def test_evaluation_actor_load_requires_complete_bound_update(tmp_path, fault):
    from gem.closedloop.dppo.training_checkpoint import load_evaluation_actor
    actor = torch.nn.Linear(3, 2)
    original = copy.deepcopy(actor.state_dict())
    candidate = {name: value + .25 for name, value in original.items()}
    kernel, assets = {'steps': 20}, {'checkpoint': 'source-sha'}
    binding=dict(reward={'version':'new_reward'},source_files={'code':'sha'},timing={'budget':.32})
    payload = dict(schema='genmo.stage2.asymmetric_actor_evaluation.v1', evaluation_only=True,
        complete_update_and_soft_kl_passed=True, actor=candidate, kernel=kernel,
        source_identity=dict(assets=assets,**binding), hard_kl={'passed': True}, source_case_sha256='a' * 64,
        update={'optimizer_steps': 8, 'planned_optimizer_steps': 8})
    if fault == 'assets': payload['source_identity'] = {'assets': {'checkpoint': 'other'}}
    if fault == 'kernel': payload['kernel'] = {'steps': 8}
    if fault == 'schema': payload['schema'] = 'old-training-checkpoint'
    if fault == 'hard_kl': payload['hard_kl']['passed'] = False
    if fault == 'soft_kl': payload['complete_update_and_soft_kl_passed'] = False
    if fault == 'partial': payload['update']['optimizer_steps'] = 1
    if fault == 'nonfinite': candidate['weight'][0, 0] = float('nan')
    if fault == 'binding': payload['source_identity']['reward']={'version':'other_reward'}
    path = tmp_path / 'evaluation_actor.pt'
    torch.save(payload, path)
    if fault:
        with pytest.raises(ValueError):
            load_evaluation_actor(path, actor=actor, kernel=kernel, assets=assets,execution_binding=binding)
        assert all(torch.equal(value, actor.state_dict()[name]) for name, value in original.items())
    else:
        result = load_evaluation_actor(path, actor=actor, kernel=kernel, assets=assets,execution_binding=binding)
        assert result['sha256'] == file_sha256(path)
        assert all(torch.equal(value, actor.state_dict()[name]) for name, value in candidate.items())


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
    report = dict(evaluation_artifacts=[],evaluations=[])
    publication = dict(path=str(checkpoint.relative_to(tmp_path)), sha256=file_sha256(checkpoint),
                       iteration=0, metadata=dict(actor_model_fingerprint='actor' if matching_actor else 'wrong',reason='evaluated_initial'))
    ready=[]
    context = SimpleNamespace(distributed=SimpleNamespace(rank=0), actor=object(), bc=object(), output=tmp_path,
        session=session,state={'iteration':0},async_writer=SimpleNamespace(poll=lambda **kwargs:ready.pop(0) if ready else None))
    monkeypatch.setattr(evaluation_runtime, '_evaluate', lambda *args,**kwargs: copy.deepcopy(result))
    monkeypatch.setattr(training, '_snapshot', lambda *args, **kwargs: ready.append(publication))
    monkeypatch.setattr(training, 'root_call', lambda distributed, call: call())
    saved = training._evaluate(context, report, 'initial')
    assert saved['selection']['best_saved'] is None
    assert not report['evaluation_artifacts']
    if not matching_actor:
        with pytest.raises(ValueError, match='Actor fingerprint differs'):
            training._poll_checkpoint(context,wait=True)
        assert not (tmp_path / 'best_checkpoint.json').exists()
        return
    training._poll_checkpoint(context,wait=True)
    assert saved['selection']['eligible'] is True
    assert saved['selection']['best_saved']['checkpoint']['sha256'] == file_sha256(checkpoint)
    assert {'best_observed', 'best_saved'}<=set(context.state)
    assert not (tmp_path / 'eligible.json').exists()
    assert json.loads((tmp_path / 'best_checkpoint.json').read_text()) == publication
    assert json.loads(artifact.read_text()) == saved
    assert report['evaluation_artifacts'][0]['sha256'] == file_sha256(artifact)


@pytest.mark.parametrize('fault',[None,'binding','fingerprint'])
def test_complete_checkpoint_evaluation_checks_execution_before_loading(tmp_path,fault):
    from gem.closedloop.dppo.training_checkpoint import load_evaluation_actor,SCHEMA
    from gem.closedloop.dppo.evaluation import _model_fingerprint
    actor=torch.nn.Linear(3,2);original=copy.deepcopy(actor.state_dict())
    candidate=copy.deepcopy(actor)
    with torch.no_grad():candidate.weight.add_(.25)
    binding={'reward':{'music_weight':0.},'source_files':{'code':'sha'}}
    identity=dict(binding,assets={'checkpoint':'sha'},kernel={'frozen_prefix':16})
    payload=dict(schema=SCHEMA,identity=identity,actor=candidate.state_dict(),
        actor_fingerprint=_model_fingerprint(candidate),state={'iteration':100,'policy_version':100})
    if fault=='binding':identity['source_files']={'code':'changed'}
    if fault=='fingerprint':payload['actor_fingerprint']='wrong'
    path=tmp_path/'checkpoint.pt';torch.save(payload,path)
    def load():return load_evaluation_actor(path,actor=actor,kernel={'frozen_prefix':16},
        assets={'checkpoint':'sha'},execution_binding=binding)
    if fault:
        with pytest.raises(ValueError):load()
        assert all(torch.equal(v,actor.state_dict()[k]) for k,v in original.items())
    else:
        assert load()['source_iteration']==100
        assert torch.equal(actor.weight,candidate.weight)


def test_complete_rollout_anomaly_preserves_execution_and_old_statistics(tmp_path):
    from gem.closedloop.dppo.diagnostic_rollout import load_rollout
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
    record = training._preserve_rollout_anomaly(context, buffer, targets, manifest, 'failed_step')
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
    from gem.closedloop.dppo.updater import critic_update

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.freeze_encoders = frozen
            self.cache_frozen_conditions = frozen
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
        critic_context=lambda selected: dict(feature=features[[row.index for row in selected]]),
        get=lambda key, selected: torch.ones(len(selected)))
    manifest = [dict(owner_rank=0, local_index=i, valid=True, has_free=True) for i in range(4)]
    distributed = SimpleNamespace(rank=0, world_size=1, all_gather_object=lambda value: [value],
        sum_tensor=lambda value: value, broadcast_object=lambda value: value, sum_gradients=lambda module: None)
    optimizer = torch.optim.SGD([p for p in critic.parameters() if p.requires_grad], lr=.1)
    report = critic_update(critic, optimizer, rows, dict(returns=[1., 2., 3., 4.]), cache, manifest,
        distributed=distributed, generator=torch.Generator().manual_seed(3), epochs=2, batch=2)
    assert report['optimizer_steps'] == 4
    assert report['frozen_encoder_passes'] == int(frozen)
    assert critic.encoding_calls == (1 if frozen else 6)
    assert torch.equal(before, critic.encoder.weight) == frozen


@pytest.mark.parametrize('device_trace', [False, True])
def test_generation_delivers_conditions_on_the_sampling_device(device_trace):
    from gem.closedloop.dppo.vector_generation import generate_batch
    from tests.closedloop.dppo.test_data_learning import context
    from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER
    import numpy as np
    device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    actor = torch.nn.Linear(1, 1).to(device)
    actor.endecoder = SimpleNamespace(codec=SimpleNamespace(apply_world_anchor=lambda value, anchor: value),
        kinematics=SimpleNamespace(joint_order=GMT_EXPECTED_JOINT_ORDER))

    def sample(conditions, **kwargs):
        count = len(conditions['decision_time'])
        zeros = lambda *shape: torch.zeros(count, *shape, device=device)
        return dict(conditions={key: value.clone() for key, value in conditions.items()},
            chain=zeros(21, 120, 30), qpos=zeros(120, 28), qpos30=zeros(120, 30), contact=zeros(120, 2),
            free_mask=conditions['future_valid'][..., None] & ~conditions['known_qpos30_mask'])

    builder = SimpleNamespace(build_many=lambda *args: ([context(), context()],
        [dict(world_anchor=[0., 0., 0.]), dict(world_anchor=[0., 0., 0.])]))
    policy = SimpleNamespace(actor=actor, sample_rollout=sample, last_sample_timing={})
    snapshot=dict(joint_names=GMT_EXPECTED_JOINT_ORDER,joint_hard_limits_rad=np.tile([-2.,2.],(21,1)),
        joint_soft_limits_rad=np.tile([-1.8,1.8],(21,1)),tracking_history_schema='gmt.causal_tracking_history90.v1',
        tracking_history_values=np.zeros((50,90)),tracking_history_valid=np.ones(50,dtype=bool),
        tracking_history_ticks=np.arange(12,601,12),tracking_reference_ticks=np.arange(12,601,12))
    packets = [dict(builder=builder, snapshot=snapshot, reservation=dict(request_id=str(i)), music=None,
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


def test_repeated_checkpoint_boundary_publishes_distinct_immutable_files(tmp_path):
    from gem.closedloop.dppo.training_checkpoint import AsyncCheckpointWriter, SCHEMA
    payload = dict(schema=SCHEMA, actor={'weight': torch.ones(3)}, actor_fingerprint='finite',
        state=dict(iteration=1, rollouts_completed=2, actor_updates=3, critic_updates=4))
    writer = AsyncCheckpointWriter(tmp_path)
    published = []
    try:
        for _ in range(2):
            writer.submit(copy.deepcopy(payload), reason='controlled_end')
            published.append(writer.poll(wait=True))
    finally:
        writer.close()
    assert published[0]['path'] != published[1]['path']
    for item in published:
        assert file_sha256(tmp_path / item['path']) == item['sha256']
    assert json.loads((tmp_path / 'latest.json').read_text())['path'] == published[-1]['path']


def test_iteration_milestones_survive_periodic_retention_without_rejected_duplicates(tmp_path):
    from gem.closedloop.dppo.run_control import iteration_checkpoint_reason
    from gem.closedloop.dppo.training_checkpoint import AsyncCheckpointWriter,SCHEMA
    storage={'checkpoint_interval_iterations':100}
    assert iteration_checkpoint_reason(99,storage) is None
    assert iteration_checkpoint_reason(0,storage) is None
    assert iteration_checkpoint_reason(100,storage,rejected=True) is None
    assert iteration_checkpoint_reason(100,storage,is_warmup=True) is None
    writer=AsyncCheckpointWriter(tmp_path,keep_last=2)
    saved=[]
    try:
        for iteration,reason in [(100,iteration_checkpoint_reason(100,storage)),
                (101,'periodic'),(102,'periodic'),(200,iteration_checkpoint_reason(200,storage)),(201,'periodic')]:
            payload=dict(schema=SCHEMA,actor={'weight':torch.tensor([float(iteration)])},actor_fingerprint='test',
                state=dict(iteration=iteration,rollouts_completed=iteration,actor_updates=iteration,critic_updates=iteration))
            writer.submit(payload,reason=reason)
            saved.append(writer.poll(wait=True))
    finally:writer.close()
    assert (tmp_path/saved[0]['path']).is_file() and (tmp_path/saved[3]['path']).is_file()
    assert not (tmp_path/saved[1]['path']).exists()
    assert len(list((tmp_path/'checkpoints').glob('*.pt')))==4
    assert torch.load(tmp_path/saved[0]['path'],weights_only=False)['state']['iteration']==100
    assert iteration_checkpoint_reason(2000,storage)=='iteration_002000'


def test_rollback_restores_nonempty_adam_and_rng_exactly():
    from gem.closedloop.dppo.gradient_merge import BCGradientController
    from gem.closedloop.dppo.rollback_audit import exact_state_digest

    class Anchor:
        def __init__(self): self.value = {'cursor': 4}
        def state_dict(self): return copy.deepcopy(self.value)
        def load_state_dict(self, value): self.value = copy.deepcopy(value)

    actor, critic = torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    actor_opt, critic_opt = (torch.optim.AdamW(module.parameters(), lr=.01) for module in (actor, critic))
    c = SimpleNamespace(actor=actor, critic=critic, actor_optimizer=actor_opt, critic_optimizer=critic_opt,
        bc=Anchor(), controller=BCGradientController(), generators={'actor': torch.Generator().manual_seed(314)},
        state=dict(actor_updates=2, critic_updates=3, training_control_steps=101))
    for model, opt in ((actor, actor_opt), (critic, critic_opt)):
        model(torch.ones(2, 3)).square().sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    saved = training._rollback_snapshot(c)
    for model, opt in ((actor, actor_opt), (critic, critic_opt)):
        model(torch.full((2, 3), 2.)).sum().backward(); opt.step()
    torch.rand(3); torch.rand(3, generator=c.generators['actor'])
    c.bc.value['cursor'] += 1; c.controller.update(3., 4.)
    c.state.update(actor_updates=8, critic_updates=12, training_control_steps=109)
    audit = training._restore_rollback(c, saved)
    assert audit['identical']
    assert exact_state_digest(saved) == exact_state_digest(training._rollback_snapshot(c))
    assert c.state['training_control_steps'] == 109
    assert all(p.grad is None for model in (actor, critic) for p in model.parameters())


def test_nan_p95_is_rejected_without_relaxing_kl_limits():
    from gem.closedloop.dppo.updater import check_kl
    report = dict(mean_coordinate_kl=0., mean_joint_kl=0., maximum_step_mean_coordinate_kl=0.,
                  coordinate_kl=dict(p95=float('nan')))
    outcome = check_kl(report, dict(p95_coordinate=.01))
    assert not outcome['passed'] and 'p95_coordinate' in outcome['failures']
