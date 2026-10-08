"""八采集器主流程的配置、保存边界和整轮回滚集成检查。

使用真实 PyTorch 小模型、Adam 状态与持久预算，替换昂贵的物理采集和网络前向，
注入 Critic 后异常、Actor 后异常及最终 KL 超限。检查模型、优化器和随机数确实
退回整轮起点，而已经持久化的优化尝试数保持不变；不把 helper 的单元通过当作
真实八卡物理执行验收。所有文件仅位于 pytest tmp_path，不读取正式训练模型。
普通非周期轮次另外经过真实小型 PPO 更新器，检查每个 step 的 PPO／BC 分解梯度
都由根调用启用，而不是只在首轮或每一百轮完整概率诊断时才输出。
哨兵覆盖无效首条、全前缀和某个 rank 无可用链的边界；KL 配置回归使用此前真实
被拒绝的数值，确认提高硬上限后允许该候选，同时仍拒绝真正超过新边界的候选。
另外运行真实 CPU 评估汇总，验证终态评估与最终计时的报告 SHA 一并登记；完整
外层循环时间与核心训练时间分别落盘，周期验证可复用，失败验证不得登记为成功。
双进程 Gloo 故障注入检查评估恢复失败的同步边界；归档排空或计时失败仍必须释放
运行管理器锁，避免其他进程误入后续 collective 或留下不可恢复的活动会话锁。
恢复校准另行覆盖单卡延迟超限、状态恢复与证据落盘失败：失败不进入租约退款，
成功保留旧延迟合同与任务游标，同时保留校准已经消耗的新请求编号。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo import parallel_training as training
from gem.closedloop.dppo.execution_profile import select_profile
from gem.closedloop.dppo.run_management import TrainingBudget
from gem.closedloop.dppo.parallel_support import begin_lease, check_kl_limits
from tools.train_closedloop_stage10 import configuration


class Solo:
    rank, world_size, device = 0, 1, torch.device('cpu')

    def broadcast_object(self, value):
        return value

    def all_gather_object(self, value):
        return [value]


def test_formal_configuration_and_outer_save_boundaries():
    root = Path(__file__).resolve().parents[3]
    config = configuration(root/'configs/closedloop/stage10_8gpu_server1_v2.yaml')
    settings = config['stage9']
    assert settings['actor_lr'] == 5e-9
    assert settings['kl_stop_joint'] == .03
    assert settings['kl_soft_stop_joint'] == .015
    assert 'actor_lr_candidates' not in settings
    assert settings['rollout_upper_steps'] == 160
    assert settings['ppo_epochs'] == 2
    assert settings['actor_minibatch_internal_transitions'] == 1600
    assert settings['bc_batch'] == 2
    assert [i for i in range(1, 601) if training.checkpoint_due(i)] == [300, 600]
    assert training.checkpoint_due(1, normal_end=True)
    assert not training.checkpoint_due(4)


@pytest.mark.parametrize('mean_kl, accepted', [(.0241416694575873, True), (.03, True), (.030001, False)])
def test_formal_kl_limit_accepts_previous_real_candidate_and_enforces_new_boundary(mean_kl, accepted):
    root = Path(__file__).resolve().parents[3]
    settings = configuration(root/'configs/closedloop/stage10_8gpu_server1_v2.yaml')['stage9']
    # 使用旧验收的末端集中比例；逐步 KL 不是平均内部 KL 的同名独立门槛。
    tail_share = .9968549524249632
    steps = settings['denoising_steps']
    total = mean_kl * steps
    per_step = [total * (1-tail_share)/(steps-2)] * (steps-2) + [total * tail_share/2] * 2
    report = dict(mean_joint_kl=sum(per_step)/steps, mean_chain_joint_kl=sum(per_step),
        per_denoising_step=[dict(mean_joint_kl=value) for value in per_step])
    # 边界值以配置的浮点数直接传入，避免测试构造求和造成额外一 ULP。
    report['mean_joint_kl'] = mean_kl
    assert settings['kl_stop_joint'] * steps == pytest.approx(.6)
    if accepted:
        check_kl_limits(report, settings)
    else:
        with pytest.raises(RuntimeError, match='Final whole-rollout KL rejected'):
            check_kl_limits(report, settings)


def test_sentinels_select_eligible_chains_per_rank_and_report_empty_ranks():
    manifest = [dict(owner_rank=rank, local_index=index, valid=index != 0, has_free=index != 1)
                for rank in range(8) for index in range(4)]
    # rank 3 的两条剩余链都是全前缀；该 rank 仍在全局清单中，不能伪造其概率覆盖。
    for row in manifest:
        if row['owner_rank'] == 3:
            row['has_free'] = False
    selected, coverage = training._probability_sentinels(manifest, world_size=8, chains_per_rank=1)
    assert [(manifest[i]['owner_rank'], manifest[i]['local_index']) for i in selected] == [
        (rank, 2) for rank in range(8) if rank != 3]
    assert coverage['checked_upper_transitions'] == 7
    assert coverage['ranks_without_eligible_chains'] == [3]
    assert coverage['rank_coverage'][3] == dict(rank=3, eligible_upper_transitions=0, checked_upper_transitions=0)
    selected, coverage = training._probability_sentinels(manifest, world_size=8, chains_per_rank=3)
    assert len(selected) == 14
    assert coverage['rank_coverage'][0]['checked_upper_transitions'] == 2
    for row in manifest:
        row['has_free'] = False
    with pytest.raises(ValueError, match='No free valid upper transition'):
        training._probability_sentinels(manifest, world_size=8, chains_per_rank=1)


def test_profile_selects_common_pass_without_changing_tolerances():
    ranks = [[dict(microbatch=m, cfg_batch=c, passed=m <= cap and not c)
              for m in (4, 2, 1) for c in (True, False)] for cap in (4, 2)]
    assert select_profile(ranks)['microbatch'] == 2
    assert select_profile(ranks)['cfg_batch'] is False
    with pytest.raises(RuntimeError):
        select_profile(ranks, required=dict(microbatch=4, cfg_batch=False))


def test_global_collection_credit_rejected_before_any_rank_reservation(tmp_path):
    budget = TrainingBudget(tmp_path/'global.json', dict(accepted_iterations=2, optimizer_attempts=8,
        generations=10, control_steps=100, physics_steps=400))
    credits = [dict(generations=6, control_steps=10, physics_steps=40)]*2
    before = budget.state_dict()
    with pytest.raises(RuntimeError, match='complete parallel collection lease'):
        begin_lease(Solo(), None, budget, tmp_path, 'collect', credits)
    assert budget.state_dict() == before
    assert not (tmp_path/'resource_lease.json').exists()


def test_calibration_measures_selected_cfg_after_numeric_fallback(tmp_path, monkeypatch):
    from tools import train_closedloop_stage10 as entry
    events = []
    env = SimpleNamespace(latency_budget_s=.26, config={'stage9':{'seed':42}},
        reset_task=lambda *a, **k: events.append('reset'),
        preview_context=lambda: ({'value':torch.zeros(1)}, None))
    context = SimpleNamespace(env=env, generators={}, base_config={'timing':{'calibration_warmup':1,'calibration_samples':2}},
        settings={'episode_seconds':10.,'denoising_microbatch':4}, distributed=Solo(), stage={'seed':42},
        state={}, policy=SimpleNamespace(cfg_batch=True), catalog=SimpleNamespace(
            samples={'train':{'Mine':[{}]}}, load_music=lambda _: None), budget=None)
    monkeypatch.setattr(training, '_new_phase', lambda *args: (tmp_path, 'profile', SimpleNamespace(reserve=lambda *a, **k:None)))
    monkeypatch.setattr(training, 'probe_profiles', lambda *a, **k: [dict(microbatch=1,cfg_batch=False,passed=True)])
    monkeypatch.setattr(training, 'finish_lease', lambda *args: {'used':{}})
    def calibrate(*args):
        assert context.policy.cfg_batch is False
        events.append('calibrate_separate_cfg')
        env.latency_budget_s=.4
        return dict(durations=[.3,.31],latency_budget_s=.4)
    monkeypatch.setattr(entry, 'calibrate', calibrate)
    training._calibrate_and_profile(context)
    assert events == ['reset','calibrate_separate_cfg']
    assert env.latency_budget_s == .4
    assert context.state['execution_profile']['microbatch'] == 1


def _resume_calibration_worker(rank, directory):
    from datetime import timedelta
    import torch.distributed as dist
    from tools import train_closedloop_stage10 as entry
    from gem.closedloop.dppo.distributed_runtime import DistributedCollectives

    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory/'resume_rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    saved = dict(new_phase=training._new_phase, probe=training.probe_profiles, settle=training.finish_lease,
        atomic=training.atomic_json, rng=training.restore_local_rng, calibrate=entry.calibrate,
        execution=entry.restore_execution_state)
    outcomes = {}
    try:
        collective = DistributedCollectives(rank, 2, device='cpu')
        for phase in ('latency', 'execution', 'rng', 'profile_io', 'success'):
            settled = []
            env = SimpleNamespace(latency_budget_s=.4, decision=4, attempt=11, episode_count=2,
                policy_version=3, iteration=2, config={'stage9':{'seed':42}},
                reset_task=lambda *a, **k: None, preview_context=lambda: ({'value':torch.zeros(1)}, None))
            context = SimpleNamespace(env=env, generators={}, distributed=collective,
                base_config={'timing':{'calibration_warmup':1, 'calibration_samples':2}},
                settings={'episode_seconds':10., 'denoising_microbatch':4}, stage={'seed':42},
                state={'policy_version':3, 'iteration':2, 'execution_profile':{'microbatch':1, 'cfg_batch':False}},
                policy=SimpleNamespace(cfg_batch=True), catalog=SimpleNamespace(
                    samples={'train':{'Mine':[{}]}}, load_music=lambda _: None), budget=None)
            def new_phase(*args):
                path = directory/phase/f'rank{rank}'
                path.mkdir(parents=True)
                return path, phase, SimpleNamespace(reserve=lambda *a, **k: None)
            def probe(*args, **kwargs):
                torch.rand(3)
                return [dict(microbatch=1, cfg_batch=False, passed=True)]
            def calibrate(*args):
                torch.rand(3)
                env.attempt, env.decision, env.episode_count, env.latency_budget_s = 17, 8, 5, .7
                return dict(durations=[.5 if rank == 1 and phase == 'latency' else .3], latency_budget_s=.7)
            def execution(*args, **kwargs):
                if rank == 1 and phase == 'execution':
                    raise RuntimeError('injected_execution_restore')
                return saved['execution'](*args, **kwargs)
            def rng_restore(*args, **kwargs):
                saved['rng'](*args, **kwargs)
                if rank == 1 and phase == 'rng':
                    raise RuntimeError('injected_calibration_rng_restore')
            def atomic(*args, **kwargs):
                if rank == 1 and phase == 'profile_io':
                    raise OSError('injected_profile_write')
                return saved['atomic'](*args, **kwargs)
            training._new_phase, training.probe_profiles = new_phase, probe
            training.finish_lease = lambda *args: (settled.append(phase), {'used':{}})[1]
            training.restore_local_rng, training.atomic_json = rng_restore, atomic
            entry.calibrate, entry.restore_execution_state = calibrate, execution
            original_rng = torch.get_rng_state().clone()
            try:
                training._calibrate_and_profile(context, restored=True)
            except RuntimeError as error:
                if phase == 'success':
                    raise
                outcomes[phase] = str(error)
                assert not settled
            else:
                assert phase == 'success', 'Rank-local resume failure was not synchronized'
                assert settled == ['success']
                assert (env.latency_budget_s, env.decision, env.episode_count, env.attempt) == (.4, 4, 2, 17)
                assert env.policy_version == 3 and env.iteration == 2
                assert context.state['budget'] == {'used':{}}
                outcomes[phase] = 'restored_without_reusing_spent_attempts'
            assert torch.equal(torch.get_rng_state(), original_rng)
            collective.barrier()
        torch.save(outcomes, directory/f'resume_rank{rank}.pt')
    finally:
        training._new_phase, training.probe_profiles, training.finish_lease = saved['new_phase'], saved['probe'], saved['settle']
        training.restore_local_rng, training.atomic_json = saved['rng'], saved['atomic']
        entry.calibrate, entry.restore_execution_state = saved['calibrate'], saved['execution']
        dist.destroy_process_group()


def test_resume_calibration_failures_are_shared_before_lease_settlement(tmp_path):
    import torch.distributed as dist
    from tests.closedloop.dppo.test_updater_v2 import _spawn_bounded
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo is unavailable')
    _spawn_bounded(_resume_calibration_worker, tmp_path)
    outcomes = [torch.load(tmp_path/f'resume_rank{rank}.pt', weights_only=False) for rank in range(2)]
    assert outcomes[0] == outcomes[1]
    assert 'rank 1: RuntimeError: Restored latency contract is insufficient' in outcomes[0]['latency']
    for phase, message in [('execution', 'injected_execution_restore'), ('rng', 'injected_calibration_rng_restore'),
                           ('profile_io', 'injected_profile_write')]:
        assert message in outcomes[0][phase]
    assert outcomes[0]['success'] == 'restored_without_reusing_spent_attempts'


def test_successful_final_evaluation_is_recorded_with_final_report_sha(tmp_path, monkeypatch):
    from tests.closedloop.dppo.test_parallel_periodic_evaluation import context as evaluation_context
    from gem.closedloop.dppo.budget import atomic_json
    from gem.robots.bumi.kinematics import sha256_file

    context, _ = evaluation_context(tmp_path, monkeypatch)
    context.initial_iteration = 0
    report = dict(evaluations=[], evaluation_artifacts=[])
    baseline = training._evaluate_and_record(context, report, 'initial')
    atomic_json(context.output/'evaluation_baseline.json', baseline)
    context.state['iteration'] = 1
    saves = []
    monkeypatch.setattr(training, '_write_checkpoint', lambda c, **kwargs: saves.append(kwargs['reason']))
    training._finish_training(context, report)
    assert saves == ['controlled_end']
    assert report['evaluations'] == ['initial', 'final_000001']
    for descriptor, expected_iteration in zip(report['evaluation_artifacts'], [0, 1]):
        source = context.output/descriptor['path']
        assert descriptor['sha256'] == sha256_file(source)
        saved = json.loads(source.read_text())
        assert saved['iteration'] == expected_iteration == descriptor['iteration']
        assert saved['wall_seconds'] >= 0
        assert saved['wall_time_scope'].startswith('evaluation_including_state_restore')
    assert context.last_evaluation['wall_seconds'] >= 0


def test_failed_final_evaluation_is_not_registered_and_periodic_final_is_reused(tmp_path, monkeypatch):
    context = SimpleNamespace(state={'iteration':100}, initial_iteration=0)
    report = dict(evaluations=[100], evaluation_artifacts=[{'label':'000100'}])
    monkeypatch.setattr(training, '_write_checkpoint', lambda *a, **k: None)
    calls = []
    def fail(*args):
        calls.append(args[-1])
        raise RuntimeError('injected evaluation failure')
    monkeypatch.setattr(training, '_evaluate', fail)
    training._finish_training(context, report)
    assert not calls
    context.state['iteration'] = 101
    before = copy.deepcopy(report)
    with pytest.raises(RuntimeError, match='injected evaluation failure'):
        training._finish_training(context, report)
    assert calls == ['final_000101']
    assert report == before


def test_complete_iteration_timing_is_separate_from_sealed_core_timing(tmp_path, monkeypatch):
    events = []
    context = SimpleNamespace(distributed=Solo(), manager=SimpleNamespace(append_metrics=events.append),
                              output=tmp_path, writer=None)
    monkeypatch.setattr(training.time, 'perf_counter', lambda: 20.)
    timing = training._record_iteration_walltime(context, 300, 5., core_seconds=7., periodic_evaluation=True)
    assert timing['seconds'] == 15.
    assert timing['core_seconds'] == 7.
    assert timing['post_update_seconds'] == 8.
    assert timing['periodic_evaluation'] is True
    assert events[0]['event'] == 'iteration_walltime'
    curve = json.loads((tmp_path/'curves.jsonl').read_text())
    assert curve['step'] == 300
    assert curve['metrics']['iteration_walltime/seconds'] == 15.
    assert curve['metrics']['iteration_walltime/core_seconds'] == 7.


def _evaluation_cleanup_worker(rank, directory):
    from datetime import timedelta
    import torch.distributed as dist
    from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
    from gem.closedloop.dppo.parallel_support import capture_local_rng

    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory/'cleanup_rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    actual_restore = training.restore_local_rng
    outcomes = {}
    try:
        distributed = DistributedCollectives(rank, 2, device='cpu')
        for phase in ('journal', 'rng'):
            original_journal = object()
            context = SimpleNamespace(distributed=distributed, generators={}, backend=SimpleNamespace(journal=object()))
            rng = capture_local_rng()
            torch.rand(5)
            def close():
                if rank == 1 and phase == 'journal':
                    raise OSError('injected_journal_close')
            def restore(value, generators):
                actual_restore(value, generators)
                if rank == 1 and phase == 'rng':
                    raise RuntimeError('injected_rng_restore')
            training.restore_local_rng = restore
            try:
                training._restore_evaluation_state(context, SimpleNamespace(close=close), original_journal, rng)
            except RuntimeError as error:
                outcomes[phase] = str(error)
            else:
                raise AssertionError('Rank-local evaluation cleanup failure was not synchronized')
            assert context.backend.journal is original_journal
            assert torch.equal(torch.get_rng_state(), rng['torch'])
            # 若失败卡先退出、健康卡误入计时，双方不可能正常到达同一 barrier。
            distributed.barrier()
        torch.save(outcomes, directory/f'cleanup_rank{rank}.pt')
    finally:
        training.restore_local_rng = actual_restore
        dist.destroy_process_group()


def test_evaluation_cleanup_failure_is_shared_before_next_collective(tmp_path):
    import torch.distributed as dist
    from tests.closedloop.dppo.test_updater_v2 import _spawn_bounded
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo is unavailable')
    _spawn_bounded(_evaluation_cleanup_worker, tmp_path)
    outputs = [torch.load(tmp_path/f'cleanup_rank{rank}.pt', weights_only=False) for rank in range(2)]
    assert outputs[0] == outputs[1]
    assert 'rank 1: OSError: injected_journal_close' in outputs[0]['journal']
    assert 'rank 1: RuntimeError: injected_rng_restore' in outputs[0]['rng']


@pytest.mark.parametrize('failure', ['drain', 'metrics'])
def test_archive_shutdown_releases_manager_when_drain_or_metrics_fail(tmp_path, monkeypatch, failure):
    events = []
    def drain():
        events.append('drain')
        if failure == 'drain':
            raise OSError('injected drain failure')
    def flush(context):
        events.append('metrics')
        if failure == 'metrics':
            raise OSError('injected metrics failure')
    monkeypatch.setattr(training, '_flush_archive_metrics', flush)
    context = SimpleNamespace(maintenance=SimpleNamespace(drain=drain), state={'iteration':1}, output=tmp_path,
        writer=None, manager=SimpleNamespace(close=lambda: events.append('close'), append_metrics=lambda _: None))
    with pytest.raises(OSError, match=f'injected {failure} failure'):
        training._close_run_manager(context, {})
    assert events == ['drain', 'metrics', 'close']


@pytest.mark.parametrize('failure', ['critic', 'actor', 'kl'])
def test_whole_iteration_rollback_preserves_spent_budget(tmp_path, monkeypatch, failure):
    actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    aopt = torch.optim.AdamW(actor.parameters(), lr=5e-9)
    copt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    budget = TrainingBudget(tmp_path/'budget.json', dict(accepted_iterations=2, optimizer_attempts=8,
        generations=20, control_steps=100, physics_steps=400))
    config = configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage10_8gpu_server1_v2.yaml')
    context = SimpleNamespace(distributed=Solo(), initial_iteration=0, stage=config['stage10'],
        settings=config['stage9'], policy=SimpleNamespace(actor=actor), actor=actor, critic=critic,
        actor_optimizer=aopt, critic_optimizer=copt, bc=None, budget=budget, session=tmp_path,
        profile={'microbatch':1}, generators={k:torch.Generator().manual_seed(13) for k in ('actor','critic')})
    old_actor, old_critic = copy.deepcopy(actor.state_dict()), copy.deepcopy(critic.state_dict())
    rng = torch.get_rng_state().clone()
    def step(model, optimizer):
        optimizer.zero_grad(); model(torch.randn(3, 2)).sum().backward(); optimizer.step()
    def fake_critic(*args, **kwargs):
        step(critic, copt)
        if failure == 'critic':
            raise FloatingPointError('injected critic failure')
        return {'optimizer_steps':80}
    def fake_actor(*args, **kwargs):
        kwargs['reserve_attempt']()
        step(actor, aopt)
        if failure == 'actor':
            raise FloatingPointError('injected actor failure')
        return {'optimizer_steps':1}
    monkeypatch.setattr(training, 'probability_check_local', lambda *a, **k: {'passed':True})
    monkeypatch.setattr(training, 'critic_update_local', fake_critic)
    monkeypatch.setattr(training, 'actor_update_v2', fake_actor)
    monkeypatch.setattr(training, 'analytic_kl_local', lambda *a, **k: {
        'mean_joint_kl':1., 'per_denoising_step':[{'mean_joint_kl':1.}]})
    monkeypatch.setattr(training, 'broadcast_state', lambda value, _: value)
    with pytest.raises((FloatingPointError, RuntimeError)):
        training._update(context, SimpleNamespace(transitions=[]), {}, [], 1)
    for name, value in actor.state_dict().items():
        assert torch.equal(value, old_actor[name])
    for name, value in critic.state_dict().items():
        assert torch.equal(value, old_critic[name])
    assert not aopt.state and not copt.state
    assert torch.equal(torch.get_rng_state(), rng)
    assert budget.state_dict()['used']['optimizer_attempts'] == (0 if failure == 'critic' else 1)
    rejected = json.loads((tmp_path/'rejected_000001.json').read_text())
    assert rejected['rolled_back']
    assert rejected['probability_check'] == dict(passed=True, scope='full_rollout_before_first_step')
    assert rejected['timings']['critic_seconds'] > 0
    assert (rejected['timings']['actor_seconds'] > 0) == (failure != 'critic')
    assert (rejected['timings']['kl_seconds'] > 0) == (failure == 'kl')


@pytest.mark.parametrize('first_chain', ['eligible', 'invalid', 'all_prefix'])
def test_ordinary_iteration_records_decomposed_gradients_on_every_actor_step(tmp_path, first_chain):
    from tests.closedloop.dppo.test_distributed_training import SmallCritic
    from tests.closedloop.dppo.test_updater_v2 import Anchor, BatchGaussianPolicy, rows

    class SingleRank(Solo):
        def sum_gradients(self, module):
            pass

        def sum_tensor(self, value):
            return value

        def gather_rows(self, values, positions, count):
            assert sorted(positions) == list(range(count))
            return values[torch.tensor(positions).argsort()].double()

    config = configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage10_8gpu_server1_v2.yaml')
    config['stage9'].update(actor_minibatch_internal_transitions=4, critic_steps=2)
    policy, critic, anchor = BatchGaussianPolicy(), SmallCritic(), Anchor()
    samples = rows(policy)
    if first_chain == 'invalid':
        samples[0].transition_valid = False
    elif first_chain == 'all_prefix':
        samples[0].free_mask.zero_()
        samples[0].context['known_qpos30_mask'].fill_(True)
        samples[0].old_log_prob.zero_()
    manifest = [dict(owner_rank=0, local_index=index, valid=row.transition_valid, has_free=bool(row.free_mask.any()))
                for index, row in enumerate(samples)]
    budget = TrainingBudget(tmp_path/'budget.json', dict(accepted_iterations=2, optimizer_attempts=8,
        generations=20, control_steps=100, physics_steps=400))
    # 使用真实状态接口让主循环也执行根进程不可变快照，而不绕过事务边界。
    anchor.state_dict = lambda: {'calls':anchor.calls}
    context = SimpleNamespace(distributed=SingleRank(), initial_iteration=0, stage=config['stage10'],
        settings=config['stage9'], policy=policy, actor=policy.actor, critic=critic,
        actor_optimizer=torch.optim.AdamW(policy.actor.parameters(), lr=5e-9, weight_decay=0.),
        critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=0.),
        bc=anchor, budget=budget, session=tmp_path, profile={'microbatch':3},
        generators={key:torch.Generator().manual_seed(13) for key in ('actor','critic')})
    targets = dict(advantages=torch.tensor([1., -.3, .4, -.7]), returns=torch.tensor([1., -.2, .7, 1.3]),
                   valid=torch.tensor([row.transition_valid for row in samples]))
    report = training._update(context, SimpleNamespace(transitions=samples), targets, manifest, 2)
    assert report['probability_check']['scope'] == 'rank_sentinel_all_denoising_steps'
    assert report['probability_check']['sentinel_coverage']['checked_upper_transitions'] == 1
    assert report['actor']['optimizer_steps'] == 4
    assert report['actor']['bc_global_samples'] == 8
    for step in report['actor']['steps']:
        gradients = step['gradient_contributions']
        assert gradients['all']['ppo_norm'] > 0
        assert gradients['all']['weighted_bc_norm'] > 0
        assert gradients['shared']['cosine'] is not None
    assert budget.state_dict()['used']['optimizer_attempts'] == 4
