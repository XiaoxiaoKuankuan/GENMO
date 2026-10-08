"""八 rank、多 minibatch、稀疏 checkpoint 与作废尾部的只读审计回归。

夹具实际调用 Writer、journal、seal、v2 save/load 和归档；八个逻辑 rank 各一条
合成轨迹，小模型每轮四次 Actor 更新。保存间隔为300，短测试仅在停止点保存。
不启动 GPU、物理后端或长训练，不证明真实八卡性能、动作质量或训练收敛。
负例重写外层SHA后仍由索引、GAE、计数、BC和冻结语义拒绝；临时文件统一回收。
周期评估夹具复用真实 CPU evaluate_policy，生成八 rank 分片中的四个有效分片、
四来源各一样本和两个 seed；检查终态缺失、报告/episode SHA、计划、指标篡改和
旧格式证据缺口。有效物理失败必须保留为效果指标，不能被审计器误报为基础设施失败。
固定目标直接调用生产 fixed_targets 与 normalize_advantages_global，不人为添加旧版
审计器才需要的 old_values/next_values；另行验证可选重复列及篡改拒绝，防止合成
夹具与真实八卡写盘格式分叉后再次产生错误的通过结论。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo.checkpoint import (
    VERSION_V2,
    capture_rank_state,
    load_checkpoint,
    save_checkpoint,
)
from gem.closedloop.dppo.long_run import LongRunMaintenance
from gem.closedloop.dppo.returns import normalize_advantages_global
from gem.closedloop.dppo.trainer import fixed_targets
from gem.closedloop.dppo.run_management import GuardedStepJournal, RolloutWriter, RunManager
from tests import test_evaluation as fixtures
from tests.closedloop.dppo.test_stage10_audit_recovery import write
from tests.closedloop.dppo.test_periodic_monitor import Catalog
from gem.closedloop.dppo.evaluation import evaluate_policy, _model_fingerprint
from gem.closedloop.dppo.periodic_monitor import build_balanced_plan, merge_periodic_reports, shard_plan
from tools.eval.audit_closedloop_stage10 import (
    audit_evaluation_restore,
    audit_run,
    read_json,
    sha256,
)


class Sampler:
    def __init__(self, state):
        self.state = copy.deepcopy(state)

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


class ParallelLifecycle:
    def __init__(self, temporary):
        self.template = fixtures._stage10_archive(temporary/'template')
        self.root = temporary/'parallel'
        self.data = read_json(self.template/'sessions/s1/data_audit.json')
        self.identity = read_json(self.template/'run.json')['identity']
        self.identity.update(stage10='genmo.closedloop.stage10.v2', base_seed=42,
            execution_contract={'timing_contract': 'deployment_critical.v2'},
            distributed_training=dict(world_size=8, backend='nccl', collection='all_ranks'))
        self.contract = self.identity['training_contract']
        self.contract.update(ppo_epochs=2, max_actor_optimizer_steps=4, actor_minibatch_internal_transitions=8,
            rollout_upper_steps_per_rank=1, rollout_upper_steps=8, critic_steps=2, actor_lr=5e-9,
            bc_batch=2, bc_weight=.1, objective_logprob_reduction='joint_sum')
        self.item = torch.load(self.template/'sessions/s1/iterations/000001/rollout/chunk_000000/transition_000000000.pt',
                               weights_only=False)
        self.item.context['future_times'] += 1.
        self.limits = dict(accepted_iterations=20, optimizer_attempts=100, generations=1000,
                           control_steps=1000, physics_steps=4000)
        self.stage = dict(run_control={}, storage=dict(archive_completed_iterations=True,
            checkpoint_every_iterations=300))
        self.manager = None

    def start(self, checkpoint=None):
        self.manager = RunManager(self.root, resume=self.root.exists())
        self.budget = self.manager.budget(self.limits)
        self.maintenance = LongRunMaintenance(self.manager, self.stage)
        self.directory = self.root/'sessions'/self.manager.session_id
        self.directory.mkdir(parents=True)
        write(self.directory/'data_audit.json', self.data)
        if not (self.root/'run.json').exists():
            write(self.root/'run.json', dict(schema='genmo.closedloop.stage10.run.v2', mode='train', identity=self.identity))
        self.actor, self.critic = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        self.optimizers = dict(actor_optimizer=torch.optim.AdamW(self.actor.parameters(), lr=5e-9),
            critic_optimizer=torch.optim.AdamW(self.critic.parameters(), lr=.001))
        self.music = Sampler(dict(split='train', catalog_identity=self.identity['dataset']))
        self.bc = Sampler(dict(batch_size=2, bc_update_steps=0))
        self.state = dict(iteration=0, policy_version=0, actor_updates=0, critic_updates=0,
            buffer_size=0, pending_plan=False, selected_actor_lr=5e-9, budget=self.budget.state_dict())
        resume = None
        if checkpoint is not None:
            self.state = load_checkpoint(checkpoint, actor=self.actor, critic=self.critic, **self.optimizers,
                identity=self.identity, rank=0, world_size=8, samplers=dict(music=self.music, bc=self.bc))
            local = self.state.pop('local_rank_state')
            self.attempt = max(local['attempt'], self.budget.state_dict()['used']['generations']//8)
            self.manager.reconcile_accepted(self.state['iteration'])
            resume = dict(training_resume=True, restored_full_state=True, old_buffer_discarded=True,
                initial_iteration=self.state['iteration'], checkpoint=str(checkpoint), sha256=sha256(checkpoint))
        else:
            self.attempt = 0
        self.initial = self.state['iteration']
        self.report = dict(schema='genmo.closedloop.stage10.session.v2', session_id=self.manager.session_id,
            world_size=8, initial_iteration=self.initial, resume=resume, identity=self.identity,
            data_audit='data_audit.json', iterations=[], evaluations=[], evaluation_artifacts=[])
        write(self.directory/'session_start.json', dict(self.report, schema='genmo.closedloop.stage10.session_start.v2'))
        write(self.directory/'resolved_config.yaml', dict(stage10=dict(evaluation=dict(
            samples_per_source=1, seeds=[42, 1729], episode_seconds=10., every_iterations=100))))
        if not (self.root/'evaluation_baseline.json').exists():
            self.evaluate('initial')
        return self

    def evaluate(self, label, *, failure=False):
        catalog = Catalog(self.root)
        catalog.identity = self.identity['dataset']
        plan = build_balanced_plan(catalog, samples_per_source=1)
        policy = SimpleNamespace(actor=self.actor)
        paths = []
        for rank in range(4):
            path = self.directory/'phases'/f'eval_{label}'/f'rank{rank:02d}'/'report'
            evaluate_policy(fixtures.Env(policy, failure=failure), policy, shard_plan(plan, rank, 8),
                path, catalog=catalog, episode_seconds=10., actor_identity=dict(
                    iteration=self.state['iteration'], policy_version=self.state['policy_version']))
            paths.append(path/'report.json')
        baseline_path = self.root/'evaluation_baseline.json'
        baseline = read_json(baseline_path) if baseline_path.exists() else None
        merged = merge_periodic_reports(plan, paths, baseline=baseline,
            evaluation_identity=dict(training_identity=self.identity, timing_contract='deployment_critical.v2'))
        merged.update(iteration=self.state['iteration'], session_id=self.directory.name)
        path = self.directory/'evaluations'/f'{label}.json'
        write(path, merged)
        self.report['evaluations'].append(int(label) if label.isdecimal() else label)
        self.report['evaluation_artifacts'].append(dict(label=label, iteration=self.state['iteration'],
            path=str(path.relative_to(self.root)), sha256=sha256(path)))
        if baseline is None:
            write(baseline_path, merged)
        return path

    def accept(self, *, save=False, archive=False):
        index = self.state['iteration']+1
        directory = self.directory/'iterations'/f'{index:06d}'
        collectors, journals, raw_targets = [], [], []
        self.attempt += 1
        for rank in range(8):
            path = directory/f'rank{rank:02d}'
            path.mkdir(parents=True)
            item = copy.deepcopy(self.item)
            item.identity.update(run_id=self.manager.run_id, backend_session_id=f'{self.manager.session_id}:rank{rank}',
                episode_id=f'{self.manager.session_id}:rank{rank}:iteration{index}', decision_id=index-1, policy_version=index-1)
            item.rewards *= rank+1
            item.metadata['value_snapshot_version'] = self.state['critic_updates']
            journal = GuardedStepJournal(path/'execution_journal.sqlite', self.manager.disk_guard)
            journal.append_result(dict(backend_session_id=item.identity['backend_session_id'], mutation_seq=index,
                result=dict(executed_control_steps=2, executed_physics_steps=8)))
            journal.close()
            journals.append(journal)
            writer = RolloutWriter(path/'rollout', policy_version=index-1, disk_guard=self.manager.disk_guard)
            writer.append(item)
            manifest = writer.finish()
            collectors.append(dict(full_train_pool=True, transition_count=1, control_steps=2, rollout_manifest=str(manifest)))
            targets = fixed_targets([item], None, 'cpu', reuse_values=True,
                critic_version=self.state['critic_updates'], normalize=False,
                gamma_upper=self.contract['gamma_upper'], lambda_upper=self.contract['lambda_upper'])
            raw_targets.append(targets)
        raw = torch.cat([target['advantages_raw'] for target in raw_targets])
        statistics = torch.stack((raw.new_tensor(raw.numel()), raw.sum(), raw.square().sum()))
        collective = SimpleNamespace(sum_tensor=lambda local_statistics: statistics.clone())
        for rank, target in enumerate(raw_targets):
            target = normalize_advantages_global(target, distributed=collective)
            assert 'old_values' not in target and 'next_values' not in target
            torch.save(target, directory/f'rank{rank:02d}'/'fixed_targets.pt')
        for model, optimizer, steps in ((self.actor, self.optimizers['actor_optimizer'], 4),
                                         (self.critic, self.optimizers['critic_optimizer'], 2)):
            for _ in range(steps):
                optimizer.zero_grad()
                model(torch.ones(1, 1)).sum().backward()
                optimizer.step()
        self.budget.reserve('train', generations=8, control_steps=16, physics_steps=64)
        self.budget.reserve('update', optimizer_attempts=4)
        self.budget.accept_iteration()
        self.state.update(iteration=index, policy_version=index, actor_updates=self.state['actor_updates']+4,
            critic_updates=self.state['critic_updates']+2, budget=self.budget.state_dict())
        self.bc.state['bc_update_steps'] = self.state['actor_updates']
        steps = [dict(epoch=position//2, optimizer_step=position+1,
            global_upper_indices=list(range((position%2)*4, (position%2)*4+4)), internal_transitions=8,
            ppo_only_gradient_norm=1., total_gradient_norm=2., clip_fraction=.1,
            ratio_scope='before_this_minibatch_step_against_fixed_rollout_old_policy', learning_rates=[5e-9],
            bc=dict(batch_size=2, weight=.1)) for position in range(4)]
        summary = dict(schema='genmo.closedloop.stage10.iteration.v2', status='accepted', iteration=index,
            policy_version_before=index-1, policy_version_after=index, collectors=collectors,
            global_manifest=[dict(owner_rank=rank, local_index=0, valid=True, has_free=True) for rank in range(8)],
            probability_check=dict(passed=True, scope='full_rollout_before_first_step',
                max_abs_log_probability_difference=0., max_abs_ratio_minus_one=0., max_abs_independent_gaussian_difference=0.),
            actor=dict(optimizer_steps=4, optimizer_attempts=4, steps=steps, epoch_orders=[list(range(8)), list(range(8))],
                epochs_started=2, included_upper_transitions=8, excluded_upper_transitions=0, old_statistics_fixed=True,
                hard_kl_pending=True, rollback_scope='caller_owned_whole_rollout', objective_logprob_reduction='joint_sum',
                bc_global_samples=8), critic=dict(optimizer_steps=2, losses=[.1, .09]),
            kl=dict(mean_joint_kl=.001, per_denoising_step=[dict(mean_joint_kl=.001), dict(mean_joint_kl=.001)]),
            actor_lr=5e-9, kl_limit=.02, actor_updates_total=self.state['actor_updates'], critic_updates_total=self.state['critic_updates'],
            gmt_frozen_by_rank=[dict(policy_unchanged=True, runtime_parameters_unchanged=True,
                execution_journal=dict(backend_session_id=f'{self.manager.session_id}:rank{rank}', executed_seq=index, acked_seq=index))
                for rank in range(8)], source_unchanged=dict(unchanged=True),
            replicas=dict(scope='sampled_parameter_values_not_full_hash', world_size=8, passed=True), budget=self.budget.state_dict())
        write(directory/'summary.json', summary)
        self.manager.seal_iteration(directory, index, closed_journals=journals)
        self.report['iterations'].append(index)
        checkpoint = self.save() if save else None
        if archive:
            self.maintenance.enqueue_archive(directory)
            self.maintenance.drain()
        return checkpoint

    def save(self, *, charge_after_accept=False):
        if charge_after_accept:
            self.budget.reserve('evaluation', generations=8, control_steps=16, physics_steps=64)
            self.state['budget'] = self.budget.state_dict()
        path = self.root/'checkpoints'/f'{self.state["iteration"]:06d}-{self.manager.session_id}.pt'
        ranks = [capture_rank_state(rank, state=dict(self.state, decision=self.state['iteration'],
            attempt=self.attempt, episode_count=self.state['iteration'], latency_budget_s=.5),
            samplers=dict(music=self.music, **({'bc': self.bc} if rank==0 else {}))) for rank in range(8)]
        save_checkpoint(path, actor=self.actor, critic=self.critic, **self.optimizers, state=self.state,
            identity=self.identity, config=dict(stage10=dict(storage=dict(checkpoint_every_iterations=300))),
            version=VERSION_V2, rank_states=ranks)
        self.manager.publish_checkpoint(self.state['iteration'], path, metadata=dict(reason='controlled_end', world_size=8,
            actor_updates=self.state['actor_updates'], critic_updates=self.state['critic_updates'],
            actor_model_fingerprint=_model_fingerprint(self.actor)))
        return path

    def finish(self, *, failed=False, hard=False):
        if not hard and not failed and not any(row['iteration']==self.state['iteration']
                and row['label']!='initial' for row in self.report['evaluation_artifacts']):
            self.evaluate(f'final_{self.state["iteration"]:06d}')
        self.manager.close()
        self.manager = None
        if hard:
            return self.directory
        self.report.update(status='failed' if failed else 'passed', exit_code=1 if failed else 0,
            final_state=copy.deepcopy(self.state), source_unchanged=dict(unchanged=True), original_assets_unchanged=True,
            rank_reports=[dict(rank=rank, status='passed', worker_shutdown=dict(gmt=dict(policy_unchanged=True,
                runtime_parameters_unchanged=True, process_exit_code=0))) for rank in range(8)])
        write(self.directory/'summary.json', self.report)
        write(self.directory/'completion.json', dict(schema='genmo.closedloop.stage10.session_completion.v2',
            status=self.report['status'], exit_code=self.report['exit_code'], summary_sha256=sha256(self.directory/'summary.json')))
        return self.directory


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    life = ParallelLifecycle(tmp_path)
    yield life
    if life.manager is not None:
        life.manager.close()


def test_v2_eight_rank_multistep_sparse_checkpoint_and_sealed_archives(lifecycle):
    life = lifecycle.start()
    life.accept(archive=True)
    life.accept(archive=True)
    life.accept(save=True, archive=True)
    life.finish()
    before = {str(path.relative_to(life.root)):sha256(path) for path in life.root.rglob('*') if path.is_file()}
    result = audit_run(life.root, require_resume=False)
    assert result['status']=='passed', result['checks']
    json.dumps(result, allow_nan=False)
    assert result['version'].endswith('.v2')
    assert result['checkpoint_storage']['outer_iterations_without_checkpoint']==2
    assert next(row for row in result['checks'] if row['name']=='v2_iteration:3')['details']['actor_updates_total']==12
    assert before=={str(path.relative_to(life.root)):sha256(path) for path in life.root.rglob('*') if path.is_file()}


def test_v2_failure_before_first_accept_keeps_its_audit_version(lifecycle):
    life = lifecycle.start()
    life.finish(failed=True)
    assert not (life.root/'accepted.json').exists()
    result = audit_run(life.root)
    assert result['status']=='failed'
    assert result['version']=='genmo.closedloop.stage10.audit.v2'
    failure = next(row for row in result['checks'] if row['name']=='parallel_training_recovery_audit')
    assert failure['status']=='failed' and 'accepted.json' in failure['error']
    assert result['minimum_iterations']==2 and result['require_resume'] is True


def test_v2_resume_supersedes_unsaved_tail_without_refunding_resources(lifecycle):
    life = lifecycle.start()
    life.accept(archive=True)
    checkpoint = life.accept(save=True, archive=True)
    life.accept(archive=True)
    old = life.finish(hard=True)
    life.start(checkpoint)
    life.accept(save=True, archive=True)
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result['checks']
    assert result['recovery_history']['superseded_seal_count']==1
    assert read_json(life.root/'budget.json')['used']['accepted_iterations']==4
    historical = next(row for row in result['checks'] if row['name']==f'v2_historical_seal:{old.name}:3')
    assert historical['details']['disposition']=='superseded_unsaved'
    next((life.root/'superseded_tails').glob('*.json')).unlink()
    assert audit_run(life.root)['status']=='failed'


def test_v2_final_checkpoint_can_include_later_evaluation_budget(lifecycle):
    life = lifecycle.start()
    life.accept()
    life.accept()
    life.save(charge_after_accept=True)
    life.finish()
    result = audit_run(life.root, require_resume=False)
    assert result['status']=='passed', result['checks']


def test_v2_failed_later_session_is_not_misclassified_as_recovered(lifecycle):
    life = lifecycle.start()
    life.accept()
    checkpoint = life.accept(save=True)
    life.finish()
    life.start(checkpoint)
    life.finish(failed=True)
    result = audit_run(life.root, require_resume=False)
    assert result['status']=='failed', result['checks']


@pytest.mark.parametrize('fault', ['owner', 'actor_count', 'bc', 'frozen', 'global_advantage', 'archive', 'completion'])
def test_v2_semantic_or_archive_tampering_never_passes(lifecycle, fault):
    life = lifecycle.start()
    life.accept()
    life.accept(save=True)
    life.finish()
    directory = life.directory/'iterations/000002'
    if fault=='completion':
        (life.directory/'completion.json').unlink()
    elif fault=='archive':
        with RunManager(life.root, resume=True) as manager:
            maintenance = LongRunMaintenance(manager, life.stage)
            maintenance.archive_iteration(directory)
        (directory/'execution_evidence.tar.gz').write_bytes(b'bad archive')
    else:
        summary = read_json(directory/'summary.json')
        if fault=='owner':
            summary['global_manifest'][0]['owner_rank']=1
        elif fault=='actor_count':
            summary['actor_updates_total']=2
        elif fault=='bc':
            summary['actor']['bc_global_samples']=2
        elif fault=='frozen':
            summary['gmt_frozen_by_rank'][0]['policy_unchanged']=False
        elif fault=='global_advantage':
            path = directory/'rank00/fixed_targets.pt'
            target = torch.load(path, weights_only=False)
            target['advantages'] += .1
            torch.save(target, path)
        write(directory/'summary.json', summary)
        seal_path = directory/'seal_manifest.json'
        seal = read_json(seal_path)
        seal['summary_sha256']=sha256(directory/'summary.json')
        for member in seal['members']:
            path = directory/member['path']
            member.update(sha256=sha256(path), size_bytes=path.stat().st_size)
        write(seal_path, seal)
        accepted = read_json(life.root/'accepted.json')
        accepted['seal_sha256']=sha256(seal_path)
        write(life.root/'accepted.json', accepted)
    result = audit_run(life.root, require_resume=False, allow_incomplete=True)
    assert result['status']!='passed' and result['failed_checks']+result['not_run_checks']>0


def test_v1_multistep_evidence_still_rejected(tmp_path):
    root = fixtures._stage10_archive(tmp_path/'old')
    path = root/'sessions/s1/iterations/000001/summary.json'
    summary = read_json(path)
    summary['actor']['optimizer_steps']=4
    write(path, summary)
    result = audit_run(root)
    assert result['status']=='failed'


def test_v2_evaluation_audits_weights_without_interpreting_training_rng(tmp_path):
    checkpoint = tmp_path/'model.pt'
    identity = {'stage10': 'genmo.closedloop.stage10.v2'}
    torch.save(dict(version=VERSION_V2, identity=identity, actor={'weight': torch.ones(1)}, critic={},
        state=dict(iteration=300), rank_states=[dict(state={'decision': 'not an evaluation counter'})]), checkpoint)
    session = dict(initial_iteration=300, resume=dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
        restore_mode='weights_only', restored_full_state=False, training_resume=False))
    rng = torch.get_rng_state().clone()
    result = audit_evaluation_restore(session, identity)
    assert result['restore_mode']=='weights_only' and not result['optimizer_restored'] and not result['rng_restored']
    assert torch.equal(rng, torch.get_rng_state())
    session['resume']['restored_full_state'] = True
    with pytest.raises(ValueError, match='falsely claims'):
        audit_evaluation_restore(session, identity)


def _rewrite_session_evaluation_publication(life):
    """负例重新绑定外层 SHA，让审计继续检查更深层的任务与指标语义。"""
    summary = read_json(life.directory/'summary.json')
    for artifact in summary.get('evaluation_artifacts', []):
        path = life.root/artifact['path']
        if path.exists():
            artifact['sha256'] = sha256(path)
    write(life.directory/'summary.json', summary)
    completion = read_json(life.directory/'completion.json')
    completion['summary_sha256'] = sha256(life.directory/'summary.json')
    write(life.directory/'completion.json', completion)


@pytest.mark.parametrize('fault', ['missing_final', 'missing_final_declaration', 'report_sha',
    'aggregate', 'task_count', 'plan', 'episode_sha', 'rank_failure'])
def test_v2_terminal_periodic_evaluation_requires_real_fixed_evidence(lifecycle, fault):
    life = lifecycle.start()
    life.accept()
    life.accept(save=True)
    life.finish()
    path = life.directory/'evaluations/final_000002.json'
    report = read_json(path)
    if fault=='missing_final':
        path.unlink()
    elif fault=='missing_final_declaration':
        session = read_json(life.directory/'summary.json')
        session['evaluations'] = ['initial']
        session['evaluation_artifacts'] = session['evaluation_artifacts'][:1]
        write(life.directory/'summary.json', session)
        _rewrite_session_evaluation_publication(life)
    elif fault=='report_sha':
        path.write_text(path.read_text()+'\n')
    elif fault in ('aggregate', 'task_count', 'plan'):
        if fault=='aggregate':
            report['source_balanced_reward'] += 1.
        elif fault=='task_count':
            report['task_count'] -= 1
        else:
            report['plan']['tasks'][0]['music_start_frame'] += 1
        write(path, report)
        _rewrite_session_evaluation_publication(life)
    elif fault=='episode_sha':
        episode = Path(report['source_episode_manifests'][0]['path'])
        value = read_json(episode)
        value['reward_sum'] += 1.
        write(episode, value)
    else:
        rank_path = life.directory/'phases/eval_final_000002/rank00/report/report.json'
        value = read_json(rank_path)
        value['status'] = 'failed'
        write(rank_path, value)
    result = audit_run(life.root, require_resume=False)
    assert result['status']=='failed'
    checked = next(row for row in result['checks'] if row['name'].startswith('v2_periodic_evaluations:'))
    assert checked['status']=='failed', result['checks']


def test_v2_periodic_physical_failures_are_valid_measured_outcomes(lifecycle):
    life = lifecycle.start()
    life.accept()
    life.accept(save=True)
    life.evaluate('final_000002', failure=True)
    life.finish()
    result = audit_run(life.root, require_resume=False)
    assert result['status']=='passed', result['checks']
    final = result['periodic_evaluations'][life.directory.name]['reports'][-1]
    assert final['task_count']==8 and final['physical_failure_count']==8
    assert final['infrastructure_complete'] and final['physical_failures_are_valid_policy_outcomes']


def test_v2_legacy_missing_evaluation_sha_is_readable_but_not_full_acceptance(lifecycle):
    life = lifecycle.start()
    life.accept()
    life.accept(save=True)
    life.finish()
    summary = read_json(life.directory/'summary.json')
    del summary['evaluation_artifacts']
    write(life.directory/'summary.json', summary)
    _rewrite_session_evaluation_publication(life)
    result = audit_run(life.root, require_resume=False, allow_incomplete=True)
    assert result['status']=='incomplete', result['checks']
    assert next(row for row in result['checks'] if row['name']=='v2_iteration:2')['status']=='passed'
    legacy = next(row for row in result['checks'] if row['name'].startswith('v2_periodic_evaluations:'))
    assert legacy['status']=='not_run' and 'legacy V2' in legacy['error']


@pytest.mark.parametrize('variant', ['production', 'matching_optional_values', 'wrong_old',
    'wrong_next', 'wrong_return', 'wrong_advantage', 'not_normalized'])
def test_v2_targets_use_production_schema_and_independent_rollout_values(lifecycle, variant):
    """目标由真实生产函数生成；重签封存 SHA 后仍独立识别值/GAE/归一化错误。"""
    life = lifecycle.start()
    life.accept()
    life.accept(save=True)
    life.finish()
    directory = life.directory/'iterations/000002'
    path = directory/'rank00/fixed_targets.pt'
    target = torch.load(path, weights_only=False)
    assert 'old_values' not in target and 'next_values' not in target
    item = torch.load(next((directory/'rank00/rollout').rglob('transition_*.pt')), weights_only=False)
    if variant in ('matching_optional_values', 'wrong_old', 'wrong_next'):
        target.update(old_values=torch.tensor([item.old_value], dtype=torch.float64),
                      next_values=torch.tensor([item.next_value], dtype=torch.float64))
        if variant=='wrong_old':
            target['old_values'] += 1.
        elif variant=='wrong_next':
            target['next_values'] += 1.
    elif variant=='wrong_return':
        target['returns'] += 1.
    elif variant=='wrong_advantage':
        target['advantages'] += .1
    elif variant=='not_normalized':
        target['advantages_normalized'] = False
    torch.save(target, path)
    seal_path = directory/'seal_manifest.json'
    seal = read_json(seal_path)
    for member in seal['members']:
        member_path = directory/member['path']
        member.update(sha256=sha256(member_path), size_bytes=member_path.stat().st_size)
    write(seal_path, seal)
    pointer = read_json(life.root/'accepted.json')
    pointer['seal_sha256'] = sha256(seal_path)
    write(life.root/'accepted.json', pointer)
    result = audit_run(life.root, require_resume=False)
    if variant in ('production', 'matching_optional_values'):
        assert result['status']=='passed', result['checks']
        detail = next(row for row in result['checks'] if row['name']=='v2_iteration:2')['details']
        target_audit = detail['fixed_targets_by_rank'][0]
        assert target_audit['value_source'].startswith('SHA-verified immutable rollout')
        assert target_audit['optional_duplicate_columns_verified']==(
            ['old_values', 'next_values'] if variant=='matching_optional_values' else [])
    else:
        assert result['status']=='failed'
        assert next(row for row in result['checks'] if row['name']=='v2_iteration:2')['status']=='failed'
