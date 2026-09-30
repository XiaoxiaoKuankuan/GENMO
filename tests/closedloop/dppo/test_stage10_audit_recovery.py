"""第十步审计对失败、恢复及 checkpoint 发布窗口的合成 CPU 回归。

夹具实际调用 RunManager、持久 TrainingBudget、逐条 RolloutWriter、完整
save_checkpoint/load_checkpoint 和两个小型 Adam 优化器，生成具有不同 session
UUID 的正式格式证据。数据目录、概率/KL 与网络隔离报告使用合成模板，因此本文件
只证明审计如何选择真实恢复链、保留失败历史及拒绝篡改，不证明 GPU 训练或动力学。
注入窗口包括：接受轮次后进程中断而无 session 摘要、不可变 publication 已写但
latest 写失败、失败后重试同一逻辑轮次，以及已恢复历史中的完成标记被篡改。
新执行契约另核 checkpoint 的噪声计数与真实归档 decision；负例重写外层 SHA 后
仍必须由计数语义校验拒绝，旧证据则明确标记 legacy_not_recorded。
所有文件位于 pytest 临时目录；无需服务器、GPU 或外部数据，统一测试入口负责清理。
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo import run_management as management
from gem.closedloop.dppo.checkpoint import load_checkpoint, save_checkpoint
from tests import test_evaluation as evaluation_fixtures
from tools.eval.audit_closedloop_stage10 import audit_run, main, read_json, sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class Lifecycle:
    def __init__(self, temporary):
        self.template = evaluation_fixtures._stage10_archive(temporary/'template')
        self.root = temporary/'run'
        self.identity = read_json(self.template/'run.json')['identity']
        self.data = read_json(self.template/'sessions/s1/data_audit.json')
        self.template_iteration = read_json(self.template/'sessions/s1/iterations/000001/summary.json')
        self.template_item = torch.load(self.template/'sessions/s1/iterations/000001/rollout/chunk_000000/transition_000000000.pt', weights_only=False)
        self.template_item.context['future_times'] += 1.
        self.targets = torch.load(self.template/'sessions/s1/iterations/000001/fixed_targets.pt', weights_only=False)
        self.limits = dict(accepted_iterations=8, optimizer_attempts=20, generations=100, control_steps=1000, physics_steps=4000)
        self.manager = None

    def with_counter_contract(self):
        self.identity.update(base_seed=42, execution_contract={'protocol_version':'synthetic-execution-contract'})
        return self

    def start(self, checkpoint=None):
        self.manager = management.RunManager(self.root, resume=self.root.exists())
        self.budget = self.manager.budget(self.limits)
        self.directory = self.root/'sessions'/self.manager.session_id
        self.directory.mkdir(parents=True)
        write(self.directory/'data_audit.json', self.data)
        if not (self.root/'run.json').exists():
            write(self.root/'run.json', dict(schema='genmo.closedloop.stage10.run.v1', identity=self.identity, mode='train'))
        self.actor, self.critic = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-8)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        sampler_state = dict(split='train', catalog_identity=self.identity['dataset'])
        self.sampler = SimpleNamespace(state_dict=lambda: sampler_state,
                                       load_state_dict=lambda value: sampler_state.update(value))
        self.state = dict(iteration=0, policy_version=0, actor_updates=0, critic_updates=0,
                          buffer_size=0, pending_plan=False, selected_actor_lr=1e-8,
                          session_id=self.manager.session_id, budget=self.budget.state_dict())
        if 'execution_contract' in self.identity:
            self.state.update(decision=0, attempt=0, episode_count=0, latency_budget_s=.5)
        resume = None
        if checkpoint is not None:
            self.state = load_checkpoint(checkpoint, actor=self.actor, critic=self.critic,
                actor_optimizer=self.actor_optimizer, critic_optimizer=self.critic_optimizer,
                identity=self.identity, samplers={'music': self.sampler})
            resume = dict(training_resume=True, checkpoint=str(checkpoint), sha256=sha256(checkpoint),
                restored_full_state=True, old_buffer_discarded=True, initial_iteration=self.state['iteration'],
                new_backend_session_id='worker:'+self.manager.session_id)
        self.report = dict(session_id=self.manager.session_id, mode='train', initial_iteration=self.state['iteration'],
            resume=resume, iterations=[], evaluations=[], data_audit=str((self.directory/'data_audit.json').relative_to(self.root)))
        self.manager.append_metrics(dict(event='session_start', mode='train', initial_iteration=self.state['iteration'],
            resume=resume, backend_session_id='worker:'+self.manager.session_id))
        if checkpoint is None:
            self.save(self.root/'checkpoints/initial.pt')
        return self

    def save(self, path):
        return save_checkpoint(path, actor=self.actor, critic=self.critic, actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer, state=self.state, identity=self.identity, config={}, samplers={'music': self.sampler})

    def publish(self, *, fail_latest=False):
        index = self.state['iteration']+1
        directory = self.directory/'iterations'/f'{index:06d}'
        directory.mkdir(parents=True)
        item = copy.deepcopy(self.template_item)
        item.identity.update(run_id=self.manager.run_id, backend_session_id='worker:'+self.manager.session_id,
                             episode_id=f'{self.manager.session_id}:{index}', policy_version=index-1)
        if 'execution_contract' in self.identity:
            item.identity['decision_id'] = self.state['decision']
        writer = management.RolloutWriter(directory/'rollout', policy_version=index-1)
        writer.append(item)
        manifest = writer.finish()
        torch.save(self.targets, directory/'fixed_targets.pt')
        for model, optimizer in ((self.actor, self.actor_optimizer), (self.critic, self.critic_optimizer)):
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
        self.budget.reserve('train', generations=1, control_steps=2, physics_steps=8)
        self.budget.reserve('update', optimizer_attempts=1)
        self.budget.accept_iteration()
        self.state.update(iteration=index, policy_version=index, actor_updates=index, critic_updates=index*20,
                          session_id=self.manager.session_id, budget=self.budget.state_dict())
        if 'execution_contract' in self.identity:
            self.state.update(decision=self.state['decision']+1, attempt=self.budget.state_dict()['used']['generations'],
                              episode_count=self.state['episode_count']+1)
        checkpoint = self.root/'checkpoints'/f'{index:06d}-{self.manager.session_id}.pt'
        self.save(checkpoint)
        summary = copy.deepcopy(self.template_iteration)
        summary.update(iteration=index, policy_version_before=index-1, policy_version_after=index,
            checkpoint=str(checkpoint.relative_to(self.root)), rollout_manifest=str(manifest.relative_to(self.root)),
            targets_path=str((directory/'fixed_targets.pt').relative_to(self.root)), budget=self.budget.state_dict())
        summary['gmt_frozen']['execution_journal']['backend_session_id'] = 'worker:'+self.manager.session_id
        summary_path = directory/'summary.json'
        write(summary_path, summary)
        metadata = dict(iteration_summary=str(summary_path.relative_to(self.root)))
        if fail_latest:
            original = management._atomic_json
            def failure(path, *args, **kwargs):
                if path==self.root/'latest.json':
                    raise OSError('synthetic latest publication failure')
                return original(path, *args, **kwargs)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(management, '_atomic_json', failure)
                with pytest.raises(OSError, match='latest publication'):
                    self.manager.publish_checkpoint(index, checkpoint, metadata)
            summary.update(status='failed', error={'message': 'latest publication failed'})
            write(summary_path, summary)
        else:
            self.manager.publish_checkpoint(index, checkpoint, metadata)
            self.report['iterations'].append(metadata['iteration_summary'])
            self.manager.append_metrics(dict(event='iteration_accepted', iteration=index, summary=metadata['iteration_summary']))
        return checkpoint

    def finish(self, *, status='passed', hard=False, **fields):
        directory = self.directory
        if not hard:
            self.report.update(status=status, exit_code=0 if status=='passed' else 1,
                final_iteration=self.state['iteration'], budget=self.budget.state_dict(), source_unchanged={'unchanged':True},
                original_assets_unchanged=True, worker_shutdown={'gmt':{'policy_unchanged':True,
                    'runtime_parameters_unchanged':True, 'process_exit_code':0}}, **fields)
            write(directory/'summary.json', self.report)
            self.manager.append_metrics(dict(event='session_end', status=status, exit_code=self.report['exit_code']))
            write(directory/'completion.json', dict(schema='genmo.closedloop.stage10.session_completion.v1',
                summary_sha256=sha256(directory/'summary.json'), status=status, exit_code=self.report['exit_code']))
        self.manager.close()
        self.manager = None
        return directory


@pytest.fixture
def lifecycle(tmp_path):
    value = Lifecycle(tmp_path)
    yield value
    if value.manager is not None:
        value.manager.close()


def test_failed_session_then_real_resume_is_audited_without_erasing_history(lifecycle):
    life = lifecycle.start()
    first = life.publish()
    failed = life.finish(status='failed')
    life.start(first).publish()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result
    assert result['execution_counter_contract']=='legacy_not_recorded'
    history = {row['session_id']:row for row in result['recovery_history']['sessions']}
    assert history[failed.name]['status']=='failed'
    assert history[failed.name]['disposition']=='recovered_history'
    assert read_json(failed/'summary.json')['status']=='failed'
    destination = life.root/'recovery_audit.json'
    assert main(['--run-dir', str(life.root), '--output', str(destination)])==0
    assert read_json(destination)['status']=='passed'


@pytest.mark.parametrize('counter_contract', [False, True])
def test_immutable_publication_then_latest_failure_retry_selects_actual_chain(lifecycle, counter_contract):
    life = (lifecycle.with_counter_contract() if counter_contract else lifecycle).start()
    first = life.publish()
    life.publish(fail_latest=True)
    failed = life.finish(status='failed')
    life.start(first).publish()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result
    rejected = result['recovery_history']['unselected_publications']
    assert len(rejected)==1 and rejected[0]['iteration']==2 and rejected[0]['session_id']==failed.name
    assert len(list((life.root/'checkpoints/publications').glob('000000002-*.json')))==2
    if counter_contract:
        counters = next(row for row in result['checks'] if row['name']=='iteration:2')['details']['checkpoint']['execution_counters']
        assert counters['decision']==2 and counters['attempt']==3


def test_latest_without_session_summary_is_checked_then_recovered(lifecycle):
    life = lifecycle.start()
    first = life.publish()
    interrupted = life.finish(hard=True)
    incomplete = audit_run(life.root, allow_incomplete=True, minimum_iterations=1, require_resume=False)
    assert incomplete['status']=='incomplete', incomplete
    assert next(row for row in incomplete['checks'] if row['name']=='iteration:1')['status']=='passed'
    life.start(first).publish()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result
    assert not (interrupted/'summary.json').exists()
    assert next(row for row in result['recovery_history']['sessions'] if row['session_id']==interrupted.name)['disposition']=='recovered_history'


@pytest.mark.parametrize('artifact', ['checkpoint', 'old_probability', 'completion'])
def test_recovered_canonical_artifact_tampering_still_fails(lifecycle, artifact):
    life = lifecycle.start()
    first = life.publish()
    old = life.finish(status='failed')
    life.start(first).publish()
    life.finish()
    if artifact=='checkpoint':
        with first.open('ab') as stream:
            stream.write(b'tampered canonical checkpoint')
    elif artifact=='completion':
        value = read_json(old/'completion.json')
        value['summary_sha256'] = 'wrong'
        write(old/'completion.json', value)
    else:
        path = old/'iterations/000001/summary.json'
        value = read_json(path)
        value['probability_check']['max_abs_ratio_minus_one'] = .2
        write(path, value)
    result = audit_run(life.root, allow_incomplete=True)
    assert result['status']=='failed', result


def test_failure_after_last_accepted_checkpoint_is_not_silently_passed(lifecycle):
    life = lifecycle.start()
    first = life.publish()
    life.finish()
    life.start(first)
    failed = life.finish(status='failed')
    result = audit_run(life.root, minimum_iterations=1, require_resume=False)
    assert result['status']=='failed', result
    assert next(row for row in result['recovery_history']['sessions'] if row['session_id']==failed.name)['disposition']=='unrecovered'


def test_failure_without_publication_recovered_by_later_durable_resume(lifecycle):
    life = lifecycle.start()
    first = life.publish()
    life.finish()
    life.start(first)
    failed = life.finish(status='failed')
    life.start(first).publish()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result
    assert next(row for row in result['recovery_history']['sessions'] if row['session_id']==failed.name)['recovered_by']


def test_successful_budget_boundary_stop_after_latest_is_allowed(lifecycle):
    life = lifecycle.start()
    first = life.publish()
    life.finish()
    life.start(first).publish()
    second = life.root/read_json(life.root/'latest.json')['path']
    life.finish()
    life.start(second)
    life.finish(stop_reason='budget_exhausted')
    result = audit_run(life.root)
    assert result['status']=='passed', result


def test_resume_without_update_or_explicit_stop_is_not_accepted(lifecycle):
    life = lifecycle.start()
    first = life.publish()
    life.finish()
    life.start(first).publish()
    second = life.root/read_json(life.root/'latest.json')['path']
    life.finish()
    life.start(second)
    life.finish()
    assert audit_run(life.root)['status']=='failed'


def test_budget_extension_preserves_old_checkpoint_snapshots_and_requires_evidence(lifecycle):
    life = lifecycle.start()
    first = life.publish()
    life.finish()
    life.start(first)
    limits = dict(life.limits, accepted_iterations=10)
    descriptor = life.budget.extend_limits(limits, reason='explicit CPU test extension',
        checkpoint_sha256=sha256(first), config_sha256='b'*64)
    life.publish()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result
    path = life.root/descriptor['path']
    value = read_json(path)
    value['reason'] = 'changed after publication'
    write(path, value)
    assert audit_run(life.root)['status']=='failed'


def test_iteration_zero_resume_after_first_failure_retains_failed_session(lifecycle):
    life = lifecycle.start()
    initial = life.root/'checkpoints/initial.pt'
    failed = life.finish(status='failed')
    life.start(initial).publish()
    life.publish()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result
    assert next(row for row in result['recovery_history']['sessions'] if row['session_id']==failed.name)['disposition']=='recovered_history'


@pytest.mark.parametrize('fault', ['missing_start', 'wrong_resume', 'same_worker'])
def test_hard_interruption_cannot_invent_missing_or_inconsistent_recovery_evidence(lifecycle, fault):
    life = lifecycle.start()
    first = life.publish()
    life.finish()
    life.start(first).publish()
    second = life.root/read_json(life.root/'latest.json')['path']
    interrupted = life.finish(hard=True)
    life.start(second).publish()
    life.finish()
    path = life.root/'metrics'/f'{interrupted.name}.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if fault=='missing_start':
        records = [row for row in records if row['event']!='session_start']
    else:
        start = next(row for row in records if row['event']=='session_start')
        if fault=='wrong_resume':
            start['resume']['sha256'] = 'f'*64
        else:
            start['resume']['new_backend_session_id'] = 'worker:invented'
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    assert audit_run(life.root, allow_incomplete=True)['status']=='failed'


def test_interrupted_partial_metrics_tail_is_retained_and_reported(lifecycle):
    life = lifecycle.start()
    first = life.publish()
    interrupted = life.finish(hard=True)
    path = life.root/'metrics'/f'{interrupted.name}.jsonl'
    with path.open('ab') as stream:
        stream.write(b'{"event": "unfinished')
    life.start(first).publish()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed', result
    assert next(row for row in result['recovery_history']['sessions'] if row['session_id']==interrupted.name)['trailing_partial_metrics']
    assert path.read_bytes().endswith(b'{"event": "unfinished')


def _rewrite_latest_checkpoint(life, mutation):
    """故意更新 SHA/大小，确保负例依靠计数语义而非旧文件哈希拒绝。"""
    latest = read_json(life.root/'latest.json')
    path = life.root/latest['path']
    saved = torch.load(path, weights_only=False)
    mutation(saved['state'])
    torch.save(saved, path)
    latest.update(sha256=sha256(path), size_bytes=path.stat().st_size)
    write(life.root/latest['publication'], {key:value for key,value in latest.items() if key!='publication'})
    write(life.root/'latest.json', latest)


def test_new_execution_contract_checks_rows_counters_and_resume_continuity(lifecycle):
    life = lifecycle.with_counter_contract().start()
    life.budget.reserve('calibration', generations=4)
    life.state.update(decision=4, attempt=4, episode_count=1, budget=life.budget.state_dict())
    life.save(life.root/'checkpoints/initial.pt')
    first = life.publish()
    life.finish()
    life.start(first).publish()
    life.publish()
    life.finish()
    result = audit_run(life.root)
    assert result['status']=='passed' and result['execution_counter_contract']=='verified', result
    checks = {row['name']:row for row in result['checks']}
    counters = checks['iteration:3']['details']['checkpoint']['execution_counters']
    assert counters['decision']==7 and counters['attempt']==7 and counters['episode_count']==4
    assert counters['first_archived_decision_id']==counters['last_archived_decision_id']==6
    resumed = checks['resume_then_optimize']['details'][0]['execution_counters']
    assert resumed['saved']['decision']==resumed['next_first_decision_id']==5
    assert resumed['decision_continuous'] is True


@pytest.mark.parametrize('field,value,error', [
    ('decision', 1, 'last archived decision plus one'),
    ('attempt', 3, 'spent generation budget'),
    ('episode_count', -1, 'episode_count'),
    ('latency_budget_s', 0., 'latency budget'),
    ('decision', True, 'integer'),
])
def test_new_counter_tampering_fails_after_checkpoint_publication_sha_is_updated(lifecycle, field, value, error):
    life = lifecycle.with_counter_contract().start()
    first = life.publish()
    life.finish()
    life.start(first).publish()
    life.finish()
    _rewrite_latest_checkpoint(life, lambda state:state.update({field:value}))
    result = audit_run(life.root)
    assert result['status']=='failed' and result['execution_counter_contract']=='required_not_verified', result
    check = next(row for row in result['checks'] if row['name']=='iteration:2')
    assert error in check['error'] and 'SHA' not in check['error']


def test_new_counter_resume_reset_cannot_hide_behind_self_consistent_last_row(lifecycle):
    life = lifecycle.with_counter_contract().start()
    first = life.publish()
    life.finish()
    life.start(first).publish()
    directory = life.directory/'iterations/000002/rollout'
    life.finish()
    chunk_path = directory/'chunk_000000/manifest.json'
    chunk = read_json(chunk_path)
    record = chunk['records'][0]
    path = chunk_path.parent/record['path']
    item = torch.load(path, weights_only=False)
    item.identity['decision_id'] = 0
    torch.save(item, path)
    record.update(sha256=sha256(path), size_bytes=path.stat().st_size)
    record['identity']['decision_id'] = 0
    write(chunk_path, chunk)
    manifest = read_json(directory/'manifest.json')
    manifest['chunks'][0]['sha256'] = sha256(chunk_path)
    write(directory/'manifest.json', manifest)
    _rewrite_latest_checkpoint(life, lambda state:state.update(decision=1))
    result = audit_run(life.root)
    assert result['status']=='failed', result
    check = next(row for row in result['checks'] if row['name']=='iteration:2')
    assert 'does not continue the saved checkpoint' in check['error']
