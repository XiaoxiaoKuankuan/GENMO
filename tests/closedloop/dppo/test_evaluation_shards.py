"""完整评估分片和合并的 CPU 证据链测试。

本测试在自动清理的临时目录建立真实四库小清单及配对文件，使用既有评估环境替身
产生逐控制区间奖励和原始评估报告，然后经原 audit_run 验证各片并合并。测试不加载
真实 checkpoint、不调用 GPU 或物理仿真，微型 checkpoint 字节只用于文件 SHA 身份。
正例核对任务全集、每样本多 seed、任务噪声与文件 SHA 保持，以及独立重算统计；负例
覆盖漏片、重复、计划/证据篡改、checkpoint/训练身份/实现身份不同和未完成会话。
工具源码用临时文件模拟，包含分片、driver 和复用审计器，避免依赖尚未冻结的新文件。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gem.closedloop.dppo.evaluation import _digest, aggregate_evaluation, build_evaluation_tasks, evaluate_policy
from gem.closedloop.dppo.evaluation_shards import (EXTENSION_FILES, extension_provenance,
    verify_extension_provenance, partition_evaluation_plan, publish_shard_manifest, merge_evaluation_shards)
from gem.closedloop.dppo.full_dataset import FullMusicCatalog
from gem.closedloop.dppo.run_management import TrainingBudget
from gem.closedloop.evaluation_music import sha256_file
from tests.closedloop.dppo.test_full_dataset import make_catalog_data
from tests.test_evaluation import Env, policy


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def evidence(tmp_path):
    catalog = FullMusicCatalog(make_catalog_data(tmp_path))
    audited = catalog.audit_files()
    parent = build_evaluation_tasks(catalog, eval_count='all', seeds=(11, 22), episode_seconds=.04)
    extension_root = tmp_path/'extension'
    for name in EXTENSION_FILES:
        path = extension_root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# 本测试的独立源码身份占位\n')
    extension = extension_provenance(extension_root)
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'only SHA identity; never deserialized')
    identity = dict(dataset=catalog.identity, data_content_sha256=audited['data_content_sha256'],
                    training_contract={'test': True}, source_manifest_sha256='original_unchanged')
    loaded = policy()
    return dict(catalog=catalog, audited=audited, parent=parent, extension=extension,
                checkpoint=checkpoint, identity=identity, policy=loaded)


def make_shard(tmp_path, evidence, index, *, count=2, checkpoint=None, identity=None, extension=None,
               control_limit=1000):
    root = tmp_path/f'shard_{index}'
    session = root/'sessions'/'session'
    root.mkdir(parents=True)
    plan = partition_evaluation_plan(evidence['parent'], index, count)
    checkpoint = evidence['checkpoint'] if checkpoint is None else checkpoint
    identity = evidence['identity'] if identity is None else identity
    extension = evidence['extension'] if extension is None else extension
    write(root/'run.json', dict(schema='genmo.closedloop.stage10.run.v1', identity=identity, mode='eval'))
    write(session/'data_audit.json', evidence['audited'])
    report_path = session/'evaluation_report'/'report.json'
    report = evaluate_policy(Env(evidence['policy']), evidence['policy'], plan, report_path.parent,
        catalog=evidence['catalog'], actor_identity=dict(checkpoint=str(checkpoint), sha256=sha256_file(checkpoint), iteration=4))
    budget = TrainingBudget(root/'budget.json', dict(accepted_iterations=1, optimizer_attempts=1,
        generations=100, control_steps=control_limit, physics_steps=4*control_limit))
    controls = report['aggregate']['overall']['executed_control_steps']
    budget.reserve('evaluation', generations=report['aggregate']['overall']['decision_count'], control_steps=controls, physics_steps=4*controls)
    summary = dict(status='passed', exit_code=0, mode='eval', session_id='session',
        data_audit=str((session/'data_audit.json').relative_to(root)),
        evaluations=[str(report_path.relative_to(root))], source_unchanged={'unchanged':True},
        original_assets_unchanged=True, evaluated_checkpoint_unchanged=True,
        worker_shutdown={'gmt':dict(policy_unchanged=True,
            runtime_parameters_unchanged=True, process_exit_code=0, forced_shutdown=False)},
        budget=budget.state_dict(), extension_source_provenance=extension,
        extension_source_unchanged=verify_extension_provenance(extension))
    summary_path, completion = session/'summary.json', session/'completion.json'
    write(summary_path, summary)
    write(completion, dict(schema='genmo.closedloop.stage10.session_completion.v1',status='passed',exit_code=0,
                           summary_sha256=sha256_file(summary_path)))
    # driver提前保存完整计划，同一内容不能被publisher覆盖或判为冲突。
    write(root/'parent_plan.json', evidence['parent'])
    publish_shard_manifest(root, parent_plan=evidence['parent'], shard_plan=plan,
        evaluation_report=report_path, session_summary=summary_path, completion=completion,
        training_identity=identity, extension=extension)
    return root


def test_partition_preserves_complete_pool_exact_tasks_and_all_seeds(evidence):
    parent = evidence['parent']
    children = [partition_evaluation_plan(parent, i, 3) for i in range(3)]
    ids = [task['task_id'] for child in children for task in child['tasks']]
    assert len(ids) == len(set(ids)) == len(parent['tasks'])
    assert set(ids) == {task['task_id'] for task in parent['tasks']}
    for i, child in enumerate(children):
        assert child['partition']['parent_sample_indices'] == list(range(i, 4, 3))
        assert child['complete_pool_count'] == 4 and child['seeds'] == [11, 22]
        assert len(child['tasks']) == child['requested_eval_count']*2
        assert all(task in parent['tasks'] for task in child['tasks'])
        assert child == partition_evaluation_plan(parent, i, 3)
    assert 'partition' not in parent and parent['requested_eval_count'] == 'all'


@pytest.mark.parametrize('index,count', [(-1,2),(2,2),(True,2),(0,0),(0,5),(0,1.5)])
def test_partition_rejects_invalid_indices(evidence, index, count):
    with pytest.raises(ValueError):
        partition_evaluation_plan(evidence['parent'], index, count)


@pytest.mark.parametrize('field', ['hash', 'subset', 'noise', 'duplicate'])
def test_parent_cannot_be_tampered_or_preselected(evidence, field):
    parent = copy.deepcopy(evidence['parent'])
    if field == 'hash': parent['plan_sha256'] = 'wrong'
    if field == 'subset': parent['requested_eval_count'] = 4
    if field == 'noise': parent['tasks'][0]['noise_index_start'] += 1
    if field == 'duplicate': parent['tasks'][2:4] = copy.deepcopy(parent['tasks'][0:2])
    if field != 'hash': parent['plan_sha256'] = _digest({k:v for k,v in parent.items() if k!='plan_sha256'})
    with pytest.raises(ValueError):
        partition_evaluation_plan(parent, 0, 2)


def test_complete_merge_reaudits_every_shard_and_uses_parent_order(tmp_path, evidence):
    roots = [make_shard(tmp_path, evidence, i) for i in range(2)]
    result = merge_evaluation_shards(evidence['parent'], roots[::-1], tmp_path/'merged',
        expected_checkpoint_sha256=sha256_file(evidence['checkpoint']))
    assert result['status'] == 'passed' and result['completed_episode_count'] == 8
    assert [row['task_id'] for row in result['episode_manifests']] == [task['task_id'] for task in evidence['parent']['tasks']]
    episodes = [json.loads((tmp_path/'merged'/row['path']).read_text()) for row in result['episode_manifests']]
    assert result['aggregate'] == aggregate_evaluation(episodes)
    assert result['aggregate']['overall']['executed_control_steps'] == 16
    assert json.loads((tmp_path/'merged'/'independent_audit.json').read_text())['status'] == 'passed'
    assert (tmp_path/'merged'/'completion.json').is_file()
    for row in result['episode_manifests']:
        assert sha256_file(tmp_path/'merged'/row['path']) == row['sha256'] == sha256_file(row['source_episode_path'])
    with pytest.raises(ValueError, match='must be new'):
        merge_evaluation_shards(evidence['parent'], roots, tmp_path/'merged')


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'episode', 'completion', 'parent', 'checkpoint_content', 'unexpected_checkpoint'])
def test_merge_fails_closed_before_publication(tmp_path, evidence, fault):
    roots = [make_shard(tmp_path, evidence, i) for i in range(2)]
    kwargs = {}
    if fault == 'missing': roots = roots[:1]
    if fault == 'duplicate': roots = [roots[0], roots[0]]
    if fault == 'episode':
        path = next((roots[0]/'sessions'/'session'/'evaluation_report'/'episodes').glob('*.json'))
        value=json.loads(path.read_text()); value['reward_sum'] += 1; write(path,value)
    if fault == 'completion': (roots[0]/'sessions'/'session'/'completion.json').unlink()
    if fault == 'parent':
        value=json.loads((roots[0]/'parent_plan.json').read_text()); value['tasks'][0]['seed'] += 1
        write(roots[0]/'parent_plan.json',value)
    if fault == 'checkpoint_content': evidence['checkpoint'].write_bytes(b'changed')
    if fault == 'unexpected_checkpoint': kwargs['expected_checkpoint_sha256']='0'*64
    with pytest.raises((ValueError,FileNotFoundError)):
        merge_evaluation_shards(evidence['parent'], roots, tmp_path/'merged', **kwargs)
    assert not (tmp_path/'merged').exists()


@pytest.mark.parametrize('fault', ['checkpoint', 'identity', 'implementation'])
def test_individually_valid_shards_must_share_checkpoint_training_and_tool_identity(tmp_path, evidence, fault):
    roots = [make_shard(tmp_path, evidence, 0)]
    kwargs={}
    if fault == 'checkpoint':
        other=tmp_path/'other.pt';other.write_bytes(b'other checkpoint');kwargs['checkpoint']=other
    if fault == 'identity':
        other=copy.deepcopy(evidence['identity']);other['training_contract']={'test':'changed'};kwargs['identity']=other
    if fault == 'implementation':
        other_root=tmp_path/'other_source'
        for name in EXTENSION_FILES:
            path=other_root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# 不同实现\n')
        kwargs['extension']=extension_provenance(other_root)
    roots.append(make_shard(tmp_path,evidence,1,**kwargs))
    with pytest.raises(ValueError,match='Shards changed'):
        merge_evaluation_shards(evidence['parent'],roots,tmp_path/'merged')


def test_extension_is_separate_and_detects_live_file_changes(evidence):
    snapshot=evidence['extension']
    assert snapshot['included_in_original_checkpoint_identity'] is False
    assert verify_extension_provenance(snapshot)['unchanged']
    path=Path(snapshot['files'][0]['path'])
    path.write_text('# 修改后的工具\n')
    assert not verify_extension_provenance(snapshot)['unchanged']


def test_combined_budget_cannot_multiply_the_original_full_run_limit(tmp_path, evidence):
    roots = [make_shard(tmp_path, evidence, i, control_limit=10) for i in range(2)]
    with pytest.raises(ValueError, match='Combined evaluation exceeds'):
        merge_evaluation_shards(evidence['parent'], roots, tmp_path/'merged')
    assert not (tmp_path/'merged').exists()
