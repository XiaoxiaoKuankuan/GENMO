"""第十步完整评估计划的确定性分片、独立实现身份和只读合并验收。

这是现有串行评估的新增外围工具，不修改 Actor、训练入口、checkpoint 身份或正在
运行的已哈希源码。父计划必须来自完整 val/test 池并包含全部配对样本和多个 seed；
按父计划中样本的索引模分片数分配，同一样本的全部 seed 始终放在同一片。子任务的
task_id、音乐起点、噪声种子和配对数据保持逐字一致，子计划明确声明父计划和分片。

每片仍调用原 evaluate_policy、执行日志、磁盘预算和 session completion，再由原
CPU audit_run 独立复核。分片实现和复用审计文件单独记录 SHA，绝不冒称旧 checkpoint 已包含
这些工具。合并要求所有分片完成、原 checkpoint 和训练身份完全相同、工具实现相同，
并证明所有父 task_id 恰好出现一次。统计从逐 episode 文件重新计算，不平均各片均值。

合并目录按父任务顺序硬链接原 episode JSON（跨文件系统时复制），保留原文件 SHA
和片内 episode_index，不复制原始去噪链或物理 journal。正式 merged report 仍可由
原 audit_evaluation 核查，同时保留每片的独立审计和原始文件位置。所有发布均拒绝
覆盖既有证据。模块 CLI 只做 CPU 读文件合并，不加载策略或启动仿真。
"""
from __future__ import annotations

import argparse
import copy
import errno
import json
import os
from pathlib import Path
import shutil

from gem.closedloop.dppo.evaluation import VERSION as EVALUATION_VERSION, _digest, _publish_json, aggregate_evaluation
from gem.closedloop.evaluation_music import sha256_file
from gem.closedloop.frozen_actor import stable_noise_seed
from tools.eval.audit_closedloop_stage10 import audit_run, audit_data, audit_evaluation, read_json, require


VERSION = 'genmo.closedloop.stage10.evaluation_shard.v1'
MERGE_VERSION = 'genmo.closedloop.stage10.evaluation_merge.v1'
EXTENSION_FILES = ('gem/closedloop/dppo/evaluation_shards.py', 'tools/eval/run_closedloop_stage10_shard.py',
                   'tools/eval/audit_closedloop_stage10.py')


def _full_plan_samples(plan):
    require(plan.get('version') == EVALUATION_VERSION, 'Unsupported parent evaluation plan')
    require(plan.get('plan_sha256') == _digest({k: v for k, v in plan.items() if k != 'plan_sha256'}),
            'Parent plan SHA mismatch')
    require(plan.get('split') in ('val', 'test') and plan.get('training_buffer_allowed') is False,
            'Parent plan must be held-out and read-only')
    require(plan.get('requested_eval_count') == 'all' and 'partition' not in plan,
            'Partition requires the original full all-sample parent plan')
    seeds = plan['seeds']
    require(len(seeds) >= 2 and len(set(seeds)) == len(seeds), 'Full acceptance requires distinct multiple seeds')
    tasks, samples, seen = plan['tasks'], [], set()
    require(len(tasks) == plan['task_count'] == plan['complete_pool_count'] * len(seeds)
            and plan['selected_sample_count'] == plan['complete_pool_count'], 'Parent full Cartesian count differs')
    for offset in range(0, len(tasks), len(seeds)):
        group = tasks[offset:offset+len(seeds)]
        first = group[0]
        key = (first['dataset'], first['sample_id'])
        require(key not in seen, 'Parent plan duplicates a paired sample')
        seen.add(key)
        samples.append(first['sample'])
        for seed, task in zip(seeds, group):
            require((task['dataset'], task['sample_id']) == key and task['seed'] == seed
                    and task['sample'] == first['sample'] and task['split'] == plan['split'],
                    'Parent sample seed order or split differs')
            identity = {name: task[name] for name in ('dataset', 'sample_id', 'split', 'seed', 'music_start_frame')}
            require(task['task_id'] == _digest(identity)[:24]
                    and task['noise_index_start'] == stable_noise_seed(seed, _digest(identity)),
                    'Parent task identity or noise seed differs')
    require(len({task['task_id'] for task in tasks}) == len(tasks), 'Parent task IDs collide')
    require(_digest(samples) == plan['complete_pool_identity_sha256'], 'Parent full pool identity differs')
    return samples


def partition_evaluation_plan(parent_plan, shard_index, shard_count):
    """完整父计划按配对样本索引模 N 分片，原任务身份和噪声保持不变。"""
    samples = _full_plan_samples(parent_plan)
    require(type(shard_count) is int and 1 <= shard_count <= len(samples), 'Invalid shard count')
    require(type(shard_index) is int and 0 <= shard_index < shard_count, 'Invalid shard index')
    indices = list(range(shard_index, len(samples), shard_count))
    selected = [samples[index] for index in indices]
    seeds = parent_plan['seeds']
    result = copy.deepcopy(parent_plan)
    result.update(requested_eval_count=len(indices), selected_sample_count=len(indices),
        selected_unique_group_count=len({sample['group_id'] for sample in selected}),
        selected_unique_audio_count=len({sample['row']['source_audio_sha256'] for sample in selected}),
        selection_rule='parent_full_plan_paired_sample_index_mod_shard_count',
        tasks=[copy.deepcopy(task) for index in indices
               for task in parent_plan['tasks'][index*len(seeds):(index+1)*len(seeds)]],
        task_count=len(indices)*len(seeds), partition=dict(schema=VERSION,
            parent_plan_sha256=parent_plan['plan_sha256'], shard_index=shard_index,
            shard_count=shard_count, parent_sample_indices=indices,
            parent_selected_sample_count=len(samples), original_task_identity_and_noise_preserved=True))
    result['plan_sha256'] = _digest({k: v for k, v in result.items() if k != 'plan_sha256'})
    return result


def extension_provenance(repository_root=None):
    """分片及复用审计工具独立 SHA 清单，不改写已有 checkpoint 的训练源码身份。"""
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[3]
    files = []
    for name in EXTENSION_FILES:
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'Missing or escaping extension source')
        files.append(dict(relative_path=name, path=str(path), sha256=sha256_file(path)))
    return dict(schema=VERSION+'.source', files=files,
                source_manifest_sha256=_digest([(item['relative_path'], item['sha256']) for item in files]),
                included_in_original_checkpoint_identity=False)


def verify_extension_provenance(snapshot):
    require(snapshot.get('schema') == VERSION+'.source', 'Wrong extension source schema')
    require(snapshot.get('included_in_original_checkpoint_identity') is False, 'Extension must remain separately identified')
    files = snapshot['files']
    require([item['relative_path'] for item in files] == list(EXTENSION_FILES), 'Incomplete extension source list')
    require(snapshot['source_manifest_sha256'] == _digest([(item['relative_path'], item['sha256']) for item in files]),
            'Extension source manifest SHA differs')
    changed = [item['relative_path'] for item in files
               if not Path(item['path']).is_file() or sha256_file(item['path']) != item['sha256']]
    return dict(unchanged=not changed, changed_files=changed, source_manifest_sha256=snapshot['source_manifest_sha256'])


def _inside(root, path):
    value = Path(path)
    value = value.resolve() if value.is_absolute() else (root/value).resolve()
    require(value.is_relative_to(root) and value.is_file(), 'Shard evidence missing or outside its run')
    return value


def _reference(root, path):
    path = _inside(root, path)
    return dict(path=str(path.relative_to(root)), sha256=sha256_file(path))


def _read_reference(root, value):
    path = _inside(root, value['path'])
    require(sha256_file(path) == value['sha256'], 'Shard evidence file SHA differs')
    return path, read_json(path)


def publish_shard_manifest(run_dir, *, parent_plan, shard_plan, evaluation_report,
                           session_summary, completion, training_identity, extension):
    """全部运行证据成功完成后才发布分片清单；未完成或故障片不能混入合并结果。"""
    root = Path(run_dir).resolve()
    part = shard_plan['partition']
    require(shard_plan == partition_evaluation_plan(parent_plan, part['shard_index'], part['shard_count']),
            'Child plan differs from its declared deterministic partition')
    report_path, summary_path, completion_path = (_inside(root, path) for path in
                                                  (evaluation_report, session_summary, completion))
    report, summary, done = (read_json(path) for path in (report_path, summary_path, completion_path))
    require(report['status'] == summary['status'] == done['status'] == 'passed'
            and summary['exit_code'] == done['exit_code'] == 0, 'Cannot publish an incomplete shard')
    require(done['summary_sha256'] == sha256_file(summary_path), 'Completion does not bind successful summary')
    require(completion_path.parent == summary_path.parent
            and str(report_path.relative_to(root)) in summary['evaluations'],
            'Published report is not bound to this completed session')
    require(report['plan_sha256'] == shard_plan['plan_sha256'], 'Evaluated child plan SHA differs')
    selection = _inside(root, report_path.parent/report['selection_path'])
    require(read_json(selection) == shard_plan, 'Evaluated task plan differs from the declared shard')
    require(read_json(root/'run.json')['identity'] == training_identity, 'Run training identity differs')
    require(summary.get('extension_source_provenance') == extension
            and summary.get('extension_source_unchanged', {}).get('unchanged') is True,
            'Completed session did not verify the declared extension')
    require(summary.get('evaluated_checkpoint_unchanged') is True,
            'Completed session did not verify its evaluated checkpoint file')
    require(verify_extension_provenance(extension)['unchanged'], 'Extension changed before shard publication')
    audited = audit_run(root)
    require(audited['status'] == 'passed', 'Original independent audit did not pass for this shard')
    require(verify_extension_provenance(extension)['unchanged'], 'Extension changed during independent audit')
    if (root/'parent_plan.json').exists():
        require(read_json(root/'parent_plan.json') == parent_plan, 'Stored parent plan differs')
    else:
        _publish_json(root/'parent_plan.json', parent_plan)
    value = dict(schema=VERSION, status='passed', shard_index=part['shard_index'], shard_count=part['shard_count'],
        parent_plan=_reference(root, root/'parent_plan.json'), parent_plan_sha256=parent_plan['plan_sha256'],
        child_plan_sha256=shard_plan['plan_sha256'], evaluation_report=_reference(root, report_path),
        session_summary=_reference(root, summary_path), completion=_reference(root, completion_path),
        training_identity=copy.deepcopy(training_identity), checkpoint_sha256=report['actor_identity']['sha256'],
        extension_provenance=copy.deepcopy(extension), independent_shard_audit=audited)
    _publish_json(root/'shard_manifest.json', value)
    return root/'shard_manifest.json'


def merge_evaluation_shards(parent_plan, shard_dirs, output_dir, *, expected_checkpoint_sha256=None):
    """只读验收完整分片集合，按原父计划任务顺序发布全量重算结果。"""
    _full_plan_samples(parent_plan)
    roots = [Path(value).resolve() for value in shard_dirs]
    require(roots and len(set(roots)) == len(roots), 'Shard directories must be distinct and nonempty')
    output = Path(output_dir).resolve()
    require(not output.exists(), 'Merged evidence directory must be new')
    gathered, source_audits, manifests, indices = {}, [], [], set()
    shared, first_report, lookup, checkpoint_hash_cache = None, None, None, {}
    combined_budget = None
    for root in roots:
        manifest = read_json(root/'shard_manifest.json')
        require(manifest.get('schema') == VERSION and manifest.get('status') == 'passed', 'Invalid shard publication')
        index, count = manifest['shard_index'], manifest['shard_count']
        require(count == len(roots) and index not in indices, 'Missing or duplicate shard index')
        indices.add(index)
        _, stored_parent = _read_reference(root, manifest['parent_plan'])
        require(stored_parent == parent_plan and manifest['parent_plan_sha256'] == parent_plan['plan_sha256'],
                'Shards do not share the exact original parent plan')
        report_path, report = _read_reference(root, manifest['evaluation_report'])
        summary_path, summary = _read_reference(root, manifest['session_summary'])
        _, completion = _read_reference(root, manifest['completion'])
        require(completion['summary_sha256'] == sha256_file(summary_path), 'Shard completion summary SHA differs')
        child = read_json(_inside(root, report_path.parent/report['selection_path']))
        require(child == partition_evaluation_plan(parent_plan, index, count)
                and manifest['child_plan_sha256'] == child['plan_sha256'], 'Shard task partition differs')
        extension = manifest['extension_provenance']
        require(verify_extension_provenance(extension)['unchanged'], 'Shard extension source changed')
        run_identity = read_json(root/'run.json')['identity']
        require(run_identity == manifest['training_identity'], 'Published shard training identity differs')
        require(run_identity['dataset'] == parent_plan['catalog_identity'], 'Parent data differs from checkpoint training identity')
        require(summary.get('extension_source_provenance') == extension
                and summary.get('extension_source_unchanged', {}).get('unchanged') is True,
                'Session extension evidence differs from shard publication')
        require(summary.get('evaluated_checkpoint_unchanged') is True,
                'Shard did not verify its evaluated checkpoint file')
        common = dict(training_identity=run_identity, checkpoint_sha256=manifest['checkpoint_sha256'],
            extension_sha256=extension['source_manifest_sha256'],
            network_fingerprints=report['network_fingerprints_before'], deterministic=report['deterministic'],
            episode_seconds_limit=report['episode_seconds_limit'], budget_limits=summary['budget']['limits'])
        if shared is None:
            shared, first_report = common, report
        require(common == shared, 'Shards changed checkpoint, training identity, extension, network or evaluation semantics')
        require(report['actor_identity']['sha256'] == shared['checkpoint_sha256'], 'Shard report checkpoint SHA differs')
        if expected_checkpoint_sha256 is not None:
            require(shared['checkpoint_sha256'] == expected_checkpoint_sha256, 'Unexpected checkpoint selected for merge')
        checkpoint = Path(report['actor_identity']['checkpoint']).resolve()
        if checkpoint not in checkpoint_hash_cache:
            checkpoint_hash_cache[checkpoint] = sha256_file(checkpoint)
        require(checkpoint_hash_cache[checkpoint] == shared['checkpoint_sha256'], 'Evaluated checkpoint file changed')
        audited = audit_run(root)
        require(audited['status'] == 'passed', 'Independent shard audit failed during merge')
        if combined_budget is None:
            combined_budget = dict(limits=copy.deepcopy(summary['budget']['limits']),
                                   used=dict.fromkeys(summary['budget']['used'], 0))
        for name, value in summary['budget']['used'].items():
            combined_budget['used'][name] += value
            require(combined_budget['used'][name] <= combined_budget['limits'][name],
                    'Combined evaluation exceeds the original full-run budget')
        require(combined_budget['used']['accepted_iterations'] == combined_budget['used']['optimizer_attempts'] == 0,
                'Evaluation shards consumed optimizer/update budget')
        source_audits.append(audited)
        data_path = _inside(root, summary['data_audit'])
        current_lookup = audit_data(read_json(data_path))
        if lookup is None:
            lookup = current_lookup
        for reference, task in zip(report['episode_manifests'], child['tasks']):
            episode_path = _inside(root, report_path.parent/reference['path'])
            require(task['task_id'] not in gathered, 'A parent task was executed by more than one shard')
            episode = read_json(episode_path)
            require(episode.get('music_start_frame') == task['music_start_frame']
                    and episode.get('noise_index_start') == task['noise_index_start'], 'Shard changed task start or noise seed')
            gathered[task['task_id']] = (episode_path, reference['sha256'], episode, index)
        manifests.append(dict(run_dir=str(root), manifest_sha256=sha256_file(root/'shard_manifest.json'),
                              shard_index=index, completed_episodes=len(child['tasks'])))
    require(indices == set(range(len(roots))), 'Incomplete shard index set')
    require(set(gathered) == {task['task_id'] for task in parent_plan['tasks']}, 'Merged tasks are not the exact full parent set')
    expected_samples = {(task['dataset'], parent_plan['split'], task['sample_id']) for task in parent_plan['tasks']}
    require(expected_samples == {key for key in lookup if key[1] == parent_plan['split']}, 'Parent plan omitted audited held-out samples')
    require(verify_extension_provenance(manifest['extension_provenance'])['unchanged'], 'Extension changed during merge audit')
    output.mkdir(parents=True, exist_ok=False)
    episode_dir = output/'episodes'
    episode_dir.mkdir()
    episodes, references = [], []
    for position, task in enumerate(parent_plan['tasks']):
        original, digest, episode, shard_index = gathered[task['task_id']]
        target = episode_dir/f'{position:06d}.json'
        try:
            os.link(original, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            with original.open('rb') as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        require(sha256_file(target) == digest, 'Episode changed while merging')
        episodes.append(episode)
        references.append(dict(path=str(target.relative_to(output)), sha256=digest, task_id=task['task_id'],
            transition_valid=True, parent_task_index=position, source_shard_index=shard_index,
            source_episode_path=str(original), source_episode_index=episode['episode_index']))
    _publish_json(output/'selection.json', parent_plan)
    report = copy.deepcopy(first_report)
    report.update(plan_sha256=parent_plan['plan_sha256'], selection_path='selection.json',
        requested_task_count=len(episodes), completed_episode_count=len(episodes), episode_manifests=references,
        aggregate=aggregate_evaluation(episodes), merge_schema=MERGE_VERSION,
        source_shards=sorted(manifests, key=lambda item: item['shard_index']),
        episode_index_scope='original_shard_local; episode_manifests_order_is_original_parent_order',
        execution_evidence='independent acknowledged worker journals and original raw sample paths in each shard',
        merge_assertions=dict(exact_full_parent_task_set=True, duplicate_tasks=0,
            original_task_identity_and_noise_preserved=True, all_shards_independently_audited=True),
        extension_provenance=manifest['extension_provenance'], training_identity=shared['training_identity'],
        combined_budget=combined_budget)
    _publish_json(output/'report.json', report)
    merged_audit = audit_evaluation(output, output/'report.json', lookup, parent_plan['catalog_identity'])
    _publish_json(output/'independent_audit.json', dict(schema=MERGE_VERSION, status='passed',
        merge=merged_audit, shards=source_audits, report_sha256=sha256_file(output/'report.json')))
    _publish_json(output/'completion.json', dict(schema=MERGE_VERSION+'.completion', status='passed',
        report_sha256=sha256_file(output/'report.json'), audit_sha256=sha256_file(output/'independent_audit.json')))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', required=True, type=Path)
    parser.add_argument('--shard-dir', required=True, action='append', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--checkpoint-sha256')
    args = parser.parse_args(argv)
    result = merge_evaluation_shards(read_json(args.parent_plan), args.shard_dir, args.output_dir,
                                      expected_checkpoint_sha256=args.checkpoint_sha256)
    print(json.dumps(dict(status=result['status'], episodes=result['completed_episode_count'],
                          output=str(args.output_dir.resolve())), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
