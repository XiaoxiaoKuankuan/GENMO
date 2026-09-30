"""第十步完整验证集的独立只读评估分片入口。

每个进程显式加载同一个完整训练checkpoint，先用原Stage10入口生成的身份核验
Actor、Critic、优化器、音乐采样器和全部RNG，再按配对样本索引模分片数分配任务，
同一样本的所有seed始终在同一分片。分片只改变任务分工，不修改训练配置、原源码
身份、概率核、任务ID或噪声种子。新driver和分片模块单独记录扩展源码SHA，以便
在训练仍运行时增加评估工具而不改变其已冻结的源文件清单。

每片拥有独立RunManager、磁盘预算、GMT worker及先落盘再ACK的SQLite执行日志。
完整数据审计、模型/源文件/资产不变、真实执行计数和ACK水位全部通过，且worker
正常关闭、session_end指标与completion落盘后，才允许发布shard_manifest供严格
合并器读取。SIGINT/SIGTERM只在episode边界中断；未完成的分片明确为incomplete，
不发布成功清单。这里没有训练或优化步骤，不修改任何已有正式证据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from tools import train_closedloop_stage10 as training
from gem.closedloop.dppo.evaluation_shards import (
    extension_provenance, partition_evaluation_plan, publish_shard_manifest,
    verify_extension_provenance,
)


class EvaluationInterrupted(RuntimeError):
    """信号请求在任务边界结束，保留已有episode但不冒称分片完成。"""


def verify_execution_journal(path, frozen):
    """只读逐条检查落盘回复；不接入worker socket，不重新执行任何物理动作。"""
    watermarks = frozen['execution_journal']
    training._assert_frozen(frozen)
    session_id, sequences = watermarks['backend_session_id'], []
    controls = physics = advances = 0
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        for identity, digest, payload in connection.execute('SELECT identity, sha256, payload FROM replies'):
            if hashlib.sha256(payload.encode()).hexdigest() != digest:
                raise ValueError('Durable execution payload SHA mismatch')
            reply = json.loads(payload)
            sequence = reply['mutation_seq']
            if (reply['backend_session_id'] != session_id or type(sequence) is not int
                    or json.loads(identity) != [session_id, sequence]):
                raise ValueError('Durable execution identity differs from the active worker')
            sequences.append(sequence)
            if reply['operation'] != 'advance':
                continue
            if not reply['ok']:
                raise ValueError('An advance failed without complete execution evidence')
            result = reply['result']
            count, rows = result['executed_control_steps'], result['trace']
            if (type(count) is not int or count < 0 or len(rows) != count
                    or result['executed_physics_steps'] != count * 4
                    or result.get('physics_count_exact') is not True
                    or result.get('transition_valid') is not True
                    or result['control_tick_end'] - result['control_tick_begin'] != count * 12):
                raise ValueError('Durable execution trace/control/physics evidence is incomplete')
            for index, row in enumerate(rows):
                if (row['episode_id'] != result['episode_id']
                        or row['tick'] != result['control_tick_begin'] + (index + 1) * 12):
                    raise ValueError('Durable physical trace is not contiguous')
            advances += 1
            controls += count
            physics += result['executed_physics_steps']
    finally:
        connection.close()
    sequences.sort()
    if (sequences != list(range(1, watermarks['executed_seq'] + 1))
            or watermarks.get('outstanding_seq') is not None):
        raise ValueError('Durable execution journal has missing or unacknowledged mutations')
    return dict(complete=True, backend_session_id=session_id, mutation_count=len(sequences),
        executed_seq=watermarks['executed_seq'], acked_seq=watermarks['acked_seq'],
        advance_calls=advances, executed_control_steps=controls, executed_physics_steps=physics)


def _models(config, catalog, check, provenance, data_audit, checkpoint):
    """原训练身份保持完全相等；构造优化器只为严格恢复检查，不执行optimizer.step。"""
    s, stage = config['stage9'], config['stage10']
    actor, train_config, loading = training.load_actor(config)
    critic = training.UpperCritic(qpos_mean=actor.endecoder.mean, qpos_std=actor.endecoder.std,
        proprio_scales=tuple(train_config.model.proprio_scales)).to(config['runtime']['genmo_device'])
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=s['actor_lr'], weight_decay=0.)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=s['critic_lr'], weight_decay=0.)
    sampler = training.FullMusicSampler(catalog, split='train', seed=stage['seed'],
        window_seconds=s['episode_seconds'], random_start=stage['dataset']['random_start'],
        source_probabilities=stage['dataset']['source_probabilities'])
    bc = training.SupervisedAnchor(config, actor, train_config)
    generator = torch.Generator().manual_seed(stage['seed'] + 2002)
    expected = training.identity(config, check, provenance, catalog, data_audit, actor)
    checkpoint_sha256 = training.sha256_file(checkpoint)
    state = training.load_checkpoint(checkpoint, actor=actor, critic=critic,
        actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, identity=expected,
        samplers=dict(music=sampler, bc=bc), generators=dict(critic=generator))
    if training.sha256_file(checkpoint) != checkpoint_sha256:
        raise RuntimeError('Evaluated checkpoint changed while loading full state')
    if any(group['lr'] != state['selected_actor_lr'] for group in actor_optimizer.param_groups):
        raise RuntimeError('Restored Actor optimizer learning rate differs from checkpoint state')
    policy = training.DPPODiffusionPolicy(actor, steps=s['denoising_steps'], eta=s['eta'],
        std_floor=s['std_floor'], guidance_scale=s['guidance_scale'])
    restored = dict(checkpoint=str(checkpoint), sha256=checkpoint_sha256,
        checkpoint_unchanged_during_load=True,
        restored_full_state=True, initial_iteration=state['iteration'], old_buffer_discarded=True,
        actor_optimizer_lrs=[g['lr'] for g in actor_optimizer.param_groups],
        critic_optimizer_lrs=[g['lr'] for g in critic_optimizer.param_groups], training_resume=False)
    # 评估不使用已核验的优化器/监督支路，及时释放其显存与Python对象。
    del actor_optimizer, critic_optimizer, sampler, bc, generator
    return actor, critic, policy, state, expected, loading, restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/closedloop/stage10_prepare_server1.yaml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-count', type=int, required=True)
    parser.add_argument('--eval-split', choices=('val', 'test'))
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error('Require 0 <= shard-index < shard-count')
    checkpoint = args.checkpoint.resolve(strict=True)
    config = training.configuration(args.config)
    stage, storage = config['stage10'], config['stage10']['storage']
    output = args.output_dir.resolve()
    manager = training.RunManager(output, min_free_bytes=storage['min_free_bytes'],
                                  max_run_bytes=storage['max_run_bytes'])
    session = output/'sessions'/manager.session_id
    workers = journal = backend = provenance = check = extension = budget = None
    parent_plan = plan = expected = result = None
    stop = training.StopSignal()
    code = 0
    report = dict(schema=training.VERSION, mode='eval', status='running', session_id=manager.session_id,
        iterations=[], evaluations=[], shard_index=args.shard_index, shard_count=args.shard_count,
        initialization=dict(mode='explicit_stage10_full_checkpoint'), resume=None)
    try:
        session.mkdir(parents=True, exist_ok=False)
        config['stage9']['run_id'], config['output_root'] = manager.run_id, str(output)
        config_path = session/'resolved_config.yaml'
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        budget = manager.budget(stage['limits'])
        stop.install()
        random.seed(stage['seed']); np.random.seed(stage['seed']); torch.manual_seed(stage['seed'])
        torch.set_num_threads(config['runtime']['torch_threads'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        check = training.runtime_preflight(config, check_gpu=True)
        training.atomic_json(session/'preflight.json', check)
        if not check['ready']:
            raise RuntimeError('Runtime assets did not pass preflight')
        catalog = training.FullMusicCatalog(config['paths']['data_root'])
        audited = 0
        def data_progress(record):
            nonlocal audited
            audited += 1
            if audited % 500 == 0:
                print(f'[DATA_AUDIT] {audited} {record["dataset"]}/{record["split"]}', flush=True)
        data_audit = catalog.audit_files(data_progress, require_audio=stage['dataset']['require_audio'])
        training.atomic_json(session/'data_audit.json', data_audit)
        report['data_audit'] = training._relative(session/'data_audit.json', output)
        provenance = training._sources(config, check)
        extension = extension_provenance()
        training.atomic_json(session/'source_identity.json', provenance)
        training.atomic_json(session/'extension_source_identity.json', extension)
        actor, critic, policy, state, expected, loading, restored = _models(
            config, catalog, check, provenance, data_audit, checkpoint)
        training.atomic_json(session/'actor_loading.json', loading)
        report.update(resume=restored, initial_iteration=state['iteration'], final_iteration=state['iteration'],
                      extension_source_provenance=extension)
        training.atomic_json(output/'run.json', dict(schema='genmo.closedloop.stage10.run.v1', identity=expected,
            config_sha256=training.sha256_file(args.config), dataset_identity=catalog.identity,
            initialization=report['initialization'], mode='eval'))
        evaluation = stage['evaluation']
        parent_plan = training.build_evaluation_tasks(catalog, split=args.eval_split or evaluation['split'],
            eval_count='all', seeds=evaluation['seeds'], selection_seed=evaluation['selection_seed'],
            start_mode=evaluation['start_mode'], episode_seconds=evaluation['episode_seconds'])
        plan = partition_evaluation_plan(parent_plan, args.shard_index, args.shard_count)
        training.atomic_json(output/'parent_plan.json', parent_plan)
        if stop.stop_requested:
            raise EvaluationInterrupted('Stop requested before physical evaluation')
        workers = training.Workers(config, session)
        socket = Path(workers.temp.name)/'gmt.sock'
        client = workers.start('gmt', [config['paths']['isaac_python'], '-B',
            str(Path(config['paths']['gmt_repo'])/'scripts/rsl_rl/serve_frozen_gmt.py'),
            '--config', str(config_path), '--socket', str(socket), '--headless'], config['paths']['gmt_repo'], socket)
        journal = training.GuardedStepJournal(session/'execution_journal.sqlite', manager.disk_guard)
        backend = training.AcknowledgedBackend(client, journal, socket_path=socket,
                                                timeout_s=config['runtime']['rpc_timeout_s'])
        builder = training.OnlineConditionBuilder(training.BumiMotionFeatureCodec(
            training.BumiKinematics(config['paths']['kinematics'])))
        env = training.UpperEnvironment(config, backend, builder, policy, budget, session/'evaluation')
        env.disk_guard = manager.disk_guard
        env.policy_version, env.iteration = state['policy_version'], state['iteration']
        env.episode_count, env.attempt = state['episode_count'], state['attempt']
        env.latency_budget_s = state['latency_budget_s']
        restored['new_backend_session_id'] = backend.session_id
        manager.append_metrics(dict(event='session_start', mode='eval', initial_iteration=state['iteration'],
            checkpoint_sha256=restored['sha256'], shard_index=args.shard_index, shard_count=args.shard_count))
        def progress(item):
            if item['event'] == 'evaluation_episode_start':
                if stop.stop_requested:
                    raise EvaluationInterrupted('Stop requested at the next episode boundary')
                manager.check_disk(refresh=True)
            print('[EVAL_SHARD] ' + str(item), flush=True)
        before = dict(actor=training._fingerprint(actor), critic=training._fingerprint(critic))
        result = training.evaluate_policy(env, policy, plan, session/'evaluation_report', catalog=catalog,
            episode_seconds=evaluation['episode_seconds'],
            actor_identity=dict(checkpoint=str(checkpoint), sha256=restored['sha256'], iteration=state['iteration']),
            frozen_modules={'critic': critic}, progress=progress)
        if before != dict(actor=training._fingerprint(actor), critic=training._fingerprint(critic)):
            raise RuntimeError('Read-only evaluation changed Actor or Critic')
        report['gmt_frozen'] = backend.call('verify_frozen')
        training._assert_frozen(report['gmt_frozen'])
        report['evaluations'].append(training._relative(session/'evaluation_report/report.json', output))
        report.update(status='passed', evaluation_status=result['status'], full_dataset_checked=True,
                      stopped_on_signal=stop.stop_requested)
    except BaseException as exc:
        code = 130 if isinstance(exc, EvaluationInterrupted) else 1
        report.update(status='incomplete' if isinstance(exc, EvaluationInterrupted) else 'failed',
            error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        try:
            if workers is not None:
                if backend is not None:
                    workers.entries[-1]['client'] = backend.client
                workers.close()
                report['worker_shutdown'] = workers.shutdown
                gmt = workers.shutdown.get('gmt', {})
                if (gmt.get('close_error') or gmt.get('forced_shutdown') or gmt.get('process_exit_code') != 0
                        or gmt.get('policy_unchanged') is not True or gmt.get('runtime_parameters_unchanged') is not True):
                    report['status'], code = 'failed', 1
            if journal is not None:
                journal.close()
            if code == 0:
                report['execution_journal_integrity'] = verify_execution_journal(
                    session/'execution_journal.sqlite', report['gmt_frozen'])
                used = budget.state_dict()['used']
                evidence = report['execution_journal_integrity']
                if (used['control_steps'] != evidence['executed_control_steps']
                        or used['physics_steps'] != evidence['executed_physics_steps']
                        or used['accepted_iterations'] != 0 or used['optimizer_attempts'] != 0):
                    raise RuntimeError('Evaluation budget differs from durable physical execution')
            if provenance is not None:
                report['source_unchanged'] = training.verify_source_provenance(provenance)
                if not report['source_unchanged']['unchanged']:
                    report['status'], code = 'failed', 1
            if extension is not None:
                report['extension_source_unchanged'] = verify_extension_provenance(extension)
                if not report['extension_source_unchanged']['unchanged']:
                    report['status'], code = 'failed', 1
            if check is not None:
                report['original_assets_unchanged'] = all(training.sha256_file(config['paths'][key]) == digest
                    for key, digest in check['asset_sha256'].items())
                if not report['original_assets_unchanged']:
                    report['status'], code = 'failed', 1
            if report['resume'] is not None:
                report['evaluated_checkpoint_unchanged'] = (checkpoint.is_file()
                    and training.sha256_file(checkpoint) == report['resume']['sha256'])
                if not report['evaluated_checkpoint_unchanged']:
                    report['status'], code = 'failed', 1
            report.update(exit_code=code, budget=None if budget is None else budget.state_dict())
            training.atomic_json(session/'summary.json', report)
            manager.append_metrics(dict(event='session_end', status=report['status'], exit_code=code,
                                         summary=training._relative(session/'summary.json', output)))
            training.atomic_json(session/'completion.json', dict(
                schema='genmo.closedloop.stage10.session_completion.v1',
                summary_sha256=training.sha256_file(session/'summary.json'), status=report['status'], exit_code=code))
            if code == 0:
                publish_shard_manifest(output, parent_plan=parent_plan, shard_plan=plan,
                    evaluation_report=session/'evaluation_report/report.json', session_summary=session/'summary.json',
                    completion=session/'completion.json', training_identity=expected, extension=extension)
        finally:
            stop.restore()
            manager.close()
    print(json.dumps(dict(status=report['status'], output=str(output), exit_code=code)), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
