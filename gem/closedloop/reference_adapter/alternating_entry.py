"""Server-1单GPU正式SM交替训练入口；八组之间没有模型、Replay或通信共享。

数据流是当前Adapter真实采集(含实际动作探索)→原SM H8世界模型十次更新→冻结
Actor/WM后的逐步BPTT→可选想象GAE Critic。100个WM-only交替循环后直接策略
更新，不加载旧Adapter，不运行Oracle/方向审计/物理候选回滚。GPU配置决定H8、
H16、H32或H64及动作探索消融；WM仍固定真实H8递归监督，十帧参考不随BPTT
增长。未经真实回报校准的Critic暂不参与训练或终端梯度。

每次有限Adam完成立即计数并记录修正传播及投影梯度；0/5/10更新固定短时对照，
20更新起执行无探索整歌validation和evaluation，只有validation选择best。评价下降只
报告。运行异常保存三个网络/优化器、Replay、RNG及约束现场；恢复保留学习状态，
真实求解器从新episode开始，不伪装隐藏物理状态续接。完整checkpoint保留最近两份
及best和真实零更新初始模型，所有编号小模型和报告永久保留。零Adapter也是best
候选，不能把比零Adapter退化的首个整歌评价模型标为最佳。
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


def normalize_extra_state(value=None):
    """补齐新增运行元数据，保留旧checkpoint的模型/优化器/计数与历史评价原值。"""
    result = dict(value or {})
    defaults = dict(step_acc=0., best=None, best_short=None, evaluations=[], completed_checkpoints=[],
                    precision_scales=None, zero_evaluations={}, zero_checkpoint=None,
                    consecutive_projection_rejections=0)
    for name, default in defaults.items():
        result.setdefault(name, default)
    return result


def validation_ranking(report):
    """只对有限且没有参考拒绝的同一评价预算比较，评价歌曲不能进入best排序。"""
    values = [report.get(k) for k in ('completion_rate', 'mean_root_position_rmse_m',
                                     'mean_full_horizon_tracking', 'mean_jerk_rms')]
    if report.get('constraint_failure_count', 0) or any(v is None or not math.isfinite(v) for v in values):
        return None
    return values[0], -values[1], values[2], -values[3]


def select_validation_best(previous, zero, candidate, zero_checkpoint, *, adapter_updates, scope):
    """先将真实零模型纳入同预算候选；相同分数保留零模型或此前best，不误标当前权重。"""
    best = previous
    zero_rank = validation_ranking(zero)
    if zero_rank is not None and (best is None or zero_rank >= tuple(best['ranking'])):
        best = dict(adapter_updates=0, ranking=zero_rank, selection_split='validation_only',
                    evaluation_scope=scope, model_identity='initial_zero_Adapter',
                    checkpoint_available=zero_checkpoint is not None,
                    path=zero_checkpoint['path'] if zero_checkpoint else None)
    ranking = validation_ranking(candidate)
    if adapter_updates > 0 and ranking is not None and (best is None or ranking > tuple(best['ranking'])):
        best = dict(adapter_updates=adapter_updates, ranking=ranking,
                    selection_split='validation_only', evaluation_scope=scope,
                    model_identity='trained_Adapter', checkpoint_available=False)
    return best


def checkpoint_keep_paths(extra):
    """零更新文件始终保留；最近两份和validation best另行保留，外部resume从不删除。"""
    keep = set(extra['completed_checkpoints'][-2:])
    for candidate in (extra.get('best'), extra.get('zero_checkpoint')):
        if candidate and candidate.get('path'):
            keep.add(candidate['path'])
    return keep


def validation_degradation(previous, zero, current, rules):
    """仅整歌validation连续明显退化才停止本组；不逐候选回滚，也不读取evaluation。"""
    if not rules or current.get('max_steps') is not None:
        return previous or dict(consecutive=0, stop=False)
    reasons = []
    for metric, limit in (('mean_root_position_rmse_m', rules['root_rmse_relative_increase']),
                          ('mean_jerk_rms', rules['jerk_relative_increase'])):
        if current[metric] is not None and zero[metric] and current[metric] > zero[metric]*(1+limit):
            reasons.append(metric)
    if current['mean_full_horizon_tracking'] < zero['mean_full_horizon_tracking']*(1-rules['tracking_relative_drop']):
        reasons.append('full_horizon_tracking')
    if current['completion_rate'] < zero['completion_rate']-rules['completion_absolute_drop']+1e-12:
        reasons.append('completion_rate')
    consecutive = (previous or {}).get('consecutive', 0)+1 if reasons else 0
    return dict(consecutive=consecutive, reasons=reasons, stop=consecutive >= rules['consecutive_full_evaluations'],
                selection_split='validation_only', rules=rules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(preliminary.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import numpy as np
    import torch
    from supermimic.networks import Critic
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.bridge.bundle import atomic_save, file_digest
    from .application import read_config, build_engine, read_tasks, seed_all, direct_reference_config
    from .vector_bank import VectorBank
    from .vector_control import VectorControl
    from .vector_collector import VectorCollector
    from .vector_replay import VectorReplay
    from .vector_learning import TangentReplay
    from .vector_entry import json_write, append_json, compile_kinematics, gpu_usage
    from .vector_evaluation import evaluate_physical
    from .paired_evaluation import paired_comparison
    from .vector_checkpoint import save_vector, load_vector
    from .precision import PrecisionObjective
    from .update_diagnostics import parameter_identity
    from .alternating_learning import (enable_wm_normalization, update_wm_normalization, wm_update,
                                      policy_update, critic_update, PolicyProjectionRejected)

    cfg = direct_reference_config(read_config(args.config))
    settings, train, vector = cfg['alternating'], cfg['training'], cfg['vector']
    if settings['method'] != 'supermimic_alternating_WM_BPTT' or train['wm_fit_window'] != 8:
        raise ValueError('正式入口只允许原SM H8监督交替算法')
    if cfg['bptt_steps'] not in (8, 16, 32, 64):
        raise ValueError('本次正式BPTT配置只使用H8/H16/H32/H64')
    if settings['critic_enabled']:
        raise ValueError('10月28日修复实验暂时关闭未经独立真实回报校准的Critic；不能使用旧想象EV放行')
    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('每组必须仅可见一张GPU；禁止DDP/跨GPU通信')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[3]), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=False)
    state = dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), phase='initializing',
        accepted_updates=0, adapter_optimizer_updates=0, wm_updates=0, critic_optimizer_updates=0,
        iterations=0, real_transitions=0, num_envs=vector['num_envs'], started_unix=time.time(),
        horizon=cfg['bptt_steps'], critic_enabled=settings['critic_enabled'], method=settings['method'],
        source=revision.stdout.strip() if revision.returncode == 0 else 'unversioned_source_snapshot', formal=True)
    runtime_source = state['source']  # 恢复旧实验后仍记录这次实际执行源码，而非覆盖历史身份。
    json_write(output/'manifest.json', dict(config=cfg, command=[sys.executable, *sys.argv], state=state.copy(),
        initialization='full_resume_same_experiment' if args.resume else 'zero_Adapter_original_pretrained_WM_no_old_policy', communication='none',
        acceptance='finite_Adam_step_not_a_physical_improvement_certificate',
        critic_status='disabled_unqualified_real_MC_anchor_and_independent_calibration_required',
        full_checkpoint_retention='initial_zero_latest_two_and_validation_best_all_numbered_weights_kept'))
    (output/'pid').write_text(str(os.getpid())+'\n')
    engine = critic = collector = replay = tangent = None
    optimizers = {}
    extra = normalize_extra_state()
    frozen = None

    def publish(phase, **fields):
        state.update(phase=phase, updated_unix=time.time(), **fields)
        if collector is not None:
            state.update(real_transitions=collector.real_transitions, real_by_role=collector.real_by_role,
                         vector_steps=collector.vector_steps, resets=collector.reset_count,
                         projection_recovered_environments=collector.projection_recoveries)
        if engine is not None:
            state['wm_updates'] = engine.dynamics.model_version
        state['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        json_write(output/'status.json', state, replace=True)

    def checkpoint(label):
        from .projection_evidence import checkpoint_projection_evidence
        checkpoint_begin = time.perf_counter()
        publish('saving_complete_checkpoint')
        path = output/f'full_{state["accepted_updates"]:06d}_{label}_{time.time_ns()}.pt'
        extra['completed_checkpoints'].append(str(path))
        initial_zero = label == 'zero_update' and state['accepted_updates'] == 0 and extra['zero_checkpoint'] is None
        if initial_zero:
            # 先写进完整checkpoint本身，使后续restore仍能找到原始零权重文件。
            extra['zero_checkpoint'] = dict(path=str(path), adapter_updates=0, wm_updates=state['wm_updates'],
                physics_resume='new_episode_with_preserved_replay', adapter_sha256=parameter_identity(control.adapter))
            for key in ('best', 'best_short'):
                if extra[key] and extra[key].get('model_identity') == 'initial_zero_Adapter':
                    extra[key].update(path=str(path), checkpoint_available=True)
        current_is_best = (extra['best'] is not None and
            extra['best']['adapter_updates'] == state['accepted_updates'] and
            extra['best'].get('model_identity') != 'initial_zero_Adapter')
        if current_is_best:
            extra['best']['path'] = str(path)
            extra['best']['checkpoint_available'] = True
        extra['action_noise_rng'] = collector.action_generator.get_state()
        extra['projection_recovery_events'] = collector.projection_recovery_events
        checkpoint_projection_evidence(extra, control)
        save_vector(path, engine, critic, optimizers, replay, tangent, collector, state.copy(), cfg,
                    extra_state=extra)
        pointer = dict(path=str(path), adapter_updates=state['accepted_updates'], wm_updates=state['wm_updates'],
                       bytes=path.stat().st_size, physics_resume='new_episode_with_preserved_replay')
        json_write(output/'latest_checkpoint.json', pointer, replace=True)
        if initial_zero:
            extra['zero_checkpoint'].update(bytes=path.stat().st_size)
            json_write(output/'zero_checkpoint.json', extra['zero_checkpoint'])
        if current_is_best:
            json_write(output/'best_checkpoint.json', dict(pointer, validation=extra['best']), replace=True)
        elif initial_zero and extra['best']:
            json_write(output/'best_checkpoint.json', dict(extra['zero_checkpoint'], validation=extra['best']), replace=True)
        # 仅清理本运行自己已原子保存、且有新完整替代的文件；不删除旧实验/外部resume。
        keep = checkpoint_keep_paths(extra)
        for old in list(extra['completed_checkpoints']):
            old_path = Path(old)
            if old not in keep and old_path.parent == output and old_path.name.startswith('full_'):
                if old_path.exists():
                    old_path.unlink()
                extra['completed_checkpoints'].remove(old)
        pointer['seconds'] = time.perf_counter()-checkpoint_begin
        append_json(output/'checkpoint_history.jsonl', pointer)
        append_json(output/'timing.jsonl', dict(phase='checkpoint', **pointer, timestamp=time.time()))
        from .training_report import safe_update_report as update_report
        update_report(output.parent)
        return path

    def collect(n):
        totals = {}
        breakdown = {}
        begin = time.perf_counter()
        collector.policy_version = state['accepted_updates']
        for _ in range(n):
            row = collector.step(replay, action_noise=settings['action_noise'], split='train', perturb=True,
                exploration_fraction=settings.get('exploration_fraction', 0.),
                exploration_std=settings.get('exploration_std', .1))
            update_wm_normalization(engine.dynamics, row)
            for name, milliseconds in row['timing'].items():
                if name.endswith('_ms'):
                    breakdown[name[:-3]+'_seconds'] = breakdown.get(name[:-3]+'_seconds', 0.)+milliseconds/1000
            values = dict(real_reward=row['reward'].mean(), real_tracking=row['raw']['track'].mean(),
                real_reference_rms=row['proposal']['residual'].square().mean().sqrt())
            values.update({'real_'+k: value.mean() for k, value in row['raw'].items()
                if isinstance(value, torch.Tensor) and value.ndim == 1 and
                (k in ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori',
                       'action_rate') or k.startswith(('absolute_', 'error_')))})
            if 'exploration_std' in row['proposal']:
                values['executed_noise_rms'] = (row['proposal']['action']-row['proposal']['policy_action']).square().mean().sqrt()
                values['large_exploration_fraction'] = (row['proposal']['exploration_std'] >= .099).float().mean()
            for key, value in values.items():
                totals[key] = totals.get(key, 0.)+value.detach()/n
        keys = list(totals)
        values = torch.stack(list(totals.values())).cpu().tolist()
        totals = dict(zip(keys, values))
        totals.update(collection_seconds=time.perf_counter()-begin, collection_vector_steps=n,
                      collection_breakdown_seconds=breakdown,
                      physical_metrics_scope='training_distribution_not_paired_policy_improvement')
        if collector.projection_recovery_events:
            append_json(output/'projection_recovery.jsonl', dict(
                adapter_updates=state['accepted_updates'], events=collector.projection_recovery_events))
            collector.projection_recovery_events.clear()
        return totals

    def evaluate():
        # Newton/Isaac上下文是进程级singleton；专用4环境在独立评价子进程创建。
        # 父训练环境完全不推进、不reset，物理episode和训练RNG均保持因果连续。
        from .checkpoint import shared_cpu_state
        begin = time.perf_counter()
        full = state['accepted_updates'] >= settings.get('full_song_from_update', 20)
        budget = None if full else settings.get('early_evaluation_steps', 256)
        scope = 'full_song' if full else f'prefix_{budget}'
        actual_envs = settings.get('evaluation_num_envs', 4)
        snapshot = output/f'evaluation_snapshot_{state["accepted_updates"]:06d}.pt'
        atomic_save(shared_cpu_state(dict(adapter=control.adapter.state_dict())), snapshot)
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve().parents[3]/'tools/evaluate_stage2_dedicated.py'),
            '--config', str(Path(args.config).resolve()), '--snapshot', str(snapshot), '--output', str(output),
            '--num-envs', str(actual_envs), '--adapter-updates', str(state['accepted_updates'])]
        if budget is not None:
            command += ['--max-steps', str(budget)]
        for split in ('validation', 'evaluation'):
            cached = extra['zero_evaluations'].get(split+'_'+scope)
            cache_path = output/f'{split}_{scope}_zero_dedicated{actual_envs}.json'
            if (cached is not None and cached.get('evaluation_backend') == 'dedicated_physics'
                    and cached['num_envs'] == actual_envs and not cache_path.exists()):
                json_write(cache_path, cached)
        publish('dedicated_fixed_physics', evaluation_scope=scope, actual_evaluation_envs=actual_envs)
        with (output/f'evaluation_{state["accepted_updates"]:06d}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports = {split: json.loads((output/f'{split}_{state["accepted_updates"]:06d}.json').read_text())
                   for split in ('validation', 'evaluation')}
        for split in ('validation', 'evaluation'):
            extra['zero_evaluations'][split+'_'+scope] = json.loads(
                (output/f'{split}_{scope}_zero_dedicated{actual_envs}.json').read_text())
        zero = json.loads((output/f'validation_{scope}_zero_dedicated{actual_envs}.json').read_text())
        key = 'best' if full else 'best_short'
        extra[key] = select_validation_best(extra[key], zero, reports['validation'], extra['zero_checkpoint'],
            adapter_updates=state['accepted_updates'], scope=scope)
        if full and state['accepted_updates'] > 0:
            extra['validation_quality'] = validation_degradation(extra.get('validation_quality'),
                zero, reports['validation'], settings.get('quality_stop'))
            append_json(output/'validation_quality.jsonl', dict(extra['validation_quality'],
                adapter_updates=state['accepted_updates'], timestamp=time.time()))
        if full and extra['best'] and extra['best'].get('model_identity') == 'initial_zero_Adapter':
            if extra['zero_checkpoint'] is not None:
                json_write(output/'best_checkpoint.json', dict(extra['zero_checkpoint'], validation=extra['best']), replace=True)
        extra['evaluations'].append(state['accepted_updates'])
        append_json(output/'timing.jsonl', dict(phase='dedicated_evaluation', adapter_updates=state['accepted_updates'],
            seconds=time.perf_counter()-begin, actual_physics_envs=actual_envs,
            evaluation_physics=json.loads((output/f'dedicated_timing_{state["accepted_updates"]:06d}.json').read_text()),
            training_physics_advanced=False, timestamp=time.time()))
        return {split: {k: r[k] for k in ('completion_rate', 'mean_tracking', 'mean_root_position_rmse_m',
                                        'mean_jerk_rms', 'mean_beat_f1', 'paired_zero_difference')} for split, r in reports.items()}

    try:
        publish('loading_frozen_assets')
        seed_all(train['seed']); torch.set_num_threads(1)
        # 延续原FP32数值口径，保留旋转差分与跟踪度量精度；这里没有导数投影。
        torch.backends.cuda.matmul.allow_tf32 = False
        engine = build_engine(cfg, 'cuda:0')
        enable_wm_normalization(engine.dynamics)
        bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
        if len(bank.by_split['train']) != 64:
            raise ValueError('正式配置必须加载既有64首GENMO生成训练缓存')
        control = VectorControl(engine, bank)
        control.adapter.disable_root_xy = True
        control.disable_root_orientation = not settings.get('root_orientation', True)
        # 直接使用有界十帧输出；正式训练不构造导数投影或参考拒绝器。
        calibration = json.loads(Path(settings['calibration']).read_text())
        control.precision = PrecisionObjective(calibration['scales'], **settings['precision'])
        extra['precision_scales'] = control.precision.scales
        control.soft_joint_limit_cost = False
        control.sm_soft_termination_only = True
        control.relative_critic_features = True
        critic = Critic(2835).cuda()
        critic.requires_grad_(settings['critic_enabled'])
        optimizers = dict(adapter=torch.optim.Adam(control.adapter.parameters(), lr=train['adapter_lr'], betas=(.9, .95)),
            wm=torch.optim.Adam(engine.dynamics.model.net.parameters(), lr=train['wm_lr']),
            critic=torch.optim.Adam(critic.parameters(), lr=train['critic_lr']))
        replay = VectorReplay(train['replay_capacity'], vector['num_envs'], 'cuda:0', compute_mc=False)
        tangent = TangentReplay(1)  # 完整旧checkpoint容器保持空，不参与监督。
        frozen = engine.step.bundle.frozen_snapshot()
        initial = dict(adapter=parameter_identity(control.adapter), wm=parameter_identity(engine.dynamics),
            actor=parameter_identity(control.bundle.actor), checkpoint_sha256=file_digest(cfg['supermimic']['checkpoint']),
            tasks_sha256=file_digest(cfg['tasks']), calibration_sha256=file_digest(settings['calibration']),
            wm_in_count=float(engine.dynamics.model.in_norm.count), wm_out_count=float(engine.dynamics.model.out_norm.count),
            normalizer_policy='original_RunningNorm_preserve_pretrained_mean_var_count_and_max_count',
            adapter_dimension=control.spec.total_dim if hasattr(control.spec, 'total_dim') else 2654)
        json_write(output/'initial_assets.json', initial)
        publish('compiling_shared_FK')
        json_write(output/'compiled_fk.json', compile_kinematics(engine, bank))
        if settings.get('compile_components', False):
            from .compute import compile_components
            compile_components(control, engine.dynamics)
        sim_cfg = make_sim('cuda:0', args)
        with launch_simulation(sim_cfg, args):
            env = StreamingPhysicalEnv(sim_cfg, vector['num_envs'], engine.step.bundle)
            collector = VectorCollector(env, control, seed=vector['sampling_seed'])
            collector.uniform_start = True
            collector.projection_recovery_attempts = 0
            collector.reset(split='train', perturb=True)
            if args.resume:
                ck = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
                for key in ('alternating', 'bptt_steps', 'training'):
                    if ck['config'].get(key) != cfg[key]:
                        raise ValueError('恢复算法/时域/优化合同不符；不能加载旧审计或退化Adapter')
                state.update(load_vector(args.resume, engine, critic, optimizers, replay, tangent, collector, cfg))
                extra = normalize_extra_state(ck.get('extra_state'))
                control.last_imagination_failure = extra.get('imagination_failure')
                if 'action_noise_rng' in extra:
                    collector.action_generator.set_state(extra['action_noise_rng'])
                collector.projection_recoveries = state.get('projection_recovered_environments', 0)
                state.update(pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'], resumed_from=args.resume)
                state.pop('error', None)
                json_write(output/'resume_info.json', dict(source_checkpoint=args.resume,
                    adapter_sha256=parameter_identity(control.adapter), wm_sha256=parameter_identity(engine.dynamics),
                    adapter_updates=state['accepted_updates'], wm_updates=engine.dynamics.model_version,
                    replay_transitions=len(replay), physics_resume='new_real_episode_no_cross_reset_windows', tf32=False,
                    exploration_rng=('restored_independent_action_stream' if 'action_noise_rng' in extra
                                     else 'legacy_checkpoint_missing_new_action_stream_initialized_from_seed')))
                del ck
            else:
                if 0 in settings['evaluation_updates']:
                    publish('zero_update_fixed_physics')
                    summary = evaluate()
                    print(json.dumps(dict(adapter_updates=0, physical_evaluation=summary), ensure_ascii=False), flush=True)
                checkpoint('zero_update')
                publish('initial_real_collection')
                collect(settings['warmup_vector_steps'])
            torch.cuda.reset_peak_memory_stats()
            while state['accepted_updates'] < vector['target_accepted_updates']:
                begin = time.perf_counter()
                state['iterations'] += 1
                extra['step_acc'] += 1/settings['utd']
                n = int(extra['step_acc']); extra['step_acc'] -= n
                publish('real_collect')
                row = collect(n)
                publish('WM_H8_updates')
                wm_begin = time.perf_counter()
                for _ in range(train['wm_updates_per_iteration']):
                    wm_report = wm_update(engine.dynamics, optimizers['wm'], replay.windows(train['wm_batch'], 8))
                torch.cuda.synchronize()
                row.update(wm_report, wm_seconds=time.perf_counter()-wm_begin)
                warmup = state['iterations'] <= settings['wm_only_iterations']
                if not warmup:
                    publish('Adapter_BPTT')
                    starts = replay.starts(train['actor_batch'], horizon=cfg['bptt_steps'], ends=bank.ends)
                    def completed(metrics):
                        state['accepted_updates'] += 1
                        state['adapter_optimizer_updates'] += 1
                        publish('Adapter_Adam_completed', last_adapter_gradient_norm=metrics['adapter_gradient_norm'],
                                last_parameter_delta=metrics['parameter_delta'])
                    pi_begin = time.perf_counter()
                    try:
                        pi, imagined = policy_update(control, engine.dynamics, critic, optimizers['adapter'], starts,
                            cfg['bptt_steps'], terminal_value=False, microbatch=train['backward_microbatch'],
                            gamma=train['gamma_per_control'], diagnostics=True, timing=True, on_step=completed,
                            failure_context=dict(source=runtime_source, experiment_source=state['source'],
                                iteration=state['iterations'],
                                adapter_updates_before=state['accepted_updates'],
                                candidate_adapter_update=state['accepted_updates']+1))
                    except PolicyProjectionRejected as error:
                        optimizers['adapter'].zero_grad(set_to_none=True)
                        extra['consecutive_projection_rejections'] += 1
                        state['rejected_adapter_iterations'] = state.get('rejected_adapter_iterations', 0)+1
                        append_json(output/'rejected_updates.jsonl', dict(error.report,
                            consecutive=extra['consecutive_projection_rejections'],
                            adapter_updates=state['accepted_updates'], wm_updates=engine.dynamics.model_version,
                            iteration=state['iterations'], timestamp=time.time()))
                        if extra['consecutive_projection_rejections'] >= settings.get('max_consecutive_projection_rejections', 5):
                            raise RuntimeError('连续全部投影失败达到有限修复预算；保存现场，停止本组无效WM循环') from error
                        publish('projection_batch_rejected_no_Adam')
                        del starts
                        continue
                    extra['consecutive_projection_rejections'] = 0
                    row.update(pi, policy_seconds=time.perf_counter()-pi_begin)
                    publish('imagined_GAE_Critic' if settings['critic_enabled'] else 'Critic_disabled')
                    v = critic_update(critic, optimizers['critic'], imagined, gamma=train['gamma_per_control'],
                        lam=train['lambda_return_per_control'], epochs=settings['critic_epochs'],
                        minibatches=settings['critic_minibatches'])
                    state['critic_optimizer_updates'] += v['critic_optimizer_steps']
                    row.update(v)
                    del starts, imagined
                if state['iterations'] % 10 == 0 or state['accepted_updates'] == 1:
                    engine.step.bundle.assert_unchanged(frozen)
                update = state['accepted_updates']
                row.update(iteration=state['iterations'], adapter_updates=update, wm_updates=engine.dynamics.model_version,
                    phase='WM_only_warmup' if warmup else 'formal_alternating', real_transitions=collector.real_transitions,
                    train_transitions=collector.real_by_role.get('train', 0), iteration_seconds=time.perf_counter()-begin,
                    timestamp=time.time(), peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    wm_in_norm_count=float(engine.dynamics.model.in_norm.count),
                    wm_out_norm_count=float(engine.dynamics.model.out_norm.count))
                append_json(output/'training.jsonl', row)
                append_json(output/'timing.jsonl', dict(phase='alternating_iteration', adapter_updates=update,
                    real_collection_seconds=row['collection_seconds'], wm10_seconds=row['wm_seconds'],
                    real_collection_regions=row['collection_breakdown_seconds'],
                    adapter_seconds=row.get('policy_seconds'), adapter_regions=row.get('timing'), timestamp=time.time()))
                if warmup and state['iterations'] % 10 == 0 or not warmup and (update == 1 or update % settings['log_every'] == 0):
                    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
                    from .training_report import safe_update_report as update_report
                    update_report(output.parent)
                evaluated = False
                if not warmup and update in settings['evaluation_updates'] and update not in extra['evaluations']:
                    summary = evaluate()
                    evaluated = True
                    print(json.dumps(dict(adapter_updates=update, physical_evaluation=summary), ensure_ascii=False), flush=True)
                if (warmup and state['iterations'] == settings['wm_only_iterations']) or (not warmup and
                        (update == 1 or update % vector['checkpoint_every'] == 0 or evaluated)):
                    if not warmup:
                        atomic_save(dict(adapter=control.adapter.state_dict(), dynamics=engine.dynamics.state_dict(),
                            critic=critic.state_dict(), config=cfg, state=state.copy(), continuation='weights_only'),
                            output/f'weights_{update:06d}_{time.time_ns()}.pt')
                    checkpoint('warmup' if warmup else 'periodic')
                if extra.get('validation_quality', {}).get('stop'):
                    json_write(output/'quality_stop.json', dict(extra['validation_quality'],
                        adapter_updates=state['accepted_updates'], timestamp=time.time()))
                    publish('stopped_sustained_validation_degradation',
                            stop_reasons=extra['validation_quality']['reasons'])
                    return 0
                publish('WM_only_warmup' if warmup else 'formal_alternating', gpu_process_mib=gpu_usage())
            publish('completed_configured_Adapter_updates', target_updates=vector['target_accepted_updates'])
            return 0
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(), state=state.copy(), timestamp=time.time())
        print(failure['traceback'], flush=True)
        json_write(output/f'failure_{time.time_ns()}.json', failure)
        if collector is not None and replay is not None:
            try:
                extra['failure'] = failure
                extra['constraint_failure'] = getattr(collector, 'last_constraint_failure', None)
                extra['constraint_state'] = getattr(collector, 'last_constraint_state', None)
                extra['constraint_proposal'] = getattr(collector, 'last_constraint_proposal', None)
                extra['imagination_failure'] = getattr(control, 'last_imagination_failure', None)
                checkpoint('failure')
            except BaseException:
                json_write(output/f'failure_save_error_{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        publish('failed_runtime_protection', error=repr(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
