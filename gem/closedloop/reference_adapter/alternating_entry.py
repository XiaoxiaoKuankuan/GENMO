"""Server-1单GPU正式SM交替训练入口；八组之间没有模型、Replay或通信共享。

数据流是当前Adapter真实采集(含实际动作探索)→原SM H8世界模型十次更新→冻结
Actor/WM后的逐步BPTT→可选想象GAE Critic。100个WM-only交替循环后直接策略
更新，不加载旧Adapter，不运行Oracle/方向审计/物理候选回滚。GPU配置决定H8、
H16或H32及动作探索消融；未经真实回报校准的Critic暂不参与训练或终端梯度。

每次有限Adam完成立即计数并记录修正传播及投影梯度；0/5/10更新固定短时对照，
20更新起执行无探索整歌validation和evaluation，只有validation选择best。评价下降只
报告。运行异常保存三个网络/优化器、Replay、RNG及约束现场；恢复保留学习状态，
真实求解器从新episode开始，不伪装隐藏物理状态续接。完整checkpoint保留最近两份
及best，所有编号小模型和报告永久保留，避免八卡完整Replay耗尽服务器磁盘。
"""

import argparse
import json
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
                    precision_scales=None, zero_evaluations={}, consecutive_projection_rejections=0)
    for name, default in defaults.items():
        result.setdefault(name, default)
    return result


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
    from .application import read_config, build_engine, read_tasks, seed_all
    from .vector_bank import VectorBank
    from .vector_control import VectorControl
    from .vector_collector import VectorCollector
    from .vector_replay import VectorReplay
    from .vector_learning import TangentReplay
    from .vector_entry import json_write, append_json, compile_kinematics, gpu_usage
    from .vector_evaluation import evaluate_physical
    from .paired_evaluation import paired_comparison
    from .vector_checkpoint import save_vector, load_vector
    from .reference_constraints import ReferenceProjection
    from .precision import PrecisionObjective
    from .update_diagnostics import parameter_identity
    from .alternating_learning import (enable_wm_normalization, update_wm_normalization, wm_update,
                                      policy_update, critic_update, PolicyProjectionRejected)

    cfg = read_config(args.config)
    settings, train, vector = cfg['alternating'], cfg['training'], cfg['vector']
    if settings['method'] != 'supermimic_alternating_WM_BPTT' or train['wm_fit_window'] != 8:
        raise ValueError('正式入口只允许原SM H8监督交替算法')
    if cfg['bptt_steps'] not in (8, 16, 32):
        raise ValueError('本次正式BPTT配置只使用H8/H16/H32')
    if settings['critic_enabled']:
        raise ValueError('10月28日修复实验暂时关闭未经独立真实回报校准的Critic；不能使用旧想象EV放行')
    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('每组必须仅可见一张GPU；禁止DDP/跨GPU通信')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), phase='initializing',
        accepted_updates=0, adapter_optimizer_updates=0, wm_updates=0, critic_optimizer_updates=0,
        iterations=0, real_transitions=0, num_envs=vector['num_envs'], started_unix=time.time(),
        horizon=cfg['bptt_steps'], critic_enabled=settings['critic_enabled'], method=settings['method'],
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), formal=True)
    json_write(output/'manifest.json', dict(config=cfg, command=[sys.executable, *sys.argv], state=state.copy(),
        initialization='full_resume_same_experiment' if args.resume else 'zero_Adapter_original_pretrained_WM_no_old_policy', communication='none',
        acceptance='finite_Adam_step_not_a_physical_improvement_certificate',
        critic_status='disabled_unqualified_real_MC_anchor_and_independent_calibration_required',
        full_checkpoint_retention='latest_two_and_validation_best_all_numbered_weights_kept'))
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
        publish('saving_complete_checkpoint')
        path = output/f'full_{state["accepted_updates"]:06d}_{label}_{time.time_ns()}.pt'
        extra['completed_checkpoints'].append(str(path))
        if extra['best'] is not None and extra['best']['adapter_updates'] == state['accepted_updates']:
            extra['best']['path'] = str(path)
        extra['action_noise_rng'] = collector.action_generator.get_state()
        extra['projection_recovery_events'] = collector.projection_recovery_events
        save_vector(path, engine, critic, optimizers, replay, tangent, collector, state.copy(), cfg,
                    extra_state=extra)
        pointer = dict(path=str(path), adapter_updates=state['accepted_updates'], wm_updates=state['wm_updates'],
                       bytes=path.stat().st_size, physics_resume='new_episode_with_preserved_replay')
        json_write(output/'latest_checkpoint.json', pointer, replace=True)
        if extra['best'] is not None and extra['best']['adapter_updates'] == state['accepted_updates']:
            json_write(output/'best_checkpoint.json', dict(pointer, validation=extra['best']), replace=True)
        # 仅清理本运行自己已原子保存、且有新完整替代的文件；不删除旧实验/外部resume。
        keep = set(extra['completed_checkpoints'][-2:])
        if extra['best'] and extra['best'].get('path'):
            keep.add(extra['best']['path'])
        for old in list(extra['completed_checkpoints']):
            old_path = Path(old)
            if old not in keep and old_path.parent == output and old_path.name.startswith('full_'):
                if old_path.exists():
                    old_path.unlink()
                extra['completed_checkpoints'].remove(old)
        append_json(output/'checkpoint_history.jsonl', pointer)
        return path

    def collect(n):
        totals = {}
        begin = time.perf_counter()
        collector.policy_version = state['accepted_updates']
        for _ in range(n):
            row = collector.step(replay, action_noise=settings['action_noise'], split='train', perturb=True,
                exploration_fraction=settings.get('exploration_fraction', 0.),
                exploration_std=settings.get('exploration_std', .1))
            update_wm_normalization(engine.dynamics, row)
            values = dict(real_reward=row['reward'].mean(), real_tracking=row['raw']['track'].mean(),
                real_reference_rms=row['proposal']['residual'].square().mean().sqrt(),
                real_action_vs_zero_rms=(row['proposal']['action']-row['proposal']['zero_action']).square().mean().sqrt(),
                policy_action_vs_zero_rms=(row['proposal'].get('policy_action', row['proposal']['action'])-
                                          row['proposal']['zero_action']).square().mean().sqrt())
            values.update({'real_'+k: value.mean() for k, value in row['raw'].items()
                if isinstance(value, torch.Tensor) and value.ndim == 1 and
                (k in ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori',
                       'action_rate', 'action_deviation') or k.startswith(('absolute_', 'error_')))})
            if 'exploration_std' in row['proposal']:
                values['executed_noise_rms'] = (row['proposal']['action']-row['proposal']['policy_action']).square().mean().sqrt()
                values['large_exploration_fraction'] = (row['proposal']['exploration_std'] >= .099).float().mean()
            for key, value in values.items():
                totals[key] = totals.get(key, 0.)+float(value)/n
        totals.update(collection_seconds=time.perf_counter()-begin, collection_vector_steps=n,
                      physical_metrics_scope='training_distribution_not_paired_policy_improvement')
        if collector.projection_recovery_events:
            append_json(output/'projection_recovery.jsonl', dict(
                adapter_updates=state['accepted_updates'], events=collector.projection_recovery_events))
            collector.projection_recovery_events.clear()
        return totals

    def evaluate():
        # 固定评价随机流不改变后续训练抽样；恢复真实环境明确新episode。
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all(),
               collector.generator.get_state())
        reports = {}
        full = state['accepted_updates'] >= settings.get('full_song_from_update', 20)
        budget = None if full else settings.get('early_evaluation_steps', 256)
        scope = 'full_song' if full else f'prefix_{budget}'
        try:
            for split in ('validation', 'evaluation'):
                publish('fixed_real_'+split, evaluation_scope=scope)
                baseline_key = split+'_'+scope
                if baseline_key not in extra['zero_evaluations']:
                    seed_all(42); collector.generator.manual_seed(42)
                    zero = evaluate_physical(collector, zero=True, split=split, max_steps=budget)
                    zero.update(adapter_updates=0, exploration_noise=0., evaluation_scope=scope)
                    extra['zero_evaluations'][baseline_key] = zero
                    json_write(output/f'{baseline_key}_zero.json', zero)
                seed_all(42); collector.generator.manual_seed(42)
                report = evaluate_physical(collector, zero=False, split=split, max_steps=budget)
                report.update(adapter_updates=state['accepted_updates'], exploration_noise=0.,
                              evaluation_scope=scope, zero_baseline_key=baseline_key)
                zero = extra['zero_evaluations'][baseline_key]
                report['paired_zero_difference'] = {k: report[k]-zero[k]
                    if report[k] is not None and zero[k] is not None else None for k in
                    ('completion_rate', 'mean_tracking', 'mean_full_horizon_tracking',
                     'mean_root_position_rmse_m', 'mean_jerk_rms', 'mean_beat_f1')}
                report['paired_zero'] = paired_comparison(zero, report)
                json_write(output/f'{split}_{state["accepted_updates"]:06d}.json', report)
                reports[split] = report
            val = reports['validation']
            rankable = (val.get('constraint_failure_count', 0) == 0 and
                        val['mean_root_position_rmse_m'] is not None and val['mean_jerk_rms'] is not None)
            ranking = ((val['completion_rate'], -val['mean_root_position_rmse_m'],
                       val['mean_full_horizon_tracking'], -val['mean_jerk_rms']) if rankable else None)
            # 前缀评价不能与整歌混排；best_checkpoint仅由完整歌曲validation选择。
            key = 'best' if full else 'best_short'
            if rankable and (extra[key] is None or ranking > tuple(extra[key]['ranking'])):
                extra[key] = dict(adapter_updates=state['accepted_updates'], ranking=ranking,
                                 selection_split='validation_only', evaluation_scope=scope)
            extra['evaluations'].append(state['accepted_updates'])
        finally:
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state_all(rng[3]); collector.generator.set_state(rng[4])
            collector.reset(split='train', random_start=True, perturb=True)
        return {split: {k: r[k] for k in ('completion_rate', 'mean_tracking', 'mean_root_position_rmse_m',
                                        'mean_jerk_rms', 'mean_beat_f1', 'paired_zero_difference')} for split, r in reports.items()}

    try:
        publish('loading_frozen_assets')
        seed_all(train['seed']); torch.set_num_threads(1)
        # SM原式没有本项目的SO(3)投影。全局TF32会让旋转小差分的加速度失真，
        # 并破坏QP精化；原FP32数学/投影阈值不变，不能以增大容差掩盖误差。
        torch.backends.cuda.matmul.allow_tf32 = False
        engine = build_engine(cfg, 'cuda:0')
        enable_wm_normalization(engine.dynamics)
        bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
        if len(bank.by_split['train']) != 64:
            raise ValueError('正式配置必须加载既有64首GENMO生成训练缓存')
        control = VectorControl(engine, bank)
        control.adapter.disable_root_xy = True
        control.disable_root_orientation = not settings.get('root_orientation', True)
        control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds,
                                                 **settings['projection']).cuda()
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
        sim_cfg = make_sim('cuda:0', args)
        with launch_simulation(sim_cfg, args):
            env = StreamingPhysicalEnv(sim_cfg, vector['num_envs'], engine.step.bundle)
            collector = VectorCollector(env, control, seed=vector['sampling_seed'])
            collector.uniform_start = True
            collector.projection_recovery_attempts = settings.get('projection_recovery_attempts', 2)
            collector.reset(split='train', perturb=True)
            if args.resume:
                ck = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
                for key in ('alternating', 'bptt_steps', 'training'):
                    if ck['config'].get(key) != cfg[key]:
                        raise ValueError('恢复算法/时域/优化合同不符；不能加载旧审计或退化Adapter')
                state.update(load_vector(args.resume, engine, critic, optimizers, replay, tangent, collector, cfg))
                extra = normalize_extra_state(ck.get('extra_state'))
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
                            gamma=train['gamma_per_control'], diagnostics=True, on_step=completed)
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
                if warmup and state['iterations'] % 10 == 0 or not warmup and (update == 1 or update % settings['log_every'] == 0):
                    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
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
