"""失败方法修复后的A/B/C真实物理对照入口，不恢复旧饱和策略或启动八卡搜索。

A仅关闭根XY；B同时施加确定性参考导数约束；C在B上用真实物理潜变量PPO更新。
所有组零Adapter、同冻结资产、真实任务划分、基础随机流和25/50/100个Adam参数步
预算。A/B的每个候选都绑定当前WM和完整目标的H8物理参数方向；监督修复最多限定
次数，未通过即保存现场退出。C不使用WM到Adapter的梯度，Critic仅训练真实MC/GAE。
完整MC与WM短窗Replay分开，完整checkpoint含三优化器、归一化、随机流和未提交批次。
真实评价同时报告完整歌曲预算Tracking、完成率、根RMSE、jerk和参考导数分布。
"""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--mode', choices=('train', 'numerical', 'integration'), default='train')
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    cfg0 = OmegaConf.load(preliminary.config)
    sys.path.insert(0, str(cfg0.supermimic.repo))
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from .application import read_config, read_tasks, build_engine, seed_all
    from .vector_bank import VectorBank
    from .vector_control import VectorControl, select_state
    from .vector_collector import VectorCollector
    from .vector_replay import VectorReplay
    from .vector_learning import GroupNormalizedCritic, TangentReplay, fit_wm, fit_critic, validate_critic, adapter_update
    from .vector_entry import json_write, append_json, gpu_usage, gpu_utilization, numerical_check, compile_kinematics
    from .vector_checkpoint import save_vector, load_vector
    from .vector_evaluation import evaluate_physical, collect_validation
    from .real_returns import complete_real_mc, coverage, dataset_to
    from .physical_ppo import collect_on_policy, ppo_update
    from .parameter_audit import parameter_update_audit, audit_matches
    from .vector_audit import physical_directions
    from .update_diagnostics import fixed_probes, parameter_identity
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    cfg = read_config(args.config)
    repair, train, vc = cfg['repair'], cfg['training'], cfg['vector']
    if repair['variant'] not in ('A', 'B', 'C') or vc['target_accepted_updates'] > 100:
        raise ValueError('本入口仅执行A/B/C最多100次从零开始的修复对照')
    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('每组只能一个GPU进程，不使用DDP')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(train['seed']); torch.set_num_threads(1)
    state = dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), variant=repair['variant'],
        started_unix=time.time(), phase='initializing', accepted_updates=0, attempted_adapter_updates=0,
        wm_updates=0, repair_attempts=0, num_envs=vc['num_envs'],
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        budget_definition='one_full_batch_Adam_step_per_accepted_update')
    json_write(output/'manifest.json', dict(config=cfg, command=[sys.executable, *sys.argv], state=state.copy(),
        initialization='zero_last_layer_not_old_adapter', communication='none', seed_policy='paired_initial_rng_then_independent'))
    engine = build_engine(cfg, 'cuda:0')
    tasks = read_tasks(cfg['tasks'])
    bank = VectorBank(tasks, engine, 'cuda:0')
    control = VectorControl(engine, bank)
    critic = GroupNormalizedCritic(2835, control.spec).cuda()
    # 新约束前先证明优化后的Goal/Actor/原SM梯度仍等价；不会修改网络。
    json_write(output/'numerical_legacy_contract.json', numerical_check(engine, bank, control))
    control.adapter.disable_root_xy = True
    control.constraint_beta = 1. if repair['variant'] == 'A' else repair['constraint_beta']
    control.root_cost_weight = repair['root_position_cost']
    control.velocity_cost_weight = repair['reference_velocity_cost']
    control.soft_imagination_continuation = False
    control.relative_critic_features = True
    json_write(output/'reference_derivative_certificate.json', bank.derivative_certificate(control.adapter, control.constraint_beta))
    if args.mode == 'numerical':
        return 0
    if vc.get('compile_fk', True):
        json_write(output/'compiled_kinematics.json', compile_kinematics(engine, bank))
    frozen = engine.step.bundle.frozen_snapshot()
    optimizers = dict(adapter=torch.optim.Adam(control.adapter.parameters(), lr=train['adapter_lr']),
        critic=torch.optim.Adam(critic.parameters(), lr=train['critic_lr']),
        wm=torch.optim.Adam(engine.dynamics.model.net.parameters(), lr=train['wm_lr']))
    replay = VectorReplay(train['replay_capacity'], vc['num_envs'], 'cuda:0')
    validation = VectorReplay(vc['validation_capacity'], min(32, vc['num_envs']), 'cuda:0')
    tangent = TangentReplay(capacity=16384)
    collector = None
    extra = dict(mc={}, mc_episodes={}, pending_batch=None, fixed_probes=None, last_valid_adapter=None)
    pending_before = None
    sim_cfg = make_sim('cuda:0', args)

    def publish(phase=None):
        if phase:
            state['phase'] = phase
        state.update(wm_updates=engine.dynamics.model_version, wall_unix=time.time(),
            real_transitions=collector.real_transitions if collector else 0,
            real_transitions_by_role=collector.real_by_role.copy() if collector else {},
            replay_transitions=len(replay), gpu_process_mib=gpu_usage(), gpu_utilization=gpu_utilization(),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        json_write(output/'status.json', state, replace=True)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def checkpoint(label):
        publish()
        destination = output/f'{label}_{state["accepted_updates"]:06d}_{time.time_ns()}.pt'
        extra['last_valid_adapter'] = copy.deepcopy(control.adapter.state_dict())
        save_vector(destination, engine, critic, optimizers, replay, tangent, collector, state.copy(), cfg, validation, extra)
        json_write(output/'latest_checkpoint.json', dict(path=str(destination), **state), replace=True)
        return str(destination)

    def evaluate(label):
        publish('fixed_full_song_physical_evaluation')
        result = evaluate_physical(collector)
        result['accepted_updates'] = state['accepted_updates']
        json_write(output/f'evaluation_{label}.json', result)
        collector.reset(split='train', perturb=True)
        return result

    def critic_reports():
        trainset = dataset_to(extra['mc']['train'], 'cuda:0')
        reports = {}
        for split in ('train', 'validation'):
            data = dataset_to(extra['mc'][split], 'cuda:0')
            reports[split] = validate_critic(critic, data, trainset['y'].mean(), train['critic_quality'], len(trainset['y']), tasks)
        return reports

    try:
        with launch_simulation(sim_cfg, args):
            env = StreamingPhysicalEnv(sim_cfg, vc['num_envs'], engine.step.bundle)
            collector = VectorCollector(env, control, seed=vc['sampling_seed'])
            collector.reset(perturb=True)
            torch.cuda.reset_peak_memory_stats()
            if args.mode == 'integration':
                publish('real_small_batch_integration_not_formal_training')
                before = copy.deepcopy(control.adapter.state_dict())
                batch = collect_on_policy(collector, critic, replay, steps=32, std=repair['latent_std'])
                extra['fixed_probes'] = batch['probes']
                update = ppo_update(control, optimizers['adapter'], batch, batch=repair['ppo_microbatch'])
                fitting = fit_critic(critic, optimizers['critic'],
                    dict(x=batch['critic_x'], y=batch['target'], split='train'), epochs=1, batch=1024)
                pair = collector.tangent_pairs(); tangent.add(pair)
                wm = fit_wm(engine.dynamics, optimizers['wm'], replay.windows(32, 8), tangent.sample(32), vc['wm_tangent'])
                gate = parameter_update_audit(collector, control, engine.dynamics, critic, before, criteria=train['wm_quality'])
                state['integration_parameter_updates'] = int(update['effective'])
                saved = checkpoint('integration_complete')
                identity = parameter_identity(control.adapter)
                load_vector(saved, engine, critic, optimizers, replay, tangent, collector, cfg, validation)
                assert parameter_identity(control.adapter) == identity
                engine.step.bundle.assert_unchanged(frozen)
                json_write(output/'integration.json', dict(update=update, critic=fitting, wm=wm,
                    full_parameter_audit=gate, checkpoint_restored=True, frozen_actor_unchanged=True,
                    actual_transitions=collector.real_transitions, formal_updates=0))
                publish('integration_complete_not_formal_training')
                return 0
            if args.resume:
                ck = torch.load(args.resume, weights_only=False, map_location='cpu', mmap=True)
                if ck['config'].get('repair') != repair or not ck.get('extra_state'):
                    raise ValueError('只允许同一修复方法的完整状态恢复，不接受旧饱和Adapter')
                state.update(load_vector(args.resume, engine, critic, optimizers, replay, tangent, collector, cfg, validation))
                extra = ck['extra_state']
                extra['pending_batch'] = None  # 求解器不连续；保存的未提交批次作诊断，不伪称新在策略数据。
                extra['fixed_probes'] = {k: v.cuda() for k, v in extra['fixed_probes'].items()}
                state.update(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), resumed_from=args.resume)
            else:
                for split in ('train', 'validation'):
                    publish('independent_complete_real_MC_'+split)
                    dataset, episodes = complete_real_mc(collector, split=split, version=0,
                        repetitions=repair['mc_repetitions'], stride=repair['mc_stride'],
                        latent_std=repair['latent_std'] if repair['variant'] == 'C' else 0.)
                    extra['mc'][split], extra['mc_episodes'][split] = dataset, episodes
                    json_write(output/f'mc_coverage_{split}_000000.json', coverage(dataset, episodes, tasks))
                publish('critic_warmup_from_verified_real_MC')
                fit = fit_critic(critic, optimizers['critic'], dataset_to(extra['mc']['train'], 'cuda:0'),
                                epochs=repair['critic_warmup_epochs'], batch=train['critic_batch'])
                json_write(output/'critic_initial.json', dict(fit=fit, calibration=critic_reports()))
                evaluate('000000')
                # 固定探针来自真实初态与合法训练歌曲，不混入评价/审计歌曲。
                selected = bank.by_split['train']
                songs = selected[torch.arange(collector.num_envs, device=bank.device)%len(selected)]
                frames = ((torch.arange(collector.num_envs, device=bank.device)//len(selected))%4)*50
                frames = torch.minimum(frames, bank.ends[songs]-32)
                collector.reset(songs=songs, frames=frames, perturb=True)
                extra['fixed_probes'] = fixed_probes(collector.state)
                for _ in range(vc['warmup_vector_steps']):
                    collector.step(replay)
                if repair['variant'] != 'C':
                    tangent.add(collector.tangent_pairs())
                    collect_validation(collector, validation, vc['validation_vector_steps'])
                    collector.reset(split='train', perturb=True)
                checkpoint('initialized')
            while state['accepted_updates'] < vc['target_accepted_updates']:
                collector.policy_version = state['accepted_updates']
                loop_started = time.time()
                state['attempted_adapter_updates'] += 1
                if repair['variant'] == 'C':
                    publish('real_physics_on_policy_collection')
                    batch = collect_on_policy(collector, critic, replay, steps=repair['ppo_rollout_steps'], std=repair['latent_std'])
                    # 固定多歌曲探针跨更新不变；在线分布另外由采样统计/真实评价观测。
                    batch['probes'] = extra['fixed_probes']
                    extra['pending_batch'] = batch
                    pending_before = (copy.deepcopy(control.adapter.state_dict()), copy.deepcopy(optimizers['adapter'].state_dict()))
                    publish('real_physics_PPO_parameter_update')
                    update = ppo_update(control, optimizers['adapter'], batch, batch=repair['ppo_microbatch'])
                    critic_fit = fit_critic(critic, optimizers['critic'],
                        dict(x=batch['critic_x'], y=batch['target'], split='train'),
                        epochs=train['critic_epochs'], batch=train['critic_batch'])
                    update.update(real_reward_mean=batch['real_reward_mean'], real_batch_transitions=batch['real_transitions'],
                                  real_terminals=batch['true_terminals'], critic_fit=critic_fit)
                    extra['pending_batch'] = None
                    pending_before = None
                else:
                    publish('real_supervision_and_current_WM_repair')
                    for _ in range(train['collect_vector_steps_per_update']):
                        collector.step(replay)
                    wm_reports = []
                    steps = repair['wm_initial_steps'] if engine.dynamics.model_version == 0 else (
                        repair['wm_repair_steps'] if state['repair_attempts'] else train['wm_updates_per_iteration'])
                    for _ in range(steps):
                        wm_reports.append(fit_wm(engine.dynamics, optimizers['wm'],
                            replay.windows(train['wm_batch'], train['wm_fit_window']), tangent.sample(train['wm_batch']), vc['wm_tangent']))
                    prediction = engine.dynamics.validate_real(validation.windows(32, 16))
                    # H1局部诊断只记录一次；绝不读取其passed_horizons作为训练许可。
                    if not (output/'local_derivative_diagnostic.json').exists():
                        local = physical_directions(collector, control, engine.dynamics, critic,
                            json.loads(Path(cfg['physical_criteria']).read_text()), horizons=(1,))
                        json_write(output/'local_derivative_diagnostic.json', local)
                        collector.reset(split='train', perturb=True)
                    pending_before = (copy.deepcopy(control.adapter.state_dict()), copy.deepcopy(optimizers['adapter'].state_dict()))
                    starts = replay.starts(train['actor_batch'])
                    update, _ = adapter_update(control, engine.dynamics, critic, optimizers['adapter'], starts, 8,
                        terminal_value=False, microbatch=train['backward_microbatch'], probes=extra['fixed_probes'])
                    if update['effective']:
                        publish('exact_candidate_H8_H16_physical_parameter_audit')
                        gate = parameter_update_audit(collector, control, engine.dynamics, critic, pending_before[0],
                            criteria=train['wm_quality'])
                        if not audit_matches(gate, engine.dynamics, control.adapter):
                            raise RuntimeError('WM/Adapter参数身份在审计后改变')
                        json_write(output/f'candidate_audit_{state["attempted_adapter_updates"]:06d}.json', gate)
                        update.update(audit=gate['gates'], wm_prediction=prediction, wm_fits=wm_reports)
                        if 8 not in gate['passed_horizons']:
                            control.adapter.load_state_dict(pending_before[0]); optimizers['adapter'].load_state_dict(pending_before[1])
                            update['effective'] = False
                            update['diagnosis'] = 'current_complete_parameter_H8_direction_not_verified'
                            state['repair_attempts'] += 1
                        else:
                            state['repair_attempts'] = 0
                    pending_before = None
                if update['effective']:
                    state['accepted_updates'] += 1
                    state['phase'] = update['phase']
                elif repair['variant'] == 'C' or update['diagnosis'] != 'current_complete_parameter_H8_direction_not_verified':
                    state['phase'] = 'paused_optimizer_inactive_with_diagnostics'
                    append_json(output/'training.jsonl', dict(state=state.copy(), update=update))
                    checkpoint('inactive_complete')
                    return 3
                elif state['repair_attempts'] >= repair['max_repair_attempts']:
                    state['phase'] = 'stopped_bounded_WM_repair_exhausted'
                    append_json(output/'training.jsonl', dict(state=state.copy(), update=update))
                    checkpoint('direction_failure_complete')
                    return 2
                state['iteration_seconds'] = time.time()-loop_started
                state['accepted_updates_per_hour'] = state['accepted_updates']*3600/max(1, time.time()-state['started_unix'])
                engine.step.bundle.assert_unchanged(frozen)
                append_json(output/'training.jsonl', dict(state=state.copy(), update=update))
                publish()
                if update['effective'] and state['accepted_updates'] in repair['evaluation_updates']:
                    evaluate(f'{state["accepted_updates"]:06d}')
                    # 同当前策略独立真实MC复核；与用于GAE/WM的训练环形缓冲完全解耦。
                    for split in ('train', 'validation'):
                        publish('independent_complete_real_MC_'+split)
                        dataset, episodes = complete_real_mc(collector, split=split, version=state['accepted_updates'],
                            repetitions=repair['mc_repetitions'], stride=repair['mc_stride'],
                            latent_std=repair['latent_std'] if repair['variant'] == 'C' else 0.)
                        extra['mc'][split], extra['mc_episodes'][split] = dataset, episodes
                        json_write(output/f'mc_coverage_{split}_{state["accepted_updates"]:06d}.json', coverage(dataset, episodes, tasks))
                    json_write(output/f'critic_{state["accepted_updates"]:06d}.json', critic_reports())
                    fit_critic(critic, optimizers['critic'], dataset_to(extra['mc']['train'], 'cuda:0'),
                               epochs=train['critic_epochs'], batch=train['critic_batch'])
                    collector.reset(split='train', perturb=True)
                    checkpoint('full')
                if (output/'stop_after_checkpoint.request').exists():
                    state['phase'] = 'stopped_after_explicit_request'
                    checkpoint('requested_exit')
                    return 0
            publish('completed_100_or_configured_paired_update_budget')
            checkpoint('complete')
            return 0
    except Exception as error:
        if pending_before is not None:
            extra['rejected_candidate'] = copy.deepcopy(control.adapter.state_dict())
            control.adapter.load_state_dict(pending_before[0]); optimizers['adapter'].load_state_dict(pending_before[1])
        state.update(phase='hard_failure_saved_for_recovery', error=repr(error), traceback=traceback.format_exc())
        if collector is not None:
            try:
                state['failure_checkpoint'] = checkpoint('exception_complete')
            except Exception as save_error:
                state['checkpoint_error'] = repr(save_error)
        json_write(output/'failure.json', state)
        publish()
        raise
