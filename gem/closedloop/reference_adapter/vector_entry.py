"""8组实验共用的单进程单GPU向量训练入口，运行目录/随机流/Replay完全独立。

启动后依次创建真实批量仿真、采集局部响应和困难真实转移、预热WM与归一化Critic。
H1/4/8/16分别真实验收，只选择已通过时域；Critic未通过时仅用经方向验收的短时
密集目标，不引入不可信终端价值。效果下降只报告，不回滚或停训。每100次有效更新
保存完整checkpoint并做相同真实评价；未发生参数/参考/动作变化绝不增加accepted。
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def json_write(path, value, *, replace=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if replace:
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(text); os.replace(temp, path)
    else:
        with path.open('x') as f:
            f.write(text)


def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def gpu_usage():
    command = ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']
    rows = subprocess.run(command, capture_output=True, text=True).stdout.splitlines()
    for line in rows:
        fields = [x.strip() for x in line.split(',')]
        if fields[0] == str(os.getpid()):
            try:
                return int(fields[1])
            except ValueError:
                pass
    return None


def numerical_check(engine, bank, control):
    import torch
    from .collector import initial_state
    from .reference import reconstruct
    task, initial, nominal, audio = bank.tasks[0], bank.initial[:1], bank.nominal[0], bank.audio[0]
    xi = initial.new_zeros(1, 25); xi[:, 0] = 1
    observed = dict(state=initial, xi=xi, previous_action=initial.new_zeros(1, 21))
    old = initial_state(observed, nominal, audio, engine.step, episode_id='numerical', song_id=task['song_id'])
    z = control.initial(observed, torch.zeros(1, device=initial.device, dtype=torch.long),
                        torch.zeros(1, device=initial.device, dtype=torch.long), torch.zeros(1, device=initial.device, dtype=torch.long))
    errors = []
    for scale in (0., .001):
        a = (torch.randn(1, 10, 27, device=initial.device)*scale).requires_grad_()
        b = a.detach().clone().requires_grad_()
        reference = reconstruct(old, a, engine.step.runtime.kin)
        fast = control.reference(z, b)
        action_a = engine.step.bundle.act(old.control_history, reference.goal)
        action_b = control.bundle.act(z['history'], fast[2])
        probe = torch.randn_like(action_a)
        ga, = torch.autograd.grad((action_a*probe).sum(), a)
        gb, = torch.autograd.grad((action_b*probe).sum(), b)
        torch.testing.assert_close(reference.goal, fast[2], atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(action_a, action_b, atol=5e-6, rtol=2e-5)
        torch.testing.assert_close(ga, gb, atol=2e-5, rtol=2e-4)
        errors.append(dict(residual_scale=scale, goal_max_abs=float((reference.goal-fast[2]).abs().max()),
                           action_max_abs=float((action_a-action_b).abs().max()), gradient_max_abs=float((ga-gb).abs().max())))
    current = bank.ref(z['song'], z['frame'])
    state = initial.detach().clone().requires_grad_()
    old_r, old_terms = engine.step.runtime.reward(state, action_a.detach(), old.previous_action, current)
    new_r, new_terms = engine.step.runtime.reward(state, action_a.detach(), old.previous_action, current, diagnostics=True)
    torch.testing.assert_close(old_r, new_r, atol=0, rtol=0)
    g1, = torch.autograd.grad(old_r.sum(), state, retain_graph=True)
    g2, = torch.autograd.grad(new_r.sum(), state)
    torch.testing.assert_close(g1, g2, atol=0, rtol=0)
    return dict(passed=True, goal_actor_gradient=errors, reward_max_abs=float((old_r-new_r).abs().max()),
                reward_gradient_max_abs=float((g1-g2).abs().max()), source='actual_SM_weights_float32')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('train', 'pressure', 'numerical', 'audit'), default='train')
    parser.add_argument('--num-envs', type=int); parser.add_argument('--resume')
    parser.add_argument('--pressure-steps', type=int, default=96)
    parser.add_argument('--no-compile', action='store_true')
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    raw_cfg = OmegaConf.load(preliminary.config)
    sys.path.insert(0, str(raw_cfg.supermimic.repo))
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from .application import read_config, build_engine, read_tasks, seed_all
    from .vector_bank import VectorBank
    from .vector_control import VectorControl, select_state
    from .vector_collector import VectorCollector
    from .vector_replay import VectorReplay
    from .vector_learning import GroupNormalizedCritic, TangentReplay, fit_wm, rollout, adapter_update, fit_critic, validate_critic
    from .vector_audit import physical_directions
    from .vector_evaluation import evaluate_physical, collect_validation
    from .vector_datasets import import_zero_mc, replay_mc_dataset, combine_mc
    from .vector_checkpoint import save_vector, load_vector
    from .dynamics import frozen_parameters
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    cfg = read_config(args.config)
    vc = cfg['vector']; train = cfg['training']
    if args.num_envs:
        vc['num_envs'] = args.num_envs
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1:
        raise ValueError('每个实验只允许可见一张GPU，不启动DDP/跨GPU通信')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(train['seed']); torch.set_num_threads(1)
    started = time.time()
    status = dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), phase='initializing', accepted_updates=0,
        attempted_adapter_updates=0, wm_updates=0, real_transitions=0, num_envs=vc['num_envs'], started_unix=started,
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), formal=args.mode == 'train')
    manifest = dict(config=cfg, command=[sys.executable, *sys.argv], status=status.copy(),
        independent_experiment=True, communication='none', sampling_seed=vc['sampling_seed'],
        seed_policy='same_initial_random_stream_across_eight_configs_then_independent_online_replay')
    json_write(output/'manifest.json', manifest)
    json_write(output/'status.json', status, replace=True)
    engine = build_engine(cfg, 'cuda:0')
    tasks = read_tasks(cfg['tasks'])
    bank = VectorBank(tasks, engine, 'cuda:0')
    control = VectorControl(engine, bank)
    critic = GroupNormalizedCritic(2835, engine.step.adapter.input_spec).cuda()
    numerical = numerical_check(engine, bank, control)
    json_write(output/'numerical.json', numerical)
    if args.mode == 'numerical':
        return 0
    if not args.no_compile and vc.get('compile_fk', True):
        # 只编译纯Tensor FK，冻结权重、参考公式和Actor架构均不变。
        engine.step.runtime.kin.forward = torch.compile(engine.step.runtime.kin.forward, dynamic=True)
    optimizers = dict(adapter=torch.optim.Adam(engine.step.adapter.parameters(), lr=train['adapter_lr']),
        critic=torch.optim.Adam(critic.parameters(), lr=train['critic_lr']),
        wm=torch.optim.Adam(engine.dynamics.model.net.parameters(), lr=train['wm_lr']))
    sim_cfg = make_sim('cuda:0', args)
    collector = replay = tangent = validation = None
    try:
        with launch_simulation(sim_cfg, args):
            env = StreamingPhysicalEnv(sim_cfg, vc['num_envs'], engine.step.bundle)
            collector = VectorCollector(env, control, seed=vc['sampling_seed'])
            replay = VectorReplay(train['replay_capacity'], vc['num_envs'], 'cuda:0')
            validation = VectorReplay(vc.get('validation_capacity', 65536), min(64, vc['num_envs']), 'cuda:0')
            tangent = TangentReplay()
            criteria = json.loads(Path(cfg['physical_criteria']).read_text())
            torch.cuda.reset_peak_memory_stats()
            collector.reset(perturb=True)
            if args.mode == 'pressure':
                # 第一次分配完整百万容量Replay；统计真实向量步和采集/反传峰值。
                for _ in range(8):
                    collector.step(replay, zero=False)
                tick = time.perf_counter()
                timings = []
                for _ in range(args.pressure_steps):
                    timings.append(collector.step(replay)['timing'])
                elapsed = time.perf_counter()-tick
                pair = collector.tangent_pairs(); tangent.add(pair)
                fitted = fit_wm(engine.dynamics, optimizers['wm'], replay.windows(train['wm_batch'], 8),
                                tangent.sample(train['wm_batch']), vc['wm_tangent'])
                engine.step.adapter.zero_grad(set_to_none=True)
                with engine.dynamics.adapter_phase(), frozen_parameters(critic):
                    trajectory = rollout(control, engine.dynamics, critic, replay.starts(train['backward_microbatch']),
                                          8, terminal_value=False)
                    (-trajectory['objective'].mean()).backward()
                norm = torch.nn.utils.clip_grad_norm_(engine.step.adapter.parameters(), 1., error_if_nonfinite=True)
                engine.step.adapter.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                result = dict(passed=True, num_envs=vc['num_envs'], vector_steps=args.pressure_steps,
                    actual_transitions=args.pressure_steps*vc['num_envs'], transitions_per_second=args.pressure_steps*vc['num_envs']/elapsed,
                    wall_seconds=elapsed, replay_capacity=replay.capacity, replay_bytes=sum(v.numel()*v.element_size() for v in replay.data.values()),
                    torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, nvidia_process_mib=gpu_usage(),
                    timing_mean_ms={k: sum(t[k] for t in timings)/len(timings) for k in timings[0]},
                    wm_fit=fitted, bptt8_microbatch=train['backward_microbatch'], adapter_gradient_norm=float(norm),
                    accepted_adapter_updates=0, numerical=numerical)
                json_write(output/'pressure.json', result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return 0
            if args.resume:
                status.update(load_vector(args.resume, engine, critic, optimizers, replay, tangent, collector, cfg, validation))
                status.update(pid=os.getpid(), started_unix=started, phase='resume_requires_fresh_physics_gate')
            if args.mode == 'audit':
                result = physical_directions(collector, control, engine.dynamics, critic, criteria)
                json_write(output/'directions.json', result)
                return 0 if result['passed_horizons'] else 2
            baseline = import_zero_mc(vc['initial_real_checkpoint'], control)
            warm_critic = fit_critic(critic, optimizers['critic'], baseline['train'], train['critic_epochs'], train['critic_batch'])
            json_write(output/'critic_initial_fit.json', warm_critic)
            collector.reset(perturb=True)
            for _ in range(vc.get('warmup_vector_steps', 128)):
                collector.step(replay, zero=True)
            pair = collector.tangent_pairs(); tangent.add(pair)
            collect_validation(collector, validation, vc.get('validation_vector_steps', 300), zero=True)
            zero_evaluation = evaluate_physical(collector, zero=True)
            json_write(output/'evaluation_zero.json', zero_evaluation)
            collector.reset(perturb=True)
            horizon = None
            critic_ready = False
            iteration = int(status.get('iteration', 0))
            last_audit_wm = -1
            last_saved = -1
            peak_nvidia = gpu_usage() or 0

            def checkpoint(label):
                nonlocal last_saved
                status.update(wm_updates=engine.dynamics.model_version, real_transitions=collector.real_transitions,
                              collector_resets=collector.reset_count, iteration=iteration)
                destination = output/f'{label}_{status["accepted_updates"]:06d}_{time.time_ns()}.pt'
                save_vector(destination, engine, critic, optimizers, replay, tangent, collector, status.copy(), cfg, validation)
                json_write(output/'latest_checkpoint.json', dict(path=str(destination), **status), replace=True)
                last_saved = status['accepted_updates']

            while status['accepted_updates'] < vc['target_accepted_updates']:
                loop_start = time.perf_counter()
                iteration += 1
                collector.policy_version = status['accepted_updates']
                collect_start = time.perf_counter()
                for _ in range(train['collect_vector_steps_per_update']):
                    collector.step(replay, zero=status['accepted_updates'] == 0)
                collect_seconds = time.perf_counter()-collect_start
                if iteration % vc.get('tangent_every_iterations', 32) == 1:
                    tangent.add(collector.tangent_pairs())
                    collector.reset(perturb=True)
                wm_reports = []
                for _ in range(train['wm_updates_per_iteration']):
                    wm_reports.append(fit_wm(engine.dynamics, optimizers['wm'], replay.windows(train['wm_batch'], train['wm_fit_window']),
                        tangent.sample(train['wm_batch']), vc['wm_tangent']))
                training_mc = combine_mc(baseline['train'], replay_mc_dataset(replay, control, 'train'))
                validation_mc = combine_mc(baseline['validation'], replay_mc_dataset(validation, control, 'validation'))
                critic_fit = fit_critic(critic, optimizers['critic'], training_mc, train['critic_epochs'], train['critic_batch'])
                critic_report = validate_critic(critic, validation_mc, training_mc['y'].mean(), train['critic_quality'],
                                                len(training_mc['y']), tasks)
                critic_ready = critic_report['gate']['passed']
                # 初始至少25次真实WM监督；之后未通过时继续监督修复，每100 WM步重测。
                due_audit = engine.dynamics.model_version >= 25 and (last_audit_wm < 0 or
                    (horizon is None and engine.dynamics.model_version-last_audit_wm >= vc.get('repair_audit_wm_interval', 100)))
                if due_audit:
                    status['phase'] = 'real_physics_direction_audit'
                    json_write(output/'status.json', status, replace=True)
                    gate = physical_directions(collector, control, engine.dynamics, critic, criteria)
                    last_audit_wm = engine.dynamics.model_version
                    json_write(output/f'directions_wm{last_audit_wm:06d}_adapter{status["accepted_updates"]:06d}.json', gate)
                    permitted = [h for h in gate['passed_horizons'] if h <= vc['initial_bptt_steps']]
                    horizon = max(permitted) if permitted else None
                    status['direction_passed_horizons'] = gate['passed_horizons']
                    collector.reset(perturb=True)
                update = None
                if horizon is not None:
                    status['attempted_adapter_updates'] += 1
                    starts = replay.starts(train['actor_batch'])
                    update, lambda_data = adapter_update(control, engine.dynamics, critic, optimizers['adapter'], starts, horizon,
                        terminal_value=critic_ready, microbatch=train['backward_microbatch'])
                    status['accepted_updates'] += 1
                    if lambda_data:
                        fit_critic(critic, optimizers['critic'], lambda_data, train['critic_epochs'], train['critic_batch'])
                    status['phase'] = update['phase']
                else:
                    status['phase'] = 'wm_and_critic_supervision_adapter_waiting_for_verified_direction'
                if status['accepted_updates'] and (status['accepted_updates'] == 1 or status['accepted_updates'] % vc['evaluation_every'] == 0) and last_saved != status['accepted_updates']:
                    physical = evaluate_physical(collector, zero=False)
                    physical['accepted_updates'] = status['accepted_updates']
                    physical['tracking_change_from_zero'] = physical['mean_tracking']-zero_evaluation['mean_tracking']
                    json_write(output/f'evaluation_{status["accepted_updates"]:06d}.json', physical)
                    collect_validation(collector, validation, vc.get('validation_vector_steps', 300), zero=False)
                    gate = physical_directions(collector, control, engine.dynamics, critic, criteria)
                    last_audit_wm = engine.dynamics.model_version
                    json_write(output/f'directions_wm{last_audit_wm:06d}_adapter{status["accepted_updates"]:06d}_posteval.json', gate)
                    permitted = [h for h in gate['passed_horizons'] if h <= vc['initial_bptt_steps']]
                    horizon = max(permitted) if permitted else None
                    status['direction_passed_horizons'] = gate['passed_horizons']
                    collector.reset(perturb=True)
                    checkpoint('full')
                if horizon is None and iteration % 100 == 0:
                    checkpoint('supervised_waiting')
                peak_nvidia = max(peak_nvidia, gpu_usage() or 0)
                status.update(iteration=iteration, wm_updates=engine.dynamics.model_version, real_transitions=collector.real_transitions,
                    replay_transitions=len(replay), tangent_pairs=tangent.total_pairs, critic_ready=critic_ready, bptt_steps=horizon,
                    num_envs=collector.num_envs, nvidia_peak_observed_mib=peak_nvidia,
                    torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    real_transitions_per_second=train['collect_vector_steps_per_update']*collector.num_envs/collect_seconds,
                    accepted_updates_per_hour=status['accepted_updates']*3600/max(1, time.time()-started),
                    wall_unix=time.time(), iteration_seconds=time.perf_counter()-loop_start,
                    complete_reset_segments=replay.completed_segments, complete_full_songs=replay.completed_full_songs,
                    head_truncated_terminals=replay.head_truncated_terminals)
                report = dict(status=status.copy(), wm=wm_reports, critic=critic_report, critic_fit=critic_fit, adapter=update,
                              control_timing=collector.timing)
                append_json(output/'training.jsonl', report)
                json_write(output/'status.json', status, replace=True)
                print(json.dumps(status, ensure_ascii=False), flush=True)
            checkpoint('complete')
            status['phase'] = 'completed_target_accepted_updates'
            json_write(output/'status.json', status, replace=True)
            return 0
    except Exception as error:
        status.update(phase='hard_failure', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), wall_unix=time.time())
        if collector:
            status.update(real_transitions=collector.real_transitions, num_envs=collector.num_envs)
        status['wm_updates'] = engine.dynamics.model_version
        json_write(output/'failure.json', status)
        json_write(output/'status.json', status, replace=True)
        raise
