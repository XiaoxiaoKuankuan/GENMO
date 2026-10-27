"""Stage2-A冻结GENMO缓存的唯一WM+BPTT正式训练入口，单GPU独立进程。

训练数据来自当前策略真实连续物理转移和经投影/冻结Actor的正负参考响应。WM有限
监督修复后提出完整Adam候选，训练内部保留起点的H32/H64真实信任域接受后才增加
有效更新数；H8/H16只诊断。连续有限次失败保存全现场退出。Critic保留独立状态但
完全冻结、不训练、禁用终端
价值，没有可信完整MC校准时不偷偷接入梯度。零/25/50/100/200次更新作固定整歌评价，
评价歌曲不用于逐更新选择。每组模型、Replay、RNG及输出独立，不使用任何跨GPU通信。
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
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--mode', choices=('train', 'integration', 'wm-repair'), default='train')
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(preliminary.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs is not None:
        cfg['vector']['num_envs'] = args.num_envs
    import torch
    from .application import build_engine, read_tasks, seed_all
    from .vector_bank import VectorBank
    from .vector_control import VectorControl, select_state
    from .vector_collector import VectorCollector
    from .vector_replay import VectorReplay
    from .vector_learning import GroupNormalizedCritic, TangentReplay, fit_wm, adapter_update
    from .vector_entry import json_write, append_json, numerical_check, compile_kinematics, gpu_usage
    from .vector_evaluation import evaluate_physical, collect_validation
    from .vector_checkpoint import save_vector, load_vector
    from .reference_constraints import ReferenceProjection
    from .precision import PrecisionObjective
    from .stage2a_physics import difficulty_scan, collect_reference_response
    from .reference_response import ReferenceResponseReplay, validate_response
    from .physical_trust import trust_region_update
    from .update_diagnostics import parameter_identity, fixed_probes
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.env import make_sim
    from supermimic.bridge.bundle import atomic_save, file_digest
    settings, train, vector = cfg['stage2a'], cfg['training'], cfg['vector']
    if settings['method'] != 'WM_BPTT_physical_trust' or settings.get('terminal_value', False):
        raise ValueError('Stage2-A初版只允许WM+BPTT且禁用未校准Critic终端价值')
    if args.mode == 'train' and vector['initial_bptt_steps'] not in (32, 64):
        raise ValueError('本轮正式目标为H32/H64，H8/H16仅作诊断，不能默认回退')
    long_settings = settings['long_horizon']
    if max(long_settings['response_train_horizons']) > 32 or max(map(int, long_settings['state_weights'])) > 32:
        raise ValueError('当前H64只做独立验证，获得可靠方向证据前不得直接加入监督')
    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('每个进程只允许一张独立GPU，不使用DDP/torchrun')
    tasks = read_tasks(cfg['tasks'])
    if settings['purpose'] == 'formal' and sum(t['split'] == 'train' for t in tasks) < 64:
        raise ValueError('正式64首实验不能用现有8首训练歌曲冒充')
    if args.mode == 'train':
        oracle = json.loads(Path(settings['oracle_evidence']).read_text())
        if not any(r['credible_gain'] and r['horizon'] >= 16 for r in oracle['rows']):
            raise ValueError('Oracle未证明H16真实可补偿性，不扩大Adapter训练')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(train['seed']); torch.set_num_threads(1)
    engine = build_engine(cfg, 'cuda:0')
    bank = VectorBank(tasks, engine, 'cuda:0')
    control = VectorControl(engine, bank)
    json_write(output/'numerical_original_SM_contract.json', numerical_check(engine, bank, control))
    control.adapter.disable_root_xy = True
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds, **settings['projection']).cuda()
    calibration = json.loads(Path(settings['calibration']).read_text())
    control.precision = PrecisionObjective(calibration['scales'], **settings['precision'])
    control.soft_joint_limit_cost = settings.get('soft_joint_limit_cost', False)
    if not control.soft_joint_limit_cost and cfg['rewards']['reference_limit'] != 0:
        raise ValueError('关闭软限位时修正参考的同一软限位成本也必须明确配置为0')
    control.soft_imagination_continuation = False
    control.relative_critic_features = True
    critic = GroupNormalizedCritic(2835, control.spec).cuda()
    critic.requires_grad_(False)
    optimizers = dict(adapter=torch.optim.Adam(control.adapter.parameters(), lr=train['adapter_lr']),
        wm=torch.optim.Adam(engine.dynamics.model.net.parameters(), lr=train['wm_lr']),
        critic=torch.optim.Adam(critic.parameters(), lr=train['critic_lr']))
    replay = VectorReplay(train['replay_capacity'], vector['num_envs'], 'cuda:0')
    validation = VectorReplay(vector['validation_capacity'], min(32, vector['num_envs']), 'cuda:0')
    tangent = TangentReplay(16384)
    response = ReferenceResponseReplay(8192)
    background = (VectorReplay(65536, vector['num_envs']-len(bank.by_split['evaluation']), 'cuda:0')
                  if vector['num_envs'] > len(bank.by_split['evaluation']) else None)
    collector = None
    pending = None
    state = dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started_unix=time.time(),
        phase='initializing', accepted_updates=0, attempted_adapter_updates=0, consecutive_rejections=0,
        wm_updates=0, num_envs=vector['num_envs'], critic_terminal_enabled=False,
        method=settings['method'], source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    extra = dict(mc={}, critic_status='unqualified_disabled_no_terminal_value', response={}, safety=None,
        high_error_starts=None, precision_scales=control.precision.scales, pending_candidate=None, best=None)
    frozen = engine.step.bundle.frozen_snapshot()
    json_write(output/'manifest.json', dict(config=cfg, command=[sys.executable, *sys.argv], state=state,
        calibration_sha256=file_digest(settings['calibration']), initialization='zero_adapter_original_independent_WM',
        sampling='60_percent_real_zero_policy_high_error_40_percent_uniform_legal_frames',
        split_boundary='train_safety_internal_only_validation_audit_evaluation_not_for_candidate_selection'))
    json_write(output/'compiled_fk.json', compile_kinematics(engine, bank))

    def publish(phase):
        state.update(phase=phase, wm_updates=engine.dynamics.model_version, wall_unix=time.time(),
            real_transitions=collector.real_transitions if collector else 0,
            real_transitions_by_role=collector.real_by_role if collector else {},
            response_pairs=response.total_pairs, replay_transitions=len(replay), gpu_process_mib=gpu_usage(),
            response_coverage=response.coverage(),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        json_write(output/'status.json', state, replace=True)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def checkpoint(label):
        publish(state['phase'])
        extra['response'] = response.state_dict()
        extra['evaluation_background_replay'] = background.state_dict() if background else None
        extra['constraint_failure'] = getattr(collector, 'last_constraint_failure', None)
        extra['constraint_state'] = getattr(collector, 'last_constraint_state', None)
        extra['constraint_proposal'] = getattr(collector, 'last_constraint_proposal', None)
        path = output/f'{label}_{state["accepted_updates"]:06d}_{time.time_ns()}.pt'
        save_vector(path, engine, critic, optimizers, replay, tangent, collector, state.copy(), cfg, validation, extra)
        json_write(output/'latest_checkpoint.json', dict(path=str(path), **state), replace=True)
        return path

    def evaluate(label):
        publish('independent_full_song_physical_evaluation')
        collector.policy_version = state['accepted_updates']
        report = evaluate_physical(collector, zero=state['accepted_updates'] == 0, background_replay=background)
        report['accepted_updates'] = state['accepted_updates']
        json_write(output/f'evaluation_{label}.json', report)
        score = (report['completion_rate'], -report['mean_root_position_rmse_m'],
                 report['mean_full_horizon_tracking'], -report['mean_jerk_rms'])
        if extra['best'] is None or score > tuple(extra['best']['ranking']):
            extra['best'] = dict(ranking=score, accepted_updates=state['accepted_updates'])
            atomic_save(dict(adapter=control.adapter.state_dict(), dynamics=engine.dynamics.state_dict(),
                config=cfg, state=state.copy(), evidence=report, continuation='weights_only_not_full_resume'),
                output/f'best_{state["accepted_updates"]:06d}_{time.time_ns()}.pt')
        collector.reset(split='train', perturb=True)
        return report

    try:
        sim_cfg = make_sim('cuda:0', args)
        with launch_simulation(sim_cfg, args):
            env = StreamingPhysicalEnv(sim_cfg, vector['num_envs'], engine.step.bundle)
            collector = VectorCollector(env, control, seed=vector['sampling_seed'])
            collector.uniform_start = True
            collector.reset(perturb=False)
            if args.resume:
                ck = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
                for key in ('method', 'projection', 'precision', 'terminal_value', 'objective_version', 'long_horizon'):
                    if ck['config']['stage2a'].get(key) != settings[key]:
                        raise ValueError('恢复算法/参考/目标不兼容，不能加载旧饱和策略')
                if ck['config']['stage2a'].get('soft_joint_limit_cost', True) != control.soft_joint_limit_cost:
                    raise ValueError('软限位目标改变不能冒充同目标断点续训；新实验从零初始化，旧权重只用于明确物理对照')
                state.update(load_vector(args.resume, engine, critic, optimizers, replay, tangent, collector, cfg, validation))
                extra = ck['extra_state']
                response.load_state_dict(extra['response'], 'cuda:0')
                if extra.get('evaluation_background_replay') and background is not None:
                    background.load_state_dict(extra['evaluation_background_replay'])
                for key in ('safety', 'high_error_starts'):
                    extra[key] = {k: v.cuda() for k, v in extra[key].items()}
                control.precision.scales = extra['precision_scales']
                state.update(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), resumed_from=args.resume)
            else:
                publish('zero_policy_real_difficulty_collection')
                pool, distribution = difficulty_scan(collector, steps=settings['difficulty_steps'])
                extra['high_error_starts'] = pool
                json_write(output/'zero_difficulty_distribution.json', distribution)
                # 安全起点使用独立随机流与时间段，来源仍是训练音乐中的实际执行。
                collector.reset(split='train', random_start=True, perturb=False)
                for _ in range(24):
                    collector.step(zero=True, split='train_safety', reset_split='train', perturb=False)
                eligible = collector.state['frame']+64 < bank.ends[collector.state['song']]
                safety = fixed_probes(select_state(collector.state, eligible), per_song=2)
                extra['safety'] = {k: v.detach().clone() for k, v in safety.items()}
                publish('real_transition_warmup')
                collector.reset(split='train', perturb=True)
                for _ in range(vector['warmup_vector_steps']):
                    collector.step(replay)
                collect_validation(collector, validation, vector['validation_vector_steps'])
                eligible = collector.state['frame']+64 < bank.ends[collector.state['song']]
                validation_source = fixed_probes(select_state(collector.state, eligible), per_song=2)
                extra['response_validation'] = collect_reference_response(collector, validation_source,
                    split='validation')
                collector.reset(split='train', perturb=True)
                if args.mode == 'train':
                    evaluate('000000')
            collector.high_error_starts = extra['high_error_starts']
            torch.cuda.reset_peak_memory_stats()
            if args.mode == 'integration':
                eligible = (collector.state['frame']+64 < bank.ends[collector.state['song']]).nonzero().flatten()
                source = select_state(collector.state, eligible[:min(8, collector.num_envs//4)])
                data = collect_reference_response(collector, source, replay=replay)
                response.add(data)
                fitting = fit_wm(engine.dynamics, optimizers['wm'], replay.windows(32, 8), None, 0.,
                    reference_pairs=response.sample(32, horizon=32), reference_weight=settings['reference_response_weight'],
                    reference_kin=control.runtime.kin, response_settings=long_settings['response'],
                    long_windows=replay.windows(32, 32), long_weights=long_settings['state_weights'], checkpoint_steps=True)
                identity = dict(adapter=parameter_identity(control.adapter), wm=parameter_identity(engine.dynamics),
                                critic=parameter_identity(critic))
                saved = checkpoint('integration')
                payload = torch.load(saved, map_location='cpu', weights_only=False, mmap=True)
                state.update(load_vector(saved, engine, critic, optimizers, replay, tangent, collector, cfg, validation))
                assert identity == dict(adapter=parameter_identity(control.adapter), wm=parameter_identity(engine.dynamics),
                                        critic=parameter_identity(critic))
                restored_response = ReferenceResponseReplay(8192)
                restored_response.load_state_dict(payload['extra_state']['response'], 'cuda:0')
                assert restored_response.total_pairs == response.total_pairs
                for horizon, bucket in response.buckets.items():
                    for key, value in bucket.items():
                        torch.testing.assert_close(value, restored_response.buckets[horizon][key], atol=0, rtol=0)
                assert set(payload['optimizers']) == {'adapter', 'wm', 'critic'}
                for name, opt in optimizers.items():
                    for index, slots in opt.state_dict()['state'].items():
                        for key, value in slots.items():
                            torch.testing.assert_close(value.cpu(), payload['optimizers'][name]['state'][index][key], atol=0, rtol=0)
                for key, value in replay.data.items():
                    torch.testing.assert_close(value[:replay.size].cpu(), payload['replay']['data'][key], atol=0, rtol=0)
                engine.step.bundle.assert_unchanged(frozen)
                json_write(output/'integration.json', dict(checkpoint_restored=True, adapter_updates=0,
                    model_identities=identity, three_optimizers_exact=True, replay_exact=True, reference_response_exact=True,
                    saved_rng_streams=list(payload['rng'])+['collector'], solver_resume='new_real_episode',
                    wm=fitting, response_rejected=response.rejected, real_transitions=collector.real_transitions))
                publish('integration_complete_no_policy_update')
                return 0
            while state['accepted_updates'] < vector['target_accepted_updates']:
                collector.policy_version = state['accepted_updates']
                collector.reset(split='train', perturb=True) if collector.state is None else None
                publish('current_policy_real_collection')
                for _ in range(train['collect_vector_steps_per_update']):
                    collector.step(replay)
                eligible = collector.state['frame']+64 < bank.ends[collector.state['song']]
                source = fixed_probes(select_state(collector.state, eligible),
                                     per_song=max(2, (settings['response_pairs']+len(bank.by_split['train'])-1)//len(bank.by_split['train'])))
                n = min(settings['response_pairs'], collector.num_envs//4, len(source['state']))
                source = select_state(source, torch.linspace(0, len(source['state'])-1, n, device='cuda').long())
                data = collect_reference_response(collector, source, replay=replay)
                response.add(data)
                publish('bounded_current_policy_WM_supervision')
                steps = settings['wm_repair_steps'] if args.mode == 'wm-repair' or state['consecutive_rejections'] or engine.dynamics.model_version == 0 else train['wm_updates_per_iteration']
                for wm_step in range(steps):
                    if wm_step and wm_step % 10 == 0:
                        # 固定监督预算内刷新真实方向，避免对同一16个配对反复拟合50遍。
                        for _ in range(8):
                            collector.step(replay)
                        eligible = collector.state['frame']+64 < bank.ends[collector.state['song']]
                        fresh = fixed_probes(select_state(collector.state, eligible), per_song=4)
                        n_fresh = min(settings['response_pairs'], len(fresh['state']), collector.num_envs//4)
                        ids = torch.randperm(len(fresh['state']), generator=collector.generator, device='cuda')[:n_fresh]
                        fresh = select_state(fresh, ids)
                        data = collect_reference_response(collector, fresh, replay=replay)
                        response.add(data)
                    source_replay = background if background is not None and background.size >= 8 and wm_step % 5 == 0 else replay
                    response_h = long_settings['response_train_horizons'][engine.dynamics.model_version % len(long_settings['response_train_horizons'])]
                    report = fit_wm(engine.dynamics, optimizers['wm'], source_replay.windows(train['wm_batch'], 8), None, 0.,
                        reference_pairs=response.sample(long_settings['response_batch'], horizon=response_h),
                        reference_weight=settings['reference_response_weight'], reference_kin=control.runtime.kin,
                        response_settings=long_settings['response'], long_windows=replay.windows(long_settings['state_batch'], 32),
                        long_weights=long_settings['state_weights'], checkpoint_steps=True)
                    append_json(output/'wm.jsonl', report)
                response_report = validate_response(engine.dynamics, data, kin=control.runtime.kin,
                                                    scales=long_settings['response']['scales'])
                json_write(output/f'wm_response_{engine.dynamics.model_version:06d}.json', response_report)
                independent = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                               for k, v in extra['response_validation'].items()}
                json_write(output/f'wm_response_validation_{engine.dynamics.model_version:06d}.json',
                           validate_response(engine.dynamics, independent, kin=control.runtime.kin,
                                             scales=long_settings['response']['scales']))
                json_write(output/f'wm_validation_{engine.dynamics.model_version:06d}.json',
                           engine.dynamics.validate_real(validation.windows(32, 64), kin=control.runtime.kin))
                if args.mode == 'wm-repair':
                    checkpoint('supervised_WM_no_Adapter_update')
                    if engine.dynamics.model_version >= long_settings['repair_update_budget']:
                        publish('bounded_long_WM_repair_complete_zero_Adapter_updates')
                        return 0
                    collector.reset(split='train', perturb=True)
                    continue
                state['attempted_adapter_updates'] += 1
                before, adam_before = copy.deepcopy(control.adapter.state_dict()), copy.deepcopy(optimizers['adapter'].state_dict())
                pending = (before, adam_before)
                publish('WM_BPTT_candidate_not_yet_accepted')
                candidate, _ = adapter_update(control, engine.dynamics, critic, optimizers['adapter'],
                    replay.starts(train['actor_batch'], horizon=vector['initial_bptt_steps'], ends=bank.ends),
                    vector['initial_bptt_steps'], terminal_value=False,
                    microbatch=train['backward_microbatch'], probes=extra['safety'], checkpoint_steps=True)
                extra['pending_candidate'] = dict(adapter=copy.deepcopy(control.adapter.state_dict()),
                    optimizer=copy.deepcopy(optimizers['adapter'].state_dict()), report=candidate)
                publish('real_physical_candidate_trust_region')
                safety = extra['safety']
                count = min(settings['safety_pairs'], len(safety['state']), collector.num_envs//4)
                indices = (torch.arange(count, device='cuda')+state['attempted_adapter_updates']*count)%len(safety['state'])
                criteria = json.loads(Path(cfg['physical_criteria']).read_text()) | train['wm_quality']
                result = trust_region_update(collector, select_state(safety, indices), engine.dynamics, critic,
                    optimizers['adapter'], before, adam_before, criteria, horizon=vector['initial_bptt_steps'], replay=replay,
                    response_replay=response)
                accepted = result['accepted'] and candidate['effective']
                if accepted:
                    state['accepted_updates'] += 1
                    state['consecutive_rejections'] = 0
                else:
                    control.adapter.load_state_dict(before); optimizers['adapter'].load_state_dict(adam_before)
                    state['consecutive_rejections'] += 1
                    extra['last_rejected_candidate'] = extra['pending_candidate'] | dict(physical_evidence=result)
                    atomic_save(dict(**extra['last_rejected_candidate'], config=cfg,
                        dynamics=engine.dynamics.state_dict(), wm_version=engine.dynamics.model_version),
                        output/f'rejected_candidate_{state["attempted_adapter_updates"]:03d}.pt')
                pending = None
                append_json(output/'updates.jsonl', dict(**state, candidate=candidate, trust_region=result, accepted=accepted))
                extra['pending_candidate'] = None
                engine.step.bundle.assert_unchanged(frozen)
                publish('accepted_real_physical_parameter_update' if accepted else 'rejected_and_rolled_back')
                if state['consecutive_rejections'] >= settings['max_consecutive_rejections']:
                    checkpoint('bounded_direction_failure')
                    publish(f'stopped_current_WM_has_no_verified_H{vector["initial_bptt_steps"]}_update')
                    return 2
                if accepted and state['accepted_updates'] in (25, 50, 100, 200):
                    evaluate(f'{state["accepted_updates"]:06d}')
                if accepted and (state['accepted_updates'] % vector['checkpoint_every'] == 0 or state['accepted_updates'] == 1):
                    checkpoint('accepted')
            if state['accepted_updates'] not in (0, 25, 50, 100, 200):
                evaluate(f'{state["accepted_updates"]:06d}')
            checkpoint('complete')
            publish('completed_accepted_update_budget')
            return 0
    except BaseException:
        state['failure_traceback'] = traceback.format_exc()
        if pending:
            control.adapter.load_state_dict(pending[0]); optimizers['adapter'].load_state_dict(pending[1])
        if collector is not None:
            state['phase'] = 'failed_saved_last_valid_model'
            try:
                checkpoint('failure')
            except BaseException:
                json_write(output/'checkpoint_failure.json', dict(traceback=traceback.format_exc()))
        json_write(output/'failure.json', state)
        raise
