"""Stage2-A真实权重、GPU压力及物理Oracle入口，不启动策略优化。

在独立运行目录加载冻结GENMO缓存和冻结SM权重，使用新的有限时域投影与原始
GENMO精度目标。数值模式核对零Adapter、投影、Actor及梯度；压力模式真实推进
指定批量物理；Oracle模式采集训练难例，保存完整正负参考响应。所有产物明确标注
阶段和accepted_updates=0，不能将环境创建、Oracle或WM监督冒充正式训练。
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    import json
    import os
    import time
    import traceback
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('numerical', 'pressure', 'oracle', 'evaluate', 'direction', 'calibrate', 'trace'), required=True)
    parser.add_argument('--checkpoint', help='已接受或被拒候选的真实模型文件；评价/审计均不产生策略更新')
    parser.add_argument('--wm-checkpoint', help='显式替换独立WM用于同一Adapter候选复验；不恢复旧Replay/优化器')
    parser.add_argument('--horizons', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--bptt-horizon', type=int, choices=(32, 64), default=32)
    parser.add_argument('--starts', help='已保存的训练集真实零策略高误差起点，仅用于梯度尺度标定')
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--steps', type=int, default=96)
    parser.add_argument('--oracle-batches', type=int, default=2)
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(preliminary.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl, select_state
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection, constrain_plan
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    from gem.closedloop.reference_adapter.vector_entry import json_write, numerical_check, compile_kinematics, gpu_usage
    from gem.closedloop.reference_adapter.stage2a_physics import difficulty_scan, oracle_experiment
    from gem.closedloop.reference_adapter.vector_evaluation import evaluate_physical
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.bridge.bundle import atomic_save
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(42); torch.set_num_threads(1)
    engine = build_engine(cfg, 'cuda:0')
    bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
    control = VectorControl(engine, bank)
    json_write(output/'sm_contract.json', numerical_check(engine, bank, control))
    control.adapter.disable_root_xy = True
    control.soft_imagination_continuation = False
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds,
        **cfg.get('stage2a', {}).get('projection', {})).cuda()
    settings = cfg.get('stage2a', {})
    scales = json.loads(Path(settings['calibration']).read_text())['scales'] if settings.get('calibration') else None
    control.precision = PrecisionObjective(scales, **settings.get('precision', {}))
    control.soft_joint_limit_cost = settings.get('soft_joint_limit_cost', True)
    before = {k: v.detach().clone() for k, v in control.adapter.state_dict().items()}
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        for key in ('projection', 'precision'):
            if checkpoint['config']['stage2a'][key] != settings.get(key):
                raise ValueError('审计必须使用checkpoint实际参考投影及精度目标，不能悄悄替换')
        if args.mode != 'calibrate':
            control.adapter.load_state_dict(checkpoint['adapter'])
        engine.dynamics.load_state_dict(checkpoint['dynamics'])
        engine.dynamics.model_version = checkpoint.get('wm_version', checkpoint.get('state', {}).get('wm_updates', 0))
        saved_scales = checkpoint.get('extra_state', {}).get('precision_scales')
        if saved_scales and args.mode != 'calibrate':
            control.precision.scales = {k: saved_scales[k] for k in control.precision.scales}
    if args.wm_checkpoint:
        wm = torch.load(args.wm_checkpoint, map_location='cpu', weights_only=False, mmap=True)
        engine.dynamics.load_state_dict(wm['dynamics'])
        engine.dynamics.model_version = wm.get('wm_version', wm.get('state', {}).get('wm_updates', 0))
    json_write(output/'manifest.json', dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        command=[sys.executable, *sys.argv], config=cfg, mode=args.mode, num_envs=args.num_envs,
        accepted_updates=0, scope='Stage2-A_validation_only', physical_resume='explicit_reset_of_real_state_with_repeat_test'))
    # CPU同一QP与GPU内核交叉核对，包括提案和历史的反向导数。
    torch.manual_seed(123)
    proposal = (torch.randn(8, 10, 27, device='cuda')*.01).requires_grad_()
    proposal.data[..., :2] = 0
    anchor = torch.zeros(8, 27, device='cuda')
    projected = control.projection(proposal, anchor, anchor)
    cpu = ReferenceProjection(control.bundle.spec, control.adapter.bounds.cpu(),
                              **cfg.get('stage2a', {}).get('projection', {}))
    expected = cpu(proposal.detach().cpu(), anchor.cpu(), anchor.cpu())
    torch.testing.assert_close(projected.cpu(), expected, atol=2e-5, rtol=2e-4)
    gradient, = torch.autograd.grad(projected.square().sum(), proposal)
    json_write(output/'projection_gpu.json', dict(max_abs_cpu_error=float((projected.cpu()-expected).abs().max()),
        finite_gradient=bool(torch.isfinite(gradient).all()), gradient_norm=float(gradient.norm()),
        zero_identity=bool((control.projection(proposal.detach()*0, anchor, anchor) == 0).all())))
    if args.mode == 'numerical':
        return 0
    json_write(output/'compiled_fk.json', compile_kinematics(engine, bank))
    if args.mode == 'calibrate':
        from gem.closedloop.reference_adapter.precision import calibrate_gradient_weights
        from gem.closedloop.reference_adapter.update_diagnostics import fixed_probes
        starts = torch.load(args.starts, map_location='cuda:0', weights_only=False)
        torch.cuda.reset_peak_memory_stats()
        report = calibrate_gradient_weights(control, engine.dynamics, fixed_probes(starts, per_song=2),
                                            horizon=args.bptt_horizon)
        report.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, gpu_process_mib=gpu_usage(),
                      gradient_checkpoint=True, backward_microbatch=32)
        json_write(output/'train_gradient_calibration.json', report)
        return 0
    collector = None
    try:
        with launch_simulation(make_sim('cuda:0', args), args):
            env = StreamingPhysicalEnv(make_sim('cuda:0', args), args.num_envs, engine.step.bundle)
            collector = VectorCollector(env, control)
            torch.cuda.reset_peak_memory_stats()
            if args.mode == 'pressure':
                collector.reset(perturb=False)
                started = time.perf_counter()
                for _ in range(args.steps):
                    collector.step(zero=True, perturb=False)
                json_write(output/'pressure.json', dict(num_envs=args.num_envs, steps=args.steps,
                    real_transitions=collector.real_transitions, seconds=time.perf_counter()-started,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, gpu_process_mib=gpu_usage(),
                    last_step_timing=collector.timing, accepted_updates=0))
            elif args.mode == 'evaluate':
                report = evaluate_physical(collector, zero=not bool(args.checkpoint))
                report['checkpoint'] = args.checkpoint
                report['accepted_updates'] = checkpoint.get('state', {}).get('accepted_updates') if checkpoint else 0
                json_write(output/('model_full_song_evaluation.json' if checkpoint else 'zero_full_song_evaluation.json'), report)
            elif args.mode == 'trace':
                if checkpoint is None:
                    raise ValueError('三路径H64审计需要明确完整参数候选')
                from gem.closedloop.reference_adapter.long_horizon import independent_long_trace
                criteria = json.loads(Path(cfg['physical_criteria']).read_text()) | cfg['training']['wm_quality']
                report, data = independent_long_trace(collector, before, criteria)
                report.update(checkpoint=args.checkpoint, wm_checkpoint=args.wm_checkpoint)
                json_write(output/'three_path_H1_H64.json', report)
                atomic_save(data, output/'three_path_physical_tensors.pt')
            elif args.mode == 'direction':
                if checkpoint is None:
                    raise ValueError('独立闭环H8/H16/H32审计需要明确候选checkpoint')
                from gem.closedloop.reference_adapter.parameter_audit import parameter_update_audit
                from gem.closedloop.reference_adapter.vector_learning import GroupNormalizedCritic
                critic = GroupNormalizedCritic(2835, control.spec).cuda()
                criteria = json.loads(Path(cfg['physical_criteria']).read_text()) | cfg['training']['wm_quality']
                result = parameter_update_audit(collector, control, engine.dynamics, critic, before,
                    horizons=tuple(args.horizons), criteria=criteria)
                result['comparison'] = 'checkpoint_candidate_against_exact_zero_adapter_independent_audit_songs'
                result['checkpoint'] = args.checkpoint
                result['wm_checkpoint'] = args.wm_checkpoint
                result.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              gpu_process_mib=gpu_usage(), gradient_checkpoint=True)
                json_write(output/'independent_parameter_direction.json', result)
            else:
                pool, calibration = difficulty_scan(collector, args.steps)
                control.precision.scales = calibration['scales']
                json_write(output/'zero_train_calibration.json', calibration)
                atomic_save({k: v.cpu() for k, v in pool.items()}, output/'real_high_error_starts.pt')
                rows = []
                per_batch = min(args.num_envs//16, len(bank.by_split['train']))
                train = bank.by_split['train']
                for batch in range(args.oracle_batches):
                    chosen = []
                    for song in train[:per_batch]:
                        ids = (pool['song'] == song).nonzero().flatten()
                        chosen.append(ids[min(batch, len(ids)-1)])
                    indices = torch.stack(chosen)
                    report, pairs = oracle_experiment(collector, select_state(pool, indices))
                    atomic_save(pairs, output/f'oracle_reference_pairs_{batch:02d}.pt')
                    json_write(output/f'oracle_batch_{batch:02d}.json', report)
                    rows += report
                    print(json.dumps(dict(batch=batch, credible_gains=sum(r['credible_gain'] for r in report),
                        real_transitions=collector.real_transitions), ensure_ascii=False), flush=True)
                json_write(output/'oracle_summary.json', dict(rows=rows,
                    credible_gain_count=sum(r['credible_gain'] for r in rows),
                    real_transitions=collector.real_transitions, real_transitions_by_role=collector.real_by_role,
                    accepted_updates=0, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    boundary='offline_best_candidate_not_a_learned_observable_policy'))
        return 0
    except BaseException:
        json_write(output/'failure.json', dict(traceback=traceback.format_exc(), accepted_updates=0,
            real_transitions=collector.real_transitions if collector else 0))
        if collector:
            atomic_save(dict(state=getattr(collector, 'last_constraint_state', collector.state),
                        proposal=getattr(collector, 'last_constraint_proposal', None),
                        constraints=getattr(collector, 'last_constraint_failure', None)),
                        output/'failure_physical_state.pt')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
